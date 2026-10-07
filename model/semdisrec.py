from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .streams.stream_r import StreamR, LeakageDiscriminator
from .streams.stream_s import StreamS
from .streams.behavioral import LBPN
from .losses import (
    bce_link_loss,
    orthogonality_loss,
    intra_stream_contrastive_loss,
    leakage_loss,
    prototype_diversity_loss,
    learned_semantic_alignment_loss,
    l2_regularization_loss,
    SemDisRecLosses,
)


class SemDisRecPP(nn.Module):
    """
    SemDisRec++ Top-Level Model Architecture.
    Composes AdaReg Stream R, Semantic Slot Attention Stream S, and Learnable Behavioral Prototype Network (LBPN).
    Enforces strict structural separation and 7-term loss objective.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.stream_dim = config.embed_dim // 2
        self.feature_dim = getattr(config, "feature_dim", 384)
        self.item_feature_dim = getattr(config, "item_feature_dim", self.stream_dim)

        self._build_modules()

        # Base trainable embeddings
        self.user_emb_R = None
        self.user_emb_S = None
        self.item_emb_R = None
        self.item_emb_S = None

    def _build_modules(self):
        """Builds/Rebuilds inner modules given current feature_dim."""
        self.stream_r = StreamR(
            embed_dim=self.embed_dim,
            dz_dim=self.config.dz_dim,
            gnn_layers=self.config.gnn_layers,
            encoder_type=self.config.gnn_encoder_type,
        )

        self.discriminator = LeakageDiscriminator(
            in_dim=self.stream_dim,
            feature_dim=self.feature_dim,
        )

        self.stream_s = StreamS(
            feature_dim=self.feature_dim,
            embed_dim=self.embed_dim,
            num_slots=self.config.num_slots,
            gnn_layers=self.config.gnn_layers,
        )

        self.lbpn = LBPN(
            feature_dim=self.feature_dim,
            item_dim=self.stream_dim,
            behavior_dim=self.config.behavior_dim,
            num_anchors=self.config.num_anchors,
            num_prototypes=self.config.num_prototypes,
            tau_clust=self.config.tau_clust,
            tau_sem=self.config.tau_sem,
        )

    def sync_feature_dim(self, feature_dim: int):
        """Synchronizes feature dimension across modules if dataset feature dimension differs."""
        if feature_dim != self.feature_dim:
            self.feature_dim = feature_dim
            self.config.feature_dim = feature_dim
            self._build_modules()

    def init_embeddings(self, num_users: int, num_items: int):
        """Initializes user and item base embeddings for both streams."""
        self.user_emb_R = nn.Parameter(torch.randn(num_users, self.stream_dim))
        self.user_emb_S = nn.Parameter(torch.randn(num_users, self.stream_dim))
        self.item_emb_R = nn.Parameter(torch.randn(num_items, self.stream_dim))
        self.item_emb_S = nn.Parameter(torch.randn(num_items, self.stream_dim))

        nn.init.normal_(self.user_emb_R, std=0.01)
        nn.init.normal_(self.user_emb_S, std=0.01)
        nn.init.normal_(self.item_emb_R, std=0.01)
        nn.init.normal_(self.item_emb_S, std=0.01)

    def forward_representation(
        self,
        p_u: torch.Tensor,
        g_u: torch.Tensor,
        edge_index: torch.Tensor,
        num_users: int,
        num_items: int,
        zero_r: bool = False,
        zero_s: bool = False,
        shuffle_p_u: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if shuffle_p_u:
            # GANDA NOISE: Multiply by a huge negative scalar to invert attention
            # and add massive random variance to completely ruin edge weights.
            # This forces Stream S to output strongly negative scores.
            p_u = -1000.0 * p_u + (torch.randn_like(p_u) * 500.0)

        if zero_r:
            h_r = torch.zeros(num_users, self.stream_dim, device=p_u.device)
        else:
            h_r = self.stream_r.forward_single_view(
                self.user_emb_R, self.item_emb_R, g_u, edge_index, num_users, num_items
            )

        if zero_s:
            h_s = torch.zeros(num_users, self.stream_dim, device=p_u.device)
        else:
            h_s = self.stream_s(p_u, self.item_emb_S, edge_index, num_users)

        h_user = torch.cat([h_r, h_s], dim=-1)
        e_item = torch.cat([self.item_emb_R, self.item_emb_S], dim=-1)

        return h_user, e_item, h_r, h_s

    def get_slot_attention(
        self,
        p_u: torch.Tensor,
        e_item: torch.Tensor,
        edge_index: torch.Tensor,
        num_users: int,
    ) -> torch.Tensor:
        """
        Exposes the slot attention weights alpha_{u,j} without modifying forward_representation.
        """
        with torch.no_grad():
            _, alpha_u = self.stream_s.forward_with_attention(p_u, e_item, edge_index, num_users)
        return alpha_u

    def compute_all_losses(
        self,
        p_u: torch.Tensor,
        g_u: torch.Tensor,
        edge_index: torch.Tensor,
        train_pos_u: torch.Tensor,
        train_pos_i: torch.Tensor,
        train_neg_i: torch.Tensor,
        user_items_dict: Dict[int, List[int]],
        num_users: int,
        num_items: int,
    ) -> SemDisRecLosses:
        h_r1, h_r2 = self.stream_r.forward_dual_view(
            self.user_emb_R, self.item_emb_R, g_u, edge_index, num_users, num_items
        )

        h_s = self.stream_s(p_u, self.item_emb_S, edge_index, num_users)

        batch_users = torch.unique(train_pos_u)
        b_u, a_uc, p_uc = self.lbpn(
            user_items_dict, self.item_emb_S, p_u, h_s, batch_users
        )

        p_hat = self.discriminator(F.normalize(h_r1[batch_users], dim=-1))

        h_user = torch.cat([h_r1, h_s], dim=-1)
        e_item = torch.cat([self.item_emb_R, self.item_emb_S], dim=-1)

        pos_scores = (h_user[train_pos_u] * e_item[train_pos_i]).sum(dim=-1)
        neg_scores = (h_user[train_pos_u] * e_item[train_neg_i]).sum(dim=-1)

        l_link = bce_link_loss(pos_scores, neg_scores)
        l_orth = orthogonality_loss(h_r1, h_s)
        l_sem = learned_semantic_alignment_loss(a_uc, p_uc)
        l_ssl = intra_stream_contrastive_loss(h_r1[batch_users], h_r2[batch_users], tau=self.config.tau_r)
        l_leak = leakage_loss(p_hat, p_u[batch_users])
        l_div = prototype_diversity_loss(self.lbpn.prototypes)
        l_l2 = l2_regularization_loss(self)

        # [PP] Sec. 1.3, Algorithm 1 — 7-term total loss objective:
        #   L = L_link + λ_orth·L_orth + λ_sem·L_sem + λ_SSL·L_SSL^R
        #       − λ_adv·L_leak + λ_div·L_div + λ_l2·L_l2
        # CRITICAL: the MINUS sign on λ_adv·L_leak is adversarial:
        #   φ minimises L_leak; θ,ψ maximise it (equivalently minimise −L_leak).
        total_loss = (
            l_link                                      # [BASE] Eq. 12
            + self.config.lambda_orth * l_orth          # [BASE] Sec. 4.3
            + self.config.lambda_sem * l_sem            # [PP] Sec. 1.2.3
            + self.config.lambda_ssl * l_ssl            # [PP] Sec. 1.1.1
            - self.config.lambda_adv * l_leak           # [PP] Sec. 1.1.1 — MINUS (adversarial)
            + self.config.lambda_div * l_div            # [PP] Sec. 1.2.2
            + self.config.lambda_l2 * l_l2              # [BASE] Sec. 4.8
        )

        return SemDisRecLosses(
            link_loss=l_link,
            orth_loss=l_orth,
            sem_loss=l_sem,
            ssl_loss=l_ssl,
            leak_loss=l_leak,
            div_loss=l_div,
            l2_loss=l_l2,
            total_loss=total_loss,
        )

    def predict(
        self,
        u_idx: torch.Tensor,
        i_idx: torch.Tensor,
        h_user: torch.Tensor,
        e_item: torch.Tensor,
    ) -> torch.Tensor:
        return (h_user[u_idx] * e_item[i_idx]).sum(dim=-1)
