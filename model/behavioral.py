from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BehavioralSetEncoder(nn.Module):
    """
    Non-temporal behavioral set encoder for user interaction history.
    1. Projects items v_i = W_b e_i in R^{d_b}.
    2. Computes soft anchor histogram h_u in R^{K_b}.
    3. Refines item sequence H^{(0)} using 1-layer MHA with 3D padded tensor batching.
    4. Fuses histogram and pooled set vector into b_u in R^{d_b}.
    """

    def __init__(
        self,
        item_dim: int,
        behavior_dim: int = 32,
        num_anchors: int = 8,
        tau_anchor: float = 1.0,
        n_max: int = 100,
    ):
        super().__init__()
        self.item_dim = item_dim
        self.behavior_dim = behavior_dim
        self.num_anchors = num_anchors
        self.tau_anchor = tau_anchor
        self.n_max = n_max

        # Item-to-behavior projection matrix W_b in R^{d_b x d/2}
        self.W_b = nn.Linear(item_dim, behavior_dim, bias=False)

        # Learnable behavioral anchors A in R^{d_b x K_b}
        self.anchors = nn.Parameter(torch.randn(behavior_dim, num_anchors))
        nn.init.xavier_uniform_(self.anchors)

        # Set attention refinement (MHA)
        self.ln = nn.LayerNorm(behavior_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=behavior_dim, num_heads=4, batch_first=True
        )

        # Fused MLP: [h_u || v_u^pool] in R^{K_b + d_b} -> R^{d_b}
        self.mlp_fuse = nn.Sequential(
            nn.Linear(num_anchors + behavior_dim, behavior_dim),
            nn.LeakyReLU(),
            nn.Linear(behavior_dim, behavior_dim),
        )

    def forward(
        self,
        user_items_dict: Dict[int, List[int]],
        e_item: torch.Tensor,
        batch_users: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(batch_users, int):
            batch_users = torch.arange(batch_users, device=device)
        bz = len(batch_users)
        pad_items = np.zeros((bz, self.n_max), dtype=np.int64)
        masks = np.ones((bz, self.n_max), dtype=bool)

        for i, u in enumerate(batch_users.cpu().numpy()):
            items = user_items_dict.get(u, [])
            if len(items) > 0:
                if len(items) > self.n_max:
                    items = items[: self.n_max]
                pad_items[i, : len(items)] = items
                masks[i, :len(items)] = False

        items_tensor = torch.tensor(pad_items, dtype=torch.long, device=device)
        key_padding_mask = torch.tensor(masks, dtype=torch.bool, device=device)

        # Step 1: Item projection v_i = W_b e_i
        v_i = self.W_b(e_item[items_tensor])  # (num_users, n_max, behavior_dim)

        # Step 2: Learnable behavior histogram h_u
        logits = torch.matmul(v_i, self.anchors) / self.tau_anchor  # (num_users, n_max, num_anchors)
        alpha_ik = F.softmax(logits, dim=-1)
        alpha_ik = alpha_ik.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        h_u = alpha_ik.sum(dim=1)  # (num_users, num_anchors)

        # Step 3: Set attention refinement (MHA)
        H0 = v_i
        H1_attn, _ = self.mha(self.ln(H0), self.ln(H0), self.ln(H0), key_padding_mask=key_padding_mask, need_weights=False)
        H1 = H0 + torch.nan_to_num(H1_attn)

        # Step 4: Masked mean pooling v_pool_u
        valid_counts = (~key_padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
        v_pool_u = (H1 * (~key_padding_mask).unsqueeze(-1)).sum(dim=1) / valid_counts

        # Step 5: Fused behavioral embedding b_u
        fused_in = torch.cat([h_u, v_pool_u], dim=-1)  # (num_users, num_anchors + behavior_dim)
        b_u = self.mlp_fuse(fused_in)                  # (num_users, behavior_dim)
        return b_u


class ColdStartFallback(nn.Module):
    """
    Cold-start behavioral fallback MLP: R^k -> R^{d_b}.
    Computes b_u^cold from auxiliary features p_u for users with zero interaction history.
    """

    def __init__(self, feature_dim: int, behavior_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, behavior_dim),
            nn.LeakyReLU(),
            nn.Linear(behavior_dim, behavior_dim),
        )

    def forward(self, p_u: torch.Tensor) -> torch.Tensor:
        return self.mlp(p_u)


class LBPN(nn.Module):
    """
    Learnable Behavioral Prototype Network (LBPN).
    End-to-end differentiable behavioral set encoder + M learnable prototypes mu_c + soft assignment + soft semantic loss.
    """

    def __init__(
        self,
        feature_dim: int,
        item_dim: int,
        behavior_dim: int = 32,
        num_anchors: int = 8,
        num_prototypes: int = 5,
        tau_clust: float = 1.0,
        tau_sem: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.item_dim = item_dim
        self.behavior_dim = behavior_dim
        self.num_prototypes = num_prototypes
        self.tau_clust = tau_clust
        self.tau_sem = tau_sem

        # Behavioral Set Encoder
        self.set_encoder = BehavioralSetEncoder(
            item_dim=item_dim,
            behavior_dim=behavior_dim,
            num_anchors=num_anchors,
        )

        # Cold-Start Fallback MLP
        self.cold_start_mlp = ColdStartFallback(feature_dim, behavior_dim)

        # M Learnable Prototypes {mu_c}_{c=1}^M in R^{d_b}
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, behavior_dim))
        nn.init.xavier_uniform_(self.prototypes)

        # Projection matrix from h_u^S (d/2) to prototype dimension (d_b) for alignment
        self.h_s_proj = nn.Linear(item_dim, behavior_dim)

    def forward(
        self,
        user_items_dict: Dict[int, List[int]],
        e_item: torch.Tensor,
        p_u: torch.Tensor,
        h_s: torch.Tensor,
        batch_users: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(batch_users, int):
            batch_users = torch.arange(batch_users, device=p_u.device)
        # Compute behavioral embeddings b_u
        b_u = self.set_encoder(user_items_dict, e_item, batch_users, p_u.device)

        # Apply cold-start fallback for users with 0 interactions.
        # BUG-06 fix: in-place assignment b_u[u] = ... inside a loop corrupts the
        # autograd graph when b_u has a grad_fn. Use a vectorised torch.where instead.
        cold_mask = torch.tensor(
            [len(user_items_dict.get(u, [])) == 0 for u in batch_users.cpu().numpy()],
            dtype=torch.bool,
            device=b_u.device,
        )  # (bz,)
        if cold_mask.any():
            b_u_cold = self.cold_start_mlp(p_u[batch_users])                          # (bz, behavior_dim)
            b_u = torch.where(cold_mask.unsqueeze(-1), b_u_cold, b_u)    # safe, graph-preserving

        # Compute soft cluster assignments a_{u,c} = exp(-||b_u - mu_c||^2 / tau_clust) / sum_{c'} ...
        dist_sq = (b_u.unsqueeze(1) - self.prototypes.unsqueeze(0)).pow(2).sum(dim=-1)  # (batch_size, M)
        a_uc = F.softmax(-dist_sq / self.tau_clust, dim=-1)     # (batch_size, M)

        # Compute predicted semantic alignment p_{u,c} from h_u^S
        h_s_mapped = self.h_s_proj(h_s[batch_users])                        # (batch_size, d_b)
        scores = torch.matmul(h_s_mapped, self.prototypes.t()) / self.tau_sem  # (batch_size, M)
        p_uc = F.softmax(scores, dim=-1)                        # (batch_size, M)

        return b_u, a_uc, p_uc
