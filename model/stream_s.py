import torch
import torch.nn as nn
import torch.nn.functional as F
from ..encoders import SlotGNN


class PriorExtractionNetwork(nn.Module):
    """
    Lightweight convolutional/linear prior extraction network phi_prior: R^k -> R^d.
    Pre-trained or initialized to produce degraded prior P. Frozen during joint training.
    """

    def __init__(self, feature_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
        )

    def forward(self, p_u: torch.Tensor) -> torch.Tensor:
        return self.net(p_u)


class StreamS(nn.Module):
    """
    Semantic Slot Attention Encoder (Stream S).
    Processes authentic auxiliary feature vector p_u through K learnable semantic slots,
    weights GNN neighborhood aggregation by per-slot behavioral compatibility scores beta_{u,i,j},
    and concatenates slot representations into h_u^S in R^{d/2}.
    """

    def __init__(
        self,
        feature_dim: int,
        embed_dim: int,
        num_slots: int = 4,
        gnn_layers: int = 2,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.stream_dim = embed_dim // 2
        self.slot_dim = self.stream_dim // num_slots

        # Prior extraction network phi_prior
        self.prior_net = PriorExtractionNetwork(feature_dim, embed_dim)

        # Slot query vectors q_j in R^k
        self.slot_queries = nn.Parameter(torch.randn(num_slots, feature_dim))
        nn.init.xavier_uniform_(self.slot_queries)

        # Slot weight matrices W_j in R^{(d / 2K) x k}
        self.slot_weights = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.slot_dim, feature_dim))
                for _ in range(num_slots)
            ]
        )
        for w in self.slot_weights:
            nn.init.xavier_uniform_(w)

        # Shared bilinear compatibility matrix W_compat in R^{(d / 2K) x (d / 2)}
        self.W_compat = nn.Parameter(torch.randn(self.slot_dim, self.stream_dim))
        nn.init.xavier_uniform_(self.W_compat)

        # Slot GNNs
        self.slot_gnns = nn.ModuleList(
            [
                SlotGNN(item_dim=self.stream_dim, slot_dim=self.slot_dim, num_layers=gnn_layers)
                for _ in range(num_slots)
            ]
        )

    def forward_with_attention(
        self,
        p_u: torch.Tensor,
        e_item: torch.Tensor,
        edge_index: torch.Tensor,
        num_users: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [PP] Sec. 1.2.1, slot attention: α_{u,j} = softmax_j(p_u^T q_j / √k)
        # Scaling factor is √k where k = feature_dim (dimension of p_u and slot queries)
        attn_scores = torch.matmul(p_u, self.slot_queries.t()) / (self.feature_dim ** 0.5)
        alpha_u = F.softmax(attn_scores, dim=-1)  # (batch_size, K) — sums to 1 over K slots

        u_idx = edge_index[0]
        i_idx = edge_index[1]

        slot_outputs = []

        for j in range(self.num_slots):
            alpha_uj = alpha_u[:, j].unsqueeze(-1)
            z_uj = alpha_uj * torch.matmul(p_u, self.slot_weights[j].t())

            z_proj = torch.matmul(z_uj, self.W_compat)
            z_proj_u = z_proj[u_idx]
            e_item_i = e_item[i_idx]
            beta_uij = torch.sigmoid(torch.sum(z_proj_u * e_item_i, dim=-1))

            s_uj = self.slot_gnns[j](
                z_uj, e_item, beta_uij, edge_index, num_users
            )
            slot_outputs.append(s_uj)

        h_s = torch.cat(slot_outputs, dim=-1)
        return h_s, alpha_u

    def forward(
        self,
        p_u: torch.Tensor,
        e_item: torch.Tensor,
        edge_index: torch.Tensor,
        num_users: int,
    ) -> torch.Tensor:
        h_s, alpha_u = self.forward_with_attention(p_u, e_item, edge_index, num_users)
        return h_s
