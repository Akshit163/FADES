import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GraphConv, GATConv
from typing import Tuple


class HeteroLightGCN(nn.Module):
    """
    LightGCN encoder operating on bipartite user-item interaction graph.
    Performs weighted neighborhood aggregation without non-linearities or feature transformations.
    """

    def __init__(self, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers

    def forward(
        self,
        x_user: torch.Tensor,
        x_item: torch.Tensor,
        edge_index: torch.Tensor,
        num_users: int,
        num_items: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        u_idx = edge_index[0]
        i_idx = edge_index[1]

        u_deg = torch.zeros(num_users, device=x_user.device).scatter_add_(
            0, u_idx, torch.ones_like(u_idx, dtype=torch.float)
        )
        i_deg = torch.zeros(num_items, device=x_item.device).scatter_add_(
            0, i_idx, torch.ones_like(i_idx, dtype=torch.float)
        )

        u_norm = torch.pow(u_deg, -0.5)
        u_norm[torch.isinf(u_norm)] = 0.0
        i_norm = torch.pow(i_deg, -0.5)
        i_norm[torch.isinf(i_norm)] = 0.0

        edge_weight = u_norm[u_idx] * i_norm[i_idx]

        u_emb_all = [x_user]
        i_emb_all = [x_item]

        curr_u, curr_i = x_user, x_item

        for _ in range(self.num_layers):
            msg_i2u = edge_weight.unsqueeze(-1) * curr_i[i_idx]
            next_u = torch.zeros_like(curr_u).scatter_add_(
                0, u_idx.unsqueeze(-1).expand_as(msg_i2u), msg_i2u
            )

            msg_u2i = edge_weight.unsqueeze(-1) * curr_u[u_idx]
            next_i = torch.zeros_like(curr_i).scatter_add_(
                0, i_idx.unsqueeze(-1).expand_as(msg_u2i), msg_u2i
            )

            curr_u, curr_i = next_u, next_i
            u_emb_all.append(curr_u)
            i_emb_all.append(curr_i)

        final_u = torch.mean(torch.stack(u_emb_all, dim=0), dim=0)
        final_i = torch.mean(torch.stack(i_emb_all, dim=0), dim=0)
        return final_u, final_i


class BaseEncoder(nn.Module):
    """
    General GNN Encoder supporting LightGCN, GraphConv, SAGEConv, and GATConv backbones.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int = 2,
        encoder_type: str = "LightGCN",
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.num_layers = num_layers

        if encoder_type == "LightGCN":
            self.gcn = HeteroLightGCN(num_layers=num_layers)
        else:
            self.convs = nn.ModuleList()
            for _ in range(num_layers):
                if encoder_type == "SAGE":
                    self.convs.append(SAGEConv((in_dim, in_dim), out_dim))
                elif encoder_type == "GAT":
                    self.convs.append(GATConv((in_dim, in_dim), out_dim, add_self_loops=False))
                else:  # GCN
                    self.convs.append(GraphConv((in_dim, in_dim), out_dim))

    def forward(
        self,
        x_user: torch.Tensor,
        x_item: torch.Tensor,
        edge_index: torch.Tensor,
        num_users: int,
        num_items: int,
    ) -> torch.Tensor:
        if self.encoder_type == "LightGCN":
            final_u, _ = self.gcn(x_user, x_item, edge_index, num_users, num_items)
            return final_u
        else:
            curr_u, curr_i = x_user, x_item
            for conv in self.convs:
                next_u = F.leaky_relu(conv((curr_i, curr_u), edge_index.flip(0)))
                curr_u = next_u
            return curr_u


class SlotGNN(nn.Module):
    """
    Slot-specific 2-layer GNN graph convolution.
    Aggregates neighbor item embeddings weighted by behavioral compatibility scores beta_{u,i,j}.
    """

    def __init__(self, item_dim: int, slot_dim: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.item_proj = nn.Linear(item_dim, slot_dim, bias=False)
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(slot_dim, slot_dim)) for _ in range(num_layers)]
        )
        for w in self.weights:
            nn.init.xavier_uniform_(w)

    def forward(
        self,
        z_uj: torch.Tensor,
        e_items: torch.Tensor,
        beta_uij: torch.Tensor,
        edge_index: torch.Tensor,
        num_users: int,
    ) -> torch.Tensor:
        u_idx = edge_index[0]
        i_idx = edge_index[1]

        # Project item embedding e_items from item_dim (stream_dim) to slot_dim
        e_items_proj = self.item_proj(e_items)  # (num_items, slot_dim)

        z_norm = torch.zeros(num_users, device=z_uj.device).scatter_add_(0, u_idx, beta_uij)
        z_norm = torch.clamp(z_norm, min=1e-8)
        norm_beta = beta_uij / z_norm[u_idx]

        curr_s = z_uj
        for l in range(self.num_layers):
            msg = norm_beta.unsqueeze(-1) * torch.matmul(e_items_proj, self.weights[l])[i_idx]
            next_s = torch.zeros_like(curr_s).scatter_add_(
                0, u_idx.unsqueeze(-1).expand_as(msg), msg
            )
            curr_s = F.leaky_relu(next_s)

        return curr_s
