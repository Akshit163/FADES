from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..encoders import BaseEncoder


class NoiseGenerator(nn.Module):
    """
    Adversarially Learned Noise Generator (ALNG) G_theta: R^{d_z} -> R^{d/2}.
    2-layer MLP with LeakyReLU activations.
    """

    def __init__(self, dz_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dz_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, z_u: torch.Tensor) -> torch.Tensor:
        # [PP] Sec. 1.1.1, ALNG: G_θ: z_u → n_u  (2-layer MLP with LeakyReLU, no output activation)
        h = F.leaky_relu(self.fc1(z_u))
        n_u = self.fc2(h)  # no final activation
        return n_u


class LeakageDiscriminator(nn.Module):
    """
    Auditor Discriminator D_phi: R^{d/2} -> R^k.
    2-layer MLP attempting to regress auxiliary features p_u from Stream R output h_u^R.
    """

    def __init__(self, in_dim: int, feature_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, feature_dim)

    def forward(self, h_r: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.fc1(h_r))
        p_hat = self.fc2(h)
        return p_hat


class GraphIntensityGate(nn.Module):
    """
    Graph-Conditional Noise Intensity Gate gamma(u) = sigma(MLP_gamma(g_u)) in (0, 1).
    Conditioned on local graph topology g_u = [deg(u), ||e_u||_2, CC(u)]^T in R^3.
    """

    def __init__(self, stats_dim: int = 3, hidden_dim: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(stats_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, g_u: torch.Tensor) -> torch.Tensor:
        # [PP] Sec. 1.1.2, gate: γ(u) = σ(MLP_γ(g_u)) ∈ (0,1)
        # g_u = [log1p(deg(u)), ||e_u^R||_2, CC(u)]^T ∈ R^3 (updated per epoch)
        return self.mlp(g_u)


class StreamR(nn.Module):
    """
    AdaReg: Adaptive Regularization Stream.
    Receives base embeddings e_u, intensity-gated synthetic noise gamma(u) * n_u, and interaction graph.
    Processes zero authentic auxiliary features p_u (enforces hard structural separation).
    """

    def __init__(
        self,
        embed_dim: int,
        dz_dim: int = 32,
        gnn_layers: int = 2,
        encoder_type: str = "LightGCN",
    ):
        super().__init__()
        self.stream_dim = embed_dim // 2
        self.dz_dim = dz_dim

        self.noise_generator = NoiseGenerator(dz_dim, self.stream_dim)
        self.intensity_gate = GraphIntensityGate(stats_dim=3)
        self.gnn_encoder = BaseEncoder(
            in_dim=self.stream_dim,
            out_dim=self.stream_dim,
            num_layers=gnn_layers,
            encoder_type=encoder_type,
        )

    def forward_single_view(
        self,
        e_user: torch.Tensor,
        e_item: torch.Tensor,
        g_u: torch.Tensor,
        edge_index: torch.Tensor,
        num_users: int,
        num_items: int,
    ) -> torch.Tensor:
        """Single view forward pass with fresh random seed z_u."""
        batch_size = e_user.size(0)
        # [PP] Sec. 1.1.1: z_u ~ N(0,I) resampled fresh per forward (NOT stored — no embedding table)
        z_u = torch.randn(batch_size, self.dz_dim, device=e_user.device)

        # [PP] Sec. 1.1.1: n_u = G_θ(z_u)
        n_u = self.noise_generator(z_u)

        # [PP] Sec. 1.1.2: γ(u) ∈ (0,1) — scalar gate per user, no gradient into g_u
        gamma_u = self.intensity_gate(g_u)

        # [PP] Sec. 1.1.2: x̃_u = e_u + γ(u) ⊙ n_u  (γ scalar broadcasts over d/2 dims)
        x_tilde_user = e_user + gamma_u * n_u

        # GNN propagation over bipartite interaction graph
        h_r = self.gnn_encoder(
            x_tilde_user, e_item, edge_index, num_users, num_items
        )
        return h_r

    def forward_dual_view(
        self,
        e_user: torch.Tensor,
        e_item: torch.Tensor,
        g_u: torch.Tensor,
        edge_index: torch.Tensor,
        num_users: int,
        num_items: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dual view forward pass for intra-stream contrastive loss L_SSL^R."""
        h_r1 = self.forward_single_view(
            e_user, e_item, g_u, edge_index, num_users, num_items
        )
        h_r2 = self.forward_single_view(
            e_user, e_item, g_u, edge_index, num_users, num_items
        )
        return h_r1, h_r2
