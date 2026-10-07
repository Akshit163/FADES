"""
losses.py — SemDisRec++ loss objective (7 terms).

All formulas reference:
  [BASE]  Disentangling_Semantics_from_Regularization__SimDisRec_.pdf
  [PP]    simdesrec (2).pdf (SemDisRec++ upgrade)
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


def bce_link_loss(
    pos_scores: torch.Tensor, neg_scores: torch.Tensor
) -> torch.Tensor:
    """
    Binary Cross-Entropy link prediction loss over positive and negative user-item pairs.
    [BASE] Eq. 12:
      L_link = − Σ_{(u,i)∈R} log σ(y_ui) − Σ_{(u,j)∉R} log(1 − σ(y_uj))
    Implementation: MEAN over pos/neg pairs so that L_link magnitude is O(1) and
    batch-size-independent, consistent with all other loss terms (L_orth, L_sem,
    L_ssl, L_leak, L_div). Using SUM caused L_link to scale linearly with batch_size,
    making the effective strength of all λ weights implicitly batch-size-dependent.
    """
    # [BASE] Eq. 12 — mean over positive pairs (batch-size invariant)
    pos_loss = -F.logsigmoid(pos_scores).mean()
    # [BASE] Eq. 12 — mean over negative pairs (batch-size invariant)
    neg_loss = -F.logsigmoid(-neg_scores).mean()
    return pos_loss + neg_loss


def orthogonality_loss(
    h_r: torch.Tensor, h_s: torch.Tensor
) -> torch.Tensor:
    """
    Batch-level Frobenius norm penalty on the centered cross-covariance between
    Stream R and Stream S.
    [BASE] Sec. 4.3:
      L_orth = ||(1/|B|) Σ_{u∈B} (h_u^R − h̄^R)(h_u^S − h̄^S)^T||_F²
    Mean-centering is applied before computing the cross-covariance (required).
    """
    batch_size = h_r.size(0)
    if batch_size <= 1:
        return torch.tensor(0.0, device=h_r.device)

    # [BASE] Sec. 4.3 — center embeddings across batch before cross-covariance
    h_r_centered = h_r - h_r.mean(dim=0, keepdim=True)
    h_s_centered = h_s - h_s.mean(dim=0, keepdim=True)

    # Cross-covariance matrix C ∈ R^{(d/2) × (d/2)}
    cross_cov = torch.matmul(h_r_centered.t(), h_s_centered) / float(batch_size)
    return torch.norm(cross_cov, p="fro") ** 2


def intra_stream_contrastive_loss(
    h_r1: torch.Tensor, h_r2: torch.Tensor, tau: float = 0.1
) -> torch.Tensor:
    """
    Intra-stream contrastive consistency loss for Stream R across two independent
    noise-resampled view realizations.
    [PP] Sec. 1.1.1, L_SSL^R — NT-Xent / InfoNCE formulation:
      L_SSL^R = −(1/|B|) Σ_u log [exp(sim(h_u^{R,(1)}, h_u^{R,(2)}) / τ) /
                                   Σ_{u'} exp(sim(h_u^{R,(1)}, h_{u'}^{R,(2)}) / τ)]
    Implementation: F.cross_entropy on the full (B × B) cosine-similarity matrix.
    Off-diagonal entries are all negatives; diagonal entries are the positives.
    """
    batch_size = h_r1.size(0)
    if batch_size <= 1:
        return torch.tensor(0.0, device=h_r1.device)

    h1_norm = F.normalize(h_r1, dim=-1)
    h2_norm = F.normalize(h_r2, dim=-1)

    # [PP] Sec. 1.1.1 — cosine similarity matrix, scaled by temperature τ_R
    sim_matrix = torch.matmul(h1_norm, h2_norm.t()) / tau
    # Labels: index i is the positive pair for row i (diagonal)
    labels = torch.arange(batch_size, device=h_r1.device)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss


def leakage_loss(
    p_hat: torch.Tensor, p_u: torch.Tensor
) -> torch.Tensor:
    """
    Audit discriminator leakage loss: MSE between predicted auxiliary features
    and true p_u.
    [PP] Sec. 1.1.1, L_leak:
      L_leak = (1/|B|) Σ_{u∈B} ||p̂_u − p_u||_2²
    """
    # [PP] Sec. 1.1.1 — mean MSE over batch
    return F.mse_loss(p_hat, p_u, reduction="mean")


def prototype_diversity_loss(prototypes: torch.Tensor) -> torch.Tensor:
    """
    Prototype diversity loss to prevent prototype collapse in LBPN.
    [PP] Sec. 1.2.2, L_div:
      L_div = −(1/C(M,2)) Σ_{1 ≤ c < c' ≤ M} ||μ_c − μ_{c'}||_2²
    The MINUS sign is explicit: minimising L_div pushes prototypes apart.
    Normalised by the number of unique pairs C(M,2) = M*(M-1)/2 so that
    lambda_div has consistent magnitude regardless of num_prototypes M.
    """
    M = prototypes.size(0)
    if M <= 1:
        return torch.tensor(0.0, device=prototypes.device)

    dist_sq = (prototypes.unsqueeze(1) - prototypes.unsqueeze(0)).pow(2).sum(dim=-1)
    # [PP] Sec. 1.2.2 — strictly upper triangular entries (c < c')
    triu_indices = torch.triu_indices(M, M, offset=1)
    # Normalise by C(M,2) so loss is O(1) and M-independent
    num_pairs = max(1.0, M * (M - 1) / 2.0)
    # Negative sign: loss is negative so minimizing it maximises pairwise distances
    loss = -dist_sq[triu_indices[0], triu_indices[1]].sum() / num_pairs
    return loss


def learned_semantic_alignment_loss(
    a_uc: torch.Tensor, p_uc: torch.Tensor
) -> torch.Tensor:
    """
    Soft cross-entropy alignment loss between behavioral cluster assignment a_{u,c}
    and predicted semantic alignment p_{u,c}.
    [PP] Sec. 1.2.3, L_sem:
      L_sem = − Σ_{u∈B} Σ_{c=1}^{M} a_{u,c} · log p_{u,c}
    Averaged over batch (mean over u, sum over c).
    """
    eps = 1e-8
    # [PP] Sec. 1.2.3 — clamp to avoid log(0)
    log_p = torch.log(torch.clamp(p_uc, min=eps))
    loss = -(a_uc * log_p).sum(dim=-1).mean()
    return loss


def l2_regularization_loss(model: nn.Module) -> torch.Tensor:
    """
    L2 weight decay penalty over all trainable weight matrices in model.
    [BASE] Sec. 4.8:
      L_l2 = Σ_{W ∈ Weights} ||W||_F²  (= Σ_{W} Σ_{i,j} W_{ij}²)
    Includes: all nn.Linear / nn.Conv weight matrices (names containing 'weight').
    Excludes: embedding tables (user_emb_R, item_emb_R, etc.) and learnable
              prototype / query vectors (no 'weight' in their name).
    Implementation: torch.norm(param, p=2)**2 == torch.sum(param**2) (Frobenius).
    """
    l2 = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        # [BASE] Sec. 4.8 — only include named weight matrices, exclude embedding tables
        if param.requires_grad and "weight" in name:
            l2 = l2 + torch.norm(param, p=2) ** 2
    return l2


@dataclass
class SemDisRecLosses:
    link_loss: torch.Tensor
    orth_loss: torch.Tensor
    sem_loss: torch.Tensor
    ssl_loss: torch.Tensor
    leak_loss: torch.Tensor
    div_loss: torch.Tensor
    l2_loss: torch.Tensor
    total_loss: torch.Tensor
