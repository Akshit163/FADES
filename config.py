from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class SemDisRecConfig:
    """
    Hydra/Dataclass configuration for SemDisRec++ hyperparameters and paths.
    """
    # Dataset and Data loading
    dataset_name: str = "movielens"       # 'movielens', 'personality', or 'lastfm'
    base_dir: str = "./datasets"
    seed: int = 42
    batch_size: int = 512
    num_workers: int = 0
    min_interactions: int = 5
    n_max_items: int = 200                # Uniform random cap for behavioral set encoder

    # Feature and Embedding Dimensions
    feature_dim: int = 384                 # Auxiliary user feature dimension (p_u)
    item_feature_dim: int = 384            # Item feature dimension (p_i), if applicable
    embed_dim: int = 64                    # Total user representation dimension d (Stream R: d/2, Stream S: d/2)
    dz_dim: int = 32                       # Random seed dimension for ALNG noise generator
    behavior_dim: int = 32                 # Dimensionality db for LBPN behavioral vectors
    num_anchors: int = 8                   # Number of behavioral anchors Kb in LBPN histogram
    num_prototypes: int = 5                # Number of learnable prototypes M in LBPN

    # GNN and Stream S Architecture
    gnn_layers: int = 2
    gnn_encoder_type: str = "LightGCN"     # 'LightGCN', 'GCN', 'SAGE', or 'GAT'
    num_slots: int = 4                     # Number of semantic slots K in Stream S

    # Training and Optimization
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 30

    # Loss Weights (7-term loss objective)
    lambda_orth: float = 0.1               # lambda_1: Orthogonality disentanglement penalty
    lambda_sem: float = 0.1                # lambda_2: Learned soft semantic alignment loss
    lambda_ssl: float = 0               # lambda_SSL: Intra-stream contrastive loss for Stream R
    lambda_adv: float = 0.1               # lambda_adv: Leakage loss penalty (minus sign in main objective)
    lambda_div: float = 0.1               # lambda_div: Prototype diversity bonus weight
    lambda_l2: float = 1e-4                # lambda_3: L2 weight decay penalty

    # Temperatures
    tau_r: float = 0.1                     # Temperature for Stream R SSL InfoNCE loss
    tau_anchor: float = 1.0                # Temperature for LBPN anchor soft assignment
    tau_clust: float = 1.0                 # Temperature for LBPN prototype soft assignment
    tau_sem: float = 0.1                   # Temperature tau for predicted semantic alignment

    # Evaluation Protocol
    eval_protocol: str = "full"            # 'full' (all un-interacted items) or 'sampled' (1:99 negative sampling)
    k_list: List[int] = field(default_factory=lambda: [10, 20])
    num_negatives: int = 99                # Negative samples per test user (1:99 sampled ranking)
