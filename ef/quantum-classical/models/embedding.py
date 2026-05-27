"""Hit embedding network for metric-learning graph construction.

Acorn-style training:
  - Loss: weighted hinge loss on squared L2 distances (margin-based).
  - Hard Negative Mining (HNM): radius search in *current* embedding space
    finds the hardest negatives (non-track pairs that happen to be close).
  - Signal injection: ALL consecutive same-particle doublets are included
    every step (no sampling loss).
  - Random pairs: for training stability and coverage of easy negatives.

This replaces the previous NTXent/SimCLR approach, which does not support
HNM and treats all non-paired samples equally (missing the hard cases).

Architecture:
  Input: 5 features [r/1000, sin(φ), cos(φ), z/1000, η/4]
  sin/cos encoding makes φ periodic-safe.
  Residual MLP with LayerNorm + GELU → L2-normalised output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Input preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_hits(r, phi, z, eta):
    """Raw hit coordinates → 5D model input, all features in [-1, 1].

    sin/cos for phi correctly handles the circular geometry: hits at
    φ=+π and φ=-π have zero angular distance, not distance=2.
    """
    return torch.stack([
        r   / 1000.0,
        torch.sin(phi),
        torch.cos(phi),
        z   / 1000.0,
        eta / 4.0,
    ], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class _ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class HitEmbedding(nn.Module):
    """Residual MLP: 5-D hit features → L2-normalised embedding.

    Args:
        embed_dim : Output embedding dimension (default 16).
        hidden    : Hidden layer width (default 256).
        n_blocks  : Number of residual blocks (default 4).
        dropout   : Dropout rate inside residual blocks (default 0.1).
    """

    INPUT_DIM = 5   # r/1000, sin(φ), cos(φ), z/1000, η/4

    def __init__(self, embed_dim: int = 16, hidden: int = 256,
                 n_blocks: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        self.input_proj = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[_ResBlock(hidden, dropout) for _ in range(n_blocks)]
        )
        self.output_proj = nn.Linear(hidden, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 5) preprocessed features → (N, embed_dim) unit vectors."""
        h = self.input_proj(x)
        h = self.blocks(h)
        return F.normalize(self.output_proj(h), dim=-1)

    @staticmethod
    def preprocess(r, phi, z, eta):
        return preprocess_hits(r, phi, z, eta)


# ─────────────────────────────────────────────────────────────────────────────
# Edge building
# ─────────────────────────────────────────────────────────────────────────────

def build_edges_radius(embedding: torch.Tensor,
                       r_max: float,
                       k_max: int = 50) -> torch.Tensor:
    """Find all hit pairs within L2 distance r_max in embedding space.

    Uses PyG's `radius` function as the fallback (FRNN if available).
    Returns edge_index of shape (2, E) with self-loops removed.
    """
    try:
        import frnn
        dists, idxs, _, _ = frnn.frnn_grid_points(
            embedding.unsqueeze(0), embedding.unsqueeze(0),
            lengths1=None, lengths2=None,
            K=k_max, r=r_max, grid=None,
            return_nn=False, return_sorted=True,
        )
        idxs = idxs.squeeze(0).int()
        ind = (torch.arange(idxs.shape[0], device=embedding.device)
               .repeat(idxs.shape[1], 1).T.int())
        pos = idxs >= 0
        edge_index = torch.stack([ind[pos], idxs[pos]]).long()
    except ImportError:
        from torch_geometric.nn import radius as pyg_radius
        edge_index = pyg_radius(embedding, embedding, r=r_max,
                                max_num_neighbors=k_max)

    # Remove self-loops
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask]


def build_consecutive_doublets(pid: torch.Tensor,
                               layer_key: torch.Tensor,
                               pt: torch.Tensor,
                               min_pt: float = 0.5) -> torch.Tensor:
    """Build consecutive same-particle hit pairs sorted by detector layer.

    For each particle with pT >= min_pt, hits are sorted by their layer
    key (vol*1000 + layer) and adjacent pairs are returned as doublets.
    These are the 'ground truth' edges the embedding should learn to
    bring close together in embedding space.

    Returns edge_index (2, E) with global hit indices.
    """
    signal_mask = (pt >= min_pt) & (pid != 0)
    if signal_mask.sum() < 2:
        return torch.empty((2, 0), dtype=torch.long, device=pid.device)

    global_idx = signal_mask.nonzero(as_tuple=True)[0]
    sig_pid = pid[global_idx]
    sig_lk  = layer_key[global_idx]

    src_list, dst_list = [], []
    for p in sig_pid.unique():
        mask = sig_pid == p
        hits = global_idx[mask]
        if hits.numel() < 2:
            continue
        order = sig_lk[mask].argsort()
        h = hits[order]
        src_list.append(h[:-1])
        dst_list.append(h[1:])

    if not src_list:
        return torch.empty((2, 0), dtype=torch.long, device=pid.device)

    return torch.stack([torch.cat(src_list), torch.cat(dst_list)])


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def weighted_hinge_loss(d: torch.Tensor,
                        truth: torch.Tensor,
                        margin: float = 1.0,
                        pos_weight: float = 1.0) -> torch.Tensor:
    """Hinge loss on squared L2 distances.

    Positives (truth=1): loss = d          → pulls d toward 0
    Negatives (truth=0): loss = max(0, margin²-d) → pushes d above margin²

    Since embeddings are L2-normalised, d ∈ [0, 4].
    Set margin = r_train so the loss boundary matches the efficiency radius.

    Args:
        d          : Squared L2 distances (E,)
        truth      : Binary edge labels (E,) — 1 = same particle
        margin     : Hinge boundary in L2 units (applied as margin²)
        pos_weight : Upweight positive (signal) loss to counter class imbalance.
                     With HNM purity ~10%, set pos_weight ≈ 5–10.
    """
    margin_sq = margin ** 2

    pos_mask = truth == 1
    neg_mask = truth == 0

    losses = []
    if pos_mask.any():
        losses.append(pos_weight * F.hinge_embedding_loss(
            d[pos_mask],
            torch.ones(pos_mask.sum(), device=d.device),
            margin=margin_sq,
            reduction='mean',
        ))
    if neg_mask.any():
        losses.append(F.hinge_embedding_loss(
            d[neg_mask],
            -torch.ones(neg_mask.sum(), device=d.device),
            margin=margin_sq,
            reduction='mean',
        ))

    return sum(losses) if losses else d.sum() * 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Deprecated — kept for backward compatibility only
# ─────────────────────────────────────────────────────────────────────────────

class HitPairDataset:
    """Deprecated: use event-level HNM training instead."""
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "HitPairDataset is deprecated. "
            "Use the Acorn-style event-level training in train_embedding.py."
        )


def ntxent_loss(*args, **kwargs):
    raise RuntimeError(
        "ntxent_loss is deprecated. "
        "Use weighted_hinge_loss with hard negative mining instead."
    )


def contrastive_loss(emb_i, emb_j, labels, margin: float = 1.0):
    """Hinge contrastive loss (kept for backward compatibility)."""
    dist = F.pairwise_distance(emb_i, emb_j)
    pos  = labels       * dist.pow(2)
    neg  = (1 - labels) * F.relu(margin - dist).pow(2)
    return (pos + neg).mean()
