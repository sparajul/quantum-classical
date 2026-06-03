"""
data/dataset.py — GraphDataset for particle track graphs (.pyg files).

Features
--------
  add_edge_features   : compute dr, dz, dphi, deta
  apply_hard_cuts     : remove electron edges (pdgId ±11)
  compute_edge_weights: per-edge loss weights via rule-based conditions
  GraphDataset        : torch Dataset loading serialised PyG graphs

Bug fix: _check_condition ``not_within`` branch compared ``torch.ones_like(mask)``
(a Tensor) directly in a Python ``if`` — always True, breaking the upper-bound
check for infinite hi. Now correctly uses a boolean sentinel.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ── Edge-feature computation ──────────────────────────────────────────────────

_EDGE_FEATURES       = frozenset({"dr", "dz", "dphi", "deta"})
_SKIP_IN_NODE_FILTER = frozenset({"edge_index", "y", "weights"} | _EDGE_FEATURES)


def add_edge_features(data, edge_features: List[str]):
    """
    Compute and attach requested edge features to a PyG Data object.

    Supported features
    ------------------
    dr   : r[dst] − r[src]
    dz   : z[dst] − z[src]
    dphi : phi[dst] − phi[src], wrapped to (−π, π] via atan2
    deta : eta[dst] − eta[src]
    """
    if not edge_features:
        return data

    src, dst = data.edge_index

    if "dr" in edge_features and not hasattr(data, "dr"):
        data.dr = data.r[dst] - data.r[src]

    if "dz" in edge_features and not hasattr(data, "dz"):
        data.dz = data.z[dst] - data.z[src]

    needs_dphi = any(f in edge_features for f in ("dphi", "phislope", "rphislope"))
    if needs_dphi and not hasattr(data, "dphi"):
        raw = data.phi[dst] - data.phi[src]
        data.dphi = torch.atan2(torch.sin(raw), torch.cos(raw))

    if "deta" in edge_features and not hasattr(data, "deta"):
        if hasattr(data, "eta"):
            data.deta = data.eta[dst] - data.eta[src]
        else:
            logger.warning("deta requested but graph has no 'eta' attribute — skipping")

    if "phislope" in edge_features and not hasattr(data, "phislope"):
        dr = data.r[dst] - data.r[src]
        data.phislope = data.dphi / (dr + 1e-8)

    if "rphislope" in edge_features and not hasattr(data, "rphislope"):
        dr    = data.r[dst] - data.r[src]
        r_avg = 0.5 * (data.r[src] + data.r[dst])
        data.rphislope = r_avg * data.dphi / (dr + 1e-8)

    return data


# ── Attribute resolution ──────────────────────────────────────────────────────

def _resolve_edge_field(data, key: str, src, dst):
    """
    Resolve an attribute to a per-edge tensor, handling three storage layouts
    found in GNN4ITk / ACTS tracking graphs:

    1. Edge-level     shape [n_edges]     — use directly
    2. Node-level     shape [n_nodes]     — average over src/dst endpoints
    3. Particle-level shape [n_particles] — look up via data.pid (edge→particle map)

    Returns None if the attribute cannot be resolved (warning already logged).
    """
    raw = getattr(data, key)
    n_e = data.edge_index.shape[1]
    if n_e == 0:
        return raw.new_zeros(0)
    n_v = int(max(src.max(), dst.max()).item()) + 1

    if raw.shape[0] == n_e:
        return raw

    if raw.shape[0] == n_v:
        return 0.5 * (raw[src].float() + raw[dst].float())

    # Particle-level: resolve via data.pid (edge → particle index)
    if hasattr(data, "pid") and data.pid.shape[0] == n_e:
        pid_idx = data.pid.long()
        if pid_idx.max() < raw.shape[0]:
            return raw[pid_idx].float()

    logger.debug(
        "Attribute '%s' shape %s cannot be resolved to edges "
        "(n_edges=%d, n_nodes=%d) — skipping condition",
        key, tuple(raw.shape), n_e, n_v,
    )
    return None


# ── Hard cuts ──────────────────────────────────────────────────────────────────

def apply_hard_cuts(data, hparams: dict):
    """
    Remove edges based on hard cuts defined in hparams.

    remove_electrons : remove edges from electron tracks (pdgId ±11).
    pt_cut           : remove edges where resolved pt < pt_cut (GeV).
                       Applies to all edges; background edges with pt=0 are removed too.
    """
    if hparams.get("remove_electrons", False):
        if not hasattr(data, "pdgId"):
            logger.debug("remove_electrons=True but graph has no pdgId — skipping")
        else:
            src, dst   = data.edge_index
            edge_pdgid = _resolve_edge_field(data, "pdgId", src, dst)
            if edge_pdgid is None:
                logger.warning(
                    "Could not resolve pdgId to edge-level — skipping electron removal. "
                    "Ensure data.pid (edge→particle index) is present in your graphs."
                )
            else:
                keep = edge_pdgid.abs() != 11
                if not keep.all():
                    logger.debug("Removing %d electron edges", (~keep).sum().item())
                    data = _filter_edges(data, keep)

    pt_cut = hparams.get("pt_cut")
    if pt_cut is not None:
        if not hasattr(data, "pt"):
            logger.debug("pt_cut=%.2f set but graph has no pt attribute — skipping", pt_cut)
        else:
            src, dst = data.edge_index
            edge_pt  = _resolve_edge_field(data, "pt", src, dst)
            if edge_pt is not None:
                keep = edge_pt >= pt_cut
                if not keep.all():
                    logger.debug(
                        "pt_cut=%.2f GeV: removing %d / %d edges",
                        pt_cut, (~keep).sum().item(), keep.shape[0],
                    )
                    data = _filter_edges(data, keep)

    data = _remove_isolated_nodes(data)
    return data


# ── Sample weighting ───────────────────────────────────────────────────────────

def _check_condition(data, conditions: dict) -> torch.Tensor:
    """
    Return boolean edge mask where ALL conditions hold.

    Condition value formats
    -----------------------
    bool                       : field == bool
    [lo, hi]                   : lo <= field < hi  (.inf ok)
    ["not_within", [lo, hi]]   : NOT in range
    ["not_in", [v1, v2, ...]]  : field not in list

    Attributes are resolved to edge-level via ``_resolve_edge_field``,
    which handles edge-level, node-level, and particle-level storage.
    """
    src, dst = data.edge_index
    n_edges  = data.edge_index.shape[1]
    mask     = torch.ones(n_edges, dtype=torch.bool)

    for key, val in conditions.items():
        if key == "y":
            field = data.y.bool()
        elif hasattr(data, key):
            field = _resolve_edge_field(data, key, src, dst)
            if field is None:
                continue  # warning already logged inside helper
        else:
            logger.debug("Condition key '%s' not on graph — skipping", key)
            continue

        if isinstance(val, bool):
            mask &= field.bool() == val

        elif isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
            op, operand = val
            if op == "not_within":
                lo   = float(operand[0])
                hi_s = str(operand[1])
                hi   = math.inf if hi_s in (".inf", "inf") else float(operand[1])
                in_range = field >= lo
                if hi < math.inf:
                    in_range &= field < hi
                mask &= ~in_range
            elif op == "not_in":
                for v in operand:
                    mask &= field != float(v)

        elif isinstance(val, list) and len(val) == 2:
            lo   = float(val[0])
            hi_s = str(val[1])
            hi   = math.inf if hi_s in (".inf", "inf") else float(val[1])
            mask &= field >= lo
            if hi < math.inf:
                mask &= field < hi

        else:
            mask &= field == val

    return mask


def compute_edge_weights(data, weighting: list) -> torch.Tensor:
    """Compute per-edge loss weights; last matching rule wins."""
    weights = torch.ones(data.edge_index.shape[1])
    for rule in weighting:
        w    = float(rule["weight"])
        cond = rule.get("conditions", {})
        mask = _check_condition(data, cond)
        weights[mask] = w
    return weights


# ── Edge-index filtering helper ────────────────────────────────────────────────

def _filter_edges(data, keep: torch.Tensor):
    """Apply a boolean edge mask to edge_index, y, and all edge attributes."""
    data.edge_index = data.edge_index[:, keep]
    data.y          = data.y[keep]
    for attr in _EDGE_FEATURES | {"weights"}:
        if hasattr(data, attr):
            setattr(data, attr, getattr(data, attr)[keep])
    return data


def _remove_isolated_nodes(data):
    """Remove nodes with no incident edges and re-index edge_index to stay contiguous."""
    n_old = data.num_nodes
    if not n_old or data.edge_index.shape[1] == 0:
        return data

    used = torch.zeros(n_old, dtype=torch.bool)
    used[data.edge_index[0]] = True
    used[data.edge_index[1]] = True
    if used.all():
        return data

    # Contiguous re-mapping: old index → new index (-1 = removed)
    new_idx = torch.full((n_old,), -1, dtype=torch.long)
    new_idx[used] = torch.arange(used.sum().item(), dtype=torch.long)
    data.edge_index = new_idx[data.edge_index]

    # Filter every tensor whose first dim equals n_old, skipping known edge-level attrs.
    # In tracking graphs n_nodes (~10K) << n_edges (~40K), so shape is unambiguous.
    for key in list(data.keys()):
        if key in _SKIP_IN_NODE_FILTER:
            continue
        val = getattr(data, key, None)
        if isinstance(val, torch.Tensor) and val.shape[0] == n_old:
            setattr(data, key, val[used])

    return data


# ── Dataset ───────────────────────────────────────────────────────────────────

class GraphDataset(Dataset):
    """
    Loads serialised PyG graphs (``.pyg`` files) from disk.

    Parameters
    ----------
    home_dir   : Base path containing split subdirectories.
    sub_dir    : Subdirectory name, e.g. ``"train_set/"``.
    preprocess : Apply edge-feature computation, hard cuts, and weighting.
    hparams    : Training hyperparameters dict.
    max_events : Cap on number of files loaded (for quick iteration).
    """

    def __init__(
        self,
        home_dir: str,
        sub_dir: str,
        preprocess: bool = True,
        hparams: Optional[Dict] = None,
        max_events: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.base_path  = os.path.join(home_dir, sub_dir)
        self.preprocess = preprocess
        self.hparams    = hparams or {}
        self.file_names = self._scan_files()

        if max_events is not None:
            self.file_names = self.file_names[:max_events]

        pt_cut = self.hparams.get("pt_cut")
        logger.info(
            "GraphDataset | path=%s | files=%d | remove_electrons=%s | "
            "pt_cut=%s | weighting=%s",
            self.base_path, len(self.file_names),
            self.hparams.get("remove_electrons", False),
            ("%.2f GeV" % pt_cut) if pt_cut is not None else "none",
            "yes" if self.hparams.get("weighting") else "no",
        )
        if preprocess:
            self._log_dataset_stats()

    def _log_dataset_stats(self) -> None:
        """Scan all graphs and log mean node/edge counts before and after hard cuts."""
        n_before, e_before, n_after, e_after = [], [], [], []
        for fname in self.file_names:
            data = self._load(os.path.join(self.base_path, fname))
            n_before.append(data.num_nodes or 0)
            e_before.append(data.edge_index.shape[1])
            data = apply_hard_cuts(data, self.hparams)
            n_after.append(data.num_nodes or 0)
            e_after.append(data.edge_index.shape[1])
        mean = lambda lst: sum(lst) / max(len(lst), 1)
        logger.info(
            "Dataset stats (mean per graph) | "
            "nodes: %d → %d (%.1f%% kept) | "
            "edges: %d → %d (%.1f%% kept)",
            round(mean(n_before)), round(mean(n_after)),
            100.0 * mean(n_after) / max(mean(n_before), 1),
            round(mean(e_before)), round(mean(e_after)),
            100.0 * mean(e_after) / max(mean(e_before), 1),
        )

    def _scan_files(self) -> List[str]:
        if not os.path.isdir(self.base_path):
            raise FileNotFoundError(f"Dataset directory not found: {self.base_path}")
        files = sorted(f for f in os.listdir(self.base_path) if f.endswith(".pyg"))
        if not files:
            raise RuntimeError(f"No .pyg files found in {self.base_path}")
        return files

    @staticmethod
    def _load(path: str):
        data = torch.load(path, weights_only=False)
        # Remove stale score attributes from previous runs
        if hasattr(data, "scores"):
            del data.scores
        return data

    def _preprocess(self, data):
        data = add_edge_features(data, self.hparams.get("edge_features") or [])
        data = apply_hard_cuts(data, self.hparams)

        weighting = self.hparams.get("weighting")
        if weighting:
            data.weights = compute_edge_weights(data, weighting)

        max_size = self.hparams.get("max_training_graph_size")
        if max_size and data.edge_index.shape[1] > max_size:
            data = self._trim_graph(data, max_size)

        return data

    @staticmethod
    def _trim_graph(data, max_size: int):
        """Randomly subsample edges to at most max_size (deterministic sort for reproducibility)."""
        n   = data.edge_index.shape[1]
        idx, _ = torch.randperm(n)[:max_size].sort()
        return _filter_edges(data, idx)   # reuse filter helper

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int):
        path = os.path.join(self.base_path, self.file_names[idx])
        data = self._load(path)
        if self.preprocess:
            data = self._preprocess(data)
        return data
