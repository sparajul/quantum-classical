"""
utils/metrics.py
─────────────────
Physics-motivated metrics for particle track edge classification.

Tracking Metrics
----------------
  Efficiency (Recall / True Positive Rate)
      = TP / (TP + FN)
      "Of all true track edges, what fraction did we keep?"
      → Want this HIGH (target ≥ 0.95 for publication)

  Purity (Precision)
      = TP / (TP + FP)
      "Of all edges we kept, what fraction are real track edges?"
      → Want this HIGH (higher purity = less noise for downstream reco)

  Fake Rate (False Discovery Rate)
      = FP / (TP + FP)  = 1 - Purity
      "Of edges we kept, what fraction are fake?"
      → Want this LOW

  F1 Score
      = 2 * (Purity * Efficiency) / (Purity + Efficiency)
      → Harmonic mean; penalises imbalance between the two

  AUC (Area Under ROC Curve)
      → Threshold-independent discriminating power [0.5 random, 1.0 perfect]

Standard ML Metrics
-------------------
  Accuracy, Binary Cross-Entropy Loss
  Confusion matrix components: TP, FP, TN, FN (raw counts)

Publication Metrics
-------------------
  ROC curve data (fpr, tpr arrays) for plotting
  Efficiency vs purity curve (threshold sweep)
  Score distributions for signal and background

All functions operate on raw logits + binary labels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrackingMetrics:
    """
    Container for all metrics computed in one step/epoch.
    All scalar values are plain Python floats for easy WandB logging.
    """
    # Core tracking physics metrics
    efficiency:  float = 0.0   # TP / (TP + FN)   — recall
    purity:      float = 0.0   # TP / (TP + FP)   — precision
    fake_rate:   float = 0.0   # FP / (TP + FP)   — 1 - purity
    f1:          float = 0.0   # harmonic mean of efficiency & purity

    # Standard ML metrics
    auc:         float = 0.0   # ROC-AUC
    avg_precision: float = 0.0 # Average Precision (PR-AUC)
    accuracy:    float = 0.0   # (TP + TN) / N
    loss:        float = 0.0   # BCE loss value

    # Raw confusion matrix counts
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    n_true_positive: int = 0   # total true edges (y=1)
    n_true_negative: int = 0   # total fake edges (y=0)
    n_predicted_positive: int = 0  # edges kept by threshold

    # Curve data (for plotting — None in step mode, filled in epoch mode)
    fpr:    Optional[np.ndarray] = field(default=None, repr=False)
    tpr:    Optional[np.ndarray] = field(default=None, repr=False)
    thresholds_roc: Optional[np.ndarray] = field(default=None, repr=False)
    precision_curve: Optional[np.ndarray] = field(default=None, repr=False)
    recall_curve:    Optional[np.ndarray] = field(default=None, repr=False)
    thresholds_pr:   Optional[np.ndarray] = field(default=None, repr=False)

    # Score distributions (for histogram plots)
    scores_pos: Optional[np.ndarray] = field(default=None, repr=False)  # scores where y=1
    scores_neg: Optional[np.ndarray] = field(default=None, repr=False)  # scores where y=0

    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        """Return scalar metrics as a flat dict, optionally prefixed."""
        p = f"{prefix}/" if prefix else ""
        return {
            f"{p}efficiency":      self.efficiency,
            f"{p}purity":          self.purity,
            f"{p}fake_rate":       self.fake_rate,
            f"{p}f1":              self.f1,
            f"{p}auc":             self.auc,
            f"{p}avg_precision":   self.avg_precision,
            f"{p}accuracy":        self.accuracy,
            f"{p}loss":            self.loss,
            f"{p}tp":              float(self.tp),
            f"{p}fp":              float(self.fp),
            f"{p}tn":              float(self.tn),
            f"{p}fn":              float(self.fn),
            f"{p}n_signal":        float(self.n_true_positive),
            f"{p}n_background":    float(self.n_true_negative),
            f"{p}n_kept":          float(self.n_predicted_positive),
        }

    def summary_str(self) -> str:
        return (
            f"loss={self.loss:.4f}  eff={self.efficiency:.4f}  "
            f"pur={self.purity:.4f}  fake={self.fake_rate:.4f}  "
            f"f1={self.f1:.4f}  auc={self.auc:.4f}  "
            f"tp={self.tp} fp={self.fp} tn={self.tn} fn={self.fn}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    logits: torch.Tensor,
    y: torch.Tensor,
    loss: float = 0.0,
    edge_cut: float = 0.5,
    compute_curves: bool = False,
) -> TrackingMetrics:
    """
    Compute all tracking and ML metrics from raw logits and binary labels.

    Parameters
    ----------
    logits : Tensor [N]
        Raw model output (before sigmoid).
    y : Tensor [N]
        Binary ground truth (float, 0 or 1).
    loss : float
        Pre-computed loss value to embed in the result.
    edge_cut : float
        Probability threshold for binary classification.
    compute_curves : bool
        If True, compute ROC/PR curve arrays and score distributions.
        Expensive — use only for epoch-end logging, not per-step.

    Returns
    -------
    TrackingMetrics
    """
    scores = torch.sigmoid(logits).detach()
    y_bin  = y.detach()

    s_np = scores.cpu().float().numpy()
    y_np = y_bin.cpu().float().numpy()

    # ── Confusion matrix ──────────────────────────────────────────────────────
    pred = (s_np >= edge_cut).astype(float)
    tp = int(((pred == 1) & (y_np == 1)).sum())
    fp = int(((pred == 1) & (y_np == 0)).sum())
    tn = int(((pred == 0) & (y_np == 0)).sum())
    fn = int(((pred == 0) & (y_np == 1)).sum())

    n_pos = int(y_np.sum())
    n_neg = int((1 - y_np).sum())
    n_kept = tp + fp

    # ── Physics metrics ───────────────────────────────────────────────────────
    efficiency = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    purity     = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fake_rate  = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * purity * efficiency / (purity + efficiency)
        if (purity + efficiency) > 0 else 0.0
    )
    accuracy = (tp + tn) / len(y_np) if len(y_np) > 0 else 0.0

    # ── AUC / AP ─────────────────────────────────────────────────────────────
    has_both_classes = len(np.unique(y_np)) >= 2
    if has_both_classes:
        auc = float(roc_auc_score(y_np, s_np))
        avg_precision = float(average_precision_score(y_np, s_np))
    else:
        logger.debug("Single class in batch — AUC/AP set to 0.0")
        auc = 0.0
        avg_precision = 0.0

    # ── Curve data (epoch-end only) ───────────────────────────────────────────
    fpr = tpr = thr_roc = None
    prec_c = rec_c = thr_pr = None
    scores_pos = scores_neg = None

    if compute_curves and has_both_classes:
        fpr, tpr, thr_roc = roc_curve(y_np, s_np)
        prec_c, rec_c, thr_pr = precision_recall_curve(y_np, s_np)
        scores_pos = s_np[y_np == 1]
        scores_neg = s_np[y_np == 0]

    return TrackingMetrics(
        efficiency=efficiency,
        purity=purity,
        fake_rate=fake_rate,
        f1=f1,
        auc=auc,
        avg_precision=avg_precision,
        accuracy=accuracy,
        loss=loss,
        tp=tp, fp=fp, tn=tn, fn=fn,
        n_true_positive=n_pos,
        n_true_negative=n_neg,
        n_predicted_positive=n_kept,
        fpr=fpr, tpr=tpr, thresholds_roc=thr_roc,
        precision_curve=prec_c, recall_curve=rec_c, thresholds_pr=thr_pr,
        scores_pos=scores_pos, scores_neg=scores_neg,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Epoch-level accumulator
# ─────────────────────────────────────────────────────────────────────────────

class MetricsAccumulator:
    """
    Accumulates logits and labels across batches within an epoch,
    then computes epoch-level metrics including full ROC/PR curves.

    Usage
    -----
    acc = MetricsAccumulator()
    for batch in loader:
        logits = model(batch)
        acc.update(logits, batch.y.float(), loss.item())
    metrics = acc.compute(edge_cut=0.5)
    acc.reset()
    """

    def __init__(self):
        self._logits: list = []
        self._labels: list = []
        self._losses: list = []

    def update(self, logits: torch.Tensor, y: torch.Tensor, loss: float):
        self._logits.append(logits.detach().cpu().float())
        self._labels.append(y.detach().cpu().float())
        self._losses.append(loss)

    def reset(self):
        self._logits.clear()
        self._labels.clear()
        self._losses.clear()

    def compute(self, edge_cut: float = 0.5) -> TrackingMetrics:
        if not self._logits:
            return TrackingMetrics()
        all_logits = torch.cat(self._logits)
        all_labels = torch.cat(self._labels)
        mean_loss  = float(np.mean(self._losses))
        return compute_metrics(
            all_logits, all_labels,
            loss=mean_loss,
            edge_cut=edge_cut,
            compute_curves=True,
        )


def threshold_scores(scores: torch.Tensor, edge_cut: float = 0.5) -> torch.Tensor:
    """Apply probability threshold, return boolean mask."""
    return scores > edge_cut
