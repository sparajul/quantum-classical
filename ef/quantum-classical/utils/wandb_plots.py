"""
utils/wandb_plots.py
─────────────────────
Publication-quality WandB plot helpers.

All functions return wandb objects ready to pass to wandb.log().
Import is guarded — if wandb is not installed, everything degrades gracefully.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    logger.warning("wandb not installed — plot logging disabled.")


def _check():
    return _WANDB_AVAILABLE and wandb.run is not None


# ─────────────────────────────────────────────────────────────────────────────
# ROC Curve
# ─────────────────────────────────────────────────────────────────────────────

def roc_curve_plot(fpr: np.ndarray, tpr: np.ndarray, auc: float, split: str = "val"):
    """
    WandB ROC curve plot.
    Returns a wandb.plot.line object, or None if wandb unavailable.
    """
    if not _check():
        return None
    # Subsample to 500 points for clean rendering
    idx = np.linspace(0, len(fpr) - 1, min(500, len(fpr)), dtype=int)
    data = [[float(fpr[i]), float(tpr[i])] for i in idx]
    table = wandb.Table(data=data, columns=["FPR", "TPR"])
    return wandb.plot.line(
        table, "FPR", "TPR",
        title=f"ROC Curve [{split}] — AUC={auc:.4f}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Precision-Recall Curve
# ─────────────────────────────────────────────────────────────────────────────

def pr_curve_plot(precision: np.ndarray, recall: np.ndarray, ap: float, split: str = "val"):
    """WandB Precision-Recall curve."""
    if not _check():
        return None
    idx = np.linspace(0, len(precision) - 1, min(500, len(precision)), dtype=int)
    data = [[float(recall[i]), float(precision[i])] for i in idx]
    table = wandb.Table(data=data, columns=["Recall (Efficiency)", "Precision (Purity)"])
    return wandb.plot.line(
        table, "Recall (Efficiency)", "Precision (Purity)",
        title=f"Efficiency–Purity Curve [{split}] — AP={ap:.4f}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Score distribution
# ─────────────────────────────────────────────────────────────────────────────

def score_histogram(
    scores_pos: np.ndarray,
    scores_neg: np.ndarray,
    split: str = "val",
    n_bins: int = 50,
):
    """
    Overlapping score histograms for signal (y=1) and background (y=0).
    Well-separated distributions → good discriminator.
    """
    if not _check():
        return None

    bins = np.linspace(0, 1, n_bins + 1)
    pos_counts, _ = np.histogram(scores_pos, bins=bins, density=True)
    neg_counts, _ = np.histogram(scores_neg, bins=bins, density=True)
    bin_centres = 0.5 * (bins[:-1] + bins[1:])

    data = [
        [float(bc), float(p), float(n)]
        for bc, p, n in zip(bin_centres, pos_counts, neg_counts)
    ]
    table = wandb.Table(data=data, columns=["score", "signal (y=1)", "background (y=0)"])
    return wandb.plot.line(
        table, "score", "signal (y=1)",
        title=f"Edge Score Distribution [{split}]",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def confusion_matrix_plot(tp: int, fp: int, tn: int, fn: int, split: str = "val"):
    """WandB confusion matrix heatmap using actual TP/FP/TN/FN counts."""
    if not _check():
        return None
    # Reconstruct label/prediction arrays from counts
    y_true = [1] * (tp + fn) + [0] * (tn + fp)
    preds  = [1] * tp + [0] * fn + [0] * tn + [1] * fp
    return wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_true,
        preds=preds,
        class_names=["Background", "Signal"],
    )


def confusion_matrix_table(tp: int, fp: int, tn: int, fn: int, split: str = "val"):
    """Log confusion matrix counts as a WandB table."""
    if not _check():
        return None
    data = [
        ["Predicted Signal",     "True Signal",     tp],
        ["Predicted Signal",     "True Background", fp],
        ["Predicted Background", "True Signal",     fn],
        ["Predicted Background", "True Background", tn],
    ]
    return wandb.Table(data=data, columns=["Prediction", "Truth", "Count"])


# ─────────────────────────────────────────────────────────────────────────────
# Efficiency vs threshold sweep
# ─────────────────────────────────────────────────────────────────────────────

def efficiency_purity_vs_threshold(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
    split: str = "val",
):
    """
    Plot efficiency (TPR) and purity vs decision threshold.
    Useful for choosing the operating point.
    """
    if not _check():
        return None
    # purity from ROC = TP / (TP + FP) — need to reconstruct from tpr/fpr
    # We approximate: purity ≈ tpr / (tpr + fpr) when base rates are equal
    # For actual purity we'd need class proportions — approximate here
    idx = np.linspace(0, len(tpr) - 1, min(200, len(tpr)), dtype=int)
    data = []
    for i in idx:
        eff = float(tpr[i])
        thr = float(thresholds[i]) if i < len(thresholds) else 1.0
        data.append([thr, eff])
    table = wandb.Table(data=data, columns=["Threshold", "Efficiency (TPR)"])
    return wandb.plot.line(
        table, "Threshold", "Efficiency (TPR)",
        title=f"Efficiency vs Threshold [{split}]",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gradient norm histogram
# ─────────────────────────────────────────────────────────────────────────────

def gradient_histogram(model, step: int):
    """Log per-layer gradient L2 norms as a WandB histogram."""
    if not _check():
        return
    grad_data = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.detach().norm(2).item()
            grad_data.append([name, norm])
    if grad_data:
        table = wandb.Table(data=grad_data, columns=["layer", "grad_norm"])
        wandb.log({"gradients/norms": wandb.plot.bar(table, "layer", "grad_norm",
                   title=f"Gradient Norms (step {step})")}, step=step)


# ─────────────────────────────────────────────────────────────────────────────
# Log all epoch plots at once
# ─────────────────────────────────────────────────────────────────────────────

def log_epoch_plots(metrics, split: str, epoch: int):
    """
    Log all publication-quality plots for one epoch end.
    Call from on_validation_epoch_end / on_test_epoch_end.

    Parameters
    ----------
    metrics : TrackingMetrics
    split   : "train" | "val" | "test"
    epoch   : current epoch number
    """
    if not _check():
        return

    log_dict = {}

    if metrics.fpr is not None and metrics.tpr is not None:
        roc = roc_curve_plot(metrics.fpr, metrics.tpr, metrics.auc, split)
        if roc:
            log_dict[f"{split}/roc_curve"] = roc

        eff_thr = efficiency_purity_vs_threshold(
            metrics.fpr, metrics.tpr,
            metrics.thresholds_roc if metrics.thresholds_roc is not None else np.array([]),
            split,
        )
        if eff_thr:
            log_dict[f"{split}/efficiency_vs_threshold"] = eff_thr

    if metrics.precision_curve is not None and metrics.recall_curve is not None:
        pr = pr_curve_plot(metrics.precision_curve, metrics.recall_curve,
                           metrics.avg_precision, split)
        if pr:
            log_dict[f"{split}/pr_curve"] = pr

    if metrics.scores_pos is not None and metrics.scores_neg is not None:
        hist = score_histogram(metrics.scores_pos, metrics.scores_neg, split)
        if hist:
            log_dict[f"{split}/score_distribution"] = hist

    cm = confusion_matrix_table(metrics.tp, metrics.fp, metrics.tn, metrics.fn, split)
    if cm:
        log_dict[f"{split}/confusion_matrix"] = cm

    if log_dict:
        wandb.log(log_dict, step=epoch)
