#!/usr/bin/env python
"""
scripts/evaluate.py
────────────────────
Load predictions from one or more trained models and produce comparison plots.

Usage
-----
python scripts/evaluate.py \
    --predictions results/classical/predictions_test.pt:Classical \
                  results/edge_quantum/predictions_test.pt:Edge-Quantum \
                  results/quantum/predictions_test.pt:Quantum \
    --output_dir  plots/comparison/

Each --predictions entry is  <path>:<label>.

Outputs (PDF + PNG per plot, plus a YAML summary)
-------------------------------------------------
  eff_vs_pt.pdf / .png       Efficiency vs pt bin
  purity_vs_pt.pdf / .png    Purity vs pt bin
  eff_vs_eta.pdf / .png      Efficiency vs |η| bin
  purity_vs_eta.pdf / .png   Purity vs |η| bin
  roc_curves.pdf / .png      Overlaid ROC curves
  pr_curves.pdf / .png       Overlaid PR curves
  score_dist.pdf / .png      Score histograms (signal vs background)
  metrics_summary.pdf / .png Bar chart of scalar metrics
  metrics_summary.yaml       Aggregated scalars for all models
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    auc as sk_auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate")

# ── matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size":        12,
    "axes.labelsize":   13,
    "axes.titlesize":   14,
    "legend.fontsize":  10,
    "figure.dpi":       150,
})
COLORS     = ["steelblue", "tomato", "mediumseagreen", "mediumpurple", "darkorange"]
MARKERS    = ["o", "s", "^", "D", "v"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate and compare trained GNN models")
    p.add_argument(
        "--predictions", nargs="+", required=True,
        metavar="PATH:LABEL",
        help="One or more predictions_*.pt files with a colon-separated display label",
    )
    p.add_argument("--output_dir", type=str, default="plots/", help="Directory for output figures")
    p.add_argument("--edge_cut",   type=float, default=0.5,   help="Score threshold for eff/purity")
    p.add_argument(
        "--pt_bins", nargs="+", type=float,
        default=list(np.linspace(1.0, 10.0, 21)),
        help="pt bin edges in GeV (default: 20 uniform bins 1–10 GeV)",
    )
    p.add_argument(
        "--eta_bins", nargs="+", type=float,
        default=list(np.linspace(-4.0, 4.0, 21)),
        help="η bin edges (default: 20 uniform bins −4 to 4)",
    )
    p.add_argument("--min_edges", type=int, default=10,
                   help="Minimum true edges in a bin to plot (fewer → gap)")
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_predictions(path: str) -> dict[str, np.ndarray]:
    """Pool all per-graph tensors into flat numpy arrays."""
    records = torch.load(path, map_location="cpu")
    pooled: dict[str, list] = {}
    for r in records:
        for key in ("scores", "y", "preds", "pt", "eta"):
            if key in r:
                pooled.setdefault(key, []).append(r[key].numpy())
    return {k: np.concatenate(v) for k, v in pooled.items()}


# ── Binned efficiency / purity ────────────────────────────────────────────────

def binned_eff_purity(
    scores: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    bins: list[float],
    edge_cut: float,
    min_edges: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (centres, eff, eff_err, pur, pur_err). NaN where bin has < min_edges true edges."""
    preds = (scores >= edge_cut).astype(float)
    centres, eff, eff_err, pur, pur_err = [], [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (values >= lo) & (values < hi)
        y_b, p_b = y[mask], preds[mask]
        tp = ((y_b == 1) & (p_b == 1)).sum()
        fn = ((y_b == 1) & (p_b == 0)).sum()
        fp = ((y_b == 0) & (p_b == 1)).sum()
        centres.append(0.5 * (lo + hi))
        if (tp + fn) >= min_edges:
            e = tp / (tp + fn + 1e-9)
            p = tp / (tp + fp + 1e-9) if (tp + fp) > 0 else float("nan")
            eff.append(e)
            pur.append(p)
            eff_err.append(np.sqrt(e * (1 - e) / max(tp + fn, 1)))
            pur_err.append(np.sqrt(p * (1 - p) / max(tp + fp, 1)) if not np.isnan(p) else float("nan"))
        else:
            eff.append(float("nan"))
            pur.append(float("nan"))
            eff_err.append(float("nan"))
            pur_err.append(float("nan"))
    return (np.array(centres), np.array(eff), np.array(eff_err),
            np.array(pur), np.array(pur_err))


# ── Scalar metrics ────────────────────────────────────────────────────────────

def scalar_metrics(scores: np.ndarray, y: np.ndarray, edge_cut: float) -> dict:
    preds = (scores >= edge_cut).astype(float)
    tp = ((y == 1) & (preds == 1)).sum()
    fp = ((y == 0) & (preds == 1)).sum()
    fn = ((y == 1) & (preds == 0)).sum()
    eff = tp / (tp + fn + 1e-9)
    pur = tp / (tp + fp + 1e-9) if (tp + fp) > 0 else 0.0
    f1  = 2 * pur * eff / (pur + eff + 1e-9)
    try:
        auc_val = roc_auc_score(y, scores)
        ap_val  = average_precision_score(y, scores)
    except Exception:
        auc_val = ap_val = float("nan")
    return {
        "efficiency":     float(eff),
        "purity":         float(pur),
        "fake_rate":      float(1.0 - pur),
        "f1":             float(f1),
        "auc":            float(auc_val),
        "avg_precision":  float(ap_val),
        "n_edges":        int(len(y)),
        "n_true_pos":     int((y == 1).sum()),
    }


# ── Style helpers ─────────────────────────────────────────────────────────────

def _stats_box(ax: plt.Axes, lines: list[str]) -> None:
    ax.text(0.02, 0.97, "\n".join(lines),
            transform=ax.transAxes, fontsize=8.5,
            va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="lightgrey", alpha=0.85))


# ── Saving helpers ────────────────────────────────────────────────────────────

def savefig(fig: plt.Figure, output_dir: str, name: str) -> None:
    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight")
    logger.info("Saved %s", os.path.join(output_dir, name + ".pdf"))
    plt.close(fig)


# ── Individual plot functions ─────────────────────────────────────────────────

def plot_binned(
    model_data: list[tuple[str, dict]],
    bin_field: str,
    bins: list[float],
    metric: str,       # "eff" or "purity"
    edge_cut: float,
    min_edges: int,
    output_dir: str,
) -> None:
    ylabel   = "Efficiency" if metric == "eff" else "Purity"
    xlabel   = r"$p_T$ [GeV]" if bin_field == "pt" else r"$\eta$"
    ref_line = 0.95 if metric == "eff" else 0.90

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (label, data) in enumerate(model_data):
        if bin_field not in data:
            logger.warning("'%s' not found in predictions for %s — skipping", bin_field, label)
            continue

        centres, eff, eff_err, pur, pur_err = binned_eff_purity(
            data["scores"], data["y"], data[bin_field], bins, edge_cut, min_edges
        )
        values = eff if metric == "eff" else pur
        errors = eff_err if metric == "eff" else pur_err
        valid  = ~np.isnan(values)

        # global metric for this model
        sm     = scalar_metrics(data["scores"], data["y"], edge_cut)
        gval   = sm["efficiency"] if metric == "eff" else sm["purity"]

        color = COLORS[i % len(COLORS)]
        ls    = LINESTYLES[i % len(LINESTYLES)]

        ax.errorbar(
            centres[valid], values[valid], yerr=errors[valid],
            fmt=MARKERS[i % len(MARKERS)], color=color,
            linestyle=ls, linewidth=2, markersize=5, capsize=3, zorder=3,
            label=f"{label}  (global={gval:.3f})",
        )
        ax.axhline(gval, color=color, linestyle="--", linewidth=1.2, alpha=0.5, zorder=2)

    ax.axhline(ref_line, color="grey", linestyle=":", linewidth=1,
               label=f"{int(ref_line * 100)}% reference", zorder=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.5)
    ax.set_title(f"{ylabel} vs {xlabel}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, zorder=0)
    fig.tight_layout()
    savefig(fig, output_dir, f"{metric}_vs_{bin_field}")


def plot_roc(model_data: list[tuple[str, dict]], output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    for i, (label, data) in enumerate(model_data):
        try:
            fpr, tpr, _ = roc_curve(data["y"], data["scores"])
            auc_val = sk_auc(fpr, tpr)
            idx = np.linspace(0, len(fpr) - 1, min(500, len(fpr))).astype(int)
            ax.plot(fpr[idx], tpr[idx], color=COLORS[i % len(COLORS)],
                    linestyle=LINESTYLES[i % len(LINESTYLES)],
                    label=f"{label}  AUC={auc_val:.4f}", linewidth=1.8)
        except Exception as exc:
            logger.warning("ROC failed for %s: %s", label, exc)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, zorder=0)
    fig.tight_layout()
    savefig(fig, output_dir, "roc_curves")


def plot_pr(model_data: list[tuple[str, dict]], output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (label, data) in enumerate(model_data):
        try:
            prec, rec, _ = precision_recall_curve(data["y"], data["scores"])
            ap = average_precision_score(data["y"], data["scores"])
            idx = np.linspace(0, len(prec) - 1, min(500, len(prec))).astype(int)
            ax.plot(rec[idx], prec[idx], color=COLORS[i % len(COLORS)],
                    linestyle=LINESTYLES[i % len(LINESTYLES)],
                    label=f"{label}  AP={ap:.4f}", linewidth=1.8)
        except Exception as exc:
            logger.warning("PR failed for %s: %s", label, exc)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, zorder=0)
    fig.tight_layout()
    savefig(fig, output_dir, "pr_curves")


def plot_score_dist(
    model_data: list[tuple[str, dict]],
    output_dir: str,
    edge_cut: float = 0.5,
) -> None:
    n = len(model_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    score_bins = np.linspace(0, 1, 51)
    for ax, (label, data) in zip(axes, model_data):
        scores    = data["scores"]
        mask_pos  = data["y"] == 1
        mask_neg  = data["y"] == 0
        sig_label = (f"Signal  (n={mask_pos.sum():,}, "
                     f"mean={scores[mask_pos].mean():.3f})" if mask_pos.any() else "Signal")
        bkg_label = (f"Bkg  (n={mask_neg.sum():,}, "
                     f"mean={scores[mask_neg].mean():.3f})" if mask_neg.any() else "Background")
        ax.hist(scores[mask_pos], bins=score_bins, density=True,
                alpha=0.6, color="mediumseagreen", label=sig_label)
        ax.hist(scores[mask_neg], bins=score_bins, density=True,
                alpha=0.6, color="tomato", label=bkg_label)
        ax.axvline(edge_cut, color="grey", linestyle="--", linewidth=1,
                   label=f"cut={edge_cut:.2f}")
        ax.set_xlabel("Edge score")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, zorder=0)
    axes[0].set_ylabel("Density")
    fig.suptitle("Score Distributions", y=1.02)
    fig.tight_layout()
    savefig(fig, output_dir, "score_dist")


def plot_metrics_summary(
    summary: dict[str, dict],
    output_dir: str,
) -> None:
    keys    = ["efficiency", "purity", "f1", "auc", "avg_precision"]
    labels  = list(summary.keys())
    x       = np.arange(len(keys))
    width   = 0.8 / max(len(labels), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, label in enumerate(labels):
        vals   = [summary[label].get(k, float("nan")) for k in keys]
        offset = (i - len(labels) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=label,
                        color=COLORS[i % len(COLORS)], alpha=0.85)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(["Efficiency", "Purity", "F1", "AUC", "Avg Precision"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison Summary")
    ax.legend()
    ax.grid(True, alpha=0.3, zorder=0, axis="y")
    fig.tight_layout()
    savefig(fig, output_dir, "metrics_summary")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse "path:label" pairs
    model_data: list[tuple[str, dict]] = []
    for entry in args.predictions:
        if ":" in entry:
            path, label = entry.rsplit(":", 1)
        else:
            path  = entry
            label = os.path.basename(os.path.dirname(entry))
        if not os.path.exists(path):
            logger.warning("Skipping %s (%s) — file not found", label, path)
            continue
        logger.info("Loading %-30s  (%s)", path, label)
        model_data.append((label, load_predictions(path)))

    if not model_data:
        logger.error("No predictions files found — nothing to plot.")
        sys.exit(1)

    # ── Scalar summary ────────────────────────────────────────────────────────
    summary: dict[str, dict] = {}
    for label, data in model_data:
        m = scalar_metrics(data["scores"], data["y"], args.edge_cut)
        summary[label] = m
        logger.info(
            "[%s]  eff=%.4f  pur=%.4f  f1=%.4f  auc=%.4f  ap=%.4f",
            label, m["efficiency"], m["purity"], m["f1"], m["auc"], m["avg_precision"],
        )

    yaml_path = os.path.join(args.output_dir, "metrics_summary.yaml")
    with open(yaml_path, "w") as fh:
        yaml.dump(summary, fh, default_flow_style=False)
    logger.info("Saved metrics → %s", yaml_path)

    # ── Plots ─────────────────────────────────────────────────────────────────
    pt_bins  = sorted(args.pt_bins)
    eta_bins = sorted(args.eta_bins)

    plot_binned(model_data, "pt",  pt_bins,  "eff",    args.edge_cut, args.min_edges, args.output_dir)
    plot_binned(model_data, "pt",  pt_bins,  "purity", args.edge_cut, args.min_edges, args.output_dir)
    plot_binned(model_data, "eta", eta_bins, "eff",    args.edge_cut, args.min_edges, args.output_dir)
    plot_binned(model_data, "eta", eta_bins, "purity", args.edge_cut, args.min_edges, args.output_dir)
    plot_roc(model_data, args.output_dir)
    plot_pr(model_data, args.output_dir)
    plot_score_dist(model_data, args.output_dir, edge_cut=args.edge_cut)
    plot_metrics_summary(summary, args.output_dir)

    logger.info("All plots saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
