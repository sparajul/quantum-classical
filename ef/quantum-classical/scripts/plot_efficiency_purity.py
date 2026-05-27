#!/usr/bin/env python
"""
scripts/plot_efficiency_purity.py
──────────────────────────────────
Plot efficiency and purity as a function of eta and pt.

Works in two modes:
  1. From a saved predictions file (fast):
       python scripts/plot_efficiency_purity.py --predictions results/predictions_test.pt

  2. Directly from a checkpoint + data (runs inference internally):
       python scripts/plot_efficiency_purity.py \
           --config configs/default.yaml \
           --checkpoint checkpoints/best.ckpt \
           --split test

Output
------
  plots/efficiency_purity_eta.png
  plots/efficiency_purity_pt.png

Notes
-----
  - eta is computed from node feature 'z' and 'r':  eta = -ln(tan(arctan2(r, z) / 2))
  - pt  is read directly from node feature 'pt' if available, else estimated from 'r' and 'phi'
  - Edge eta/pt = average of src and dst node values
  - If your graphs use different field names, adjust ETA_FIELD / PT_FIELD below
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("plot")

# ── Field name config — change these if your graphs use different names ────────
PT_FIELD  = "pt"      # node-level pt  (GeV); set to None to estimate from r/phi
ETA_FIELD = None      # node-level eta; set to None to compute from r/z


# ─────────────────────────────────────────────────────────────────────────────
# Eta / pt helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_eta(r: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Pseudorapidity from cylindrical r, z."""
    theta = torch.atan2(r, z)                      # polar angle
    # clamp to avoid log(0)
    theta = theta.clamp(1e-6, np.pi - 1e-6)
    return -torch.log(torch.tan(theta / 2.0))


def get_node_eta(data) -> torch.Tensor:
    if ETA_FIELD and hasattr(data, ETA_FIELD):
        return getattr(data, ETA_FIELD).float()
    if hasattr(data, "r") and hasattr(data, "z"):
        return compute_eta(data.r.float(), data.z.float())
    raise ValueError("Graph has no 'eta' field and no 'r'/'z' to compute it from.")


def get_node_pt(data) -> torch.Tensor:
    if PT_FIELD and hasattr(data, PT_FIELD):
        return getattr(data, PT_FIELD).float()
    raise ValueError(
        f"Graph has no '{PT_FIELD}' field. "
        "Set PT_FIELD=None and add estimation logic, or ensure your graphs have pt."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core metric computation in bins
# ─────────────────────────────────────────────────────────────────────────────

def binned_eff_pur(edge_values: np.ndarray, y: np.ndarray, pred: np.ndarray,
                   bins: np.ndarray):
    """
    Compute efficiency and purity in each bin of edge_values.

    Parameters
    ----------
    edge_values : [N_edges]  — e.g. edge eta or edge pt
    y           : [N_edges]  — ground truth (0/1)
    pred        : [N_edges]  — binary predictions (0/1)
    bins        : bin edges

    Returns
    -------
    centres, efficiency, purity, eff_err, pur_err
    """
    centres    = 0.5 * (bins[:-1] + bins[1:])
    efficiency = np.zeros(len(centres))
    purity     = np.zeros(len(centres))
    eff_err    = np.zeros(len(centres))
    pur_err    = np.zeros(len(centres))

    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (edge_values >= lo) & (edge_values < hi)
        y_b, p_b = y[mask], pred[mask]
        if mask.sum() == 0:
            efficiency[i] = purity[i] = np.nan
            continue

        tp = int(((p_b == 1) & (y_b == 1)).sum())
        fp = int(((p_b == 1) & (y_b == 0)).sum())
        fn = int(((p_b == 0) & (y_b == 1)).sum())

        eff = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        pur = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        efficiency[i] = eff
        purity[i]     = pur

        # Binomial errors
        n_eff = tp + fn
        n_pur = tp + fp
        eff_err[i] = np.sqrt(eff * (1 - eff) / n_eff) if n_eff > 0 else 0.0
        pur_err[i] = np.sqrt(pur * (1 - pur) / n_pur) if n_pur > 0 else 0.0

    return centres, efficiency, purity, eff_err, pur_err


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_eff_pur(centres, efficiency, purity, eff_err, pur_err,
                xlabel: str, title: str, out_path: str, edge_cut: float):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(centres, efficiency, yerr=eff_err,
                fmt="o-", color="steelblue", label="Efficiency (recall)",
                capsize=3, linewidth=1.5, markersize=4)
    ax.errorbar(centres, purity, yerr=pur_err,
                fmt="s--", color="tomato", label="Purity (precision)",
                capsize=3, linewidth=1.5, markersize=4)

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Rate", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.95, color="grey", linestyle=":", linewidth=1, label="95% reference")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.04, f"edge_cut = {edge_cut:.2f}",
            transform=ax.transAxes, fontsize=9, color="dimgrey")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_from_predictions(pred_path: str, edge_cut: float):
    """Load from a saved predictions_test.pt file."""
    results = torch.load(pred_path, weights_only=False)
    all_scores, all_y = [], []
    for r in results:
        all_scores.append(r["scores"])
        all_y.append(r["y"])
    scores = torch.cat(all_scores).numpy()
    y      = torch.cat(all_y).numpy()
    pred   = (scores >= edge_cut).astype(float)
    logger.info("Loaded %d edges from %s", len(y), pred_path)
    # No geometry info in predictions file — need raw graphs for eta/pt plots
    raise RuntimeError(
        "predictions_test.pt does not contain graph geometry (eta/pt). "
        "Run with --checkpoint instead, or re-run inference saving full graph data."
    )


def load_from_checkpoint(config_path: str, ckpt_path: str,
                          split: str, edge_cut: float):
    """Run inference and extract edge eta, pt, scores, labels."""
    import torch
    from torch_geometric.loader import DataLoader
    from data.dataset import GraphDataset
    from models.gnn import QuantumInteractionGNN

    with open(config_path) as f:
        hparams = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Inference device: %s", device)

    split_dir = {"train": "train_set/", "val": "val_set/", "test": "test_set/"}[split]
    dataset = GraphDataset(hparams["input_dir"], split_dir,
                           preprocess=True, hparams=hparams)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = QuantumInteractionGNN.load_from_checkpoint(
        ckpt_path, hparams=hparams, map_location=device)
    model.eval().to(device)

    all_eta, all_pt, all_scores, all_y = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            scores = torch.sigmoid(logits).cpu()
            y      = batch.y.float().cpu()

            src, dst = batch.edge_index.cpu()

            # Eta
            try:
                node_eta = get_node_eta(batch).cpu()
                edge_eta = 0.5 * (node_eta[src] + node_eta[dst])
                all_eta.append(edge_eta)
            except ValueError as e:
                logger.warning("eta: %s", e)

            # Pt
            try:
                node_pt = get_node_pt(batch).cpu()
                edge_pt = 0.5 * (node_pt[src] + node_pt[dst])
                all_pt.append(edge_pt)
            except ValueError as e:
                logger.warning("pt: %s", e)

            all_scores.append(scores)
            all_y.append(y)

    scores_np = torch.cat(all_scores).numpy()
    y_np      = torch.cat(all_y).numpy()
    pred_np   = (scores_np >= edge_cut).astype(float)

    eta_np = torch.cat(all_eta).numpy() if all_eta else None
    pt_np  = torch.cat(all_pt).numpy()  if all_pt  else None

    logger.info("Total edges: %d  |  signal: %d  |  predicted positive: %d",
                len(y_np), int(y_np.sum()), int(pred_np.sum()))

    return eta_np, pt_np, scores_np, y_np, pred_np


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Plot efficiency & purity vs eta and pt")
    p.add_argument("--config",      type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint",  type=str, default=None, help=".ckpt file")
    p.add_argument("--predictions", type=str, default=None,
                   help="predictions_test.pt (alternative to --checkpoint)")
    p.add_argument("--split",       type=str, default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--edge_cut",    type=float, default=0.5)
    p.add_argument("--output_dir",  type=str, default="plots/")
    p.add_argument("--n_eta_bins",  type=int,   default=20)
    p.add_argument("--n_pt_bins",   type=int,   default=20)
    p.add_argument("--pt_max",      type=float, default=10.0,
                   help="Upper pt bin edge in GeV")
    p.add_argument("--pt_log",      action="store_true",
                   help="Use log-spaced pt bins")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.predictions:
        eta_np, pt_np, scores_np, y_np, pred_np = load_from_predictions(
            args.predictions, args.edge_cut)
    elif args.checkpoint:
        eta_np, pt_np, scores_np, y_np, pred_np = load_from_checkpoint(
            args.config, args.checkpoint, args.split, args.edge_cut)
    else:
        print("ERROR: provide --checkpoint or --predictions")
        sys.exit(1)

    # ── Eta plot ──────────────────────────────────────────────────────────────
    if eta_np is not None:
        eta_bins = np.linspace(-4, 4, args.n_eta_bins + 1)
        centres, eff, pur, eff_err, pur_err = binned_eff_pur(
            eta_np, y_np, pred_np, eta_bins)
        plot_eff_pur(
            centres, eff, pur, eff_err, pur_err,
            xlabel=r"Edge $\eta$ (avg of src, dst)",
            title=f"Efficiency & Purity vs $\\eta$  [{args.split}]",
            out_path=os.path.join(args.output_dir, "efficiency_purity_eta.png"),
            edge_cut=args.edge_cut,
        )
    else:
        logger.warning("Skipping eta plot — no eta data available.")

    # ── Pt plot ───────────────────────────────────────────────────────────────
    if pt_np is not None:
        if args.pt_log:
            pt_bins = np.logspace(np.log10(0.1), np.log10(args.pt_max),
                                  args.n_pt_bins + 1)
        else:
            pt_bins = np.linspace(0, args.pt_max, args.n_pt_bins + 1)

        centres, eff, pur, eff_err, pur_err = binned_eff_pur(
            pt_np, y_np, pred_np, pt_bins)
        plot_eff_pur(
            centres, eff, pur, eff_err, pur_err,
            xlabel=r"Edge $p_T$ [GeV] (avg of src, dst)",
            title=f"Efficiency & Purity vs $p_T$  [{args.split}]",
            out_path=os.path.join(args.output_dir, "efficiency_purity_pt.png"),
            edge_cut=args.edge_cut,
        )
    else:
        logger.warning("Skipping pt plot — no pt data available.")

    logger.info("Done. Plots saved to %s/", args.output_dir)


if __name__ == "__main__":
    main()
