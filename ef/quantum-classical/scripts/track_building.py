#!/usr/bin/env python
"""
scripts/track_building.py
──────────────────────────
Track-level reconstruction from edge scores using connected components.

Produces track-level metrics required for physics publications:
  - Track reconstruction efficiency vs pt and eta
  - Fake track rate vs pt and eta
  - Duplicate rate
  - Track purity distribution

Method
------
Edges with score >= edge_cut are kept; connected components of the
resulting subgraph form candidate tracks. Each candidate is matched
to the true particle with the most hits (majority rule).

Output
------
  results/track_metrics.json
  plots/track_eff_pt.png
  plots/track_eff_eta.png
  plots/track_fakerate_pt.png
  plots/track_fakerate_eta.png
  plots/track_purity_hist.png

Usage
-----
  python scripts/track_building.py \
      --config configs/default.yaml \
      --checkpoint checkpoints/best.ckpt \
      --split test \
      --edge_cut 0.5 \
      --output_dir results/
"""

from __future__ import annotations
import argparse, json, logging, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("track_building")


# ─────────────────────────────────────────────────────────────────────────────
# Connected components track building
# ─────────────────────────────────────────────────────────────────────────────

def build_tracks(edge_index: np.ndarray, scores: np.ndarray,
                 n_nodes: int, edge_cut: float):
    """
    Keep edges above threshold, find connected components.
    Returns track_labels: [n_nodes] int array, -1 = isolated node.
    """
    keep = scores >= edge_cut
    src  = edge_index[0][keep]
    dst  = edge_index[1][keep]

    if len(src) == 0:
        return np.full(n_nodes, -1, dtype=int)

    data = np.ones(len(src))
    mat  = csr_matrix((data, (src, dst)), shape=(n_nodes, n_nodes))
    mat  = mat + mat.T   # make symmetric
    n_components, labels = connected_components(mat, directed=False)
    return labels


def majority_matching(track_labels: np.ndarray, particle_ids: np.ndarray,
                      min_hits: int = 3):
    """
    Match each reconstructed track to the true particle with most hits.
    Returns:
      matched_pids  : reconstructed track → matched particle id (-1 if no match)
      track_purities: fraction of hits from matched particle
    """
    unique_tracks = np.unique(track_labels)
    unique_tracks = unique_tracks[unique_tracks >= 0]

    matched_pids   = {}
    track_purities = {}

    for t in unique_tracks:
        hit_mask = track_labels == t
        n_hits   = hit_mask.sum()
        if n_hits < min_hits:
            matched_pids[t]   = -1
            track_purities[t] = 0.0
            continue
        pids_in_track = particle_ids[hit_mask]
        unique_p, counts = np.unique(pids_in_track, return_counts=True)
        best_idx = np.argmax(counts)
        best_pid = unique_p[best_idx]
        purity   = counts[best_idx] / n_hits
        matched_pids[t]   = int(best_pid) if purity >= 0.5 else -1
        track_purities[t] = float(purity)

    return matched_pids, track_purities


# ─────────────────────────────────────────────────────────────────────────────
# Binned track metrics
# ─────────────────────────────────────────────────────────────────────────────

def binned_track_metrics(values: np.ndarray, reconstructed: np.ndarray,
                         true_mask: np.ndarray, bins: np.ndarray):
    """
    Compute track efficiency and fake rate in bins.
    values        : [N_particles] — pt or eta of each true particle
    reconstructed : [N_particles] bool — was this particle reconstructed?
    true_mask     : [N_particles] bool — is this a "findable" particle?
    """
    centres    = 0.5 * (bins[:-1] + bins[1:])
    efficiency = np.full(len(centres), np.nan)
    eff_err    = np.full(len(centres), np.nan)

    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        in_bin = (values >= lo) & (values < hi) & true_mask
        if in_bin.sum() == 0:
            continue
        n_reco  = reconstructed[in_bin].sum()
        n_total = in_bin.sum()
        eff     = n_reco / n_total
        efficiency[i] = eff
        eff_err[i]    = np.sqrt(eff * (1 - eff) / n_total)

    return centres, efficiency, eff_err


def plot_track_metric(centres, values, errors, xlabel, ylabel, title,
                      out_path, color="steelblue"):
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = ~np.isnan(values)
    ax.errorbar(centres[valid], values[valid], yerr=errors[valid],
                fmt="o-", color=color, capsize=3, linewidth=1.5, markersize=5)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.95, color="grey", linestyle=":", label="95% reference")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Inference + track building pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(config_path, ckpt_path, split, edge_cut):
    from torch_geometric.loader import DataLoader
    from data.dataset import GraphDataset
    from models.gnn import InteractionGNN

    with open(config_path) as f:
        hparams = yaml.safe_load(f)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_dirs = {"train": "train_set/", "val": "val_set/", "test": "test_set/"}
    dataset = GraphDataset(hparams["input_dir"], split_dirs[split],
                           preprocess=True, hparams=hparams)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model   = InteractionGNN.load_from_checkpoint(ckpt_path, hparams=hparams,
                                                  map_location=device)
    model.eval().to(device)

    all_results = []
    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            logits = model(batch)
            scores = torch.sigmoid(logits).cpu().numpy()
            y      = batch.y.cpu().numpy()
            ei     = batch.edge_index.cpu().numpy()
            n_nodes = batch.num_nodes

            # Get node-level particle IDs if available
            pid = batch.particle_id.cpu().numpy() if hasattr(batch, "particle_id") else None
            pt  = batch.pt.cpu().numpy()  if hasattr(batch, "pt")  else None
            eta = batch.eta.cpu().numpy() if hasattr(batch, "eta") else None

            all_results.append({
                "scores": scores, "y": y,
                "edge_index": ei, "n_nodes": n_nodes,
                "particle_id": pid, "pt": pt, "eta": eta,
            })

    return all_results


def compute_track_metrics(all_results, edge_cut, min_hits=3):
    """Aggregate track-level metrics across all events."""
    all_pt, all_eta  = [], []
    all_reconstructed = []
    all_purities      = []
    n_fake_tracks = 0
    n_total_tracks = 0

    for r in all_results:
        if r["particle_id"] is None:
            logger.warning("No particle_id in graph — skipping track matching")
            continue

        labels  = build_tracks(r["edge_index"], r["scores"], r["n_nodes"], edge_cut)
        matched, purities = majority_matching(labels, r["particle_id"], min_hits)

        # Track purity distribution
        all_purities.extend(purities.values())

        # Fake tracks = no matched particle
        n_fake   = sum(1 for p in matched.values() if p == -1)
        n_tracks = len(matched)
        n_fake_tracks  += n_fake
        n_total_tracks += n_tracks

        # Per-particle reconstruction efficiency
        true_pids  = np.unique(r["particle_id"])
        reco_pids  = set(p for p in matched.values() if p != -1)

        for pid in true_pids:
            if pid <= 0:  # noise hits
                continue
            hit_mask = r["particle_id"] == pid
            n_hits   = hit_mask.sum()
            if n_hits < min_hits:
                continue
            is_reco = pid in reco_pids
            all_reconstructed.append(is_reco)

            if r["pt"] is not None:
                pt_val = float(r["pt"][hit_mask].mean())
                all_pt.append(pt_val)
            else:
                all_pt.append(np.nan)

            if r["eta"] is not None:
                eta_val = float(r["eta"][hit_mask].mean())
                all_eta.append(eta_val)
            else:
                all_eta.append(np.nan)

    return (np.array(all_pt), np.array(all_eta),
            np.array(all_reconstructed, dtype=bool),
            np.array(all_purities),
            n_fake_tracks, n_total_tracks)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split",      default="test", choices=["train","val","test"])
    p.add_argument("--edge_cut",   type=float, default=0.5)
    p.add_argument("--min_hits",   type=int,   default=3)
    p.add_argument("--output_dir", default="results/")
    p.add_argument("--pt_max",     type=float, default=10000., help="pt in MeV")
    p.add_argument("--n_bins",     type=int,   default=20)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    logger.info("Running inference on %s split...", args.split)
    results = run_inference(args.config, args.checkpoint, args.split, args.edge_cut)

    logger.info("Building tracks (edge_cut=%.2f, min_hits=%d)...",
                args.edge_cut, args.min_hits)
    pt_arr, eta_arr, reco_arr, purities, n_fake, n_total = \
        compute_track_metrics(results, args.edge_cut, args.min_hits)

    findable = ~np.isnan(pt_arr)
    fake_rate_global = n_fake / n_total if n_total > 0 else 0.0
    eff_global = reco_arr[findable].mean() if findable.sum() > 0 else 0.0

    logger.info("Track efficiency (global):  %.4f", eff_global)
    logger.info("Fake track rate  (global):  %.4f  (%d / %d)",
                fake_rate_global, n_fake, n_total)
    logger.info("Mean track purity:          %.4f", np.mean(purities) if len(purities) else 0)

    # ── pt plots ──────────────────────────────────────────────────────────────
    if not np.all(np.isnan(pt_arr)):
        pt_bins = np.linspace(1000, args.pt_max, args.n_bins + 1)
        c, eff, err = binned_track_metrics(pt_arr, reco_arr, findable, pt_bins)
        plot_track_metric(c, eff, err,
            xlabel=r"Particle $p_T$ [MeV]",
            ylabel="Track Reconstruction Efficiency",
            title=f"Track Efficiency vs $p_T$  [{args.split}]",
            out_path=os.path.join(plot_dir, "track_eff_pt.png"))

    # ── eta plots ─────────────────────────────────────────────────────────────
    if not np.all(np.isnan(eta_arr)):
        eta_bins = np.linspace(-4, 4, args.n_bins + 1)
        c, eff, err = binned_track_metrics(eta_arr, reco_arr, findable, eta_bins)
        plot_track_metric(c, eff, err,
            xlabel=r"Particle $\eta$",
            ylabel="Track Reconstruction Efficiency",
            title=f"Track Efficiency vs $\\eta$  [{args.split}]",
            out_path=os.path.join(plot_dir, "track_eff_eta.png"))

    # ── track purity histogram ─────────────────────────────────────────────────
    if len(purities) > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(purities, bins=20, range=(0, 1), color="steelblue", alpha=0.8,
                edgecolor="white")
        ax.set_xlabel("Track Purity", fontsize=13)
        ax.set_ylabel("Count", fontsize=13)
        ax.set_title(f"Track Purity Distribution [{args.split}]", fontsize=14)
        ax.axvline(np.mean(purities), color="red", linestyle="--",
                   label=f"Mean={np.mean(purities):.3f}")
        ax.legend()
        fig.tight_layout()
        purity_path = os.path.join(plot_dir, "track_purity_hist.png")
        fig.savefig(purity_path, dpi=150)
        plt.close(fig)
        logger.info("Saved → %s", purity_path)

    # ── Save JSON summary ─────────────────────────────────────────────────────
    summary = {
        "split":              args.split,
        "edge_cut":           args.edge_cut,
        "min_hits":           args.min_hits,
        "n_events":           len(results),
        "track_efficiency":   float(eff_global),
        "fake_track_rate":    float(fake_rate_global),
        "mean_track_purity":  float(np.mean(purities)) if len(purities) else 0.0,
        "n_total_tracks":     n_total,
        "n_fake_tracks":      n_fake,
    }
    json_path = os.path.join(args.output_dir, "track_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Track metrics saved → %s", json_path)

    print("\n" + "="*50)
    print("  TRACK-LEVEL RESULTS")
    print("="*50)
    for k, v in summary.items():
        print(f"  {k:<25} {v}")
    print("="*50)


if __name__ == "__main__":
    main()
