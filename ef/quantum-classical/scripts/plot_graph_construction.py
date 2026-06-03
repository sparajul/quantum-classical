"""
scripts/plot_graph_construction.py
───────────────────────────────────
Graph construction efficiency and purity vs pt and eta.
Produces both separate and combined plots. No GNN checkpoint needed.

Definitions
-----------
  Efficiency : fraction of true same-particle hit-pairs captured as edges
               (per findable particle with >= min_hits hits)
  Purity     : fraction of graph edges that are true (y==1)

Output (plots/ directory)
------
  graph_efficiency_pt_<split>.png
  graph_efficiency_eta_<split>.png
  graph_purity_pt_<split>.png
  graph_purity_eta_<split>.png
  graph_combined_pt_<split>.png
  graph_combined_eta_<split>.png
  results/graph_construction_metrics_<split>.json

Usage
-----
  python scripts/plot_graph_construction.py --input-dir data/graphs/train_set
  python scripts/plot_graph_construction.py --input-dir data/graphs/test_set --split test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("graph_construction")


# ─────────────────────────────────────────────────────────────────────────────
# Per-graph computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_graph_stats(data, min_hits: int = 2):
    """
    Returns:
      eff_records : list of (pt, eta, n_possible_pairs, n_found_pairs) per particle
      pur_records : list of (pt, eta, is_true_edge) per edge

    Efficiency denominator: consecutive adjacent-layer pairs (n_hits - 1),
    not all C(n,2) pairs. This matches what the geometric graph builder covers.
    Hits are sorted by r to define the r-ordered adjacent segments.
    """
    pid = data.particle_id.numpy()
    pt  = data.pt.numpy()
    eta = data.eta.numpy()
    r   = data.r.numpy()
    src = data.edge_index[0].numpy()
    dst = data.edge_index[1].numpy()
    y   = data.y.numpy()

    edge_pt  = 0.5 * (pt[src]  + pt[dst])
    edge_eta = 0.5 * (eta[src] + eta[dst])
    pur_records = list(zip(edge_pt.tolist(), edge_eta.tolist(), y.tolist()))

    edge_set = {
        (int(min(s, d)), int(max(s, d)))
        for s, d in zip(src.tolist(), dst.tolist())
    }

    eff_records = []
    for p in np.unique(pid):
        if p <= 0:
            continue
        hits = np.where(pid == p)[0]
        if len(hits) < min_hits:
            continue
        p_pt  = float(pt[hits].mean())
        p_eta = float(eta[hits].mean())

        # Sort hits by r so consecutive pairs are adjacent-layer doublets
        order = np.argsort(r[hits])
        sorted_hits = hits[order]

        n_possible = len(sorted_hits) - 1
        n_found = sum(
            1 for i in range(n_possible)
            if (min(int(sorted_hits[i]), int(sorted_hits[i + 1])),
                max(int(sorted_hits[i]), int(sorted_hits[i + 1]))) in edge_set
        )
        eff_records.append((p_pt, p_eta, n_possible, n_found))

    return eff_records, pur_records


# ─────────────────────────────────────────────────────────────────────────────
# Binned statistics
# ─────────────────────────────────────────────────────────────────────────────

def binned_efficiency(records, bins, coord: int):
    centres = 0.5 * (bins[:-1] + bins[1:])
    num = np.zeros(len(centres))
    den = np.zeros(len(centres))
    counts = np.zeros(len(centres))          # particles per bin
    for rec in records:
        v = rec[coord]
        idx = np.searchsorted(bins, v, side="right") - 1
        if 0 <= idx < len(centres):
            num[idx]    += rec[3]
            den[idx]    += rec[2]
            counts[idx] += 1
    eff = np.where(den > 0, num / den, np.nan)
    err = np.where(den > 0,
                   np.sqrt(np.where(np.isnan(eff), 0, eff) *
                           (1 - np.where(np.isnan(eff), 0, eff)) /
                           np.maximum(den, 1)), np.nan)
    return centres, eff, err, counts


def binned_purity(records, bins, coord: int):
    centres = 0.5 * (bins[:-1] + bins[1:])
    num = np.zeros(len(centres))
    den = np.zeros(len(centres))
    for edge_pt, edge_eta, is_true in records:
        v = edge_pt if coord == 0 else edge_eta
        idx = np.searchsorted(bins, v, side="right") - 1
        if 0 <= idx < len(centres):
            den[idx] += 1
            if is_true:
                num[idx] += 1
    pur = np.where(den > 0, num / den, np.nan)
    err = np.where(den > 0,
                   np.sqrt(np.where(np.isnan(pur), 0, pur) *
                           (1 - np.where(np.isnan(pur), 0, pur)) /
                           np.maximum(den, 1)), np.nan)
    return centres, pur, err, den          # den = edges per bin


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stats_box(ax, lines: list[str]):
    """Draw a text box in the upper-left with key stats."""
    ax.text(0.02, 0.97, "\n".join(lines),
            transform=ax.transAxes, fontsize=8.5,
            va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec="lightgrey", alpha=0.85))


def _count_panel(ax_cnt, centres, counts, ylabel, color):
    """Filled bar chart of per-bin counts on a secondary axes."""
    width = (centres[1] - centres[0]) * 0.8 if len(centres) > 1 else 1.0
    ax_cnt.bar(centres, counts, width=width, color=color, alpha=0.25,
               edgecolor="none")
    ax_cnt.set_ylabel(ylabel, fontsize=9, color=color)
    ax_cnt.tick_params(axis="y", labelcolor=color, labelsize=8)
    ax_cnt.set_ylim(bottom=0)


# ─────────────────────────────────────────────────────────────────────────────
# Individual plots (metric + count bar on twin axis)
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric(centres, values, errors, counts,
                xlabel, ylabel, title, out_path,
                color, global_val, stats_lines, ref_line=0.95,
                show_count_panel=False):
    fig, ax = plt.subplots(figsize=(8, 5))

    if show_count_panel:
        ax2 = ax.twinx()
        _count_panel(ax2, centres, counts, "Count per bin", color)

    # Main metric
    valid = ~np.isnan(values)
    ax.errorbar(centres[valid], values[valid], yerr=errors[valid],
                fmt="o-", color=color, capsize=3, linewidth=2,
                markersize=5, zorder=3)
    ax.axhline(ref_line, color="grey", linestyle=":", linewidth=1,
               label=f"{int(ref_line*100)}% reference")

    ax.axhline(global_val, color=color, linestyle="--", linewidth=1.2, alpha=0.6,
               label=f"Global = {global_val:.3f}", zorder=2)

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1.5)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3, zorder=0)

    _stats_box(ax, stats_lines)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Combined efficiency + purity plot
# ─────────────────────────────────────────────────────────────────────────────

def combined_plot(centres,
                  eff, eff_err, eff_counts,
                  pur, pur_err, pur_counts,
                  xlabel, title, out_path,
                  global_eff, global_pur, stats_lines):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Count bars (particles) on twin axis — use eff_counts
    ax2 = ax.twinx()
    _count_panel(ax2, centres, eff_counts, "Particles per bin", "steelblue")

    v_eff = ~np.isnan(eff)
    v_pur = ~np.isnan(pur)

    ax.errorbar(centres[v_eff], eff[v_eff], yerr=eff_err[v_eff],
                fmt="o-", color="steelblue", capsize=3, linewidth=2,
                markersize=5, zorder=3, label="Efficiency")
    ax.errorbar(centres[v_pur], pur[v_pur], yerr=pur_err[v_pur],
                fmt="s--", color="tomato", capsize=3, linewidth=2,
                markersize=5, zorder=3, label="Purity")

    ax.axhline(0.95, color="grey", linestyle=":", linewidth=1,
               label="95% reference")

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Rate", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1.5)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3, zorder=0)

    _stats_box(ax, stats_lines)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",   default="data/graphs/train_set")
    p.add_argument("--output-dir",  default="plots/")
    p.add_argument("--results-dir", default="results/")
    p.add_argument("--split",       default="train")
    p.add_argument("--min-hits",    type=int, default=2)
    p.add_argument("--n-bins",      type=int, default=20)
    p.add_argument("--max-events",  type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir,  exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    files = sorted(Path(args.input_dir).glob("*.pyg"))
    if args.max_events:
        files = files[:args.max_events]
    if not files:
        logger.error("No .pyg files found in %s", args.input_dir)
        sys.exit(1)
    logger.info("Processing %d graphs ...", len(files))

    all_eff, all_pur = [], []
    node_counts, edge_counts = [], []

    for i, f in enumerate(files):
        data = torch.load(f, weights_only=False)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            node_counts.append(data.num_nodes)
        edge_counts.append(data.edge_index.shape[1])
        e, p = compute_graph_stats(data, min_hits=args.min_hits)
        all_eff.extend(e)
        all_pur.extend(p)
        if (i + 1) % 100 == 0:
            logger.info("  %d / %d", i + 1, len(files))

    mean_nodes = float(np.mean(node_counts))
    mean_edges = float(np.mean(edge_counts))
    logger.info("Mean nodes/event: %.0f  |  Mean edges/event: %.0f",
                mean_nodes, mean_edges)

    # ── Bins — fixed pt range 1–10 GeV ───────────────────────────────────────
    pt_lo, pt_hi = 1.0, 10.0
    pt_bins  = np.linspace(pt_lo, pt_hi, args.n_bins + 1)
    eta_bins = np.linspace(-4, 4,        args.n_bins + 1)

    # ── Binned stats ──────────────────────────────────────────────────────────
    c_pt,  eff_pt,  eff_err_pt,  eff_cnt_pt  = binned_efficiency(all_eff, pt_bins,  coord=0)
    c_eta, eff_eta, eff_err_eta, eff_cnt_eta = binned_efficiency(all_eff, eta_bins, coord=1)
    _,     pur_pt,  pur_err_pt,  pur_cnt_pt  = binned_purity(all_pur, pt_bins,  coord=0)
    _,     pur_eta, pur_err_eta, pur_cnt_eta = binned_purity(all_pur, eta_bins, coord=1)

    # ── Global numbers ────────────────────────────────────────────────────────
    total_pos  = sum(r[2] for r in all_eff)
    total_fnd  = sum(r[3] for r in all_eff)
    n_true     = sum(1 for r in all_pur if r[2])
    global_eff = total_fnd / total_pos if total_pos > 0 else 0.0
    global_pur = n_true / len(all_pur)  if all_pur  else 0.0

    # ── Stats text for annotations ────────────────────────────────────────────
    stats_lines = [
        f"Split          : {args.split}",
        f"Events         : {len(files)}",
        f"Mean nodes/evt : {mean_nodes:,.0f}",
        f"Mean edges/evt : {mean_edges:,.0f}",
    ]

    tag       = args.split
    pt_label  = r"Particle $p_T$ [GeV]"
    eta_label = r"Particle $\eta$"

    # ── Efficiency plots ──────────────────────────────────────────────────────
    plot_metric(
        c_pt, eff_pt, eff_err_pt, eff_cnt_pt,
        xlabel=pt_label,
        ylabel="Graph Construction Efficiency",
        title=f"Graph Efficiency vs $p_T$  [{args.split}]",
        out_path=os.path.join(args.output_dir, f"graph_efficiency_pt_{tag}.png"),
        color="steelblue", global_val=global_eff, stats_lines=stats_lines,
    )
    plot_metric(
        c_eta, eff_eta, eff_err_eta, eff_cnt_eta,
        xlabel=eta_label,
        ylabel="Graph Construction Efficiency",
        title=f"Graph Efficiency vs $\\eta$  [{args.split}]",
        out_path=os.path.join(args.output_dir, f"graph_efficiency_eta_{tag}.png"),
        color="steelblue", global_val=global_eff, stats_lines=stats_lines,
    )

    # ── Purity plots ──────────────────────────────────────────────────────────
    plot_metric(
        c_pt, pur_pt, pur_err_pt, pur_cnt_pt,
        xlabel=pt_label,
        ylabel="Graph Purity (signal fraction)",
        title=f"Graph Purity vs $p_T$  [{args.split}]",
        out_path=os.path.join(args.output_dir, f"graph_purity_pt_{tag}.png"),
        color="tomato", global_val=global_pur, stats_lines=stats_lines,
    )
    plot_metric(
        c_eta, pur_eta, pur_err_eta, pur_cnt_eta,
        xlabel=eta_label,
        ylabel="Graph Purity (signal fraction)",
        title=f"Graph Purity vs $\\eta$  [{args.split}]",
        out_path=os.path.join(args.output_dir, f"graph_purity_eta_{tag}.png"),
        color="tomato", global_val=global_pur, stats_lines=stats_lines,
    )

    # ── Combined plots ────────────────────────────────────────────────────────
    combined_plot(
        c_pt, eff_pt, eff_err_pt, eff_cnt_pt,
               pur_pt, pur_err_pt, pur_cnt_pt,
        xlabel=pt_label,
        title=f"Graph Efficiency & Purity vs $p_T$  [{args.split}]",
        out_path=os.path.join(args.output_dir, f"graph_combined_pt_{tag}.png"),
        global_eff=global_eff, global_pur=global_pur, stats_lines=stats_lines,
    )
    combined_plot(
        c_eta, eff_eta, eff_err_eta, eff_cnt_eta,
               pur_eta, pur_err_eta, pur_cnt_eta,
        xlabel=eta_label,
        title=f"Graph Efficiency & Purity vs $\\eta$  [{args.split}]",
        out_path=os.path.join(args.output_dir, f"graph_combined_eta_{tag}.png"),
        global_eff=global_eff, global_pur=global_pur, stats_lines=stats_lines,
    )

    # ── JSON summary ──────────────────────────────────────────────────────────
    summary = {
        "split":             args.split,

        "n_graphs":          len(files),
        "mean_nodes":        round(mean_nodes, 1),
        "mean_edges":        round(mean_edges, 1),
        "n_particles":       len(all_eff),
        "n_edges":           len(all_pur),
        "global_efficiency": round(global_eff, 4),
        "global_purity":     round(global_pur, 4),
        "pt_range":          [round(pt_lo, 4), round(pt_hi, 4)],
        "min_hits":          args.min_hits,
    }
    json_path = os.path.join(args.results_dir,
                             f"graph_construction_metrics_{tag}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 52)
    print("  GRAPH CONSTRUCTION QUALITY")
    print("=" * 52)
    for k, v in summary.items():
        print(f"  {k:<25} {v}")
    print("=" * 52)
    logger.info("Done. Plots → %s/  Metrics → %s", args.output_dir, json_path)


if __name__ == "__main__":
    main()
