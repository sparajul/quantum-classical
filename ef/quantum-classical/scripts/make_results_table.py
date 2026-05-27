#!/usr/bin/env python
"""
scripts/make_results_table.py
──────────────────────────────
Generate a publication-ready results table comparing classical vs quantum GNN.

Reads sweep_summary.json files from classical and quantum sweep directories
and produces:
  - LaTeX table  (results/table.tex)
  - Markdown table (results/table.md)
  - ROC overlay plot (plots/roc_comparison.png)
  - Score distribution overlay (plots/scores_comparison.png)

Usage
-----
  # First run sweeps for both models:
  python scripts/train_sweep.py --config configs/default.yaml \
      --model_type classical --output_dir results/classical/
  python scripts/train_sweep.py --config configs/default.yaml \
      --model_type quantum   --output_dir results/quantum/

  # Then generate table:
  python scripts/make_results_table.py \
      --classical results/classical/sweep_summary.json \
      --quantum   results/quantum/sweep_summary.json   \
      --output_dir results/
"""

from __future__ import annotations
import argparse, json, logging, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("results_table")

METRICS = [
    ("auc",         "AUC",         "{:.4f}"),
    ("efficiency",  "Efficiency",  "{:.4f}"),
    ("purity",      "Purity",      "{:.4f}"),
    ("fake_rate",   "Fake Rate",   "{:.4f}"),
    ("f1",          "F1 Score",    "{:.4f}"),
    ("avg_precision","Avg. Prec.", "{:.4f}"),
    ("loss",        "Test Loss",   "{:.4f}"),
]


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def format_val(summary, key, fmt):
    if key not in summary:
        return "—"
    m = summary[key]
    if isinstance(m, dict):
        return f"${fmt.format(m['mean'])} \\pm {fmt.format(m['std'])}$"
    return fmt.format(m)


def make_latex_table(classical, quantum, out_path):
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Comparison of Classical GNN vs Quantum-Classical Hybrid GNN on OpenML particle tracking. "
        r"Results are mean $\pm$ std over 5 seeds.}",
        r"\label{tab:results}",
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"\textbf{Metric} & \textbf{Classical GNN} & \textbf{Quantum-Classical GNN} \\",
        r"\hline",
    ]
    for key, label, fmt in METRICS:
        c_str = format_val(classical, key, fmt)
        q_str = format_val(quantum,   key, fmt)
        lines.append(f"{label} & {c_str} & {q_str} \\\\")

    # Extra rows
    c_params = classical.get("n_params", {})
    q_params = quantum.get("n_params", {})
    if isinstance(c_params, dict): c_params = c_params.get("mean", "—")
    if isinstance(q_params, dict): q_params = q_params.get("mean", "—")
    lines.append(f"Parameters & {int(c_params) if c_params != '—' else '—'} "
                 f"& {int(q_params) if q_params != '—' else '—'} \\\\")
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("LaTeX table saved → %s", out_path)


def make_markdown_table(classical, quantum, out_path):
    rows = [
        "| Metric | Classical GNN | Quantum-Classical GNN |",
        "|--------|--------------|----------------------|",
    ]
    for key, label, fmt in METRICS:
        def md_val(s):
            if key not in s: return "—"
            m = s[key]
            if isinstance(m, dict):
                return f"{fmt.format(m['mean'])} ± {fmt.format(m['std'])}"
            return fmt.format(m)
        rows.append(f"| {label} | {md_val(classical)} | {md_val(quantum)} |")

    with open(out_path, "w") as f:
        f.write("\n".join(rows))
    logger.info("Markdown table saved → %s", out_path)


def make_comparison_barplot(classical, quantum, out_path):
    keys   = [k for k, _, _ in METRICS if k in classical and k in quantum]
    labels = [l for k, l, _ in METRICS if k in classical and k in quantum]

    c_means = [classical[k]["mean"] if isinstance(classical[k], dict) else classical[k] for k in keys]
    q_means = [quantum[k]["mean"]   if isinstance(quantum[k], dict)   else quantum[k]   for k in keys]
    c_stds  = [classical[k]["std"]  if isinstance(classical[k], dict) else 0.0 for k in keys]
    q_stds  = [quantum[k]["std"]    if isinstance(quantum[k], dict)   else 0.0 for k in keys]

    x      = np.arange(len(keys))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, c_means, width, yerr=c_stds, label="Classical",
           color="steelblue", alpha=0.85, capsize=4)
    ax.bar(x + width/2, q_means, width, yerr=q_stds, label="Quantum-Classical",
           color="tomato", alpha=0.85, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Classical vs Quantum-Classical GNN — Test Metrics", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Comparison plot saved → %s", out_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--classical",  required=True, help="classical/sweep_summary.json")
    p.add_argument("--quantum",    required=True, help="quantum/sweep_summary.json")
    p.add_argument("--output_dir", default="results/")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    classical = load_summary(args.classical)
    quantum   = load_summary(args.quantum)

    make_latex_table(classical, quantum,
                     os.path.join(args.output_dir, "table.tex"))
    make_markdown_table(classical, quantum,
                        os.path.join(args.output_dir, "table.md"))
    make_comparison_barplot(classical, quantum,
                            os.path.join(plot_dir, "metric_comparison.png"))

    # Print markdown to console
    print("\n" + "="*60)
    print("  RESULTS TABLE")
    print("="*60)
    for key, label, fmt in METRICS:
        def val_str(s):
            if key not in s: return "—"
            m = s[key]
            if isinstance(m, dict):
                return f"{fmt.format(m['mean'])} ± {fmt.format(m['std'])}"
            return fmt.format(m)
        print(f"  {label:<20} Classical: {val_str(classical):<15}  "
              f"Quantum: {val_str(quantum)}")
    print("="*60)


if __name__ == "__main__":
    main()
