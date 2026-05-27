#!/usr/bin/env python
"""
scripts/scaling_study.py
─────────────────────────
Scaling study: AUC and efficiency vs n_qubits and n_qlayers.

Trains all combinations defined in configs/scaling_sweep.yaml and
produces comparison plots required for publication.

Output
------
  results/scaling/all_results.csv
  results/scaling/all_results.json
  plots/scaling_auc_vs_qubits.png
  plots/scaling_auc_vs_layers.png
  plots/scaling_parameters.png     — param count vs AUC

Usage
-----
  python scripts/scaling_study.py \
      --sweep_config configs/scaling_sweep.yaml \
      --base_config configs/default.yaml \
      --output_dir results/scaling/
"""

from __future__ import annotations
import argparse, json, logging, os, subprocess, sys
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("scaling_study")


def run_config(base_config, overrides: dict, seed: int, run_dir: str) -> dict:
    """Train one configuration and return test metrics."""
    os.makedirs(run_dir, exist_ok=True)

    # Write merged config
    with open(base_config) as f:
        cfg = yaml.safe_load(f)
    cfg.update(overrides)
    cfg["stage_dir"] = run_dir
    tmp_cfg = os.path.join(run_dir, "config.yaml")
    with open(tmp_cfg, "w") as f:
        yaml.dump(cfg, f)

    cmd = [
        sys.executable, "scripts/train.py",
        "--config", tmp_cfg,
        "--seed", str(seed),
        "--no_wandb",
    ]
    log_path = os.path.join(run_dir, "train.log")
    with open(log_path, "w") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)

    # Parse result
    metrics = {"seed": seed, **overrides}
    try:
        with open(log_path) as f:
            lines = f.readlines()
        for line in reversed(lines):
            if "[TEST epoch" in line:
                for kv in line.split():
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        try: metrics[k] = float(v)
                        except ValueError: pass
                break
    except Exception as e:
        logger.warning("Could not parse %s: %s", log_path, e)

    return metrics


def plot_scaling(df, x_col, x_label, y_col, y_label, title, out_path,
                 hue_col="model_type"):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"classical": "steelblue", "quantum": "tomato"}
    markers = {"classical": "o", "quantum": "s"}

    for model_type, grp in df.groupby(hue_col):
        # Average across seeds
        agg = grp.groupby(x_col)[y_col].agg(["mean", "std"]).reset_index()
        color  = colors.get(model_type, "grey")
        marker = markers.get(model_type, "o")
        ax.errorbar(agg[x_col], agg["mean"], yerr=agg["std"],
                    fmt=f"{marker}-", color=color, label=model_type.capitalize(),
                    capsize=3, linewidth=2, markersize=7)

    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved → %s", out_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_config", default="configs/scaling_sweep.yaml")
    p.add_argument("--base_config",  default="configs/default.yaml")
    p.add_argument("--output_dir",   default="results/scaling/")
    p.add_argument("--dry_run",      action="store_true",
                   help="Print configs without running")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    with open(args.sweep_config) as f:
        sweep_cfg = yaml.safe_load(f)

    sweep   = sweep_cfg["sweep"]
    fixed   = sweep_cfg.get("fixed", {})
    base_ov = sweep_cfg.get("base_overrides", {})
    seeds   = sweep["seeds"]

    # Build all combinations
    runs = []
    for model_type in sweep["model_type"]:
        if model_type == "classical":
            # Classical doesn't vary with n_qubits/n_qlayers
            for seed in seeds:
                overrides = {**fixed, **base_ov, "model_type": model_type}
                run_id = f"{model_type}_seed{seed}"
                runs.append((overrides, seed, run_id))
        else:
            for nq in sweep["n_qubits"]:
                for nl in sweep["n_qlayers"]:
                    for seed in seeds:
                        overrides = {**fixed, **base_ov,
                                     "model_type": model_type,
                                     "n_qubits": nq, "n_qlayers": nl}
                        run_id = f"{model_type}_q{nq}_l{nl}_seed{seed}"
                        runs.append((overrides, seed, run_id))

    logger.info("Total runs: %d", len(runs))

    all_results = []
    for i, (overrides, seed, run_id) in enumerate(runs):
        logger.info("[%d/%d] %s", i+1, len(runs), run_id)
        if args.dry_run:
            logger.info("  DRY RUN: %s", overrides)
            continue
        run_dir = os.path.join(args.output_dir, run_id)
        result  = run_config(args.base_config, overrides, seed, run_dir)
        all_results.append(result)

        # Save incrementally
        pd.DataFrame(all_results).to_csv(
            os.path.join(args.output_dir, "all_results.csv"), index=False)

    if args.dry_run or not all_results:
        return

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.output_dir, "all_results.csv"), index=False)
    with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if "auc" in df.columns and "n_qubits" in df.columns:
        quantum_df = df[df["model_type"] == "quantum"].copy()

        plot_scaling(quantum_df, "n_qubits", "Number of Qubits", "auc",
                     "Test AUC",
                     "AUC vs Number of Qubits",
                     os.path.join(plot_dir, "scaling_auc_vs_qubits.png"),
                     hue_col="n_qlayers")

    if "auc" in df.columns and "n_qlayers" in df.columns:
        quantum_df = df[df["model_type"] == "quantum"].copy()
        plot_scaling(quantum_df, "n_qlayers", "Number of VQC Layers", "auc",
                     "Test AUC",
                     "AUC vs Number of VQC Layers",
                     os.path.join(plot_dir, "scaling_auc_vs_layers.png"),
                     hue_col="n_qubits")

    logger.info("Scaling study complete. Results in %s", args.output_dir)


if __name__ == "__main__":
    main()
