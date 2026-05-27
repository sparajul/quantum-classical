#!/usr/bin/env python
"""
scripts/train_sweep.py
──────────────────────
Multi-seed sweep for statistical robustness (required for publication).

Runs training N times with different seeds and produces:
  results/sweep_summary.json   — mean ± std for all metrics
  results/sweep_runs.csv       — per-run results
  plots/sweep_auc.png          — AUC distribution across seeds

Usage
-----
  python scripts/train_sweep.py --config configs/default.yaml \
      --seeds 42 123 456 789 1337 \
      --output_dir results/sweep/

For quantum vs classical comparison:
  python scripts/train_sweep.py --config configs/default.yaml \
      --model_type classical --seeds 42 123 456 --output_dir results/classical/
  python scripts/train_sweep.py --config configs/default.yaml \
      --model_type quantum   --seeds 42 123 456 --output_dir results/quantum/
"""

from __future__ import annotations
import argparse, json, os, subprocess, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      default="configs/default.yaml")
    p.add_argument("--seeds",       nargs="+", type=int, default=[42, 123, 456, 789, 1337])
    p.add_argument("--model_type",  default=None, choices=["quantum", "classical"])
    p.add_argument("--output_dir",  default="results/sweep/")
    p.add_argument("--no_wandb",       action="store_true", default=True)
    p.add_argument("--summarise_only", action="store_true",
                   help="Skip training; just aggregate existing seed_*/train.log files")
    p.add_argument("--extra",          nargs=argparse.REMAINDER, default=[])
    return p.parse_args()


def run_seed(seed: int, config: str, model_type, output_dir: str,
             no_wandb: bool, extra: list) -> dict:
    """Launch a single training run and parse its result."""
    ckpt_dir = os.path.join(output_dir, f"seed_{seed}", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    cmd = [
        sys.executable, "scripts/train.py",
        "--config", config,
        "--seed", str(seed),
        "--stage_dir", os.path.join(output_dir, f"seed_{seed}"),
    ]
    if model_type:
        cmd += ["--model_type", model_type]
    if no_wandb:
        cmd += ["--no_wandb"]
    cmd += extra

    log_path = os.path.join(output_dir, f"seed_{seed}", "train.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Running seed {seed}")
    print(f"  Log: {log_path}")
    print(f"{'='*60}")

    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)

    # Parse last TEST epoch line from log
    metrics = {"seed": seed, "returncode": result.returncode}
    try:
        with open(log_path) as f:
            lines = f.readlines()
        for line in reversed(lines):
            if "[TEST epoch" in line:
                # e.g. loss=0.50  eff=0.68  pur=0.84  fake=0.16  f1=0.75  auc=0.82
                for kv in line.split():
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        try: metrics[k] = float(v)
                        except ValueError: pass
                break
    except Exception as e:
        print(f"  Warning: could not parse log for seed {seed}: {e}")

    print(f"  Seed {seed} done. AUC={metrics.get('auc', 'N/A')}")
    return metrics


def summarise(results: list, output_dir: str, model_type: str):
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "sweep_runs.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nPer-run results saved → {csv_path}")

    # Compute mean ± std for numeric columns
    numeric = df.select_dtypes(include=[float, int]).drop(
        columns=["seed", "returncode"], errors="ignore")
    summary = {
        col: {"mean": float(numeric[col].mean()),
              "std":  float(numeric[col].std()),
              "min":  float(numeric[col].min()),
              "max":  float(numeric[col].max())}
        for col in numeric.columns
    }
    summary["model_type"] = model_type or "unknown"
    summary["n_seeds"] = len(results)

    json_path = os.path.join(output_dir, "sweep_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {json_path}")

    # Print table
    print(f"\n{'─'*50}")
    print(f"  {'Metric':<20} {'Mean':>8}  {'±Std':>8}")
    print(f"{'─'*50}")
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k:<20} {v['mean']:>8.4f}  {v['std']:>8.4f}")
    print(f"{'─'*50}")

    # Plot AUC distribution
    if "auc" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(len(df)), df["auc"], color="steelblue", alpha=0.8)
        ax.axhline(df["auc"].mean(), color="red", linestyle="--",
                   label=f"Mean={df['auc'].mean():.4f} ± {df['auc'].std():.4f}")
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([f"seed\n{s}" for s in df["seed"]])
        ax.set_ylabel("Test AUC")
        ax.set_title(f"AUC across seeds — {model_type or 'model'}")
        ax.legend()
        ax.set_ylim(0, 1)
        fig.tight_layout()
        plot_path = os.path.join(output_dir, "sweep_auc.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"AUC plot saved → {plot_path}")

    return summary


def collect_existing(seeds: list, output_dir: str) -> list:
    """Parse results from already-completed seed_*/train.log files."""
    results = []
    for seed in seeds:
        log_path = os.path.join(output_dir, f"seed_{seed}", "train.log")
        metrics = {"seed": seed, "returncode": 0}
        if not os.path.exists(log_path):
            print(f"  Warning: log not found for seed {seed}: {log_path}")
            metrics["returncode"] = 1
            results.append(metrics)
            continue
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
            print(f"  Warning: could not parse log for seed {seed}: {e}")
        print(f"  Seed {seed}: AUC={metrics.get('auc', 'N/A')}")
        results.append(metrics)
    return results


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.summarise_only:
        print("Summarise-only mode: reading existing logs …")
        results = collect_existing(args.seeds, args.output_dir)
    else:
        results = []
        for seed in args.seeds:
            r = run_seed(seed, args.config, args.model_type,
                         args.output_dir, args.no_wandb, args.extra or [])
            results.append(r)

    summarise(results, args.output_dir, args.model_type)


if __name__ == "__main__":
    main()
