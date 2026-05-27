#!/usr/bin/env python
"""
utils/barren_plateau.py
────────────────────────
Barren plateau analysis for the quantum model.

Measures gradient variance as a function of:
  - Number of qubits  (n_qubits)
  - Number of layers  (n_qlayers)

A barren plateau is indicated when gradient variance → 0 exponentially
with circuit depth/width. Publication requires showing this does NOT
occur at the scales used, or explicitly discussing mitigation strategies.

Usage
-----
  python utils/barren_plateau.py \
      --config configs/default.yaml \
      --checkpoint checkpoints/best.ckpt \
      --output_dir results/barren_plateau/
      --n_samples 50          # number of random parameter initialisations
      --qubit_range 2 4 6 8   # n_qubits values to sweep
      --layer_range 1 2 3 4   # n_qlayers values to sweep

Output
------
  results/barren_plateau/gradient_variance_qubits.png
  results/barren_plateau/gradient_variance_layers.png
  results/barren_plateau/barren_plateau_data.json
"""

from __future__ import annotations
import argparse, json, logging, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("barren_plateau")


def compute_gradient_variance(hparams: dict, n_samples: int,
                              n_qubits: int, n_qlayers: int) -> dict:
    """
    Randomly initialise VQC parameters n_samples times and compute
    the gradient of a simple loss w.r.t. each parameter.
    Returns variance statistics across initialisations.
    """
    from models.quantum_mlp import make_quantum_mlp

    hp = dict(hparams)
    hp["n_qubits"]  = n_qubits
    hp["n_qlayers"] = n_qlayers

    input_size  = len(hp["node_features"])
    output_size = hp["hidden"]

    all_grads = []

    for trial in range(n_samples):
        try:
            model = make_quantum_mlp(
                input_size=input_size,
                sizes=[output_size],
                hidden_activation=hp.get("hidden_activation", "ReLU"),
                output_activation=None,
                n_qubits=n_qubits,
                n_qlayers=n_qlayers,
                device=hp.get("quantum_device", "default.qubit"),
                layernorm=hp.get("layernorm", False),
            )

            # Random input
            x = torch.randn(8, input_size)
            out = model(x)
            loss = out.mean()
            loss.backward()

            # Collect gradients from VQC weights
            grads = []
            for name, p in model.named_parameters():
                if p.grad is not None:
                    grads.append(p.grad.detach().cpu().numpy().flatten())

            if grads:
                all_grads.append(np.concatenate(grads))

            # Zero grads for next trial
            model.zero_grad()

        except Exception as e:
            logger.warning("Trial %d failed (qubits=%d, layers=%d): %s",
                           trial, n_qubits, n_qlayers, e)

    if not all_grads:
        return {"variance": np.nan, "mean": np.nan, "std": np.nan, "n_samples": 0}

    all_grads = np.array(all_grads)   # [n_samples, n_params]
    # Variance averaged across parameters (then across samples)
    param_vars = np.var(all_grads, axis=0)   # variance across trials per param
    return {
        "variance":      float(np.mean(param_vars)),
        "mean":          float(np.mean(all_grads)),
        "std":           float(np.std(all_grads)),
        "n_params":      all_grads.shape[1],
        "n_samples":     len(all_grads),
        "n_qubits":      n_qubits,
        "n_qlayers":     n_qlayers,
    }


def plot_variance_sweep(x_vals, variances, xlabel, title, out_path, color="steelblue"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Linear scale
    axes[0].plot(x_vals, variances, "o-", color=color, linewidth=2, markersize=8)
    axes[0].set_xlabel(xlabel, fontsize=13)
    axes[0].set_ylabel("Gradient Variance", fontsize=13)
    axes[0].set_title(f"{title} (linear)", fontsize=13)
    axes[0].grid(True, alpha=0.3)

    # Log scale — barren plateau shows as straight line
    valid = [(x, v) for x, v in zip(x_vals, variances) if v > 0 and not np.isnan(v)]
    if valid:
        xv, vv = zip(*valid)
        axes[1].semilogy(xv, vv, "o-", color=color, linewidth=2, markersize=8)
        axes[1].set_xlabel(xlabel, fontsize=13)
        axes[1].set_ylabel("Gradient Variance (log)", fontsize=13)
        axes[1].set_title(f"{title} (log scale — linear = barren plateau)", fontsize=13)
        axes[1].grid(True, alpha=0.3)

    fig.suptitle("Barren Plateau Analysis", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved → %s", out_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       default="configs/default.yaml")
    p.add_argument("--output_dir",   default="results/barren_plateau/")
    p.add_argument("--n_samples",    type=int, default=50)
    p.add_argument("--qubit_range",  nargs="+", type=int, default=[2, 4, 6, 8, 10])
    p.add_argument("--layer_range",  nargs="+", type=int, default=[1, 2, 3, 4, 5])
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        hparams = yaml.safe_load(f)
    hparams["quantum_device"] = "default.qubit"   # always use simulator for this analysis

    all_data = {}

    # ── Sweep n_qubits ────────────────────────────────────────────────────────
    logger.info("Sweeping n_qubits: %s (n_qlayers fixed at %d)",
                args.qubit_range, hparams.get("n_qlayers", 2))
    qubit_variances = []
    fixed_layers    = hparams.get("n_qlayers", 2)
    for nq in args.qubit_range:
        logger.info("  n_qubits=%d ...", nq)
        r = compute_gradient_variance(hparams, args.n_samples, nq, fixed_layers)
        qubit_variances.append(r["variance"])
        all_data[f"qubits_{nq}_layers_{fixed_layers}"] = r
        logger.info("  variance=%.6e", r["variance"])

    plot_variance_sweep(
        args.qubit_range, qubit_variances,
        xlabel="Number of Qubits",
        title=f"Gradient Variance vs Qubits (n_qlayers={fixed_layers})",
        out_path=os.path.join(args.output_dir, "gradient_variance_qubits.png"),
    )

    # ── Sweep n_qlayers ───────────────────────────────────────────────────────
    fixed_qubits = hparams.get("n_qubits", 4)
    logger.info("Sweeping n_qlayers: %s (n_qubits fixed at %d)",
                args.layer_range, fixed_qubits)
    layer_variances = []
    for nl in args.layer_range:
        logger.info("  n_qlayers=%d ...", nl)
        r = compute_gradient_variance(hparams, args.n_samples, fixed_qubits, nl)
        layer_variances.append(r["variance"])
        all_data[f"qubits_{fixed_qubits}_layers_{nl}"] = r
        logger.info("  variance=%.6e", r["variance"])

    plot_variance_sweep(
        args.layer_range, layer_variances,
        xlabel="Number of VQC Layers",
        title=f"Gradient Variance vs Layers (n_qubits={fixed_qubits})",
        out_path=os.path.join(args.output_dir, "gradient_variance_layers.png"),
        color="tomato",
    )

    # ── Save data ─────────────────────────────────────────────────────────────
    json_path = os.path.join(args.output_dir, "barren_plateau_data.json")
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=2)
    logger.info("Data saved → %s", json_path)

    # ── Interpretation ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  BARREN PLATEAU INTERPRETATION")
    print("="*60)
    print("  If gradient variance decreases EXPONENTIALLY with qubits/layers,")
    print("  your circuit suffers from barren plateaus (untrainable at scale).")
    print("  If variance is ROUGHLY CONSTANT → safe from barren plateaus.")
    print()
    for i, (nq, var) in enumerate(zip(args.qubit_range, qubit_variances)):
        print(f"  n_qubits={nq:2d}  variance={var:.4e}")
    print("="*60)


if __name__ == "__main__":
    main()
