#!/usr/bin/env python
"""
scripts/inference.py
─────────────────────
Run inference (forward pass + metric evaluation) on a trained checkpoint.

Usage
-----
# Evaluate on the test split and save per-graph predictions
python scripts/inference.py \\
    --config   configs/default.yaml \\
    --checkpoint checkpoints/best.ckpt \\
    --split    test \\
    --output_dir results/

# Quick validation-set check (no file output)
python scripts/inference.py \\
    --config   configs/default.yaml \\
    --checkpoint checkpoints/best.ckpt \\
    --split    val

Output files (in --output_dir)
──────────────────────────────
  predictions_<split>.pt  — list of dicts, one per graph:
      {
        "file": str,            # source filename
        "scores": Tensor[N],    # sigmoid edge probabilities
        "y":      Tensor[N],    # ground-truth labels
        "loss":   float,
        "acc":    float,
        "auc":    float,
      }
  metrics_<split>.yaml    — aggregated loss / acc / auc over the split
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from torch_geometric.loader import DataLoader

from data.dataset import GraphDataset
from models.gnn import QuantumInteractionGNN
from utils.metrics import compute_metrics, threshold_scores

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inference")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference with QuantumInteractionGNN")
    parser.add_argument("--config",          type=str, required=True,  help="Path to base YAML config")
    parser.add_argument("--config_override", type=str, default=None,   help="Optional override YAML (e.g. configs/ibm_hardware.yaml)")
    parser.add_argument("--checkpoint",      type=str, required=True,  help="Path to .ckpt file")
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"],
        help="Dataset split to run inference on",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save predictions and metrics (optional)",
    )
    parser.add_argument(
        "--edge_cut",    type=float, default=None, help="Override edge_cut threshold")
    parser.add_argument(
        "--ibm_backend", type=str,   default=None, help="IBM machine name e.g. ibm_brisbane")
    parser.add_argument(
        "--batch_size",  type=int,   default=1,    help="Inference batch size")
    parser.add_argument(
        "--num_workers", type=int,   default=0,    help="DataLoader workers")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    with open(args.config, "r") as fh:
        hparams = yaml.safe_load(fh)

    # Merge hardware override config on top (e.g. configs/ibm_hardware.yaml)
    if args.config_override:
        with open(args.config_override, "r") as fh:
            overrides = yaml.safe_load(fh)
        hparams.update(overrides)
        logger.info("Applied config override: %s", args.config_override)

    # CLI takes highest priority
    if args.ibm_backend:
        hparams["ibm_backend"] = args.ibm_backend

    edge_cut = args.edge_cut if args.edge_cut is not None else hparams.get("edge_cut", 0.5)
    logger.info("Using edge_cut = %.3f", edge_cut)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Inference device: %s", device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    split_dir_map = {"train": "train_set/", "val": "val_set/", "test": "test_set/"}
    sub_dir = split_dir_map[args.split]
    dataset = GraphDataset(
        hparams["input_dir"], sub_dir, preprocess=True, hparams=hparams
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    logger.info("Loaded %s split: %d graphs", args.split, len(dataset))

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Loading checkpoint: %s", args.checkpoint)
    # Checkpoint's saved hparams take priority over the YAML so that model
    # architecture (model_type, n_qubits, hidden, …) is always reconstructed
    # correctly regardless of what default.yaml says.
    ckpt_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_hparams = ckpt_data.get("hyper_parameters", {})
    merged_hparams = {**hparams, **ckpt_hparams}
    model = QuantumInteractionGNN.load_from_checkpoint(
        args.checkpoint,
        hparams=merged_hparams,
        map_location=device,
    )
    model.eval()
    model.to(device)
    logger.info("Model loaded and set to eval mode.")

    # ── Inference loop ────────────────────────────────────────────────────────
    all_results = []
    total_loss = 0.0
    total_acc  = 0.0
    total_auc  = 0.0
    n_batches  = 0

    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            logits = model(batch)
            y = batch.y.float()

            loss = criterion(logits, y).item()
            metrics = compute_metrics(logits, y, edge_cut=edge_cut)
            acc = metrics.accuracy
            auc = metrics.auc

            scores = torch.sigmoid(logits)
            pred_labels = threshold_scores(scores, edge_cut)

            # file_names indexing is only safe with batch_size=1
            file_name = dataset.file_names[batch_idx] if args.batch_size == 1 else f"batch_{batch_idx}"
            src, dst = batch.edge_index
            result = {
                "file":   file_name,
                "scores": scores.cpu(),
                "y":      y.cpu(),
                "preds":  pred_labels.cpu(),
                "loss":   loss,
                "acc":    acc,
                "auc":    auc,
                "n_edges":       scores.shape[0],
                "n_positive":    int(y.sum().item()),
                "n_pred_positive": int(pred_labels.sum().item()),
            }
            if hasattr(batch, "pt"):
                result["pt"]  = (0.5 * (batch.pt[src]  + batch.pt[dst])).float().cpu()
            if hasattr(batch, "eta"):
                result["eta"] = (0.5 * (batch.eta[src] + batch.eta[dst])).float().cpu()
            all_results.append(result)

            total_loss += loss
            total_acc  += acc
            total_auc  += auc
            n_batches  += 1

            logger.info(
                "[%s %d/%d]  loss=%.4f  acc=%.4f  auc=%.4f  "
                "edges=%d  true_pos=%d  pred_pos=%d",
                args.split, batch_idx + 1, len(loader),
                loss, acc, auc,
                result["n_edges"], result["n_positive"], result["n_pred_positive"],
            )

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    agg = {
        "split":    args.split,
        "n_graphs": n_batches,
        "mean_loss": total_loss / max(1, n_batches),
        "mean_acc":  total_acc  / max(1, n_batches),
        "mean_auc":  total_auc  / max(1, n_batches),
    }
    logger.info(
        "── Aggregate [%s] ─────────────────────────────────",
        args.split.upper(),
    )
    for k, v in agg.items():
        logger.info("  %-12s %s", k, f"{v:.4f}" if isinstance(v, float) else v)

    # ── Save outputs ──────────────────────────────────────────────────────────
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        pred_path = os.path.join(args.output_dir, f"predictions_{args.split}.pt")
        torch.save(all_results, pred_path)
        logger.info("Saved predictions → %s", pred_path)

        metrics_path = os.path.join(args.output_dir, f"metrics_{args.split}.yaml")
        with open(metrics_path, "w") as fh:
            yaml.dump(agg, fh, default_flow_style=False)
        logger.info("Saved metrics     → %s", metrics_path)


if __name__ == "__main__":
    main()
