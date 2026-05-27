#!/usr/bin/env python
"""
scripts/train.py
─────────────────
Publication-quality training entry point for QuantumInteractionGNN.

Features
--------
  - WandB experiment tracking with run naming, tags, and config logging
  - Full physics metrics: efficiency, purity, fake_rate, F1, AUC, avg_precision
  - ROC curve, PR curve, score distribution, confusion matrix plots in WandB
  - Gradient monitoring
  - GPU/CPU auto-detection with lightning.gpu quantum device if available
  - Checkpoint best by val/auc_epoch
  - WandB sweep compatible (pass --sweep for hyperparameter search)

Usage
-----
# Basic
python scripts/train.py --config configs/default.yaml

# With WandB (recommended)
python scripts/train.py --config configs/default.yaml --wandb_project my_project

# Override any hparam
python scripts/train.py --config configs/default.yaml --lr 0.0002 --n_qubits 4

# Resume
python scripts/train.py --config configs/default.yaml --resume checkpoints/last.ckpt

# WandB sweep (define sweep config in wandb, then call this script)
python scripts/train.py --config configs/default.yaml --sweep
"""

from __future__ import annotations

import argparse
import logging
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger, CSVLogger

from data.dataset import GraphDataset
from models.gnn import QuantumInteractionGNN
from utils.callbacks import (
    WandBMetricsCallback, GradientMonitor, TimingCallback, SummaryCallback,
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train QuantumInteractionGNN")
    p.add_argument("--config",    type=str, default="configs/default.yaml")
    p.add_argument("--resume",    type=str, default=None,  help=".ckpt to resume from")
    p.add_argument("--sweep",     action="store_true",     help="WandB sweep mode")

    # WandB
    p.add_argument("--wandb_project", type=str, default=None, help="WandB project name")
    p.add_argument("--wandb_entity",  type=str, default=None, help="WandB entity/team")
    p.add_argument("--wandb_name",    type=str, default=None, help="WandB run name")
    p.add_argument("--wandb_tags",    type=str, default=None, help="Comma-separated tags")
    p.add_argument("--no_wandb",      action="store_true",    help="Disable WandB entirely")
    p.add_argument("--run",           type=str, default=None, help="Run ID for checkpoint dir (default: SLURM_JOB_ID or timestamp)")

    # Hparam overrides
    p.add_argument("--model_type",    type=str,   default=None, choices=["classical", "quantum"])
    p.add_argument("--max_epochs",    type=int,   default=None)
    p.add_argument("--lr",            type=float, default=None)
    p.add_argument("--min_lr",        type=float, default=None)
    p.add_argument("--n_qubits",      type=int,   default=None)
    p.add_argument("--n_qlayers",     type=int,   default=None)
    p.add_argument("--hidden",        type=int,   default=None)
    p.add_argument("--n_graph_iters", type=int,   default=None)
    p.add_argument("--batch_size",    type=int,   default=None)
    p.add_argument("--edge_cut",      type=float, default=None)
    p.add_argument("--pos_weight",    type=float, default=None)
    p.add_argument("--weight_decay",  type=float, default=None)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--stage_dir",     type=str,   default=None, help="Override stage_dir from config")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(hparams: dict, args: argparse.Namespace) -> dict:
    skip = {"config", "resume", "sweep", "wandb_project", "wandb_entity",
            "wandb_name", "wandb_tags", "no_wandb", "seed", "run"}
    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k not in skip}
    if overrides:
        logger.info("CLI overrides: %s", overrides)
    hparams.update(overrides)
    return hparams


# ─────────────────────────────────────────────────────────────────────────────
# Device / quantum backend selection
# ─────────────────────────────────────────────────────────────────────────────

def _test_device(device_name: str) -> bool:
    """Run a tiny circuit to confirm the device works without hanging."""
    import signal
    import pennylane as qml

    def _timeout(signum, frame):
        raise TimeoutError(f"{device_name} timed out")

    try:
        dev = qml.device(device_name, wires=2)

        @qml.qnode(dev, interface="torch")
        def _circuit():
            qml.PauliX(wires=0)
            return qml.expval(qml.PauliZ(0))

        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(10)
        try:
            result = _circuit()
            _ = float(result)
        finally:
            signal.alarm(0)
        return True
    except Exception as e:
        logger.debug("Device %s failed test: %s", device_name, e)
        return False


def resolve_quantum_device(hparams: dict) -> str:
    """
    Auto-select the best quantum simulation device, verified with an actual
    circuit execution. Falls back down the chain if a device hangs or fails.
      lightning.gpu   → fastest (CUDA), tested before use
      lightning.qubit → fast CPU
      default.qubit   → pure Python fallback, always works
    """
    requested = hparams.get("quantum_device", "default.qubit")

    # User explicitly named a non-auto device — respect it
    if requested not in {"default.qubit", "lightning.qubit", "lightning.gpu", "auto"}:
        logger.info("Quantum device: %s (user-specified)", requested)
        return requested

    try:
        import pennylane as qml  # noqa: F401
    except ImportError:
        logger.info("Quantum device: default.qubit (pennylane not found)")
        return "default.qubit"

    if requested == "lightning.gpu":
        candidates = ["lightning.gpu", "lightning.qubit", "default.qubit"]
    elif requested == "lightning.qubit":
        candidates = ["lightning.qubit", "default.qubit"]
    else:  # "auto" or "default.qubit"
        if torch.cuda.is_available():
            candidates = ["lightning.gpu", "lightning.qubit", "default.qubit"]
        else:
            candidates = ["lightning.qubit", "default.qubit"]

    for device_name in candidates:
        logger.info("Testing quantum device: %s ...", device_name)
        if _test_device(device_name):
            logger.info("Quantum device: %s ✓", device_name)
            return device_name
        logger.warning("Quantum device %s failed or timed out — trying next.", device_name)

    logger.info("Quantum device: default.qubit (fallback)")
    return "default.qubit"


# ─────────────────────────────────────────────────────────────────────────────
# Loggers
# ─────────────────────────────────────────────────────────────────────────────

def build_loggers(args, hparams: dict):
    loggers = []

    # Always log to CSV (offline backup)
    csv_logger = CSVLogger(
        save_dir=hparams.get("stage_dir", "outputs/"),
        name="csv_logs",
    )
    loggers.append(csv_logger)

    # TensorBoard (always — free, local)
    tb_logger = TensorBoardLogger(
        save_dir=hparams.get("stage_dir", "outputs/"),
        name="tensorboard",
    )
    loggers.append(tb_logger)

    # WandB (optional but strongly recommended for publication)
    if not args.no_wandb and (hparams.get("wandb_project") or args.wandb_project):
        try:
            project = args.wandb_project or hparams.get("wandb_project", "quantum-gnn")
            entity  = args.wandb_entity  or hparams.get("wandb_entity", None)
            tags    = []
            if args.wandb_tags:
                tags = [t.strip() for t in args.wandb_tags.split(",")]
            tags += hparams.get("wandb_tags", [])

            run_name = args.wandb_name or (
                f"qgnn-q{hparams['n_qubits']}"
                f"-l{hparams.get('n_qlayers',2)}"
                f"-h{hparams['hidden']}"
                f"-lr{hparams['lr']}"
            )

            wandb_logger = WandbLogger(
                project=project,
                entity=entity,
                name=run_name,
                tags=tags,
                config=hparams,
                log_model=True,   # saves checkpoints as WandB artifacts
            )
            loggers.append(wandb_logger)
            logger.info("WandB logger: project=%s  run=%s", project, run_name)
        except Exception as exc:
            logger.warning("WandB init failed (%s) — continuing without it.", exc)

    return loggers


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(hparams: dict, args: argparse.Namespace) -> None:

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    # ── Auto-select quantum device ────────────────────────────────────────────
    hparams["quantum_device"] = resolve_quantum_device(hparams)

    # ── Datasets ──────────────────────────────────────────────────────────────
    input_dir = hparams["input_dir"]
    logger.info("Loading datasets from %s …", input_dir)

    data_split = hparams.get("data_split", None)
    n_train = data_split[0] if data_split else None
    n_val   = data_split[1] if data_split else None
    n_test  = data_split[2] if data_split else None

    train_dataset = GraphDataset(input_dir, "train_set/", preprocess=True, hparams=hparams, max_events=n_train)
    val_dataset   = GraphDataset(input_dir, "val_set/",   preprocess=True, hparams=hparams, max_events=n_val)
    test_dataset  = GraphDataset(input_dir, "test_set/",  preprocess=True, hparams=hparams, max_events=n_test)

    logger.info("train=%d  val=%d  test=%d",
                len(train_dataset), len(val_dataset), len(test_dataset))

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.resume:
        logger.info("Resuming from: %s", args.resume)
        model = QuantumInteractionGNN.load_from_checkpoint(args.resume, hparams=hparams)
    else:
        model = QuantumInteractionGNN(hparams)

    model.train_dataset = train_dataset
    model.val_dataset   = val_dataset
    model.test_dataset  = test_dataset

    # ── Run ID → per-run checkpoint directory ────────────────────────────────
    run_id  = args.run or os.environ.get("SLURM_JOB_ID") or \
              datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join(hparams.get("stage_dir", "outputs"),
                            "checkpoints", f"run_{run_id}")
    logger.info("Checkpoint dir: %s", ckpt_dir)

    callbacks = [
        # Save best by F1 and last checkpoint only
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-f1-{epoch:03d}-{val/f1_epoch:.4f}",
            monitor="val/f1_epoch",
            mode="max",
            save_top_k=1,
            save_last=True,
            verbose=True,
        ),
        EarlyStopping(
            monitor="val/f1_epoch",
            mode="max",
            patience=hparams.get("patience", 15),
            min_delta=1e-4,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        WandBMetricsCallback(),    # epoch-end WandB plots
        GradientMonitor(log_every_n_steps=50),
        TimingCallback(),
        SummaryCallback(),
    ]

    # ── Loggers ───────────────────────────────────────────────────────────────
    pl_loggers = build_loggers(args, hparams)

    # ── Trainer ───────────────────────────────────────────────────────────────
    accelerator = hparams.get("accelerator", "auto")
    if accelerator == "gpu" and not torch.cuda.is_available():
        logger.warning("GPU requested but not available — using CPU.")
        accelerator = "cpu"

    trainer = pl.Trainer(
        max_epochs=hparams["max_epochs"],
        accelerator=accelerator,
        devices=hparams.get("devices", 1),
        callbacks=callbacks,
        logger=pl_loggers,
        log_every_n_steps=1,
        deterministic=False,  # required for PennyLane GPU compatibility
        gradient_clip_val=hparams.get("gradient_clip_val", 1.0),  # prevent exploding grads
        default_root_dir=hparams.get("stage_dir", "outputs/"),
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info("  quantum_device : %s", hparams["quantum_device"])
    logger.info("  n_qubits       : %d", hparams["n_qubits"])
    logger.info("  n_graph_iters  : %d", hparams["n_graph_iters"])
    logger.info("  hidden         : %d", hparams["hidden"])
    logger.info("  lr             : %g", hparams["lr"])
    logger.info("=" * 60)

    trainer.fit(model, ckpt_path=args.resume)

    # ── Test ──────────────────────────────────────────────────────────────────
    logger.info("Running test set evaluation …")
    trainer.test(model, ckpt_path="best")

    best = trainer.checkpoint_callback.best_model_path
    logger.info("Best checkpoint saved: %s", best)

    return best


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    hparams = load_config(args.config)
    hparams = apply_overrides(hparams, args)

    if args.sweep:
        # WandB sweep: override hparams with sweep config
        try:
            import wandb
            with wandb.init() as run:
                # Sweep agent provides values via wandb.config
                hparams.update(dict(run.config))
                train(hparams, args)
        except ImportError:
            logger.error("wandb not installed — cannot run sweep.")
            sys.exit(1)
    else:
        train(hparams, args)


if __name__ == "__main__":
    main()
