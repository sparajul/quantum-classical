"""
utils/callbacks.py — PyTorch Lightning callbacks for publication-quality training.

Callbacks
---------
  WandBMetricsCallback : logs all physics metrics + plots to WandB at epoch end
  GradientMonitor      : tracks gradient norms, warns on vanishing/exploding
  TimingCallback       : wall-clock time per epoch
  SummaryCallback      : clean per-epoch summary table to stdout (HPC-friendly)

Bug fix: TimingCallback.on_train_epoch_end would crash if on_train_epoch_start
was never called (e.g. resumed training). Initialise _t0 in __init__.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import pytorch_lightning as pl
import torch

logger = logging.getLogger(__name__)

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from utils.wandb_plots import log_epoch_plots


def _wandb_active() -> bool:
    return _WANDB_AVAILABLE and wandb.run is not None


# ── WandB epoch-end callback ──────────────────────────────────────────────────
class WandBMetricsCallback(pl.Callback):

    def _log(self, trainer, pl_module, split: str) -> None:
        if not _wandb_active():
            return
        metrics = getattr(pl_module, f"{split}_metrics_epoch", None)
        if metrics is None:
            return
        step = trainer.global_step
        wandb.log(metrics.to_dict(prefix=split), step=step)
        log_epoch_plots(metrics, split=split, epoch=step)

    def on_train_epoch_end(self, trainer, pl_module):
        self._log(trainer, pl_module, "train")

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log(trainer, pl_module, "val")

    def on_test_epoch_end(self, trainer, pl_module):
        self._log(trainer, pl_module, "test")

# ── Gradient monitor ──────────────────────────────────────────────────────────

class GradientMonitor(pl.Callback):
    """
    Monitors gradient norms every N steps.
    Warns on vanishing (norm < threshold) or exploding (norm > threshold) gradients.
    """

    def __init__(
        self,
        log_every_n_steps: int = 50,
        warn_vanish: float = 1e-6,
        warn_explode: float = 100.0,
    ) -> None:
        self.log_every    = log_every_n_steps
        self.warn_vanish  = warn_vanish
        self.warn_explode = warn_explode

    def on_after_backward(self, trainer, pl_module) -> None:
        step = trainer.global_step
        if step % self.log_every != 0:
            return

        total_sq = 0.0
        max_norm = 0.0

        for param in pl_module.parameters():
            if param.grad is not None:
                norm = param.grad.detach().norm(2).item()
                total_sq += norm ** 2
                max_norm = max(max_norm, norm)

        total_norm = total_sq ** 0.5

        if total_norm < self.warn_vanish:
            logger.warning(
                "Vanishing gradients: total_norm=%.2e (step %d)", total_norm, step,
            )
        if total_norm > self.warn_explode:
            logger.warning(
                "Exploding gradients: total_norm=%.2e (step %d)", total_norm, step,
            )

        if _wandb_active():
            wandb.log(
                {"gradients/total_norm": total_norm, "gradients/max_layer_norm": max_norm},
                step=step,
            )


# ── Timing callback ───────────────────────────────────────────────────────────

class TimingCallback(pl.Callback):
    """Logs wall-clock time per epoch."""

    def __init__(self) -> None:
        super().__init__()
        self._t0: float = 0.0  # default prevents crash on resume

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._t0 = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        elapsed = time.perf_counter() - self._t0
        logger.info("[Timing] epoch=%d  elapsed=%.1fs", trainer.current_epoch, elapsed)
        if _wandb_active():
            wandb.log({"timing/epoch_seconds": elapsed}, step=trainer.current_epoch)


# ── Summary callback ──────────────────────────────────────────────────────────

class SummaryCallback(pl.Callback):
    """
    Prints a clean aligned summary to stdout at each validation epoch end.
    Designed for HPC cluster logs where rich progress bars break.

    Example output::

        ──────────────────────────────────────────────────────────────────────
          Epoch  12/100
          TRAIN  loss=0.3421  eff=0.9123  pur=0.8741  fake=0.1259  f1=0.8923  auc=0.9512
          VAL    loss=0.3701  eff=0.9002  pur=0.8620  fake=0.1380  f1=0.8806  auc=0.9488
        ──────────────────────────────────────────────────────────────────────
    """

    @staticmethod
    def _fmt(m, label: str) -> str:
        if m is None:
            return f"  {label:<6} — (no data)"
        return (
            f"  {label:<6} "
            f"loss={m.loss:.4f}  "
            f"eff={m.efficiency:.4f}  "
            f"pur={m.purity:.4f}  "
            f"fake={m.fake_rate:.4f}  "
            f"f1={m.f1:.4f}  "
            f"auc={m.auc:.4f}"
        )

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        epoch   = trainer.current_epoch
        max_ep  = trainer.max_epochs
        train_m = getattr(pl_module, "train_metrics_epoch", None)
        val_m   = getattr(pl_module, "val_metrics_epoch",   None)
        sep     = "─" * 78
        print(f"\n{sep}")
        print(f"  Epoch {epoch + 1:>3}/{max_ep}")
        print(self._fmt(train_m, "TRAIN"))
        print(self._fmt(val_m,   "VAL"))
        print(f"{sep}\n")

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        test_m = getattr(pl_module, "test_metrics_epoch", None)
        if test_m is None:
            return
        print(f"\n{'═' * 78}")
        print("  TEST RESULTS")
        print(
            f"  loss={test_m.loss:.4f}  eff={test_m.efficiency:.4f}  "
            f"pur={test_m.purity:.4f}  fake={test_m.fake_rate:.4f}  "
            f"f1={test_m.f1:.4f}  auc={test_m.auc:.4f}"
        )
        print(f"  TP={test_m.tp}  FP={test_m.fp}  TN={test_m.tn}  FN={test_m.fn}")
        print(f"{'═' * 78}\n")
