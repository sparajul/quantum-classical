"""
models/gnn.py — InteractionGNN: quantum or classical edge classifier.

    model_type: "quantum"    variational quantum circuits in every MLP block
    model_type: "classical"  plain Linear layers, identical architecture

Architecture
------------
  NodeEncoder(node_features → hidden)
  EdgeEncoder(edge_features[dr,dphi,dz,deta] → hidden)
  n_graph_iters × [EdgeNetwork, NodeNetwork]
  EdgeDecoder → EdgeOutputTransform(hidden → 1)

The edge and node networks can be shared (recurrent=True) or unshared
(recurrent=False, one set of weights per iteration).

``QuantumInteractionGNN`` is a backwards-compatible alias.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_add
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl

from models.classical_mlp import make_classical_mlp
from models.quantum_mlp import make_quantum_mlp
from utils.metrics import TrackingMetrics, MetricsAccumulator, compute_metrics

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

_HPARAM_DEFAULTS: Dict[str, Any] = {
    "model_type":                  "quantum",
    "concat":                      True,
    "in_out_diff_agg":             True,
    "edge_net_recurrent":          True,
    "node_net_recurrent":          True,
    "checkpointing":               True,
    "undirected_message_passing":  True,
    "quantum_device":              "default.qubit",
    "n_qubits":                    4,
    "n_qlayers":                   1,
    "n_shots":                     1024,
    "noise_mitigation":            False,
    "ibm_backend":                 None,
    "lr":                          1e-4,
    "min_lr":                      5e-5,
    "warmup":                      5,
    "max_epochs":                  100,
    "patience":                    15,
    "edge_cut":                    0.5,
    "pos_weight":                  None,
    "layernorm":                   False,
    "batchnorm":                   False,
    "track_running_stats":         False,
    "remove_electrons":            False,
    "factor":                      0.9,
    "quantum_lr_factor":           1.0,
    "vqc_warmup_epochs":           0,
    "data_reuploading":            False,
    "ring_entanglement":           False,
    "weight_decay":                0.0,
    "batch_size":                  1,
    "num_workers":                 [4, 4, 4],
}

# Keys that are consumed by other tools / codebases but must not crash here.
_IGNORED_KEYS = frozenset({
    "output_layer_norm", "edge_output_transform_final_layer_norm",
    "output_batch_norm", "edge_output_transform_final_batch_norm",
    "bn_track_running_stats", "edge_output_transform_final_activation",
    "max_training_graph_size", "debug",
})


# ── Parameter counting ────────────────────────────────────────────────────────

def _count_params(model: nn.Module):
    total = quantum = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        total += n
        if name.endswith(".weights") or "encoding" in name:
            quantum += n
    return total, quantum, total - quantum


_QUANTUM_MODEL_TYPES = frozenset({"quantum", "edge_quantum"})

_MODEL_TYPE_LABELS = {
    "quantum":      "QUANTUM-CLASSICAL HYBRID GNN (CTD 2025)",
    "edge_quantum": "EDGE-QUANTUM GNN (quantum edge_network only)",
    "classical":    "CLASSICAL GNN (baseline)",
}

# Maps model_type → which MLP blocks use the quantum circuit.
_QUANTUM_BLOCKS: Dict[str, frozenset] = {
    "quantum":      frozenset({"node_encoder", "edge_encoder",
                               "edge_network", "node_network",
                               "output_edge_classifier"}),
    "edge_quantum": frozenset({"edge_network"}),
    "classical":    frozenset(),
}


def _log_parameter_summary(model: nn.Module, model_type: str) -> None:
    total, quantum, classical = _count_params(model)
    # frozen_quantum params are not trainable — count them separately for display
    frozen = sum(
        p.numel() for name, p in model.named_parameters()
        if not p.requires_grad and (name.endswith(".weights") or "encoding" in name)
    )
    sep   = "=" * 62
    label = _MODEL_TYPE_LABELS.get(model_type, model_type.upper())
    print(sep)
    print("  Model type        : %s" % label)
    print("  Total  parameters : %d  (trainable)" % total)
    if model_type in _QUANTUM_MODEL_TYPES:
        print("  Quantum  (VQC)    : %d  (%.1f%%)" % (quantum,   100 * quantum   / max(total, 1)))
        if frozen:
            print("  Frozen VQC params : %d  (not in grad graph)" % frozen)
        print("  Classical (Linear): %d  (%.1f%%)" % (classical, 100 * classical / max(total, 1)))
    else:
        print("  Quantum  (VQC)    : 0  (0.0%%)")
        print("  Classical (Linear): %d  (100.0%%)" % total)
    print(sep)


# ── Model ─────────────────────────────────────────────────────────────────────

class InteractionGNN(pl.LightningModule):
    """
    Quantum-classical (or purely classical) Interaction Network GNN
    for particle track edge classification.

    Datasets are attached externally before ``trainer.fit()``:

        model.train_dataset = train_ds
        model.val_dataset   = val_ds
        model.test_dataset  = test_ds
    """

    def __init__(self, hparams: Dict[str, Any]) -> None:
        super().__init__()
        hp = {**_HPARAM_DEFAULTS, **hparams}
        for k in _IGNORED_KEYS:
            hp.setdefault(k, None)
        self.save_hyperparameters(hp)

        # Loss
        if hp["pos_weight"] is not None:
            pw = torch.tensor([hp["pos_weight"]], dtype=torch.float)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # Networks
        self._build_networks()

        # Node-feature normalisation scales (registered as buffer → saved in ckpt)
        self.register_buffer(
            "node_scales",
            torch.tensor(self.hparams["node_scales"], dtype=torch.float),
        )

        # Epoch-level metric accumulators
        self._train_acc = MetricsAccumulator()
        self._val_acc   = MetricsAccumulator()
        self._test_acc  = MetricsAccumulator()

        # Metrics written by on_*_epoch_end — readable by callbacks
        self.train_metrics_epoch: Optional[TrackingMetrics] = None
        self.val_metrics_epoch:   Optional[TrackingMetrics] = None
        self.test_metrics_epoch:  Optional[TrackingMetrics] = None

        _log_parameter_summary(self, self.hparams["model_type"])

    # ── Network construction ──────────────────────────────────────────────────

    def _mlp(self, input_size: int, sizes: List[int],
             output_activation: Optional[str] = "__cfg__",
             block: str = "") -> nn.Sequential:
        h   = self.hparams
        oa  = h["output_activation"] if output_activation == "__cfg__" else output_activation
        kw  = dict(
            input_size=input_size,
            sizes=sizes,
            hidden_activation=h["hidden_activation"],
            output_activation=oa,
            layernorm=h.get("layernorm", False),
            batchnorm=h.get("batchnorm", False),
            track_running_stats=h.get("track_running_stats", False),
        )
        quantum_blocks = _QUANTUM_BLOCKS.get(h["model_type"], frozenset())
        if block in quantum_blocks:
            return make_quantum_mlp(
                **kw,
                n_qubits=h["n_qubits"],
                n_qlayers=h["n_qlayers"],
                data_reuploading=h.get("data_reuploading", False),
                ring_entanglement=h.get("ring_entanglement", False),
            )
        return make_classical_mlp(**kw)

    def _build_networks(self) -> None:
        h    = self.hparams["hidden"]
        slot = 2 if self.hparams["concat"] else 1   # 2h if concat, h otherwise
        da   = self.hparams["in_out_diff_agg"]

        # Edge network input:
        #   concat=True  : cat([e, init_e, x[src], init_x[src], x[dst], init_x[dst]]) → 6h
        #   concat=False : cat([e, x[src], x[dst]])                                   → 3h
        in_e = h * 6 if self.hparams["concat"] else h * 3

        # Node network input:
        #   in_out_diff_agg=True  : cat([agg_in, agg_out, x_cat]) → h + h + slot*h
        #   in_out_diff_agg=False : cat([agg_in+agg_out, x_cat])  → h + slot*h
        in_n = h * 2 + slot * h if da else h + slot * h

        self.node_encoder = self._mlp(
            len(self.hparams["node_features"]),
            [h] * self.hparams["n_node_encoder_layers"],
            block="node_encoder",
        )
        edge_in = (len(self.hparams["edge_features"])
                   if self.hparams.get("edge_features")
                   else slot * h)
        self.edge_encoder = self._mlp(
            edge_in,
            [h] * self.hparams["n_edge_encoder_layers"],
            block="edge_encoder",
        )
        self.edge_network = self._make_recurrent_or_list(
            "edge_net_recurrent", in_e, self.hparams["n_edge_net_layers"],
            block="edge_network",
        )
        self.node_network = self._make_recurrent_or_list(
            "node_net_recurrent", in_n, self.hparams["n_node_net_layers"],
            block="node_network",
        )
        # Acorn-style output: cat([x[src], x[dst], e]) → MLP → 1
        # Incorporates node context into the final edge score (3h input always).
        self.output_edge_classifier = self._mlp(
            3 * h,
            [h] * self.hparams["n_edge_decoder_layers"] + [1],
            output_activation=None,
            block="output_edge_classifier",
        )

    def _make_recurrent_or_list(self, key: str, input_size: int, n_layers: int,
                                block: str = ""):
        h = self.hparams["hidden"]
        if self.hparams[key]:
            return self._mlp(input_size, [h] * n_layers, block=block)
        return nn.ModuleList([
            self._mlp(input_size, [h] * n_layers, block=block)
            for _ in range(self.hparams["n_graph_iters"])
        ])

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, batch) -> torch.Tensor:
        dev = batch.edge_index.device
        x = (
            torch.stack([batch[f] for f in self.hparams["node_features"]], dim=-1).float()
            / self.node_scales.to(dev)
        )
        edge_attr = (
            torch.stack([batch[f] for f in self.hparams["edge_features"]], dim=-1).float()
            if self.hparams.get("edge_features")
            else None
        )
        src, dst = batch.edge_index

        # Undirected: double edges so both directions are propagated simultaneously.
        # Scores are averaged over the two directions at the end (Acorn style).
        if self.hparams["undirected_message_passing"]:
            start = torch.cat([src, dst])
            end   = torch.cat([dst, src])
        else:
            start, end = src, dst

        if self.hparams["checkpointing"]:
            x   = checkpoint(self.node_encoder, x, use_reentrant=False)
            ein = edge_attr if edge_attr is not None else torch.cat([x[src], x[dst]], dim=-1)
            e   = checkpoint(self.edge_encoder, ein, use_reentrant=False)
        else:
            x   = self.node_encoder(x)
            ein = edge_attr if edge_attr is not None else torch.cat([x[src], x[dst]], dim=-1)
            e   = self.edge_encoder(ein)

        # Double edge embeddings to match doubled edge index.
        if self.hparams["undirected_message_passing"]:
            e = torch.cat([e, e])

        init_x, init_e = x, e
        for i in range(self.hparams["n_graph_iters"]):
            x, e = self._message_passing_step(x, e, init_x, init_e, start, end, i)

        # Acorn-style final classifier: incorporates node context into edge score.
        logits = self.output_edge_classifier(
            torch.cat([x[start], x[end], e], dim=-1)
        ).squeeze(-1)

        # Average forward and reverse scores for symmetric undirected output.
        if self.hparams["undirected_message_passing"]:
            logits = logits.view(2, -1).mean(dim=0)

        return logits

    def _message_passing_step(self, x, e, init_x, init_e, start, end, i: int):
        # ── Edge update ───────────────────────────────────────────────────────
        e_in = (
            torch.cat([e, init_e, x[start], init_x[start], x[end], init_x[end]], dim=-1)
            if self.hparams["concat"]
            else torch.cat([e, x[start], x[end]], dim=-1)
        )
        e_net = self.edge_network if self.hparams["edge_net_recurrent"] else self.edge_network[i]
        e_new = e_net(e_in)

        # ── Aggregation ───────────────────────────────────────────────────────
        n_nodes = x.shape[0]
        agg_in  = scatter_add(e_new, end,   dim=0, dim_size=n_nodes)
        agg_out = scatter_add(e_new, start, dim=0, dim_size=n_nodes)

        # ── Node update ───────────────────────────────────────────────────────
        x_cat = torch.cat([x, init_x], dim=-1) if self.hparams["concat"] else x
        n_in  = (
            torch.cat([agg_in, agg_out, x_cat], dim=-1)
            if self.hparams["in_out_diff_agg"]
            else torch.cat([agg_in + agg_out, x_cat], dim=-1)
        )
        n_net = self.node_network if self.hparams["node_net_recurrent"] else self.node_network[i]
        x_new = n_net(n_in)

        return x_new, e_new

    # ── Shared step ───────────────────────────────────────────────────────────

    def _shared_step(self, batch, split: str, acc: MetricsAccumulator) -> torch.Tensor:
        logits = self(batch)
        y      = batch.y.float()

        if getattr(batch, "weights", None) is not None:
            pw = getattr(self.criterion, "pos_weight", None)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, y,
                weight=batch.weights.to(logits.device),
                pos_weight=pw.to(logits.device) if pw is not None else None,
            )
        else:
            loss = self.criterion(logits, y)

        m   = compute_metrics(logits, y, loss=loss.item(),
                              edge_cut=self.hparams["edge_cut"], compute_curves=False)
        acc.update(logits, y, loss.item())

        bs = batch.edge_index.shape[1]  # number of edges = actual batch size

        epoch_kw = dict(on_step=False, on_epoch=True, prog_bar=True,
                        sync_dist=True, batch_size=bs)
        self.log(f"{split}/loss",       loss,         **epoch_kw)
        self.log(f"{split}/efficiency", m.efficiency, **epoch_kw)
        self.log(f"{split}/purity",     m.purity,     **epoch_kw)
        self.log(f"{split}/fake_rate",  m.fake_rate,  **epoch_kw)
        self.log(f"{split}/f1",         m.f1,         **epoch_kw)
        self.log(f"{split}/auc",        m.auc,        **epoch_kw)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train", self._train_acc)

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val", self._val_acc)

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test", self._test_acc)

    # ── Epoch end ─────────────────────────────────────────────────────────────

    def _epoch_end(self, acc: MetricsAccumulator, split: str) -> None:
        m = acc.compute(edge_cut=self.hparams["edge_cut"])
        acc.reset()
        setattr(self, f"{split}_metrics_epoch", m)

        kw = dict(on_step=False, on_epoch=True, prog_bar=False,
                  sync_dist=True, batch_size=1)
        scalars = [
            ("auc_epoch",           m.auc),
            ("avg_precision_epoch", m.avg_precision),
            ("efficiency_epoch",    m.efficiency),
            ("purity_epoch",        m.purity),
            ("fake_rate_epoch",     m.fake_rate),
            ("f1_epoch",            m.f1),
            ("loss_epoch",          m.loss),
            ("tp",                  float(m.tp)),
            ("fp",                  float(m.fp)),
            ("tn",                  float(m.tn)),
            ("fn",                  float(m.fn)),
        ]
        for key, val in scalars:
            self.log(f"{split}/{key}", val, **kw)

        logger.info("[%s epoch %d] %s", split.upper(), self.current_epoch, m.summary_str())

    def on_train_epoch_end(self):      self._epoch_end(self._train_acc, "train")
    def on_validation_epoch_end(self): self._epoch_end(self._val_acc,   "val")
    def on_test_epoch_end(self):       self._epoch_end(self._test_acc,  "test")

    # ── Optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("Optimizer: %d trainable params (lr=%.5f)", n_params, self.hparams["lr"])

        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams.get("weight_decay", 0.0),
        )
        warmup  = self.hparams["warmup"]
        total   = self.hparams["max_epochs"]
        ws = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda e: (e + 1) / max(1, warmup),
        )
        cs = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, total - warmup), eta_min=self.hparams["min_lr"],
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[ws, cs], milestones=[warmup],
        )
        return {
            "optimizer":    opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval":  "epoch",
                "monitor":   "val/auc_epoch",
                "frequency": 1,
            },
        }

    # ── DataLoaders ───────────────────────────────────────────────────────────

    def _loader(self, split: str) -> DataLoader:
        attr = f"{split}_dataset"
        if not hasattr(self, attr):
            raise AttributeError(
                f"Attach a dataset before training: model.{attr} = <GraphDataset>"
            )
        nw_cfg = self.hparams.get("num_workers", [4, 4, 4])
        nw = (
            nw_cfg[["train", "val", "test"].index(split)]
            if isinstance(nw_cfg, (list, tuple))
            else nw_cfg
        )
        return DataLoader(
            getattr(self, attr),
            batch_size=self.hparams.get("batch_size", 1),
            shuffle=(split == "train"),
            num_workers=nw,
            pin_memory=torch.cuda.is_available(),
        )

    def train_dataloader(self): return self._loader("train")
    def val_dataloader(self):   return self._loader("val")
    def test_dataloader(self):  return self._loader("test")


# Backwards-compatible alias
QuantumInteractionGNN = InteractionGNN