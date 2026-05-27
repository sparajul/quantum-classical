# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Quantum-classical hybrid Graph Neural Network for real-time particle track edge classification at ATLAS/HL-LHC. Given a graph of detector hits (~10K nodes, ~40K edges per event), the model classifies each edge as belonging to a real particle track (signal, ~5%) or random noise (background, ~95%). Variational quantum circuits (VQCs) can replace classical MLP blocks inside message-passing layers.

## Common Commands

```bash
# Training (YAML config + CLI overrides via dot notation)
python scripts/train.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml model_type=quantum n_qubits=4 n_qlayers=2

# Inference on a data split
python scripts/inference.py --config configs/default.yaml --checkpoint path/to/best.ckpt --split test

# IBM hardware inference
python scripts/inference.py --config configs/ibm_hardware.yaml --checkpoint path/to/best.ckpt

# WandB hyperparameter sweep
python scripts/train_sweep.py --config configs/scaling_sweep.yaml

# IBM backend utilities (list available devices)
python scripts/ibm_utils.py --list-backends --n-qubits 4
```

There are no unit tests or linting configuration in this repository.

## Architecture Overview

### Model Pipeline

```
Input: PyG Data
  nodes: [r, φ, z, η]  (cylindrical detector coords + pseudorapidity)
  edges: [Δr, Δφ, Δz, Δη, φslope, rφslope]  (auto-computed in dataset.py)
  labels: y ∈ {0, 1}
  │
  ├─ NodeEncoder(coords → hidden dim h)
  ├─ EdgeEncoder(6 features → h)
  ├─ × n_graph_iters (default 6):
  │   ├─ EdgeNetwork: MLP(6h → h)       — edge embedding update
  │   ├─ Aggregation: scatter_add       — separate in/out edges per node
  │   └─ NodeNetwork: MLP(3h or 4h → h) — node embedding update
  ├─ EdgeDecoder: MLP(h → h)
  └─ EdgeOutputTransform: Linear(h → 1) + sigmoid → edge probability
```

Each MLP block is either **classical** (`models/classical_mlp.py`) or **quantum** (`models/quantum_mlp.py`), controlled by `model_type` in config. Initial embeddings (e₀, x₀) are concatenated at every iteration to prevent over-smoothing.

### Key Files

| File | Purpose |
|------|---------|
| `models/gnn.py` | `InteractionGNN` — PyTorch Lightning module; message-passing loop, train/val/test steps, metric accumulation |
| `models/quantum_mlp.py` | `VQCLayer` — QPLT: VQC runs once/batch with learnable encoding angles → expectation values → weight matrix → batched matmul |
| `models/classical_mlp.py` | Drop-in classical MLP with optional LayerNorm |
| `models/backend_factory.py` | Routes `make_quantum_mlp()` to pennylane / qiskit-ml / qiskit-native backends |
| `models/qiskit_native_layer.py` | Manual parameter-shift differentiation for IBM hardware (no autograd dependency) |
| `data/dataset.py` | `GraphDataset` — loads `.pyg` files, computes edge features, applies hard cuts (remove electron edges), handles graph trimming |
| `utils/metrics.py` | `TrackingMetrics` + `MetricsAccumulator` — efficiency, purity, fake_rate, F1, AUC, avg_precision |
| `utils/callbacks.py` | WandB logging, gradient monitoring, timing, stdout summary (HPC-friendly) |
| `scripts/train.py` | Training pipeline: YAML + CLI config, WandB integration, two checkpoints (best-F1 + last) |
| `scripts/inference.py` | Load checkpoint → evaluate → save `predictions.pt` + `metrics.yaml` |

### Quantum Circuit Design (`models/quantum_mlp.py`)

**QPLT (Quantum-Parametrised Linear Transform) — runs once per forward pass, not per sample:**
1. VQC encodes learnable angles `self.encoding ∈ ℝ^{n_qubits}` (not per-sample data) via `RY` gates
2. Variational layers: `[RX, RZ, RY] → [CNOT ladder] → [CRZ pairs]` × n_qlayers
3. Measurement: `⟨Z⟩` expectation value per qubit → `e ∈ [-1,1]^{n_qubits}`
4. Build weight matrix: `W = outer(e, e) + diag(e)` — classical
5. Apply to batch: `output = tanh(x @ W.T)` — GPU matmul over all ~40K edges

Data features `x` enter through the classical matmul in step 5, **not** through the quantum circuit directly.

**Device auto-selection:** `lightning.gpu` → `lightning.qubit` → `default.qubit`

**Differentiation method by backend:**
- `lightning.gpu` / `lightning.qubit`: adjoint (fast)
- `default.qubit`: backprop
- Qiskit Aer / IBM hardware: parameter_shift (universal, slower)

### Configuration System

YAML files in `configs/`. Key parameters:

| Parameter | Description |
|-----------|-------------|
| `model_type` | `"classical"` or `"quantum"` |
| `n_qubits`, `n_qlayers` | Quantum circuit dimensions |
| `hidden` | GNN hidden dimension |
| `n_graph_iters` | Message-passing iterations (default 6) |
| `pos_weight` | BCE loss upweighting for rare true edges (~30) |
| `quantum_device` | PennyLane device string or `"qiskit.ibmq"` |

`configs/ibm_hardware.yaml` overlays `default.yaml` for real-hardware inference with reduced qubit counts (n_qubits=4). The `noise_mitigation` flag is accepted but not yet implemented — ZNE is planned for a future release.

### Training Details

- **Loss:** `BCEWithLogitsLoss` with `pos_weight ≈ 30` for class imbalance (~5% signal)
- **Optimizer:** Adam (lr=5e-4), linear warmup 5 epochs + cosine annealing
- **Early stopping:** patience=15 on `val/f1_epoch`
- **Two checkpoints saved:** `best-f1-{epoch}-{val/f1_epoch}.ckpt` (best validation F1) + `last.ckpt`
- **Gradient checkpointing:** Enabled to handle ~40K edges/graph (recomputes activations on backward)
- **WandB run naming:** `qgnn-q<n_qubits>-l<n_qlayers>-h<hidden>-lr<lr>`

### Primary Metrics (computed in `utils/metrics.py`)

| Metric | Target | Definition |
|--------|--------|------------|
| Efficiency | ≥0.95 | TP/(TP+FN) — fraction of true edges retained |
| Purity | ≥0.90 | TP/(TP+FP) — quality of retained edges |
| AUC | logged | Tracked via `MetricsAccumulator`; F1 is used for checkpointing/early-stopping |
| Avg Precision | secondary | PR-curve AUC; preferred for imbalanced data |

Per-step metrics (fast, single-batch) are logged every step; full-dataset AUC/AP are accumulated per epoch via `MetricsAccumulator`.

### IBM Hardware Workflow

Train on simulation → export checkpoint → swap to `configs/ibm_hardware.yaml` → run `inference.py` on real hardware. PennyLane is used throughout; setting `quantum_device: qiskit.ibmq` routes circuit execution through PennyLane's Qiskit plugin to IBM hardware. `qiskit_native_layer.py` provides a pure-Qiskit alternative with manual parameter-shift gradients (no PennyLane dependency) for maximum hardware compatibility. ZNE noise mitigation is **not yet implemented** — the `noise_mitigation` flag is a placeholder for a future release.

## Data Format

**Source:** OpenML ttbar simulation (no pileup) — `ttbar_pu0_tracker_hits` and `ttbar_pu0_particles` parquet shards.

**Pipeline:**
1. `data/colliderml.py` (`ColliderMLEvents`) reads the parquet shards and converts Cartesian → cylindrical coordinates per event.
2. `scripts/train_embedding.py` trains a hit metric-learning embedding on the raw hits.
3. `scripts/build_graphs.py` uses the embedding to build a k-NN graph per event and saves each as a `.pyg` file (`data/graphs/train_set/`, `val_set/`, `test_set/`).
4. `data/dataset.py` (`GraphDataset`) loads those `.pyg` files at training time, computes all 6 edge features on-the-fly from raw node coordinates, and applies per-edge loss weights.

**`.pyg` file schema:** serialized `torch_geometric.data.Data` with `x` (node features `[r, φ, z, η]`), `edge_index`, `y` (edge labels), `particle_id`, `pt`.

Set `input_dir` in `configs/default.yaml` to the directory containing `train_set/`, `val_set/`, `test_set/` subdirectories.
