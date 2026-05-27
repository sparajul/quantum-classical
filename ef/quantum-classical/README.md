# Quantum-Classical GNN for Particle Track Reconstruction

[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b?logo=arxiv&logoColor=white)](#)
[![CTD 2025](https://img.shields.io/badge/CTD%202025-Tokyo-blue)](https://indico.cern.ch)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.35%2B-00B2FF)](https://pennylane.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Santosh Parajuli** — Postdoctoral Research Associate, UIUC · ATLAS Collaboration  
[`santosh.prj@gmail.com`](mailto:santosh.prj@gmail.com) | [LinkedIn](https://www.linkedin.com/in/sparajul/) | [INSPIRE-HEP](https://inspirehep.net/authors/1627735)

---

Quantum-classical hybrid Interaction Network GNN for real-time edge classification at the HL-LHC. Given a graph of ATLAS detector hits (~10K nodes, ~40K edges per event), the model classifies each edge as a real particle track (signal, ~5%) or noise (background, ~95%). Variational quantum circuits (VQCs) replace classical MLP blocks inside the message-passing layers.

Architecture follows the [Acorn (GNN4ITk)](https://gitlab.cern.ch/gnn4itkteam/acorn/-/tree/CTD25) framework:
- Acorn-style pre-norm MLP: `Linear → LayerNorm → ReLU`
- Undirected message passing: edges doubled `[src→dst, dst→src]`, scores averaged
- Output classifier: `cat([x_src, x_dst, e]) → MLP → sigmoid`
- Embedding-based graph construction with Hard Negative Mining

---

## Table of Contents

1. [Setup](#setup)
2. [Data Pipeline](#data-pipeline)
3. [Running Locally](#running-locally)
4. [Running on SLURM](#running-on-slurm)
5. [Multi-seed Sweep (Publication)](#multi-seed-sweep-publication)
6. [Inference & Evaluation](#inference--evaluation)
7. [Plotting — Graph Construction Quality](#plotting--graph-construction-quality)
8. [Plotting — Edge Classification (GNN)](#plotting--edge-classification-gnn)
9. [Configuration Reference](#configuration-reference)
10. [Project Structure](#project-structure)
11. [IBM Quantum Hardware](#ibm-quantum-hardware)
12. [Metrics](#metrics)
13. [Citation](#citation)

---

## Setup

```bash
# 1. Create and activate the conda environment
conda env create -f environment.yml
conda activate qgnn

# 2. Install PyTorch + torch-geometric (CUDA auto-detected)
bash install.sh
```

`install.sh` detects your GPU and fetches the right CUDA wheels automatically.

**Optional — GPU quantum simulation:**
```bash
pip install pennylane-lightning[gpu]   # requires CUDA 11.8+
```

**Optional — IBM hardware access:**
```bash
python scripts/ibm_utils.py --save-token YOUR_IBM_TOKEN
```

---

## Data Pipeline

The pipeline has three sequential stages:

```
Raw hits/particles (OpenML parquet)
        │
        ▼
[Stage 1] Train metric-learning embedding (scripts/train_embedding.py)
        │  → outputs/embedding.pt
        ▼
[Stage 2] Build .pyg graphs via embedding radius search (scripts/build_graphs.py)
        │  → data/graphs/train_set/, val_set/, test_set/
        ▼
[Stage 3] Train GNN edge classifier (scripts/train.py)
```

### Download the data

```bash
pip install openml
python - <<'EOF'
import openml, shutil, pathlib
for did, name in [(45068, "ttbar_pu0_tracker_hits"), (45069, "ttbar_pu0_particles")]:
    ds = openml.datasets.get_dataset(did, download_data=True)
    src = pathlib.Path(ds.data_file)
    dst = pathlib.Path(f"data/openml/{name}/data/{name}")
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst / src.name)
    print(f"Saved {name} -> {dst}")
EOF
```

Set the paths in `configs/default.yaml`:
```yaml
input_dir: "data/graphs"    # directory that will contain train_set/, val_set/, test_set/
stage_dir: "outputs/"       # checkpoints and logs
```

---

## Running Locally

Run all commands from the **project root** (`quantum-classical/`).

### Stage 1 — Train the hit embedding

```bash
python -u scripts/train_embedding.py \
    --hits-dir      data/openml/ttbar_pu0_tracker_hits/data/ttbar_pu0_tracker_hits \
    --particles-dir data/openml/ttbar_pu0_particles/data/ttbar_pu0_particles \
    --output        outputs/embedding.pt \
    --max-events    800 \
    --val-events    100 \
    --min-pt        1.0 \
    --epochs        120 \
    --embed-dim     16 \
    --hidden        512 \
    --n-blocks      4 \
    --r-train       0.6 \
    --margin        0.6 \
    --k-hnm         50 \
    --n-random      2000 \
    --pos-weight    2.0 \
    --lr            3e-4 \
    --min-lr        1e-5 \
    --warmup-epochs 5 \
    --patience      30 \
    --grad-clip     1.0
```

### Stage 2 — Build graphs

```bash
python -u scripts/build_graphs.py \
    --hits-dir      data/openml/ttbar_pu0_tracker_hits/data/ttbar_pu0_tracker_hits \
    --particles-dir data/openml/ttbar_pu0_particles/data/ttbar_pu0_particles \
    --embedding     outputs/embedding.pt \
    --output-dir    data/graphs \
    --split         800 100 100 \
    --method        embedding \
    --r-infer       1.0 \
    --k-infer       500 \
    --min-pt        1.0
```

This writes `data/graphs/train_set/*.pyg`, `val_set/*.pyg`, `test_set/*.pyg`.

`--r-infer 1.0 > --r-train 0.6` gives a safety margin to capture boundary doublets.  
`--k-infer 500` limits neighbours per hit at high density.

### Stage 3a — Train the classical GNN

```bash
python -u scripts/train.py \
    --config configs/default.yaml \
    --model_type classical
```

### Stage 3b — Train the quantum GNN

```bash
python -u scripts/train.py \
    --config     configs/default.yaml \
    --model_type quantum \
    --n_qubits   4 \
    --n_qlayers  2
```

### Useful training overrides

```bash
# Resume from a checkpoint
python scripts/train.py --config configs/default.yaml --resume outputs/checkpoints/last.ckpt

# Override any hyperparameter
python scripts/train.py --config configs/default.yaml \
    --model_type classical \
    --hidden 64 \
    --lr 0.0002 \
    --n_graph_iters 6 \
    --max_epochs 50

# Enable WandB logging
python scripts/train.py --config configs/default.yaml \
    --wandb_project my_project \
    --wandb_tags "classical,test"
```

---

## Running on SLURM

All SLURM scripts live in `run/`. Submit from the **project root**.

### Option A — Submit the full pipeline automatically (recommended)

`run/submit_all.sh` chains all four jobs with SLURM dependencies so each stage
waits for the previous one to succeed:

```
01_train_embedding → 02_build_graphs → 03_train_classical
                                     → 04_train_quantum   (parallel)
```

```bash
# Full pipeline (all four steps)
bash run/submit_all.sh

# Skip embedding if outputs/embedding.pt already exists
bash run/submit_all.sh --skip-embedding

# Skip embedding + graph building if data/graphs/ already exists
bash run/submit_all.sh --skip-graphs

# Only classical GNN (no quantum job)
bash run/submit_all.sh --classical-only

# Only quantum GNN (no classical job)
bash run/submit_all.sh --quantum-only
```

Monitor jobs and logs:
```bash
squeue -u $USER
tail -f run/logs/embed_<jobid>.out
tail -f run/logs/classical_<jobid>.out
tail -f run/logs/quantum_<jobid>.out
```

### Option B — Submit individual jobs

```bash
# Stage 1: train embedding (~15 hours, GPU)
sbatch run/01_train_embedding.sh

# Stage 2: build graphs (~4 hours, GPU) — run after stage 1 completes
sbatch run/02_build_graphs.sh

# Stage 3: train classical GNN (~10 hours, GPU)
sbatch run/03_train_classical.sh

# Stage 3: train quantum GNN (~15 hours, GPU — VQC simulation is slower)
sbatch run/04_train_quantum.sh
```

### SLURM resource summary

| Script | Time | CPUs | GPU | RAM | Notes |
|--------|------|------|-----|-----|-------|
| `01_train_embedding.sh` | 900 min | 8 | 1 | 32G | HNM per step; watch `eff@r_train` |
| `02_build_graphs.sh` | 240 min | 8 | 1 | 32G | Radius search in embedding space |
| `03_train_classical.sh` | 900 min | 8 | 1 | 32G | Acorn-style classical GNN |
| `04_train_quantum.sh` | 1200 min | 8 | 1 | 64G | VQC simulation; auto-selects `lightning.gpu` |
| `05_sweep_classical.sh` | 900 min × 5 parallel | 8 | 1 | 32G | Multi-seed classical sweep (array job) |
| `06_sweep_quantum.sh` | 1200 min × 5 parallel | 8 | 1 | 64G | Multi-seed quantum sweep (array job) |

### Edit qubit count for the quantum job

Open `run/04_train_quantum.sh` and change `--n_qubits` / `--n_qlayers`:
```bash
python -u scripts/train.py \
    --config configs/default.yaml \
    --model_type quantum \
    --n_qubits 6 \    # ← change here
    --n_qlayers 3     # ← and here
```

---

## Multi-seed Sweep (Publication)

Journal submission requires reporting **mean ± std** across multiple independent
runs. Each run uses a different random seed, which changes weight initialisation
and data-shuffle order. Five seeds is the standard minimum.

### Why this matters

A single training run cannot distinguish a good model from a lucky initialisation.
Reporting `AUC = 0.971 ± 0.003` (5 seeds) is publishable; `AUC = 0.971` (1 seed)
will be rejected by reviewers.

### Running the sweeps on SLURM (recommended)

Each seed is submitted as a separate SLURM array job so all 5 run **in parallel**
— wall time equals one run, not five.

```bash
# Step 1 — classical sweep (5 seeds in parallel)
sbatch run/05_sweep_classical.sh

# Step 2 — quantum sweep (5 seeds in parallel, n_qubits=4, n_qlayers=2)
sbatch run/06_sweep_quantum.sh
```

Monitor progress:
```bash
squeue -u $USER
# logs land in run/logs/classical_sweep_<arrayID>_<taskID>.out
```

### Aggregating results after all jobs finish

```bash
# Classical summary → results/classical/sweep_summary.json
python scripts/train_sweep.py \
    --config     configs/default.yaml \
    --model_type classical \
    --output_dir results/classical/ \
    --summarise_only

# Quantum summary → results/quantum/sweep_summary.json
python scripts/train_sweep.py \
    --config     configs/default.yaml \
    --model_type quantum \
    --output_dir results/quantum/ \
    --summarise_only
```

### Generating the results table

```bash
python scripts/make_results_table.py \
    --classical results/classical/sweep_summary.json \
    --quantum   results/quantum/sweep_summary.json \
    --output_dir results/
```

Outputs:
- `results/table.tex` — LaTeX table for the paper (mean ± std per metric)
- `results/table.md` — same table in Markdown

### Output structure

```
results/
├── classical/
│   ├── seed_42/train.log        # full training log per seed
│   ├── seed_123/train.log
│   ├── ...
│   ├── sweep_runs.csv           # per-seed metrics (AUC, eff, purity, ...)
│   └── sweep_summary.json       # mean ± std across all seeds
├── quantum/
│   └── (same structure)
├── table.tex                    # publication-ready LaTeX table
└── table.md
```

### Running sweeps locally (sequential, no SLURM)

```bash
python scripts/train_sweep.py \
    --config     configs/default.yaml \
    --model_type classical \
    --seeds      42 123 456 789 1337 \
    --output_dir results/classical/
```

This runs each seed one after another in the same process. Use this for testing
locally or when a cluster is not available. The `--summarise_only` flag is not
needed here — summary is produced automatically at the end.

### Full publication pipeline

`run/run_publication.sh` chains all steps: classical sweep → quantum sweep →
barren plateau analysis → scaling study → plots → LaTeX table:

```bash
bash run/run_publication.sh
```

Edit `CKPT_CLASSICAL` and `CKPT_QUANTUM` at the top of the script after the
sweeps complete.

---

## Inference & Evaluation

Run inference on a saved checkpoint to get per-graph predictions and aggregate metrics:

```bash
# Test split (saves predictions_test.pt + metrics_test.yaml)
python scripts/inference.py \
    --config      configs/default.yaml \
    --checkpoint  outputs/checkpoints/best-f1.ckpt \
    --split       test \
    --output_dir  results/

# Quick validation check (no file output)
python scripts/inference.py \
    --config     configs/default.yaml \
    --checkpoint outputs/checkpoints/best-f1.ckpt \
    --split      val
```

**Output files** in `results/`:
- `predictions_test.pt` — list of per-graph dicts: `{scores, y, loss, acc, auc}`
- `metrics_test.yaml` — aggregated efficiency, purity, AUC, avg_precision over the split

---

## Plotting — Graph Construction Quality

These plots show how well the embedding-based graph builder captures true particle
doublets **before** the GNN runs. No checkpoint needed — plots come directly from
the `.pyg` graph files.

### What is produced

| File | Description |
|------|-------------|
| `graph_efficiency_pt_<split>.png` | Efficiency vs particle pT |
| `graph_efficiency_eta_<split>.png` | Efficiency vs particle η |
| `graph_purity_pt_<split>.png` | Purity (signal fraction) vs edge pT |
| `graph_purity_eta_<split>.png` | Purity vs edge η |
| `graph_combined_pt_<split>.png` | Efficiency + purity overlaid vs pT |
| `graph_combined_eta_<split>.png` | Efficiency + purity overlaid vs η |
| `results/graph_construction_metrics_<split>.json` | Global numbers (efficiency, purity, mean graph size) |

**Efficiency** = fraction of consecutive same-particle hit-pairs captured as edges  
**Purity** = fraction of graph edges that are true track doublets (y == 1)

### Commands

```bash
# Plot test set — default for make graph-quality (final quality number for paper)
python scripts/plot_graph_construction.py \
    --input-dir  data/graphs/test_set \
    --output-dir plots/ \
    --results-dir results/ \
    --split      test

# Or use make (shortcut — always runs on test_set)
make graph-quality

# Plot train/val set for quick sanity checks during development
python scripts/plot_graph_construction.py \
    --input-dir  data/graphs/train_set \
    --output-dir plots/ \
    --results-dir results/ \
    --split      train

# Limit to first N events (fast check)
python scripts/plot_graph_construction.py \
    --input-dir  data/graphs/test_set \
    --output-dir plots/ \
    --split      test \
    --max-events 50
```

**All options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | `data/graphs/train_set` | Directory of `.pyg` files |
| `--output-dir` | `plots/` | Where to write PNG files |
| `--results-dir` | `results/` | Where to write JSON summary |
| `--split` | `train` | Label for file names and plot titles |
| `--min-hits` | `2` | Minimum hits per particle to count as findable |
| `--n-bins` | `20` | Number of bins for pT and η histograms |
| `--max-events` | all | Limit to first N graphs |

### Interpret the output

```
====================================================
  GRAPH CONSTRUCTION QUALITY
====================================================
  split                     test
  n_graphs                  100
  mean_nodes                9847
  mean_edges                41320
  global_efficiency         0.9612      ← target ≥ 0.95
  global_purity             0.0821      ← ~5–15% is expected (GNN filters later)
====================================================
```

Low efficiency (< 0.90) means the embedding did not converge — retrain `01_train_embedding.sh`
and watch `eff@r_train` logs until it exceeds 0.95.

---

## Plotting — Edge Classification (GNN)

These plots show how well the **trained GNN** classifies edges, as a function of
particle pT and η. Requires a trained checkpoint.

### What is produced

| File | Description |
|------|-------------|
| `efficiency_purity_eta.png` | GNN efficiency + purity vs edge η |
| `efficiency_purity_pt.png` | GNN efficiency + purity vs edge pT |

**Efficiency** = TP / (TP + FN) at the chosen `edge_cut` threshold  
**Purity** = TP / (TP + FP) at the chosen `edge_cut` threshold

### Commands

```bash
# Classical model — test split
python scripts/plot_efficiency_purity.py \
    --config     configs/default.yaml \
    --checkpoint outputs/checkpoints/best-f1.ckpt \
    --split      test \
    --edge_cut   0.3 \
    --output_dir plots/classical/

# Quantum model — test split
python scripts/plot_efficiency_purity.py \
    --config     configs/default.yaml \
    --checkpoint outputs/checkpoints/best-quantum.ckpt \
    --split      test \
    --edge_cut   0.3 \
    --output_dir plots/quantum/

# Validation split (fast, during training)
python scripts/plot_efficiency_purity.py \
    --config     configs/default.yaml \
    --checkpoint outputs/checkpoints/last.ckpt \
    --split      val \
    --edge_cut   0.3 \
    --output_dir plots/

# Log-spaced pT bins (better for wide pT range)
python scripts/plot_efficiency_purity.py \
    --config     configs/default.yaml \
    --checkpoint outputs/checkpoints/best-f1.ckpt \
    --split      test \
    --edge_cut   0.3 \
    --pt_log \
    --pt_max     20.0 \
    --output_dir plots/
```

**All options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `configs/default.yaml` | YAML config file |
| `--checkpoint` | required | Path to `.ckpt` file |
| `--split` | `test` | `train`, `val`, or `test` |
| `--edge_cut` | `0.5` | Score threshold for binary classification (use `0.3` — matches config) |
| `--output_dir` | `plots/` | Where to write PNG files |
| `--n_eta_bins` | `20` | Number of η bins |
| `--n_pt_bins` | `20` | Number of pT bins |
| `--pt_max` | `10.0` | Upper edge of pT range (GeV) |
| `--pt_log` | off | Use log-spaced pT bins |

**Note:** always use `--edge_cut 0.3` to match the `edge_cut: 0.3` in `configs/default.yaml`.

### Full paper plots (classical + quantum, all metrics)

```bash
# Step 1 — run inference to save predictions
python scripts/inference.py \
    --config     configs/default.yaml \
    --checkpoint results/classical/seed_42/checkpoints/best-f1.ckpt \
    --split      test \
    --output_dir results/classical/

python scripts/inference.py \
    --config     configs/default.yaml \
    --checkpoint results/quantum/seed_42/checkpoints/best-f1.ckpt \
    --split      test \
    --output_dir results/quantum/

# Step 2 — efficiency/purity vs eta and pt
python scripts/plot_efficiency_purity.py \
    --config     configs/default.yaml \
    --checkpoint results/classical/seed_42/checkpoints/best-f1.ckpt \
    --split test --edge_cut 0.3 \
    --output_dir plots/classical/

python scripts/plot_efficiency_purity.py \
    --config     configs/default.yaml \
    --checkpoint results/quantum/seed_42/checkpoints/best-f1.ckpt \
    --split test --edge_cut 0.3 \
    --output_dir plots/quantum/

# Step 3 — graph construction quality (no checkpoint needed)
python scripts/plot_graph_construction.py \
    --input-dir data/graphs/test_set \
    --output-dir plots/graphs/ \
    --results-dir results/ \
    --split test
```

---

## Configuration Reference

All settings live in `configs/default.yaml`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | `"classical"` | `"classical"` or `"quantum"` |
| `n_qubits` | `8` | Qubits per VQC block (quantum only) |
| `n_qlayers` | `1` | Variational layers per VQC (quantum only) |
| `hidden` | `32` | GNN hidden dimension |
| `n_graph_iters` | `8` | Message-passing iterations |
| `undirected_message_passing` | `true` | Double edges + average scores (Acorn style) |
| `layernorm` | `true` | Pre-norm in MLP: `Linear → LayerNorm → ReLU` |
| `lr` | `5e-4` | Peak learning rate |
| `min_lr` | `5e-5` | Cosine annealing floor |
| `warmup` | `5` | Linear warmup epochs |
| `patience` | `15` | Early stopping patience (on `val/f1_epoch`) |
| `max_epochs` | `100` | Maximum training epochs |
| `pos_weight` | `30` | BCE upweight for class imbalance (~5% signal) |
| `edge_cut` | `0.3` | Binary classification threshold |
| `gradient_clip_val` | `1.0` | Gradient clipping |
| `quantum_device` | `"auto"` | `lightning.gpu` → `lightning.qubit` → `default.qubit` |
| `input_dir` | `"data/graphs"` | Directory containing `train_set/`, `val_set/`, `test_set/` |
| `stage_dir` | `"outputs/"` | Output directory for checkpoints and logs |

`configs/ibm_hardware.yaml` overlays `default.yaml` with IBM-specific settings:
4 qubits, `quantum_device: qiskit.ibmq`. Note: `noise_mitigation: true` is accepted
but not yet implemented — ZNE is planned for a future release.

### Checkpoints saved during training

Two checkpoints are saved per run in `outputs/checkpoints/run_<SLURM_JOB_ID>/`:

| Filename pattern | Monitors |
|-----------------|---------|
| `best-f1-{epoch:03d}-{val/f1_epoch:.4f}.ckpt` | Best validation F1 (top 1 kept) |
| `last.ckpt` | Most recent epoch |

Each run gets its own sub-directory named after the SLURM job ID (e.g.
`run_449606/`), so parallel multi-seed runs never overwrite each other.
Pass `--run <name>` to override the directory name for local runs.

---

## Project Structure

```
quantum-classical/
├── configs/
│   ├── default.yaml          # main config — edit input_dir and stage_dir here
│   ├── ibm_hardware.yaml     # IBM hardware overrides (qiskit.ibmq, n_qubits=4, n_shots=1024)
│   └── scaling_sweep.yaml    # qubit/layer scaling grid for sweep study
│
├── models/
│   ├── gnn.py                # InteractionGNN — PyTorch Lightning module
│   ├── quantum_mlp.py        # VQC layer (PennyLane TorchLayer)
│   ├── classical_mlp.py      # Acorn-style pre-norm MLP (drop-in classical)
│   ├── embedding.py          # HitEmbedding + HNM training utilities
│   └── backend_factory.py    # routes make_quantum_mlp() to pennylane/qiskit
│
├── data/
│   ├── dataset.py            # GraphDataset — loads .pyg files, computes edge features
│   └── colliderml.py         # OpenML parquet → per-event cylindrical tensors
│
├── utils/
│   ├── metrics.py            # efficiency, purity, AUC, avg_precision
│   └── callbacks.py          # WandB logging, gradient monitoring, timing
│
├── scripts/
│   ├── train_embedding.py    # Stage 1: metric learning with HNM + hinge loss
│   ├── build_graphs.py       # Stage 2: embedding radius search → .pyg files
│   ├── train.py              # Stage 3: GNN training entry point
│   ├── inference.py          # evaluate checkpoint → predictions.pt + metrics.yaml
│   ├── plot_graph_construction.py   # graph eff/purity vs pT and eta (no checkpoint)
│   ├── plot_efficiency_purity.py    # GNN eff/purity vs pT and eta (needs checkpoint)
│   ├── train_sweep.py        # multi-seed sweep
│   ├── scaling_study.py      # qubit/layer scaling grid
│   ├── track_building.py     # connected-components track reconstruction
│   ├── make_results_table.py # LaTeX + Markdown results table
│   └── ibm_utils.py          # IBM backend listing and token management
│
├── run/
│   ├── submit_all.sh         # submit full pipeline with SLURM dependencies
│   ├── 01_train_embedding.sh # SLURM: Stage 1 (900 min, 1 GPU)
│   ├── 02_build_graphs.sh    # SLURM: Stage 2 (240 min, 1 GPU)
│   ├── 03_train_classical.sh # SLURM: Stage 3 classical (900 min, 1 GPU)
│   ├── 04_train_quantum.sh   # SLURM: Stage 3 quantum (1200 min, 1 GPU)
│   ├── 05_sweep_classical.sh # SLURM array: 5-seed classical sweep (parallel)
│   ├── 06_sweep_quantum.sh   # SLURM array: 5-seed quantum sweep (parallel)
│   ├── run_publication.sh    # full 7-step publication pipeline (local/sequential)
│   └── logs/                 # SLURM stdout/stderr (created on first submit)
│
├── environment.yml           # conda env (Python + most packages)
├── install.sh                # installs PyTorch + torch-geometric (CUDA-aware)
├── Makefile                  # convenience targets
└── requirements.txt          # raw pip requirements (reference)
```

---

## IBM Quantum Hardware

Train on simulation first, then run inference on real hardware.

```bash
# Step 1 — train on simulation (quantum, n_qubits=4 for hardware compatibility)
python scripts/train.py \
    --config     configs/default.yaml \
    --model_type quantum \
    --n_qubits   4 \
    --n_qlayers  1

# Step 2 — find the best available backend
python scripts/ibm_utils.py --recommend --n_qubits 4

# Step 3 — inference on IBM hardware (overlays default.yaml with IBM settings)
python scripts/inference.py \
    --config          configs/default.yaml \
    --config_override configs/ibm_hardware.yaml \
    --checkpoint      outputs/checkpoints/run_<ID>/best-f1-*.ckpt \
    --split           test \
    --output_dir      results/ibm/ \
    --ibm_backend     ibm_brisbane   # replace with your machine

# List all available IBM backends (filtered by qubit count)
python scripts/ibm_utils.py --list --min_qubits 4
```

`configs/ibm_hardware.yaml` sets:
- `n_qubits: 4`, `n_qlayers: 1`, `quantum_device: qiskit.ibmq`
- `n_shots: 1024` — measurement shots per circuit evaluation
- `noise_mitigation: true` — flag accepted but **not yet implemented** (ZNE planned)
- `ibm_backend: ibm_brisbane` — change to your target device

---

## Metrics

| Metric | Target | Definition |
|--------|--------|------------|
| Efficiency | ≥ 0.95 | TP / (TP + FN) — fraction of true track edges kept |
| Purity | ≥ 0.90 | TP / (TP + FP) — quality of kept edges |
| AUC | primary | Used for early stopping and best-checkpoint selection |
| Avg Precision | secondary | PR-curve AUC; preferred for imbalanced data |

Graph construction efficiency (Stage 2 target): ≥ 0.95 before training the GNN.

---

## Citation

```bibtex
@inproceedings{parajuli2025quantum,
  title     = {Quantum-Enhanced Graph Neural Networks for Particle Tracking},
  author    = {Parajuli, Santosh},
  booktitle = {Connecting The Dots 2025},
  address   = {Tokyo, Japan},
  year      = {2025},
  url       = {https://github.com/sparajul/quantum-gnn-tracking}
}
```

Related:

