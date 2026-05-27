CONFIG      ?= configs/default.yaml
CKPT        ?= outputs/checkpoints/best-auc.ckpt
SPLIT       ?= test
HITS_DIR    ?= data/openml/ttbar_pu0_tracker_hits/data/ttbar_pu0_tracker_hits
PARTICLES   ?= data/openml/ttbar_pu0_particles/data/ttbar_pu0_particles
EMBEDDING   ?= outputs/embedding.pt
GRAPHS_DIR  ?= data/graphs

.PHONY: setup build-embedding build-graphs train train-quantum eval plots tracks graph-quality sweep ibm list-backends help

help:
	@echo "Usage: make <target> [VAR=value ...]"
	@echo ""
	@echo "  setup            Create conda env and install all dependencies"
	@echo "  build-embedding  Train hit embedding from OpenML parquet data"
	@echo "  build-graphs     Build .pyg graph files from OpenML parquet data"
	@echo "  train            Train classical GNN"
	@echo "  train-quantum    Train quantum GNN (4 qubits, 2 layers)"
	@echo "  eval             Evaluate a checkpoint  (CKPT=... SPLIT=...)"
	@echo "  graph-quality    Graph construction efficiency & purity vs eta/pt"
	@echo "  plots            Edge efficiency & purity vs eta/pt  (CKPT=...)"
	@echo "  tracks           Track-level efficiency, fake rate, purity  (CKPT=...)"
	@echo "  sweep            Run scaling sweep"
	@echo "  ibm              IBM hardware inference (CKPT=...)"
	@echo "  list-backends    Show available IBM backends"

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	conda env create -f environment.yml
	conda run -n qgnn bash install.sh

# ── Data preparation ─────────────────────────────────────────────────────────
build-embedding:
	python scripts/train_embedding.py \
	    --hits-dir $(HITS_DIR) \
	    --particles-dir $(PARTICLES) \
	    --output $(EMBEDDING) \
	    --epochs 60 --embed-dim 16 --hidden 256 --n-blocks 4 \
	    --batch-size 1024 --pairs-per-event 2000

build-graphs:
	python scripts/build_graphs.py \
	    --hits-dir $(HITS_DIR) \
	    --particles-dir $(PARTICLES) \
	    --embedding $(EMBEDDING) \
	    --output-dir $(GRAPHS_DIR) \
	    --method geometric \
	    --phi-slope-cut 1.0 \
	    --z-slope-cut 100.0 \
	    --sim-threshold 0.0

# ── Training ──────────────────────────────────────────────────────────────────
train:
	python scripts/train.py --config $(CONFIG)

train-quantum:
	python scripts/train.py --config $(CONFIG) model_type=quantum n_qubits=4 n_qlayers=2

# ── Evaluation ────────────────────────────────────────────────────────────────
eval:
	python scripts/inference.py --config $(CONFIG) --checkpoint $(CKPT) --split $(SPLIT)

# ── Graph construction quality (no checkpoint needed) ─────────────────────────
graph-quality:
	python scripts/plot_graph_construction.py \
	    --input-dir $(GRAPHS_DIR)/test_set \
	    --output-dir plots/ \
	    --results-dir results/ \
	    --split test

# ── Plots ─────────────────────────────────────────────────────────────────────
plots:
	python scripts/plot_efficiency_purity.py \
	    --config $(CONFIG) \
	    --checkpoint $(CKPT) \
	    --split $(SPLIT) \
	    --output_dir plots/ \
	    --pt_log

tracks:
	python scripts/track_building.py \
	    --config $(CONFIG) \
	    --checkpoint $(CKPT) \
	    --split $(SPLIT) \
	    --output_dir results/

# ── Sweeps ────────────────────────────────────────────────────────────────────
sweep:
	python scripts/train_sweep.py --config configs/scaling_sweep.yaml

# ── IBM Quantum ───────────────────────────────────────────────────────────────
ibm:
	python scripts/inference.py --config configs/ibm_hardware.yaml --checkpoint $(CKPT)

list-backends:
	python scripts/ibm_utils.py --list-backends --n-qubits 4
