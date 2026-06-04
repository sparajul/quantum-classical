# Quantum-Classical GNN for Particle Track Reconstruction

Quantum-classical hybrid Interaction Network GNN for edge classification at the HL-LHC.
Given a graph of ATLAS detector hits, the model
classifies each edge as a real particle track (signal) or noise (background).

```bash
conda env create -f environment.yml
conda activate qgnn
bash install.sh
```

---

## Models

| Model | Script | What's quantum |
|-------|--------|----------------|
| `classical` | `03_train_classical.sh` | nothing — plain MLP baseline | 
| `edge_quantum` | `04_train_edge_quantum.sh` | edge_network block only (message-passing update) |
| `quantum` | `05_train_quantum.sh` | all five MLP blocks |

**classical** — every block (node encoder, edge encoder, edge network, node network, output classifier) is a standard `Linear → BatchNorm → ReLU` stack. Baseline for comparison.

**edge_quantum** — only the `edge_network` (the block that updates edge embeddings each GNN iteration) is replaced with a VQC; everything else stays classical. The edge network is the most compute-heavy block and most directly responsible for edge classification, so it is the natural candidate for a quantum swap.

**quantum** — all five MLP blocks replaced with VQCs. Maximum quantum involvement, slowest to simulate.

### Classical vs full quantum only

To skip `edge_quantum` and compare just classical vs quantum:

```bash
sbatch run/03_train_classical.sh
sbatch run/05_train_quantum.sh

sbatch run/06_infer_classical.sh   # after 03 finishes
sbatch run/08_infer_quantum.sh     # after 05 finishes

python scripts/evaluate.py \
    --predictions \
        results/classical/predictions_test.pt:Classical \
        results/quantum/predictions_test.pt:Quantum \
    --output_dir plots/comparison/ \
    --edge_cut   0.7
```

---

## Download Sample Data

**Option A — Pre-built graphs (fastest):** download the ready-to-use `.pyg` graph files
directly from CERNBox and extract them into `data/graphs/`:

```bash
# Download and unpack into data/graphs/ (contains train_set/, val_set/, test_set/)
wget -O graphs.tar.gz "https://cernbox.cern.ch/s/iRf2h4iGaeiZdDI"
mkdir -p data/graphs && tar -xzf graphs.tar.gz -C data/graphs/
```

Then skip straight to [Stage 3](#stage-by-stage-job-submission) — no embedding training or graph building needed.

**Option B — Build from raw hits:**

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

Set `input_dir` in `configs/default.yaml` to `"data/graphs"` (will be created by Stage 2).

---

## Stage-by-Stage Job Submission

Run from the **project root**. Each stage depends on the previous one completing.

### Individual jobs

```bash
# Stage 1 — train hit embedding 
sbatch run/01_train_embedding.sh

# Stage 2 — build graphs — run after Stage 1
sbatch run/02_build_graphs.sh

# Stage 3 — train GNN — run after Stage 2 (all three can run in parallel)
sbatch run/03_train_classical.sh        # classical baseline  → outputs/classical_16/
sbatch run/04_train_edge_quantum.sh     # quantum edge_network only  → outputs/edge_quantum_22qb_2l/
sbatch run/05_train_quantum.sh          # full quantum (all blocks)  → outputs/quantum_20qb_2l/

# Inference — run after the corresponding training job
sbatch run/06_infer_classical.sh        # → results/classical/
sbatch run/07_infer_edge_quantum.sh     # → results/edge_quantum/
sbatch run/08_infer_quantum.sh          # → results/quantum/
```

Monitor logs:
```bash
squeue -u $USER
tail -f run/logs/classical_<jobid>.out
```

---

## Plotting

### Graph construction quality (no checkpoint needed)

Shows embedding-based graph efficiency and purity vs particle pT and η.

```bash
python scripts/plot_graph_construction.py \
    --input-dir   data/graphs/test_set \
    --output-dir  plots/ \
    --results-dir results/ \
    --split       test
```

Output: `plots/graph_{efficiency,purity,combined}_{pt,eta}_test.png`  
Summary: `results/graph_construction_metrics_test.json`

Target: `global_efficiency ≥ 0.95` before training the GNN.

### GNN edge classification (per model)

Shows GNN efficiency and purity vs pT and η at the chosen score threshold.

```bash
# Classical
python scripts/plot_efficiency_purity.py \
    --config     configs/default.yaml \
    --checkpoint outputs/classical_16/checkpoints/run_<ID>/best-f1-*.ckpt \
    --split      test \
    --output_dir plots/classical/
    --edge_cut   0.7

# Edge-quantum
python scripts/plot_efficiency_purity.py \
    --config     configs/default.yaml \
    --checkpoint outputs/edge_quantum_22qb_2l/checkpoints/run_<ID>/best-f1-*.ckpt \
    --split      test \
    --output_dir plots/edge_quantum/
    --edge_cut   0.7

# Full-quantum
python scripts/plot_efficiency_purity.py \
    --config     configs/default.yaml \
    --checkpoint outputs/quantum_20qb_2l/checkpoints/run_<ID>/best-f1-*.ckpt \
    --split      test \
    --output_dir plots/quantum/
    --edge_cut   0.7
```

### Multi-model comparison (after all three inference runs)

ROC curves, PR curves, and efficiency/purity overlaid for all three models.

```bash
sbatch run/09_evaluate.sh
# or directly:
python scripts/evaluate.py \
    --predictions \
        results/classical/predictions_test.pt:Classical \
        results/edge_quantum/predictions_test.pt:Edge-Quantum \
        results/quantum/predictions_test.pt:Quantum \
    --output_dir plots/edge_classification/ \
    --edge_cut   0.7
```

Output: `plots/edge_classification/`
