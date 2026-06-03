#!/bin/bash
#SBATCH -A adeiana_smu_atlas_research_0001
#SBATCH -J embed
#SBATCH -t 900
#SBATCH -c 8
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -o run/logs/embed_%j.out
#SBATCH -e run/logs/embed_%j.err

# Acorn-style metric learning: Hard Negative Mining + hinge loss.
# Per-step: embed all hits → HNM radius search → signal doublets → hinge loss.
# Watch eff@r_train in logs — target >0.95 before building graphs.
# --min-pt 1.0 = 1 GeV (OpenML px/py are in GeV; matches signal: pt: [1.0, .inf])

module load spack conda gcc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qgnn

echo "Job:   $SLURM_JOB_ID"
echo "Node:  $SLURMD_NODENAME"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

python -u scripts/train_embedding.py \
    --hits-dir      data/openml/ttbar_pu0_tracker_hits/data/ttbar_pu0_tracker_hits \
    --particles-dir data/openml/ttbar_pu0_particles/data/ttbar_pu0_particles \
    --output        graphs_outputs/embedding.pt \
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
    --pos-weight    1.5 \
    --lr            3e-4 \
    --min-lr        1e-5 \
    --warmup-epochs 5 \
    --patience      30 \
    --grad-clip     1.0

echo ""
echo "Done: $(date)"
