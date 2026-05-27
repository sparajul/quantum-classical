#!/bin/bash
#SBATCH -A adeiana_smu_atlas_research_0001
#SBATCH -J graphs
#SBATCH -t 240
#SBATCH -c 8
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -o run/logs/graphs_%j.out
#SBATCH -e run/logs/graphs_%j.err

# Acorn-style graph construction: pure radius search in embedding space.
# For each hit, find all other hits within L2 distance r_infer.
# Handles barrel/endcap transitions that geometric layer adjacency misses.
#
# r_infer = 1.0 > r_train = 0.6 (safety margin for boundary doublets).
# k_infer = 500: max neighbours per hit — set large to avoid missing hits
#   at high local density. Purity will be low (~5-15%) but the GNN handles it.
# --min-pt 1.0 = 1 GeV (OpenML px/py are in GeV; matches signal: pt: [1.0, .inf])

module load spack conda gcc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qgnn

echo "Job:   $SLURM_JOB_ID"
echo "Node:  $SLURMD_NODENAME"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

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

echo ""
echo "Done: $(date)"
