#!/bin/bash
#SBATCH -A adeiana_smu_atlas_research_0001
#SBATCH -J gnn-classical
#SBATCH -t 900
#SBATCH -c 8
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -o run/logs/classical_%j.out
#SBATCH -e run/logs/classical_%j.err

# Acorn-style classical GNN baseline.
# Architecture changes vs old version:
#   - undirected_message_passing=True: doubles edges, averages scores both ways
#   - output_edge_classifier(cat[x_src, x_dst, e]) replaces edge_decoder+linear
#   - MLP norm placement fixed: Linear→LayerNorm→ReLU (pre-norm, Acorn style)
# These are all defaults in configs/default.yaml — no overrides needed.
# NOTE: old checkpoints are incompatible (architecture changed).

module load spack conda gcc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qgnn

echo "Job:   $SLURM_JOB_ID"
echo "Node:  $SLURMD_NODENAME"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

python -u scripts/train.py \
    --config     configs/default.yaml \
    --model_type classical \
    --stage_dir  outputs/classical

echo ""
echo "Done: $(date)"
