#!/bin/bash
#SBATCH -A adeiana_smu_atlas_research_0001
#SBATCH -J gnn-quantum
#SBATCH -t 1200
#SBATCH -c 8
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -o run/logs/quantum_%j.out
#SBATCH -e run/logs/quantum_%j.err

# Acorn-style quantum GNN baseline.
# Architecture changes vs old version (same as classical):
#   - undirected_message_passing=True: doubles edges, averages scores both ways
#   - output_edge_classifier(cat[x_src, x_dst, e]) replaces edge_decoder+linear
#   - MLP norm placement fixed: Linear→LayerNorm→ReLU (pre-norm, Acorn style)
# Quantum: VQC replaces EdgeNetwork/NodeNetwork MLPs.
# Longer time limit: quantum simulation is ~5-10x slower than classical.
# NOTE: old checkpoints are incompatible (architecture changed).

module load spack conda gcc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qgnn

echo "Job:   $SLURM_JOB_ID"
echo "Node:  $SLURMD_NODENAME"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# Edit --n_qubits / --n_qlayers as needed
python -u scripts/train.py \
    --config     configs/default.yaml \
    --model_type quantum \
    --n_qubits   4 \
    --n_qlayers  1 \
    --stage_dir  outputs/quantum

echo ""
echo "Done: $(date)"
