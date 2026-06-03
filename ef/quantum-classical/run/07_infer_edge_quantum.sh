#!/bin/bash
#SBATCH -A adeiana_smu_atlas_research_0001
#SBATCH -J infer-edgeq
#SBATCH -t 480
#SBATCH -c 16
#SBATCH -G 1
#SBATCH --mem=64G
#SBATCH -o run/logs/infer_edge_quantum_%j.out
#SBATCH -e run/logs/infer_edge_quantum_%j.err

# Inference for the edge-quantum GNN.
# Quantum edge_network only; all other blocks are classical.
# Auto-discovers the best checkpoint under outputs/edge_quantum_22qb_2l/.

module load spack conda gcc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qgnn

echo "Job:   $SLURM_JOB_ID"
echo "Node:  $SLURMD_NODENAME"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

STAGE_DIR=outputs/edge_quantum_22qb_2l
CKPT=$(find "${STAGE_DIR}/checkpoints" -name "*.ckpt" ! -name "last.ckpt" 2>/dev/null | sort | tail -1)

if [[ -z "$CKPT" ]]; then
    echo "ERROR: no best checkpoint found under ${STAGE_DIR}/checkpoints/"
    echo "       Run 04_train_edge_quantum.sh first, or pass the path manually."
    exit 1
fi
echo "Checkpoint: $CKPT"

python -u scripts/inference.py \
    --config      configs/default.yaml \
    --checkpoint  "$CKPT" \
    --split       test \
    --output_dir  results/edge_quantum/

echo ""
echo "Done: $(date)"
