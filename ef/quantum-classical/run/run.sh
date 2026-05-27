#!/bin/bash
#SBATCH -A adeiana_smu_atlas_research_0001
#SBATCH -J QGNN
#SBATCH -t 600
#SBATCH -c 8
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err

set -euo pipefail

# Activate conda env
module load spack conda gcc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qgnn

echo "Job:   $SLURM_JOB_ID"
echo "Node:  $SLURMD_NODENAME"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU info)"
echo ""

# ── Training ──────────────────────────────────────────────────────────────────
# Default: classical model (model_type: "classical" in configs/default.yaml)
# For quantum: add --model_type quantum --n_qubits 4 --n_qlayers 2
srun python -u scripts/train.py \
    --config configs/default.yaml \
    --wandb_project ICHEP_2026 \
    --wandb_tags "classical,atlas,hl-lhc,ichep2026"

# Useful overrides (add any of these to the command above):
#   --model_type quantum      run quantum model with VQC layers
#   --n_qubits 6              change qubit count
#   --n_qlayers 3             change VQC depth
#   --hidden 32               larger hidden dimension
#   --lr 0.0002               learning rate
#   --resume checkpoints/last.ckpt   resume from checkpoint

echo ""
echo "Training done."
echo "End: $(date)"
