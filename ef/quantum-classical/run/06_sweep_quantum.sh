#!/bin/bash
#SBATCH -A adeiana_smu_atlas_research_0001
#SBATCH -J gnn-quantum-sweep
#SBATCH -t 1200
#SBATCH -c 8
#SBATCH -G 1
#SBATCH --mem=64G
#SBATCH --array=0-4
#SBATCH -o run/logs/quantum_sweep_%A_%a.out
#SBATCH -e run/logs/quantum_%A_%a.err

# Multi-seed quantum sweep — 5 jobs run in parallel (one per seed).
# Edit N_QUBITS / N_QLAYERS to match your target quantum config.
#
# Usage:
#   sbatch run/06_sweep_quantum.sh
#
# After all 5 jobs finish, aggregate with:
#   python scripts/train_sweep.py \
#       --config configs/default.yaml --model_type quantum \
#       --output_dir results/quantum/ --summarise_only

SEEDS=(42 123 456 789 1337)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUTPUT_DIR="results/quantum"
N_QUBITS=4
N_QLAYERS=2

module load spack conda gcc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qgnn

echo "Job:      $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "Node:     $SLURMD_NODENAME"
echo "Seed:     $SEED"
echo "Qubits:   $N_QUBITS  Layers: $N_QLAYERS"
echo "Start:    $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

mkdir -p "${OUTPUT_DIR}/seed_${SEED}"

python -u scripts/train.py \
    --config configs/default.yaml \
    --model_type quantum \
    --n_qubits "$N_QUBITS" \
    --n_qlayers "$N_QLAYERS" \
    --seed "$SEED" \
    --run "qnn_${SLURM_JOB_ID}_s${SEED}" \
    --no_wandb \
    --stage_dir "${OUTPUT_DIR}/seed_${SEED}" \
    > "${OUTPUT_DIR}/seed_${SEED}/train.log" 2>&1

echo ""
echo "Done: $(date)"
