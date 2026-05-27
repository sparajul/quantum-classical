#!/bin/bash
#SBATCH -A adeiana_smu_atlas_research_0001
#SBATCH -J gnn-classical-sweep
#SBATCH -t 900
#SBATCH -c 8
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH --array=0-4
#SBATCH -o run/logs/classical_sweep_%A_%a.out
#SBATCH -e run/logs/classical_sweep_%A_%a.err

# Multi-seed classical sweep — 5 jobs run in parallel (one per seed).
#
# Usage:
#   sbatch run/05_sweep_classical.sh
#
# After all 5 jobs finish, aggregate with:
#   python scripts/train_sweep.py \
#       --config configs/default.yaml --model_type classical \
#       --output_dir results/classical/ --summarise_only
#
# Or submit the aggregation as a dependent job automatically:
#   SWEEP_ID=$(sbatch --parsable run/05_sweep_classical.sh)
#   sbatch --dependency=afterok:$SWEEP_ID run/05_sweep_classical_agg.sh

SEEDS=(42 123 456 789 1337)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUTPUT_DIR="results/classical"

module load spack conda gcc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qgnn

echo "Job:    $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "Node:   $SLURMD_NODENAME"
echo "Seed:   $SEED"
echo "Start:  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

mkdir -p "${OUTPUT_DIR}/seed_${SEED}"

python -u scripts/train.py \
    --config configs/default.yaml \
    --model_type classical \
    --seed "$SEED" \
    --run "cls_${SLURM_JOB_ID}_s${SEED}" \
    --no_wandb \
    --stage_dir "${OUTPUT_DIR}/seed_${SEED}" \
    > "${OUTPUT_DIR}/seed_${SEED}/train.log" 2>&1

echo ""
echo "Done: $(date)"
