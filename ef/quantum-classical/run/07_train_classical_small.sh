#!/bin/bash
#SBATCH -A adeiana_smu_atlas_research_0001
#SBATCH -J gnn-classical-small
#SBATCH -t 900
#SBATCH -c 8
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -o run/logs/classical_small_%j.out
#SBATCH -e run/logs/classical_small_%j.err

# Size-matched classical baseline (hidden=5, ~15K params).
# Provides a fair comparison against the quantum model (n_qubits=4 → ~14K params).
#
# Paper comparison table:
#   Classical (h=32) : ~126K params  ← full-capacity baseline
#   Classical (h=5)  : ~15K  params  ← this job (size-matched)
#   Quantum   (4q/1L): ~14K  params  ← run/04_train_quantum.sh

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
    --hidden     8 \
    --stage_dir  outputs/classical_small

echo ""
echo "Done: $(date)"
