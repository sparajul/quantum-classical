#!/bin/bash
#SBATCH -A adeiana_smu_atlas_research_0001
#SBATCH -J gnn-classical
#SBATCH -t 900
#SBATCH -c 8
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -o run/logs/classical_%j.out
#SBATCH -e run/logs/classical_%j.err

# Classical GNN baseline — no explicit normalisation, full hidden dim.

module load spack conda gcc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qgnn

echo "Job:   $SLURM_JOB_ID"
echo "Node:  $SLURMD_NODENAME"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

python -u scripts/train.py \
    --config      configs/default.yaml \
    --model_type  classical \
    --stage_dir   outputs/classical_16

echo ""
echo "Done: $(date)"
