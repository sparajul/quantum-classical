#!/bin/bash
#SBATCH -A adeiana_smu_atlas_research_0001
#SBATCH -J eval
#SBATCH -t 60
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=16G
#SBATCH -o run/logs/eval_%j.out
#SBATCH -e run/logs/eval_%j.err

# Comparison plots for all three models: classical vs edge-quantum vs full-quantum.
# Requires predictions_test.pt from 06_infer_classical, 07_infer_edge_quantum,
# and 08_infer_quantum to exist first.

module load spack conda gcc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qgnn

echo "Job:   $SLURM_JOB_ID"
echo "Node:  $SLURMD_NODENAME"
echo "Start: $(date)"
echo ""

for PRED in \
    results/classical/predictions_test.pt \
    results/edge_quantum/predictions_test.pt \
    results/quantum/predictions_test.pt
do
    [[ ! -f "$PRED" ]] && echo "WARNING: $PRED not found — will be skipped in plots"
done

python -u scripts/evaluate.py \
    --predictions \
        results/classical/predictions_test.pt:Classical \
        results/edge_quantum/predictions_test.pt:Edge-Quantum \
        results/quantum/predictions_test.pt:Quantum \
    --output_dir  plots/edge_classification/ \
    --edge_cut    0.5

echo ""
echo "Plots saved to plots/edge_classification/"
echo "Done: $(date)"
