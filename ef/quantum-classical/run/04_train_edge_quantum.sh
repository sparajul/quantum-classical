#!/bin/bash
#SBATCH -A adeiana_smu_atlas_research_0001
#SBATCH -J gnn-edgeq
#SBATCH -t 2880
#SBATCH -c 32
#SBATCH -G 1
#SBATCH --mem=300G
#SBATCH -o run/logs/edge_quantum_%j.out
#SBATCH -e run/logs/edge_quantum_%j.err

# Edge-quantum GNN — quantum edge_network only; all other blocks are classical.
# n_qlayers=2 + data_reuploading adds Fourier frequency modes (Pérez-Salinas 2020).
# ring_entanglement closes the CNOT ladder for full qubit connectivity.

module load spack conda gcc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qgnn

echo "Job:   $SLURM_JOB_ID"
echo "Node:  $SLURMD_NODENAME"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

python -u scripts/train.py \
    --config              configs/default.yaml \
    --model_type          edge_quantum \
    --n_qubits            22 \
    --n_qlayers           2 \
    --data_reuploading    true \
    --ring_entanglement   true \
    --stage_dir           outputs/edge_quantum_22qb_2l \
#    --resume              outputs/edge_quantum_22qb_2l/checkpoints/run_<OLD_JOB_ID>/last.ckpt

echo ""
echo "Done: $(date)"
