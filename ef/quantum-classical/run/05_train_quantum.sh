#!/bin/bash
#SBATCH -A adeiana_smu_atlas_research_0001
#SBATCH -J gnn-quantum
#SBATCH -t 2880
#SBATCH -c 32
#SBATCH -G 1
#SBATCH --mem=300G
#SBATCH -o run/logs/quantum_%j.out
#SBATCH -e run/logs/quantum_%j.err

# Full quantum GNN — all MLP blocks use VQC.
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
    --model_type          quantum \
    --n_qubits            8 \
    --n_qlayers           2 \
    --data_reuploading    true \
    --ring_entanglement   true \
    --stage_dir           outputs/quantum_8qb_2l \
#    --resume              outputs/quantum_8qb_2l/checkpoints/run_<OLD_JOB_ID>/last.ckpt


echo ""
echo "Done: $(date)"
