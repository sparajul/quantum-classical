#!/bin/bash
# Submit the full pipeline with automatic job dependencies.
# Run from the project root:  bash run/submit_all.sh
#
# Dependency chain:
#   01_train_embedding → 02_build_graphs → 03_train_classical
#                                        → 04_train_quantum   (parallel)
#
# Options:
#   --skip-embedding   skip step 1 (reuse existing outputs/embedding.pt)
#   --skip-graphs      skip steps 1-2 (reuse existing data/graphs/)
#   --classical-only   skip quantum training
#   --quantum-only     skip classical training



SKIP_EMBEDDING=0
SKIP_GRAPHS=0
CLASSICAL=1
QUANTUM=1

for arg in "$@"; do
    case $arg in
        --skip-embedding) SKIP_EMBEDDING=1 ;;
        --skip-graphs)    SKIP_EMBEDDING=1; SKIP_GRAPHS=1 ;;
        --classical-only) QUANTUM=0 ;;
        --quantum-only)   CLASSICAL=0 ;;
    esac
done

# ── Step 1: Train embedding ───────────────────────────────────────────────────
if [[ $SKIP_EMBEDDING -eq 0 ]]; then
    JID1=$(sbatch --parsable run/01_train_embedding.sh)
    echo "Submitted 01_train_embedding  → job $JID1"
    DEP_GRAPHS="--dependency=afterok:${JID1}"
else
    echo "Skipping 01_train_embedding (--skip-embedding)"
    DEP_GRAPHS=""
fi

# ── Step 2: Build graphs ──────────────────────────────────────────────────────
if [[ $SKIP_GRAPHS -eq 0 ]]; then
    JID2=$(sbatch --parsable $DEP_GRAPHS run/02_build_graphs.sh)
    echo "Submitted 02_build_graphs     → job $JID2"
    DEP_TRAIN="--dependency=afterok:${JID2}"
else
    echo "Skipping 02_build_graphs (--skip-graphs)"
    DEP_TRAIN=""
fi

# ── Steps 3 & 4: Train GNN (classical + quantum in parallel) ──────────────────
if [[ $CLASSICAL -eq 1 ]]; then
    JID3=$(sbatch --parsable $DEP_TRAIN run/03_train_classical.sh)
    echo "Submitted 03_train_classical  → job $JID3"
fi

if [[ $QUANTUM -eq 1 ]]; then
    JID4=$(sbatch --parsable $DEP_TRAIN run/04_train_quantum.sh)
    echo "Submitted 04_train_quantum    → job $JID4"
fi

echo ""
echo "Monitor with:  squeue -u $USER"
echo "Logs in:       run/logs/"
