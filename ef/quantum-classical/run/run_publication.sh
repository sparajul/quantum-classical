#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_publication.sh — Full publication pipeline
#
# Steps:
#   1. Multi-seed sweep: classical model (5 seeds)
#   2. Multi-seed sweep: quantum model  (5 seeds)
#   3. Barren plateau analysis
#   4. Scaling study (quantum n_qubits x n_qlayers grid)
#   5. Track-level reconstruction on best checkpoint
#   6. Efficiency/purity vs eta and pt plots
#   7. Results table (LaTeX + Markdown)
#
# Edit CKPT_CLASSICAL and CKPT_QUANTUM below after step 1-2 complete.
# ─────────────────────────────────────────────────────────────────────────────

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG="configs/default.yaml"
RESULTS="results"
SEEDS="42 123 456 789 1337"

# ── After steps 1-2, set these to your best checkpoints ──────────────────────
CKPT_CLASSICAL="${RESULTS}/classical/seed_42/checkpoints/best.ckpt"
CKPT_QUANTUM="${RESULTS}/quantum/seed_42/checkpoints/best.ckpt"

echo "================================================================"
echo "  STEP 1: Classical multi-seed sweep"
echo "================================================================"
python scripts/train_sweep.py \
    --config "$CONFIG" \
    --model_type classical \
    --seeds $SEEDS \
    --output_dir "${RESULTS}/classical/"

echo "================================================================"
echo "  STEP 2: Quantum multi-seed sweep"
echo "================================================================"
python scripts/train_sweep.py \
    --config "$CONFIG" \
    --model_type quantum \
    --seeds $SEEDS \
    --output_dir "${RESULTS}/quantum/"

echo "================================================================"
echo "  STEP 3: Barren plateau analysis"
echo "================================================================"
python utils/barren_plateau.py \
    --config "$CONFIG" \
    --output_dir "${RESULTS}/barren_plateau/" \
    --n_samples 50 \
    --qubit_range 2 4 6 8 10 12 \
    --layer_range 1 2 3 4 5 6

echo "================================================================"
echo "  STEP 4: Scaling study (quantum)"
echo "================================================================"
python scripts/scaling_study.py \
    --sweep_config configs/scaling_sweep.yaml \
    --base_config "$CONFIG" \
    --output_dir "${RESULTS}/scaling/"

echo "================================================================"
echo "  STEP 5: Track-level reconstruction (classical)"
echo "================================================================"
python scripts/track_building.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT_CLASSICAL" \
    --split test \
    --edge_cut 0.3 \
    --output_dir "${RESULTS}/tracks_classical/"

echo "================================================================"
echo "  STEP 5b: Track-level reconstruction (quantum)"
echo "================================================================"
python scripts/track_building.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT_QUANTUM" \
    --split test \
    --edge_cut 0.3 \
    --output_dir "${RESULTS}/tracks_quantum/"

echo "================================================================"
echo "  STEP 6: Efficiency/purity vs eta and pt (classical)"
echo "================================================================"
python scripts/plot_efficiency_purity.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT_CLASSICAL" \
    --split test --edge_cut 0.3 \
    --output_dir "${RESULTS}/plots_classical/"

echo "================================================================"
echo "  STEP 6b: Efficiency/purity vs eta and pt (quantum)"
echo "================================================================"
python scripts/plot_efficiency_purity.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT_QUANTUM" \
    --split test --edge_cut 0.3 \
    --output_dir "${RESULTS}/plots_quantum/"

echo "================================================================"
echo "  STEP 7: Results table"
echo "================================================================"
python scripts/make_results_table.py \
    --classical "${RESULTS}/classical/sweep_summary.json" \
    --quantum   "${RESULTS}/quantum/sweep_summary.json" \
    --output_dir "${RESULTS}/"

echo ""
echo "================================================================"
echo "  ALL DONE — results in ${RESULTS}/"
echo "  Key files:"
echo "    ${RESULTS}/table.tex            — LaTeX results table"
echo "    ${RESULTS}/table.md             — Markdown results table"
echo "    ${RESULTS}/barren_plateau/      — Barren plateau plots"
echo "    ${RESULTS}/scaling/             — Scaling study"
echo "    ${RESULTS}/tracks_*/            — Track-level metrics"
echo "    ${RESULTS}/plots_*/             — Efficiency/purity plots"
echo "================================================================"
