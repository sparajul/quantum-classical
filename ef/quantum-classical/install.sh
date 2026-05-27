#!/usr/bin/env bash
# Installs PyTorch, torch-geometric, and torch-scatter with the correct
# CUDA-specific wheel URLs. Run this after: conda activate qgnn
set -euo pipefail

# ── Detect CUDA ───────────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
    CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    case "$MAJOR" in
        12) CUDA_TAG="cu121" ;;
        11) CUDA_TAG="cu118" ;;
        *)  CUDA_TAG="cu121" ;;
    esac
    echo "GPU detected — CUDA $CUDA_VER → using wheel tag: $CUDA_TAG"
else
    CUDA_TAG="cpu"
    echo "No GPU detected — installing CPU-only PyTorch"
fi

TORCH_URL="https://download.pytorch.org/whl/${CUDA_TAG}"

# ── PyTorch ───────────────────────────────────────────────────────────────────
echo ""
echo "Installing PyTorch from ${TORCH_URL} ..."
pip install torch --index-url "$TORCH_URL"

# ── torch-geometric + torch-scatter ──────────────────────────────────────────
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html"

echo "Installing torch-geometric and torch-scatter from ${PYG_URL} ..."
pip install torch-geometric torch-scatter torch-cluster -f "$PYG_URL"

echo ""
echo "All done. Run:  conda activate qgnn"
