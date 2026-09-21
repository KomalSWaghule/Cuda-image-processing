#!/bin/bash

set -e

echo "=========================================="
echo " CUDA Image Processing at Scale"
echo "=========================================="

echo ""
echo "[1/3] Generating dataset..."

python3 generate_images.py

echo ""
echo "[2/3] Compiling CUDA program..."

make build

echo ""
echo "[3/3] Running GPU image processing..."

./image_processor

echo ""
echo "=========================================="
echo " Completed successfully"
echo "=========================================="

echo ""
echo "Outputs:"
echo "  input/  -> Original images"
echo "  output/ -> GPU processed images"
echo "  results/ -> Execution logs"
