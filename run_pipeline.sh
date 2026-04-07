#!/bin/bash
set -e

echo "=== DrugResponce Pipeline ==="

echo "[1/7] Loading and preprocessing data..."
python src/data_loader.py

echo "[2/7] TME deconvolution..."
python src/tme_deconvolution.py

echo "[3/7] Feature engineering..."
python src/feature_engineering.py

echo "[4/7] Training baseline model (no TME)..."
python src/models.py --mode baseline

echo "[5/7] Training full model (with TME)..."
python src/models.py --mode full

echo "[6/7] SHAP analysis..."
python src/shap_analysis.py

echo "[7/7] Generating final report..."
python src/evaluation.py --report

echo "=== Done. Results in results/REPORT.md ==="
