#!/bin/bash
set -e

echo "=== DrugResponce TME-Only Pipeline ==="

echo "[1/7] Loading and preprocessing data..."
python src/data_loader.py

echo "[2/7] TME deconvolution..."
python src/tme_deconvolution.py

echo "[3/7] Feature engineering (with TME)..."
python src/feature_engineering.py --use-tme

echo "[4/7] Training TME-only model..."
python src/models.py --mode tme

echo "[5/7] SHAP analysis (TME model)..."
python src/shap_analysis.py --model tme

echo "[6/7] Generating final report (TME model)..."
python src/evaluation.py --model tme --report

echo "[7/7] Example drug recommendation (TME model)..."
python src/treatment_recommendation.py --model tme --top-k 3

echo "=== Done. Results in results/REPORT.md ==="
