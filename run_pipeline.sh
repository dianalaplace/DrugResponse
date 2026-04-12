#!/bin/bash
set -e

echo "=== OncoPrecision — Real Data Pipeline ==="

echo "[1/8] Loading TCGA-BRCA..."
python src/data_loader.py

echo "[2/8] Downloading GEO pCR datasets (GSE25065, GSE20194, GSE41998)..."
python src/geo_loader.py

echo "[3/8] Downloading Wu 2021 scRNA atlas (GSE176078)..."
echo "[4/8] Building scRNA-calibrated TME signatures..."
python src/scrna_deconvolution.py

echo "[5/8] LM22 TME deconvolution (baseline)..."
python src/tme_deconvolution.py

echo "[6/8] Feature engineering..."
python src/feature_engineering.py --use-tme

echo "[7/8] Ablation study (4 model variants)..."
python src/models.py --mode all --ablation

echo "[8/8] SHAP + spatial validation + report..."
python src/shap_analysis.py --model full
python src/spatial_validation.py
python src/evaluation.py --model full --report

echo ""
echo "Results: results/REPORT.md"
echo "Figures: results/figures/"
echo "Metrics: results/metrics/ablation_study.json"
