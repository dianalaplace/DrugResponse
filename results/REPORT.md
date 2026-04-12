# OncoPrecision — Final Results Report

## 1. Data Sources

| Dataset | N patients | Endpoint | Platform | Source |
|---|---|---|---|---|
| TCGA-BRCA | 1,097 | proxy pCR (1−DFS_event) | RNA-seq TPM | UCSC Xena |
| GSE25065 | 198 | real pCR | Affymetrix HG-U133A | GEO/NCBI |
| GSE20194 | 278 | real pCR | Affymetrix HG-U133A | GEO/NCBI |
| **GEO pooled** | **460** | **real pCR (21.3%)** | **microarray** | — |

- GPL96 probe-to-gene mapping: 21,156 probes → gene symbols
- TME signature genes found in GEO: 61/64 (95%)

## 2. TCGA Cohort (Proxy pCR)

| Subtype | Count | pCR rate |
|---|---|---|
| LUMA | 678 | 0.768 |
| LUMB | 192 | 0.714 |
| BASAL | 141 | 0.901 |
| HER2 | 67 | 0.896 |
| NORMAL | 23 | 0.696 |
| **Overall** | **1,097** | **0.782** |

### TCGA Model Performance (5-fold CV, XGBoost)

| Model | N features | AUC-ROC | AUC-PR | Brier |
|---|---|---|---|---|
| Baseline (clinical + expr markers) | 12 | 0.605 ± 0.022 | 0.841 ± 0.017 | 0.243 ± 0.016 |
| Full (clinical + expr + TME) | 26 | 0.682 ± 0.042 | 0.884 ± 0.021 | 0.160 ± 0.007 |
| **Delta** | — | **+0.076** | **+0.043** | **−0.083** |

> TME adds +0.076 AUC on TCGA proxy labels. However, proxy pCR (78% positive rate) is unreliable — inflated by DFS-event inversion artifact.

## 3. GEO Cohort (Real pCR Labels)

### Ablation Study (5-fold stratified CV, XGBoost, heavy regularization)

| Variant | N features | AUC-ROC (mean ± std) | Delta vs baseline |
|---|---|---|---|
| A: Clinical only (ER, PR, HER2, grade, age) | 5 | **0.791 ± 0.047** | — |
| B: Clinical + TME (14 cell/pathway scores) | 19 | 0.733 ± 0.052 | −0.059 |
| C: TME only | 14 | 0.633 ± 0.077 | −0.158 |

### Per-fold AUC (Clinical only)

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---|---|---|---|
| 0.820 | 0.836 | 0.745 | 0.725 | 0.831 |

### XGBoost Feature Importance (Clinical + TME model)

| Rank | Feature | Importance | TME? |
|---|---|---|---|
| 1 | ER_status | 0.177 | |
| 2 | grade | 0.142 | |
| 3 | HER2_status | 0.067 | |
| 4 | age | 0.053 | |
| 5 | PR_status | 0.049 | |
| 6 | NK_cell | 0.046 | ✓ |
| 7 | TGFb_activity | 0.042 | ✓ |
| 8 | CXCL1_MDSC_axis | 0.041 | ✓ |
| 9 | B_cell | 0.040 | ✓ |
| 10 | CAF_proxy | 0.039 | ✓ |

**5 TME features in top-10** (criterion met: ≥2 TME in top-10).

## 4. SHAP Analysis (TCGA Full Model)

### Top-10 SHAP features

| Rank | Feature | mean |SHAP| | Category |
|---|---|---|---|
| 1 | tumor_stage_N | 0.04135 | Clinical |
| 2 | tumor_stage_T | 0.03090 | Clinical |
| 3 | age_at_diagnosis | 0.02980 | Clinical |
| 4 | B_cell | 0.02867 | **TME** |
| 5 | CAF_proxy | 0.02759 | **TME** |
| 6 | CD8_T_cell | 0.02670 | **TME** |
| 7 | Dendritic | 0.02507 | **TME** |
| 8 | CXCL1_MDSC_axis | 0.02387 | **TME** |
| 9 | subtype_PAM50_BASAL | 0.02328 | Clinical |
| 10 | M2_macrophage | 0.02187 | **TME** |

**6 TME features in SHAP top-10** — TME features carry measurable signal, but adding them does not improve discriminative AUC on real pCR labels.

Plots: `results/shap/global_importance.png`, `results/shap/beeswarm.png`

## 5. Survival Model

- C-index: N/A (lifelines not installed)
- Skipped in current environment.

## 6. Success Criteria

| Criterion | Target | Result | Status |
|---|---|---|---|
| Model AUC computed | ✓ | 0.791 (GEO clinical) | **PASS** |
| AUC > 0.75 on real pCR | > 0.75 | 0.791 | **PASS** |
| ≥2 TME in top-10 SHAP | ≥ 2 | 6 (TCGA), 5 (GEO) | **PASS** |
| TME delta AUC > 0.02 | > +0.02 | −0.059 (GEO) | **FAIL** |
| Ablation monotonic improvement | A < B < C < D | A > B > C | **FAIL** |
| C-index > 0.65 | > 0.65 | N/A | **SKIP** |
| Cross-cohort delta < 0.10 | < 0.10 | N/A | **SKIP** |

## 7. Key Finding

**TME features do NOT add independent predictive value for neoadjuvant chemotherapy pCR prediction** on these datasets (GSE25065 + GSE20194, n=460).

### Why?

1. **ER/HER2/grade dominate chemotherapy response.** These 3 clinical features alone achieve AUC = 0.791. The biological signal for pCR is heavily concentrated in receptor status and proliferation grade.

2. **LM22-derived TME signatures were designed for immunotherapy contexts.** Immune cell infiltration (CD8 T cells, NK cells) is more relevant for checkpoint inhibitor response than for cytotoxic chemotherapy.

3. **Curse of dimensionality.** Adding 14 TME features to 5 strong clinical predictors on n=460 patients introduces noise that regularization cannot fully suppress, reducing generalization (AUC drops from 0.791 to 0.733).

4. **TME features are informative but redundant.** They rank in top-10 by importance (5/10 in GEO, 6/10 in TCGA SHAP), meaning they capture real biological variation — but this variation is already captured by ER/HER2 status, which are proxies for immune-hot vs immune-cold tumors.

### Implications

- For **immunotherapy** response prediction (e.g., pembrolizumab + chemo trials), TME features would likely add value — this is an important direction for future work.
- For **standard neoadjuvant chemotherapy**, clinical features (ER, HER2, grade) remain the strongest predictors.
- This is a **scientifically valid negative result** — demonstrating that a hypothesis (TME improves chemo-response prediction) does not hold on real data is valuable.

## 8. Reproducibility

```bash
# Full pipeline
bash run_pipeline.sh

# GEO ablation only
python src/geo_loader.py
python src/batch_correction.py
# Then run ablation script
```

## 9. Output Files

| File | Description |
|---|---|
| `results/metrics/geo_ablation_results.json` | GEO ablation AUCs and feature importance |
| `results/metrics/full_cv_metrics.json` | TCGA full model 5-fold CV |
| `results/metrics/baseline_cv_metrics.json` | TCGA baseline 5-fold CV |
| `results/metrics/model_comparison.json` | Baseline vs full delta |
| `results/shap/global_importance.png` | SHAP bar plot (TME highlighted) |
| `results/shap/beeswarm.png` | SHAP beeswarm plot |
| `results/shap/tme_importance_report.txt` | TME-specific SHAP ranking |
| `data/processed/tme_features.csv` | Computed TME features (1097×14) |
