# DrugResponce

Breast-cancer pathological complete response (pCR) prediction pipeline, trained on real histopathological response labels from GEO neoadjuvant cohorts with a leakage-safe, gene-first protocol.

## What this repo is

A deliberately conservative, reviewer-defensible predictor of pCR to neoadjuvant chemotherapy. The canonical model is **B_clinical_genes** - clinical covariates plus the top-500 variable genes selected *inside* each training fold of a nested cross-validation - chosen because HLA/APM and TME summary blocks do not add stable pooled-level value once variance is honestly controlled. `C_clinical_genes_hla` is retained only as a pre-specified TNBC-specific sensitivity analysis.

The full write-up (Methods, Results, Discussion, Limitations) lives in `results/best_response_model_run/OncoPrecision_BestResponse_Article.docx`.

## Repository layout

```
run_best_response_model.py          canonical CLI entry point
src/
  best_response_model.py            nested-CV pipeline with leakage-safe feature selection
  publication_data.py               GEO series-matrix loader + I-SPY2 biomarker table loader
  publication_features.py           I-SPY2 reduced-feature proxy helper
  data_loader.py                    shared constants (RANDOM_STATE, PROCESSED_DIR, RAW_DIR)
scripts/
  make_best_response_figures.py     publication figure generation (ROC/PR, SHAP, subgroups, DCA, external)
create_best_response_article.js     DOCX article builder (docx.js)
signatures/                         curated gene-set reference files
data/raw/                           cohort source files (gitignored)
data/processed/                     derived tables (gitignored)
results/                            all outputs (gitignored)
```

## Setup

```bash
pip install -r requirements.txt
npm install      # only needed to rebuild the DOCX article
```

## Run the canonical pipeline

```bash
python run_best_response_model.py \
  --top-genes 500 \
  --outer-splits 3 \
  --inner-splits 2 \
  --bootstrap 1000 \
  --output-dir results/best_response_model_run
```

This will:

- Build the pooled GEO training cohort (GSE20194 + GSE25065, n = 460 after QC, pCR rate 21.3%).
- Run nested 3 × 2 stratified cross-validation across four nested feature representations (A clinical / B +genes / C +HLA / D +TME).
- Fit the final production models on the full cohort using the in-fold-selected hyperparameters.
- Produce SHAP interpretability for the selected primary model.
- Run subgroup analyses by molecular subtype (HER2+, HR+/HER2-, TNBC).
- Run reduced-feature proxy external validation on the I-SPY2 biomarker table.
- Write `results/best_response_model_run/REPORT.md` and `manifest.json`.

## Regenerate figures and article

```bash
python scripts/make_best_response_figures.py
node create_best_response_article.js
```

Outputs land in `results/best_response_model_run/figures/` and `results/best_response_model_run/OncoPrecision_BestResponse_Article.docx`.

## Honest limitations

- Training uses Affymetrix HG-U133A (GPL96) microarray data, which is dated relative to modern RNA-seq.
- Full gene-level external transfer to I-SPY2 is blocked by the missing GPL20078 probe-to-gene annotation; the reported external number is a **reduced-feature proxy lower bound**, not the canonical external metric.
- n = 460 training samples; subgroup ns are modest (HER2+ n = 61), reflected in wide bootstrap 95% CIs.
- Training chemotherapy regimens (taxane / anthracycline) do not cover modern checkpoint-inhibitor combinations - those are represented only indirectly via the I-SPY2 durvalumab/olaparib arm in the proxy validation.

The critical next step before any clinical claim is matched validation on a modern RNA-seq neoadjuvant cohort with real histopathological pCR labels and a shared gene universe.
