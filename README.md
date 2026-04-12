# DrugResponce

Breast cancer targeted-therapy response modeling pipeline integrating genomic/clinical and tumor microenvironment (TME) predictors.

## Structure

This project follows the required structure:

- `data/raw/` raw cohort files
- `data/processed/` model-ready tables
- `src/` pipeline modules
- `results/` metrics, SHAP outputs, survival plots, final report

## Expected raw inputs

- `data/raw/tcga_brca_rnaseq_tpm.tsv`
- `data/raw/tcga_brca_clinical.tsv`

If missing, `src/data_loader.py` attempts programmatic retrieval via GDC API (clinical fully, RNA-seq metadata scaffold).

## Run

```bash
cd DrugResponce
pip install -r requirements.txt
./run_pipeline.sh
```

TME-only run:

```bash
./run_tme_pipeline.sh
```

## Outputs

- `results/metrics/*.json`
- `results/shap/*.png`
- `results/survival/km_curves.png`
- `results/REPORT.md`
