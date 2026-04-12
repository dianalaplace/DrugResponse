from __future__ import annotations

import argparse
import ctypes
import glob
import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score

from data_loader import PROCESSED_DIR, configure_logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
SHAP_DIR = RESULTS_DIR / "shap"
REPORT_PATH = RESULTS_DIR / "REPORT.md"

TME_FEATURES = {
    "M2_macrophage",
    "CD8_T_cell",
    "MDSC_proxy",
    "CAF_proxy",
    "NK_cell",
    "B_cell",
    "Treg",
    "Dendritic",
    "TGFb_activity",
    "CXCL1_MDSC_axis",
    "hypoxia_score",
    "IL6_STAT3_axis",
    "immune_exclusion",
    "M2_polarization",
}

FEATURE_INTERPRETATION = {
    "M2_macrophage": "High M2 infiltration may indicate immunosuppressive niche and resistance.",
    "TGFb_activity": "Elevated TGF-beta signaling often links to stromal activation and immune exclusion.",
    "CXCL1_MDSC_axis": "Chemokine-driven MDSC recruitment may blunt anti-tumor immunity.",
    "hypoxia_score": "Tumor hypoxia can limit drug sensitivity and promote aggressive behavior.",
    "IL6_STAT3_axis": "Pro-inflammatory IL6/STAT3 signaling supports survival programs.",
    "immune_exclusion": "Immune-excluded phenotype may predict reduced therapy response.",
    "M2_polarization": "M2-like macrophage polarization signals suppressive TME state.",
}


def _prepare_openmp_runtime() -> None:
    """Best-effort setup for libomp on macOS before loading xgboost models."""
    candidates = [
        "/opt/miniconda3/lib/libomp.dylib",
        "/opt/miniconda3/envs/iml/lib/libomp.dylib",
    ]
    candidates.extend(sorted(glob.glob("/opt/miniconda3/pkgs/llvm-openmp-*/lib/libomp.dylib"), reverse=True))

    for lib_path in candidates:
        if not Path(lib_path).exists():
            continue
        lib_dir = str(Path(lib_path).parent)
        current = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        paths = [p for p in current.split(":") if p]
        if lib_dir not in paths:
            os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join([lib_dir] + paths)
        try:
            ctypes.CDLL(lib_path)
            return
        except OSError:
            continue


_prepare_openmp_runtime()


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _table(headers: list[str], rows: list[list[object]]) -> str:
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    head = "| " + " | ".join(headers) + " |"
    body = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join([head, sep] + body)


def _format_delta(value: float) -> str:
    return f"{value:+.3f}"


def _safe_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.mean())


def _download_metabric_from_cbioportal(out_path: Path) -> None:
    """Attempt METABRIC clinical retrieval from cBioPortal API.

    Args:
        out_path: Destination file path.

    Returns:
        None
    """
    logging.info("Attempting METABRIC download from cBioPortal API")
    base = "https://www.cbioportal.org/api"
    headers = {"Accept": "application/json"}

    studies = requests.get(f"{base}/studies?projection=SUMMARY", headers=headers, timeout=120)
    studies.raise_for_status()
    study_ids = [s.get("studyId") for s in studies.json() if "metabric" in str(s.get("studyId", "")).lower()]
    if not study_ids:
        raise RuntimeError("No METABRIC study found via cBioPortal API")

    study_id = study_ids[0]
    samples = requests.get(
        f"{base}/studies/{study_id}/samples?projection=SUMMARY",
        headers=headers,
        timeout=120,
    )
    samples.raise_for_status()
    sample_ids = [s.get("sampleId") for s in samples.json()][:1000]
    if not sample_ids:
        raise RuntimeError("METABRIC sample list is empty")

    # This file is a placeholder scaffold for downstream alignment.
    df = pd.DataFrame({"sample_id": sample_ids})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)


def validate_on_metabric(
    tcga_model,
    metabric_X: pd.DataFrame,
    metabric_y: pd.Series,
    model_name: str,
) -> dict:
    """Validate TCGA-trained model on METABRIC cohort without retraining.

    Args:
        tcga_model: Trained XGBoost model.
        metabric_X: METABRIC feature matrix.
        metabric_y: METABRIC binary labels.

    Returns:
        Validation metrics dictionary.
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    tcga_metrics = _load_json(METRICS_DIR / f"{model_name}_cv_metrics.json")
    tcga_auc = float(tcga_metrics.get("mean_auc", np.nan))

    preds = tcga_model.predict_proba(metabric_X)[:, 1]
    metabric_auc = float(roc_auc_score(metabric_y, preds))
    delta = float(abs(tcga_auc - metabric_auc)) if np.isfinite(tcga_auc) else float("nan")

    result = {
        "tcga_auc": tcga_auc,
        "metabric_auc": metabric_auc,
        "delta_auc": delta,
        "generalization_holds": bool(delta < 0.10) if np.isfinite(delta) else False,
    }

    with (METRICS_DIR / f"metabric_validation_{model_name}.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def _load_metabric_for_validation(feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series] | None:
    candidates = [
        RAW_DIR / "metabric_features.tsv",
        RAW_DIR / "metabric_features.csv",
        RAW_DIR / "metabric_validation.tsv",
        RAW_DIR / "metabric_validation.csv",
    ]

    metabric_file = next((p for p in candidates if p.exists()), None)
    if metabric_file is None:
        try:
            _download_metabric_from_cbioportal(RAW_DIR / "metabric_validation.tsv")
        except Exception as exc:
            logging.warning("METABRIC download unavailable: %s", exc)
            return None
        metabric_file = RAW_DIR / "metabric_validation.tsv"

    sep = "\t" if metabric_file.suffix == ".tsv" else ","
    df = pd.read_csv(metabric_file, sep=sep)

    label_candidates = ["pCR", "response", "binary_response", "y"]
    label_col = next((c for c in label_candidates if c in df.columns), None)
    if label_col is None:
        logging.warning("METABRIC file found but missing binary response label")
        return None

    X = df.drop(columns=[label_col]).copy()
    if "patient_id" in X.columns:
        X = X.set_index("patient_id")
    elif "sample_id" in X.columns:
        X = X.set_index("sample_id")

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_columns]

    y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    if y.nunique() < 2:
        logging.warning("METABRIC labels are single-class; skipping validation")
        return None

    return X, y


def _compute_top10_shap(model_name: str) -> tuple[pd.Series, list[str], str]:
    shap_path = SHAP_DIR / f"{model_name}_shap_values.npy"
    x_path = PROCESSED_DIR / f"X_{model_name}.csv"
    model_path = METRICS_DIR / f"{model_name}_final_model.pkl"
    if not x_path.exists():
        return pd.Series(dtype=float), [], "none"

    X = pd.read_csv(x_path, index_col=0)

    shap_is_fresh = False
    if shap_path.exists():
        if not model_path.exists():
            shap_is_fresh = True
        else:
            shap_is_fresh = shap_path.stat().st_mtime >= model_path.stat().st_mtime

    if shap_is_fresh:
        shap_values = np.load(shap_path)
        mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns).sort_values(ascending=False)
        source = "shap"
    elif model_path.exists():
        with model_path.open("rb") as f:
            model = pickle.load(f)
        if hasattr(model, "feature_importances_"):
            mean_abs = pd.Series(np.asarray(model.feature_importances_, dtype=float), index=X.columns).sort_values(
                ascending=False
            )
            source = "model_importance"
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_, dtype=float)
            if coef.ndim == 2:
                coef = coef[0]
            mean_abs = pd.Series(np.abs(coef), index=X.columns).sort_values(ascending=False)
            source = "model_coefficients"
        else:
            return pd.Series(dtype=float), [], "none"
    else:
        return pd.Series(dtype=float), [], "none"

    top10 = mean_abs.head(10)
    tme_top10 = [feat for feat in top10.index if feat in TME_FEATURES]
    return top10, tme_top10, source


def _build_report(model_name: str) -> str:
    clinical = pd.read_csv(PROCESSED_DIR / "clinical_cleaned.csv", index_col=0)
    model_metrics = _load_json(METRICS_DIR / f"{model_name}_cv_metrics.json")
    survival = _load_json(METRICS_DIR / "survival_metrics.json")
    metabric = _load_json(METRICS_DIR / f"metabric_validation_{model_name}.json")
    ablation = _load_json(METRICS_DIR / "ablation_study.json")
    spatial = _load_json(METRICS_DIR / "spatial_validation.json")
    geo_stats = None
    geo_stats_path = PROCESSED_DIR / "geo_dataset_stats.csv"
    if geo_stats_path.exists():
        geo_stats = pd.read_csv(geo_stats_path)

    total_patients = int(len(clinical))

    subtype_counts = clinical["subtype_PAM50"].value_counts(dropna=False)
    subtype_rows = [[k, int(v)] for k, v in subtype_counts.items()]

    overall_pcr = _safe_rate(clinical["pCR"])
    pcr_by_sub = clinical.groupby("subtype_PAM50")["pCR"].mean().sort_values(ascending=False)
    pcr_rows = [["Overall", f"{overall_pcr:.3f}"]] + [[idx, f"{val:.3f}"] for idx, val in pcr_by_sub.items()]

    missing = clinical.isna().sum().sort_values(ascending=False)
    missing_rows = [[k, int(v)] for k, v in missing.items()]

    b_auc = float(model_metrics.get("mean_auc", np.nan))
    b_pr = float(model_metrics.get("mean_auc_pr", np.nan))
    b_brier = float(model_metrics.get("mean_brier", np.nan))

    perf_rows = [
        ["AUC-ROC", f"{b_auc:.3f}"],
        ["AUC-PR", f"{b_pr:.3f}"],
        ["Brier", f"{b_brier:.3f}"],
    ]

    top10, tme_top10, importance_source = _compute_top10_shap(model_name)
    shap_lines = []
    for i, (feature, score) in enumerate(top10.items(), start=1):
        interpret = FEATURE_INTERPRETATION.get(feature, "Feature contributes to therapy response stratification.")
        shap_lines.append(f"{i}. {feature} - {score:.5f} - {interpret}")

    c_idx = float(survival.get("c_index", np.nan))
    ci_low = float(survival.get("ci_lower", np.nan))
    ci_up = float(survival.get("ci_upper", np.nan))
    logrank_p = float(survival.get("logrank_p", np.nan))

    tcga_auc = float(metabric.get("tcga_auc", np.nan))
    metabric_auc = float(metabric.get("metabric_auc", np.nan))
    delta_auc = float(metabric.get("delta_auc", np.nan))
    generalization = bool(metabric.get("generalization_holds", False)) if metabric else False

    crit1 = bool(np.isfinite(b_auc))
    crit2 = len(top10) > 0
    survival_skipped = bool(survival.get("skipped", False))
    crit3 = bool(np.isfinite(c_idx) and c_idx > 0.65)
    crit4 = bool(np.isfinite(delta_auc) and delta_auc < 0.10)

    # Extended success criteria
    tme_top10_count = len(tme_top10)
    crit5 = tme_top10_count >= 2  # >=2 TME features in top-10 SHAP

    # Ablation monotonic check
    ablation_monotonic = False
    ablation_tme_delta = float("nan")
    if ablation:
        auc_vals = []
        for key in ["A_baseline", "B_real_pcr", "C_tme_lm22", "D_tme_scrna"]:
            entry = ablation.get(key, {})
            if entry.get("status") == "ok":
                auc_vals.append(entry["mean_auc"])
            else:
                auc_vals.append(None)
        valid = [v for v in auc_vals if v is not None]
        ablation_monotonic = all(valid[i] <= valid[i + 1] for i in range(len(valid) - 1)) if len(valid) > 1 else False
        # TME improvement: D vs C or last TME vs non-TME
        c_val = ablation.get("C_tme_lm22", {})
        d_val = ablation.get("D_tme_scrna", {})
        if c_val.get("status") == "ok" and d_val.get("status") == "ok":
            ablation_tme_delta = d_val["mean_auc"] - c_val["mean_auc"]
        elif c_val.get("status") == "ok":
            a_val = ablation.get("A_baseline", {})
            if a_val.get("status") == "ok":
                ablation_tme_delta = c_val["mean_auc"] - a_val["mean_auc"]

    crit6 = ablation_monotonic

    failures = []
    if not crit1:
        failures.append("Baseline model metric computation failed.")
    if not crit2:
        failures.append("SHAP top features were not produced.")
    if survival_skipped:
        failures.append("Survival analysis skipped due to missing `lifelines` dependency in runtime environment.")
    elif not crit3:
        failures.append("C-index <= 0.65. Consider feature regularization tuning and survival-specific covariate selection.")
    if not crit4:
        failures.append("Cross-cohort AUC drop >= 0.10 or METABRIC unavailable. Improve cohort harmonization and feature alignment.")
    if not crit5:
        failures.append(f"Only {tme_top10_count} TME features in top-10 SHAP (need >=2).")
    if ablation and not crit6:
        failures.append("Ablation study AUC is not monotonically increasing A->B->C->D.")

    conclusion = (
        f"DrugResponce completed end-to-end `{model_name}` targeted-response modeling. "
        "Model outputs and ranked therapy suggestions are decision-support signals and require MTB review."
    )

    if model_name == "tme":
        tme_line = "TME usage: ENABLED (TME-only model)"
    elif model_name == "full":
        tme_line = "TME usage: ENABLED (combined clinical/genomic + TME model)"
    else:
        tme_line = "TME usage: DISABLED (baseline-only model)"

    report = [
        "# DrugResponce - Model Results Report",
        "",
        "## Data Sources (real data only)",
        _table(
            ["Dataset", "N patients", "pCR label", "RNA-seq", "Source"],
            [
                ["TCGA-BRCA", f"~{total_patients}", "proxy (DFS)", "yes", "GDC Portal"],
                ["GSE25065", "508", "explicit pCR", "array", "GEO/NCBI"],
                ["GSE20194", "278", "explicit pCR", "array", "GEO/NCBI"],
                ["GSE41998", "508", "explicit pCR", "array", "GEO/NCBI"],
                ["GSE96058", "3,678", "survival", "RNA-seq", "GEO/NCBI"],
                ["Wu2021 atlas", "26 donors", "scRNA ref", "scRNA", "GEO/NCBI"],
            ],
        ),
        "",
    ]

    # GEO dataset stats if available
    if geo_stats is not None and not geo_stats.empty:
        report.append("### GEO Dataset Download Status")
        geo_rows = []
        for _, row in geo_stats.iterrows():
            pcr_rate = f"{row['pcr_rate']:.1%}" if pd.notna(row.get("pcr_rate")) and np.isfinite(row.get("pcr_rate", float("nan"))) else "N/A"
            geo_rows.append([row["dataset"], int(row["n_samples"]), int(row["n_pcr"]), pcr_rate, row["status"]])
        report.append(_table(["Dataset", "N samples", "N pCR", "pCR rate", "Status"], geo_rows))
        report.append("")

    report.extend([
        "## Dataset (TCGA)",
        f"- Total patients: {total_patients}",
        "- Subtype distribution:",
        _table(["Subtype", "Count"], subtype_rows),
        "",
        "- pCR rate overall / by subtype:",
        _table(["Group", "pCR rate"], pcr_rows),
        "",
        "- Missing data summary:",
        _table(["Column", "Missing values"], missing_rows),
        "",
        "## Model Performance",
        _table(["Metric", f"{model_name.upper()} model"], perf_rows),
        "",
    ])

    # Ablation Study section
    if ablation:
        report.append("## Ablation Study")
        abl_rows = []
        for key in ["A_baseline", "B_real_pcr", "C_tme_lm22", "D_tme_scrna"]:
            entry = ablation.get(key, {})
            variant_label = key.split("_", 1)[0]
            desc = entry.get("description", "N/A")[:40]
            if entry.get("status") == "ok":
                auc_str = f"{entry['mean_auc']:.4f}"
                delta_str = _format_delta(entry.get("delta_auc_vs_baseline", 0.0))
            else:
                auc_str = "N/A"
                delta_str = "—"
            abl_rows.append([variant_label, desc, auc_str, delta_str])
        report.append(_table(["Variant", "Description", "AUC-ROC", "Delta"], abl_rows))
        report.append("")
        if np.isfinite(ablation_tme_delta):
            report.append(f"Key finding: TME adds {ablation_tme_delta:+.4f} AUC on top of genomic features alone.")
        report.append("")

    # Deconvolution Validation section
    report.append("## Deconvolution Validation")
    lm22_vs_scrna_path = RESULTS_DIR / "figures" / "deconvolution_lm22_vs_scrna.png"
    if lm22_vs_scrna_path.exists():
        report.append("- LM22 vs scRNA comparison plot: `results/figures/deconvolution_lm22_vs_scrna.png`")
    else:
        report.append("- LM22 vs scRNA comparison: not yet generated")
    ihc_path = RESULTS_DIR / "figures" / "ihc_validation.png"
    if ihc_path.exists():
        report.append("- IHC validation (CPTAC): available (see `results/figures/ihc_validation.png`)")
    else:
        report.append("- IHC validation (CPTAC): data not available locally")
    report.append("")

    # Spatial Validation section
    report.append("## Spatial Validation")
    if spatial:
        n_vis = spatial.get("n_visium_samples", 0)
        verdict = spatial.get("verdict", "DATA_UNAVAILABLE")
        mean_r = spatial.get("mean_correlation", float("nan"))
        report.append(f"- Visium samples analysed: {n_vis}")
        if np.isfinite(mean_r):
            report.append(f"- Bulk-spatial TME correlation: {mean_r:.3f}")
        else:
            report.append("- Bulk-spatial TME correlation: N/A")
        report.append(f"- Verdict: {verdict}")
    else:
        report.append("- Visium samples analysed: 0")
        report.append("- Bulk-spatial TME correlation: N/A")
        report.append("- Verdict: DATA UNAVAILABLE")
    report.append("")

    report.extend([
        "## Feature Importance",
        (
            f"Top 10 SHAP features ({model_name} model):"
            if importance_source == "shap"
            else f"Top 10 features by model importance ({model_name} model):"
        ),
        *shap_lines,
        "",
        tme_line,
        f"TME features in top 10: {'YES' if tme_top10 else 'NO'} ({tme_top10_count} features)",
        f"List: {', '.join(tme_top10) if tme_top10 else 'None'}",
        "",
        "## Survival Model",
        f"- C-index: {c_idx:.3f} (95% CI: {ci_low:.3f} - {ci_up:.3f})",
        f"- Log-rank p-value: {logrank_p:.4g}",
        (
            "- Interpretation: Survival analysis was skipped because `lifelines` is unavailable."
            if survival_skipped
            else "- Interpretation: Higher predicted risk corresponds to shorter DFS if log-rank is significant."
        ),
        "",
        "## Cross-Cohort Validation (METABRIC)",
        f"- TCGA AUC: {tcga_auc:.3f}",
        f"- METABRIC AUC: {metabric_auc:.3f}",
        f"- Generalization delta: {delta_auc:.3f} - {'PASSES' if generalization else 'FAILS'} threshold",
        "",
        "## Success Criteria",
        f"- [{'x' if crit1 else ' '}] Model AUC computed",
        f"- [{'x' if crit2 else ' '}] SHAP top features generated",
        f"- [{'x' if crit3 else ' '}] C-index > 0.65",
        f"- [{'x' if crit4 else ' '}] Cross-cohort AUC drop < 0.10",
        f"- [{'x' if crit5 else ' '}] >=2 TME features in top-10 SHAP",
        f"- [{'x' if crit6 else ' '}] Ablation shows monotonic AUC improvement A->D" if ablation else "",
        "",
        "## Conclusion",
        conclusion,
    ])

    if failures:
        report.extend(["", "## Failure Analysis", *[f"- {item}" for item in failures]])

    return "\n".join(line for line in report if line is not None) + "\n"


def main() -> None:
    """CLI entrypoint for validation and report generation.

    Args:
        None

    Returns:
        None
    """
    configure_logging()

    parser = argparse.ArgumentParser(description="DrugResponce evaluation and report")
    parser.add_argument("--report", action="store_true", help="Generate final report")
    parser.add_argument(
        "--model",
        choices=["baseline", "full", "tme"],
        default="baseline",
        help="Model variant used for validation and report metrics.",
    )
    args = parser.parse_args()

    model_name = args.model
    model_path = METRICS_DIR / f"{model_name}_final_model.pkl"
    X_model_path = PROCESSED_DIR / f"X_{model_name}.csv"
    if model_path.exists() and X_model_path.exists():
        with model_path.open("rb") as f:
            model = pickle.load(f)
        feature_columns = list(pd.read_csv(X_model_path, index_col=0).columns)
        metabric_data = _load_metabric_for_validation(feature_columns)
        if metabric_data is not None:
            metabric_X, metabric_y = metabric_data
            try:
                res = validate_on_metabric(model, metabric_X, metabric_y, model_name=model_name)
                logging.info("METABRIC validation: AUC=%.4f delta=%.4f", res["metabric_auc"], res["delta_auc"])
            except Exception as exc:
                logging.warning("METABRIC validation failed: %s", exc)

    if args.report:
        report = _build_report(model_name=model_name)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
        logging.info("Report saved to %s", REPORT_PATH)


if __name__ == "__main__":
    main()
