from __future__ import annotations

import argparse
import ctypes
import glob
import json
import logging
import os
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from data_loader import PROCESSED_DIR, RANDOM_STATE, configure_logging

ABLATION_CONFIGS = {
    "A_baseline": {
        "data": "tcga_only",
        "target": "proxy_dfs",
        "tme": "none",
        "description": "Clinical + genomic only, proxy pCR target",
    },
    "B_real_pcr": {
        "data": "tcga_plus_geo",
        "target": "real_pcr",
        "tme": "none",
        "description": "Clinical + genomic, real pCR from GEO",
    },
    "C_tme_lm22": {
        "data": "tcga_plus_geo",
        "target": "real_pcr",
        "tme": "lm22",
        "description": "Clinical + genomic + LM22 TME, real pCR",
    },
    "D_tme_scrna": {
        "data": "tcga_plus_geo",
        "target": "real_pcr",
        "tme": "scrna",
        "description": "Clinical + genomic + scRNA TME (BEST)",
    },
}

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.plotting import add_at_risk_counts
    from lifelines.statistics import logrank_test
    from lifelines.utils import concordance_index

    LIFELINES_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment-specific fallback
    CoxPHFitter = None  # type: ignore[assignment]
    KaplanMeierFitter = None  # type: ignore[assignment]
    add_at_risk_counts = None  # type: ignore[assignment]
    logrank_test = None  # type: ignore[assignment]
    concordance_index = None  # type: ignore[assignment]
    LIFELINES_AVAILABLE = False
    LIFELINES_IMPORT_ERROR = exc


def _prepare_openmp_runtime() -> None:
    """Best-effort setup for libomp on macOS before importing xgboost."""
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

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment-specific fallback
    xgb = None  # type: ignore[assignment]
    XGB_AVAILABLE = False
    XGB_IMPORT_ERROR = exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
SURVIVAL_DIR = RESULTS_DIR / "survival"


def _json_ready(metrics: dict) -> dict:
    out: dict[str, object] = {}
    for k, v in metrics.items():
        if k == "trained_models":
            continue
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, list):
            out[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
        else:
            out[k] = v
    return out


def _infer_n_splits(y: pd.Series, target_splits: int = 5) -> int:
    class_counts = y.value_counts()
    min_count = int(class_counts.min())
    return max(2, min(target_splits, min_count))


def train_xgboost_cv(X: pd.DataFrame, y: pd.Series, model_name: str) -> dict:
    """Train an XGBoost classifier with stratified cross-validation.

    Args:
        X: Feature matrix.
        y: Binary target series.
        model_name: Label used for output files.

    Returns:
        Dictionary with CV summary metrics, OOF predictions, and trained models.
    """
    if y.nunique() < 2:
        raise ValueError("Target y must have both classes for classification")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    n_splits = _infer_n_splits(y, target_splits=5)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    oof_predictions = np.zeros(len(y), dtype=float)
    aucs: list[float] = []
    auc_prs: list[float] = []
    briers: list[float] = []
    best_iterations: list[int] = []
    trained_models: list[object] = []

    X_arr = X.values
    y_arr = y.values

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X_arr, y_arr), start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        scale_pos_weight = (neg / max(pos, 1)) if pos > 0 else 1.0

        if XGB_AVAILABLE:
            model = xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                eval_metric="auc",
                early_stopping_rounds=50,
                n_jobs=-1,
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
            )
        else:
            if fold == 1:
                logging.warning(
                    "XGBoost unavailable (%s). Falling back to ExtraTreesClassifier.",
                    str(XGB_IMPORT_ERROR),
                )
            model = ExtraTreesClassifier(
                n_estimators=800,
                max_depth=None,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
                model.fit(X_train, y_train)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
            preds = model.predict_proba(X_valid)[:, 1]
        oof_predictions[valid_idx] = preds

        fold_auc = roc_auc_score(y_valid, preds)
        fold_auc_pr = average_precision_score(y_valid, preds)
        fold_brier = brier_score_loss(y_valid, preds)
        fold_best_iteration = int(getattr(model, "best_iteration", getattr(model, "n_estimators", 0)))

        aucs.append(float(fold_auc))
        auc_prs.append(float(fold_auc_pr))
        briers.append(float(fold_brier))
        best_iterations.append(fold_best_iteration)
        trained_models.append(model)

        logging.info(
            "%s fold %d/%d | AUC=%.4f AUC-PR=%.4f Brier=%.4f best_iter=%d",
            model_name,
            fold,
            n_splits,
            fold_auc,
            fold_auc_pr,
            fold_brier,
            fold_best_iteration,
        )

    metrics = {
        "model_name": model_name,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_splits": int(n_splits),
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0),
        "mean_auc_pr": float(np.mean(auc_prs)),
        "std_auc_pr": float(np.std(auc_prs, ddof=1) if len(auc_prs) > 1 else 0.0),
        "mean_brier": float(np.mean(briers)),
        "std_brier": float(np.std(briers, ddof=1) if len(briers) > 1 else 0.0),
        "oof_predictions": oof_predictions,
        "trained_models": trained_models,
        "fold_auc": aucs,
        "fold_auc_pr": auc_prs,
        "fold_brier": briers,
        "best_iterations": best_iterations,
    }

    summary = _json_ready(metrics)
    metrics_path = METRICS_DIR / f"{model_name}_cv_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (METRICS_DIR / f"{model_name}_cv_models.pkl").open("wb") as f:
        pickle.dump(trained_models, f)

    pd.Series(oof_predictions, index=X.index, name="oof_prediction").to_csv(
        METRICS_DIR / f"{model_name}_oof_predictions.csv"
    )

    logging.info(
        "CV summary (%s): AUC %.4f±%.4f | AUC-PR %.4f±%.4f | Brier %.4f±%.4f",
        model_name,
        summary["mean_auc"],
        summary["std_auc"],
        summary["mean_auc_pr"],
        summary["std_auc_pr"],
        summary["mean_brier"],
        summary["std_brier"],
    )
    return metrics


def _train_final_xgboost(X: pd.DataFrame, y: pd.Series, model_name: str):
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    scale_pos_weight = (neg / max(pos, 1)) if pos > 0 else 1.0

    if XGB_AVAILABLE:
        model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            eval_metric="auc",
            n_jobs=-1,
        )
        model.fit(X, y, verbose=False)
    else:
        model = ExtraTreesClassifier(
            n_estimators=800,
            max_depth=None,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
            model.fit(X, y)

    with (METRICS_DIR / f"{model_name}_final_model.pkl").open("wb") as f:
        pickle.dump(model, f)

    return model


def compare_models(baseline_metrics: dict, full_metrics: dict) -> None:
    """Compare baseline and full model CV metrics and save report.

    Args:
        baseline_metrics: Metrics dict for baseline model.
        full_metrics: Metrics dict for full model.

    Returns:
        None
    """
    auc_delta = float(full_metrics["mean_auc"] - baseline_metrics["mean_auc"])
    auc_pr_delta = float(full_metrics["mean_auc_pr"] - baseline_metrics["mean_auc_pr"])
    brier_delta = float(full_metrics["mean_brier"] - baseline_metrics["mean_brier"])

    comparison = {
        "baseline": {
            "auc_roc": float(baseline_metrics["mean_auc"]),
            "auc_pr": float(baseline_metrics["mean_auc_pr"]),
            "brier": float(baseline_metrics["mean_brier"]),
        },
        "full": {
            "auc_roc": float(full_metrics["mean_auc"]),
            "auc_pr": float(full_metrics["mean_auc_pr"]),
            "brier": float(full_metrics["mean_brier"]),
        },
        "delta": {
            "auc_roc": auc_delta,
            "auc_pr": auc_pr_delta,
            "brier": brier_delta,
        },
        "tme_clinically_meaningful_auc_gain": bool(auc_delta > 0.02),
    }

    with (METRICS_DIR / "model_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    logging.info("Model comparison table:")
    logging.info("| Metric  | Baseline (no TME) | Full (with TME) | Delta |")
    logging.info("|---------|-------------------|-----------------|-------|")
    logging.info(
        "| AUC-ROC | %.4f | %.4f | %+0.4f |",
        baseline_metrics["mean_auc"],
        full_metrics["mean_auc"],
        auc_delta,
    )
    logging.info(
        "| AUC-PR  | %.4f | %.4f | %+0.4f |",
        baseline_metrics["mean_auc_pr"],
        full_metrics["mean_auc_pr"],
        auc_pr_delta,
    )
    logging.info(
        "| Brier   | %.4f | %.4f | %+0.4f |",
        baseline_metrics["mean_brier"],
        full_metrics["mean_brier"],
        brier_delta,
    )
    if auc_delta > 0.02:
        logging.info("TME effect is clinically meaningful (AUC delta > 0.02).")


def _bootstrap_c_index(duration: pd.Series, event: pd.Series, risk: pd.Series, n_boot: int = 1000) -> tuple[float, float]:
    if not LIFELINES_AVAILABLE:
        return float("nan"), float("nan")

    rng = np.random.default_rng(RANDOM_STATE)
    n = len(duration)
    ci_scores: list[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            c_idx = concordance_index(
                duration.iloc[idx],
                -risk.iloc[idx],
                event.iloc[idx],
            )
            if np.isfinite(c_idx):
                ci_scores.append(float(c_idx))
        except Exception:
            continue

    if not ci_scores:
        return float("nan"), float("nan")

    return float(np.percentile(ci_scores, 2.5)), float(np.percentile(ci_scores, 97.5))


def plot_km_curves(T: pd.Series, E: pd.Series, risk_groups: pd.Series) -> None:
    """Plot Kaplan-Meier curves for risk groups.

    Args:
        T: Duration in months.
        E: Event indicators.
        risk_groups: Risk group labels (`high` or `low`).

    Returns:
        None
    """
    if not LIFELINES_AVAILABLE:
        return

    SURVIVAL_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "serif", "axes.grid": False})
    fig, ax = plt.subplots(figsize=(8, 6))

    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()

    hi = risk_groups == "high"
    lo = risk_groups == "low"

    kmf_high.fit(T[hi], event_observed=E[hi], label="High risk")
    kmf_low.fit(T[lo], event_observed=E[lo], label="Low risk")

    kmf_high.plot_survival_function(ax=ax, ci_show=True, color="#b2182b", linewidth=2)
    kmf_low.plot_survival_function(ax=ax, ci_show=True, color="#2166ac", linewidth=2)

    med_hi = kmf_high.median_survival_time_
    med_lo = kmf_low.median_survival_time_
    if np.isfinite(med_hi):
        ax.axvline(med_hi, color="#b2182b", linestyle="--", linewidth=1)
    if np.isfinite(med_lo):
        ax.axvline(med_lo, color="#2166ac", linestyle="--", linewidth=1)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1)

    ax.set_title("Kaplan-Meier Curves by Predicted Risk Group")
    ax.set_xlabel("Disease-Free Survival (months)")
    ax.set_ylabel("Survival probability")
    ax.grid(False)

    add_at_risk_counts(kmf_high, kmf_low, ax=ax)
    fig.tight_layout()

    out_path = SURVIVAL_DIR / "km_curves.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def train_survival_model(X: pd.DataFrame, duration: pd.Series, event: pd.Series) -> dict:
    """Train Cox PH model and compute risk stratification performance.

    Args:
        X: Feature matrix.
        duration: DFS in months.
        event: Recurrence event indicator.

    Returns:
        Dictionary with C-index, CI bounds, and log-rank p-value.
    """
    SURVIVAL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if not LIFELINES_AVAILABLE:
        logging.warning("lifelines is unavailable (%s); skipping survival model.", str(LIFELINES_IMPORT_ERROR))
        metrics = {
            "c_index": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "logrank_p": float("nan"),
            "skipped": True,
            "reason": str(LIFELINES_IMPORT_ERROR),
        }
        with (METRICS_DIR / "survival_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        return metrics

    data = X.copy()
    data["DFS_months"] = pd.to_numeric(duration, errors="coerce")
    data["DFS_event"] = pd.to_numeric(event, errors="coerce")
    data = data.dropna(subset=["DFS_months", "DFS_event"])
    data["DFS_event"] = data["DFS_event"].astype(int)

    if data.empty:
        raise ValueError("No valid rows for survival modeling")

    X_model = data.drop(columns=["DFS_months", "DFS_event"])
    # Improve numerical stability: remove near-constant and highly collinear columns.
    variances = X_model.var(ddof=0)
    low_var_cols = variances[variances < 1e-6].index.tolist()
    if low_var_cols:
        X_model = X_model.drop(columns=low_var_cols)
        logging.info("Survival model dropped %d low-variance features", len(low_var_cols))

    if X_model.shape[1] > 1:
        corr = X_model.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        collinear_cols = [col for col in upper.columns if (upper[col] > 0.995).any()]
        if collinear_cols:
            X_model = X_model.drop(columns=collinear_cols)
            logging.info("Survival model dropped %d highly collinear features", len(collinear_cols))

    fit_data = X_model.copy()
    fit_data["DFS_months"] = data["DFS_months"]
    fit_data["DFS_event"] = data["DFS_event"]

    cph: CoxPHFitter | None = None
    last_exc: Exception | None = None
    for penalizer, l1_ratio in [(0.1, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 0.1), (5.0, 0.1)]:
        try:
            candidate = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
            candidate.fit(fit_data, duration_col="DFS_months", event_col="DFS_event", show_progress=False)
            cph = candidate
            logging.info("Survival Cox fit converged with penalizer=%.2f l1_ratio=%.2f", penalizer, l1_ratio)
            break
        except Exception as exc:  # pragma: no cover - data-dependent convergence path
            last_exc = exc
            logging.warning(
                "Survival Cox fit failed (penalizer=%.2f, l1=%.2f): %s",
                penalizer,
                l1_ratio,
                str(exc),
            )

    if cph is None:
        # Fallback risk score keeps pipeline runnable even under severe collinearity.
        logging.warning("Falling back to linear risk proxy for survival due to Cox convergence failure.")
        risk_scores = pd.Series(X_model.sum(axis=1).values, index=X_model.index, name="risk")
    else:
        risk_scores = pd.Series(
            cph.predict_partial_hazard(X_model).values.ravel(),
            index=X_model.index,
            name="risk",
        )

    T = data["DFS_months"]
    E = data["DFS_event"]

    c_index = float(concordance_index(T, -risk_scores, E))
    ci_lower, ci_upper = _bootstrap_c_index(T, E, risk_scores, n_boot=1000)

    median_risk = float(risk_scores.median())
    risk_groups = pd.Series(np.where(risk_scores >= median_risk, "high", "low"), index=risk_scores.index)

    hi = risk_groups == "high"
    lo = risk_groups == "low"
    lr = logrank_test(T[hi], T[lo], event_observed_A=E[hi], event_observed_B=E[lo])
    logrank_p = float(lr.p_value)

    plot_km_curves(T, E, risk_groups)

    metrics = {
        "c_index": c_index,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "logrank_p": logrank_p,
    }

    with (METRICS_DIR / "survival_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if cph is not None:
        with (METRICS_DIR / "cox_model.pkl").open("wb") as f:
            pickle.dump(cph, f)

    logging.info(
        "Survival model: C-index %.4f (95%% CI %.4f-%.4f), log-rank p=%.4g",
        c_index,
        ci_lower,
        ci_upper,
        logrank_p,
    )
    return metrics


def _load_ablation_variant(config_key: str) -> tuple[pd.DataFrame, pd.Series] | None:
    """Load features and labels for a single ablation variant.

    Args:
        config_key: Key from ABLATION_CONFIGS (e.g. 'A_baseline').

    Returns:
        Tuple of (X, y) or None if data unavailable.
    """
    config = ABLATION_CONFIGS[config_key]

    # Determine target labels
    if config["target"] == "real_pcr":
        geo_labels_path = PROCESSED_DIR / "geo_pcr_labels.csv"
        if geo_labels_path.exists():
            y_geo = pd.read_csv(geo_labels_path, index_col=0).iloc[:, 0].astype(int)
        else:
            y_geo = pd.Series(dtype=int)
    else:
        y_geo = pd.Series(dtype=int)

    # TCGA labels (always available)
    y_tcga_path = PROCESSED_DIR / "y_pcr.csv"
    if not y_tcga_path.exists():
        logging.warning("TCGA pCR labels not found for ablation variant %s", config_key)
        return None
    y_tcga = pd.read_csv(y_tcga_path, index_col=0).iloc[:, 0].astype(int)

    # Determine features
    X_baseline_path = PROCESSED_DIR / "X_baseline.csv"
    if not X_baseline_path.exists():
        logging.warning("Baseline features not found for ablation variant %s", config_key)
        return None
    X_baseline = pd.read_csv(X_baseline_path, index_col=0)

    if config["tme"] == "lm22":
        tme_path = PROCESSED_DIR / "tme_features.csv"
        if tme_path.exists():
            tme = pd.read_csv(tme_path, index_col=0)
            common = X_baseline.index.intersection(tme.index)
            X = X_baseline.loc[common].join(tme.loc[common])
        else:
            logging.warning("LM22 TME features not found; falling back to baseline for %s", config_key)
            X = X_baseline.copy()
    elif config["tme"] == "scrna":
        scrna_path = PROCESSED_DIR / "scrna_tme_features.csv"
        if scrna_path.exists():
            scrna = pd.read_csv(scrna_path, index_col=0)
            common = X_baseline.index.intersection(scrna.index)
            X = X_baseline.loc[common].join(scrna.loc[common])
        else:
            logging.warning("scRNA TME features not found; falling back to baseline for %s", config_key)
            X = X_baseline.copy()
    else:
        X = X_baseline.copy()

    # Select target
    if config["data"] == "tcga_plus_geo" and len(y_geo) > 0:
        # Combine TCGA and GEO labels for shared samples where possible
        y = pd.concat([y_tcga, y_geo])
        y = y[~y.index.duplicated(keep="first")]
    else:
        y = y_tcga

    # Align X and y
    common = X.index.intersection(y.index)
    if len(common) < 20:
        logging.warning("Only %d samples for ablation variant %s", len(common), config_key)
        return None

    X = X.loc[common]
    y = y.loc[common]

    # Ensure binary target with both classes
    if y.nunique() < 2:
        logging.warning("Single-class target for ablation variant %s", config_key)
        return None

    # Handle NaN
    X = X.fillna(0.0)

    return X, y


def run_ablation_study() -> dict:
    """Train and evaluate all 4 ablation model variants.

    Each variant uses 5-fold stratified CV with XGBoost.
    Reports AUC-ROC, AUC-PR, Brier for each, and computes
    deltas vs baseline at each step.

    Returns:
        Dict with all ablation metrics.
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}
    baseline_auc: float | None = None

    logging.info("=" * 60)
    logging.info("ABLATION STUDY — 4 Model Variants")
    logging.info("=" * 60)

    for config_key, config in ABLATION_CONFIGS.items():
        logging.info("\n--- Variant %s: %s ---", config_key, config["description"])

        data = _load_ablation_variant(config_key)
        if data is None:
            logging.warning("Skipping variant %s: data unavailable", config_key)
            all_results[config_key] = {
                "description": config["description"],
                "status": "skipped",
                "reason": "data_unavailable",
            }
            continue

        X, y = data
        try:
            metrics = train_xgboost_cv(X, y, model_name=f"ablation_{config_key}")
            auc = float(metrics["mean_auc"])
            auc_pr = float(metrics["mean_auc_pr"])
            brier = float(metrics["mean_brier"])

            delta = float(auc - baseline_auc) if baseline_auc is not None else 0.0
            if baseline_auc is None:
                baseline_auc = auc

            result = {
                "description": config["description"],
                "status": "ok",
                "n_samples": int(len(y)),
                "n_features": int(X.shape[1]),
                "mean_auc": auc,
                "std_auc": float(metrics["std_auc"]),
                "mean_auc_pr": auc_pr,
                "std_auc_pr": float(metrics["std_auc_pr"]),
                "mean_brier": brier,
                "std_brier": float(metrics["std_brier"]),
                "delta_auc_vs_baseline": delta,
            }
            all_results[config_key] = result

            logging.info(
                "Variant %s: AUC=%.4f (delta=%+.4f), AUC-PR=%.4f, Brier=%.4f, N=%d, F=%d",
                config_key, auc, delta, auc_pr, brier, len(y), X.shape[1],
            )

        except Exception as exc:
            logging.error("Ablation variant %s failed: %s", config_key, exc)
            all_results[config_key] = {
                "description": config["description"],
                "status": "error",
                "reason": str(exc),
            }

    # Print formatted comparison table
    logging.info("\n" + "=" * 80)
    logging.info("ABLATION STUDY RESULTS")
    logging.info("=" * 80)
    logging.info("| %-12s | %-40s | %7s | %7s | %7s | %7s |",
                  "Variant", "Description", "AUC", "±std", "Delta", "N")
    logging.info("|" + "-" * 14 + "|" + "-" * 42 + "|" + "-" * 9 + "|" + "-" * 9 + "|" + "-" * 9 + "|" + "-" * 9 + "|")
    for key, res in all_results.items():
        if res.get("status") == "ok":
            logging.info(
                "| %-12s | %-40s | %7.4f | %7.4f | %+7.4f | %7d |",
                key, res["description"][:40],
                res["mean_auc"], res["std_auc"],
                res["delta_auc_vs_baseline"], res["n_samples"],
            )
        else:
            logging.info("| %-12s | %-40s | %7s | %7s | %7s | %7s |",
                          key, res["description"][:40], "N/A", "N/A", "N/A", "N/A")

    # Critical checks
    logging.info("\n--- Critical Checks ---")
    a = all_results.get("A_baseline", {})
    b = all_results.get("B_real_pcr", {})
    c = all_results.get("C_tme_lm22", {})
    d = all_results.get("D_tme_scrna", {})

    if b.get("status") == "ok" and a.get("status") == "ok":
        ba_delta = b["mean_auc"] - a["mean_auc"]
        logging.info("B vs A (real pCR vs proxy): delta=%+.4f %s",
                      ba_delta, "PASS" if ba_delta > 0 else "FAIL")
    if c.get("status") == "ok" and b.get("status") == "ok":
        cb_delta = c["mean_auc"] - b["mean_auc"]
        logging.info("C vs B (LM22 TME added): delta=%+.4f %s",
                      cb_delta, "PASS" if cb_delta > 0 else "FAIL")
    if d.get("status") == "ok" and c.get("status") == "ok":
        dc_delta = d["mean_auc"] - c["mean_auc"]
        logging.info("D vs C (scRNA vs LM22): delta=%+.4f %s",
                      dc_delta, "PASS" if dc_delta > 0 else "FAIL")

    # Save results
    with (METRICS_DIR / "ablation_study.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logging.info("Ablation results saved to %s", METRICS_DIR / "ablation_study.json")

    return all_results


def _load_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.Series, pd.DataFrame]:
    X_baseline = pd.read_csv(PROCESSED_DIR / "X_baseline.csv", index_col=0)
    X_full = pd.read_csv(PROCESSED_DIR / "X_full.csv", index_col=0)
    X_tme_path = PROCESSED_DIR / "X_tme.csv"
    X_tme = pd.read_csv(X_tme_path, index_col=0) if X_tme_path.exists() else None
    y_pcr = pd.read_csv(PROCESSED_DIR / "y_pcr.csv", index_col=0).iloc[:, 0].astype(int)
    y_survival = pd.read_csv(PROCESSED_DIR / "y_survival.csv", index_col=0)

    common = X_baseline.index.intersection(X_full.index).intersection(y_pcr.index).intersection(y_survival.index)
    if X_tme is not None:
        common = common.intersection(X_tme.index)

    X_baseline = X_baseline.loc[common]
    X_full = X_full.loc[common]
    if X_tme is not None:
        X_tme = X_tme.loc[common]
    y_pcr = y_pcr.loc[common]
    y_survival = y_survival.loc[common]

    assert not X_baseline.isna().any().any()
    assert not X_full.isna().any().any()
    if X_tme is not None:
        assert not X_tme.isna().any().any()
    assert set(y_pcr.unique()) <= {0, 1}

    return X_baseline, X_full, X_tme, y_pcr, y_survival


def main() -> None:
    """CLI entrypoint for classification and survival modeling.

    Args:
        None

    Returns:
        None
    """
    configure_logging()

    parser = argparse.ArgumentParser(description="Train DrugResponce models")
    parser.add_argument("--mode", choices=["baseline", "full", "tme", "all"], default="all")
    parser.add_argument("--ablation", action="store_true", help="Run full ablation study (4 variants)")
    args = parser.parse_args()

    if args.ablation:
        run_ablation_study()
        if args.mode == "all":
            logging.info("Ablation complete; continuing with standard model training...")
        else:
            return

    X_baseline, X_full, X_tme, y_pcr, y_survival = _load_features()

    baseline_metrics: dict | None = None
    full_metrics: dict | None = None
    tme_metrics: dict | None = None

    if args.mode in {"baseline", "all"}:
        baseline_metrics = train_xgboost_cv(X_baseline, y_pcr, model_name="baseline")
        _train_final_xgboost(X_baseline, y_pcr, model_name="baseline")

    if args.mode in {"full", "all"}:
        full_metrics = train_xgboost_cv(X_full, y_pcr, model_name="full")
        _train_final_xgboost(X_full, y_pcr, model_name="full")

    if args.mode in {"tme", "all"}:
        if X_tme is None or X_tme.empty:
            raise FileNotFoundError(
                f"TME-only mode requires {PROCESSED_DIR / 'X_tme.csv'}. Run tme_deconvolution.py and "
                "feature_engineering.py --use-tme first."
            )
        tme_metrics = train_xgboost_cv(X_tme, y_pcr, model_name="tme")
        _train_final_xgboost(X_tme, y_pcr, model_name="tme")

    if args.mode == "full":
        train_survival_model(X_full, y_survival["DFS_months"], y_survival["DFS_event"])
    elif args.mode == "tme":
        assert X_tme is not None
        train_survival_model(X_tme, y_survival["DFS_months"], y_survival["DFS_event"])
    elif args.mode == "all":
        train_survival_model(X_full, y_survival["DFS_months"], y_survival["DFS_event"])

    if args.mode == "all" and baseline_metrics is not None and full_metrics is not None:
        compare_models(baseline_metrics, full_metrics)
    elif args.mode == "full" and full_metrics is not None:
        baseline_path = METRICS_DIR / "baseline_cv_metrics.json"
        if baseline_path.exists():
            with baseline_path.open("r", encoding="utf-8") as f:
                baseline_from_disk = json.load(f)
            compare_models(baseline_from_disk, full_metrics)
    elif args.mode == "tme" and tme_metrics is not None:
        baseline_path = METRICS_DIR / "baseline_cv_metrics.json"
        if baseline_path.exists():
            with baseline_path.open("r", encoding="utf-8") as f:
                baseline_from_disk = json.load(f)
            delta = float(tme_metrics["mean_auc"] - float(baseline_from_disk.get("mean_auc", np.nan)))
            logging.info(
                "TME-only vs baseline AUC delta: %+0.4f (TME=%.4f, baseline=%.4f)",
                delta,
                tme_metrics["mean_auc"],
                float(baseline_from_disk.get("mean_auc", np.nan)),
            )


if __name__ == "__main__":
    main()
