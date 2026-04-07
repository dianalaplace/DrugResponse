from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from data_loader import PROCESSED_DIR, RANDOM_STATE, configure_logging

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
                    "XGBoost unavailable (%s). Falling back to RandomForestClassifier.",
                    str(XGB_IMPORT_ERROR),
                )
            model = RandomForestClassifier(
                n_estimators=500,
                max_depth=8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )
            model.fit(X_train, y_train)

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
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
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

    data = X.copy()
    data["DFS_months"] = pd.to_numeric(duration, errors="coerce")
    data["DFS_event"] = pd.to_numeric(event, errors="coerce").astype(int)
    data = data.dropna(subset=["DFS_months", "DFS_event"])

    if data.empty:
        raise ValueError("No valid rows for survival modeling")

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(data, duration_col="DFS_months", event_col="DFS_event", show_progress=False)

    X_model = data.drop(columns=["DFS_months", "DFS_event"])
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


def _load_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    X_baseline = pd.read_csv(PROCESSED_DIR / "X_baseline.csv", index_col=0)
    X_full = pd.read_csv(PROCESSED_DIR / "X_full.csv", index_col=0)
    y_pcr = pd.read_csv(PROCESSED_DIR / "y_pcr.csv", index_col=0).iloc[:, 0].astype(int)
    y_survival = pd.read_csv(PROCESSED_DIR / "y_survival.csv", index_col=0)

    common = X_baseline.index.intersection(X_full.index).intersection(y_pcr.index).intersection(y_survival.index)
    X_baseline = X_baseline.loc[common]
    X_full = X_full.loc[common]
    y_pcr = y_pcr.loc[common]
    y_survival = y_survival.loc[common]

    assert not X_baseline.isna().any().any()
    assert not X_full.isna().any().any()
    assert set(y_pcr.unique()) <= {0, 1}

    return X_baseline, X_full, y_pcr, y_survival


def main() -> None:
    """CLI entrypoint for classification and survival modeling.

    Args:
        None

    Returns:
        None
    """
    configure_logging()

    parser = argparse.ArgumentParser(description="Train DrugResponce models")
    parser.add_argument("--mode", choices=["baseline", "full", "all"], default="all")
    args = parser.parse_args()

    X_baseline, X_full, y_pcr, y_survival = _load_features()

    baseline_metrics: dict | None = None
    full_metrics: dict | None = None

    if args.mode in {"baseline", "all"}:
        baseline_metrics = train_xgboost_cv(X_baseline, y_pcr, model_name="baseline")
        _train_final_xgboost(X_baseline, y_pcr, model_name="baseline")

    if args.mode in {"full", "all"}:
        full_metrics = train_xgboost_cv(X_full, y_pcr, model_name="full")
        _train_final_xgboost(X_full, y_pcr, model_name="full")
        train_survival_model(X_full, y_survival["DFS_months"], y_survival["DFS_event"])

    if args.mode == "all" and baseline_metrics is not None and full_metrics is not None:
        compare_models(baseline_metrics, full_metrics)
    elif args.mode == "full" and full_metrics is not None:
        baseline_path = METRICS_DIR / "baseline_cv_metrics.json"
        if baseline_path.exists():
            with baseline_path.open("r", encoding="utf-8") as f:
                baseline_from_disk = json.load(f)
            compare_models(baseline_from_disk, full_metrics)


if __name__ == "__main__":
    main()
