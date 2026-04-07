from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from data_loader import PROCESSED_DIR, configure_logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHAP_DIR = PROJECT_ROOT / "results" / "shap"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

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


def compute_shap_values(model, X: pd.DataFrame, model_name: str) -> np.ndarray:
    """Compute SHAP values for an XGBoost model.

    Args:
        model: Trained model.
        X: Feature matrix.
        model_name: Name for output filenames.

    Returns:
        SHAP values array.
    """
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_values = np.asarray(shap_values)

    out_path = SHAP_DIR / f"{model_name}_shap_values.npy"
    np.save(out_path, shap_values)
    logging.info("Saved SHAP matrix to %s", out_path)

    with (SHAP_DIR / f"{model_name}_expected_value.json").open("w", encoding="utf-8") as f:
        json.dump({"expected_value": float(np.ravel(explainer.expected_value)[0])}, f)

    return shap_values


def plot_global_importance(shap_values: np.ndarray, X: pd.DataFrame) -> None:
    """Plot top-20 global SHAP importance with TME-specific coloring.

    Args:
        shap_values: SHAP matrix.
        X: Feature matrix.

    Returns:
        None
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=X.columns).sort_values(ascending=False).head(20)

    colors = ["#0f766e" if feat in TME_FEATURES else "#6b7280" for feat in importance.index]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance.index[::-1], importance.values[::-1], color=colors[::-1])
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Global Feature Importance (Top 20)")

    from matplotlib.patches import Patch

    legend_items = [
        Patch(facecolor="#0f766e", label="TME features"),
        Patch(facecolor="#6b7280", label="Genomic/Clinical"),
    ]
    ax.legend(handles=legend_items)
    fig.tight_layout()
    fig.savefig(SHAP_DIR / "global_importance.png", dpi=300)
    plt.close(fig)


def plot_beeswarm(shap_values: np.ndarray, X: pd.DataFrame) -> None:
    """Create SHAP beeswarm plot.

    Args:
        shap_values: SHAP matrix.
        X: Feature matrix.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "beeswarm.png", dpi=300)
    plt.close()


def plot_patient_waterfall(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    explainer,
    patient_indices: list,
) -> None:
    """Plot SHAP waterfall charts for selected patients.

    Args:
        shap_values: SHAP matrix.
        X: Feature matrix.
        explainer: SHAP explainer with expected value and optional proba cache.
        patient_indices: List of row indices.

    Returns:
        None
    """
    base_value = float(np.ravel(explainer.expected_value)[0])
    probas = getattr(explainer, "_drugresponce_proba", None)

    for idx in patient_indices:
        row_values = shap_values[idx]
        row_data = X.iloc[idx].values
        patient_id = X.index[idx]

        pred_prob = float(probas[idx]) if probas is not None else float("nan")
        explanation = shap.Explanation(
            values=row_values,
            base_values=base_value,
            data=row_data,
            feature_names=list(X.columns),
        )

        plt.figure(figsize=(10, 5))
        shap.plots.waterfall(explanation, max_display=20, show=False)
        plt.title(f"Patient {patient_id} - predicted pCR probability: {pred_prob:.2f}")
        plt.tight_layout()
        plt.savefig(SHAP_DIR / f"waterfall_patient_{patient_id}.png", dpi=300)
        plt.close()


def tme_importance_report(shap_values: np.ndarray, X: pd.DataFrame) -> None:
    """Create TME-specific SHAP ranking report.

    Args:
        shap_values: SHAP matrix.
        X: Feature matrix.

    Returns:
        None
    """
    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns).sort_values(ascending=False)
    top10 = importance.head(10)

    tme_only = importance[importance.index.isin(TME_FEATURES)]
    tme_top10 = [f for f in top10.index if f in TME_FEATURES]

    lines = ["TME SHAP Importance Report", "==========================", "", "Top TME features:"]
    for rank, (feat, val) in enumerate(tme_only.head(14).items(), start=1):
        lines.append(f"{rank}. {feat}: {val:.6f}")

    lines.append("")
    lines.append(f"TME features in global top 10: {'YES' if tme_top10 else 'NO'}")
    lines.append(f"List: {', '.join(tme_top10) if tme_top10 else 'None'}")
    lines.append(f"M2_macrophage in top 10: {'YES' if 'M2_macrophage' in top10.index else 'NO'}")
    lines.append(f"TGFb_activity in top 10: {'YES' if 'TGFb_activity' in top10.index else 'NO'}")

    report_path = SHAP_DIR / "tme_importance_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _select_patient_indices(prob: np.ndarray) -> list[int]:
    highest = int(np.argmax(prob))
    lowest = int(np.argmin(prob))
    uncertain = int(np.argmin(np.abs(prob - 0.5)))
    return [highest, lowest, uncertain]


def main() -> None:
    """Run SHAP analyses for the full model.

    Args:
        None

    Returns:
        None
    """
    configure_logging()
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    X = pd.read_csv(PROCESSED_DIR / "X_full.csv", index_col=0)

    model_path = METRICS_DIR / "full_final_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model for SHAP: {model_path}")

    with model_path.open("rb") as f:
        model = pickle.load(f)

    shap_values = compute_shap_values(model, X, model_name="full")
    explainer = shap.TreeExplainer(model)

    probs = model.predict_proba(X)[:, 1]
    setattr(explainer, "_drugresponce_proba", probs)

    plot_global_importance(shap_values, X)
    plot_beeswarm(shap_values, X)
    patient_indices = _select_patient_indices(probs)
    plot_patient_waterfall(shap_values, X, explainer, patient_indices)
    tme_importance_report(shap_values, X)

    logging.info("Saved SHAP outputs to %s", SHAP_DIR)


if __name__ == "__main__":
    main()
