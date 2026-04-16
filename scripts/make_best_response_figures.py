"""Generate publication-grade figures for the B_clinical_genes response model.

All figures are saved under results/best_response_model_run/figures/.
They rely only on artifacts already produced by the pipeline so the script
can be re-run without re-training: the out-of-fold (OOF) predictions, the
cached clinical feature block, the SHAP payload, the nested-CV summary and
the I-SPY2 proxy validation file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path("/Users/dianalysenko/Documents/DrugResponce/results/best_response_model_run")
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

MODELS = ["A_clinical", "B_clinical_genes", "C_clinical_genes_hla", "D_clinical_genes_hla_tme"]
PRETTY = {
    "A_clinical": "A. Clinical",
    "B_clinical_genes": "B. Clinical + Genes (primary)",
    "C_clinical_genes_hla": "C. + HLA/APM",
    "D_clinical_genes_hla_tme": "D. + HLA + TME",
}
COLORS = {
    "A_clinical": "#7f7f7f",
    "B_clinical_genes": "#1f77b4",
    "C_clinical_genes_hla": "#2ca02c",
    "D_clinical_genes_hla_tme": "#d62728",
}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})


def load_predictions() -> pd.DataFrame:
    frames = []
    for model in MODELS:
        df = pd.read_csv(ROOT / "metrics" / f"{model}_oof.csv", index_col=0)
        df = df.rename(columns={"oof_prediction": model})
        frames.append(df)
    preds = pd.concat(frames, axis=1)
    # attach clinical block + derive subtype
    clin = pd.read_csv(ROOT / "models" / "A_clinical_features.csv", index_col=0)
    merged = preds.join(clin, how="inner")
    merged["dataset"] = merged.index.to_series().str.split(":").str[0]
    # load y labels from the richest feature set (any of them has clinical+label link)
    # y not stored directly; recover by using pCR from manifest-independent logic:
    # reconstruct via the subgroup_analysis which contains per-subgroup counts.
    # Simpler: derive y from the nested-CV tuning trace predictions. Not enough -> reuse the
    # subgroup file's pCR rate. We'll instead reconstruct y by joining on
    # data/processed GEO label file if present; otherwise infer from threshold.
    # Fallback: read y from the raw clinical data produced by build_geo_training_dataset.
    return merged


def load_labels(sample_ids: pd.Index) -> pd.Series:
    # The pipeline stores no standalone y file; reconstruct via fold-merged true labels
    # saved alongside OOF predictions? The CSV only has oof_prediction + sample id.
    # We reload from the GEO data module to get y.
    from src.publication_data import build_geo_training_dataset  # lazy import

    _, clin, _ = build_geo_training_dataset()
    y = clin["pCR"].astype(int)
    y = y.reindex(sample_ids)
    return y


def subtype(row: pd.Series) -> str:
    her2 = row.get("HER2_status")
    er = row.get("ER_status")
    pr = row.get("PR_status")
    hr = row.get("HR_status")
    if her2 == 1.0:
        return "HER2+"
    if (hr == 1.0) or (er == 1.0) or (pr == 1.0):
        return "HR+/HER2-"
    if (her2 == 0.0) and (hr == 0.0 or (er == 0.0 and pr == 0.0)):
        return "TNBC"
    return "Unknown"


def figure_performance_overview(preds: pd.DataFrame, y: pd.Series) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # Panel A - ROC
    ax = axes[0]
    for model in MODELS:
        fpr, tpr, _ = roc_curve(y, preds[model])
        auc = roc_auc_score(y, preds[model])
        ax.plot(fpr, tpr, color=COLORS[model], lw=2,
                label=f"{PRETTY[model]} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("A. Receiver Operating Characteristic")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Panel B - PR
    ax = axes[1]
    baseline = float(y.mean())
    for model in MODELS:
        prec, rec, _ = precision_recall_curve(y, preds[model])
        ap = average_precision_score(y, preds[model])
        ax.plot(rec, prec, color=COLORS[model], lw=2,
                label=f"{PRETTY[model]} (AP={ap:.3f})")
    ax.hlines(baseline, 0, 1, ls="--", color="#999999", lw=1,
              label=f"Prevalence = {baseline:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("B. Precision-Recall")
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Panel C - Calibration (primary model only)
    ax = axes[2]
    prob_true, prob_pred = calibration_curve(y, preds["B_clinical_genes"], n_bins=10, strategy="quantile")
    ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=1, label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, "o-", color=COLORS["B_clinical_genes"], lw=2,
            label="B. Clinical + Genes")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed pCR rate")
    ax.set_title("C. Calibration (primary model)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_performance_overview.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_performance_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_shap_bar() -> None:
    shap_payload = json.loads((ROOT / "shap" / "B_clinical_genes_shap.json").read_text())
    top = shap_payload["top_features"][:15]
    names = [entry["feature"].replace("GENE::", "") for entry in top][::-1]
    values = [entry["mean_abs_shap"] for entry in top][::-1]

    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    bars = ax.barh(names, values, color="#1f77b4", edgecolor="white")
    for bar, value in zip(bars, values):
        ax.text(value + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}", va="center", fontsize=8.5)
    ax.set_xlabel("Mean |SHAP value| (primary model B)")
    ax.set_title("Top 15 gene-level drivers of pCR prediction")
    ax.set_xlim(0, max(values) * 1.15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_shap_top_features.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig2_shap_top_features.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_subgroup_performance() -> None:
    payload = json.loads((ROOT / "metrics" / "subgroup_analysis.json").read_text())
    subgroups = payload["clinical_subtype"]

    order = ["HER2+", "HR+/HER2-", "TNBC"]
    xs = np.arange(len(order))
    width = 0.20

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for i, model in enumerate(MODELS):
        vals = [subgroups[g][model]["roc_auc"] for g in order]
        ax.bar(xs + (i - 1.5) * width, vals, width,
               label=PRETTY[model], color=COLORS[model], edgecolor="white")
        for x, v in zip(xs + (i - 1.5) * width, vals):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)

    # Annotate n per subgroup
    labels = [f"{g}\n(n={subgroups[g]['n_samples']}, pCR={subgroups[g]['pcr_rate']:.2f})" for g in order]
    ax.set_xticks(xs, labels)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Subgroup performance by molecular subtype")
    ax.set_ylim(0.5, 0.9)
    ax.axhline(0.5, ls="--", color="#cccccc", lw=1)
    ax.legend(loc="upper right", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_subgroup_performance.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_subgroup_performance.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_decision_curve(preds: pd.DataFrame, y: pd.Series) -> None:
    thresholds = np.linspace(0.01, 0.6, 60)
    n = len(y)
    prev = float(y.mean())

    def net_benefit(scores: np.ndarray, y_arr: np.ndarray, thr: float) -> float:
        chosen = scores >= thr
        tp = np.sum((chosen) & (y_arr == 1))
        fp = np.sum((chosen) & (y_arr == 0))
        if n == 0:
            return 0.0
        return (tp / n) - (fp / n) * (thr / (1 - thr))

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    y_arr = y.values

    for model in MODELS:
        scores = preds[model].values
        nb = [net_benefit(scores, y_arr, t) for t in thresholds]
        ax.plot(thresholds, nb, lw=2, color=COLORS[model], label=PRETTY[model])

    # Treat-all / treat-none lines
    treat_all = [prev - (1 - prev) * (t / (1 - t)) for t in thresholds]
    ax.plot(thresholds, treat_all, color="#cccccc", lw=1.5, ls="-", label="Treat all")
    ax.axhline(0, color="#888888", lw=1.0, ls="--", label="Treat none")

    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision Curve Analysis (nested-CV OOF predictions)")
    ax.set_ylim(-0.04, max(0.2, prev + 0.02))
    ax.set_xlim(0.01, 0.6)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_decision_curve.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig4_decision_curve.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_external_validation() -> None:
    payload = json.loads((ROOT / "metrics" / "external_proxy_validation.json").read_text())
    by_arm = payload.get("by_arm", {})
    if not by_arm:
        return

    arms = list(by_arm.keys())
    aucs = [by_arm[a]["roc_auc"] for a in arms]
    praucs = [by_arm[a]["pr_auc"] for a in arms]
    ns = [by_arm[a]["n_samples"] for a in arms]

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    xs = np.arange(len(arms))
    width = 0.35
    ax.bar(xs - width / 2, aucs, width, color="#1f77b4", label="ROC-AUC")
    ax.bar(xs + width / 2, praucs, width, color="#ff7f0e", label="PR-AUC")
    for x, v in zip(xs - width / 2, aucs):
        ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    for x, v in zip(xs + width / 2, praucs):
        ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)

    ax.axhline(payload["roc_auc"], ls="--", color="#1f77b4", lw=1,
               label=f"Pooled ROC-AUC={payload['roc_auc']:.2f}")
    ax.set_xticks(xs, [f"{a}\n(n={n})" for a, n in zip(arms, ns)])
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Discrimination")
    ax.set_title("External proxy validation on I-SPY2 (reduced-feature transfer)")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_external_validation.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig5_external_validation.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    preds = load_predictions()
    y = load_labels(preds.index)
    assert y.notna().all(), "missing labels in reconstructed y series"

    figure_performance_overview(preds, y)
    figure_shap_bar()
    figure_subgroup_performance()
    figure_decision_curve(preds, y)
    figure_external_validation()

    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
