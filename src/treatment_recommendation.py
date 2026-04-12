from __future__ import annotations

import argparse
import ctypes
import glob
import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from data_loader import PROCESSED_DIR, configure_logging, normalize_patient_id

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
RECOMMEND_DIR = RESULTS_DIR / "recommendations"
TME_COLUMNS = [
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
]


@dataclass
class DrugRecommendation:
    name: str
    mechanism: str
    probability: float
    confidence: str
    reasons: list[str]


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


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _clip_prob(p: float) -> float:
    return float(np.clip(p, 0.02, 0.95))


def _zscore_value(value: float, population: pd.Series) -> float:
    std = float(population.std(ddof=0))
    if std == 0.0 or not np.isfinite(std):
        return 0.0
    return float((value - float(population.mean())) / std)


def _gene_z(expr: pd.DataFrame, patient_id: str, genes: list[str]) -> tuple[float, float]:
    vals: list[float] = []
    total = len(genes)
    for g in genes:
        if g not in expr.columns:
            continue
        val = float(expr.at[patient_id, g])
        vals.append(_zscore_value(val, expr[g]))
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(len(vals) / total)


def _confidence_label(score: float) -> str:
    if score >= 0.72:
        return "High"
    if score >= 0.52:
        return "Medium"
    return "Low"


def _combine_probability(base_prob: float, drug_score: float, eligible: bool) -> float:
    p = 0.20 + 0.45 * base_prob + 0.35 * drug_score
    p += 0.08 if eligible else -0.08
    return _clip_prob(p)


def _build_drug_scores_baseline(
    patient_id: str,
    base_prob: float,
    clinical: pd.DataFrame,
    expr: pd.DataFrame,
) -> list[DrugRecommendation]:
    er = float(clinical.at[patient_id, "ER_status"])
    pr = float(clinical.at[patient_id, "PR_status"])
    her2 = float(clinical.at[patient_id, "HER2_status"])

    ki67 = float(clinical.at[patient_id, "Ki67"])
    ki67_z = _zscore_value(ki67, clinical["Ki67"])

    erbb2_z, erbb2_cov = _gene_z(expr, patient_id, ["ERBB2"])
    pi3k_z, pi3k_cov = _gene_z(expr, patient_id, ["PIK3CA", "AKT1", "MTOR"])
    cdk_z, cdk_cov = _gene_z(expr, patient_id, ["CCND1", "CDK4", "CDK6", "E2F1"])
    parp_z, parp_cov = _gene_z(expr, patient_id, ["PARP1", "PARP2", "BRCA1", "BRCA2", "RAD51"])
    top2a_z, top2a_cov = _gene_z(expr, patient_id, ["TOP2A", "MKI67", "AURKA"])

    recs: list[DrugRecommendation] = []

    # 1) HER2-targeted
    elig = her2 >= 0.5
    drug_score = _sigmoid(1.4 * (1.0 if elig else -1.0) + 1.2 * erbb2_z + 0.2 * ki67_z)
    prob = _combine_probability(base_prob, drug_score, elig)
    conf = _confidence_label(0.45 * erbb2_cov + 0.35 * abs(prob - 0.5) * 2 + 0.20 * (1.0 if elig else 0.4))
    reasons = [
        f"HER2 status: {'positive' if elig else 'negative'}",
        f"ERBB2 expression z-score: {erbb2_z:+.2f}",
        f"Proliferation z-score: {ki67_z:+.2f}",
    ]
    recs.append(DrugRecommendation("Trastuzumab", "HER2 inhibitor", prob, conf, reasons))

    # 2) PI3K-targeted
    elig = (er >= 0.5) and (her2 < 0.5)
    drug_score = _sigmoid(1.0 * pi3k_z + 0.8 * (1.0 if er >= 0.5 else -1.0) + 0.2 * ki67_z)
    prob = _combine_probability(base_prob, drug_score, elig)
    conf = _confidence_label(0.45 * pi3k_cov + 0.35 * abs(prob - 0.5) * 2 + 0.20 * (1.0 if elig else 0.5))
    reasons = [
        f"Hormone/HER2 context: ER={int(er)}, HER2={int(her2)}",
        f"PI3K pathway score z: {pi3k_z:+.2f}",
        f"Proliferation z-score: {ki67_z:+.2f}",
    ]
    recs.append(DrugRecommendation("Alpelisib", "PI3K inhibitor", prob, conf, reasons))

    # 3) CDK4/6-targeted
    elig = (er >= 0.5) and (her2 < 0.5)
    drug_score = _sigmoid(0.9 * cdk_z + 0.7 * (1.0 if er >= 0.5 else -1.0) + 0.3 * ki67_z)
    prob = _combine_probability(base_prob, drug_score, elig)
    conf = _confidence_label(0.45 * cdk_cov + 0.35 * abs(prob - 0.5) * 2 + 0.20 * (1.0 if elig else 0.5))
    reasons = [
        f"CDK pathway score z: {cdk_z:+.2f}",
        f"Ki67 z-score: {ki67_z:+.2f}",
        f"ER/HER2 eligibility: {'met' if elig else 'not met'}",
    ]
    recs.append(DrugRecommendation("Palbociclib", "CDK4/6 inhibitor", prob, conf, reasons))

    # 4) PARP inhibitor proxy
    elig = parp_z > 0.0
    drug_score = _sigmoid(0.9 * parp_z + 0.2 * ki67_z)
    prob = _combine_probability(base_prob, drug_score, elig)
    conf = _confidence_label(0.45 * parp_cov + 0.35 * abs(prob - 0.5) * 2 + 0.20 * (1.0 if elig else 0.5))
    reasons = [
        f"PARP/HRD proxy score z: {parp_z:+.2f}",
        f"Proliferation z-score: {ki67_z:+.2f}",
        f"Eligibility signal: {'high' if elig else 'moderate'}",
    ]
    recs.append(DrugRecommendation("Olaparib", "PARP inhibitor", prob, conf, reasons))

    # 5) ADC/anti-proliferative targeted option
    elig = top2a_z > 0.0
    drug_score = _sigmoid(0.8 * top2a_z + 0.4 * ki67_z)
    prob = _combine_probability(base_prob, drug_score, elig)
    conf = _confidence_label(0.45 * top2a_cov + 0.35 * abs(prob - 0.5) * 2 + 0.20 * (1.0 if elig else 0.5))
    reasons = [
        f"TOP2A/proliferation score z: {top2a_z:+.2f}",
        f"Ki67 z-score: {ki67_z:+.2f}",
        f"Eligibility signal: {'high' if elig else 'moderate'}",
    ]
    recs.append(DrugRecommendation("Trastuzumab Deruxtecan", "HER2-directed ADC", prob, conf, reasons))

    # 6) Chemotherapy baseline option
    elig = True
    drug_score = _sigmoid(0.5 * ki67_z - 0.3 * er + 0.2 * top2a_z)
    prob = _combine_probability(base_prob, drug_score, elig)
    conf = _confidence_label(0.55 + 0.35 * abs(prob - 0.5) * 2)
    reasons = [
        f"Proliferation (Ki67 z): {ki67_z:+.2f}",
        f"TOP2A score z: {top2a_z:+.2f}",
        "General cytotoxic fallback option",
    ]
    recs.append(DrugRecommendation("Paclitaxel", "Chemotherapy", prob, conf, reasons))

    return sorted(recs, key=lambda r: r.probability, reverse=True)


def _tme_z(tme: pd.DataFrame, patient_id: str, col: str) -> float:
    if col not in tme.columns:
        return 0.0
    return _zscore_value(float(tme.at[patient_id, col]), tme[col])


def _build_drug_scores_tme(
    patient_id: str,
    base_prob: float,
    tme: pd.DataFrame,
) -> list[DrugRecommendation]:
    cd8 = _tme_z(tme, patient_id, "CD8_T_cell")
    nk = _tme_z(tme, patient_id, "NK_cell")
    dendritic = _tme_z(tme, patient_id, "Dendritic")
    m2 = _tme_z(tme, patient_id, "M2_macrophage")
    mdsc = _tme_z(tme, patient_id, "MDSC_proxy")
    caf = _tme_z(tme, patient_id, "CAF_proxy")
    tgfb = _tme_z(tme, patient_id, "TGFb_activity")
    cxcl1 = _tme_z(tme, patient_id, "CXCL1_MDSC_axis")
    hypoxia = _tme_z(tme, patient_id, "hypoxia_score")
    il6 = _tme_z(tme, patient_id, "IL6_STAT3_axis")
    excl = _tme_z(tme, patient_id, "immune_exclusion")
    m2_pol = _tme_z(tme, patient_id, "M2_polarization")

    inflamed = 0.45 * cd8 + 0.30 * nk + 0.25 * dendritic - 0.35 * excl - 0.25 * tgfb
    myeloid_suppression = 0.45 * m2 + 0.35 * mdsc + 0.20 * m2_pol
    stromal_barrier = 0.50 * tgfb + 0.30 * caf + 0.20 * excl
    angiogenic_stress = 0.60 * hypoxia + 0.40 * il6

    recs: list[DrugRecommendation] = []

    # 1) PD-1 inhibitor
    elig = inflamed > -0.2
    score = _sigmoid(1.2 * inflamed - 0.8 * stromal_barrier - 0.7 * myeloid_suppression)
    prob = _combine_probability(base_prob, score, elig)
    conf = _confidence_label(0.55 + 0.35 * abs(prob - 0.5) * 2)
    reasons = [
        f"Inflamed-TME index: {inflamed:+.2f}",
        f"Suppression index (M2/MDSC): {myeloid_suppression:+.2f}",
        f"Immune exclusion z-score: {excl:+.2f}",
    ]
    recs.append(DrugRecommendation("Pembrolizumab", "PD-1 inhibitor", prob, conf, reasons))

    # 2) CSF1R inhibitor (macrophage reprogramming)
    elig = myeloid_suppression > 0.0
    score = _sigmoid(1.3 * myeloid_suppression + 0.3 * stromal_barrier)
    prob = _combine_probability(base_prob, score, elig)
    conf = _confidence_label(0.55 + 0.35 * abs(prob - 0.5) * 2)
    reasons = [
        f"M2 macrophage z-score: {m2:+.2f}",
        f"MDSC proxy z-score: {mdsc:+.2f}",
        f"M2 polarization z-score: {m2_pol:+.2f}",
    ]
    recs.append(DrugRecommendation("Pexidartinib", "CSF1R inhibitor", prob, conf, reasons))

    # 3) CXCR1/2 inhibitor
    elig = cxcl1 > -0.1
    score = _sigmoid(1.1 * cxcl1 + 0.5 * mdsc + 0.2 * il6)
    prob = _combine_probability(base_prob, score, elig)
    conf = _confidence_label(0.55 + 0.35 * abs(prob - 0.5) * 2)
    reasons = [
        f"CXCL1-MDSC axis z-score: {cxcl1:+.2f}",
        f"MDSC proxy z-score: {mdsc:+.2f}",
        f"IL6-STAT3 axis z-score: {il6:+.2f}",
    ]
    recs.append(DrugRecommendation("Reparixin", "CXCR1/2 inhibitor", prob, conf, reasons))

    # 4) TGF-beta pathway inhibitor
    elig = stromal_barrier > 0.0
    score = _sigmoid(1.2 * tgfb + 0.7 * excl + 0.4 * caf)
    prob = _combine_probability(base_prob, score, elig)
    conf = _confidence_label(0.55 + 0.35 * abs(prob - 0.5) * 2)
    reasons = [
        f"TGF-beta activity z-score: {tgfb:+.2f}",
        f"CAF proxy z-score: {caf:+.2f}",
        f"Immune exclusion z-score: {excl:+.2f}",
    ]
    recs.append(DrugRecommendation("Galunisertib", "TGF-beta inhibitor", prob, conf, reasons))

    # 5) Anti-VEGF option
    elig = angiogenic_stress > -0.1
    score = _sigmoid(1.3 * angiogenic_stress + 0.3 * hypoxia)
    prob = _combine_probability(base_prob, score, elig)
    conf = _confidence_label(0.55 + 0.35 * abs(prob - 0.5) * 2)
    reasons = [
        f"Hypoxia score z: {hypoxia:+.2f}",
        f"IL6-STAT3 axis z-score: {il6:+.2f}",
        f"Angiogenic stress index: {angiogenic_stress:+.2f}",
    ]
    recs.append(DrugRecommendation("Bevacizumab", "VEGF inhibitor", prob, conf, reasons))

    # 6) Cytotoxic fallback
    elig = True
    score = _sigmoid(0.5 * angiogenic_stress - 0.2 * stromal_barrier)
    prob = _combine_probability(base_prob, score, elig)
    conf = _confidence_label(0.55 + 0.35 * abs(prob - 0.5) * 2)
    reasons = [
        f"Angiogenic stress index: {angiogenic_stress:+.2f}",
        f"Stromal barrier index: {stromal_barrier:+.2f}",
        "Fallback option when targeted TME effects are uncertain",
    ]
    recs.append(DrugRecommendation("Paclitaxel", "Chemotherapy", prob, conf, reasons))

    return sorted(recs, key=lambda r: r.probability, reverse=True)


def _render_markdown(
    patient_id: str,
    base_prob: float,
    recs: list[DrugRecommendation],
    model_name: str,
    top_k: int = 3,
) -> str:
    lines = [
        f"# DrugResponce Treatment Ranking - {patient_id}",
        "",
        f"Model type: {model_name}",
        f"Base model response probability: {base_prob * 100:.1f}%",
        "",
        "Recommended Treatments:",
        "",
    ]

    for idx, rec in enumerate(recs[:top_k], start=1):
        lines.extend(
            [
                f"{idx}. {rec.name} ({rec.mechanism})",
                f"   Probability of response: {rec.probability * 100:.0f}%",
                f"   Confidence: {rec.confidence}",
                f"   Rationale: {rec.reasons[0]}; {rec.reasons[1]}",
                "",
            ]
        )

    lines.append("Clinical note: ranking is model-assisted and must be reviewed at MTB.")
    return "\n".join(lines) + "\n"


def recommend_for_patient(patient_id: str, model_name: str = "baseline", top_k: int = 3) -> tuple[float, list[DrugRecommendation], str]:
    """Generate ranked treatment options for one patient.

    Args:
        patient_id: Patient identifier.
        model_name: Model variant (`baseline` or `tme`).
        top_k: Number of top drugs to include in report.

    Returns:
        Tuple of (base response probability, sorted recommendations, markdown report path).
    """
    pid = normalize_patient_id(patient_id)
    if model_name not in {"baseline", "tme"}:
        raise ValueError("model_name must be one of: baseline, tme")

    X_model = pd.read_csv(PROCESSED_DIR / f"X_{model_name}.csv", index_col=0)

    if pid not in X_model.index:
        raise KeyError(f"Patient {pid} is not present in processed dataset")

    _prepare_openmp_runtime()
    with (METRICS_DIR / f"{model_name}_final_model.pkl").open("rb") as f:
        model = pickle.load(f)

    base_prob = float(model.predict_proba(X_model.loc[[pid]])[:, 1][0])
    if model_name == "tme":
        tme = pd.read_csv(PROCESSED_DIR / "tme_features.csv", index_col=0)
        missing = [col for col in TME_COLUMNS if col not in tme.columns]
        if missing:
            raise KeyError(f"Missing TME columns for recommendation: {missing}")
        recs = _build_drug_scores_tme(pid, base_prob, tme)
    else:
        clinical = pd.read_csv(PROCESSED_DIR / "clinical_cleaned.csv", index_col=0)
        expr = pd.read_csv(PROCESSED_DIR / "rnaseq_processed.csv", index_col=0)
        recs = _build_drug_scores_baseline(pid, base_prob, clinical, expr)

    RECOMMEND_DIR.mkdir(parents=True, exist_ok=True)
    md_text = _render_markdown(pid, base_prob, recs, model_name=model_name, top_k=top_k)
    md_path = RECOMMEND_DIR / f"{pid}_{model_name}_recommendations.md"
    md_path.write_text(md_text, encoding="utf-8")

    json_payload = {
        "patient_id": pid,
        "model_name": model_name,
        "base_model_probability": base_prob,
        "recommendations": [
            {
                "drug": r.name,
                "mechanism": r.mechanism,
                "probability": r.probability,
                "confidence": r.confidence,
                "reasons": r.reasons,
            }
            for r in recs
        ],
    }
    with (RECOMMEND_DIR / f"{pid}_{model_name}_recommendations.json").open("w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    return base_prob, recs, str(md_path)


def main() -> None:
    """CLI entrypoint for patient-level treatment ranking."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Generate drug recommendations for a patient")
    parser.add_argument("--patient-id", default=None, help="Patient ID (TCGA-XX-XXXX style)")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model", choices=["baseline", "tme"], default="baseline")
    args = parser.parse_args()

    X_model = pd.read_csv(PROCESSED_DIR / f"X_{args.model}.csv", index_col=0)
    pid = normalize_patient_id(args.patient_id) if args.patient_id else str(X_model.index[0])

    base_prob, recs, out_path = recommend_for_patient(pid, model_name=args.model, top_k=args.top_k)
    logging.info("Recommendations saved to %s", out_path)
    logging.info("\\n%s", _render_markdown(pid, base_prob, recs, model_name=args.model, top_k=args.top_k))


if __name__ == "__main__":
    main()
