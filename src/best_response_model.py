from __future__ import annotations

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
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

try:
    import shap

    SHAP_AVAILABLE = True
except Exception as exc:  # pragma: no cover - runtime dependent
    shap = None  # type: ignore[assignment]
    SHAP_AVAILABLE = False
    SHAP_IMPORT_ERROR = exc

from src.data_loader import RANDOM_STATE
from src.publication_data import build_geo_training_dataset, load_ispy2_biomarker_external
from src.publication_features import compute_ispy2_proxy_train_features


def _prepare_openmp_runtime() -> None:
    candidates = [
        "/opt/miniconda3/lib/libomp.dylib",
        "/opt/miniconda3/envs/iml/lib/libomp.dylib",
    ]
    candidates.extend(
        sorted(glob.glob("/opt/miniconda3/pkgs/llvm-openmp-*/lib/libomp.dylib"), reverse=True)
    )
    for lib_path in candidates:
        if not Path(lib_path).exists():
            continue
        lib_dir = str(Path(lib_path).parent)
        current = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        paths = [item for item in current.split(":") if item]
        if lib_dir not in paths:
            os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join([lib_dir] + paths)
        try:
            ctypes.CDLL(lib_path)
            return
        except OSError:
            continue


_prepare_openmp_runtime()

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception as exc:  # pragma: no cover - runtime dependent
    XGBClassifier = None  # type: ignore[assignment]
    XGBOOST_AVAILABLE = False
    XGBOOST_IMPORT_ERROR = exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

HLA_APM_GENES = ["HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2", "PSMB8", "PSMB9"]
TME_SIGNATURES = {
    "B_cell": ["CD19", "MS4A1", "CD79A", "CD79B", "BLK", "BANK1"],
    "CD8_T_cell": ["CD8A", "CD8B", "GZMB", "PRF1", "IFNG", "NKG7"],
    "Dendritic": ["ITGAX", "CD1C", "FCER1A", "CLEC10A", "BATF3"],
    "M2_macrophage": ["CD163", "MRC1", "MSR1", "IL10", "CSF1R"],
    "CAF_proxy": ["ACTA2", "FAP", "PDGFRB", "COL1A1", "COL3A1", "FN1"],
    "TGFb_activity": ["TGFB1", "TGFB2", "SMAD2", "SMAD3", "SERPINE1"],
    "hypoxia_score": ["HIF1A", "VEGFA", "LDHA", "CA9", "SLC2A1"],
    "immune_exclusion": ["TGFB1", "WNT5A", "FGF2", "VEGFA", "IL10", "IDO1"],
}
CLINICAL_NUMERIC = ["age", "grade", "ER_status", "PR_status", "HER2_status", "HR_status"]

PARAM_GRID = [
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.7},
    {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.7},
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    include_genes: bool
    include_hla: bool
    include_tme: bool


MODEL_SPECS = [
    ModelSpec("A_clinical", "Clinical only", include_genes=False, include_hla=False, include_tme=False),
    ModelSpec("B_clinical_genes", "Clinical + variable genes", include_genes=True, include_hla=False, include_tme=False),
    ModelSpec("C_clinical_genes_hla", "Clinical + genes + HLA/APM", include_genes=True, include_hla=True, include_tme=False),
    ModelSpec(
        "D_clinical_genes_hla_tme",
        "Clinical + genes + HLA/APM + TME",
        include_genes=True,
        include_hla=True,
        include_tme=True,
    ),
]


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _save_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2)


def _all_tme_genes() -> list[str]:
    genes: set[str] = set()
    for values in TME_SIGNATURES.values():
        genes.update(gene.upper() for gene in values)
    return sorted(genes)


def bootstrap_metrics(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    n_bootstrap: int,
    random_state: int = RANDOM_STATE,
) -> dict[str, dict[str, float]]:
    y = np.asarray(y_true)
    score = np.asarray(y_score)
    rng = np.random.default_rng(random_state)

    auc_values: list[float] = []
    pr_values: list[float] = []
    brier_values: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), size=len(y))
        if np.unique(y[idx]).size < 2:
            continue
        auc_values.append(float(roc_auc_score(y[idx], score[idx])))
        pr_values.append(float(average_precision_score(y[idx], score[idx])))
        brier_values.append(float(brier_score_loss(y[idx], score[idx])))

    def _summ(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}
        return {
            "mean": float(np.mean(values)),
            "ci_lower": float(np.percentile(values, 2.5)),
            "ci_upper": float(np.percentile(values, 97.5)),
        }

    return {"roc_auc": _summ(auc_values), "pr_auc": _summ(pr_values), "brier": _summ(brier_values)}


def _make_model(params: dict[str, object], y_train: pd.Series) -> XGBClassifier:
    if not XGBOOST_AVAILABLE:
        raise RuntimeError(f"XGBoost unavailable: {XGBOOST_IMPORT_ERROR}")
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
        reg_alpha=1.0,
        reg_lambda=3.0,
        min_child_weight=5,
        gamma=0.0,
        scale_pos_weight=scale_pos_weight,
        **params,
    )


class GeoFeatureBuilder:
    def __init__(self, expression: pd.DataFrame, clinical: pd.DataFrame, top_n_genes: int) -> None:
        self.expression = expression  # genes x samples
        self.clinical = clinical
        self.top_n_genes = top_n_genes
        self.selected_genes: list[str] = []
        self.expr_means = pd.Series(dtype=float)
        self.expr_stds = pd.Series(dtype=float)
        self.hla_genes: list[str] = []
        self.numeric_medians = pd.Series(dtype=float)

    def fit(self, train_ids: pd.Index) -> "GeoFeatureBuilder":
        train_expr = self.expression.loc[:, train_ids]
        variances = train_expr.var(axis=1, ddof=0).sort_values(ascending=False)
        self.selected_genes = variances.head(self.top_n_genes).index.tolist()

        support = sorted(set(self.selected_genes) | set(HLA_APM_GENES) | set(_all_tme_genes()))
        support_expr = train_expr.reindex(support).fillna(0.0)
        self.expr_means = support_expr.mean(axis=1)
        self.expr_stds = support_expr.std(axis=1, ddof=0).replace(0.0, 1.0)
        self.hla_genes = [gene for gene in HLA_APM_GENES if gene in self.expression.index]
        self.numeric_medians = self.clinical.loc[train_ids, CLINICAL_NUMERIC].apply(pd.to_numeric, errors="coerce").median()
        return self

    def _zscore_expr(self, sample_ids: pd.Index, genes: list[str]) -> pd.DataFrame:
        frame = self.expression.reindex(genes).loc[:, sample_ids].fillna(0.0)
        z = frame.sub(self.expr_means.reindex(genes), axis=0).div(self.expr_stds.reindex(genes).replace(0.0, 1.0), axis=0)
        return z.T

    def _clinical_block(self, sample_ids: pd.Index) -> pd.DataFrame:
        block = self.clinical.loc[sample_ids, CLINICAL_NUMERIC].copy()
        for column in CLINICAL_NUMERIC:
            block[column] = pd.to_numeric(block[column], errors="coerce").fillna(self.numeric_medians[column])
        return block

    def _tme_block(self, sample_ids: pd.Index) -> pd.DataFrame:
        z = self._zscore_expr(sample_ids, _all_tme_genes()).T
        scores: dict[str, pd.Series] = {}
        for name, genes in TME_SIGNATURES.items():
            matched = [gene.upper() for gene in genes if gene.upper() in z.index]
            if matched:
                scores[f"TME::{name}"] = z.loc[matched].mean(axis=0)
            else:
                scores[f"TME::{name}"] = pd.Series(0.0, index=sample_ids)
        return pd.DataFrame(scores, index=sample_ids)

    def transform(self, sample_ids: pd.Index, spec: ModelSpec) -> pd.DataFrame:
        blocks = [self._clinical_block(sample_ids)]
        if spec.include_genes:
            gene_block = self._zscore_expr(sample_ids, self.selected_genes)
            gene_block.columns = [f"GENE::{column}" for column in gene_block.columns]
            blocks.append(gene_block)
        if spec.include_hla:
            hla_block = self._zscore_expr(sample_ids, self.hla_genes)
            hla_block.columns = [f"HLA::{column}" for column in hla_block.columns]
            hla_score = pd.Series(hla_block.mean(axis=1), index=sample_ids, name="HLA::immune_presentation_score")
            blocks.extend([hla_block, hla_score.to_frame()])
        if spec.include_tme:
            blocks.append(self._tme_block(sample_ids))
        return pd.concat(blocks, axis=1).fillna(0.0)

    def train_ispy2_proxy_features(self, sample_ids: pd.Index, spec: ModelSpec) -> pd.DataFrame:
        train_expr = self.expression.loc[:, sample_ids]
        tme_scores = self._tme_block(sample_ids).copy()
        tme_scores.columns = [column.split("::", 1)[1] for column in tme_scores.columns]
        return compute_ispy2_proxy_train_features(train_expr, self.clinical.loc[sample_ids], tme_scores)


def _inner_select_params(
    expression: pd.DataFrame,
    clinical: pd.DataFrame,
    train_ids: pd.Index,
    spec: ModelSpec,
    top_n_genes: int,
    inner_splits: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    y_train = clinical.loc[train_ids, "pCR"].astype(int)
    n_splits = max(2, min(inner_splits, int(y_train.value_counts().min())))
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    best_params = PARAM_GRID[0]
    best_score = float("-inf")
    trace: list[dict[str, object]] = []

    for params in PARAM_GRID:
        fold_aucs: list[float] = []
        for inner_train_pos, inner_valid_pos in inner_cv.split(train_ids, y_train):
            inner_train_ids = train_ids[inner_train_pos]
            inner_valid_ids = train_ids[inner_valid_pos]

            builder = GeoFeatureBuilder(expression, clinical, top_n_genes=top_n_genes).fit(inner_train_ids)
            x_train = builder.transform(inner_train_ids, spec)
            x_valid = builder.transform(inner_valid_ids, spec)
            y_inner_train = clinical.loc[inner_train_ids, "pCR"].astype(int)
            y_inner_valid = clinical.loc[inner_valid_ids, "pCR"].astype(int)

            model = _make_model(params, y_inner_train)
            model.fit(x_train, y_inner_train, verbose=False)
            preds = model.predict_proba(x_valid)[:, 1]
            fold_aucs.append(float(roc_auc_score(y_inner_valid, preds)))

        mean_auc = float(np.mean(fold_aucs))
        trace.append({"params": params, "mean_inner_auc": mean_auc, "fold_auc": fold_aucs})
        if mean_auc > best_score:
            best_score = mean_auc
            best_params = params

    return best_params, trace


def run_nested_cv(
    expression: pd.DataFrame,
    clinical: pd.DataFrame,
    top_n_genes: int,
    outer_splits: int,
    inner_splits: int,
    n_bootstrap: int,
    output_dir: Path,
) -> tuple[dict[str, object], dict[str, pd.Series], dict[str, list[dict[str, object]]]]:
    y = clinical["pCR"].astype(int)
    n_splits = max(2, min(outer_splits, int(y.value_counts().min())))
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    oof_predictions = {spec.name: pd.Series(np.nan, index=clinical.index, dtype=float) for spec in MODEL_SPECS}
    tuning_records: dict[str, list[dict[str, object]]] = {spec.name: [] for spec in MODEL_SPECS}
    fold_metrics: dict[str, list[dict[str, float]]] = {spec.name: [] for spec in MODEL_SPECS}

    for fold_idx, (train_pos, test_pos) in enumerate(outer_cv.split(clinical.index, y), start=1):
        train_ids = clinical.index[train_pos]
        test_ids = clinical.index[test_pos]
        y_train = clinical.loc[train_ids, "pCR"].astype(int)
        y_test = clinical.loc[test_ids, "pCR"].astype(int)

        for spec in MODEL_SPECS:
            best_params, trace = _inner_select_params(
                expression=expression,
                clinical=clinical,
                train_ids=train_ids,
                spec=spec,
                top_n_genes=top_n_genes,
                inner_splits=inner_splits,
            )
            tuning_records[spec.name].append({"outer_fold": fold_idx, "selected_params": best_params, "inner_trace": trace})

            builder = GeoFeatureBuilder(expression, clinical, top_n_genes=top_n_genes).fit(train_ids)
            x_train = builder.transform(train_ids, spec)
            x_test = builder.transform(test_ids, spec)
            model = _make_model(best_params, y_train)
            model.fit(x_train, y_train, verbose=False)
            preds = model.predict_proba(x_test)[:, 1]
            oof_predictions[spec.name].loc[test_ids] = preds

            fold_metric = {
                "outer_fold": fold_idx,
                "roc_auc": float(roc_auc_score(y_test, preds)),
                "pr_auc": float(average_precision_score(y_test, preds)),
                "brier": float(brier_score_loss(y_test, preds)),
            }
            fold_metrics[spec.name].append(fold_metric)
            logging.info(
                "%s | outer fold %d/%d | AUC=%.4f PR-AUC=%.4f Brier=%.4f",
                spec.name,
                fold_idx,
                n_splits,
                fold_metric["roc_auc"],
                fold_metric["pr_auc"],
                fold_metric["brier"],
            )

    summary: dict[str, object] = {}
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for spec in MODEL_SPECS:
        preds = oof_predictions[spec.name].dropna()
        y_valid = clinical.loc[preds.index, "pCR"].astype(int)
        summary[spec.name] = {
            "description": spec.description,
            "n_samples": int(len(preds)),
            "roc_auc": float(roc_auc_score(y_valid, preds)),
            "pr_auc": float(average_precision_score(y_valid, preds)),
            "brier": float(brier_score_loss(y_valid, preds)),
            "bootstrap": bootstrap_metrics(y_valid, preds, n_bootstrap=n_bootstrap),
            "fold_metrics": fold_metrics[spec.name],
        }
        preds.rename("oof_prediction").to_csv(metrics_dir / f"{spec.name}_oof.csv")

    _save_json(summary, metrics_dir / "nested_cv_summary.json")
    _save_json(tuning_records, metrics_dir / "nested_cv_tuning_trace.json")
    return summary, oof_predictions, tuning_records


def _params_mode(records: list[dict[str, object]]) -> dict[str, object]:
    counts: dict[str, int] = {}
    decoded: dict[str, dict[str, object]] = {}
    for record in records:
        params = record["selected_params"]
        key = json.dumps(params, sort_keys=True)
        counts[key] = counts.get(key, 0) + 1
        decoded[key] = params  # type: ignore[assignment]
    return decoded[max(counts.items(), key=lambda item: item[1])[0]]


def fit_final_models(
    expression: pd.DataFrame,
    clinical: pd.DataFrame,
    tuning_records: dict[str, list[dict[str, object]]],
    top_n_genes: int,
    output_dir: Path,
) -> tuple[dict[str, object], dict[str, pd.DataFrame], dict[str, GeoFeatureBuilder]]:
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    fitted_models: dict[str, object] = {}
    fitted_matrices: dict[str, pd.DataFrame] = {}
    builders: dict[str, GeoFeatureBuilder] = {}
    y = clinical["pCR"].astype(int)

    for spec in MODEL_SPECS:
        params = _params_mode(tuning_records[spec.name])
        builder = GeoFeatureBuilder(expression, clinical, top_n_genes=top_n_genes).fit(clinical.index)
        x_all = builder.transform(clinical.index, spec)
        model = _make_model(params, y)
        model.fit(x_all, y, verbose=False)

        fitted_models[spec.name] = model
        fitted_matrices[spec.name] = x_all
        builders[spec.name] = builder

        with (model_dir / f"{spec.name}.pkl").open("wb") as handle:
            pickle.dump(model, handle)
        x_all.to_csv(model_dir / f"{spec.name}_features.csv")

    return fitted_models, fitted_matrices, builders


def _subgroup_label(clinical: pd.DataFrame) -> pd.Series:
    hr = pd.to_numeric(clinical["HR_status"], errors="coerce").fillna(0.0)
    her2 = pd.to_numeric(clinical["HER2_status"], errors="coerce").fillna(0.0)

    labels = pd.Series("Other", index=clinical.index, dtype=object)
    labels.loc[(hr >= 0.5) & (her2 < 0.5)] = "HR+/HER2-"
    labels.loc[her2 >= 0.5] = "HER2+"
    labels.loc[(hr < 0.5) & (her2 < 0.5)] = "TNBC"
    return labels


def subgroup_analysis(clinical: pd.DataFrame, oof_predictions: dict[str, pd.Series], output_dir: Path) -> dict[str, object]:
    y = clinical["pCR"].astype(int)
    output: dict[str, object] = {"dataset": {}, "clinical_subtype": {}}

    for dataset_name, idx in clinical.groupby("dataset").groups.items():
        sample_ids = pd.Index(idx)
        entry: dict[str, object] = {"n_samples": int(len(sample_ids)), "pcr_rate": float(y.loc[sample_ids].mean())}
        for spec in MODEL_SPECS:
            preds = oof_predictions[spec.name].loc[sample_ids].dropna()
            if len(preds) < 30 or y.loc[preds.index].nunique() < 2:
                continue
            entry[spec.name] = {
                "roc_auc": float(roc_auc_score(y.loc[preds.index], preds)),
                "pr_auc": float(average_precision_score(y.loc[preds.index], preds)),
                "brier": float(brier_score_loss(y.loc[preds.index], preds)),
            }
        output["dataset"][str(dataset_name)] = entry

    subtype_labels = _subgroup_label(clinical)
    for subtype_name, idx in subtype_labels.groupby(subtype_labels).groups.items():
        sample_ids = pd.Index(idx)
        if len(sample_ids) < 40 or y.loc[sample_ids].nunique() < 2:
            continue
        entry = {"n_samples": int(len(sample_ids)), "pcr_rate": float(y.loc[sample_ids].mean())}
        for spec in MODEL_SPECS:
            preds = oof_predictions[spec.name].loc[sample_ids].dropna()
            if len(preds) < 30 or y.loc[preds.index].nunique() < 2:
                continue
            entry[spec.name] = {
                "roc_auc": float(roc_auc_score(y.loc[preds.index], preds)),
                "pr_auc": float(average_precision_score(y.loc[preds.index], preds)),
                "brier": float(brier_score_loss(y.loc[preds.index], preds)),
            }
        output["clinical_subtype"][str(subtype_name)] = entry

    _save_json(output, output_dir / "metrics" / "subgroup_analysis.json")
    return output


def run_shap(model_name: str, model: object, feature_matrix: pd.DataFrame, output_dir: Path, max_samples: int = 300) -> dict[str, object]:
    if not SHAP_AVAILABLE:
        payload = {"skipped": True, "reason": str(SHAP_IMPORT_ERROR)}
        _save_json(payload, output_dir / "shap" / f"{model_name}_shap.json")
        return payload

    sample = feature_matrix.sample(min(len(feature_matrix), max_samples), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if getattr(shap_values, "ndim", 0) == 3:
        shap_values = shap_values[:, :, 1]
    importance = pd.Series(np.abs(np.asarray(shap_values)).mean(axis=0), index=sample.columns).sort_values(ascending=False)
    payload = {
        "model": model_name,
        "top_features": [{"feature": feature, "mean_abs_shap": float(value)} for feature, value in importance.head(20).items()],
    }
    _save_json(payload, output_dir / "shap" / f"{model_name}_shap.json")
    return payload


def external_proxy_validation(
    expression: pd.DataFrame,
    clinical: pd.DataFrame,
    builder: GeoFeatureBuilder,
    output_dir: Path,
) -> dict[str, object]:
    try:
        external = load_ispy2_biomarker_external()
    except Exception as exc:
        payload = {"skipped": True, "reason": str(exc)}
        _save_json(payload, output_dir / "metrics" / "external_proxy_validation.json")
        return payload

    train_proxy = builder.train_ispy2_proxy_features(clinical.index, MODEL_SPECS[-1])
    feature_columns = [column for column in train_proxy.columns if column in external.columns]
    if not feature_columns:
        payload = {"skipped": True, "reason": "no_overlap_features"}
        _save_json(payload, output_dir / "metrics" / "external_proxy_validation.json")
        return payload

    x_train = train_proxy[feature_columns]
    y_train = clinical["pCR"].astype(int)
    x_external = external[feature_columns]
    y_external = external["pCR"].astype(int)

    params = PARAM_GRID[0]
    model = _make_model(params, y_train)
    model.fit(x_train, y_train, verbose=False)
    preds = model.predict_proba(x_external)[:, 1]

    payload: dict[str, object] = {
        "mode": "reduced_proxy_transfer",
        "features": feature_columns,
        "n_external": int(len(y_external)),
        "roc_auc": float(roc_auc_score(y_external, preds)),
        "pr_auc": float(average_precision_score(y_external, preds)),
        "brier": float(brier_score_loss(y_external, preds)),
        "bootstrap": bootstrap_metrics(y_external, preds, n_bootstrap=500),
    }

    if "arm" in external.columns:
        by_arm: dict[str, object] = {}
        external_scored = external.copy()
        external_scored["prediction"] = preds
        for arm_name, subset in external_scored.groupby("arm"):
            if subset["pCR"].nunique() < 2:
                continue
            by_arm[str(arm_name)] = {
                "n_samples": int(len(subset)),
                "roc_auc": float(roc_auc_score(subset["pCR"], subset["prediction"])),
                "pr_auc": float(average_precision_score(subset["pCR"], subset["prediction"])),
            }
        payload["by_arm"] = by_arm

    _save_json(payload, output_dir / "metrics" / "external_proxy_validation.json")
    return payload


def _choose_best_model(summary: dict[str, object]) -> str:
    return max(MODEL_SPECS, key=lambda spec: summary[spec.name]["roc_auc"]).name


def _interpret_feature(feature: str) -> str:
    if feature in {"age", "grade", "ER_status", "PR_status", "HER2_status", "HR_status"}:
        return "Established clinical covariate."
    if feature.startswith("HLA::"):
        return "Antigen-presentation feature probing immune visibility."
    if feature.startswith("TME::"):
        return "Microenvironment summary feature."
    if feature.startswith("GENE::CD19"):
        return "B-cell lineage or immune infiltration-associated signal."
    if feature.startswith("GENE::"):
        return "Tumor gene-level feature selected inside training folds."
    return "Model feature."


def write_report(
    summary: dict[str, object],
    best_model_name: str,
    shap_payload: dict[str, object],
    subgroup_payload: dict[str, object],
    external_payload: dict[str, object],
    output_dir: Path,
    top_n_genes: int,
) -> Path:
    performance_rows = []
    for spec in MODEL_SPECS:
        entry = summary[spec.name]
        performance_rows.append(
            {
                "model": spec.name,
                "description": spec.description,
                "roc_auc": entry["roc_auc"],
                "roc_auc_ci_lower": entry["bootstrap"]["roc_auc"]["ci_lower"],
                "roc_auc_ci_upper": entry["bootstrap"]["roc_auc"]["ci_upper"],
                "pr_auc": entry["pr_auc"],
                "brier": entry["brier"],
            }
        )
    pd.DataFrame(performance_rows).to_csv(output_dir / "metrics" / "performance_table.csv", index=False)

    baseline = summary["A_clinical"]["roc_auc"]
    genes = summary["B_clinical_genes"]["roc_auc"]
    hla = summary["C_clinical_genes_hla"]["roc_auc"]
    full = summary["D_clinical_genes_hla_tme"]["roc_auc"]

    lines = [
        "# Best Available Clinical-Response Model",
        "",
        "## Design Choice",
        "- Primary training task: real pCR on GEO neoadjuvant cohorts, not TCGA proxy response.",
        f"- Gene-first representation with top `{top_n_genes}` variable genes selected inside training folds only.",
        "- HLA/APM treated as a structured incremental block.",
        "- TME treated as a secondary biological block, not assumed to help.",
        "",
        "## Performance Comparison",
        "| Model | ROC-AUC | PR-AUC | Brier |",
        "|---|---:|---:|---:|",
    ]
    for row in performance_rows:
        lines.append(
            f"| {row['model']} | {row['roc_auc']:.3f} ({row['roc_auc_ci_lower']:.3f}-{row['roc_auc_ci_upper']:.3f}) | {row['pr_auc']:.3f} | {row['brier']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## What Changed",
            f"- Clinical -> Clinical+Genes: {genes - baseline:+.3f} ROC-AUC",
            f"- Clinical+Genes -> +HLA/APM: {hla - genes:+.3f} ROC-AUC",
            f"- Clinical+Genes+HLA -> +TME: {full - hla:+.3f} ROC-AUC",
            "",
            "## Recommended Main Model",
            f"- Selected model: `{best_model_name}` based on highest nested-CV ROC-AUC.",
        ]
    )

    if best_model_name == "B_clinical_genes":
        lines.append("- Interpretation: gene-level signal is useful, but HLA/TME do not add stable incremental value.")
    elif best_model_name == "C_clinical_genes_hla":
        lines.append("- Interpretation: HLA/APM adds enough incremental value to retain as a structured block.")
    elif best_model_name == "D_clinical_genes_hla_tme":
        lines.append("- Interpretation: full multilayer representation is retained, but the TME gain should still be treated as modest until externally validated.")
    else:
        lines.append("- Interpretation: clinical features remain the strongest and most stable signal under the current data regime.")

    lines.extend(["", "## Biological Drivers"])
    for rank, entry in enumerate(shap_payload.get("top_features", [])[:10], start=1):
        lines.append(
            f"{rank}. {entry['feature']} - mean |SHAP| {entry['mean_abs_shap']:.4f} - {_interpret_feature(entry['feature'])}"
        )

    lines.extend(["", "## Clinically Meaningful Subgroups"])
    for group_name, group_payload in subgroup_payload.get("clinical_subtype", {}).items():
        lines.append(f"- {group_name}: n={group_payload['n_samples']}, pCR rate={group_payload['pcr_rate']:.3f}")
        best_subgroup_model = None
        best_subgroup_auc = float("-inf")
        for spec in MODEL_SPECS:
            metrics = group_payload.get(spec.name)
            if isinstance(metrics, dict) and metrics["roc_auc"] > best_subgroup_auc:
                best_subgroup_auc = metrics["roc_auc"]
                best_subgroup_model = spec.name
        if best_subgroup_model is not None:
            lines.append(f"  Best model: {best_subgroup_model} (ROC-AUC {best_subgroup_auc:.3f})")

    lines.extend(["", "## External Validation Status"])
    if external_payload.get("skipped"):
        lines.append(f"- Full external validation unavailable: {external_payload.get('reason', 'unknown')}")
    else:
        lines.append("- Full gene-level transfer remains blocked by missing GPL20078 gene annotation in the local workspace.")
        lines.append(
            f"- Reduced proxy transfer model on I-SPY2: ROC-AUC {external_payload['roc_auc']:.3f}, PR-AUC {external_payload['pr_auc']:.3f}, Brier {external_payload['brier']:.3f}."
        )
        if "by_arm" in external_payload:
            for arm_name, entry in external_payload["by_arm"].items():
                lines.append(f"  {arm_name}: ROC-AUC {entry['roc_auc']:.3f}, PR-AUC {entry['pr_auc']:.3f}")

    lines.extend(
        [
            "",
            "## Conclusion",
            "- This is the strongest clinically oriented option available with the current local data.",
            "- Real response labels on GEO are more trustworthy than TCGA proxy response for model development.",
            "- External generalization of the full gene-first model is still not claimable without a matched RNA-seq or properly annotated gene-level holdout.",
        ]
    )

    report_path = output_dir / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_best_response_model(
    top_n_genes: int,
    outer_splits: int,
    inner_splits: int,
    n_bootstrap: int,
    output_dir: Path,
) -> dict[str, object]:
    expression, clinical, _batch = build_geo_training_dataset()
    if not XGBOOST_AVAILABLE:
        raise RuntimeError(f"XGBoost unavailable: {XGBOOST_IMPORT_ERROR}")

    summary, oof_predictions, tuning_records = run_nested_cv(
        expression=expression,
        clinical=clinical,
        top_n_genes=top_n_genes,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        n_bootstrap=n_bootstrap,
        output_dir=output_dir,
    )
    best_model_name = _choose_best_model(summary)
    fitted_models, fitted_matrices, builders = fit_final_models(
        expression=expression,
        clinical=clinical,
        tuning_records=tuning_records,
        top_n_genes=top_n_genes,
        output_dir=output_dir,
    )
    shap_payload = run_shap(best_model_name, fitted_models[best_model_name], fitted_matrices[best_model_name], output_dir)
    subgroup_payload = subgroup_analysis(clinical, oof_predictions, output_dir)
    external_payload = external_proxy_validation(expression, clinical, builders["D_clinical_genes_hla_tme"], output_dir)
    report_path = write_report(
        summary=summary,
        best_model_name=best_model_name,
        shap_payload=shap_payload,
        subgroup_payload=subgroup_payload,
        external_payload=external_payload,
        output_dir=output_dir,
        top_n_genes=top_n_genes,
    )

    manifest = {
        "dataset": "GEO real pCR cohorts",
        "n_samples": int(len(clinical)),
        "positive_rate": float(clinical["pCR"].mean()),
        "top_n_genes": int(top_n_genes),
        "outer_splits": int(outer_splits),
        "inner_splits": int(inner_splits),
        "bootstrap_iterations": int(n_bootstrap),
        "best_model": best_model_name,
        "report_path": str(report_path),
        "summary": summary,
    }
    _save_json(manifest, output_dir / "manifest.json")
    return manifest
