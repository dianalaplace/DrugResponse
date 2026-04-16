from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pycombat import Combat

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

try:
    import gseapy as gp

    GSEAPY_AVAILABLE = True
except Exception as exc:  # pragma: no cover - import is environment dependent
    gp = None  # type: ignore[assignment]
    GSEAPY_AVAILABLE = False
    GSEAPY_IMPORT_ERROR = exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GMT_PATH = PROJECT_ROOT / "signatures" / "tme_signatures.gmt"

CLINICAL_FEATURE_COLUMNS = [
    "age",
    "grade",
    "ER_status",
    "PR_status",
    "HER2_status",
    "HR_status",
]

T_CELL_GENES = ["CD3D", "CD3E", "CD2", "CD8A", "CD8B", "GZMB", "PRF1", "IFNG"]
STAT1_GENES = ["STAT1", "IRF1", "CXCL9", "CXCL10", "IDO1"]
TIS_GENES = ["IFNG", "CXCL9", "CXCL10", "IDO1", "HLA-DRA", "STAT1", "CD8A"]


def combat_fit_transform(train_df: pd.DataFrame, batch_train: pd.Series) -> tuple[pd.DataFrame, Combat]:
    combat = Combat()
    corrected = combat.fit_transform(train_df.T.values, batch_train.reindex(train_df.columns).to_numpy())
    corrected_df = pd.DataFrame(corrected.T, index=train_df.index, columns=train_df.columns)
    return corrected_df, combat


def combat_transform(test_df: pd.DataFrame, combat_model: Combat, batch_test: pd.Series) -> pd.DataFrame:
    corrected = combat_model.transform(test_df.T.values, batch_test.reindex(test_df.columns).to_numpy())
    return pd.DataFrame(corrected.T, index=test_df.index, columns=test_df.columns)


def _load_gmt(gmt_file: Path | str) -> dict[str, list[str]]:
    signatures: dict[str, list[str]] = {}
    with Path(gmt_file).open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            signatures[parts[0]] = [gene.upper() for gene in parts[2:] if gene]
    return signatures


def _zscore_rows(expression_df: pd.DataFrame) -> pd.DataFrame:
    centered = expression_df.sub(expression_df.mean(axis=1), axis=0)
    scaled = centered.div(expression_df.std(axis=1, ddof=0).replace(0.0, np.nan), axis=0)
    return scaled.fillna(0.0)


def _compute_zscore_fallback(expression_df: pd.DataFrame, gmt_file: Path | str) -> pd.DataFrame:
    signatures = _load_gmt(gmt_file)
    z_expr = _zscore_rows(expression_df)
    scores: dict[str, pd.Series] = {}
    for name, genes in signatures.items():
        matched = [gene for gene in genes if gene in z_expr.index]
        if matched:
            scores[name] = z_expr.loc[matched].mean(axis=0)
        else:
            scores[name] = pd.Series(0.0, index=expression_df.columns)
    return pd.DataFrame(scores, index=expression_df.columns)


def compute_ssgsea(expression_df: pd.DataFrame, gmt_file: Path | str = DEFAULT_GMT_PATH) -> pd.DataFrame:
    if not GSEAPY_AVAILABLE:
        logging.warning("gseapy unavailable (%s); using z-score fallback", str(GSEAPY_IMPORT_ERROR))
        return _compute_zscore_fallback(expression_df, gmt_file)

    try:
        result = gp.ssgsea(
            data=expression_df,
            gene_sets=str(gmt_file),
            sample_norm_method="rank",
            outdir=None,
            permutation_num=0,
            no_plot=True,
            min_size=1,
            threads=1,
            verbose=False,
        )
        metric = "ES" if "ES" in result.res2d.columns else "NES"
        scores = result.res2d.pivot(index="Term", columns="Name", values=metric)
        return scores.T
    except Exception as exc:  # pragma: no cover - depends on gseapy runtime
        logging.warning("ssGSEA failed (%s); using z-score fallback", exc)
        return _compute_zscore_fallback(expression_df, gmt_file)


def clinical_feature_frame(clinical_df: pd.DataFrame) -> pd.DataFrame:
    features = clinical_df.reindex(columns=CLINICAL_FEATURE_COLUMNS).copy()
    return features.apply(pd.to_numeric, errors="coerce")


def merge_feature_blocks(tme_scores: pd.DataFrame, clinical_df: pd.DataFrame) -> pd.DataFrame:
    return tme_scores.join(clinical_feature_frame(clinical_df), how="left")


def _gene_series(expression_df: pd.DataFrame, gene_symbol: str) -> pd.Series:
    gene = gene_symbol.upper()
    if gene in expression_df.index:
        return pd.to_numeric(expression_df.loc[gene], errors="coerce")
    return pd.Series(0.0, index=expression_df.columns)


def _signature_mean(z_expr: pd.DataFrame, genes: list[str]) -> pd.Series:
    matched = [gene.upper() for gene in genes if gene.upper() in z_expr.index]
    if not matched:
        return pd.Series(0.0, index=z_expr.columns)
    return z_expr.loc[matched].mean(axis=0)


def compute_ispy2_proxy_train_features(
    expression_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    tme_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    z_expr = _zscore_rows(expression_df)
    if tme_scores is None:
        tme_scores = pd.DataFrame(index=expression_df.columns)

    features = pd.DataFrame(index=expression_df.columns)
    features["HR_status"] = pd.to_numeric(clinical_df.reindex(expression_df.columns)["HR_status"], errors="coerce")
    features["HER2_status"] = pd.to_numeric(clinical_df.reindex(expression_df.columns)["HER2_status"], errors="coerce")
    features["B_cell"] = tme_scores.reindex(expression_df.columns).get("B_cell", pd.Series(0.0, index=expression_df.columns))
    features["Dendritic"] = tme_scores.reindex(expression_df.columns).get(
        "Dendritic", pd.Series(0.0, index=expression_df.columns)
    )
    features["T_cells_sig"] = _signature_mean(z_expr, T_CELL_GENES)
    features["CD68"] = _gene_series(z_expr, "CD68")
    features["PD1"] = _gene_series(z_expr, "PDCD1")
    features["PDL1"] = _gene_series(z_expr, "CD274")
    features["STAT1_sig"] = _signature_mean(z_expr, STAT1_GENES)
    features["TIS_sig"] = _signature_mean(z_expr, TIS_GENES)
    return features
