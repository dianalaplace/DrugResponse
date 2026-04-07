from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from data_loader import PROCESSED_DIR, RNASEQ_FILE, configure_logging, load_rnaseq

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LM22_GENES: dict[str, list[str]] = {
    "M2_macrophage": ["CD163", "MRC1", "MSR1", "FOLR2", "IL10"],
    "CD8_T_cell": ["CD8A", "CD8B", "GZMB", "PRF1", "IFNG"],
    "MDSC_proxy": ["ARG1", "S100A8", "S100A9", "CXCR2", "ITGAM"],
    "CAF_proxy": ["ACTA2", "FAP", "PDGFRA", "PDGFRB", "COL1A1"],
    "NK_cell": ["NCAM1", "NKG7", "GNLY", "KLRB1", "KLRD1"],
    "B_cell": ["CD19", "MS4A1", "CD79A", "PAX5", "BLK"],
    "Treg": ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TNFRSF18"],
    "Dendritic": ["ITGAX", "HLA-DRA", "CD1C", "CLEC9A", "XCR1"],
}

TME_SIGNATURES: dict[str, list[str]] = {
    "TGFb_activity": ["TGFB1", "TGFB2", "SMAD2", "SMAD3", "ACTA2", "COL1A1"],
    "CXCL1_MDSC_axis": ["CXCL1", "CXCL2", "CXCL5", "CXCR2", "S100A8"],
    "hypoxia_score": ["HIF1A", "VEGFA", "LDHA", "CA9", "SLC2A1", "BNIP3"],
    "IL6_STAT3_axis": ["IL6", "STAT3", "JAK1", "JAK2", "SOCS1", "SOCS3"],
    "immune_exclusion": ["TGFB1", "WNT5A", "FGF2", "VEGFA", "IL10", "IDO1"],
    "M2_polarization": ["IL10", "ARG1", "MRC1", "CCL22", "CCL17", "CD163"],
}


def _sanitize_gene_columns(rnaseq: pd.DataFrame) -> pd.DataFrame:
    cols = rnaseq.columns.astype(str)
    parsed: list[str] = []
    for c in cols:
        token = c
        if "|" in token:
            parts = token.split("|")
            if len(parts) > 1 and parts[1] not in {"", "NA", "nan"}:
                token = parts[1]
            else:
                token = parts[0]
        token = token.split(".")[0]
        parsed.append(token.upper())
    gene_symbols = pd.Index(parsed)
    out = rnaseq.copy()
    out.columns = gene_symbols
    out = out.loc[:, ~out.columns.duplicated()].copy()
    return out


def _zscore_columns(df: pd.DataFrame) -> pd.DataFrame:
    std = df.std(axis=0, ddof=0).replace(0.0, np.nan)
    z = (df - df.mean(axis=0)) / std
    return z.fillna(0.0)


def compute_cell_fractions(rnaseq: pd.DataFrame) -> pd.DataFrame:
    """Estimate TME cell fractions from signature genes.

    Args:
        rnaseq: Samples x genes expression matrix.

    Returns:
        DataFrame with patients x LM22 cell-type fractions.
    """
    expr = _sanitize_gene_columns(rnaseq)
    z_expr = _zscore_columns(expr)

    cell_scores: dict[str, pd.Series] = {}
    for cell_type, genes in LM22_GENES.items():
        available = [g.upper() for g in genes if g.upper() in z_expr.columns]
        if not available:
            logging.warning("No signature genes found for %s; assigning zeros", cell_type)
            cell_scores[cell_type] = pd.Series(0.0, index=z_expr.index)
            continue
        if len(available) < len(genes):
            logging.info(
                "%s: using %d/%d signature genes", cell_type, len(available), len(genes)
            )
        cell_scores[cell_type] = z_expr[available].mean(axis=1)

    raw_scores = pd.DataFrame(cell_scores, index=rnaseq.index)
    fractions = pd.DataFrame(
        softmax(raw_scores.to_numpy(), axis=1),
        index=raw_scores.index,
        columns=raw_scores.columns,
    )

    assert np.allclose(fractions.sum(axis=1).values, 1.0, atol=1e-6), "Fractions must sum to 1"
    return fractions


def compute_tme_signatures(rnaseq: pd.DataFrame) -> pd.DataFrame:
    """Compute TME process signatures from expression matrix.

    Args:
        rnaseq: Samples x genes expression matrix.

    Returns:
        DataFrame with patients x TME signatures.
    """
    expr = _sanitize_gene_columns(rnaseq)
    z_expr = _zscore_columns(expr)

    signature_scores: dict[str, pd.Series] = {}
    for signature, genes in TME_SIGNATURES.items():
        available = [g.upper() for g in genes if g.upper() in z_expr.columns]
        if not available:
            logging.warning("No genes found for signature %s; assigning zeros", signature)
            signature_scores[signature] = pd.Series(0.0, index=z_expr.index)
            continue
        signature_scores[signature] = z_expr[available].mean(axis=1)

    return pd.DataFrame(signature_scores, index=rnaseq.index)


def build_tme_feature_matrix(rnaseq: pd.DataFrame) -> pd.DataFrame:
    """Build combined TME feature matrix and save it to disk.

    Args:
        rnaseq: Samples x genes expression matrix.

    Returns:
        DataFrame with patients x TME features.
    """
    cell_fractions = compute_cell_fractions(rnaseq)
    signatures = compute_tme_signatures(rnaseq)

    tme = pd.concat([cell_fractions, signatures], axis=1)
    assert not tme.isna().any().any(), "TME feature matrix contains NaN"

    out_path = PROCESSED_DIR / "tme_features.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tme.to_csv(out_path)
    logging.info("Saved TME features to %s (%d x %d)", out_path, tme.shape[0], tme.shape[1])

    return tme


def main() -> None:
    """Run TME feature extraction script.

    Args:
        None

    Returns:
        None
    """
    configure_logging()

    rnaseq_processed = PROCESSED_DIR / "rnaseq_processed.csv"
    if rnaseq_processed.exists():
        rnaseq = pd.read_csv(rnaseq_processed, index_col=0)
    else:
        rnaseq = load_rnaseq(str(RNASEQ_FILE))
        rnaseq.to_csv(rnaseq_processed)

    tme = build_tme_feature_matrix(rnaseq)
    logging.info("TME matrix ready with %d patients", tme.shape[0])


if __name__ == "__main__":
    main()
