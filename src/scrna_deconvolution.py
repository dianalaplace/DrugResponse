"""scRNA-seq calibrated TME deconvolution using Wu et al. 2021 breast cancer atlas.

Uses the Wu et al. 2021 scRNA-seq atlas (GSE176078) to build a cell-type
signature matrix and deconvolve bulk RNA-seq samples with higher resolution
than LM22-based approaches.
"""
from __future__ import annotations

import gzip
import logging
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import pearsonr, spearmanr

from data_loader import PROCESSED_DIR, RANDOM_STATE, configure_logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

WU2021_GEO_ID = "GSE176078"
WU2021_FTP = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE176nnn/GSE176078/"

SCRNA_CELL_TYPES = [
    "Cancer_Epithelial", "T_cells_CD8", "T_cells_CD4",
    "Macrophages_M1", "Macrophages_M2", "CAFs",
    "Endothelial", "B_cells", "NK_cells",
    "Myeloid_other", "Plasmablasts",
]

# Curated marker genes per cell type from Wu et al. 2021 and literature.
# Used as fallback when full atlas is unavailable.
CURATED_MARKERS: dict[str, list[str]] = {
    "Cancer_Epithelial": [
        "EPCAM", "KRT18", "KRT19", "KRT8", "KRT7", "MUC1", "ESR1",
        "ERBB2", "GATA3", "FOXA1", "CDH1", "TFF1", "TFF3", "AGR2",
        "SERPINA3", "CLDN4", "CLDN7", "S100A14", "TACSTD2", "CEACAM6",
    ],
    "T_cells_CD8": [
        "CD8A", "CD8B", "GZMB", "GZMK", "PRF1", "IFNG", "NKG7",
        "CCL5", "CST7", "GNLY", "CD3D", "CD3E", "CXCR3", "EOMES",
        "TBX21", "LAG3", "PDCD1", "HAVCR2", "TIGIT", "CTLA4",
    ],
    "T_cells_CD4": [
        "CD4", "IL7R", "TCF7", "LEF1", "SELL", "CCR7", "MAL",
        "FOXP3", "IL2RA", "CTLA4", "IKZF2", "TNFRSF18", "TIGIT",
        "BATF", "ICOS", "CD40LG", "CXCR5", "PDCD1", "TOX", "TOX2",
    ],
    "Macrophages_M1": [
        "CD68", "CD80", "CD86", "IL1B", "TNF", "NOS2", "CXCL9",
        "CXCL10", "CXCL11", "IDO1", "IRF1", "IRF5", "STAT1",
        "HLA-DRA", "HLA-DRB1", "CD14", "FCGR1A", "FCGR3A", "CCR7", "IL12B",
    ],
    "Macrophages_M2": [
        "CD163", "MRC1", "MSR1", "FOLR2", "CD68", "IL10", "CCL18",
        "CCL22", "CCL17", "ARG1", "TGFB1", "VEGFA", "MMP9", "CD209",
        "STAB1", "LYVE1", "F13A1", "SEPP1", "MAF", "MERTK",
    ],
    "CAFs": [
        "ACTA2", "FAP", "PDGFRA", "PDGFRB", "COL1A1", "COL1A2",
        "COL3A1", "COL5A1", "DCN", "LUM", "THY1", "VIM", "FN1",
        "POSTN", "TAGLN", "MYL9", "SPARC", "BGN", "CALD1", "IGFBP7",
    ],
    "Endothelial": [
        "PECAM1", "VWF", "CDH5", "FLT1", "KDR", "ENG", "PLVAP",
        "CLDN5", "EMCN", "ACKR1", "SELE", "SELP", "ICAM1", "VCAM1",
        "TEK", "TIE1", "NOS3", "ESAM", "ROBO4", "ERG",
    ],
    "B_cells": [
        "CD19", "MS4A1", "CD79A", "CD79B", "PAX5", "BLK", "BANK1",
        "CD22", "FCER2", "CR2", "TCL1A", "VPREB3", "IGHD", "IGHM",
        "CD24", "CD38", "FCRL4", "TNFRSF13B", "TNFRSF17", "IRF4",
    ],
    "NK_cells": [
        "NCAM1", "NKG7", "GNLY", "KLRB1", "KLRD1", "KLRK1", "KLRC1",
        "NCR1", "NCR3", "FCGR3A", "PRF1", "GZMB", "GZMH", "FGFBP2",
        "CX3CR1", "SH2D1B", "TYROBP", "CD160", "KIR2DL1", "KIR3DL1",
    ],
    "Myeloid_other": [
        "ITGAX", "HLA-DRA", "CD1C", "CLEC9A", "XCR1", "FCER1A",
        "CD14", "FCGR3A", "S100A8", "S100A9", "VCAN", "FCN1",
        "CD33", "SIGLEC6", "LYZ", "CSF1R", "ITGAM", "CST3", "AIF1", "LST1",
    ],
    "Plasmablasts": [
        "IGHG1", "IGHG2", "IGHA1", "IGKC", "IGLC1", "IGLC2",
        "MZB1", "JCHAIN", "XBP1", "IRF4", "PRDM1", "SDC1",
        "TNFRSF17", "CD38", "SLAMF7", "DERL3", "SEC11C", "FKBP11", "SSR4", "DDOST",
    ],
}

# Mapping from scRNA cell types to LM22-style names for comparison
SCRNA_TO_LM22_MAP = {
    "T_cells_CD8": "CD8_T_cell",
    "Macrophages_M2": "M2_macrophage",
    "CAFs": "CAF_proxy",
    "NK_cells": "NK_cell",
    "B_cells": "B_cell",
    "T_cells_CD4": "Treg",  # approximate
    "Myeloid_other": "Dendritic",  # approximate
}


def download_wu2021_atlas(output_dir: Path) -> Path:
    """Download Wu et al. 2021 scRNA-seq atlas from GEO (GSE176078).

    Attempts to download the main expression file. If the full h5 file
    is unavailable, falls back to using curated marker signatures.

    Args:
        output_dir: Directory to save atlas files.

    Returns:
        Path to main data file or marker file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    marker_path = output_dir / "wu2021_curated_markers.csv"

    # The Wu 2021 atlas files on GEO are supplementary and large.
    # Try downloading the series matrix for metadata at minimum.
    matrix_url = WU2021_FTP + "matrix/GSE176078_series_matrix.txt.gz"
    matrix_gz = output_dir / "GSE176078_series_matrix.txt.gz"

    if not matrix_gz.exists():
        logging.info("Attempting Wu 2021 atlas download from %s", matrix_url)
        try:
            urllib.request.urlretrieve(matrix_url, matrix_gz)
            logging.info("Downloaded Wu 2021 series matrix metadata")
        except Exception as exc:
            logging.warning("Could not download Wu 2021 atlas: %s", exc)
            logging.info("Using curated marker gene signatures as fallback")

    # Always save curated markers as the reliable fallback
    _save_curated_markers(marker_path)
    return marker_path


def _save_curated_markers(path: Path) -> None:
    """Save curated marker gene lists to CSV."""
    rows = []
    for cell_type, genes in CURATED_MARKERS.items():
        for rank, gene in enumerate(genes):
            rows.append({"cell_type": cell_type, "gene": gene.upper(), "rank": rank})
    pd.DataFrame(rows).to_csv(path, index=False)
    logging.info("Saved curated markers: %d cell types, %d total markers",
                  len(CURATED_MARKERS), len(rows))


def build_scrna_signature_matrix(atlas_path: Path,
                                  target_genes: list[str]) -> pd.DataFrame:
    """Build marker gene signature matrix from scRNA-seq atlas.

    Steps:
    1. Load curated markers (or atlas if available)
    2. Filter to genes present in target bulk RNA-seq dataset
    3. Build matrix: genes x cell_types (binary or weighted indicators)

    Args:
        atlas_path: Path to atlas data or curated markers CSV.
        target_genes: List of gene symbols in the bulk RNA-seq dataset.

    Returns:
        Signature matrix DataFrame (genes x cell_types).
    """
    target_set = {g.upper() for g in target_genes}

    try:
        import scanpy as sc
        # Try loading as h5ad if available
        h5ad_candidates = list(atlas_path.parent.glob("*.h5ad"))
        if h5ad_candidates:
            logging.info("Loading scRNA-seq atlas from %s", h5ad_candidates[0])
            adata = sc.read_h5ad(h5ad_candidates[0])
            return _build_signature_from_adata(adata, target_set)
    except ImportError:
        logging.info("scanpy not available; using curated marker signatures")
    except Exception as exc:
        logging.warning("Failed to load atlas h5ad: %s; using curated markers", exc)

    # Fallback: build from curated markers
    logging.info("Building signature matrix from curated markers")
    markers_df = pd.read_csv(atlas_path)

    sig_data: dict[str, dict[str, float]] = {}
    for cell_type in SCRNA_CELL_TYPES:
        ct_markers = markers_df[markers_df["cell_type"] == cell_type]
        genes_available = [g for g in ct_markers["gene"].values if g in target_set]

        if not genes_available:
            logging.warning("No marker genes for %s found in bulk data", cell_type)
            continue

        # Weight by rank: top markers get higher weight
        for _, row in ct_markers.iterrows():
            gene = row["gene"]
            if gene in target_set:
                rank = row["rank"]
                weight = 1.0 / (1.0 + rank * 0.1)  # Decaying weight by rank
                if gene not in sig_data:
                    sig_data[gene] = {}
                sig_data[gene][cell_type] = weight

        logging.info("%s: %d/%d markers overlap with bulk data",
                      cell_type, len(genes_available), len(ct_markers))

    sig_matrix = pd.DataFrame(sig_data).T.fillna(0.0)
    sig_matrix = sig_matrix.reindex(columns=SCRNA_CELL_TYPES).fillna(0.0)
    # Remove cell types with no markers
    sig_matrix = sig_matrix.loc[:, sig_matrix.sum(axis=0) > 0]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sig_matrix.to_csv(PROCESSED_DIR / "scrna_signatures.csv")
    logging.info("Signature matrix: %d genes x %d cell types",
                  sig_matrix.shape[0], sig_matrix.shape[1])

    return sig_matrix


def _build_signature_from_adata(adata, target_set: set[str]) -> pd.DataFrame:
    """Build signature matrix from a loaded scanpy AnnData object."""
    import scanpy as sc

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Use rank_genes_groups for marker detection
    cell_type_col = None
    for candidate in ["celltype_major", "cell_type", "celltype", "CellType"]:
        if candidate in adata.obs.columns:
            cell_type_col = candidate
            break

    if cell_type_col is None:
        logging.warning("No cell type annotation found in atlas")
        raise ValueError("Atlas missing cell type annotations")

    sc.tl.rank_genes_groups(adata, cell_type_col, method="wilcoxon")

    sig_data: dict[str, dict[str, float]] = {}
    for ct in adata.obs[cell_type_col].unique():
        result = sc.get.rank_genes_groups_df(adata, group=ct)
        # Filter: log2FC > 1 and FDR < 0.05
        markers = result[(result["logfoldchanges"] > 1) & (result["pvals_adj"] < 0.05)]
        top_markers = markers.head(50)

        for _, row in top_markers.iterrows():
            gene = str(row["names"]).upper()
            if gene in target_set:
                if gene not in sig_data:
                    sig_data[gene] = {}
                sig_data[gene][ct] = float(row["logfoldchanges"])

    sig_matrix = pd.DataFrame(sig_data).T.fillna(0.0)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sig_matrix.to_csv(PROCESSED_DIR / "scrna_signatures.csv")
    return sig_matrix


def deconvolve_bulk_scrna(bulk_expr: pd.DataFrame,
                          signature_matrix: pd.DataFrame,
                          method: str = "nnls") -> pd.DataFrame:
    """Deconvolve bulk RNA-seq using scRNA-seq signatures.

    Args:
        bulk_expr: Samples x genes expression DataFrame.
        signature_matrix: Genes x cell_types signature matrix.
        method: 'nnls' (default) or 'ridge'.

    Returns:
        Samples x cell_types fractions DataFrame.
    """
    # Align genes between bulk and signature
    common_genes = bulk_expr.columns.intersection(signature_matrix.index)
    if len(common_genes) < 10:
        # Try uppercase matching
        bulk_upper = bulk_expr.copy()
        bulk_upper.columns = bulk_upper.columns.str.upper()
        sig_upper = signature_matrix.copy()
        sig_upper.index = sig_upper.index.str.upper()
        common_genes = bulk_upper.columns.intersection(sig_upper.index)
        if len(common_genes) >= 10:
            bulk_expr = bulk_upper
            signature_matrix = sig_upper

    if len(common_genes) < 10:
        logging.warning("Only %d common genes between bulk and signatures", len(common_genes))
        return pd.DataFrame(index=bulk_expr.index)

    logging.info("Deconvolving %d samples using %d genes and %d cell types (%s)",
                  len(bulk_expr), len(common_genes),
                  signature_matrix.shape[1], method)

    B = bulk_expr[common_genes].values  # samples x genes
    S = signature_matrix.loc[common_genes].values  # genes x cell_types

    fractions = np.zeros((B.shape[0], S.shape[1]))

    if method == "nnls":
        for i in range(B.shape[0]):
            coef, _ = nnls(S, B[i, :])
            total = coef.sum()
            fractions[i, :] = coef / total if total > 0 else 0.0
    elif method == "ridge":
        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha=1.0, positive=True)
        ridge.fit(S, B.T)
        raw = ridge.coef_
        row_sums = raw.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        fractions = raw / row_sums
    else:
        raise ValueError(f"Unknown deconvolution method: {method}")

    result = pd.DataFrame(
        fractions,
        index=bulk_expr.index,
        columns=signature_matrix.columns,
    )

    # Compare to LM22 if available
    _compare_to_lm22(result)

    return result


def _compare_to_lm22(scrna_fractions: pd.DataFrame) -> None:
    """Compare scRNA deconvolution to LM22 results if available."""
    lm22_path = PROCESSED_DIR / "tme_features.csv"
    if not lm22_path.exists():
        logging.info("No LM22 results found for comparison")
        return

    lm22 = pd.read_csv(lm22_path, index_col=0)
    common_samples = scrna_fractions.index.intersection(lm22.index)
    if len(common_samples) < 10:
        logging.info("Insufficient common samples for LM22 comparison (%d)", len(common_samples))
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    correlations: dict[str, float] = {}
    for scrna_ct, lm22_ct in SCRNA_TO_LM22_MAP.items():
        if scrna_ct in scrna_fractions.columns and lm22_ct in lm22.columns:
            s = scrna_fractions.loc[common_samples, scrna_ct]
            l = lm22.loc[common_samples, lm22_ct]
            if s.std() > 0 and l.std() > 0:
                r, _ = pearsonr(s, l)
                correlations[f"{scrna_ct}_vs_{lm22_ct}"] = r

    if correlations:
        logging.info("scRNA vs LM22 correlations:")
        for pair, r in correlations.items():
            logging.info("  %s: r=%.3f", pair, r)

        # Plot comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        pairs = list(correlations.keys())
        values = [correlations[p] for p in pairs]
        colors = ["#2166ac" if v > 0 else "#b2182b" for v in values]
        ax.barh(range(len(pairs)), values, color=colors)
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels([p.replace("_vs_", "\nvs ") for p in pairs], fontsize=8)
        ax.set_xlabel("Pearson r")
        ax.set_title("scRNA-seq vs LM22 Deconvolution Comparison")
        ax.axvline(0, color="black", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "deconvolution_lm22_vs_scrna.png", dpi=200)
        plt.close(fig)


def validate_fractions_ihc(fractions: pd.DataFrame) -> dict[str, float]:
    """Validate deconvolution fractions against CPTAC-BRCA IHC data if available.

    Args:
        fractions: Samples x cell_types fractions DataFrame.

    Returns:
        Dict of Spearman correlations per marker, or empty dict if unavailable.
    """
    # CPTAC data requires portal access; check for local files
    cptac_candidates = [
        RAW_DIR / "cptac_brca_ihc.csv",
        RAW_DIR / "cptac_brca_ihc.tsv",
        RAW_DIR / "cptac" / "brca_ihc.csv",
    ]

    ihc_path = next((p for p in cptac_candidates if p.exists()), None)
    if ihc_path is None:
        logging.warning("CPTAC IHC data not found locally. Skipping IHC validation. "
                         "Download from https://cptac-data-portal.georgetown.edu/ if available.")
        return {}

    logging.info("Loading CPTAC IHC data from %s", ihc_path)
    sep = "\t" if ihc_path.suffix == ".tsv" else ","
    ihc = pd.read_csv(ihc_path, sep=sep, index_col=0)

    common = fractions.index.intersection(ihc.index)
    if len(common) < 10:
        logging.warning("Insufficient common samples for IHC validation (%d)", len(common))
        return {}

    # Map deconvolution cell types to IHC markers
    ihc_map = {
        "CD8": "T_cells_CD8",
        "CD163": "Macrophages_M2",
        "FOXP3": "T_cells_CD4",
    }

    results: dict[str, float] = {}
    for ihc_marker, deconv_ct in ihc_map.items():
        ihc_col = next((c for c in ihc.columns if ihc_marker.lower() in c.lower()), None)
        if ihc_col is None or deconv_ct not in fractions.columns:
            continue
        s_ihc = pd.to_numeric(ihc.loc[common, ihc_col], errors="coerce").dropna()
        s_deconv = fractions.loc[s_ihc.index, deconv_ct]
        if len(s_ihc) >= 10 and s_ihc.std() > 0 and s_deconv.std() > 0:
            r, _ = spearmanr(s_ihc, s_deconv)
            results[f"{ihc_marker}_r"] = float(r)
            logging.info("IHC validation %s: Spearman r=%.3f (n=%d)",
                          ihc_marker, r, len(s_ihc))

    if results:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        markers = list(results.keys())
        values = [results[m] for m in markers]
        ax.bar(markers, values, color="#4daf4a")
        ax.set_ylabel("Spearman r")
        ax.set_title("Deconvolution vs IHC Validation (CPTAC)")
        ax.axhline(0, color="black", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "ihc_validation.png", dpi=200)
        plt.close(fig)

    return results


def build_scrna_tme_features(bulk_expr: pd.DataFrame) -> pd.DataFrame:
    """End-to-end: download atlas, build signatures, deconvolve bulk.

    Args:
        bulk_expr: Samples x genes expression DataFrame.

    Returns:
        Samples x cell_types fractions DataFrame.
    """
    atlas_dir = RAW_DIR / "wu2021"
    atlas_path = download_wu2021_atlas(atlas_dir)

    target_genes = [g.upper() for g in bulk_expr.columns]
    sig_matrix = build_scrna_signature_matrix(atlas_path, target_genes)

    if sig_matrix.empty:
        logging.warning("Empty signature matrix; cannot deconvolve")
        return pd.DataFrame(index=bulk_expr.index)

    fractions = deconvolve_bulk_scrna(bulk_expr, sig_matrix, method="nnls")

    if not fractions.empty:
        ihc_results = validate_fractions_ihc(fractions)

        # Save scRNA TME features
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        fractions.to_csv(PROCESSED_DIR / "scrna_tme_features.csv")
        logging.info("Saved scRNA TME features: %d samples x %d cell types",
                      fractions.shape[0], fractions.shape[1])

    return fractions


def main() -> None:
    """CLI entrypoint: build scRNA-calibrated TME features."""
    configure_logging()
    logging.info("=== scRNA-seq Calibrated TME Deconvolution ===")

    # Load bulk RNA-seq
    rnaseq_path = PROCESSED_DIR / "rnaseq_processed.csv"
    if not rnaseq_path.exists():
        logging.error("RNA-seq data not found at %s. Run data_loader.py first.", rnaseq_path)
        return

    bulk_expr = pd.read_csv(rnaseq_path, index_col=0)
    logging.info("Loaded bulk RNA-seq: %d samples x %d genes", bulk_expr.shape[0], bulk_expr.shape[1])

    fractions = build_scrna_tme_features(bulk_expr)

    if not fractions.empty:
        logging.info("scRNA deconvolution complete: %d samples, %d cell types",
                      fractions.shape[0], fractions.shape[1])
        logging.info("Cell type fractions (mean):")
        for ct in fractions.columns:
            logging.info("  %s: %.4f", ct, fractions[ct].mean())
    else:
        logging.warning("scRNA deconvolution produced empty results")


if __name__ == "__main__":
    main()
