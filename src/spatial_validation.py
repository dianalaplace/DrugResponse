"""Spatial TME validation using public Visium datasets.

Validates bulk TME deconvolution against spatially-resolved transcriptomics
from 10x Visium breast cancer samples.
"""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from data_loader import PROCESSED_DIR, RANDOM_STATE, configure_logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures" / "spatial"

PUBLIC_VISIUM_DATASETS = {
    "HTAN_BREAST": {
        "url": "https://data.humantumoratlas.org/",
        "note": "Requires HTAN data portal registration",
    },
    "10X_VISIUM_BREAST": {
        "url": "https://www.10xgenomics.com/datasets/human-breast-cancer-ductal-carcinoma-in-situ-invasive-carcinoma-ffpe-1-standard-1-3-0",
        "note": "Public demo dataset, no registration",
    },
    "GEO_VISIUM_BRCA": {
        "geo_id": "GSE176078",
        "note": "Same atlas as Wu 2021, includes Visium samples",
    },
}


def _find_visium_data() -> list[Path]:
    """Search for locally available Visium datasets."""
    candidates = []
    visium_dir = RAW_DIR / "visium"
    if visium_dir.exists():
        candidates.extend(visium_dir.glob("**/*.h5ad"))
        candidates.extend(visium_dir.glob("**/filtered_feature_bc_matrix.h5"))
        candidates.extend(visium_dir.glob("**/*.h5"))
    # Also check for 10x sample data
    for pattern in ["*visium*", "*spatial*", "*10x*breast*"]:
        candidates.extend(RAW_DIR.glob(f"**/{pattern}"))
    return sorted(set(candidates))


def load_visium_sample(data_path: Path):
    """Load 10x Visium .h5ad or filtered_feature_bc_matrix with scanpy.

    Args:
        data_path: Path to Visium data file.

    Returns:
        AnnData object with normalized expression and spatial info.
    """
    try:
        import scanpy as sc
    except ImportError:
        logging.warning("scanpy not available; cannot load Visium data")
        return None

    logging.info("Loading Visium sample from %s", data_path)

    if data_path.suffix == ".h5ad":
        adata = sc.read_h5ad(data_path)
    elif data_path.suffix == ".h5":
        try:
            adata = sc.read_10x_h5(data_path)
        except Exception:
            adata = sc.read_h5ad(data_path)
    else:
        logging.warning("Unsupported Visium file format: %s", data_path.suffix)
        return None

    # Basic preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=min(50, adata.n_vars - 1))

    # Compute spatial neighbors if spatial coordinates available
    if "spatial" in adata.obsm:
        try:
            import squidpy as sq
            sq.gr.spatial_neighbors(adata)
            logging.info("Computed spatial neighbors for Visium sample")
        except ImportError:
            logging.info("squidpy not available; skipping spatial neighbor computation")
        except Exception as exc:
            logging.warning("Spatial neighbor computation failed: %s", exc)

    logging.info("Visium sample loaded: %d spots x %d genes", adata.n_obs, adata.n_vars)
    return adata


def compute_spot_tme_scores(adata, signature_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute TME cell type scores for each Visium spot.

    Args:
        adata: AnnData object from load_visium_sample.
        signature_matrix: Genes x cell_types signature matrix.

    Returns:
        Spots x cell_types score DataFrame.
    """
    try:
        import scanpy as sc
    except ImportError:
        logging.warning("scanpy not available for spot scoring")
        return pd.DataFrame()

    scores: dict[str, np.ndarray] = {}

    for cell_type in signature_matrix.columns:
        sig_genes = signature_matrix.index[signature_matrix[cell_type] > 0].tolist()
        # Filter to genes present in Visium data
        var_names = set(adata.var_names.str.upper())
        available_genes = [g for g in sig_genes if g.upper() in var_names]

        if len(available_genes) < 3:
            logging.info("Skipping %s: only %d signature genes in Visium",
                          cell_type, len(available_genes))
            continue

        try:
            sc.tl.score_genes(adata, available_genes, score_name=f"score_{cell_type}")
            scores[cell_type] = adata.obs[f"score_{cell_type}"].values
            logging.info("%s: scored with %d genes", cell_type, len(available_genes))
        except Exception as exc:
            logging.warning("Scoring failed for %s: %s", cell_type, exc)

    if not scores:
        return pd.DataFrame()

    result = pd.DataFrame(scores, index=adata.obs_names)
    logging.info("Spot TME scores: %d spots x %d cell types", result.shape[0], result.shape[1])
    return result


def compare_bulk_spatial_tme(bulk_fractions: pd.DataFrame,
                              spatial_scores: dict[str, pd.DataFrame],
                              sample_ids: list[str]) -> dict[str, float]:
    """Compare bulk deconvolution fractions with mean spatial scores.

    Args:
        bulk_fractions: Samples x cell_types from bulk deconvolution.
        spatial_scores: Dict of {sample_id: spots x cell_types DataFrame}.
        sample_ids: Sample IDs with both bulk and spatial data.

    Returns:
        Dict of {cell_type: Pearson correlation}.
    """
    if not spatial_scores or not sample_ids:
        logging.warning("No paired bulk-spatial samples available")
        return {}

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Compute mean spatial score per sample
    mean_spatial = {}
    for sid in sample_ids:
        if sid in spatial_scores and not spatial_scores[sid].empty:
            mean_spatial[sid] = spatial_scores[sid].mean(axis=0)

    if len(mean_spatial) < 3:
        logging.warning("Fewer than 3 paired samples; correlation unreliable")
        return {}

    spatial_df = pd.DataFrame(mean_spatial).T
    common_cts = bulk_fractions.columns.intersection(spatial_df.columns)
    common_sids = bulk_fractions.index.intersection(spatial_df.index)

    if len(common_cts) == 0 or len(common_sids) < 3:
        logging.warning("Insufficient overlap for bulk-spatial comparison")
        return {}

    correlations: dict[str, float] = {}
    for ct in common_cts:
        b = bulk_fractions.loc[common_sids, ct]
        s = spatial_df.loc[common_sids, ct]
        if b.std() > 0 and s.std() > 0:
            r, _ = pearsonr(b, s)
            correlations[ct] = float(r)

    if correlations:
        logging.info("Bulk vs Spatial TME correlations:")
        for ct, r in correlations.items():
            logging.info("  %s: r=%.3f", ct, r)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        cts = list(correlations.keys())
        vals = [correlations[ct] for ct in cts]
        colors = ["#2166ac" if v > 0.3 else "#fdae61" if v > 0 else "#b2182b" for v in vals]
        ax.barh(range(len(cts)), vals, color=colors)
        ax.set_yticks(range(len(cts)))
        ax.set_yticklabels(cts)
        ax.set_xlabel("Pearson r")
        ax.set_title("Bulk Deconvolution vs Spatial TME Scores")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.axvline(0.3, color="gray", linewidth=0.5, linestyle="--")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "bulk_vs_spatial_correlation.png", dpi=200)
        plt.close(fig)

    return correlations


def run_spatial_validation() -> dict:
    """Run full spatial validation pipeline.

    Returns:
        Dict with validation results including correlations and sample counts.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "n_visium_samples": 0,
        "correlations": {},
        "verdict": "DATA_UNAVAILABLE",
    }

    # Check for available Visium data
    visium_files = _find_visium_data()
    if not visium_files:
        logging.warning(
            "No Visium data found locally. To enable spatial validation:\n"
            "  1. Download from 10x Genomics: %s\n"
            "  2. Place files in data/raw/visium/\n"
            "  3. Re-run pipeline",
            PUBLIC_VISIUM_DATASETS["10X_VISIUM_BREAST"]["url"],
        )
        return results

    logging.info("Found %d Visium data files", len(visium_files))

    # Load signature matrix
    sig_path = PROCESSED_DIR / "scrna_signatures.csv"
    if not sig_path.exists():
        logging.warning("No scRNA signature matrix found; run scrna_deconvolution.py first")
        return results

    sig_matrix = pd.read_csv(sig_path, index_col=0)

    # Load bulk fractions
    bulk_path = PROCESSED_DIR / "scrna_tme_features.csv"
    if not bulk_path.exists():
        bulk_path = PROCESSED_DIR / "tme_features.csv"
    if not bulk_path.exists():
        logging.warning("No bulk TME features found")
        return results

    bulk_fractions = pd.read_csv(bulk_path, index_col=0)

    # Process each Visium sample
    spatial_scores: dict[str, pd.DataFrame] = {}
    for vpath in visium_files:
        sample_name = vpath.stem
        adata = load_visium_sample(vpath)
        if adata is None:
            continue

        scores = compute_spot_tme_scores(adata, sig_matrix)
        if not scores.empty:
            spatial_scores[sample_name] = scores
            results["n_visium_samples"] += 1

    if spatial_scores:
        # Attempt correlation with bulk (may have limited overlap)
        sample_ids = list(spatial_scores.keys())
        correlations = compare_bulk_spatial_tme(bulk_fractions, spatial_scores, sample_ids)
        results["correlations"] = correlations

        if correlations:
            mean_r = float(np.mean(list(correlations.values())))
            results["mean_correlation"] = mean_r
            results["verdict"] = "VALIDATED" if mean_r > 0.3 else "NOT_VALIDATED"
            logging.info("Spatial validation verdict: %s (mean r=%.3f)",
                          results["verdict"], mean_r)
        else:
            results["verdict"] = "INSUFFICIENT_OVERLAP"
    else:
        logging.warning("No Visium samples successfully processed")

    # Save results
    import json
    metrics_dir = RESULTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / "spatial_validation.json").open("w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def main() -> None:
    """CLI entrypoint for spatial TME validation."""
    configure_logging()
    logging.info("=== Spatial TME Validation ===")
    results = run_spatial_validation()
    logging.info("Spatial validation complete: %s", results["verdict"])


if __name__ == "__main__":
    main()
