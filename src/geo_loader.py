"""GEO neoadjuvant breast cancer dataset loader with explicit pCR labels.

Downloads and parses GEO series matrix files for datasets with real
pathologic complete response (pCR) endpoints from neoadjuvant chemotherapy trials.
"""
from __future__ import annotations

import gzip
import io
import logging
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import PROCESSED_DIR, RANDOM_STATE, configure_logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

GEO_DATASETS = {
    "GSE25065": {
        "n": 508,
        "description": "TNBC, neoadjuvant chemo, explicit pCR labels",
        "ftp": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE25nnn/GSE25065/matrix/",
    },
    "GSE20194": {
        "n": 278,
        "description": "All subtypes, neoadjuvant, pCR available",
        "ftp": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE20nnn/GSE20194/matrix/",
    },
    "GSE41998": {
        "n": 508,
        "description": "Neoadjuvant chemo + bevacizumab, pCR labels",
        "ftp": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE41nnn/GSE41998/matrix/",
    },
    "GSE96058": {
        "n": 3678,
        "description": "Large cohort, RNA-seq, survival outcomes",
        "ftp": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96058/matrix/",
    },
}

# Patterns for extracting pCR from sample characteristics
PCR_PATTERNS = [
    re.compile(r"(?:pcr|pathologic[_ ]?complete[_ ]?response)\s*[:=]\s*(yes|no|1|0|pcr|rd|rcb-?0)", re.IGNORECASE),
    re.compile(r"(?:response)\s*[:=]\s*(pcr|rd|residual|complete)", re.IGNORECASE),
    re.compile(r"(?:rcb[_ ]?class)\s*[:=]\s*(0|rcb-?0|I|II|III)", re.IGNORECASE),
    re.compile(r"(?:drfs[_ ]?event|dfs[_ ]?event|rfs[_ ]?event)\s*[:=]\s*(0|1|yes|no)", re.IGNORECASE),
]

PCR_POSITIVE = {"yes", "1", "pcr", "complete", "0", "rcb0", "rcb-0"}
PCR_NEGATIVE = {"no", "0", "rd", "residual", "i", "ii", "iii", "rcb-i", "rcb-ii", "rcb-iii"}


def download_geo_series_matrix(geo_id: str, output_dir: Path) -> Path:
    """Download and parse GEO series matrix file.

    Steps:
    1. Download .txt.gz via urllib from NCBI FTP
    2. Parse header: !Sample_geo_accession, !Sample_characteristics_ch1
    3. Parse expression block: rows=probes, cols=samples
    4. Extract pCR from characteristics
    5. Save expression and metadata TSVs

    Args:
        geo_id: GEO series accession (e.g., GSE25065).
        output_dir: Directory to save downloaded files.

    Returns:
        Path to expression file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    expr_path = output_dir / f"{geo_id}_expr.tsv"
    meta_path = output_dir / f"{geo_id}_meta.tsv"

    if expr_path.exists() and meta_path.exists():
        logging.info("GEO %s already downloaded, skipping", geo_id)
        return expr_path

    dataset_info = GEO_DATASETS.get(geo_id)
    if dataset_info is None:
        raise ValueError(f"Unknown GEO dataset: {geo_id}")

    ftp_base = dataset_info["ftp"]
    gz_filename = f"{geo_id}_series_matrix.txt.gz"
    url = ftp_base + gz_filename
    local_gz = output_dir / gz_filename

    if not local_gz.exists():
        logging.info("Downloading %s from %s", geo_id, url)
        try:
            urllib.request.urlretrieve(url, local_gz)
        except Exception as exc:
            logging.warning("Failed to download %s: %s", url, exc)
            # Try alternative naming patterns
            for alt_suffix in [
                f"{geo_id}-GPL96_series_matrix.txt.gz",
                f"{geo_id}-GPL570_series_matrix.txt.gz",
            ]:
                alt_url = ftp_base + alt_suffix
                try:
                    logging.info("Trying alternative URL: %s", alt_url)
                    urllib.request.urlretrieve(alt_url, local_gz)
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError(
                    f"Could not download series matrix for {geo_id}. "
                    f"Place {gz_filename} manually in {output_dir}"
                ) from exc

    logging.info("Parsing series matrix for %s", geo_id)
    meta_rows: dict[str, dict[str, str]] = {}
    sample_ids: list[str] = []
    expr_lines: list[str] = []
    in_expr_block = False

    with gzip.open(local_gz, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")

            if line.startswith("!Sample_geo_accession"):
                parts = line.split("\t")
                sample_ids = [p.strip('"') for p in parts[1:]]
                for sid in sample_ids:
                    if sid not in meta_rows:
                        meta_rows[sid] = {"sample_id": sid}

            elif line.startswith("!Sample_characteristics_ch1"):
                parts = line.split("\t")
                for i, val in enumerate(parts[1:]):
                    if i < len(sample_ids):
                        sid = sample_ids[i]
                        clean_val = val.strip('"').strip()
                        # Append characteristics to capture all fields
                        existing = meta_rows.get(sid, {}).get("characteristics", "")
                        if existing:
                            meta_rows[sid]["characteristics"] = existing + " || " + clean_val
                        else:
                            meta_rows.setdefault(sid, {})["characteristics"] = clean_val

            elif line.startswith("!Sample_title"):
                parts = line.split("\t")
                for i, val in enumerate(parts[1:]):
                    if i < len(sample_ids):
                        meta_rows.setdefault(sample_ids[i], {})["title"] = val.strip('"')

            elif line.startswith("!series_matrix_table_begin"):
                in_expr_block = True
                continue
            elif line.startswith("!series_matrix_table_end"):
                in_expr_block = False
                continue

            if in_expr_block:
                expr_lines.append(line)

    # Parse expression matrix
    if expr_lines:
        expr_text = "\n".join(expr_lines)
        expr_df = pd.read_csv(io.StringIO(expr_text), sep="\t", index_col=0)
        expr_df.columns = [c.strip('"') for c in expr_df.columns]
        expr_df.index = expr_df.index.astype(str)
        expr_df = expr_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        expr_df.to_csv(expr_path, sep="\t")
        logging.info("GEO %s expression matrix: %d probes x %d samples",
                      geo_id, expr_df.shape[0], expr_df.shape[1])
    else:
        logging.warning("No expression data found in %s series matrix", geo_id)
        pd.DataFrame().to_csv(expr_path, sep="\t")

    # Save metadata
    meta_df = pd.DataFrame.from_dict(meta_rows, orient="index")
    if "sample_id" not in meta_df.columns and meta_df.index.name != "sample_id":
        meta_df["sample_id"] = meta_df.index
    meta_df.to_csv(meta_path, sep="\t", index=False)
    logging.info("GEO %s metadata: %d samples", geo_id, len(meta_df))

    return expr_path


def extract_pcr_labels(meta_df: pd.DataFrame) -> pd.Series:
    """Extract binary pCR from !Sample_characteristics_ch1 fields.

    Args:
        meta_df: Metadata DataFrame with 'characteristics' column.

    Returns:
        Series: sample_id -> int (1=pCR, 0=residual disease).
    """
    labels: dict[str, int] = {}

    char_col = None
    for candidate in ["characteristics", "response", "pcr", "pCR"]:
        if candidate in meta_df.columns:
            char_col = candidate
            break

    if char_col is None:
        logging.warning("No characteristics column found in metadata")
        return pd.Series(dtype=int)

    for idx, row in meta_df.iterrows():
        text = str(row.get(char_col, ""))
        sample_id = str(row.get("sample_id", idx))

        for pattern in PCR_PATTERNS:
            match = pattern.search(text)
            if match:
                value = match.group(1).lower().strip()
                # RCB class: 0 = pCR, I/II/III = residual disease
                if "rcb" in pattern.pattern.lower():
                    if value in {"0", "rcb0", "rcb-0"}:
                        labels[sample_id] = 1
                    else:
                        labels[sample_id] = 0
                elif value in PCR_POSITIVE:
                    labels[sample_id] = 1
                elif value in PCR_NEGATIVE:
                    labels[sample_id] = 0
                break

    result = pd.Series(labels, name="pCR", dtype=int)

    if len(result) > 0:
        n_pcr = int((result == 1).sum())
        n_rd = int((result == 0).sum())
        logging.info("pCR extraction: %d pCR, %d RD, %d total (%.1f%% pCR rate)",
                      n_pcr, n_rd, len(result), 100.0 * n_pcr / len(result))
        assert n_pcr > 0 and n_rd > 0, "Both pCR classes must be present"
    else:
        logging.warning("No pCR labels extracted from metadata")

    return result


def _probe_to_gene_mean(expr_df: pd.DataFrame) -> pd.DataFrame:
    """Map probe-level expression to gene symbols by averaging.

    For microarray data, probe IDs are often numeric or Affymetrix-style.
    We use a heuristic: if index looks like probe IDs (numeric or _at suffix),
    keep as-is and let downstream handle gene mapping. Otherwise, treat index
    as gene symbols and average duplicates.

    Args:
        expr_df: Probes x samples expression DataFrame.

    Returns:
        Genes x samples DataFrame (transposed to samples x genes).
    """
    idx_str = expr_df.index.astype(str)
    # Heuristic: if most indices are numeric or contain '_at', these are probe IDs
    is_probe = idx_str.str.match(r"^\d+$").mean() > 0.5 or idx_str.str.contains("_at").mean() > 0.3

    if is_probe:
        # Keep probe-level data, transpose to samples x probes
        result = expr_df.T
        result.columns = result.columns.astype(str)
        logging.info("Probe-level data detected (%d probes), keeping as features", result.shape[1])
    else:
        # Gene symbols: average duplicates
        expr_df.index = idx_str.str.upper()
        gene_expr = expr_df.groupby(level=0).mean()
        result = gene_expr.T
        logging.info("Gene-level data: %d genes", result.shape[1])

    return result


def harmonize_expression(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Align gene sets and normalize across datasets.

    Steps:
    1. Keep intersection of common features (genes or probes)
    2. Log2(x+1) normalize each dataset
    3. Quantile normalization within each dataset
    4. Record batch labels for downstream correction

    Args:
        datasets: Dict of {dataset_id: expression DataFrame (samples x genes/probes)}.

    Returns:
        Dict of {dataset_id: normalized expression DataFrame}.
    """
    if not datasets:
        return {}

    # Find common features across all datasets
    common_features = None
    for name, df in datasets.items():
        if common_features is None:
            common_features = set(df.columns)
        else:
            common_features &= set(df.columns)

    if not common_features:
        logging.warning("No common features across datasets; using union with zero-fill")
        all_features = set()
        for df in datasets.values():
            all_features |= set(df.columns)
        common_features = all_features

    common_features_list = sorted(common_features)
    logging.info("Harmonizing %d datasets on %d common features", len(datasets), len(common_features_list))

    harmonized = {}
    for name, df in datasets.items():
        # Subset to common features, fill missing with 0
        aligned = df.reindex(columns=common_features_list, fill_value=0.0)

        # Log2(x+1) normalization (skip if already log-scale)
        max_val = aligned.max().max()
        if max_val > 50:  # Likely raw intensities
            aligned = np.log2(aligned.clip(lower=0) + 1.0)

        # Quantile normalization within dataset
        aligned = _quantile_normalize(aligned)

        harmonized[name] = aligned
        logging.info("Harmonized %s: %d samples x %d features", name, aligned.shape[0], aligned.shape[1])

    return harmonized


def _quantile_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Quantile normalize a samples x features DataFrame."""
    if df.empty:
        return df
    rank_mean = df.stack().groupby(df.rank(method="first").stack().astype(int)).mean()
    normalized = df.rank(method="min").stack().astype(int).map(rank_mean).unstack()
    return normalized.reindex_like(df).fillna(0.0)


def build_pcr_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Combine all GEO datasets into unified pCR dataset.

    Downloads each GEO series matrix, extracts pCR labels,
    harmonizes expression data, and builds a combined dataset.

    Returns:
        Tuple of (expression DataFrame, pCR labels Series).
    """
    geo_dir = RAW_DIR / "geo"
    geo_dir.mkdir(parents=True, exist_ok=True)

    all_expr: dict[str, pd.DataFrame] = {}
    all_labels: dict[str, pd.Series] = {}
    dataset_stats: list[dict] = []

    for geo_id, info in GEO_DATASETS.items():
        logging.info("Processing %s: %s", geo_id, info["description"])
        try:
            expr_path = download_geo_series_matrix(geo_id, geo_dir)
            meta_path = geo_dir / f"{geo_id}_meta.tsv"

            # Load expression
            expr_df = pd.read_csv(expr_path, sep="\t", index_col=0)
            if expr_df.empty:
                logging.warning("Empty expression matrix for %s, skipping", geo_id)
                continue

            # Convert to samples x genes
            samples_x_genes = _probe_to_gene_mean(expr_df)

            # Load metadata and extract pCR
            meta_df = pd.read_csv(meta_path, sep="\t")
            pcr_labels = extract_pcr_labels(meta_df)

            if pcr_labels.empty:
                logging.warning("No pCR labels for %s, checking for survival data", geo_id)
                # For GSE96058 (survival-only), skip pCR extraction
                dataset_stats.append({
                    "dataset": geo_id,
                    "n_samples": len(samples_x_genes),
                    "n_pcr": 0,
                    "pcr_rate": float("nan"),
                    "status": "no_pCR_labels",
                })
                continue

            # Align expression and labels
            common_samples = samples_x_genes.index.intersection(pcr_labels.index)
            if len(common_samples) < 10:
                logging.warning("Only %d samples with both expression and pCR for %s",
                                len(common_samples), geo_id)
                continue

            expr_aligned = samples_x_genes.loc[common_samples]
            labels_aligned = pcr_labels.loc[common_samples]

            # Add batch identifier
            expr_aligned = expr_aligned.copy()

            all_expr[geo_id] = expr_aligned
            all_labels[geo_id] = labels_aligned

            n_pcr = int((labels_aligned == 1).sum())
            dataset_stats.append({
                "dataset": geo_id,
                "n_samples": len(labels_aligned),
                "n_pcr": n_pcr,
                "pcr_rate": float(n_pcr / len(labels_aligned)),
                "status": "ok",
            })

        except Exception as exc:
            logging.error("Failed to process %s: %s", geo_id, exc)
            dataset_stats.append({
                "dataset": geo_id,
                "n_samples": 0,
                "n_pcr": 0,
                "pcr_rate": float("nan"),
                "status": f"error: {exc}",
            })
            continue

    # Report stats
    logging.info("=== GEO Dataset Summary ===")
    for stat in dataset_stats:
        logging.info("  %s: N=%d, pCR=%d (%.1f%%), status=%s",
                      stat["dataset"], stat["n_samples"], stat["n_pcr"],
                      stat["pcr_rate"] * 100 if np.isfinite(stat["pcr_rate"]) else 0,
                      stat["status"])

    if not all_expr:
        logging.warning("No GEO datasets successfully processed with pCR labels")
        return pd.DataFrame(), pd.Series(dtype=int)

    # Harmonize expression across datasets
    harmonized = harmonize_expression(all_expr)

    # Combine all datasets
    combined_expr_parts = []
    combined_labels_parts = []
    batch_labels = []

    for geo_id in harmonized:
        expr = harmonized[geo_id]
        labels = all_labels[geo_id]
        # Prefix sample IDs with dataset to avoid collisions
        expr.index = [f"{geo_id}_{s}" for s in expr.index]
        labels.index = [f"{geo_id}_{s}" for s in labels.index]
        combined_expr_parts.append(expr)
        combined_labels_parts.append(labels)
        batch_labels.extend([geo_id] * len(expr))

    combined_expr = pd.concat(combined_expr_parts, axis=0)
    combined_labels = pd.concat(combined_labels_parts)

    total_pcr = int((combined_labels == 1).sum())
    total_n = len(combined_labels)
    logging.info("Combined dataset: %d samples, %d pCR (%.1f%%)",
                  total_n, total_pcr, 100.0 * total_pcr / max(total_n, 1))

    if total_pcr < 50:
        logging.warning("Fewer than 50 pCR-positive cases (%d). Proceeding but results may be underpowered.", total_pcr)

    # Save outputs
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined_expr.to_csv(PROCESSED_DIR / "geo_pcr_expr.csv")
    combined_labels.to_csv(PROCESSED_DIR / "geo_pcr_labels.csv", header=["pCR"])

    # Save batch info for downstream correction
    batch_df = pd.DataFrame({"sample_id": combined_expr.index, "batch": batch_labels})
    batch_df.to_csv(PROCESSED_DIR / "geo_batch_labels.csv", index=False)

    # Save dataset stats
    stats_df = pd.DataFrame(dataset_stats)
    stats_df.to_csv(PROCESSED_DIR / "geo_dataset_stats.csv", index=False)

    logging.info("Saved GEO pCR dataset: %d samples x %d features",
                  combined_expr.shape[0], combined_expr.shape[1])

    return combined_expr, combined_labels


def main() -> None:
    """CLI entrypoint: download GEO datasets and build unified pCR dataset."""
    configure_logging()
    logging.info("=== GEO pCR Dataset Loader ===")
    expr, labels = build_pcr_dataset()
    if not expr.empty:
        logging.info("GEO pCR dataset ready: %d samples, %d features, pCR rate %.1f%%",
                      len(labels), expr.shape[1],
                      100.0 * (labels == 1).sum() / len(labels))
    else:
        logging.warning("No GEO data available. Pipeline will use TCGA data only.")


if __name__ == "__main__":
    main()
