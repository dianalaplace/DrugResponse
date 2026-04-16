from __future__ import annotations

import gzip
import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import PROCESSED_DIR, RAW_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEO_DIR = RAW_DIR / "geo"

TRAIN_GEO_DATASETS = ("GSE25065", "GSE20194")
ISPY2_SERIES_ID = "GSE173839"
ISPY2_SERIES_PATH = GEO_DIR / f"{ISPY2_SERIES_ID}_series_matrix.txt.gz"
ISPY2_BIOMARKER_PATH = GEO_DIR / "GSE173839_ISPY2_DurvalumabOlaparibArm_biomarkers.csv.gz"

GPL96_ANNOTATION_CANDIDATES = (
    GEO_DIR / "GPL96_annotation.txt",
    GEO_DIR / "GPL96_annotation.txt.gz",
)
GPL20078_ANNOTATION_CANDIDATES = (
    GEO_DIR / "GPL20078_annotation.txt",
    GEO_DIR / "GPL20078_annotation.txt.gz",
)

CLINICAL_COLUMNS = [
    "age",
    "grade",
    "ER_status",
    "PR_status",
    "HER2_status",
    "HR_status",
    "pCR",
]


def _to_binary(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return 1.0 if float(value) >= 0.5 else 0.0

    token = str(value).strip().upper()
    if token in {"P", "POS", "POSITIVE", "YES", "TRUE", "1", "PCR", "COMPLETE"}:
        return 1.0
    if token in {"N", "NEG", "NEGATIVE", "NO", "FALSE", "0", "RD", "RESIDUAL", "INCOMPLETE"}:
        return 0.0
    return np.nan


def _parse_float(value: object) -> float:
    if pd.isna(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _normalize_expression(expr_df: pd.DataFrame) -> pd.DataFrame:
    expr = expr_df.copy()
    expr.columns = [str(col).strip('"') for col in expr.columns]
    expr.index = [str(idx).strip('"') for idx in expr.index]
    expr = expr.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if not expr.empty and float(expr.max().max()) > 50.0:
        expr = np.log2(expr.clip(lower=0.0) + 1.0)
    return expr


def parse_geo_series_matrix(path: Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    meta_lines: list[str] = []
    expr_lines: list[str] = []
    sample_ids: list[str] = []
    in_table = False

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")

            if line.startswith("!Sample_geo_accession"):
                parts = line.split("\t")
                sample_ids = [item.strip('"') for item in parts[1:]]
            elif line.startswith("!Sample_characteristics"):
                meta_lines.append(line)
            elif line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            elif line.startswith("!series_matrix_table_end"):
                in_table = False
                continue

            if in_table:
                expr_lines.append(line)

    expr_df = pd.read_csv(io.StringIO("\n".join(expr_lines)), sep="\t", index_col=0)
    expr_df = _normalize_expression(expr_df)

    if sample_ids and len(sample_ids) == expr_df.shape[1]:
        expr_df.columns = sample_ids

    metadata = {sample_id: [] for sample_id in expr_df.columns}
    for line in meta_lines:
        parts = line.split("\t")
        for idx, item in enumerate(parts[1:]):
            if idx >= len(expr_df.columns):
                continue
            metadata[expr_df.columns[idx]].append(item.strip('"').strip())

    return expr_df, metadata


def _extract_clinical_record(sample_id: str, characteristics: list[str], dataset_id: str) -> dict[str, object]:
    record: dict[str, object] = {
        "sample_id": sample_id,
        "dataset": dataset_id,
        "age": np.nan,
        "grade": np.nan,
        "ER_status": np.nan,
        "PR_status": np.nan,
        "HER2_status": np.nan,
        "HR_status": np.nan,
        "pCR": np.nan,
    }

    for characteristic in characteristics:
        if ":" not in characteristic:
            continue
        key, value = characteristic.split(":", 1)
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()

        if key in {"age_years", "age", "age_at_diagnosis"}:
            record["age"] = _parse_float(value)
        elif key in {"grade", "bmngrd", "tumor_grade"}:
            record["grade"] = _parse_float(value)
        elif key in {"er_status_ihc", "er_status", "estrogen_receptor"}:
            record["ER_status"] = _to_binary(value)
        elif key in {"pr_status_ihc", "pr_status", "progesterone_receptor"}:
            record["PR_status"] = _to_binary(value)
        elif key in {"her2_status", "her2_status_ihc", "her2_status_by_ihc", "her2_status_by_fish"}:
            record["HER2_status"] = _to_binary(value)
        elif key in {"pathologic_response_pcr_rd", "pcr_vs_rd", "pathological_response", "pcr"}:
            record["pCR"] = _to_binary(value)
        elif key == "hr":
            record["HR_status"] = _to_binary(value)

    er = record["ER_status"]
    pr = record["PR_status"]
    if pd.isna(record["HR_status"]):
        positives = [value for value in [er, pr] if not pd.isna(value)]
        if positives:
            record["HR_status"] = float(max(positives))

    return record


def extract_geo_clinical(metadata: dict[str, list[str]], dataset_id: str) -> pd.DataFrame:
    rows = [
        _extract_clinical_record(sample_id, characteristics, dataset_id)
        for sample_id, characteristics in metadata.items()
    ]
    clinical = pd.DataFrame(rows).drop_duplicates(subset="sample_id")
    clinical = clinical.set_index("sample_id").sort_index()
    clinical["pCR"] = pd.to_numeric(clinical["pCR"], errors="coerce")
    clinical = clinical.loc[clinical["pCR"].isin([0.0, 1.0])]
    clinical["pCR"] = clinical["pCR"].astype(int)
    return clinical


def _read_annotation_frame(path: Path) -> pd.DataFrame:
    compression = "gzip" if path.suffix == ".gz" else None
    skip_rows = 0
    opener = gzip.open if compression == "gzip" else open

    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("!platform_table_begin"):
                skip_rows += 1
                break
            skip_rows += 1

    return pd.read_csv(
        path,
        sep="\t",
        skiprows=skip_rows,
        compression=compression,
        comment="!",
        low_memory=False,
    )


def load_probe_annotation(path: Path) -> dict[str, str]:
    annotation = _read_annotation_frame(path)
    id_col = annotation.columns[0]
    symbol_col = next(
        (
            column
            for column in [
                "Gene Symbol",
                "Gene symbol",
                "gene_symbol",
                "GENE_SYMBOL",
                "Symbol",
            ]
            if column in annotation.columns
        ),
        None,
    )
    if symbol_col is None:
        raise KeyError(f"Could not find gene symbol column in {path}")

    mapping: dict[str, str] = {}
    subset = annotation[[id_col, symbol_col]].dropna()
    for _, row in subset.iterrows():
        probe = str(row[id_col]).strip()
        gene = str(row[symbol_col]).strip().split("///")[0].split("//")[0].strip()
        if gene and gene not in {"---", "NA", "nan"}:
            mapping[probe] = gene.upper()
    return mapping


def collapse_probes_to_genes(expr_df: pd.DataFrame, probe_to_gene: dict[str, str]) -> pd.DataFrame:
    gene_expr = expr_df.copy()
    gene_expr["gene_symbol"] = gene_expr.index.map(probe_to_gene.get)
    gene_expr = gene_expr.dropna(subset=["gene_symbol"])
    gene_expr["gene_symbol"] = gene_expr["gene_symbol"].astype(str).str.upper()
    gene_expr = gene_expr.groupby("gene_symbol").max(numeric_only=True)
    return gene_expr


def _find_first_existing(candidates: tuple[Path, ...]) -> Path | None:
    return next((path for path in candidates if path.exists()), None)


def build_geo_training_dataset(
    dataset_ids: tuple[str, ...] = TRAIN_GEO_DATASETS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    annotation_path = _find_first_existing(GPL96_ANNOTATION_CANDIDATES)
    if annotation_path is None:
        raise FileNotFoundError("GPL96 annotation file is required for GEO training data")
    probe_map = load_probe_annotation(annotation_path)

    expr_parts: list[pd.DataFrame] = []
    clinical_parts: list[pd.DataFrame] = []
    batch_parts: list[pd.Series] = []
    common_genes: set[str] | None = None

    for dataset_id in dataset_ids:
        path = GEO_DIR / f"{dataset_id}_series_matrix.txt.gz"
        if not path.exists():
            raise FileNotFoundError(f"Missing GEO series matrix: {path}")

        probe_expr, metadata = parse_geo_series_matrix(path)
        clinical = extract_geo_clinical(metadata, dataset_id)
        gene_expr = collapse_probes_to_genes(probe_expr, probe_map)

        common_samples = clinical.index.intersection(gene_expr.columns)
        if len(common_samples) == 0:
            raise ValueError(f"No overlapping clinical/expression samples in {dataset_id}")

        prefixed_samples = [f"{dataset_id}:{sample_id}" for sample_id in common_samples]
        gene_expr = gene_expr.loc[:, common_samples].copy()
        gene_expr.columns = prefixed_samples

        clinical = clinical.loc[common_samples].copy()
        clinical.index = prefixed_samples
        clinical["dataset"] = dataset_id

        if common_genes is None:
            common_genes = set(gene_expr.index)
        else:
            common_genes &= set(gene_expr.index)

        expr_parts.append(gene_expr)
        clinical_parts.append(clinical)
        batch_parts.append(pd.Series(dataset_id, index=prefixed_samples, name="batch"))

    if not expr_parts or common_genes is None:
        raise ValueError("No GEO training datasets were assembled")

    gene_order = sorted(common_genes)
    expression = pd.concat([part.loc[gene_order] for part in expr_parts], axis=1)
    clinical = pd.concat(clinical_parts, axis=0).loc[expression.columns]
    batch = pd.concat(batch_parts).loc[expression.columns]
    return expression, clinical, batch


def build_ispy2_gene_level_external_dataset() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    annotation_path = _find_first_existing(GPL20078_ANNOTATION_CANDIDATES)
    if annotation_path is None or not ISPY2_SERIES_PATH.exists():
        return None

    probe_map = load_probe_annotation(annotation_path)
    probe_expr, metadata = parse_geo_series_matrix(ISPY2_SERIES_PATH)
    clinical = extract_geo_clinical(metadata, ISPY2_SERIES_ID)
    gene_expr = collapse_probes_to_genes(probe_expr, probe_map)

    common_samples = clinical.index.intersection(gene_expr.columns)
    if len(common_samples) == 0:
        return None

    clinical = clinical.loc[common_samples].copy()
    clinical.index = [f"{ISPY2_SERIES_ID}:{sample_id}" for sample_id in common_samples]
    clinical["dataset"] = ISPY2_SERIES_ID

    gene_expr = gene_expr.loc[:, common_samples].copy()
    gene_expr.columns = clinical.index
    return gene_expr, clinical


def load_ispy2_biomarker_external() -> pd.DataFrame:
    if not ISPY2_BIOMARKER_PATH.exists():
        raise FileNotFoundError(f"Missing I-SPY2 biomarker table: {ISPY2_BIOMARKER_PATH}")

    raw = pd.read_csv(ISPY2_BIOMARKER_PATH)
    renamed = raw.rename(
        columns={
            "ResearchID": "sample_id",
            "Arm": "arm",
            "HR.status..0.HR....1..HR..": "HR_status",
            "HER2.status..0..HER2...1..HER2..": "HER2_status",
            "pCR.status": "pCR",
            "B.cells_sig": "B_cell",
            "Dendritic.cells_sig": "Dendritic",
            "T.cells_sig": "T_cells_sig",
            "CD68": "CD68",
            "PD1": "PD1",
            "PDL1": "PDL1",
            "STAT1_sig": "STAT1_sig",
            "TIS_sig": "TIS_sig",
        }
    )

    keep_columns = [
        "sample_id",
        "arm",
        "HR_status",
        "HER2_status",
        "pCR",
        "B_cell",
        "Dendritic",
        "T_cells_sig",
        "CD68",
        "PD1",
        "PDL1",
        "STAT1_sig",
        "TIS_sig",
    ]
    external = renamed[keep_columns].copy()
    external = external.loc[external["pCR"].isin([0, 1])].copy()
    external["sample_id"] = external["sample_id"].astype(str)
    external = external.set_index("sample_id")

    numeric_columns = [column for column in keep_columns if column not in {"sample_id", "arm"}]
    external[numeric_columns] = external[numeric_columns].apply(pd.to_numeric, errors="coerce")
    external["pCR"] = external["pCR"].astype(int)
    return external


def load_tcga_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    expression_path = PROCESSED_DIR / "rnaseq_processed.csv"
    clinical_path = PROCESSED_DIR / "clinical_cleaned.csv"
    if not expression_path.exists() or not clinical_path.exists():
        raise FileNotFoundError("Processed TCGA files are required before running survival analysis")

    expression = pd.read_csv(expression_path, index_col=0)
    expression.columns = expression.columns.astype(str).str.upper()
    expression = expression.T
    expression.columns = expression.columns.astype(str).str.upper()

    clinical = pd.read_csv(clinical_path, index_col=0)
    clinical.index = clinical.index.astype(str)
    clinical["HR_status"] = clinical[["ER_status", "PR_status"]].max(axis=1, skipna=True)

    common_samples = clinical.index.intersection(expression.columns)
    expression = expression.loc[:, common_samples]
    clinical = clinical.loc[common_samples]
    return expression, clinical
