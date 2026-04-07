from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RNASEQ_FILE = RAW_DIR / "tcga_brca_rnaseq_tpm.tsv"
CLINICAL_FILE = RAW_DIR / "tcga_brca_clinical.tsv"

REQUIRED_CLINICAL_COLUMNS = [
    "patient_id",
    "subtype_PAM50",
    "age_at_diagnosis",
    "tumor_stage_T",
    "tumor_stage_N",
    "ER_status",
    "PR_status",
    "HER2_status",
    "Ki67",
    "pCR",
    "DFS_months",
    "DFS_event",
]

COLUMN_ALIASES: dict[str, list[str]] = {
    "patient_id": [
        "patient_id",
        "case_submitter_id",
        "submitter_id",
        "bcr_patient_barcode",
        "case_id",
    ],
    "subtype_PAM50": ["subtype_PAM50", "pam50", "PAM50", "subtype", "tumor_subtype"],
    "age_at_diagnosis": ["age_at_diagnosis", "age", "diagnosis_age"],
    "tumor_stage_T": ["tumor_stage_T", "ajcc_pathologic_t", "pathologic_T", "t_stage"],
    "tumor_stage_N": ["tumor_stage_N", "ajcc_pathologic_n", "pathologic_N", "n_stage"],
    "ER_status": ["ER_status", "er_status", "er_status_by_ihc"],
    "PR_status": ["PR_status", "pr_status", "pr_status_by_ihc"],
    "HER2_status": ["HER2_status", "her2_status", "her2_status_by_ihc"],
    "Ki67": ["Ki67", "ki67", "proliferation_index"],
    "pCR": ["pCR", "pathologic_complete_response", "response", "binary_response"],
    "DFS_months": [
        "DFS_months",
        "disease_free_survival_months",
        "days_to_new_tumor_event_after_initial_treatment",
    ],
    "DFS_event": [
        "DFS_event",
        "disease_free_status",
        "new_tumor_event",
        "recurrence_event",
    ],
}


def configure_logging() -> None:
    """Configure root logger.

    Args:
        None

    Returns:
        None
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def normalize_patient_id(sample_id: str) -> str:
    """Normalize a sample/case ID to TCGA patient ID style.

    Args:
        sample_id: Raw sample or case identifier.

    Returns:
        Normalized patient identifier.
    """
    if not isinstance(sample_id, str):
        return str(sample_id)
    token = sample_id.strip().replace(".", "-")
    parts = token.split("-")
    if len(parts) >= 3 and parts[0].upper() == "TCGA":
        return "-".join(parts[:3]).upper()
    return token


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _encode_binary_status(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return 1.0 if float(value) >= 0.5 else 0.0
    v = str(value).strip().lower()
    if v in {"positive", "pos", "1", "1.0", "yes", "true", "amplified", "overexpressed"}:
        return 1.0
    if v in {"negative", "neg", "0", "0.0", "no", "false", "not amplified", "normal"}:
        return 0.0
    try:
        numeric_v = float(v)
        return 1.0 if numeric_v >= 0.5 else 0.0
    except ValueError:
        pass
    return np.nan


def _encode_stage(value: object, prefix: str) -> float:
    if pd.isna(value):
        return np.nan
    v = str(value).upper().replace(" ", "")
    if not v.startswith(prefix.upper()):
        return np.nan
    digits = "".join(ch for ch in v if ch.isdigit())
    if digits == "":
        return np.nan
    return float(int(digits[0]))


def _first_present_column(df: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    for col in aliases:
        if col in df.columns:
            return col
    return None


def _prepare_clinical_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        hit = _first_present_column(df, aliases)
        if hit is not None:
            renamed[hit] = canonical

    out = df.rename(columns=renamed)
    for col in REQUIRED_CLINICAL_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    out = out[REQUIRED_CLINICAL_COLUMNS].copy()
    out["patient_id"] = out["patient_id"].astype(str).map(normalize_patient_id)

    out["ER_status"] = out["ER_status"].map(_encode_binary_status)
    out["PR_status"] = out["PR_status"].map(_encode_binary_status)
    out["HER2_status"] = out["HER2_status"].map(_encode_binary_status)
    out["tumor_stage_T"] = out["tumor_stage_T"].map(lambda x: _encode_stage(x, "T"))
    out["tumor_stage_N"] = out["tumor_stage_N"].map(lambda x: _encode_stage(x, "N"))

    out["pCR"] = out["pCR"].map(_encode_binary_status)
    out["DFS_event"] = out["DFS_event"].map(_encode_binary_status)

    continuous = ["age_at_diagnosis", "Ki67", "DFS_months"]
    for col in continuous:
        out[col] = _safe_numeric(out[col])

    out["subtype_PAM50"] = out["subtype_PAM50"].astype(str).replace({"nan": np.nan})
    out["subtype_PAM50"] = out["subtype_PAM50"].str.upper().replace(
        {
            "BASAL-LIKE": "BASAL",
            "LUMINAL A": "LUMA",
            "LUMINAL B": "LUMB",
            "HER2-ENRICHED": "HER2",
            "NORMAL-LIKE": "NORMAL",
        }
    )

    categorical = ["subtype_PAM50"]
    numeric = [
        "age_at_diagnosis",
        "tumor_stage_T",
        "tumor_stage_N",
        "ER_status",
        "PR_status",
        "HER2_status",
        "Ki67",
        "pCR",
        "DFS_months",
        "DFS_event",
    ]

    for col in categorical:
        mode = out[col].mode(dropna=True)
        if not mode.empty:
            out[col] = out[col].fillna(mode.iloc[0])

    for col in numeric:
        med = out[col].median(skipna=True)
        if pd.isna(med):
            med = 0.0
        out[col] = out[col].fillna(med)

    out = out.drop_duplicates(subset=["patient_id"]).set_index("patient_id").sort_index()
    return out


def _download_tcga_clinical_gdc(output_path: Path) -> None:
    """Download TCGA-BRCA clinical table from GDC API.

    Args:
        output_path: Destination TSV path.

    Returns:
        None
    """
    logging.info("Attempting clinical download from GDC API")
    query = {
        "filters": {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "projects.project_id",
                        "value": ["TCGA-BRCA"],
                    },
                }
            ],
        },
        "fields": "case_submitter_id,diagnoses.age_at_diagnosis,diagnoses.ajcc_pathologic_t,diagnoses.ajcc_pathologic_n,diagnoses.days_to_recurrence,diagnoses.tumor_grade,diagnoses.last_known_disease_status,diagnoses.site_of_resection_or_biopsy,demographic.gender,diagnoses.days_to_last_follow_up",
        "format": "JSON",
        "size": "2000",
    }
    response = requests.get("https://api.gdc.cancer.gov/cases", params={"filters": json.dumps(query["filters"]), "fields": query["fields"], "format": "JSON", "size": query["size"]}, timeout=120)
    response.raise_for_status()
    payload = response.json()
    hits = payload.get("data", {}).get("hits", [])

    rows: list[dict[str, object]] = []
    for hit in hits:
        diagnoses = (hit.get("diagnoses") or [{}])[0]
        rows.append(
            {
                "patient_id": hit.get("case_submitter_id"),
                "age_at_diagnosis": diagnoses.get("age_at_diagnosis"),
                "tumor_stage_T": diagnoses.get("ajcc_pathologic_t"),
                "tumor_stage_N": diagnoses.get("ajcc_pathologic_n"),
                "DFS_months": diagnoses.get("days_to_recurrence"),
                "DFS_event": diagnoses.get("last_known_disease_status"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("GDC clinical query returned empty result")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    logging.info("Clinical data downloaded to %s (%d rows)", output_path, len(df))


def _download_tcga_rnaseq_stub(output_path: Path) -> None:
    """Attempt metadata-first GDC RNA-seq retrieval.

    Args:
        output_path: Destination TSV path.

    Returns:
        None

    Raises:
        RuntimeError: When direct matrix retrieval is not available.
    """
    logging.info("Attempting RNA-seq metadata query from GDC API")
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": ["TCGA-BRCA"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "data_type",
                    "value": ["Gene Expression Quantification"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "analysis.workflow_type",
                    "value": ["HTSeq - FPKM-UQ"],
                },
            },
        ],
    }
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.case_submitter_id",
        "format": "JSON",
        "size": "5",
    }
    response = requests.get("https://api.gdc.cancer.gov/files", params=params, timeout=120)
    response.raise_for_status()
    hits = response.json().get("data", {}).get("hits", [])
    if not hits:
        raise RuntimeError("No TCGA-BRCA RNA-seq files matched GDC filter")
    example = hits[0]
    msg = (
        "Direct cohort matrix assembly from raw GDC HTSeq files is not available in this offline run. "
        f"Found file example: {example.get('file_name')} (id={example.get('file_id')}). "
        "Provide pre-aggregated tcga_brca_rnaseq_tpm.tsv in data/raw/."
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (output_path.parent / "gdc_rnaseq_manifest_preview.json").write_text(
        json.dumps(hits, indent=2), encoding="utf-8"
    )
    raise RuntimeError(msg)


def ensure_raw_data_available(rnaseq_path: Path, clinical_path: Path) -> None:
    """Ensure required raw files exist; attempt programmatic downloads if absent.

    Args:
        rnaseq_path: RNA-seq matrix path.
        clinical_path: Clinical table path.

    Returns:
        None
    """
    if not clinical_path.exists():
        _download_tcga_clinical_gdc(clinical_path)

    if not rnaseq_path.exists():
        _download_tcga_rnaseq_stub(rnaseq_path)


def load_rnaseq(path: str) -> pd.DataFrame:
    """Load RNA-seq TPM matrix.

    Args:
        path: Path to RNA-seq matrix file.

    Returns:
        DataFrame in samples x genes format with normalized patient IDs.
    """
    src = Path(path)
    logging.info("Loading RNA-seq matrix from %s", src)
    df = pd.read_csv(src, sep="\t", low_memory=False)
    if df.empty:
        raise ValueError("RNA-seq matrix is empty")

    gene_col = df.columns[0]
    if gene_col.lower() in {"gene", "gene_id", "symbol", "ensembl_id"}:
        df = df.set_index(gene_col)
    else:
        df = df.set_index(df.columns[0])

    if "gene_type" in df.columns:
        mask = df["gene_type"].astype(str).str.lower().eq("protein_coding")
        df = df.loc[mask].drop(columns=["gene_type"])
    else:
        gene_idx = pd.Index(df.index.astype(str))
        ensg_mask = gene_idx.str.startswith("ENSG")
        if ensg_mask.any():
            df = df.loc[ensg_mask]

    expr = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    expr = np.log2(expr + 1.0)

    expr = expr.T
    expr.index = expr.index.map(normalize_patient_id)
    expr = expr.groupby(level=0).mean().sort_index()

    assert expr.index.is_unique, "Duplicate patient IDs in RNA-seq table"
    assert np.isfinite(expr.to_numpy()).all(), "RNA-seq matrix contains non-finite values"

    logging.info("RNA-seq loaded: %d samples x %d genes", expr.shape[0], expr.shape[1])
    return expr


def load_clinical(path: str) -> pd.DataFrame:
    """Load and clean clinical data table.

    Args:
        path: Path to clinical metadata TSV file.

    Returns:
        Cleaned clinical DataFrame indexed by patient_id.
    """
    src = Path(path)
    logging.info("Loading clinical metadata from %s", src)
    df = pd.read_csv(src, sep="\t", low_memory=False)
    cleaned = _prepare_clinical_columns(df)

    assert cleaned.index.is_unique, "Duplicate patient IDs in clinical table"
    assert not cleaned.isna().any().any(), "Clinical table contains NaN after imputation"
    assert set(REQUIRED_CLINICAL_COLUMNS) - {"patient_id"} <= set(cleaned.columns), "Missing required clinical columns"

    logging.info("Clinical table loaded: %d patients x %d columns", cleaned.shape[0], cleaned.shape[1])
    return cleaned


def merge_datasets(rnaseq: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    """Merge RNA-seq and clinical tables on patient_id.

    Args:
        rnaseq: Samples x genes RNA-seq matrix.
        clinical: Cleaned clinical table indexed by patient_id.

    Returns:
        Merged DataFrame.
    """
    assert rnaseq.index.is_unique, "RNA-seq patient IDs must be unique"
    assert clinical.index.is_unique, "Clinical patient IDs must be unique"

    merged = clinical.join(rnaseq, how="inner")
    assert merged.index.is_unique, "Merged dataset has duplicate patient IDs"
    if merged.empty:
        raise ValueError("Merged dataset is empty; check patient ID harmonization")

    logging.info("Merged dataset: %d samples x %d total features", merged.shape[0], merged.shape[1])
    return merged


def main() -> None:
    """Run data loading pipeline and persist processed files.

    Args:
        None

    Returns:
        None
    """
    configure_logging()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ensure_raw_data_available(RNASEQ_FILE, CLINICAL_FILE)

    rnaseq = load_rnaseq(str(RNASEQ_FILE))
    clinical = load_clinical(str(CLINICAL_FILE))
    merged = merge_datasets(rnaseq, clinical)

    rnaseq.to_csv(PROCESSED_DIR / "rnaseq_processed.csv")
    clinical.to_csv(PROCESSED_DIR / "clinical_cleaned.csv")
    merged.to_csv(PROCESSED_DIR / "merged_dataset.csv")

    logging.info("Saved processed RNA-seq to %s", PROCESSED_DIR / "rnaseq_processed.csv")
    logging.info("Saved processed clinical to %s", PROCESSED_DIR / "clinical_cleaned.csv")
    logging.info("Saved merged dataset to %s", PROCESSED_DIR / "merged_dataset.csv")


if __name__ == "__main__":
    main()
