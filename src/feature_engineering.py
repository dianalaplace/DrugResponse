from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_loader import PROCESSED_DIR, configure_logging

GENOMIC_CLINICAL_FEATURES = [
    "HER2_status",
    "ER_status",
    "PR_status",
    "Ki67",
    "subtype_PAM50",
    "age_at_diagnosis",
    "tumor_stage_T",
    "tumor_stage_N",
]

TME_FEATURES = [
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


def build_feature_matrix(clinical: pd.DataFrame, tme_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build baseline and full matrices for ablation.

    Args:
        clinical: Clinical dataframe indexed by patient_id.
        tme_features: TME feature dataframe indexed by patient_id.

    Returns:
        Tuple of (X_baseline, X_full, y_pcr, y_survival).
    """
    for col in GENOMIC_CLINICAL_FEATURES + ["pCR", "DFS_months", "DFS_event"]:
        if col not in clinical.columns:
            raise KeyError(f"Missing clinical column: {col}")

    X_baseline = clinical[GENOMIC_CLINICAL_FEATURES].copy()

    missing_tme = [c for c in TME_FEATURES if c not in tme_features.columns]
    if missing_tme:
        logging.warning("Missing TME features in matrix; filling zeros: %s", missing_tme)
        for col in missing_tme:
            tme_features[col] = 0.0

    tme_subset = tme_features[TME_FEATURES].copy()
    X_full = X_baseline.join(tme_subset, how="inner")

    y_pcr = clinical.loc[X_full.index, "pCR"].astype(int)
    y_survival = clinical.loc[X_full.index, ["DFS_months", "DFS_event"]].copy()

    X_baseline = X_baseline.loc[X_full.index]

    assert len(X_baseline) == len(X_full) == len(y_pcr) == len(y_survival)
    assert not y_survival.isna().any().any(), "Survival labels contain NaN"

    return X_baseline, X_full, y_pcr, y_survival


def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns and standard-scale all features.

    Args:
        X: Raw feature matrix.

    Returns:
        Preprocessed feature matrix.
    """
    categorical_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

    X_scaled = X_encoded.copy()
    binary_cols: list[str] = []
    for col in X_encoded.columns:
        values = pd.Series(X_encoded[col]).dropna().unique()
        if len(values) > 0 and set(values).issubset({0, 1, 0.0, 1.0, True, False}):
            binary_cols.append(col)
    continuous_cols = [c for c in X_encoded.columns if c not in binary_cols]

    if continuous_cols:
        scaler = StandardScaler()
        X_scaled[continuous_cols] = scaler.fit_transform(X_encoded[continuous_cols])

    assert not X_scaled.isna().any().any(), "NaN values remain after preprocessing"
    return X_scaled


def main() -> None:
    """Create and save model feature matrices and targets.

    Args:
        None

    Returns:
        None
    """
    configure_logging()

    clinical_path = PROCESSED_DIR / "clinical_cleaned.csv"
    tme_path = PROCESSED_DIR / "tme_features.csv"

    if not clinical_path.exists():
        raise FileNotFoundError(f"Missing clinical file: {clinical_path}")
    if not tme_path.exists():
        raise FileNotFoundError(f"Missing TME feature file: {tme_path}")

    clinical = pd.read_csv(clinical_path, index_col=0)
    tme = pd.read_csv(tme_path, index_col=0)

    X_baseline, X_full, y_pcr, y_survival = build_feature_matrix(clinical, tme)

    X_baseline_p = preprocess_features(X_baseline)
    X_full_p = preprocess_features(X_full)

    X_baseline_p.to_csv(PROCESSED_DIR / "X_baseline.csv")
    X_full_p.to_csv(PROCESSED_DIR / "X_full.csv")
    y_pcr.to_csv(PROCESSED_DIR / "y_pcr.csv", header=["pCR"])
    y_survival.to_csv(PROCESSED_DIR / "y_survival.csv")

    logging.info("Saved baseline features to %s", PROCESSED_DIR / "X_baseline.csv")
    logging.info("Saved full features to %s", PROCESSED_DIR / "X_full.csv")
    logging.info("Saved pCR labels to %s", PROCESSED_DIR / "y_pcr.csv")
    logging.info("Saved survival labels to %s", PROCESSED_DIR / "y_survival.csv")


if __name__ == "__main__":
    main()
