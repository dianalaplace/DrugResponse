"""Batch correction for multi-cohort expression data.

Supports ComBat (pycombat) and PCA-residual correction methods
for removing technical batch effects between TCGA and GEO datasets.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

try:
    from pycombat import pycombat
    COMBAT_AVAILABLE = True
except ImportError:
    COMBAT_AVAILABLE = False

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def correct_batch(expr: pd.DataFrame,
                  batch_labels: pd.Series,
                  method: str = "auto") -> pd.DataFrame:
    """Apply batch correction to expression matrix.

    Args:
        expr: samples x genes expression matrix.
        batch_labels: Series mapping sample_id -> batch_id (str).
        method: 'combat', 'pca_residual', or 'auto'.

    Returns:
        Batch-corrected expression matrix, same shape as input.
    """
    if method == "auto":
        method = "combat" if COMBAT_AVAILABLE else "pca_residual"

    logging.info("Batch correction method: %s", method)
    batch_dist = batch_labels.value_counts().to_dict()
    logging.info("Batch distribution: %s", batch_dist)

    if len(batch_dist) < 2:
        logging.info("Only one batch detected; skipping correction")
        return expr

    if method == "combat" and COMBAT_AVAILABLE:
        corrected = pycombat(
            expr.T,
            batch_labels.reindex(expr.index).fillna("unknown")
        )
        return corrected.T

    elif method == "pca_residual":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(expr)

        n_components = min(20, expr.shape[1] - 1, expr.shape[0] - 1)
        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(X_scaled)

        batch_enc = pd.Categorical(
            batch_labels.reindex(expr.index).fillna("unknown")
        ).codes
        correlations = np.abs([
            np.corrcoef(pcs[:, i], batch_enc)[0, 1]
            for i in range(pcs.shape[1])
        ])
        batch_pcs = np.where(correlations > 0.3)[0]
        logging.info("PCs correlated with batch (r>0.3): %s (r=%s)",
                     batch_pcs.tolist(),
                     [f"{correlations[i]:.3f}" for i in batch_pcs])

        if len(batch_pcs) == 0:
            logging.info("No strong batch PCs found; skipping correction")
            return expr

        reg = LinearRegression()
        reg.fit(pcs[:, batch_pcs], X_scaled)
        X_corrected = X_scaled - reg.predict(pcs[:, batch_pcs])
        corrected = scaler.inverse_transform(X_corrected)
        return pd.DataFrame(corrected, index=expr.index,
                            columns=expr.columns)

    else:
        logging.warning("No correction applied")
        return expr


def main():
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s")

    rnaseq = pd.read_csv(PROCESSED / "rnaseq_processed.csv", index_col=0)
    batch = rnaseq.index.map(
        lambda x: "tcga" if str(x).startswith("TCGA") else "geo"
    )
    batch = pd.Series(batch, index=rnaseq.index)

    corrected = correct_batch(rnaseq, batch)
    corrected.to_csv(PROCESSED / "rnaseq_batch_corrected.csv")
    logging.info("Saved batch-corrected matrix: %s", corrected.shape)


if __name__ == "__main__":
    main()
