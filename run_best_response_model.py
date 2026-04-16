#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from src.best_response_model import run_best_response_model
from src.data_loader import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Best available clinically oriented response model on real pCR GEO cohorts")
    parser.add_argument("--top-genes", type=int, default=500, help="Top variable genes per training fold")
    parser.add_argument("--outer-splits", type=int, default=5, help="Outer CV folds")
    parser.add_argument("--inner-splits", type=int, default=3, help="Inner CV folds")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap iterations for confidence intervals")
    parser.add_argument("--output-dir", default="results/best_response_model", help="Output directory")
    args = parser.parse_args()

    configure_logging()
    run_best_response_model(
        top_n_genes=args.top_genes,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        n_bootstrap=args.bootstrap,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
