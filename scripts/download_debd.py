#!/usr/bin/env python
"""Fetch the DEBD binary datasets into ``data/debd`` for reproducible runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from hessian_reg.datasets import download_debd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Density-Estimation Benchmark Datasets (DEBD)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Root directory for downloads (default: <repo>/data).",
    )
    args = parser.parse_args()

    dest = download_debd(args.data_dir)
    print(f"DEBD datasets are available at: {dest}")


if __name__ == "__main__":
    main()
