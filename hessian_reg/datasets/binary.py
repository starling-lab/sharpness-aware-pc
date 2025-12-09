"""
hessian_reg.datasets.binary
===========================

Loader for the *Density‑Estimation Benchmark Datasets* (DEBD) — the canonical
20‑dataset binary collection used in SPN / PC papers.

Features
--------
* First call **clones** the DEBD repo under  ``data/debd``  (≈ 6 MB).
* Loads the **train** and **valid** splits into PyTorch tensors.
* Returns two DataLoaders; the **test** split remains on disk for evaluation.
"""

from __future__ import annotations

import csv
import os
import subprocess
import pathlib
from typing import Dict, Tuple, List

import torch
from torch.utils.data import TensorDataset, DataLoader

# --------------------------------------------------------------------------- #
# Dataset registry
# --------------------------------------------------------------------------- #
BINARY_DATASETS: List[str] = [
    "accidents",
    "ad",
    "baudio",
    "bbc",
    "bnetflix",
    "book",
    "c20ng",
    "cr52",
    "cwebkb",
    "dna",
    "jester",
    "kdd",
    "kosarek",
    "moviereview",
    "msnbc",
    "msweb",
    "nltcs",
    "plants",
    "pumsb_star",
    "tmovie",
    "tretail",
    "voting",
]

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = pathlib.Path(os.getenv("DATA_ROOT", PROJECT_ROOT / "data"))
DEBD_DIR = DATA_ROOT / "debd"

REPO_URL = "https://github.com/arranger1044/DEBD"


# --------------------------------------------------------------------------- #
# Helper: clone repo once
# --------------------------------------------------------------------------- #
def download_debd(data_root: pathlib.Path | str | None = None) -> pathlib.Path:
    """
    Clone the DEBD repository into ``data/debd`` (or ``data_root``/debd) and
    return the path. Reuses an existing checkout if present.
    """
    target_root = pathlib.Path(data_root) if data_root is not None else DATA_ROOT
    debd_dir = target_root / "debd"
    if debd_dir.exists():
        return debd_dir

    print(f"[DEBD] cloning repository into {debd_dir} …")
    target_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", REPO_URL, str(debd_dir)], check=True)
    # Checkout fixed commit for reproducibility
    subprocess.run(
        ["git", "checkout", "80a4906dcf3b3463370f904efa42c21e8295e85c"],
        cwd=debd_dir,
        check=True,
    )
    # Remove .git to save space
    subprocess.run(["rm", "-rf", ".git"], cwd=debd_dir, check=True)
    print("[DEBD] repo cloned to", debd_dir)
    return debd_dir


def _maybe_clone_debd() -> None:
    """Backwards compatible wrapper around :func:`download_debd`."""
    download_debd()


# --------------------------------------------------------------------------- #
# CSV → Tensor
# --------------------------------------------------------------------------- #
def _load_csv(path: pathlib.Path, dtype=torch.float32) -> torch.Tensor:
    with open(path) as fh:
        reader = csv.reader(fh, delimiter=",")
        rows = [[float(x) for x in row] for row in reader if row]
    return torch.tensor(rows, dtype=dtype)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def get_binary_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns
    -------
    train_dl, valid_dl, test_dl  (DataLoader)
    """
    name = cfg["name"].lower()
    if name not in BINARY_DATASETS:
        raise ValueError(f"{name} not in supported DEBD list")

    debd_dir = download_debd()

    base = debd_dir / "datasets" / name
    train_file = base / f"{name}.train.data"
    valid_file = base / f"{name}.valid.data"
    test_file = base / f"{name}.test.data"

    x_train = _load_csv(train_file)
    x_valid = _load_csv(valid_file)
    x_test = _load_csv(test_file)

    bs = cfg["batch_size"]
    dev = cfg.get("device", "cpu")
    n_samples = cfg.get("n_samples", -1)
    # Check if n_samples is a fraction
    if n_samples > 0 and n_samples <= 1:
        n_samples = int(len(x_train) * n_samples)
    x_train = x_train[:n_samples]
    train_dl = DataLoader(
        TensorDataset(x_train.to(dev)), batch_size=bs, shuffle=True
    )
    valid_dl = DataLoader(
        TensorDataset(x_valid.to(dev)), batch_size=bs, shuffle=False
    )
    test_dl = DataLoader(
        TensorDataset(x_test.to(dev)), batch_size=bs, shuffle=False
    )
    return train_dl, valid_dl, test_dl
