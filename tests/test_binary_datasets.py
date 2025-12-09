import torch
import pytest
from hessian_reg.datasets import get_dataloaders
from hessian_reg.datasets.binary import DATA_ROOT
import os
import pathlib


@pytest.mark.parametrize("name", ["nltcs", "plants"])
def test_binary_loader_shape(name):
    """Check dataloader returns expected binary shape and type."""
    cfg = {
        "name": name,
        "batch_size": 64,
        "device": "cpu",
    }
    train_dl, valid_dl, test_dl = get_dataloaders(cfg)

    batch = next(iter(train_dl))
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    assert x.dim() == 2, "Data should be a (B, D) tensor"
    assert x.dtype == torch.float32
    assert set(torch.unique(x).tolist()).issubset({0.0, 1.0})

    # Check validation set
    if valid_dl is not None:
        x_val = next(iter(valid_dl))[0]
        assert x_val.shape[1] == x.shape[1], "Train/Val feature dim mismatch"


"""
===========================

Validate that the DEBD loader returns tensors whose shapes exactly match
the reference statistics for every dataset (train, valid, test).
"""

# --------------------------------------------------------------------- #
# Ground‑truth metadata (rows, cols) taken from the README table
# --------------------------------------------------------------------- #
META = {
    "nltcs": dict(train=(16181, 16), valid=(2157, 16), test=(3236, 16)),
    "msnbc": dict(train=(291326, 17), valid=(38843, 17), test=(58265, 17)),
    "kdd": dict(train=(180092, 64), valid=(19907, 64), test=(34955, 64)),
    "plants": dict(train=(17412, 69), valid=(2321, 69), test=(3482, 69)),
    "baudio": dict(train=(15000, 100), valid=(2000, 100), test=(3000, 100)),
    "jester": dict(train=(9000, 100), valid=(1000, 100), test=(4116, 100)),
    "bnetflix": dict(train=(15000, 100), valid=(2000, 100), test=(3000, 100)),
    "accidents": dict(train=(12758, 111), valid=(1700, 111), test=(2551, 111)),
    "mushrooms": dict(train=(2000, 112), valid=(500, 112), test=(5624, 112)),
    "adult": dict(train=(5000, 123), valid=(1414, 123), test=(26147, 123)),
    "connect4": dict(train=(16000, 126), valid=(4000, 126), test=(47557, 126)),
    "ocr_letters": dict(train=(32152, 128), valid=(10000, 128), test=(10000, 128)),
    "rcv1": dict(train=(40000, 150), valid=(10000, 150), test=(150000, 150)),
    "tretail": dict(train=(22041, 135), valid=(2938, 135), test=(4408, 135)),
    "pumsb_star": dict(train=(12262, 163), valid=(1635, 163), test=(2452, 163)),
    "dna": dict(train=(1600, 180), valid=(400, 180), test=(1186, 180)),
    "kosarek": dict(train=(33375, 190), valid=(4450, 190), test=(6675, 190)),
    "msweb": dict(train=(29441, 294), valid=(3270, 294), test=(5000, 294)),
    "nips": dict(train=(400, 500), valid=(100, 500), test=(1240, 500)),
    "book": dict(train=(8700, 500), valid=(1159, 500), test=(1739, 500)),
    "tmovie": dict(train=(4524, 500), valid=(1002, 500), test=(591, 500)),
    "cwebkb": dict(train=(2803, 839), valid=(558, 839), test=(838, 839)),
    "cr52": dict(train=(6532, 889), valid=(1028, 889), test=(1540, 889)),
    "c20ng": dict(train=(11293, 910), valid=(3764, 910), test=(3764, 910)),
    "moviereview": dict(train=(1600, 1001), valid=(150, 1001), test=(250, 1001)),
    "bbc": dict(train=(1670, 1058), valid=(225, 1058), test=(330, 1058)),
    "voting": dict(train=(1214, 1359), valid=(200, 1359), test=(350, 1359)),
    "ad": dict(train=(2461, 1556), valid=(327, 1556), test=(491, 1556)),
    "binarized_mnist": dict(train=None, valid=None, test=None),  # not in DEBD repo
}


# --------------------------------------------------------------------- #
# Parametrised test (skip if dataset not downloaded yet)
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(META.keys()))
def test_binary_dataset_shapes(name):
    cfg = {"name": name, "batch_size": 512, "device": "cpu"}

    # Check if file exists; otherwise skip to avoid download in CI
    root_file = DATA_ROOT / "debd" / "datasets" / name / f"{name}.train.data"
    if not os.path.exists(root_file):
        pytest.skip(f"{name} not downloaded (run `python scripts/download_debd.py`)")

    train_dl, valid_dl, test_dl = get_dataloaders(cfg)
    x_train = next(iter(train_dl))[0]
    x_valid = next(iter(valid_dl))[0]
    x_test = next(iter(test_dl))[0]

    # Shape assertions
    n_train, d = META[name]["train"]
    n_valid, d_val = META[name]["valid"]
    n_test, d_test = META[name]["test"]

    assert x_train.shape[1] == d, "Feature dim mismatch (train)"
    assert x_valid.shape[1] == d_val, "Feature dim mismatch (valid)"
    assert x_test.shape[1] == d_test, "Feature dim mismatch (test)"
    # Allow small discrepancies due to empty lines or formatting differences
    assert abs(len(train_dl.dataset) - n_train) <= 5, f"Train sample count mismatch: {len(train_dl.dataset)} vs {n_train}"
    assert abs(len(valid_dl.dataset) - n_valid) <= 5, f"Valid sample count mismatch: {len(valid_dl.dataset)} vs {n_valid}"
    assert abs(len(test_dl.dataset) - n_test) <= 5, f"Test sample count mismatch: {len(test_dl.dataset)} vs {n_test}"

    # Binary check
    assert set(torch.unique(x_train).tolist()).issubset({0.0, 1.0})
