import pytest
from hessian_reg.datasets import get_dataloaders, SUPPORTED_SYNTHETIC_DATASETS


@pytest.mark.parametrize("name", SUPPORTED_SYNTHETIC_DATASETS)
def test_synthetic_datasets(name):
    cfg = {"name": name, "n_samples": 512, "batch_size": 64, "device": "cpu"}
    train_dl, valid_dl, test_dl = get_dataloaders(cfg)
    x = next(iter(train_dl))[0]
    # Check shape - most are 2D but some 3D datasets exist
    assert x.shape[1] in [2, 3]
    assert abs(x.mean()).item() < 0.5
    assert abs(x.std() - 1.0).item() < 0.5
