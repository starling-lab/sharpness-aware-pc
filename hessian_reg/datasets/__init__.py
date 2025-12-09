from .synthetic import get_synth_dataloaders, SYNTH_DATASETS
from .binary import get_binary_dataloaders, BINARY_DATASETS

SUPPORTED_SYNTHETIC_DATASETS = SYNTH_DATASETS
SUPPORTED_BINARY_DATASETS = BINARY_DATASETS
SUPPORTED_ALL_DATASETS = SUPPORTED_SYNTHETIC_DATASETS + SUPPORTED_BINARY_DATASETS


def get_dataloaders(cfg):
    name = cfg["name"] if isinstance(cfg, dict) else cfg.name
    if name in SUPPORTED_SYNTHETIC_DATASETS:
        return get_synth_dataloaders(cfg)
    if name in SUPPORTED_BINARY_DATASETS:
        return get_binary_dataloaders(cfg)
    raise ValueError(f"Unsupported dataset: {name}")
