"""
hessian_reg.datasets.synthetic
==============================

Synthetic 2-D *and* 3-D toy datasets with automatic 80 / 10 / 10
train-valid-test splits.

`get_synth_dataloaders(cfg)` → (train_dl, valid_dl, test_dl)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Dict

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn import datasets

# --------------------------------------------------------------------------- #
# Public constants
# --------------------------------------------------------------------------- #
DATASET_SIZE = 1000

# All supported dataset names (lower-case)
SYNTH_DATASETS = [
    # ─── 2-D ─────────────────────────────────────────────────────────────── #
    "gmm",
    "two_moons",
    "two_circles",
    "spiral",
    "pinwheel",
    "checkerboard",
    "8gaussians",
    "swissroll",
    # ─── 3-D ─────────────────────────────────────────────────────────────── #
    "knotted",
    "helix",
    "bent_lissajous",
    "disjoint_circles",
    "twisted_eight",
    "interlocked_circles",
]


# --------------------------------------------------------------------------- #
# Config dataclass
# --------------------------------------------------------------------------- #
@dataclass
class SynthCfg:
    name: str
    n_samples: int
    batch_size: int
    device: str = "cpu"
    noise: float = 0.05
    k: int = 6
    seed: int = 0

    def get(self, key: str, default=None):
        return getattr(self, key, default)


# --------------------------------------------------------------------------- #
# 2-D generators (unchanged)
# --------------------------------------------------------------------------- #
def _gmm(cfg: SynthCfg):
    means = torch.randn(cfg.k, 2) * 4
    comps = torch.randint(0, cfg.k, (cfg.n_samples,))
    return means[comps] + 0.3 * torch.randn(cfg.n_samples, 2)


def _two_moons(cfg: SynthCfg):
    x, _ = datasets.make_moons(cfg.n_samples, noise=cfg.noise)
    return torch.as_tensor(x, dtype=torch.float32)


def _two_circles(cfg: SynthCfg):
    x, _ = datasets.make_circles(cfg.n_samples, noise=cfg.noise, factor=0.5)
    return torch.as_tensor(x, dtype=torch.float32)


def _spiral(cfg: SynthCfg):
    n = cfg.n_samples // 2
    theta = torch.sqrt(torch.rand(n)) * 2 * math.pi
    r = 2 * theta
    x1 = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], 1)
    x2 = torch.stack([-r * torch.cos(theta), -r * torch.sin(theta)], 1)
    return torch.cat([x1, x2], 0) + 0.1 * torch.randn(cfg.n_samples, 2)


def _pinwheel(cfg: SynthCfg):
    rads = torch.randn(cfg.n_samples) * 0.3 + 1.0
    angles = torch.randint(0, 5, (cfg.n_samples,)) * (2 * math.pi / 5)
    angles += torch.randn(cfg.n_samples) * 0.2
    return torch.stack([rads * torch.cos(angles), rads * torch.sin(angles)], 1)


def _checkerboard(cfg: SynthCfg):
    x1 = torch.rand(cfg.n_samples) * 4 - 2
    x2 = torch.rand(cfg.n_samples) * 4 - 2
    x2 += torch.floor(x1) % 2
    return torch.stack([x1, x2], 1)


def _8gaussians(cfg: SynthCfg):
    c = (
        torch.tensor(
            [
                [1, 0],
                [-1, 0],
                [0, 1],
                [0, -1],
                [1 / math.sqrt(2), 1 / math.sqrt(2)],
                [-1 / math.sqrt(2), 1 / math.sqrt(2)],
                [1 / math.sqrt(2), -1 / math.sqrt(2)],
                [-1 / math.sqrt(2), -1 / math.sqrt(2)],
            ]
        )
        * 2
    )
    comps = torch.randint(0, 8, (cfg.n_samples,))
    return c[comps] + 0.1 * torch.randn(cfg.n_samples, 2)


def _swissroll(cfg: SynthCfg):
    x, _ = datasets.make_swiss_roll(cfg.n_samples, noise=cfg.noise)
    return torch.tensor(x[:, [0, 2]], dtype=torch.float32)


# --------------------------------------------------------------------------- #
# 3-D generators  (NEW)
# --------------------------------------------------------------------------- #
def _knotted(cfg: SynthCfg):
    n = cfg.n_samples
    θ = torch.linspace(-math.pi, math.pi, n)
    x = torch.sin(θ) + 2 * torch.sin(2 * θ)
    y = torch.cos(θ) - 2 * torch.cos(2 * θ)
    z = torch.sin(3 * θ)
    data = torch.stack([x, y, z], 1)
    data += 0.1 * torch.randn_like(data)
    return data * 4


def _helix(cfg: SynthCfg):
    n = cfg.n_samples
    θ = torch.linspace(0, 6 * math.pi, n)
    x = θ
    y = torch.cos(θ)
    z = torch.sin(θ)
    data = torch.stack([x, y, z], 1)
    data += 0.05 * torch.randn_like(data)
    return data


def _bent_lissajous(cfg: SynthCfg):
    n = cfg.n_samples
    θ = torch.linspace(-math.pi, math.pi, n)
    x = torch.sin(2 * θ)
    y = torch.cos(θ)
    z = torch.cos(2 * θ)
    data = torch.stack([x, y, z], 1)
    data += 0.1 * torch.randn_like(data)
    return data * 4


def _disjoint_circles(cfg: SynthCfg):
    n = cfg.n_samples // 2
    θ = torch.linspace(-math.pi, math.pi, n)
    x = torch.cat([-2 + torch.sin(θ), 2 + torch.sin(θ)])
    y = torch.cat([-1 + torch.sin(θ), 1 + 2 * torch.cos(θ)])
    z = torch.cat([-1 + torch.sin(θ), 1 + 2 * torch.cos(θ)])
    data = torch.stack([x, y, z], 1)
    data += 0.05 * torch.randn_like(data)
    return data * 2


def _twisted_eight(cfg: SynthCfg):
    n = cfg.n_samples // 2
    θ = torch.linspace(-math.pi, math.pi, n)
    x = torch.cat([torch.sin(θ), 2 + torch.sin(θ)])
    y = torch.cat([torch.cos(θ), torch.zeros_like(θ)])
    z = torch.cat([torch.zeros_like(θ), torch.cos(θ)])
    data = torch.stack([x, y, z], 1)
    data += 0.1 * torch.randn_like(data)
    return data * 4


def _interlocked_circles(cfg: SynthCfg):
    n = cfg.n_samples // 2
    θ = torch.linspace(-math.pi, math.pi, n)
    x = torch.cat([torch.sin(θ), 1 + torch.sin(θ)])
    y = torch.cat([torch.cos(θ), torch.zeros_like(θ)])
    z = torch.cat([torch.zeros_like(θ), torch.cos(θ)])
    data = torch.stack([x, y, z], 1)
    data += 0.1 * torch.randn_like(data)
    return data * 4


# --------------------------------------------------------------------------- #
# Lookup table
# --------------------------------------------------------------------------- #
_GENERATORS = {
    # 2-D
    "gmm": _gmm,
    "two_moons": _two_moons,
    "two_circles": _two_circles,
    "spiral": _spiral,
    "pinwheel": _pinwheel,
    "checkerboard": _checkerboard,
    "8gaussians": _8gaussians,
    "swissroll": _swissroll,
    # 3-D
    "knotted": _knotted,
    "helix": _helix,
    "bent_lissajous": _bent_lissajous,
    "disjoint_circles": _disjoint_circles,
    "twisted_eight": _twisted_eight,
    "interlocked_circles": _interlocked_circles,
}


# --------------------------------------------------------------------------- #
# Public loader
# --------------------------------------------------------------------------- #
def get_synth_dataloaders(cfg_dict: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    cfg = SynthCfg(**cfg_dict)
    if cfg.name not in SYNTH_DATASETS:
        raise ValueError(f"Unsupported synthetic dataset: {cfg.name}")

    # Allow fractional n_samples (e.g. 0.05 ⇒ 5 %)
    n_samples = cfg.get("n_samples", DATASET_SIZE)
    if 0 < n_samples <= 1:
        n_samples = int(DATASET_SIZE * n_samples)

    # keep extra DATASET_SIZE*2 for val/test
    cfg.n_samples = n_samples + 2 * DATASET_SIZE
    print(f"[Synthetic] Generating {cfg.n_samples} samples for '{cfg.name}'")

    x = _GENERATORS[cfg.name](cfg)
    x = (x - x.mean(0)) / x.std(0)  # per-dimension normalisation

    # Shuffle & split
    perm = torch.randperm(cfg.n_samples)
    train_end = cfg.n_samples - 2 * DATASET_SIZE
    val_end = cfg.n_samples - DATASET_SIZE

    x_train = x[perm[:train_end]]
    x_val = x[perm[train_end:val_end]]
    x_test = x[perm[val_end:]]

    bs = cfg.batch_size
    dev = cfg.device
    train_dl = DataLoader(TensorDataset(x_train.to(dev)), batch_size=bs, shuffle=True)
    valid_dl = DataLoader(TensorDataset(x_val.to(dev)), batch_size=bs, shuffle=False)
    test_dl = DataLoader(TensorDataset(x_test.to(dev)), batch_size=bs, shuffle=False)
    return train_dl, valid_dl, test_dl
