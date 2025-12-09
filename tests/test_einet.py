import copy
import os
import sys

import numpy as np
import pytest
import torch

# Ensure the 'packages' directory is on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# ---- our framework -------------------------------------------------------
from hessian_reg.datasets.synthetic import get_synth_dataloaders  # noqa: E402
from hessian_reg.models.pfc_wrapper import build_pfc_model  # noqa: E402
from hessian_reg.trainer import Trainer  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
dl_cfg = {
    "name": "gmm",
    "n_samples": 100,
    "batch_size": 20,
    "device": device,
    "noise": 0.05,
    "seed": 42,
}
train_dl, valid_dl, _ = get_synth_dataloaders(dl_cfg)


class Cfg:
    backend = "pfc"
    model_name = "EinsumNet"
    graph_type = "random_binary_tree"  # random binary tree
    # structural hyper‑params
    num_vars = 2
    num_dims = 1
    num_input_distributions = 2
    num_sums = 2
    num_repetition = 1
    num_classes = 1
    depth = 1
    leaf_distribution = "NormalArray"
    leaf_config = {}
    use_em = 0  # gradient descent
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def get(self, key, default=None):
        return getattr(self, key, default)


model = build_pfc_model(Cfg())


@pytest.mark.parametrize("mu", [0.0, 0.5])  # can add >0 to test reg‑path too
def test_training_pipeline_pfc_spiral(mu):

    for name, p in model.m.named_parameters():
        print(name, p.shape)

    trainer = Trainer(
        model,
        train_dl,
        valid_loader=valid_dl,
        mode="sgd",
        lr=3e-3,
        mu=mu,
        device=device,
    )

    hist = trainer.fit(epochs=10)
    ll0 = hist["valid_ll"][0]
    llF = hist["valid_ll"][-1]
    assert llF > ll0, "validation log‑likelihood did not improve"

