import os
import sys
from types import SimpleNamespace

import torch

# Ensure the 'packages' directory is on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from hessian_reg.models.pfc_wrapper import PFCCircuit, build_pfc_model  # noqa: E402


def test_pfc_forward(tmp_path):
    # Build a tiny random EinsumNetwork from PFC for smoke test
    from packages.pfc.components.spn.EinsumNetwork import EinsumNetwork, Args
    from packages.pfc.components.spn.Graph import random_binary_trees

    # Create a simple binary tree structure
    graph = random_binary_trees(num_var=4, depth=2, num_repetitions=1)

    # Create args for the network
    args = Args(
        num_var=4,
        num_dims=1,
        num_input_distributions=2,
        num_sums=4,
        num_classes=1,
    )

    # Create the EinsumNetwork
    net = EinsumNetwork(graph=graph, args=args)
    net.initialize()

    # Create the PFC circuit wrapper
    m = PFCCircuit(net, device="cpu")

    x = torch.randint(0, 2, (5, 4)).float()
    lp = m.log_prob(x)
    assert lp.shape == (5, 1) or lp.shape == (5,)
