"""PFC backend wrapper used for the paper's EinsumNetwork experiments.

Implements tree/DAG PC wrappers with the analytical sum-of-squared-gradients
trace used as the sharpness regulariser.
"""

# hessian_reg/models/pfc_circuit.py
from __future__ import annotations
from typing import Any, Dict, Sequence
import torch
import torch.nn as nn
from .base import BaseCircuit, HessianTraceMixin

# --- PFC imports ---------------------------------------------------------
from packages.pfc.components.spn.Graph import (
    random_binary_trees,
    poon_domingos_structure,
)
from packages.pfc.components.spn.EinsumNetwork import EinsumNetwork, Args
from packages.pfc.components.spn.ExponentialFamilyArray import (
    NormalArray,
    BinomialArray,
    CategoricalArray,
)
from packages.pfc.models import (
    EinsumNet,
    LinearSplineEinsumFlow,
    QuadraticSplineEinsumFlow,
)
from torch.func import grad as func_grad, vmap, functional_call


class PFCCircuit(HessianTraceMixin, BaseCircuit):
    """
    Wrapper for models built with the `probabilistic-flow-circuits` package.
    Works with EinsumNetwork (discrete) and LinearSplineEinsumFlow (flow).

    Parameters
    ----------
    pfc_model : EinsumNetwork | LinearSplineEinsumFlow
        A *constructed* PFC model.
    device : str | torch.device
        Where to place the model and tensors.
    """

    def __init__(self, pfc_model: nn.Module, device="cpu"):
        super().__init__()
        self.m = pfc_model.to(device)
        self.device = torch.device(device)
        # cache: many PFC methods allocate buffers on the fly
        self._last_batch: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    #  Inference API
    # ------------------------------------------------------------------ #
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        self._last_batch = x.to(self.device)
        return self.m(x.to(self.device))

    def marginal(self, x: torch.Tensor, query_vars: Sequence[int]) -> torch.Tensor:
        self._last_batch = x.to(self.device)
        return self.m.marginal(self._last_batch, query_vars)

    def conditional(
        self,
        x: torch.Tensor,
        query_vars: Sequence[int],
        evidence_vars: Dict[int, int | float],
    ) -> torch.Tensor:
        self._last_batch = x.to(self.device)
        return self.m.conditional(self._last_batch, query_vars, evidence_vars)

    def mpe(self, x: torch.Tensor) -> torch.Tensor:
        return self.m.mpe(x.to(self.device))

    def sample(self, n: int) -> torch.Tensor:
        return self.m.sample(n).to(self.device)

    def em_accumulate(self, x: torch.Tensor, mu: float = 0.0, lambda_: float = 1.0) -> None:
        """
        Accumulate sufficient statistics for EM.
        This is a no-op for flow models (no EM).
        """
        lls = self.m(x)
        objective = lls.sum()
        objective.backward()  # accumulate gradients
        self.gradients = torch.cat(
            [theta.grad.reshape(-1) for theta in self.edge_params()]
        ).norm()
        self.m.em_process_batch(reg_lam=mu, reg_mu=lambda_)

    # ------------------------------------------------------------------ #
    #  Learning API
    # ------------------------------------------------------------------ #
    def em_update(self, step_size=0.1, mu: float = 0.0, lambda_: float = 1.0):
        r"""
        Closed-form EM M-step that mirrors Theorem 2 in the paper.

        Implements

        .. math::
            \theta_{nc} = \frac{F_{nc} + \sqrt{F_{nc}^2 + 4\lambda \mu F_{nc}}}
                                {2 \lambda}

        where :math:`F_{nc}` are the edge flows accumulated during the E-step.

        Args:
            step_size: Step size for PyTorch optimisers (kept for API parity).
            mu: :math:`\mu` regularisation multiplier (sharpness strength).
            lambda_: :math:`\lambda` simplex multiplier (typically 1.0).
        """
        _ = step_size  # retained for API parity with other backends
        self.m.em_update(reg_lam=mu, reg_mu=lambda_)

    def apply_gradients(self, batch, optimizer: torch.optim.Optimizer, mu: float = 0.0):
        """
        Run standard back‑prop on −log P(x) plus (optional) μ·Tr(∇²log P).
        """
        batch = self._last_batch if batch is None else batch.to(self.device)
        if batch is None:
            raise RuntimeError("call log_prob / forward once before training step.")

        optimizer.zero_grad()
        logP = self.log_prob(batch)
        loss = -logP.sum()

        if isinstance(mu, list) or mu > 0:
            trace = self.trace_hessian(batch)
            loss += mu * trace if not isinstance(mu, list) else trace

        loss.backward()
        params = self.edge_params()
        if isinstance(mu, list) and len(mu) == len(params):
            for theta_param, mu_scale in zip(params, mu):
                if theta_param.grad is not None:
                    # Layer-wise μ from mean-flow scheduler 
                    theta_param.grad *= mu_scale
                    
        self.gradients = torch.cat(
            [theta.grad.reshape(-1) for theta in self.edge_params()]
        ).norm()
        optimizer.step()

    def edge_params(self):
        """Return edge (non-input) parameters.

        Note: Uses self.named_parameters() to be consistent with trace_hessian.
        """
        return [
            theta
            for name, theta in self.named_parameters()
            if "0" not in name and theta.requires_grad
        ]

    def zero_grad(self):
        # Clear existing gradients
        for theta in self.m.parameters():
            if theta.grad is not None:
                theta.grad.zero_()

    def trace_hessian(self, x: torch.Tensor) -> torch.Tensor:
        """Return the sum-of-squared-gradients trace on batch `x`."""
        self.zero_grad()
        param_pytree = dict(self.named_parameters())
        input_layers = [key for key in param_pytree.keys() if "0" in key]
        for key in input_layers:
            del param_pytree[key]

        def loss_fn(pytree, xi):
            out = functional_call(self, pytree, (xi.view(1, -1),))
            return -out

        per_sample_grad = func_grad(loss_fn)

        def per_sample_trace(xi):
            grads = per_sample_grad(param_pytree, xi)
            return -sum(g.pow(2).sum() for g in grads.values())

        trace_hess = vmap(per_sample_trace)(x).sum()
        return trace_hess.abs()

    @property
    def edge_weights(self):
        "Return a *flat* tensor view of all mixture weights."
        return torch.cat([w.view(-1) for w in self.edge_params()])


_PFC_MODELS = {
    "EinsumNet": EinsumNet,
    "LinearSplineEinsumFlow": LinearSplineEinsumFlow,
    "QuadraticSplineEinsumFlow": QuadraticSplineEinsumFlow,
}


def _build_graph(cfg) -> Any:
    if cfg.graph_type in ["pd", "poon_and_domingos"]:  # Poon–Domingos
        H, W = cfg.image_h, cfg.image_w  # must be provided for 2‑D data
        delta = [[H / d, W / d] for d in cfg.pd_num_pieces]
        return poon_domingos_structure(shape=(H, W), delta=delta)

    elif cfg.graph_type in ["random_binary_tree", "ratspn", "binary_tree", "binary"]:
        depth = (
            cfg.depth if cfg.depth > 0 else int(max(1, (cfg.num_vars).bit_length() - 1))
        )
        return random_binary_trees(cfg.num_vars, depth, cfg.num_repetition)
    raise ValueError(f"Unsupported PFC graph type: {cfg.graph_type}")


# --------------------------------------------------------------------- #
#  Public builder
# --------------------------------------------------------------------- #
def build_pfc_model(cfg) -> PFCCircuit:
    """
    Parameters
    ----------
    cfg : omegaconf.DictConfig  (or any object with dotted attributes)
        Must contain at least:
          num_vars, num_dims, model_name, graph_type ("binary"|"pd"),
          num_input_distributions, num_sums, num_repetition, num_classes,
          leaf_config (dict), use_em (0/1), device.

    Returns
    -------
    PFCCircuit  (ready for the universal trainer)
    """

    # 1) graph ----------------------------------------------------------
    graph = _build_graph(cfg)

    # 2) leaf distribution & core args ---------------------------------
    args = Args(
        num_var=cfg.num_vars,
        num_dims=cfg.num_dims,
        num_input_distributions=cfg.num_input_distributions,
        num_sums=cfg.num_sums,
        num_classes=cfg.get("num_classes", 1),
        exponential_family=eval(cfg.leaf_distribution),  # will be set by the PFC model
        exponential_family_args=cfg.leaf_config,
        use_em=cfg.use_em,
        online_em_frequency=1,
        online_em_stepsize=cfg.get("online_em_stepsize", 0.1),
    )

    # 3) instantiate PFC model ----------------------------------------
    model = EinsumNetwork(graph=graph, args=args)
    model.initialize()
    model.to(torch.device(cfg.device))
    # 4) wrap ----------------------------------------------------------
    return PFCCircuit(model, device=cfg.device)
