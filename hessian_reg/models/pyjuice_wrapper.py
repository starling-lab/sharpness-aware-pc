"""PyJuice backend wrapper used for the paper's HCLT experiments."""

import torch
from .base import BaseCircuit, HessianTraceMixin
import packages.pyjuice as pj
from packages.pyjuice.queries import sample
from packages.pyjuice.structures import RAT_SPN, HCLT, HMM, PD, PDHCLT


class PyJuiceCircuit(BaseCircuit, HessianTraceMixin):
    """TensorCircuit wrapper exposing the BaseCircuit API."""

    def __init__(self, tensor_circuit):
        super().__init__()
        self.pc = tensor_circuit  # compiled TensorCircuit
        self._last_batch: torch.Tensor | None = None

    # ---------------- evidence --------------------------------------
    def log_prob(self, x):
        x = self._ensure_discrete(x)
        self._last_batch = x
        return self.pc(x)[:, 0]

    # ---------------- marginals / conditionals ----------------------
    def marginal(self, x, query_vars):
        x = self._ensure_discrete(x)
        self._last_batch = x
        return self.pc.forward(x, input_layer_fn="marginal", query_vars=query_vars)

    def conditional(self, x, query_vars, evidence_vars):
        x = self._ensure_discrete(x)
        self._last_batch = x
        return self.pc.forward(
            x,
            input_layer_fn="conditional",
            query_vars=query_vars,
            evidence_vars=evidence_vars,
        )

    def mpe(self, x):
        x = self._ensure_discrete(x)
        self._last_batch = x
        return self.pc.forward_mpe(x)

    def sample(self, n):
        return sample(self.pc, n)

    # ---------------- learning --------------------------------------
    def em_accumulate(self, x, mu: float = 0.0, lambda_: float = 1.0):
        r"""
        Accumulate sufficient statistics for EM (Definition 6).

        Args:
            x: Mini-batch tensor.
            mu: :math:`\mu` sharpness multiplier applied in the M-step.
            lambda_: :math:`\lambda` simplex multiplier (usually 1.0).
        """
        _ = mu
        _ = lambda_
        x = self._ensure_discrete(x)
        self._last_batch = x
        self.pc.cumulate_flows(x)

    @property
    def params(self):
        return [self.pc.params[self.pc.num_dummy_params :]]

    def trace_hessian(self, batch=None):
        """Return |Tr(∇² log P(x))| via PyJuice's analytical trace."""
        self.pc.zero_param_flows()
        if batch is not None:
            batch = self._ensure_discrete(batch)
        trace_hess = self.pc.compute_hessian_trace(batch)
        return trace_hess.abs()

    def edge_params(self):
        return [p for n, p in self.pc.named_parameters() if "input"]

    def zero_param_flows(self):
        """
        Zero out the parameter flows.
        This is a no-op for flow models (no EM).
        """
        self.pc.zero_param_flows()

    # ---- SGD update --------------------------------------------
    def apply_gradients(self, batch, optimizer, mu: float = 0.0):
        self.pc.zero_param_flows()
        lr = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad()
        batch = self._ensure_discrete(batch)
        self.pc.sgd_update(batch, lr=lr, reg_coeff=mu)

    # ---- EM update ---------------------------------------------
    def em_update(self, step_size=1.0, pseudocount=0.0, mu: float = 0.0, lambda_: float = 1.0):
        r"""Closed-form EM update using Theorem 2."""
        self.pc.param_flows = torch.clamp(self.pc.param_flows, min=1e-12)
        self.pc.mini_batch_em(step_size, pseudocount, reg_mu=mu, reg_lam=lambda_)

    def _ensure_discrete(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_floating_point(x):
            return x.long()
        return x


def resolve_input_distribution(type_str: str, params: dict):
    from packages.pyjuice.nodes.distributions import (
        Categorical,
        Gaussian,
        Bernoulli,
    )

    name = type_str.lower()
    dist_map = {
        "categorical": Categorical,
        "gaussian": Gaussian,
        "bernoulli": Bernoulli,
    }

    if name not in dist_map:
        raise ValueError(f"Unsupported input_node_type: {name}")

    return dist_map[name], params


def build_pyjuice_model(cfg, data):
    model_type = cfg.graph_type.lower()
    num_vars = cfg.num_vars
    num_latents = cfg.num_latents

    input_node_type_str = getattr(cfg, "leaf_distribution", "categorical")
    input_node_params = getattr(cfg, "leaf_config", {"num_cats": 256})
    input_node_type, input_node_params = resolve_input_distribution(
        input_node_type_str, input_node_params
    )

    # Construct structure
    if model_type == "ratspn":
        root = RAT_SPN(
            num_vars=num_vars,
            num_latents=num_latents,
            depth=cfg.depth,
            num_repetitions=cfg.num_repetitions,
            num_pieces=cfg.get("num_pieces", 2),
            input_node_type=input_node_type,
            input_node_params=input_node_params,
        )

    elif model_type == "hclt":
        root = HCLT(
            x=data,
            num_latents=num_latents,
            input_node_type=input_node_type,
            input_node_params=input_node_params,
            **cfg.get("hclt_kwargs", {}),
        )

    elif model_type == "pd":
        root = PD(
            data_shape=cfg.data_shape,
            num_latents=num_latents,
            split_intervals=cfg.get("split_intervals", 2),
            structure_type=cfg.get("structure_type", "sum_dominated"),
            input_node_type=input_node_type,
            input_node_params=input_node_params,
        )

    elif model_type == "pdhclt":
        root = PDHCLT(
            data=data,
            data_shape=cfg.data_shape,
            num_latents=num_latents,
            split_intervals=cfg.get("split_intervals", 2),
            structure_type=cfg.get("structure_type", "sum_dominated"),
            input_node_type=input_node_type,
            input_node_params=input_node_params,
            hclt_kwargs=cfg.get("hclt_kwargs", {}),
        )

    elif model_type == "hmm":
        root = HMM(
            seq_length=cfg.seq_length,
            num_latents=num_latents,
            num_emits=cfg.num_emits,
            homogeneous=cfg.get("homogeneous", True),
        )

    else:
        raise ValueError(f"Unsupported pyjuice structure type: {model_type}")

    model = pj.compile(root)
    model.to(torch.device(cfg.device))

    print(
        f"PyJuice model {model_type} with {num_vars} variables and {num_latents} latents created."
    )
    print(f"Number of parameters: {model.params.shape}")

    return PyJuiceCircuit(model)
