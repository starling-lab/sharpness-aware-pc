"""Adaptive curvature-regularisation utilities used in the paper.

Implements gradient-norm (SGD) and mean-flow (EM) μ schedulers described
in the paper as well as a smoother RegScheduler for ablations.
"""

import torch
import numpy as np
import math
from statistics import median


def _grad_norm(params):
    vec = [p.grad.reshape(-1) for p in params if p.grad is not None]
    return torch.cat(vec).norm().item() if vec else 0.0


def _flow(params):
    vec = [(p.data * p.grad).reshape(-1) for p in params if p.grad is not None]
    return torch.cat(vec).mean().abs().item() if vec else 0.0


def _layerwise_flow(params):
    vec = [
        torch.ones_like(p.reshape(-1)) * (p.data * p.grad).mean()
        for p in params
        if p.grad is not None
    ]
    return torch.cat(vec).abs()


def mean_flow_mu(
    model,
    train_loader,
    device,
    max_batches: int = 1,
    layerwise=True,  # use local mean flow instead of global
):
    """
    Estimate μ using per-layer mean flow statistics (EM setting).

    Chooses μ so that EM pseudo-flows match the target degree-of-overfitting.
    """
    model.train()
    flow = None
    if "PFCCircuit" in str(type(model)):
        for i, batch in enumerate(train_loader):
            if i == max_batches:
                break
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
            # data gradient
            model.zero_grad()
            (model.log_prob(x).sum()).backward(retain_graph=True)
            step_flow = (
                _flow(model.edge_params())
                if not layerwise
                else _layerwise_flow(model.edge_params())
            )
            step_flow = step_flow if step_flow is not np.nan else 0.0
            flow = step_flow if flow is None else flow + step_flow
        new_mu = flow / max_batches if max_batches > 0 else 0.0
        if layerwise:
            shapes = [p.shape for p in model.edge_params() if p.grad is not None]
            sizes = [p.numel() for p in model.edge_params() if p.grad is not None]
            recoverd, offset = [], 0
            for shape, size in zip(shapes, sizes):
                recoverd.append(new_mu[offset : offset + size].view(shape))
                offset += size
            new_mu = recoverd

    elif "PyJuice" in str(type(model)):
        for i, batch in enumerate(train_loader):
            if i == max_batches:
                break
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
            model.zero_param_flows()
            with torch.no_grad():
                model.pc.forward(x)
                model.pc.backward(
                    inputs=x,
                    ll_weights=None,
                    compute_param_flows=True,
                    flows_memory=1.0,
                )
                model.pc.cumulate_flows(x)
            if not layerwise:
                step_flow = (
                    model.pc.param_flows.mean().item()
                    if model.pc.param_flows is not None
                    else 0.0
                )
            else:
                step_flow = torch.ones_like(model.pc.param_flows).detach()
                for layer_group in model.pc.inner_layer_groups:
                    for layer in layer_group:
                        if hasattr(layer, "num_parameters"):
                            # print( layer, layer.num_parameters, layer._layer_pid_range, layer._layer_pfid_range)
                            step_flow[
                                layer._layer_pfid_range[0] : layer._layer_pfid_range[1]
                            ] = (
                                model.pc.param_flows[
                                    layer._layer_pfid_range[
                                        0
                                    ] : layer._layer_pfid_range[1]
                                ]
                                .mean()
                                .item()
                            )

            flow = step_flow if flow is None else flow + step_flow
        new_mu = flow / max_batches if max_batches > 0 else 0.0
        model.zero_param_flows()

    return new_mu


def adaptive_mu(
    model,
    train_loader,
    device,
    dof_now: float,  # current |val-train| / |val|
    delta_trn_nll=0.0,  # change in train NLL
    delta_val_nll=0.0,  # change in validation NLL
    alpha: float = 1,  # target grad ratio
    kappa: float = 1.05,  # DOF influence strength
    max_batches: int = 1,
    clip=(1e-12, 1e12),
):
    """
    Return μ that balances gradient magnitudes and DOF feedback.

    Implements the SGD heuristic from the paper:
      1. Match μ·||∇R|| to α·||∇L|| (Equation 5).
      2. Scale μ multiplicatively based on change in DOF/validation NLL.
    """
    g_data, g_reg = [], []

    model.train()
    for i, batch in enumerate(train_loader):
        if i == max_batches:
            break
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)

        # data gradient
        model.zero_grad()
        (-model.log_prob(x).sum()).backward(retain_graph=True)
        g_data.append(_grad_norm(model.edge_params()))

        # reg gradient
        model.zero_grad()
        trace_hess = model.trace_hessian(x)
        trace_hess.backward()
        g_reg.append(_grad_norm(model.edge_params()))

    if not g_data or not g_reg or median(g_reg) == 0:
        return None  # keep current μ

    base_mu = alpha * median(g_data) / median(g_reg)

    # ---- DOF feedback -------------------------------------------------
    # scale μ multiplicatively: if over-fit > target  → increase μ

    scale = 1.0
    if delta_trn_nll * delta_val_nll >= 0:
        scale = 1e-2  # both losses decreasing → keep μ conservative

    elif delta_trn_nll <= 0 and delta_val_nll >= 0:
        # When train improves but validation worsens, increase μ sharply.
        scale = kappa ** (100 * (min(dof_now, 1.0)))  # cap dof at 100%

    new_mu = base_mu * scale
    new_mu = max(min(new_mu, clip[1]), clip[0])
    return new_mu

class RegScheduler:
    """Adaptive log-μ optimizer used for ablations"""

    def __init__(
        self,
        init_mu: float = 1e-3,
        dof_target: float = 0.001,
        beta: float = 0.5,  # weight on ΔValNLL
        period: int = 1,  # epochs between updates
        ema_alpha: float = 0.5,  # smoothing for signals
        lr: float = 0.1,  # base LR for log-μ
        clip: tuple = (1e-12, 1e12),
    ):
        self.z = math.log10(init_mu)
        self.clip = clip
        self.dof_tgt = dof_target
        self.beta = beta
        self.period = period
        self.ema_a = ema_alpha
        self.lr = lr
        self.grad_sq = 0.0  # AdaGrad accumulator
        # state
        self.ema_dof = None
        self.prev_val = None
        self.step_ctr = 0

    # --------------------------------------------------------------
    def mu_value(self):
        """Return current μ in linear domain."""
        lam = 10.0**self.z
        return max(min(lam, self.clip[1]), self.clip[0])

    # --------------------------------------------------------------
    def update(self, trn_nll: float, val_nll: float):
        """
        Call once per epoch (or batch); only performs an optimisation
        step every `period` calls.
        """
        # ---- derive signals --------------------------------------
        # Degree-of-overfitting proxy (Eq. 7) in percentage.
        dof = 100 * abs(val_nll - trn_nll) / abs(val_nll)

        if self.ema_dof is None:
            self.ema_dof = dof
            self.prev_val = val_nll
            self.step_ctr += 1
            return self.mu_value()

        # ΔValNLL (positive = got worse)
        delta_val = (self.prev_val - val_nll) / abs(self.prev_val)
        self.prev_val = val_nll

        # EMA smoothing
        self.ema_dof = self.ema_a * self.ema_dof + (1 - self.ema_a) * dof
        if not hasattr(self, "ema_dval"):
            self.ema_dval = delta_val
        else:
            self.ema_dval = self.ema_a * self.ema_dval + (1 - self.ema_a) * delta_val

        # perform update only every `period`
        self.step_ctr += 1
        if self.step_ctr % self.period:
            return self.mu_value()

        # ---- compute gradient wrt z = log10(μ) -------------------
        # μ = 10^z  ⇒ dμ/dz = μ ln(10)
        lam = self.mu_value()
        dlam_dz = lam * math.log(10)

        # ∂E/∂μ  (treat DOF/τ + β ΔNLL as linear in μ for small changes)
        tau = self.dof_tgt
        grad_E_lam = (self.ema_dof - tau) / tau + self.beta * self.ema_dval
        # chain rule
        grad_z = grad_E_lam * dlam_dz

        # AdaGrad LR
        self.grad_sq += grad_z**2
        adj_lr = self.lr / (math.sqrt(self.grad_sq) + 1e-8)

        # update z
        self.z -= adj_lr * grad_z
        # clip μ (in linear domain) then map back to z
        self.z = math.log10(max(min(10**self.z, self.clip[1]), self.clip[0]))

        return self.mu_value()
