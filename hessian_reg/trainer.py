# hessian_reg/trainer.py
"""
Trainer
----------------
• Handles both optimisation modes:
      - 'sgd'  : gradient descent (Adam, SGD, …)
      - 'em'   : EM / mini‑batch EM
• Optional Hessian‑trace regulariser (μ > 0, Theorem 2)
• Works with every wrapper inheriting from BaseCircuit
"""

from __future__ import annotations

import copy
from typing import Dict

import torch
import tqdm
from torch.utils.data import DataLoader

from hessian_reg.models.base import BaseCircuit
from hessian_reg.utils.profiler_reg import adaptive_mu, mean_flow_mu


class Trainer:
    """Unified SGD/EM trainer with Hessian trace regularisation."""

    def __init__(
        self,
        model: BaseCircuit,
        train_loader: DataLoader,
        valid_loader: DataLoader | None = None,
        *,
        mode: str = "sgd",  # 'sgd' or 'em'
        optimizer_cls="Adam",
        lr: float = 1e-3,
        mu: float | str | None = None,
        lambda_: float | None = None,
        device: str | torch.device = "cpu",
        online_em: bool = False,  # True → do EM step after each batch
        pseudocount: float = 0.0,
    ):
        """Initialise trainer."""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.mode = mode.lower()

        self.mu = 0.0 if mu is None else mu
        self.lambda_ = 1.0 if lambda_ is None else lambda_

        self.device = torch.device(device)
        self.lr = lr
        self.online_em = online_em
        self.pseudocount = pseudocount

        print(
            f"Trainer: {self.mode.upper()} mode, λ={self.lambda_}, μ={self.mu}, "
            f"device={self.device}, pseudocount={self.pseudocount}"
        )
        if "sgd" in self.mode:
            if isinstance(optimizer_cls, str):
                optimizer_cls = getattr(torch.optim, optimizer_cls)
            self.optimizer = optimizer_cls(self.model.parameters(), lr=lr)
        else:
            self.optimizer = None  # EM does not need an optimiser

        self.mu_schedule = None
        if isinstance(self.mu, str):
            self.mu_schedule = self.mu
            # warm start to set μ if we requested adaptive scheduling
            self.mu = self.init_mu(ratio=1, warm_batches=5)

    def init_mu(self, ratio=1, warm_batches=1):
        if "em" in self.mode:
            return mean_flow_mu(
                self.model,
                self.train_loader,
                self.device,
                max_batches=warm_batches,
            )
        elif self.mode == "sgd":
            return adaptive_mu(
                self.model,
                self.train_loader,
                self.device,
                dof_now=0.0,  # initial DOF is 0
                alpha=ratio,  # target grad ratio
                kappa=1.05,    # influence strength
                max_batches=warm_batches,
            )

    def _update_mu(self, g_data_norm, g_reg_norm, beta=0.95, alpha=0.2):
        """
        g_data_norm : ||∇L||    on the *current* batch
        g_reg_norm  : ||∇R||    idem
        beta        : EMA smoothing factor
        alpha       : target ratio  μ·||∇R|| / ||∇L||
        """
        # 1) Exponential moving averages for smoother estimates
        if not hasattr(self, "_ema_gd"):
            self._ema_gd = g_data_norm
            self._ema_gr = g_reg_norm
        else:
            self._ema_gd = beta * self._ema_gd + (1 - beta) * g_data_norm
            self._ema_gr = beta * self._ema_gr + (1 - beta) * g_reg_norm

        # 2) multiplicative correction factor
        if self._ema_gr > 0:
            factor = alpha * self._ema_gd / self._ema_gr
            # clip to avoid drastic jumps
            factor = min(max(factor, 0.5), 2.0)
            self.mu *= factor

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def fit(self, epochs: int) -> Dict[str, list]:
        history = {"train_ll": [], "valid_ll": [], "gradients": [], "params": []}

        for epoch in range(1, epochs + 1):
            history["params"].append(copy.deepcopy(self.model.edge_params()))
            train_ll = -self.train_epoch()
            history["train_ll"].append(train_ll)
            if self.valid_loader is not None:
                valid_ll = -self.evaluate(self.valid_loader)
                history["valid_ll"].append(valid_ll)
                print(f"[{epoch:03d}]  train: {train_ll:.4f}   valid: {valid_ll:.4f}")
            else:
                print(f"[{epoch:03d}]  train: {train_ll:.4f}")
            # save params for inspection
        return history

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        tot, n = 0.0, 0
        for x in loader:
            x = x[0] if isinstance(x, (list, tuple)) else x
            lls = self.model.log_prob(x.to(self.device))
            tot += lls.sum().item()
            n += lls.numel()
        return -tot / n

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #
    def train_epoch(self) -> float:
        """
        Train for one epoch and return the (negative) average log-likelihood −𝓛̅.
        Prints a live progress-bar that shows:
            • batch-level −log p(x)
            • running (epoch) average −log p(x)
        """
        self.model.train()
        running_tot, running_n = 0.0, 0  # running sums

        with tqdm.tqdm(
            self.train_loader,
            desc=f"Epoch {getattr(self, 'epoch', '?')}",
            unit="batch",
            dynamic_ncols=True,
            position=0,
            leave=False,
        ) as pbar:

            for batch in pbar:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)

                # ── forward pass ──────────────────────────────────────────────
                lls = self.model.log_prob(x)  # log p(x|θ)
                batch_ll = lls.mean()  # scalar per-batch LL
                running_tot += batch_ll.item() * len(x)
                running_n += len(x)

                # ── optimisation step(s) ─────────────────────────────────────
                if "sgd" in self.mode:
                    self._sgd_step(x)

                elif "em" in self.mode:
                    self._em_accumulate(x)
                    if self.online_em:
                        self._em_mstep()

                # ── update the bar footer ────────────────────────────────────
                pbar.set_postfix(
                    btch_NLL=f"{-batch_ll.item():7.4f}",
                    avg_NLL=f"{-running_tot / running_n:7.4f}",
                )

        if "em" in self.mode and not self.online_em:
            self._em_mstep()

        return -running_tot / running_n

    # ------------ optimisation paths ----------------------------------
    def _sgd_step(self, batch=None):
        self.model.apply_gradients(batch=batch, optimizer=self.optimizer, mu=self.mu)

    def _em_accumulate(self, x):
        # just run backward to collect flows; gradient not needed
        self.model.em_accumulate(x, mu=self.mu, lambda_=self.lambda_)

    def _em_mstep(self):
        self.model.em_update(
            step_size=self.lr,
            pseudocount=self.pseudocount,
            mu=self.mu,
            lambda_=self.lambda_,
        )
        if hasattr(self.model, "zero_param_flows"):
            self.model.zero_param_flows()
