from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, Dict
import torch
import torch.nn as nn


class BaseCircuit(ABC, nn.Module):
    """
    Abstract wrapper for ANY probabilistic‑circuit backend.
    Concrete subclasses must implement the seven core ops.
    """

    # -------- likelihood -------------------------------------------------
    @abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        _ = x
        raise NotImplementedError

    # -------- inference --------------------------------------------------
    @abstractmethod
    def marginal(self, x: torch.Tensor, query_vars: Sequence[int]) -> torch.Tensor:
        _ = (x, query_vars)
        raise NotImplementedError

    @abstractmethod
    def conditional(
        self,
        x: torch.Tensor,
        query_vars: Sequence[int],
        evidence_vars: Dict[int, int | float],
    ) -> torch.Tensor:
        _ = (x, query_vars, evidence_vars)
        raise NotImplementedError

    @abstractmethod
    def mpe(self, x: torch.Tensor) -> torch.Tensor:
        _ = x
        raise NotImplementedError

    @abstractmethod
    def sample(self, n: int) -> torch.Tensor:
        _ = n
        raise NotImplementedError

    # -------- learning ---------------------------------------------------
    @abstractmethod
    def em_update(self, step_size: float = 1.0, pseudocount: float = 0.0):
        _ = (step_size, pseudocount)
        raise NotImplementedError

    @abstractmethod
    def apply_gradients(
        self, batch, optimizer: torch.optim.Optimizer, mu: float = 0.0
    ):
        _ = (batch, optimizer, mu)
        raise NotImplementedError

    def forward(self, x):
        return self.log_prob(x).sum()


class HessianTraceMixin(ABC):
    """
    Mixin for circuits supporting efficient Hessian‑trace computation.
    Sub‑classes override `_trace_hessian()` to compute Tr(H) for their params.
    """
    mu: float = 1e-3  # default strength
    # --------- must override in concrete wrapper ------------------
    @abstractmethod
    def trace_hessian(self) -> torch.Tensor:
        """Return Tr(H) for *sum‑node* parameter vector θ."""

    
