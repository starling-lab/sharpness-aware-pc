"""Hessian utilities implementing the SSG trace described in the paper."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from hessian_reg.models.base import BaseCircuit


def compute_hessian_trace(
    model: BaseCircuit, data_loader: DataLoader, device: torch.device
) -> float:
    """Compute the SSG trace of the log-likelihood for a PC.

    Args:
        model: BaseCircuit instance with edge parameters.
        data_loader: DataLoader yielding batches of shape (B, D).
        device: torch.device for computation.

    Returns:
        float: |Tr(∇² log P(D))| summed over dataset.

    Time Complexity:
        O(|P| · |D|) where |P| is number of parameters, |D| is dataset size.
    """
    model = model.to(device)
    model.eval()
    trace_hess_total = 0.0

    with torch.no_grad():
        for batch in data_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            trace_hess = model.trace_hessian(x.to(device))
            trace_hess_total += torch.abs(trace_hess).item()

    return float(trace_hess_total)
