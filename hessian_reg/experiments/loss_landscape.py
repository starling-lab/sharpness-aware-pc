#!/usr/bin/env python3
"""
experiments/loss_landscape.py
=============================
Quick loss‑landscape & sharpness visualisation for PC‑Hessian experiments.

Borrowed and *heavily* simplified from Tom Goldstein et al.'s
https://github.com/tomgoldstein/loss-landscape (MIT License).

* 1‑D and 2‑D loss surface evaluation along random or interpolation directions.
* Optional top‑k Hessian eigen‑value estimation (power iteration) for sharpness.
* Stand‑alone CLI that plays nicely with checkpoints saved by the PC‑Hessian
  `Trainer` (`torch.save({'state_dict': model.state_dict()})`).
* Minimal dependencies: `torch`, `numpy`, `matplotlib`, `h5py` (optional).

Example
-------
```bash
python -m experiments.loss_landscape \
    --checkpoint runs/spiral/pcnet_best.pt \
    --model-class hessian_reg.models.pfc_wrapper.PFCCircuit \
    --dataset hessian_reg.datasets.synthetic.* \
    --x -1:1:51 --y -1:1:51 --k-eig 20
```
Generates:
* `loss_landscape/surface.h5` (or .npz when *h5py* is absent)
* `loss_landscape/surface.png` – 2‑D mesh of the surface
* `loss_landscape/hessian_eigs.npy` – top‑k eigen‑values
"""
from __future__ import annotations

# ───────────────────────────────── imports ────────────────────────────────────
import math
import pathlib
from typing import List, Optional, Sequence, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors, gridspec, ticker
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3‑D)
from omegaconf import DictConfig, open_dict

try:
    import h5py  # efficient persistence
except ImportError:  # pragma: no cover
    h5py = None

from hessian_reg.datasets import get_dataloaders
from hessian_reg.models import load_model
from hessian_reg.utils import set_seed
from hessian_reg.utils.env import load_env

# Load user overrides from .env (project root or cwd)
load_env()

# ───────────────────────────── global config ─────────────────────────────────
plt.rcParams.update(
    {
        "figure.figsize": (4, 4),
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 7,
        "axes.titlesize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 2.0,
        "legend.fontsize": 10,
        "savefig.dpi": 300,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


# ─────────────────────────────── CLI entry ───────────────────────────────────
@hydra.main(version_base=None, config_path="../../configs", config_name="main")
def main(cfg: DictConfig):
    """Entry‑point – computes surfaces & sharpness, then saves pretty figures."""

    set_seed(cfg.get("seed", 0))
    mu_value = cfg.trainer.get("mu", 0.0)
    with open_dict(cfg.trainer):
        cfg.trainer.mu = mu_value
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.model["device"] = device

    exp_name = f"{cfg.dataset.name}_{cfg.model.model_name}_{cfg.trainer.mode}_R{cfg.trainer.mu}_"
    # ---------------- Dataset & model ----------------------------------------
    train_dl, valid_dl, _ = get_dataloaders(cfg.dataset)

    model = load_model(cfg.model, next(iter(train_dl))[0].to(device))
    model.load_state_dict(
        torch.load(f"./models/N{cfg.dataset.n_samples}_final_model.pt")
    )

    # ---------------- Directions & coordinates -------------------------------
    X = cfg.get("landscape", {}).get("x", "-3:3:101")
    Y = cfg.get("landscape", {}).get("y", "-3:3:101")
    k_eig = cfg.get("landscape", {}).get("k_eig", 10)

    xcoords, _ = _parse_slice(X)
    ycoords, _ = _parse_slice(Y)

    dir_x = _random_direction(model)
    dir_y = _random_direction(model)
    base_state = _clone_params(model)

    # ---------------- Surface evaluation -------------------------------------
    train_losses = _compute_surface(
        model,
        base_state,
        dir_x,
        xcoords,
        train_dl,
        device=device,
        dir_y=dir_y,
        ycoords=ycoords,
    )

    valid_losses = _compute_surface(
        model,
        base_state,
        dir_x,
        xcoords,
        valid_dl,
        device=device,
        dir_y=dir_y,
        ycoords=ycoords,
    )

    # ---------------- Persist results ----------------------------------------
    outdir = pathlib.Path(f"./loss_landscape/N{cfg.dataset.n_samples}")
    outdir.mkdir(parents=True, exist_ok=True)

    _save_surface(outdir / f"{exp_name}_train_surface", xcoords, ycoords, train_losses)
    _save_surface(outdir / f"{exp_name}_valid_surface", xcoords, ycoords, valid_losses)

    # ---------------- Plotting ------------------------------------------------
    fig1d = _plot_1d(
        xcoords,
        train_losses,
        valid_losses,
        outdir / f"{exp_name}_1d_surface",
        title=f"1D Train Loss Surface | {cfg.model.model_name} | {cfg.dataset.name} | Reg={cfg.trainer.mu!=0}",
    )

    fig_train_contour, fig_train_3d = _plot_2d(
        xcoords,
        ycoords,
        train_losses,
        outdir / f"{exp_name}_2d_train_surface",
        title=f"2D Train Loss Surface | {cfg.model.model_name}  | {cfg.dataset.name} | Reg={cfg.trainer.mu!=0}",
    )

    fig_val_contour, fig_val_3d = _plot_2d(
        xcoords,
        ycoords,
        valid_losses,
        outdir / f"{exp_name}_2d_valid_surface",
        title=f"2D Validation Loss Surface | {cfg.model.model_name}  | {cfg.dataset.name} | Reg={cfg.trainer.mu!=0}",
    )

    # ---------------- Sharpness (optional) -----------------------------------
    fig_eigs = None
    if k_eig > 0:
        eigs = _power_iter(model, train_dl, k=k_eig, device=device)
        np.save(
            outdir / f"{exp_name}_hessian_eigs.npy", np.asarray(eigs, dtype=np.float32)
        )
        print("Top‑{} Hessian eigen‑values:".format(k_eig), eigs)
        fig_eigs = _plot_eigs(
            eigs,
            outdir / f"{exp_name}_hessian_eigs",
            title=f"Hessian spectrum (top {k_eig}) |  {cfg.model.model_name}  | {cfg.dataset.name} | Reg={cfg.trainer.mu!=0}",
        )

    _plot_loss_and_eigs_compact(
        xcoords,
        train_losses,
        valid_losses,
        eigs,
        outdir / f"{exp_name}_1D_loss_plus_spectrum",
    )
    # ---------------- Summary figure ----------------------------------------
    _plot_summary(fig1d, fig_train_contour, fig_eigs, outdir / f"{exp_name}_summary")


# ───────────────────────────── helper utils ──────────────────────────────────


# ─────────────── 2) compact figure: 1-D loss curves + eigen-spectrum ─────────────
def _plot_loss_and_eigs_compact(
    xcoords: np.ndarray,
    train_losses: Sequence[float],
    valid_losses: Sequence[float],
    eigs: Sequence[float],
    out: pathlib.Path,
    *,
    title: str = "Loss curves & Hessian spectrum",
) -> plt.Figure:
    """
    Two-panel figure: left = 1-D loss curves (train & valid),
    right = eigen-spectrum (as in `_plot_eigs`).

    Parameters
    ----------
    xcoords : np.ndarray
        Coordinates α used for the 1-D scan.
    train_losses, valid_losses : Sequence[float]
        Loss values along `xcoords`.
    eigs : Sequence[float]
        Eigen-values for the spectrum.
    out : pathlib.Path
        Output stem – `.png` and `.pdf` are saved.
    title : str
        Overall title.
    """
    eigs = np.sort(np.asarray(eigs, dtype=np.float32))[::-1]
    idx = np.arange(1, len(eigs) + 1)

    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(
        1,
        2,
        width_ratios=[3, 1],
        wspace=0.0,  #  ← removes gap
        left=0.07,
        right=0.97,
        top=0.92,
        bottom=0.12,
    )

    if len(np.array(train_losses).shape) > 1:
        train_losses = np.array(train_losses)
        valid_losses = np.array(valid_losses)
        min_idx = np.unravel_index(np.argmin(train_losses), train_losses.shape)[1]
        train_losses = train_losses[:, min_idx]
        valid_losses = valid_losses[:, min_idx]

    # ─ left: loss curves ─
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_loss.plot(xcoords, train_losses, label="Training", lw=2, color="red")
    ax_loss.plot(xcoords, valid_losses, label="Validation", lw=2, ls="--", color="blue")
    # ax_loss.set_xlabel("α", fontsize=10, fontweight="bold")
    # ax_loss.set_ylabel("NLL", fontsize=10, fontweight="bold")
    # ax_loss.set_title("1-D surface", fontsize=9)
    # ax_loss.grid(alpha=0.3)
    ax_loss.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_loss.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_loss.tick_params(axis="both", which="both", direction="in", length=4)
    ax_loss.grid(True, which="major", ls="--", alpha=0.35)
    ax_loss.grid(True, which="minor", ls=":", alpha=0.15)
    ax_loss.legend(frameon=True, fontsize=13, loc="lower left")
    ax_loss.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=5,
        width=1.0,
        labelsize=13, 
    )
    # ─ right: eigen spectrum (no colour-bar) ─
    cmap = plt.get_cmap("viridis")
    norm = colors.LogNorm(vmin=eigs.min(), vmax=eigs.max())
    ax_eig = fig.add_subplot(gs[0, 1])
    ax_eig.barh(
        idx - 0.5,  # start at integer edge (align='edge' baseline)
        eigs,
        height=0.9,
        align="edge",
        color=cmap(norm(eigs)),
        edgecolor="none",
    )
    ax_eig.set_xscale("log")
    ax_eig.invert_yaxis()
    ax_eig.xaxis.set_label_position("bottom")
    ax_eig.xaxis.tick_top()
    # ax_eig.set_xlabel("$\\lambda_k$", labelpad=2, fontsize=11, fontweight="bold")
    ax_eig.set_yticks(idx)
    ax_eig.set_yticklabels(idx, fontsize=13)
    # ax_eig.set_title("spectrum", fontsize=9, pad=4)

    # major / minor ticks
    # ax_eig.xaxis.set_major_locator(ticker.LogLocator(numticks=6))
    # ax_eig.xaxis.set_minor_locator(ticker.LogLocator(subs="auto", numticks=10))
    # ax_eig.yaxis.set_minor_locator(ticker.NullLocator())  # no minor y ticks

    median_val = float(np.median(eigs))
    tick_val = 10 ** round(math.log10(median_val))
    tick_label = rf"$10^{{{int(round(math.log10(median_val)))}}}$"

    # ax_eig.set_xticks([median_val])
    # ax_eig.set_xticklabels([f"{median_val:.1e}"], fontsize=10)

    ax_eig.set_xticks([tick_val])
    ax_eig.set_xticklabels([tick_label], fontsize=13)

    ax_eig.tick_params(
        axis="x", which="major", direction="in", length=5, width=1.0, color="black"
    )
    ax_eig.tick_params(
        axis="x", which="minor", direction="in", length=3, width=0.8, color="#404040"
    )
    ax_eig.tick_params(
        axis="y", which="major", direction="in", length=5, width=1.0, color="black"
    )

    ax_eig.grid(True, which="major", ls="--", alpha=0.35, axis="x")
    ax_eig.grid(True, which="minor", ls=":", alpha=0.15, axis="x")

    # helper: place axis label *inside* the axes
    def _label_inside(ax, which: str, text: str, *, pad: float = 0.02, **kw):
        if which.lower() == "x":
            ax.text(
                0.75, pad, text, transform=ax.transAxes, ha="right", va="bottom", **kw
            )
            ax.set_xlabel("")  # hide the regular exterior label
        elif which.lower() == "y":
            ax.text(
                pad,
                0.5,
                text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                rotation=90,
                **kw,
            )
            ax.set_ylabel("")

    # ── for the loss panel ───────────────────────────────────────────────
    _label_inside(ax_loss, "x", "α", fontsize=13, fontweight="bold")
    _label_inside(ax_loss, "y", "NLL", fontsize=13, fontweight="bold")

    # ── for the spectrum panel (top-axis moved earlier) ──────────────────
    _label_inside(ax_eig, "x", "$\\lambda_k$", fontsize=13, fontweight="bold")
    # usually no y-label for spectrum; comment out if you had one

    # fig.suptitle(title, y=1.02, fontsize=11)
    fig.tight_layout()

    for ext in (".png", ".pdf"):
        fig.savefig(
            out.with_suffix(ext),
            dpi=300 if ext == ".png" else None,
            bbox_inches="tight",
            transparent=True,
        )
    plt.close(fig)
    return fig


def _save_surface(path: pathlib.Path, xcoords, ycoords, losses):
    """Save as .h5 if possible else .npz."""
    if h5py is not None:
        with h5py.File(path.with_suffix(".h5"), "w") as h:
            h.create_dataset("loss", data=losses)
            h.attrs["xcoords"] = xcoords
            if ycoords is not None:
                h.attrs["ycoords"] = ycoords
    np.savez(path.with_suffix(".npz"), loss=losses, xcoords=xcoords, ycoords=ycoords)


def _parse_slice(spec: str) -> Tuple[np.ndarray, str]:
    start_s, stop_s, num_s = spec.split(":")
    pts = np.linspace(float(start_s), float(stop_s), int(num_s))
    return pts, f"[{start_s}, {stop_s}] ({len(pts)})"


def _clone_params(model: torch.nn.Module) -> List[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters() if p.requires_grad]


def _random_direction(
    model: torch.nn.Module, *, ignore_bias_bn: bool = True
) -> List[torch.Tensor]:
    dirs = []
    for n, p in model.named_parameters():
        if ignore_bias_bn and ("bias" in n or "bn" in n or "batchnorm" in n):
            dirs.append(torch.zeros_like(p))
        else:
            dirs.append(torch.randn_like(p))
    _normalize_dirs(dirs, method="filter")
    return dirs


def _normalize_dirs(dirs: Sequence[torch.Tensor], *, method: str = "filter") -> None:
    if method == "filter":
        for d in dirs:
            if d.ndim > 1:
                d.mul_(1.0 / d.norm())
    else:  # layer
        total = math.sqrt(sum((d**2).sum() for d in dirs))
        for d in dirs:
            d.mul_(1.0 / total)


def _add_direction(
    model: torch.nn.Module,
    base: Sequence[torch.Tensor],
    direction: Sequence[torch.Tensor],
    coef: float,
) -> None:
    with torch.no_grad():
        for p, θ0, d in zip(model.parameters(), base, direction):
            p.copy_(θ0 + coef * d)


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str
) -> float:
    model.eval()
    total, N = 0.0, 0
    for batch in dataloader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        total += -model.log_prob(x).mean().item() * x.size(0)
        N += x.size(0)
    return total / max(N, 1)


# ───────────────────── Hessian eigen‑values (optional) ───────────────────────


def _hvp(model: torch.nn.Module, dataloader, vec, device):
    params = [p for p in model.parameters() if p.requires_grad]

    model.zero_grad()
    for batch in dataloader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        (-model.log_prob(x).sum()).backward(create_graph=True)
    grad = torch.cat([p.grad.reshape(-1) for p in params])

    grad_dot_vec = (grad * vec).sum()
    model.zero_grad()
    grad_dot_vec.backward(retain_graph=True)
    hvp = torch.cat([p.grad.reshape(-1) for p in params])
    return hvp


def _power_iter(
    model, dataloader, k=20, iters=100, tol=1e-3, device="cuda"
) -> List[float]:
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    eigen_vals, eigen_vecs = [], []

    for _ in range(k):
        v = torch.randn(n_params, device=device)
        v /= v.norm()
        λ_prev = None
        for _ in range(iters):
            w = _hvp(model, dataloader, v, device)
            for ev in eigen_vecs:
                w = w - (w @ ev) * ev
            λ = w.norm()
            v = w / max(λ, 1e-12)
            if λ_prev is not None and abs(λ - λ_prev) < tol * λ_prev:
                break
            λ_prev = λ
        eigen_vals.append(float(λ_prev))
        eigen_vecs.append(v)
    return eigen_vals


# ─────────────────────── surface computation ────────────────────────────────


def _compute_surface(
    model, base_state, dir_x, xcoords, dataloader, *, device, dir_y=None, ycoords=None
):
    if ycoords is None:
        losses = np.zeros(len(xcoords), dtype=np.float32)
        for i, α in enumerate(xcoords):
            _add_direction(model, base_state, dir_x, α)
            losses[i] = _evaluate(model, dataloader, device)
            _add_direction(model, base_state, dir_x, 0.0)
        return losses
    losses = np.zeros((len(xcoords), len(ycoords)), dtype=np.float32)
    for i, α in enumerate(xcoords):
        for j, β in enumerate(ycoords):
            _add_direction(model, base_state, dir_x, α)
            _add_direction(model, model.parameters(), dir_y, β)
            losses[i, j] = _evaluate(model, dataloader, device)
            _add_direction(model, base_state, dir_x, 0.0)
    return losses


# ───────────────────────────── fancy plotting ───────────────────────────────


def _plot_1d(xcoords, train_losses, valid_losses, out: pathlib.Path, *, title: str):
    fig, ax = plt.subplots()
    if len(np.array(train_losses).shape) > 1:
        train_losses = np.array(train_losses)
        valid_losses = np.array(valid_losses)
        idx = np.unravel_index(np.argmin(train_losses), train_losses.shape)[1]
        train_losses = train_losses[:, idx]
        valid_losses = valid_losses[:, idx]
    ax.plot(xcoords, train_losses, label="Train", color="red")
    ax.plot(xcoords, valid_losses, label="Valid", color="blue", linestyle="--")

    ax.set_xlabel(r"$\alpha$", fontsize=13, fontweight="bold")
    ax.set_ylabel("Negative log‑likelihood ↓", fontsize=13, fontweight="bold")
    # ax.set_title(title)
    ax.legend(frameon=False, fontsize=13)
    ax.grid(which="both", linestyle=":", linewidth=0.5)
    _finalize_and_save(fig, out)
    return fig


def _plot_2d(xcoords, ycoords, losses, out: pathlib.Path, *, title: str):
    X, Y = np.meshgrid(xcoords, ycoords, indexing="ij")

    # 2‑D contour for quick insight -----------------------------------------
    fig2, ax2 = plt.subplots()
    ax2.contourf(X, Y, losses, levels=40, cmap="viridis")
    # cbar = fig2.colorbar(ax2.collections[0], ax=ax2, pad=0.02, aspect=30)
    # cbar.ax.set_ylabel("Loss")
    # ax2.set_xlabel(r"$\alpha$")
    # ax2.set_ylabel(r"$\beta$")
    # ax2.set_title(title.replace("Surface", "Contour"))
    
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    _finalize_and_save(fig2, out.with_name(out.stem + "_contour"))

    # 3‑D surface -----------------------------------------------------------
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        X,
        Y,
        losses,
        cmap="viridis",
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=True,
        alpha=0.95,
    )

    # ax.set_xlabel(r"$\alpha$", fontsize=10, fontweight="bold")
    # ax.set_ylabel(r"$\beta$", fontsize=10, fontweight="bold")
    # ax.set_zlabel("Loss", labelpad=5)
    ax.view_init(elev=25, azim=135)
    # fig.colorbar(ax.collections[0], shrink=0.6, aspect=12, pad=0.05, label="Loss")
    # ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    _finalize_and_save(fig, out)
    return fig2, fig


# ────────────────────────────── plot utilities ──────────────────────────────
def _plot_eigs(
    eigs: Sequence[float],
    out: pathlib.Path,
    *,
    title: str = "Hessian spectrum (log-scale)",
) -> plt.Figure:
    """
    Save a publication-ready semilog plot of the Hessian eigen-values.

    Parameters
    ----------
    eigs : list | np.ndarray
        Eigen-values (will be sorted in descending order before plotting).
    out : pathlib.Path
        Output path *without* suffix – `.png` and `.pdf` are created.
    title : str, optional
        Figure title.
    """
    eigs = np.sort(np.asarray(eigs, dtype=np.float32))[::-1]  # ↓ sort desc
    cum_energy = np.cumsum(eigs) / eigs.sum()

    idx = np.arange(1, len(eigs) + 1)

    # --- styling
    cmap = plt.get_cmap("viridis")
    norm = colors.LogNorm(vmin=eigs.min(), vmax=eigs.max())
    bar_colors = cmap(norm(eigs))

    fig, ax1 = plt.subplots(figsize=(4, 4))

    # bars: λₖ on log-scale
    ax1.bar(idx, eigs, color=bar_colors, alpha=0.9, width=0.9, edgecolor="none")
    ax1.set_yscale("log")
    ax1.set_xlabel("Eigen-index  $k$", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Eigen-value  $\\lambda_k$", fontsize=10, fontweight="bold")
    # ax1.set_title(title, pad=6)
    ax1.grid(True, which="both", ls="--", alpha=0.3)

    # colour-bar (acts as legend for magnitude)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # cbar = fig.colorbar(sm, ax=ax1, fraction=0.046, pad=0.03)
    # cbar.set_label("")
    # cbar.set_ticks([])
    # cbar.set_ticklabels([])

    # secondary axis: cumulative energy
    ax2 = ax1.twinx()
    ax2.plot(idx, cum_energy * 100, color="black", lw=1.0, linestyle="--", marker="o", ms=2)
    ax2.set_ylim(0, 105)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax2.grid(False)
    ax2.set_yticklabels([], fontsize=13)
    # # highlight 90 % cutoff
    # k90 = np.searchsorted(cum_energy, 0.9) + 1
    # ax1.axvline(k90 + 0.5, color="red", ls=":", lw=1.2)
    # ax2.axhline(90, color="red", ls=":", lw=1.2)
    # ax2.text(k90 + 1, 92, f"90 % at $k={k90}$", color="red", fontsize=7)

    # fig.tight_layout()

    # save
    for ext in (".png", ".pdf"):
        fig.savefig(
            out.with_suffix(ext),
            dpi=300 if ext == ".png" else None,
            bbox_inches="tight",
            transparent=True,
        )

    plt.close(fig)
    return fig


def _plot_summary(
    fig_1d: plt.Figure,
    fig_contour: plt.Figure,
    fig_eigs: Optional[plt.Figure],
    out: pathlib.Path,
    *,
    title: str = "Loss-landscape summary",
) -> plt.Figure:
    """
    Assemble a 2 × 2 “dashboard” figure from the already-created sub-figures.

    Parameters
    ----------
    fig_1d, fig_contour, fig_eigs : matplotlib.figure.Figure
        The individual panels returned by your plotting helpers.  `fig_eigs`
        may be *None* (e.g. if you skipped the Hessian computation).
    out : pathlib.Path
        Output path *without* suffix – `.png` and `.pdf` are created.
    title : str, optional
        Title centred at the top of the dashboard.
    """

    # helper to rasterise a figure into an RGBA array we can `imshow`
    def _fig_to_rgba(fig: plt.Figure) -> np.ndarray:
        import io
        from PIL import Image

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        rgba = np.array(Image.open(buf))
        buf.close()
        return rgba

    imgs = [_fig_to_rgba(fig_1d), _fig_to_rgba(fig_contour)]
    if fig_eigs is not None:
        imgs.append(_fig_to_rgba(fig_eigs))

    n_panels = len(imgs)
    n_rows = 1
    n_cols = 3
    summary = plt.figure(figsize=(4, 4))
    gs = summary.add_gridspec(n_rows, n_cols, hspace=0.05, wspace=0.05)

    for idx, img in enumerate(imgs):
        r, c = divmod(idx, n_cols)
        ax = summary.add_subplot(gs[r, c])
        ax.imshow(img)
        ax.axis("off")  # no ticks/frames

    # fill empty slot(s) if spectrum is absent
    for idx in range(n_panels, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        ax = summary.add_subplot(gs[r, c])
        ax.axis("off")

    summary.suptitle(title, fontsize=13, y=0.98)
    summary.tight_layout(rect=[0, 0, 1, 0.97])

    for ext in (".png", ".pdf"):
        summary.savefig(
            out.with_suffix(ext),
            dpi=300 if ext == ".png" else None,
            bbox_inches="tight",
        )
    print(f"Saved summary figure to {out.with_suffix('.png')}")
    plt.close(summary)
    return summary


def _finalize_and_save(fig: plt.Figure, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"), transparent=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, transparent=True)
    plt.close(fig)


# ────────────────────────────────── run ─────────────────────────────────────
if __name__ == "__main__":
    main()
