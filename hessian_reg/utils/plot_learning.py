import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

# colour-blind-safe Set2 palette
BLUE, ORANGE, GREEN, RED = plt.get_cmap("Set2").colors[:4]
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.prop_cycle": cycler("color", [BLUE, ORANGE, GREEN, RED]),
    }
)


# ---------------------------------------------------------------
def load_trials(paths, frac=0.01):
    """Return dict with stacked arrays  (mean over trials)."""
    trials = [json.loads(Path(p).read_text()) for p in paths]
    trim = min(len(t[f"{frac}"]["trn_nll"]) for t in trials)

    def stack(key):
        arr = np.stack([t[f"{frac}"][key][:trim] for t in trials])
        return arr.mean(0), arr.std(0)

    out = {k: stack(k) for k in ("trn_nll", "ssg_sharpness", "val_nll")}
    out["epoch"] = np.asarray(trials[0][f"{frac}"]["epoch"][:trim])
    out["test_nll"] = np.asarray([t[f"{frac}"]["test_nll"][0] for t in trials])
    return out


def make_figure(stats, title, out_png, out_pdf):
    ep = stats["epoch"]
    tr_mu, tr_std = stats["trn_nll"]
    va_mu, va_std = stats["val_nll"]
    ssg_mu, _ = stats["ssg_sharpness"]

    plt.rcParams.update({"font.size": 8})

    # ---- requested colours -----------------------------------------
    RED = "red"  # train
    BLUE = "blue"  # val
    ORANGE = "orange"  # sharpness
    GREY = "green"

    fig, axL = plt.subplots(figsize=(4, 4))

    # ------------- Left axis: Train vs Val NLL ----------------------
    axL.plot(ep, tr_mu, color=BLUE, lw=1.8, label="Train NLL")
    axL.plot(ep, va_mu, color=RED, lw=1.8, label="Val NLL")
    axL.fill_between(ep, tr_mu - tr_std, tr_mu + tr_std, color=BLUE, alpha=0.2)
    axL.fill_between(ep, va_mu - va_std, va_mu + va_std, color=RED, alpha=0.2)
    axL.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    axL.set_ylabel("Negative log-likelihood", fontsize=12, fontweight="bold")

    # ------------- Right axis: Sharpness ---------------------------
    axR = axL.twinx()

    axR.plot(
        ep,
        ssg_mu,
        color=ORANGE,
        lw=1.2,
        linestyle="--",
        marker="o",
        markevery=10,
        ms=5,
        markeredgecolor="white",
        markeredgewidth=0.4,
        label="Sharpness",
        zorder=5,
    )

    axR.set_ylabel("Sharpness (Hessian trace)", fontsize=12, fontweight="bold")

    # ------------- styling ------------------------------------------
    for a in (axL, axR):
        a.minorticks_on()
        a.grid(which="major", lw=0.6, alpha=0.35)
        a.grid(which="minor", lw=0.4, ls=":", alpha=0.18)
        a.tick_params(axis="both", which="major", labelsize=10)  # ← new
        a.tick_params(axis="both", which="minor", labelsize=7)

    # unified legend
    lines = axL.get_lines() + axR.get_lines()
    labels = [line.get_label() for line in lines]
    fig.legend(
        lines,
        labels,
        ncol=1,
        frameon=True,
        fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.6, 0.62),
    )

    # fig.suptitle(title, y=0.99, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print("saved:", out_png, "&", out_pdf)


# ---------------------------------------------------------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser(
        description="Plot 4×4 in. train/test NLL + sharpness curves"
    )
    pa.add_argument(
        "json",
        nargs="+",
        help="one or more performance_summary_*.json files (same setting)",
    )
    pa.add_argument("--title", default="EinsumNet learning dynamics")
    pa.add_argument("--png", default="curve.png")
    pa.add_argument("--pdf", default="curve.pdf")
    pa.add_argument(
        "--data_frac",
        type=float,
        default=0.01,
        help="fraction of data to use for plotting (default: 0.01)",
    )
    args = pa.parse_args()
    print(
        f"Loading {len(args.json)} JSON files from {args.json}... data_frac={args.data_frac}"
    )
    stats = load_trials(args.json, args.data_frac)
    make_figure(stats, args.title, args.png, args.pdf)
