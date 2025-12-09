#!/usr/bin/env python3
"""Experiment driver for Tables 1–2 (sharpness vs samples)."""

from __future__ import annotations
import time
import json
import hydra
from omegaconf import DictConfig, open_dict
import torch
import matplotlib.pyplot as plt
import os
from hessian_reg.models import load_model
from hessian_reg.datasets import (
    get_dataloaders
)
from hessian_reg.trainer import Trainer
from pathlib import Path
import sys
import numpy as np
from hessian_reg.utils import set_seed
from hessian_reg.utils.profiler_reg import (
    adaptive_mu,
    mean_flow_mu,
    RegScheduler,
)
from hessian_reg.utils.env import load_env

# Load user overrides from .env (project root or cwd)
load_env()

sharpness = []
degree_of_overfitting = []
performance_summary = {}
loss_agg = "mean" # can also be 'sum'

@hydra.main(version_base=None, config_path="../../configs", config_name="main")
def main(cfg: DictConfig):
    # Use Hydra’s run.dir as base output
    out_dir = os.getcwd()

    set_seed(cfg.get("seed", 0))
    mu_value = cfg.trainer.get("mu", 0.0)
    lambda_value = cfg.trainer.get("lambda_", 1.0)
    print("Using μ =", mu_value, " λ =", lambda_value)
    with open_dict(cfg.trainer):
        cfg.trainer.mu = mu_value
        cfg.trainer.lambda_ = lambda_value
    cfg.model["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    # ------------------ dataset -------------------------------------- #
    train_dl, valid_dl, test_dl = get_dataloaders(cfg.dataset)

    # ------------------ model ---------------------------------------- #
    if "em" in cfg.trainer.mode:
        cfg.model["use_em"] = True
    model = load_model(cfg.model, next(iter(train_dl))[0].to(cfg.model["device"]))

    trainer = Trainer(
        model,
        train_dl,
        valid_loader=valid_dl,
        mode=cfg.trainer.mode,
        optimizer_cls=cfg.trainer.optimizer_cls,
        lr=cfg.trainer.lr,
        mu=cfg.trainer.mu,
        lambda_=cfg.trainer.lambda_,
        device=cfg.model.device,
        online_em="online" in cfg.trainer.mode.lower()
        and "em" in cfg.trainer.mode.lower(),
        pseudocount=cfg.trainer.pseudocount,
    )

    metrics = {
        "trn_nll": [],
        "val_nll": [],
        "test_nll": [],
        "ssg_sharpness": [],
        "degree_of_overfitting": [],
        "epoch": [],
    }

    os.makedirs("./models", exist_ok=True)

    sched = RegScheduler(
        init_mu=10, dof_target=0.001, beta=0.1, period=1, ema_alpha=0.1
    )

    for epoch in range(0, cfg.trainer.epochs + 1):
        t0 = time.time()
        trn_nll = (
            trainer.train_epoch() if epoch > 0 else trainer.evaluate(train_dl)
        )  # one epoch
        train_time = time.time() - t0
        val_nll = trainer.evaluate(valid_dl)

        if trn_nll == np.inf or val_nll == np.inf:
            print("Divergence detected, stopping training.")
            exit(0)

        dof = abs(val_nll - trn_nll) / abs(val_nll)

        delta_trn_nll = (
            100 * (trn_nll - metrics["trn_nll"][-1])
            if len(metrics["trn_nll"]) > 0
            else 0
        )
        delta_val_nll = (
            100 * (val_nll - metrics["val_nll"][-1])
            if len(metrics["val_nll"]) > 0
            else 0
        )

        if epoch % 1 == 0 and trainer.mu_schedule is not None:
            schedule_name = trainer.mu_schedule
            if "dof" in schedule_name:
                mu_new = dof
                
            elif "mean_flow" in schedule_name:
                mu_new = mean_flow_mu(
                    model=trainer.model,
                    train_loader=train_dl,
                    device=cfg.model.device,
                    max_batches=5,
                )
            elif "ema" in schedule_name:
                mu_new = sched.update(trn_nll, val_nll)

            elif "auto" in schedule_name:
                mu_new = adaptive_mu(
                    model=trainer.model,
                    train_loader=train_dl,
                    device=cfg.model.device,
                    dof_now=dof,                  # |val-train| / |val|
                    delta_trn_nll=delta_trn_nll,  # 100*(trn_nll - previous_trn_nll)
                    delta_val_nll=delta_val_nll,  # 100*(val_nll
                    alpha=1.0,
                    kappa=1.05,
                    max_batches=5,
                )
            else:
                mu_new = trainer.mu

            print("Adaptive μ:", str(mu_new))
            trainer.mu = mu_new

        total_sharpness = 0.0
        total_samples = 0
        
        # Compute mean sharpness across train dataset using the SSG trace
        for batch in train_dl:
            x = batch[0].to(cfg.model["device"])
            total_sharpness += model.trace_hessian(x).item()
            total_samples += len(x)

        # Sharpness w.r.t mean NLL surface would incur a normalization by N^2 from the hessian-trace formula
        N = max(total_samples, 1)**2 if cfg.trainer.loss_agg == "mean" else max(total_samples, 1) 
        ssg_pk = total_sharpness / N
        
        metrics["ssg_sharpness"].append(ssg_pk)
        metrics["degree_of_overfitting"].append(dof)

        print(
            f"Epoch {epoch:3d}  trn NLL {trn_nll:.2f} val NLL {val_nll:.2f} Sharpness {ssg_pk:.2f}  train time {train_time:.1f}s"
        )

        metrics["trn_nll"].append(trn_nll)
        metrics["val_nll"].append(val_nll)
        metrics["epoch"].append(epoch)


    save_final = cfg.trainer.get("save_final", False)
    if save_final:
        torch.save(model.state_dict(), f"models/N{cfg.dataset.n_samples}_final_model.pt")

    test_nll = trainer.evaluate(test_dl)
    metrics["test_nll"].append(test_nll)
    print(f"Test NLL: {test_nll:.2f}")

    mu_cfg = cfg.trainer.mu
    plt.figure(figsize=(12, 5))

    # Plotting Train, Val, and Test NLL across epochs
    plt.subplot(1, 2, 1)
    plt.plot(
        metrics["epoch"],
        metrics["trn_nll"],
        label="Train NLL",
        marker="o",
        color="blue",
        ms=2,
    )
    plt.plot(
        metrics["epoch"],
        metrics["val_nll"],
        label="Validation NLL",
        marker="s",
        color="green",
        ms=2,
    )
    plt.axhline(
        metrics["test_nll"][0], label="Test NLL", marker="^", color="orange", ms=3
    )
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log-Likelihood (NLL)")
    plt.title(f"NLL across Epochs, #Samples={cfg.dataset.n_samples} (Reg={mu_cfg})")
    plt.legend()
    plt.grid(True)

    plt.tick_params(which="minor")
    # Plotting sharpness across epochs
    plt.subplot(1, 2, 2)
    plt.plot(
        metrics["epoch"],
        metrics["ssg_sharpness"],
        linestyle="-",
        color="orange",
        ms=1.5,
        label="Sharpness",
    )
    plt.xlabel("Epoch")
    plt.legend()
    plt.ylabel("Sharpness (Trace of Hessian)")
    plt.title(
        f"Sharpness across Epochs, #Samples={cfg.dataset.n_samples} (Reg={mu_cfg})"
    )

    plt.grid(True)
    plt.tick_params(which="minor")
    plt.tight_layout()
    output_path = Path(f"./N{cfg.dataset.n_samples}_{cfg.exp.name}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)

    dof = abs(metrics["trn_nll"][-1] - metrics["test_nll"][-1]) / abs(
        metrics["trn_nll"][-1]
    )
    print(f"Degree of overfitting: {dof}")
    sharpness.append(metrics["ssg_sharpness"][-1])
    degree_of_overfitting.append(dof)

    performance_summary[cfg.dataset.n_samples] = metrics
    with open(f"{out_dir}/performance_summary_{cfg.exp.name}.json", "w") as f:
        json.dump(performance_summary, f, indent=4)

    # ------------------ markdown report ------------------------------ #
    report = Path(out_dir) / "report.md"
    report.write_text(
        f"# Experiment Report\n\n"
        f"**Config**: `{cfg}`\n\n"
        f"* Test NLL: **{test_nll:.2f}**\n"
    )


if __name__ == "__main__":
    num_samples_frac = [0.01, 0.05, 0.10, 0.50, 1.00]  
    for n_sample in num_samples_frac:
        sys.argv.append(f"dataset.n_samples={n_sample}")
        print(f"Running for n_samples: {n_sample}")
        main()
        sys.argv.remove(f"dataset.n_samples={n_sample}")
        
