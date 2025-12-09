#!/usr/bin/env python3
"""
Unified experiment script for *pc‑hessian*.
* Hydra‑driven configuration
* wandb logging
* final test evaluation
* optional 2‑D visualisation for synthetic datasets
* Markdown report saved to results/<run‑id>/report.md
"""

from __future__ import annotations
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
import os
from hessian_reg.models import load_model
from hessian_reg.datasets import get_dataloaders
from hessian_reg.trainer import Trainer
from pathlib import Path
from hessian_reg.utils import set_seed
from hessian_reg.utils.profiler_reg import adaptive_mu, mean_flow_mu, RegScheduler
from hessian_reg.utils.env import load_env

# Load user overrides from .env (project root or cwd)
load_env()

# Hydra entry point
@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(cfg: DictConfig):
    # Use Hydra’s run.dir as base output
    out_dir = os.getcwd()

    set_seed(cfg.get("seed", 0))
    cfg.model["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------ dataset -------------------------------------- #
    train_dl, valid_dl, test_dl = get_dataloaders(cfg.dataset)

    # ------------------ model ---------------------------------------- #
    if "em" in cfg.trainer.mode:
        cfg.model["use_em"] = True
    model = load_model(cfg.model, next(iter(train_dl))[0].to(cfg.model["device"]))

    mu_value = cfg.trainer.get("mu", 0.0)
    lambda_value = cfg.trainer.get("lambda_", 1.0)
    print("Using μ =", mu_value, " λ =", lambda_value)
    trainer = Trainer(
        model,
        train_dl,
        valid_loader=valid_dl,
        mode=cfg.trainer.mode,
        optimizer_cls=cfg.trainer.optimizer_cls,
        lr=cfg.trainer.lr,
        mu=mu_value,
        lambda_=lambda_value,
        pseudocount=cfg.trainer.get("pseudocount", 0.0),
        device=cfg.model.device,
    )

    sched = RegScheduler(
        init_mu=10, dof_target=0.001, beta=0.1, period=1, ema_alpha=0.1
    )
    prev_trn_nll, prev_val_nll = None, None

    # ------------------ wandb ---------------------------------------- #
    wandb_run = None
    if cfg.get("wandb_project"):
        wandb_run = wandb.init(
            project=cfg.get("wandb_project", "pc-hessian"),
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.exp.name,
            dir=out_dir,
        )
        wandb_run.log({"config": OmegaConf.to_container(cfg, resolve=True)})

    # ------------------ training loop  ------------ #
    save_final = cfg.trainer.get("save_final", False)
    
    for epoch in range(cfg.trainer.epochs):
        t0 = time.time()
        trn_nll = trainer.train_epoch()  # one epoch
        val_nll = trainer.evaluate(valid_dl)
        if wandb_run:
            wandb_run.log({"trn/nll": trn_nll, "epoch": epoch})
            wandb_run.log({"val/nll": val_nll, "epoch": epoch})

        denom = abs(val_nll) if abs(val_nll) > 1e-12 else 1e-12
        dof = abs(val_nll - trn_nll) / denom
        delta_trn_nll = (
            100 * (trn_nll - prev_trn_nll) if prev_trn_nll is not None else 0
        )
        delta_val_nll = (
            100 * (val_nll - prev_val_nll) if prev_val_nll is not None else 0
        )

        if trainer.mu_schedule is not None:
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
                    dof_now=dof,
                    delta_trn_nll=delta_trn_nll,
                    delta_val_nll=delta_val_nll,
                    alpha=1.0,
                    kappa=1.05,
                    max_batches=5,
                )
            else:
                mu_new = None

            if mu_new is not None:
                trainer.mu = mu_new


        print(
            f"Epoch {epoch:3d}  trn NLL {trn_nll  :.3f} val NLL {val_nll:.3f}  time {time.time()-t0:.1f}s"
        )
        prev_trn_nll, prev_val_nll = trn_nll, val_nll

    if save_final:
        print("Saving final model to ", out_dir)
        torch.save(model.state_dict(), Path(out_dir) / "final.pt")

    # ------------------ test with final weights ---------------------- #
    test_nll = trainer.evaluate(test_dl)
    if wandb_run:
        wandb_run.log({"test/nll": test_nll})
    print(f"Test NLL: {test_nll:.3f}")

    # ------------------ markdown report ------------------------------ #
    report = Path(out_dir) / "report.md"
    report.write_text(
        f"# Experiment Report\n\n"
        f"**Config**: `{cfg}`\n\n"
        f"* Test NLL: **{test_nll:.3f}**\n"
    )
    if wandb_run:
        wandb_run.save(str(report))
        wandb_run.finish()


if __name__ == "__main__":
    main()
