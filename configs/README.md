# Hydra configuration

`configs/main.yaml` holds the defaults used by `python -m hessian_reg.run` and the
experiment utilities in `hessian_reg/experiments/`.

- `model.*` – backend (`pfc`, `pyjuice`), graph shape, leaf distributions, device.
- `trainer.*` – optimisation mode (`sgd`, `em`), learning rate, Hessian
  regularisation strength (`mu`, optional `lambda_` for EM normalisation), epochs, early stopping,
  and `save_final` to persist the last model checkpoint.
- `dataset.*` – dataset name, batch size, `n_samples` (absolute or fraction for synthetic),
  device for loading.
- `exp.*` – naming and output directory template used by Hydra.

Override from the CLI:

```bash
python -m hessian_reg.run \
  dataset.name=8gaussians \
  dataset.n_samples=0.1 \
  model.backend=pfc \
  trainer.mode=sgd \
  trainer.mu=0.1 \
  trainer.lambda_=1.0 \
  trainer.lr=5e-3
```

Hydra writes each run to `results/<dataset>/<backend>/...` (see `exp.base_dir` /
`hydra.run.dir`). Override globally by exporting `RESULTS_DIR` or passing a custom path via the CLI.
