
The `hessian_reg` folder contains modules for data, models, training, and utilities.

### datasets/

- `synthetic.py`: generates 2D/3D toy datasets (spiral, pinwheel, helix, etc.) with train/validation/test splits.
- `binary.py`: DEBD binary density loader with support for sub-sampling via `n_samples`.

### models/
- `base.py`: `BaseCircuit` abstract class defining the PC interface.
- `pyjuice_wrapper.py`: wraps PyJuice TensorCircuit (HCLT) with SGD/EM + sharpness helpers.
- `pfc_wrapper.py`: wrapper for PFC EinsumNetworks with Hessian-trace support.

### experiments/
- `sharpness.py`: experiment driver used by the batch scripts.
- `loss_landscape.py`: loss-surface visualisations for sanity checks.

### run.py

Entrypoint using Hydra. Parses configs and launches training via `trainer.py`.

### trainer.py

Implements training loop, loss computation (-log-likelihood + regularizer), logging, checkpointing.

### utils/
- `hessian.py`: Hessian-trace primitives.
- `plot_synth.py`, `plot_learning.py`: plotting helpers used in the paper.
- `profiler_reg.py`: regularisation scheduling utilities.
- `env.py`: loads `.env` overrides before Hydra initialises.
