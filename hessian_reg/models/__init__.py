from importlib import import_module


def load_model(cfg, data=None):
    backend = cfg.backend.lower()
    if backend in {"pyjuice", "spn"}:
        return import_module("hessian_reg.models.pyjuice_wrapper").build_pyjuice_model(
            cfg, data
        )
    if backend == "pfc":
        return import_module("hessian_reg.models.pfc_wrapper").build_pfc_model(cfg)
    raise ValueError(backend)
