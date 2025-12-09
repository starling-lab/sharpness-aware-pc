"""Lightweight .env loader (no external dependency)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _parse_env_lines(lines: Iterable[str]) -> dict[str, str]:
    env = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def load_env(env_file: str | Path | None = None) -> Path | None:
    """
    Load key=value pairs from a `.env` file into os.environ (without overwriting
    variables that are already set). Returns the path that was loaded, or None if
    nothing was found.

    Search order:
    1) Explicit `env_file` argument (if provided)
    2) Path in $HESSIAN_ENV_FILE (if set)
    3) `.env` in the current working directory
    4) `.env` at the project root
    """
    candidates: list[Path] = []
    if env_file:
        candidates.append(Path(env_file))
    if os.getenv("HESSIAN_ENV_FILE"):
        candidates.append(Path(os.environ["HESSIAN_ENV_FILE"]))
    candidates.append(Path.cwd() / ".env")
    candidates.append(PROJECT_ROOT / ".env")

    for path in candidates:
        if path and path.exists():
            new_vars = _parse_env_lines(path.read_text().splitlines())
            for k, v in new_vars.items():
                os.environ.setdefault(k, v)
            return path
    return None
