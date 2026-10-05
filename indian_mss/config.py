"""Loader for ``configs/default.yaml``, the single source of hyperparameters."""

from __future__ import annotations

from pathlib import Path

__all__ = ["DEFAULT_CONFIG_PATH", "load_config"]

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def load_config(path: str | Path) -> dict:
    """Parse a YAML config into a nested dict. An empty file gives ``{}``."""
    import yaml

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return cfg
