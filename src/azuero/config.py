"""Project configuration loading and resource-profile resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(os.environ.get("AZUERO_CONFIG", ROOT / "config.yaml"))


def _coerce(value: str) -> Any:
    """Parse an environment-variable override with YAML semantics."""
    return yaml.safe_load(value)


@dataclass(frozen=True)
class Resources:
    """Resolved resource toggles (preset + per-key overrides)."""

    profile: str
    values: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, key: str) -> Any:
        try:
            return self.values[key]
        except KeyError as e:  # pragma: no cover - programming error
            raise AttributeError(key) from e


def resolve_resources(cfg: dict, profile: str | None = None,
                      overrides: dict | None = None) -> Resources:
    """Combine a named preset with config/env/explicit overrides.

    Precedence (highest last): preset < config `resources.overrides` <
    AZUERO_RESOURCES__<KEY> env vars < `overrides` argument.
    """
    rcfg = cfg["resources"]
    profile = profile or os.environ.get("AZUERO_PROFILE") or rcfg["profile"]
    if profile not in rcfg["presets"]:
        raise ValueError(f"unknown resource profile {profile!r}")
    values = dict(rcfg["presets"][profile])
    values.update(rcfg.get("overrides") or {})
    prefix = "AZUERO_RESOURCES__"
    for k, v in os.environ.items():
        if k.startswith(prefix):
            values[k[len(prefix):].lower()] = _coerce(v)
    values.update(overrides or {})
    return Resources(profile=profile, values=values)


@lru_cache(maxsize=1)
def load_config(path: str | Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resources(profile: str | None = None, **overrides: Any) -> Resources:
    return resolve_resources(load_config(), profile, overrides)


def path(kind: str, *parts: str) -> Path:
    """Absolute path under one of the configured roots (data/results/site)."""
    p = ROOT / load_config()["paths"][kind]
    return p.joinpath(*parts)


def data_path(*parts: str, mkdir: bool = True) -> Path:
    p = path("data", *parts)
    if mkdir:
        (p if not p.suffix else p.parent).mkdir(parents=True, exist_ok=True)
    return p


def results_path(*parts: str, mkdir: bool = True) -> Path:
    p = path("results", *parts)
    if mkdir:
        (p if not p.suffix else p.parent).mkdir(parents=True, exist_ok=True)
    return p
