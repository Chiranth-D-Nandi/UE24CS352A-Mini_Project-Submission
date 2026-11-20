"""Configuration loading and path resolution."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and resolve data paths relative to the repository root."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")

    config = deepcopy(config)
    root = config_path.parent.parent
    config["_meta"] = {"config_path": str(config_path), "project_root": str(root)}

    path_keys = {
        "spotify_csv",
        "billboard_archive_csv",
        "billboard_current_csv",
        "billboard_hits",
        "modeling_dataset",
        "match_audit",
        "unmatched_billboard",
        "manifest",
    }
    for key in path_keys:
        if key in config.get("data", {}):
            value = Path(config["data"][key]).expanduser()
            if not value.is_absolute():
                value = root / value
            config["data"][key] = str(value.resolve())
    return config


def project_path(config: dict[str, Any], *parts: str) -> Path:
    return Path(config["_meta"]["project_root"]).joinpath(*parts)


def ensure_output_directories(config: dict[str, Any]) -> None:
    """Create only generated-data/report directories, never raw source files."""
    root = Path(config["_meta"]["project_root"])
    for relative in [
        "data/raw/spotify",
        "data/raw/billboard",
        "data/interim",
        "data/processed",
        "artifacts",
        "reports/figures",
        "reports/metrics",
        "reports/experiments",
        "reports/audits",
    ]:
        (root / relative).mkdir(parents=True, exist_ok=True)
