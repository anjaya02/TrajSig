from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .utils import stable_hash


REQUIRED_CONDITIONS = {"C0", "C1", "C2", "C3", "C4", "C5"}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    validate_config(config)
    config["_config_path"] = str(config_path.resolve())
    return config


def validate_config(config: dict[str, Any]) -> None:
    for key in ["experiment", "data", "training", "conditions", "telemetry", "analysis"]:
        if key not in config:
            raise ValueError(f"Missing configuration section: {key}")
    names = [item["id"] for item in config["conditions"]]
    if len(names) != len(set(names)):
        raise ValueError("Condition IDs must be unique")
    if config["experiment"].get("phase") == "full" and set(names) != REQUIRED_CONDITIONS:
        raise ValueError(f"Full experiment requires exactly {sorted(REQUIRED_CONDITIONS)}")
    seeds = config["experiment"].get("seed_ids", [])
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError("At least two unique seed identities are required")
    if int(config["training"]["epochs"]) < 2:
        raise ValueError("At least two trajectory points are required")


def condition_config(config: dict[str, Any], condition_id: str) -> dict[str, Any]:
    matches = [item for item in config["conditions"] if item["id"] == condition_id]
    if len(matches) != 1:
        raise KeyError(condition_id)
    return deepcopy(matches[0])


def derive_seeds(seed_id: int) -> dict[str, int]:
    # Disjoint namespaces make accidental coupling explicit. C3/C4 share corruption
    # seeds, making the 10% corruption set a deterministic subset of the 20% set.
    return {
        "seed_id": int(seed_id),
        "initialization": 10_000 + int(seed_id),
        "training": 20_000 + int(seed_id),
        "data_order": 30_000 + int(seed_id),
        "augmentation": 40_000 + int(seed_id),
        "corruption": 50_000 + int(seed_id),
        "subset": 60_000 + int(seed_id),
    }


def scientific_config(config: dict[str, Any]) -> dict[str, Any]:
    return {k: deepcopy(v) for k, v in config.items() if not k.startswith("_") and k != "paths"}


def run_spec(config: dict[str, Any], condition_id: str, seed_id: int) -> dict[str, Any]:
    return {
        "schema_version": 2,
        # Only training-affecting fields belong here. In particular, analysis,
        # classifiers, signature depth, and the list of other planned runs must
        # never invalidate reusable neural-training telemetry.
        "experiment": {
            "name": config["experiment"]["name"],
            "phase": config["experiment"]["phase"],
            "deterministic": config["experiment"].get("deterministic", True),
        },
        "data": deepcopy(config["data"]),
        "training": deepcopy(config["training"]),
        "telemetry": deepcopy(config["telemetry"]),
        "condition": condition_config(config, condition_id),
        "seeds": derive_seeds(seed_id),
    }


def run_id(config: dict[str, Any], condition_id: str, seed_id: int) -> str:
    # Opaque IDs deliberately do not embed condition or seed labels.
    return stable_hash(run_spec(config, condition_id, seed_id), 20)


def experiment_id(config: dict[str, Any]) -> str:
    return stable_hash(scientific_config(config), 16)
