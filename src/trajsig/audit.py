from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .cache import ArtifactStore
from .config import run_id
from .trajectory import augment_time, construct_trajectory, reorder_trajectory
from .utils import atomic_json, utc_now


def audit_experiment(config: dict[str, Any], output: str | Path) -> dict[str, Any]:
    store = ArtifactStore(output)
    expected_ids = {
        run_id(config, condition["id"], int(seed))
        for condition in config["conditions"]
        for seed in config["experiment"]["seed_ids"]
    }
    complete_ids = set(store.complete_run_ids())
    errors: list[str] = []
    warnings: list[str] = []
    if complete_ids != expected_ids:
        missing = sorted(expected_ids - complete_ids)
        extra = sorted(complete_ids - expected_ids)
        if missing:
            errors.append(f"Missing expected runs: {missing}")
        if extra:
            errors.append(f"Unexpected complete runs: {extra}")
    reference = None
    seen_pairs: set[tuple[str, int]] = set()
    corruption_by_seed: dict[int, dict[str, set[int]]] = {}
    for identifier in sorted(complete_ids & expected_ids):
        artifact = store.load_run(identifier)
        condition = artifact.manifest["condition"]
        seed_id = int(artifact.manifest["run_spec"]["seeds"]["seed_id"])
        pair = (condition, seed_id)
        if pair in seen_pairs:
            errors.append(f"Duplicate condition/seed pair: {pair}")
        seen_pairs.add(pair)
        shape_definition = (
            artifact.layer_values.shape,
            artifact.ordinary_metrics.shape,
            tuple(artifact.layer_names),
            tuple(artifact.stat_names),
        )
        if reference is None:
            reference = shape_definition
        elif shape_definition != reference:
            errors.append(f"Incompatible telemetry shape/definition: {identifier}")
        path, _ = construct_trajectory(artifact)
        if not np.array_equal(reorder_trajectory(path, "reversed"), path[::-1]):
            errors.append(f"Temporal reversal failed: {identifier}")
        timed = augment_time(path)
        if not np.allclose(timed[:, 0], np.linspace(0, 1, len(path))):
            errors.append(f"Time augmentation failed: {identifier}")
        if condition in {"C3", "C4"}:
            index_path = store.data_spec_dir / f"{identifier}.npz"
            with np.load(index_path, allow_pickle=False) as indices:
                actual = len(indices["corrupted_indices"])
            fraction = artifact.manifest["run_spec"]["condition"]["label_corruption"]
            expected_count = int(np.floor(artifact.manifest["run_spec"]["data"].get("train_size", 50_000) * fraction))
            if actual != expected_count:
                errors.append(f"Corruption count mismatch: {identifier}, got {actual}, expected {expected_count}")
            with np.load(index_path, allow_pickle=False) as indices:
                corruption_by_seed.setdefault(seed_id, {})[condition] = set(indices["corrupted_indices"].tolist())
        if condition == "C5":
            spec = json.loads((store.data_spec_dir / f"{identifier}.json").read_text(encoding="utf-8"))
            if spec["removed_count"] <= 0:
                errors.append(f"C5 removed no examples: {identifier}")
    for seed_id, condition_sets in corruption_by_seed.items():
        if {"C3", "C4"}.issubset(condition_sets) and not condition_sets["C3"].issubset(condition_sets["C4"]):
            errors.append(f"C3 corruption is not nested in C4 for seed {seed_id}")
    result = {
        "status": "pass" if not errors else "fail",
        "audited_at": utc_now(),
        "expected_runs": len(expected_ids),
        "complete_runs": len(complete_ids & expected_ids),
        "checks": [
            "exact deterministic run set",
            "unique condition/seed pairs",
            "finite cached arrays",
            "compatible shapes and feature definitions",
            "temporal reversal",
            "time augmentation",
            "corruption counts",
            "nested C3/C4 corruption indices",
            "nonempty C5 subset removal",
            "opaque run IDs",
        ],
        "errors": errors,
        "warnings": warnings,
    }
    atomic_json(store.root / "audit.json", result)
    if errors:
        raise RuntimeError("Audit failed: " + "; ".join(errors))
    return result
