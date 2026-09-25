from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .utils import assert_finite, atomic_json, atomic_npz


@dataclass
class RunArtifact:
    run_id: str
    manifest: dict[str, Any]
    layer_values: np.ndarray
    ordinary_metrics: np.ndarray
    layer_names: list[str]
    stat_names: list[str]
    metric_names: list[str]
    parameter_counts: np.ndarray


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.telemetry_dir = self.root / "telemetry"
        self.manifest_dir = self.root / "manifests"
        self.checkpoint_dir = self.root / "checkpoints"
        self.data_spec_dir = self.root / "data_specs"
        self.representation_dir = self.root / "representations"
        self.evaluation_dir = self.root / "evaluation"
        self.figure_dir = self.root / "figures"
        self.report_dir = self.root / "reports"
        for directory in [
            self.telemetry_dir,
            self.manifest_dir,
            self.checkpoint_dir,
            self.data_spec_dir,
            self.representation_dir,
            self.evaluation_dir,
            self.figure_dir,
            self.report_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def telemetry_path(self, run_id: str) -> Path:
        return self.telemetry_dir / f"{run_id}.npz"

    def manifest_path(self, run_id: str) -> Path:
        return self.manifest_dir / f"{run_id}.json"

    def checkpoint_path(self, run_id: str) -> Path:
        return self.checkpoint_dir / f"{run_id}.pt"

    def is_complete(self, run_id: str, epochs: int) -> bool:
        if not self.telemetry_path(run_id).exists() or not self.manifest_path(run_id).exists():
            return False
        try:
            manifest = json.loads(self.manifest_path(run_id).read_text(encoding="utf-8"))
            return manifest.get("status") == "complete" and manifest.get("completed_epochs") == epochs
        except (json.JSONDecodeError, OSError):
            return False

    def save_data_spec(self, run_id: str, spec: dict[str, Any], affected: np.ndarray, removed: np.ndarray) -> None:
        atomic_json(self.data_spec_dir / f"{run_id}.json", spec)
        atomic_npz(
            self.data_spec_dir / f"{run_id}.npz",
            corrupted_indices=np.asarray(affected, dtype=np.int64),
            removed_indices=np.asarray(removed, dtype=np.int64),
        )

    def save_run(
        self,
        run_id: str,
        manifest: dict[str, Any],
        layer_values: np.ndarray,
        ordinary_metrics: np.ndarray,
        layer_names: list[str],
        stat_names: list[str],
        metric_names: list[str],
        parameter_counts: np.ndarray,
    ) -> None:
        assert_finite("layer telemetry", layer_values)
        assert_finite("ordinary metrics", ordinary_metrics)
        atomic_npz(
            self.telemetry_path(run_id),
            layer_values=np.asarray(layer_values, dtype=np.float64),
            ordinary_metrics=np.asarray(ordinary_metrics, dtype=np.float64),
            layer_names=np.asarray(layer_names, dtype=str),
            stat_names=np.asarray(stat_names, dtype=str),
            metric_names=np.asarray(metric_names, dtype=str),
            parameter_counts=np.asarray(parameter_counts, dtype=np.int64),
        )
        atomic_json(self.manifest_path(run_id), manifest)

    def load_run(self, run_id: str) -> RunArtifact:
        manifest = json.loads(self.manifest_path(run_id).read_text(encoding="utf-8"))
        with np.load(self.telemetry_path(run_id), allow_pickle=False) as data:
            artifact = RunArtifact(
                run_id=run_id,
                manifest=manifest,
                layer_values=data["layer_values"],
                ordinary_metrics=data["ordinary_metrics"],
                layer_names=data["layer_names"].astype(str).tolist(),
                stat_names=data["stat_names"].astype(str).tolist(),
                metric_names=data["metric_names"].astype(str).tolist(),
                parameter_counts=data["parameter_counts"],
            )
        assert_finite(f"{run_id} layer telemetry", artifact.layer_values)
        assert_finite(f"{run_id} ordinary metrics", artifact.ordinary_metrics)
        return artifact

    def complete_run_ids(self) -> list[str]:
        ids: list[str] = []
        for path in sorted(self.manifest_dir.glob("*.json")):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                if manifest.get("status") == "complete":
                    ids.append(path.stem)
            except (OSError, json.JSONDecodeError):
                continue
        return ids

