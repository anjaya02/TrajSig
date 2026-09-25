from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .cache import RunArtifact
from .utils import assert_finite


GLOBAL_NAMES = [
    "weight_rms",
    "gradient_rms",
    "update_rms",
    "update_weight_ratio",
    "gradient_mean",
    "gradient_std",
    "near_zero_fraction",
    "gradient_gini",
]


def _global_aggregate(artifact: RunArtifact) -> np.ndarray:
    values = artifact.layer_values
    counts = artifact.parameter_counts.astype(np.float64)
    total = counts.sum()
    if values.ndim != 3 or values.shape[1] != len(counts) or values.shape[2] != 8:
        raise ValueError(f"Unexpected raw telemetry shape for {artifact.run_id}: {values.shape}")
    weight = counts / total
    weight_norm = values[:, :, 0]
    grad_norm = values[:, :, 1]
    update_norm = values[:, :, 2]
    grad_mean_layers = values[:, :, 4]
    grad_std_layers = values[:, :, 5]
    global_weight_norm = np.sqrt(np.square(weight_norm).sum(axis=1))
    global_grad_norm = np.sqrt(np.square(grad_norm).sum(axis=1))
    global_update_norm = np.sqrt(np.square(update_norm).sum(axis=1))
    grad_mean = (grad_mean_layers * weight[None, :]).sum(axis=1)
    second_moment = ((np.square(grad_std_layers) + np.square(grad_mean_layers)) * weight[None, :]).sum(axis=1)
    grad_std = np.sqrt(np.maximum(second_moment - np.square(grad_mean), 0.0))
    result = np.column_stack(
        [
            global_weight_norm / np.sqrt(total),
            global_grad_norm / np.sqrt(total),
            global_update_norm / np.sqrt(total),
            global_update_norm / np.maximum(global_weight_norm, np.finfo(np.float64).tiny),
            grad_mean,
            grad_std,
            (values[:, :, 6] * weight[None, :]).sum(axis=1),
            (values[:, :, 7] * weight[None, :]).sum(axis=1),
        ]
    )
    assert_finite("global trajectory", result)
    return result


def _stage_name(layer_name: str) -> str:
    for stage in ["layer1", "layer2", "layer3", "layer4"]:
        if layer_name.startswith(stage + "."):
            return stage
    if layer_name.startswith("conv1") or layer_name.startswith("0.") or layer_name.startswith("3."):
        return "stem"
    if layer_name.startswith("fc") or layer_name.startswith("7."):
        return "head"
    return "other"


def construct_trajectory(artifact: RunArtifact, kind: str = "global", metric_set: str = "internal") -> tuple[np.ndarray, list[str]]:
    base = _global_aggregate(artifact)
    names = list(GLOBAL_NAMES)
    if kind == "stage_update_ratios":
        stage_values: list[np.ndarray] = []
        stage_names: list[str] = []
        for stage in ["stem", "layer1", "layer2", "layer3", "layer4", "head", "other"]:
            positions = [i for i, name in enumerate(artifact.layer_names) if _stage_name(name) == stage]
            if not positions:
                continue
            weight_l2 = np.sqrt(np.square(artifact.layer_values[:, positions, 0]).sum(axis=1))
            update_l2 = np.sqrt(np.square(artifact.layer_values[:, positions, 2]).sum(axis=1))
            stage_values.append(update_l2 / np.maximum(weight_l2, np.finfo(np.float64).tiny))
            stage_names.append(f"{stage}_update_weight_ratio")
        base = np.column_stack([base, *stage_values]) if stage_values else base
        names.extend(stage_names)
    elif kind != "global":
        raise ValueError(f"Unknown trajectory kind: {kind}")
    if metric_set == "internal_plus_metrics":
        base = np.column_stack([base, artifact.ordinary_metrics])
        names.extend(artifact.metric_names)
    elif metric_set != "internal":
        raise ValueError(f"Unknown metric set: {metric_set}")
    assert_finite("constructed trajectory", base)
    return base.astype(np.float64), names


@dataclass
class PathStandardizer:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, trajectories: Iterable[np.ndarray]) -> "PathStandardizer":
        arrays = [np.asarray(item, dtype=np.float64) for item in trajectories]
        if not arrays:
            raise ValueError("Cannot fit path standardizer without training trajectories")
        stacked = np.concatenate(arrays, axis=0)
        assert_finite("normalizer training values", stacked)
        self.mean_ = stacked.mean(axis=0)
        scale = stacked.std(axis=0)
        self.scale_ = np.where(scale > 1e-12, scale, 1.0)
        return self

    def transform(self, trajectory: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("PathStandardizer has not been fitted")
        trajectory = np.asarray(trajectory, dtype=np.float64)
        if trajectory.shape[1] != len(self.mean_):
            raise ValueError("Trajectory dimensionality does not match normalizer")
        result = (trajectory - self.mean_) / self.scale_
        assert_finite("normalized trajectory", result)
        return result


def augment_time(trajectory: np.ndarray) -> np.ndarray:
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or len(trajectory) < 2:
        raise ValueError("Time augmentation requires a 2D trajectory with at least two points")
    tau = np.linspace(0.0, 1.0, len(trajectory), dtype=np.float64)
    return np.column_stack([tau, trajectory])


def temporal_permutation(length: int, order: str) -> np.ndarray:
    if order == "normal":
        return np.arange(length)
    if order == "reversed":
        return np.arange(length - 1, -1, -1)
    if order.startswith("shuffle_"):
        shuffle_id = int(order.split("_", 1)[1])
        return np.random.default_rng(80_000 + shuffle_id).permutation(length)
    raise ValueError(f"Unknown temporal order: {order}")


def reorder_trajectory(trajectory: np.ndarray, order: str) -> np.ndarray:
    return np.asarray(trajectory)[temporal_permutation(len(trajectory), order)]

