from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .signatures import SignatureAdapter
from .trajectory import augment_time, reorder_trajectory


@dataclass(frozen=True)
class RepresentationSpec:
    name: str
    family: str
    depth: int | None = None
    time_augmented: bool = False
    temporal_order: str = "normal"
    downsample_points: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def endpoint(path: np.ndarray) -> np.ndarray:
    return np.asarray(path[-1], dtype=np.float64)


def trajectory_summary(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=np.float64)
    tau = np.linspace(0.0, 1.0, len(path))
    centered = tau - tau.mean()
    slope = (centered[:, None] * path).sum(axis=0) / np.square(centered).sum()
    # Explicit trapezoidal integral keeps compatibility with NumPy 1.26.
    auc = ((path[:-1] + path[1:]) * 0.5).sum(axis=0) / (len(path) - 1)
    return np.concatenate(
        [path.mean(axis=0), path.std(axis=0), path.min(axis=0), path.max(axis=0), path[-1], slope, auc]
    )


def downsampled(path: np.ndarray, points: int) -> np.ndarray:
    path = np.asarray(path, dtype=np.float64)
    if points < 2:
        raise ValueError("Downsample baseline needs at least two points")
    positions = np.linspace(0, len(path) - 1, points)
    left = np.floor(positions).astype(int)
    right = np.ceil(positions).astype(int)
    alpha = positions - left
    sampled = (1.0 - alpha[:, None]) * path[left] + alpha[:, None] * path[right]
    return sampled.reshape(-1)


def extract_representation(path: np.ndarray, spec: RepresentationSpec, adapter: SignatureAdapter) -> np.ndarray:
    if spec.family == "endpoint":
        return endpoint(path)
    if spec.family == "summary":
        return trajectory_summary(path)
    if spec.family == "downsampled":
        return downsampled(path, int(spec.downsample_points or 0))
    ordered_input = augment_time(path) if spec.time_augmented else np.asarray(path)
    ordered_input = reorder_trajectory(ordered_input, spec.temporal_order)
    return adapter.transform(ordered_input, int(spec.depth or 0), spec.family)


def representation_specs(analysis: dict[str, Any], include_ablations: bool = True) -> list[RepresentationSpec]:
    specs = [
        RepresentationSpec("endpoint", "endpoint"),
        RepresentationSpec("summary", "summary"),
        RepresentationSpec(
            f"downsampled_{analysis['downsample_points']}",
            "downsampled",
            downsample_points=int(analysis["downsample_points"]),
        ),
    ]
    orders = analysis["temporal_orders"] if include_ablations else ["normal"]
    for family in analysis["signature_families"]:
        for depth in analysis["signature_depths"]:
            for time_augmented in analysis["time_augmentation"]:
                for order in orders:
                    time_label = "time" if time_augmented else "notime"
                    name = f"{family}_d{depth}_{time_label}_{order}"
                    specs.append(
                        RepresentationSpec(
                            name=name,
                            family=family,
                            depth=int(depth),
                            time_augmented=bool(time_augmented),
                            temporal_order=str(order),
                        )
                    )
    return specs
