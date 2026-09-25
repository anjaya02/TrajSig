from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


STAT_NAMES = [
    "weight_l2",
    "gradient_l2",
    "update_l2",
    "update_weight_ratio",
    "gradient_mean",
    "gradient_std",
    "near_zero_fraction",
    "gradient_gini",
]


def gradient_gini(values: np.ndarray) -> float:
    """Gini concentration of absolute values; zero vector maps to zero."""
    x = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    if x.size == 0:
        return 0.0
    if not np.isfinite(x).all():
        raise ValueError("Cannot calculate Gini on non-finite gradients")
    total = float(x.sum())
    if total <= np.finfo(np.float64).tiny:
        return 0.0
    x.sort()
    n = x.size
    index = np.arange(1, n + 1, dtype=np.float64)
    value = (2.0 * np.dot(index, x) / (n * total)) - (n + 1.0) / n
    return float(np.clip(value, 0.0, 1.0))


def selected_parameters(model: Any, minimum_ndim: int = 2) -> list[tuple[str, Any]]:
    return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.ndim >= minimum_ndim]


def snapshot_parameters(model: Any, minimum_ndim: int = 2) -> dict[str, Any]:
    return {name: parameter.detach().clone() for name, parameter in selected_parameters(model, minimum_ndim)}


def _sample_for_gini(flat: Any, maximum: int) -> np.ndarray:
    array = flat.detach().float().cpu().numpy().reshape(-1)
    if array.size <= maximum:
        return array
    # Evenly spaced deterministic sample: no telemetry-specific RNG state.
    positions = np.linspace(0, array.size - 1, maximum, dtype=np.int64)
    return array[positions]


@dataclass
class TelemetrySnapshot:
    values: np.ndarray
    layer_names: list[str]
    parameter_counts: np.ndarray


def collect_telemetry(
    model: Any,
    epoch_start: dict[str, Any],
    minimum_ndim: int = 2,
    near_zero_threshold: float = 1e-12,
    gini_max_samples: int = 4096,
) -> TelemetrySnapshot:
    import torch

    rows: list[list[float]] = []
    names: list[str] = []
    counts: list[int] = []
    for name, parameter in selected_parameters(model, minimum_ndim):
        if parameter.grad is None:
            raise RuntimeError(f"Missing gradient for selected parameter {name}")
        weight = parameter.detach().float()
        gradient = parameter.grad.detach().float()
        update = weight - epoch_start[name].float()
        weight_l2 = float(torch.linalg.vector_norm(weight).item())
        gradient_l2 = float(torch.linalg.vector_norm(gradient).item())
        update_l2 = float(torch.linalg.vector_norm(update).item())
        denominator = max(weight_l2, np.finfo(np.float32).tiny)
        rows.append(
            [
                weight_l2,
                gradient_l2,
                update_l2,
                update_l2 / denominator,
                float(gradient.mean().item()),
                float(gradient.std(unbiased=False).item()),
                float((gradient.abs() <= near_zero_threshold).float().mean().item()),
                gradient_gini(_sample_for_gini(gradient, gini_max_samples)),
            ]
        )
        names.append(name)
        counts.append(parameter.numel())
    return TelemetrySnapshot(
        values=np.asarray(rows, dtype=np.float64),
        layer_names=names,
        parameter_counts=np.asarray(counts, dtype=np.int64),
    )


def compute_update_norm(current: np.ndarray, previous: np.ndarray) -> float:
    current = np.asarray(current, dtype=np.float64)
    previous = np.asarray(previous, dtype=np.float64)
    if current.shape != previous.shape:
        raise ValueError("Update norm inputs must have matching shapes")
    return float(np.linalg.norm(current - previous))

