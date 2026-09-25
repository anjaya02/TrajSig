from __future__ import annotations

import json
import math
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .cache import ArtifactStore, RunArtifact
from .representations import RepresentationSpec, extract_representation, representation_specs
from .signatures import SignatureAdapter
from .trajectory import PathStandardizer, construct_trajectory
from .utils import assert_finite, atomic_json, atomic_npz, stable_hash, utc_now


def grouped_seed_splits(seed_ids: list[int]) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    seeds = np.asarray(seed_ids, dtype=np.int64)
    for held_out in sorted(np.unique(seeds)):
        test = np.flatnonzero(seeds == held_out)
        train = np.flatnonzero(seeds != held_out)
        if np.intersect1d(train, test).size:
            raise AssertionError("Grouped split overlap")
        yield train, test


def grouped_condition_splits(conditions: list[str]) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    values = np.asarray(conditions, dtype=str)
    for held_out in sorted(np.unique(values)):
        test = np.flatnonzero(values == held_out)
        train = np.flatnonzero(values != held_out)
        yield train, test


def _metadata(artifacts: list[RunArtifact]) -> tuple[list[str], list[int]]:
    conditions = [str(item.manifest["condition"]) for item in artifacts]
    seeds = [int(item.manifest["run_spec"]["seeds"]["seed_id"]) for item in artifacts]
    return conditions, seeds


def _compatible(artifacts: list[RunArtifact]) -> None:
    if not artifacts:
        raise ValueError("No complete run artifacts found")
    reference = (
        artifacts[0].layer_values.shape,
        artifacts[0].ordinary_metrics.shape,
        artifacts[0].layer_names,
        artifacts[0].stat_names,
        artifacts[0].metric_names,
    )
    for artifact in artifacts[1:]:
        current = (
            artifact.layer_values.shape,
            artifact.ordinary_metrics.shape,
            artifact.layer_names,
            artifact.stat_names,
            artifact.metric_names,
        )
        if current != reference:
            raise ValueError(f"Incompatible telemetry definition in {artifact.run_id}")


def _fold_representations(
    store: ArtifactStore,
    artifacts: list[RunArtifact],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    trajectory_kind: str,
    metric_set: str,
    spec: RepresentationSpec,
    adapter: SignatureAdapter,
    task: str,
    fold_label: str,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    paths = [construct_trajectory(item, trajectory_kind, metric_set)[0] for item in artifacts]
    normalizer = PathStandardizer().fit(paths[index] for index in train_indices)
    cache_key = stable_hash(
        {
            "schema": 2,
            "task": task,
            "fold": fold_label,
            "train_run_ids": [artifacts[index].run_id for index in train_indices],
            "test_run_ids": [artifacts[index].run_id for index in test_indices],
            "trajectory_kind": trajectory_kind,
            "metric_set": metric_set,
            "spec": spec.as_dict(),
            "normalizer_mean": normalizer.mean_.tolist(),
            "normalizer_scale": normalizer.scale_.tolist(),
        },
        24,
    )
    cache_path = store.representation_dir / f"{cache_key}.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            return cached["x_train"], cached["x_test"], float(cached["extraction_seconds"]), int(cached["feature_dim"])
    started = time.perf_counter()
    train_rows = [extract_representation(normalizer.transform(paths[index]), spec, adapter) for index in train_indices]
    test_rows = [extract_representation(normalizer.transform(paths[index]), spec, adapter) for index in test_indices]
    x_train = np.stack(train_rows)
    x_test = np.stack(test_rows)
    elapsed = time.perf_counter() - started
    assert_finite("training representations", x_train)
    assert_finite("test representations", x_test)
    atomic_npz(
        cache_path,
        x_train=x_train,
        x_test=x_test,
        train_run_ids=np.asarray([artifacts[index].run_id for index in train_indices]),
        test_run_ids=np.asarray([artifacts[index].run_id for index in test_indices]),
        normalizer_mean=normalizer.mean_,
        normalizer_scale=normalizer.scale_,
        extraction_seconds=np.asarray(elapsed),
        feature_dim=np.asarray(x_train.shape[1]),
    )
    return x_train, x_test, elapsed, x_train.shape[1]


def _classifier(config: dict[str, Any]) -> Pipeline:
    analysis = config["analysis"]
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(analysis["classifier_c"]),
                    max_iter=int(analysis["max_iter"]),
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=int(analysis["random_seed"]),
                ),
            ),
        ]
    )


def _distance_statistics(x: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    scaler = StandardScaler().fit(x)
    normalized = scaler.transform(x)
    within: list[float] = []
    between: list[float] = []
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            distance = float(np.linalg.norm(normalized[i] - normalized[j]))
            (within if labels[i] == labels[j] else between).append(distance)
    within_mean = float(np.mean(within)) if within else math.nan
    between_mean = float(np.mean(between)) if between else math.nan
    return {
        "within_mean": within_mean,
        "between_mean": between_mean,
        "between_within_ratio": between_mean / within_mean if within_mean > 0 else math.nan,
        "within_pairs": len(within),
        "between_pairs": len(between),
    }


def _ci95(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return math.nan, math.nan
    half = 1.96 * float(values.std(ddof=1)) / math.sqrt(len(values))
    return float(values.mean() - half), float(values.mean() + half)


def _aggregate(folds: pd.DataFrame) -> pd.DataFrame:
    keys = ["task", "trajectory_kind", "metric_set", "representation"]
    rows: list[dict[str, Any]] = []
    for group_values, group in folds.groupby(keys, sort=True):
        row = dict(zip(keys, group_values))
        for metric in ["balanced_accuracy", "macro_f1"]:
            values = group[metric].to_numpy(dtype=float)
            low, high = _ci95(values)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else math.nan
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        row["folds"] = len(group)
        row["feature_dim"] = int(group["feature_dim"].iloc[0])
        row["dense_float64_bytes_per_run"] = 8 * row["feature_dim"]
        row["extraction_seconds_mean"] = float(group["extraction_seconds"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_all(config: dict[str, Any], output: str | Path) -> dict[str, Path]:
    store = ArtifactStore(output)
    artifacts = [store.load_run(identifier) for identifier in store.complete_run_ids()]
    _compatible(artifacts)
    expected = len(config["conditions"]) * len(config["experiment"]["seed_ids"])
    if len(artifacts) != expected:
        raise ValueError(f"Expected {expected} completed runs, found {len(artifacts)}")
    conditions, seeds = _metadata(artifacts)
    condition_levels = sorted(set(conditions))
    seed_levels = sorted(set(seeds))
    adapter = SignatureAdapter()
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    compression_rows: dict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(list)

    trajectory_kinds = list(config["analysis"]["trajectory_kinds"])
    for trajectory_kind in trajectory_kinds:
        include_ablations = trajectory_kind == "global"
        specs = representation_specs(config["analysis"], include_ablations=include_ablations)
        for metric_set in config["analysis"]["metric_sets"]:
            for spec in specs:
                for task in ["condition", "seed_identity"]:
                    if task == "condition":
                        split_iterator = grouped_seed_splits(seeds)
                        targets = np.asarray(conditions)
                        levels: list[Any] = condition_levels
                        fold_names = [f"seed_{item}" for item in seed_levels]
                    else:
                        split_iterator = grouped_condition_splits(conditions)
                        targets = np.asarray(seeds)
                        levels = seed_levels
                        fold_names = [f"condition_{item}" for item in condition_levels]
                    for fold_index, (train_indices, test_indices) in enumerate(split_iterator):
                        fold_label = fold_names[fold_index]
                        x_train, x_test, extraction_seconds, feature_dim = _fold_representations(
                            store,
                            artifacts,
                            train_indices,
                            test_indices,
                            trajectory_kind,
                            metric_set,
                            spec,
                            adapter,
                            task,
                            fold_label,
                        )
                        max_features = int(config["analysis"].get("max_representation_features", 200_000))
                        if feature_dim > max_features:
                            raise ValueError(
                                f"{spec.name} has {feature_dim} features, exceeding configured limit {max_features}"
                            )
                        model = _classifier(config)
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=ConvergenceWarning)
                            model.fit(x_train, targets[train_indices])
                        predicted = model.predict(x_test)
                        balanced = float(balanced_accuracy_score(targets[test_indices], predicted))
                        macro = float(f1_score(targets[test_indices], predicted, labels=levels, average="macro", zero_division=0))
                        matrix = confusion_matrix(targets[test_indices], predicted, labels=levels).tolist()
                        fold_rows.append(
                            {
                                "task": task,
                                "trajectory_kind": trajectory_kind,
                                "metric_set": metric_set,
                                "representation": spec.name,
                                "held_out_group": fold_label,
                                "balanced_accuracy": balanced,
                                "macro_f1": macro,
                                "feature_dim": feature_dim,
                                "extraction_seconds": extraction_seconds,
                                "confusion_matrix_json": json.dumps(matrix),
                            }
                        )
                        compression_rows[(trajectory_kind, metric_set, spec.name)].append((feature_dim, extraction_seconds))
                        for class_index, class_label in enumerate(levels):
                            support = int(np.sum(targets[test_indices] == class_label))
                            correct = int(matrix[class_index][class_index])
                            per_class_rows.append(
                                {
                                    "task": task,
                                    "trajectory_kind": trajectory_kind,
                                    "metric_set": metric_set,
                                    "representation": spec.name,
                                    "held_out_group": fold_label,
                                    "class_label": str(class_label),
                                    "support": support,
                                    "recall": correct / support if support else math.nan,
                                }
                            )
                        for position, artifact_index in enumerate(test_indices):
                            prediction_rows.append(
                                {
                                    "task": task,
                                    "trajectory_kind": trajectory_kind,
                                    "metric_set": metric_set,
                                    "representation": spec.name,
                                    "held_out_group": fold_label,
                                    "run_id": artifacts[artifact_index].run_id,
                                    "true_label": str(targets[artifact_index]),
                                    "predicted_label": str(predicted[position]),
                                }
                            )
                        if task == "condition":
                            distance = _distance_statistics(x_train, targets[train_indices])
                            distance_rows.append(
                                {
                                    "trajectory_kind": trajectory_kind,
                                    "metric_set": metric_set,
                                    "representation": spec.name,
                                    "held_out_group": fold_label,
                                    **distance,
                                }
                            )

    folds = pd.DataFrame(fold_rows)
    aggregate = _aggregate(folds)
    predictions = pd.DataFrame(prediction_rows)
    per_class_folds = pd.DataFrame(per_class_rows)
    per_class = (
        per_class_folds.groupby(
            ["task", "trajectory_kind", "metric_set", "representation", "class_label"], sort=True
        )["recall"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "recall_mean", "std": "recall_std", "count": "folds"})
    )
    distances = pd.DataFrame(distance_rows)
    compression: list[dict[str, Any]] = []
    for (trajectory_kind, metric_set, representation), measurements in sorted(compression_rows.items()):
        dims = [item[0] for item in measurements]
        timings = [item[1] for item in measurements]
        compression.append(
            {
                "trajectory_kind": trajectory_kind,
                "metric_set": metric_set,
                "representation": representation,
                "feature_dim": int(dims[0]),
                "dense_float64_bytes_per_run": int(dims[0] * 8),
                "actual_compressed_bytes_per_run": math.nan,
                "extraction_seconds_mean_per_fold": float(np.mean(timings)),
                "extraction_seconds_mean_per_run": float(np.mean(timings)) / len(artifacts),
            }
        )
    raw_sizes = [store.telemetry_path(item.run_id).stat().st_size for item in artifacts]
    raw_shape = list(artifacts[0].layer_values.shape)
    compression.append(
        {
            "trajectory_kind": "raw_layer_telemetry",
            "metric_set": "internal_plus_metrics",
            "representation": "compressed_npz_on_disk",
            "feature_dim": int(np.prod(raw_shape) + np.prod(artifacts[0].ordinary_metrics.shape)),
            "dense_float64_bytes_per_run": int(
                (np.prod(raw_shape) + np.prod(artifacts[0].ordinary_metrics.shape)) * 8
            ),
            "actual_compressed_bytes_per_run": int(np.mean(raw_sizes)),
            "extraction_seconds_mean_per_fold": 0.0,
            "extraction_seconds_mean_per_run": 0.0,
        }
    )
    compression_frame = pd.DataFrame(compression)
    training_costs = pd.DataFrame(
        [
            {
                "run_id": item.run_id,
                "condition": item.manifest["condition"],
                "seed_id": item.manifest["run_spec"]["seeds"]["seed_id"],
                "training_elapsed_seconds": item.manifest.get("training_elapsed_seconds", math.nan),
                "device": item.manifest["environment"].get("device"),
                "gpu": item.manifest["environment"].get("gpu"),
            }
            for item in artifacts
        ]
    )

    paths = {
        "folds": store.evaluation_dir / "per_fold_results.csv",
        "aggregate": store.evaluation_dir / "aggregate_results.csv",
        "predictions": store.evaluation_dir / "predictions.csv",
        "per_class_folds": store.evaluation_dir / "per_class_per_fold.csv",
        "per_class": store.evaluation_dir / "per_class_results.csv",
        "distances": store.evaluation_dir / "distance_results.csv",
        "compression": store.evaluation_dir / "compression_results.csv",
        "training_costs": store.evaluation_dir / "training_costs.csv",
        "summary": store.evaluation_dir / "evaluation_summary.json",
    }
    folds.to_csv(paths["folds"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    per_class_folds.to_csv(paths["per_class_folds"], index=False)
    per_class.to_csv(paths["per_class"], index=False)
    distances.to_csv(paths["distances"], index=False)
    compression_frame.to_csv(paths["compression"], index=False)
    training_costs.to_csv(paths["training_costs"], index=False)
    atomic_json(
        paths["summary"],
        {
            "status": "complete",
            "completed_at": utc_now(),
            "run_count": len(artifacts),
            "condition_levels": condition_levels,
            "seed_levels": seed_levels,
            "condition_chance_balanced_accuracy": 1.0 / len(condition_levels),
            "seed_chance_balanced_accuracy": 1.0 / len(seed_levels),
            "signature_backend": adapter.backend,
            "normalization": "Path features and classifier features fitted on training runs only in every fold",
            "files": {key: str(path.name) for key, path in paths.items() if key != "summary"},
        },
    )
    return paths
