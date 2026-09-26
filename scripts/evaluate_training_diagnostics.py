"""CPU-only, seed-held-out Phase-0 evaluation for #5 Training Diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trajsig.cache import ArtifactStore
from trajsig.trajectory import GLOBAL_NAMES, construct_trajectory
from trajsig.utils import atomic_json, utc_now


PREFIXES = (3, 5, 10, 15)
METHODS = (
    "ordinary_logistic",
    "gradient_threshold",
    "update_ratio_threshold",
    "telemetry_logistic",
    "telemetry_envelope",
)
BENIGN = ("C0", "C1", "C2")
HARMFUL = ("C3", "C4")
CONDITIONS = (*BENIGN, *HARMFUL, "C5")


def prefix_summary(paths: np.ndarray, epochs: int) -> np.ndarray:
    """Only inspect the observed prefix; return per-channel mean and last value."""
    if paths.ndim != 3 or epochs < 2 or epochs > paths.shape[1]:
        raise ValueError("Invalid paths or prefix length")
    observed = paths[:, :epochs, :]
    return np.concatenate((observed.mean(axis=1), observed[:, -1, :]), axis=1)


def load_data(root: Path) -> dict[str, object]:
    audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or audit.get("complete_runs") != 30:
        raise ValueError("A passing 30-run audit is required")
    store = ArtifactStore(root)
    artifacts = [store.load_run(identifier) for identifier in store.complete_run_ids()]
    conditions = np.asarray([a.manifest["condition"] for a in artifacts], dtype=str)
    seeds = np.asarray([a.manifest["run_spec"]["seeds"]["seed_id"] for a in artifacts], dtype=int)
    pairs = {(str(c), int(s)) for c, s in zip(conditions, seeds)}
    if len(artifacts) != 30 or pairs != {(c, s) for c in CONDITIONS for s in range(5)}:
        raise ValueError("Expected exactly one run per condition and seed identity")
    names = artifacts[0].metric_names
    metric_indices = [names.index(name) for name in ("train_loss", "train_accuracy", "test_accuracy")]
    telemetry = np.stack([construct_trajectory(a, "global", "internal")[0] for a in artifacts])
    if telemetry.shape != (30, 30, len(GLOBAL_NAMES)):
        raise ValueError(f"Unexpected telemetry shape: {telemetry.shape}")
    telemetry = np.sign(telemetry) * np.log1p(np.abs(telemetry))
    ordinary = np.stack([a.ordinary_metrics[:, metric_indices] for a in artifacts])
    if ordinary.shape != (30, 30, 3) or not np.isfinite(ordinary).all():
        raise ValueError("Unexpected ordinary metric shape or non-finite values")
    return {
        "run_ids": np.asarray([a.run_id for a in artifacts]),
        "conditions": conditions,
        "seeds": seeds,
        "telemetry": telemetry,
        "ordinary": ordinary,
    }


def feature_sets(data: dict[str, object], epochs: int) -> dict[str, np.ndarray]:
    telemetry = np.asarray(data["telemetry"])
    ordinary = np.asarray(data["ordinary"])
    summary = prefix_summary(telemetry, epochs)
    return {
        "ordinary_logistic": prefix_summary(ordinary, epochs),
        "gradient_threshold": telemetry[:, :epochs, GLOBAL_NAMES.index("gradient_rms")].mean(axis=1)[:, None],
        "update_ratio_threshold": telemetry[:, :epochs, GLOBAL_NAMES.index("update_weight_ratio")].mean(axis=1)[:, None],
        "telemetry_logistic": summary,
        "telemetry_envelope": summary,
    }


def fit_and_score(
    method: str,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    condition_fit: np.ndarray,
    x_eval: np.ndarray,
) -> np.ndarray:
    if method.endswith("_logistic"):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=731),
        )
        model.fit(x_fit, y_fit)
        return model.predict_proba(x_eval)[:, 1]
    benign = x_fit[y_fit == 0]
    if method in ("gradient_threshold", "update_ratio_threshold"):
        center = float(np.median(benign[:, 0]))
        scale = max(float(np.std(benign[:, 0])), 1e-8)
        return np.abs(x_eval[:, 0] - center) / scale
    if method == "telemetry_envelope":
        benign_conditions = condition_fit[y_fit == 0]
        centers = np.stack([benign[benign_conditions == c].mean(axis=0) for c in BENIGN])
        residual = benign - np.stack([centers[BENIGN.index(c)] for c in benign_conditions])
        within_scale = np.sqrt(np.mean(residual**2, axis=0))
        global_scale = np.std(benign, axis=0)
        scale = np.maximum(np.maximum(within_scale, 0.25 * global_scale), 1e-6)
        standardized = (x_eval[:, None, :] - centers[None, :, :]) / scale[None, None, :]
        return np.sqrt(np.mean(standardized**2, axis=2)).min(axis=1)
    raise ValueError(f"Unknown method: {method}")


def evaluate(data: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conditions = np.asarray(data["conditions"])
    seeds = np.asarray(data["seeds"])
    run_ids = np.asarray(data["run_ids"])
    rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    for epochs in PREFIXES:
        features = feature_sets(data, epochs)
        for method in METHODS:
            x = features[method]
            for held_out in range(5):
                train = (seeds != held_out) & (conditions != "C5")
                test = seeds == held_out
                y_train = np.isin(conditions[train], HARMFUL).astype(int)
                inner_scores: list[float] = []
                for inner_seed in range(5):
                    if inner_seed == held_out:
                        continue
                    inner_fit = train & (seeds != inner_seed)
                    inner_benign = train & (seeds == inner_seed) & np.isin(conditions, BENIGN)
                    scores = fit_and_score(
                        method,
                        x[inner_fit],
                        np.isin(conditions[inner_fit], HARMFUL).astype(int),
                        conditions[inner_fit],
                        x[inner_benign],
                    )
                    inner_scores.extend(scores.tolist())
                if len(inner_scores) != 12:
                    raise AssertionError("Expected 12 seed-cross-fitted benign calibration scores")
                threshold = float(np.quantile(inner_scores, 0.9, method="higher"))
                exceedances = int(np.sum(np.asarray(inner_scores) > threshold))
                calibration_rows.append(
                    {
                        "prefix_epochs": epochs,
                        "method": method,
                        "held_out_seed": held_out,
                        "threshold": threshold,
                        "calibration_benign_count": len(inner_scores),
                        "calibration_exceedances": exceedances,
                    }
                )
                test_indices = np.flatnonzero(test)
                test_scores = fit_and_score(method, x[train], y_train, conditions[train], x[test])
                for index, score in zip(test_indices, test_scores):
                    condition = str(conditions[index])
                    rows.append(
                        {
                            "prefix_epochs": epochs,
                            "method": method,
                            "held_out_seed": held_out,
                            "run_id": str(run_ids[index]),
                            "condition": condition,
                            "binary_label": 1 if condition in HARMFUL else (0 if condition in BENIGN else -1),
                            "score": float(score),
                            "threshold": threshold,
                            "alarm": bool(score > threshold),
                        }
                    )
    predictions = pd.DataFrame(rows)
    calibration = pd.DataFrame(calibration_rows)
    metrics: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for (epochs, method), group in predictions.groupby(["prefix_epochs", "method"], sort=True):
        primary = group[group.binary_label >= 0]
        benign = group[group.binary_label == 0]
        harmful = group[group.binary_label == 1]
        benign_scores = np.sort(benign.score.to_numpy(dtype=float))[::-1]
        oracle_threshold = float(benign_scores[1])  # Strict > allows at most one of 15 benign runs.
        fold_auc: list[float] = []
        fold_ap: list[float] = []
        for seed, fold in group.groupby("held_out_seed", sort=True):
            fold_primary = fold[fold.binary_label >= 0]
            fold_auc.append(float(roc_auc_score(fold_primary.binary_label, fold_primary.score)))
            fold_ap.append(float(average_precision_score(fold_primary.binary_label, fold_primary.score)))
            fold_rows.append(
                {
                    "prefix_epochs": epochs,
                    "method": method,
                    "held_out_seed": seed,
                    "benign_false_alarms": int(fold[fold.binary_label == 0].alarm.sum()),
                    "C3_detected": int(fold[fold.condition == "C3"].alarm.sum()),
                    "C4_detected": int(fold[fold.condition == "C4"].alarm.sum()),
                    "C5_warned": int(fold[fold.condition == "C5"].alarm.sum()),
                    "AUROC": fold_auc[-1],
                    "AUPRC": fold_ap[-1],
                }
            )
        cal = calibration[(calibration.prefix_epochs == epochs) & (calibration.method == method)]
        row: dict[str, object] = {
            "prefix_epochs": epochs,
            "method": method,
            "benign_false_alarms": int(benign.alarm.sum()),
            "benign_total": len(benign),
            "false_alarm_rate": float(benign.alarm.mean()),
            "C3_detected": int(group[group.condition == "C3"].alarm.sum()),
            "C4_detected": int(group[group.condition == "C4"].alarm.sum()),
            "C5_warned": int(group[group.condition == "C5"].alarm.sum()),
            "AUROC_pooled": float(roc_auc_score(primary.binary_label, primary.score)),
            "AUPRC_pooled": float(average_precision_score(primary.binary_label, primary.score)),
            "AUROC_fold_mean": float(np.mean(fold_auc)),
            "AUPRC_fold_mean": float(np.mean(fold_ap)),
            "oracle_one_false_alarm_harmful_detected": int(np.sum(harmful.score > oracle_threshold)),
            "oracle_one_false_alarm_C3_detected": int(np.sum(group[group.condition == "C3"].score > oracle_threshold)),
            "oracle_one_false_alarm_C4_detected": int(np.sum(group[group.condition == "C4"].score > oracle_threshold)),
            "calibration_exceedances": int(cal.calibration_exceedances.sum()),
            "calibration_benign_count": int(cal.calibration_benign_count.sum()),
        }
        if method.endswith("_logistic"):
            row["Brier_pooled"] = float(brier_score_loss(primary.binary_label, primary.score))
        metrics.append(row)
    delays: list[dict[str, object]] = []
    for (method, run_id), group in predictions.groupby(["method", "run_id"], sort=True):
        group = group.sort_values("prefix_epochs")
        warnings = group[group.alarm]
        delays.append(
            {
                "method": method,
                "run_id": run_id,
                "condition": str(group.condition.iloc[0]),
                "seed_id": int(group.held_out_seed.iloc[0]),
                "first_warning_epoch": int(warnings.prefix_epochs.iloc[0]) if len(warnings) else np.nan,
                "censored_after_epoch_15": len(warnings) == 0,
            }
        )
    return predictions, calibration, pd.DataFrame(metrics), pd.DataFrame(fold_rows), pd.DataFrame(delays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Extracted audited 30-run artifact directory")
    parser.add_argument("--output", type=Path, required=True, help="Separate #5 output directory")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("#5 output must be separate from the #7 source")
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_data(args.input)
    predictions, calibration, metrics, folds, delays = evaluate(data)
    predictions.to_csv(args.output / "held_out_predictions.csv", index=False)
    calibration.to_csv(args.output / "seed_calibration.csv", index=False)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    folds.to_csv(args.output / "per_seed_metrics.csv", index=False)
    delays.to_csv(args.output / "first_warning.csv", index=False)
    cumulative_rows: list[dict[str, object]] = []
    for method, group in delays.groupby("method", sort=True):
        for epochs in PREFIXES:
            warned = group.first_warning_epoch <= epochs
            cumulative_rows.append(
                {
                    "method": method,
                    "by_epoch": epochs,
                    "benign_ever_warned": int((warned & group.condition.isin(BENIGN)).sum()),
                    "C3_ever_warned": int((warned & (group.condition == "C3")).sum()),
                    "C4_ever_warned": int((warned & (group.condition == "C4")).sum()),
                    "C5_ever_warned": int((warned & (group.condition == "C5")).sum()),
                }
            )
    pd.DataFrame(cumulative_rows).to_csv(args.output / "cumulative_warnings.csv", index=False)
    atomic_json(
        args.output / "analysis_manifest.json",
        {
            "created_at": utc_now(),
            "experiment": "#5 Training Diagnostics, new Phase-0",
            "source": str(args.input.resolve()),
            "prefix_epochs": list(PREFIXES),
            "methods": list(METHODS),
            "outer_folds": 5,
            "inner_calibration_benign_per_fold": 12,
            "nominal_false_alarm_target": 0.10,
            "C5_role": "secondary unlabeled generalization case",
            "signatures_used": False,
            "gpu_training_rerun": False,
        },
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
