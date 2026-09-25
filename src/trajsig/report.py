from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .cache import ArtifactStore


def _top_table(frame: pd.DataFrame, task: str, metric_set: str, count: int = 12) -> str:
    selected = frame[(frame["task"] == task) & (frame["metric_set"] == metric_set)].copy()
    selected = selected.sort_values("balanced_accuracy_mean", ascending=False).head(count)
    columns = [
        "trajectory_kind",
        "representation",
        "feature_dim",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "macro_f1_mean",
    ]
    return selected[columns].to_markdown(index=False, floatfmt=".3f") if len(selected) else "No results."


def generate_figures(output: str | Path) -> list[Path]:
    store = ArtifactStore(output)
    aggregate = pd.read_csv(store.evaluation_dir / "aggregate_results.csv")
    selected = aggregate[
        (aggregate["task"] == "condition")
        & (aggregate["metric_set"] == "internal")
        & (aggregate["trajectory_kind"] == "global")
    ].copy()
    baselines = selected[selected["representation"].isin(["endpoint", "summary"]) | selected["representation"].str.startswith("downsampled")]
    signatures = selected[selected["representation"].str.endswith("_normal")]
    plot_data = pd.concat([baselines, signatures]).sort_values("balanced_accuracy_mean", ascending=True)
    if len(plot_data) > 25:
        plot_data = plot_data.tail(25)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(plot_data))))
    ax.barh(plot_data["representation"], plot_data["balanced_accuracy_mean"], color="#35618f")
    ax.errorbar(
        plot_data["balanced_accuracy_mean"],
        range(len(plot_data)),
        xerr=plot_data["balanced_accuracy_std"].fillna(0),
        fmt="none",
        ecolor="black",
        capsize=2,
    )
    ax.set_xlabel("Held-out-seed balanced accuracy (mean ± SD)")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = store.figure_dir / "condition_accuracy_internal.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths = [path]

    temporal = selected[
        selected["representation"].str.match(r"^(signature|logsignature)_d[1-4]_notime_")
    ].copy()
    temporal["family_depth"] = temporal["representation"].str.extract(
        r"^(signature_d[1-4]|logsignature_d[1-4])"
    )[0]
    temporal["order"] = temporal["representation"].str.extract(
        r"_(normal|reversed|shuffle_[0-9]+)$"
    )[0]
    temporal_pivot = temporal.pivot_table(
        index="family_depth", columns="order", values="balanced_accuracy_mean", aggfunc="first"
    )
    if len(temporal_pivot):
        ax = temporal_pivot.plot(kind="bar", figsize=(11, 5), ylim=(0, 1))
        ax.set_ylabel("Held-out-seed balanced accuracy")
        ax.set_xlabel("Representation")
        ax.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        temporal_path = store.figure_dir / "temporal_ablation_internal.png"
        plt.savefig(temporal_path, dpi=180)
        plt.close()
        paths.append(temporal_path)

    compression = pd.read_csv(store.evaluation_dir / "compression_results.csv")
    tradeoff = selected.merge(
        compression,
        on=["trajectory_kind", "metric_set", "representation"],
        how="inner",
        suffixes=("", "_storage"),
    )
    tradeoff = tradeoff[
        tradeoff["representation"].isin(["endpoint", "summary"])
        | tradeoff["representation"].str.startswith("downsampled")
        | tradeoff["representation"].str.endswith("_normal")
    ]
    if len(tradeoff):
        fig, ax = plt.subplots(figsize=(8, 5))
        is_signature = tradeoff["representation"].str.startswith(("signature", "logsignature"))
        ax.scatter(
            tradeoff.loc[~is_signature, "dense_float64_bytes_per_run"],
            tradeoff.loc[~is_signature, "balanced_accuracy_mean"],
            label="simple baseline",
            marker="s",
        )
        ax.scatter(
            tradeoff.loc[is_signature, "dense_float64_bytes_per_run"],
            tradeoff.loc[is_signature, "balanced_accuracy_mean"],
            label="signature/log-signature",
            alpha=0.8,
        )
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Dense float64 bytes per run (log scale)")
        ax.set_ylabel("Held-out-seed balanced accuracy")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        tradeoff_path = store.figure_dir / "accuracy_storage_tradeoff_internal.png"
        fig.savefig(tradeoff_path, dpi=180)
        plt.close(fig)
        paths.append(tradeoff_path)

    time_rows = selected[selected["representation"].str.endswith("_normal")].copy()
    time_rows["time_augmented"] = time_rows["representation"].str.contains("_time_")
    time_rows["family_depth"] = time_rows["representation"].str.extract(
        r"^(signature_d[1-4]|logsignature_d[1-4])"
    )[0]
    time_pivot = time_rows.pivot_table(
        index="family_depth", columns="time_augmented", values="balanced_accuracy_mean", aggfunc="first"
    )
    if False in time_pivot.columns and True in time_pivot.columns:
        delta = (time_pivot[True] - time_pivot[False]).sort_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#2d7f5e" if value >= 0 else "#aa4b4b" for value in delta]
        ax.bar(delta.index, delta.values, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Accuracy(time) − Accuracy(no time)")
        ax.set_xlabel("Representation")
        plt.xticks(rotation=35, ha="right")
        fig.tight_layout()
        time_path = store.figure_dir / "time_augmentation_delta_internal.png"
        fig.savefig(time_path, dpi=180)
        plt.close(fig)
        paths.append(time_path)
    return paths


def generate_report(config: dict[str, Any], output: str | Path) -> Path:
    store = ArtifactStore(output)
    summary = json.loads((store.evaluation_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
    audit = json.loads((store.root / "audit.json").read_text(encoding="utf-8"))
    aggregate = pd.read_csv(store.evaluation_dir / "aggregate_results.csv")
    compression = pd.read_csv(store.evaluation_dir / "compression_results.csv")
    phase = config["experiment"]["phase"]
    is_smoke = phase != "full"
    status = "PIPELINE VALIDATION ONLY — NOT SCIENTIFIC EVIDENCE" if is_smoke else "FULL ANALYSIS GENERATED — VERDICT REQUIRES SCIENTIFIC REVIEW"
    conditions = "\n".join(
        f"- {item['id']}: {item['description']}" for item in config["conditions"]
    )
    text = f"""# Neural Optimization Trajectory Signatures — Phase-0 Report

**Status: {status}**

## 1. Research question

Can truncated path signatures or log-signatures of low-dimensional internal optimization telemetry discriminate controlled training conditions on unseen random seeds, add value beyond simple baselines, and retain meaningful temporal information at a useful representation size?

## 2. Experimental setup

Dataset mode: `{config['data']['kind']}`. Architecture: `{config['training']['architecture']}`. Epochs: {config['training']['epochs']}. Runs completed: {summary['run_count']}. The full configuration is retained in `experiment_config.json`.

## 3. Training conditions

{conditions}

## 4. Seeds

Seed identities: {config['experiment']['seed_ids']}. Initialization, training, data-order, augmentation, corruption, and subset seeds use separate deterministic namespaces (10,000 through 60,000 plus seed identity). C3 and C4 share a corruption permutation so the 10% set is nested in the 20% set.

## 5. Telemetry construction

Once per epoch, each trainable tensor with at least two dimensions contributes weight L2 norm, last-minibatch gradient L2 norm, epoch displacement L2 norm, displacement/weight ratio, gradient mean and standard deviation, near-zero fraction, and a deterministic at-most-4096-element gradient Gini estimate. Full gradients are never saved. Ordinary loss/accuracy/LR metrics are cached separately in the same raw telemetry artifact and excluded from the primary trajectory.

## 6. Trajectory construction

The primary 8-channel trajectory contains parameter-count-aware global weight RMS, gradient RMS, update RMS, global update/weight ratio, global gradient mean and standard deviation, near-zero fraction, and Gini concentration. Raw layer-wise telemetry is retained. A configured stage-update-ratio construction is available as an aggregation sensitivity analysis. All trajectories have {config['training']['epochs']} points.

## 7. Signature/log-signature construction

The `esig` adapter uses backend `{summary['signature_backend']}`. Signature level zero is dropped. Every configured depth is reported. Time augmentation prepends normalized epoch time before any reversal or permutation, so temporal ablations permute time-value pairs together.

## 8. Baselines

Endpoint, seven-statistic per-channel summary (mean, SD, min, max, final, slope, trapezoidal area), and {config['analysis']['downsample_points']}-point linearly downsampled trajectories are evaluated with the same split protocol and classifier.

## 9. Leakage controls

Run filenames are opaque hashes. Condition, seed, affected indices, and descriptions are never representation features. For every fold, path-channel normalization is fitted only on training trajectories before representation extraction. The downstream feature scaler is also fitted only on training runs. Split-specific representations are cached with the training run IDs and normalizer parameters in the cache key. Audit status: **{audit['status']}**.

## 10. Held-out-seed protocol

Condition classification uses leave-one-seed-identity-out folds. Every condition for the held-out seed is absent from training. Chance balanced accuracy is {summary['condition_chance_balanced_accuracy']:.3f}.

## 11. Condition-classification results

### Internal telemetry only (top methods; smoke ordering is non-scientific)

{_top_table(aggregate, 'condition', 'internal')}

### Internal plus ordinary metrics

{_top_table(aggregate, 'condition', 'internal_plus_metrics')}

## 12. Seed-identity results

Seed identity is predicted with leave-one-condition-out folds, preventing a classifier from using the same condition in train and test. Chance balanced accuracy is {summary['seed_chance_balanced_accuracy']:.3f}.

{_top_table(aggregate, 'seed_identity', 'internal')}

## 13. Temporal ablations

Normal, reversed, and deterministic shuffled order variants are present in `aggregate_results.csv`. These must be compared within the same family, depth, time setting, and metric set.

## 14. Time-augmentation results

Every signature/log-signature is evaluated with and without an explicit [0,1] epoch coordinate. Paired rows are retained rather than selecting a best setting post hoc.

## 15. Representation geometry

`distance_results.csv` reports Euclidean within-condition and between-condition distances after training-fold-only feature standardization. Ratios above one indicate greater average between-condition separation. No t-SNE/UMAP evidence is used.

## 16. Compression/storage analysis

`compression_results.csv` records dimensionality, dense float64 bytes per run, extraction time, and raw compressed telemetry size. Representative rows:

{compression.head(15).to_markdown(index=False, floatfmt='.4g')}

## 17. Statistical uncertainty

Raw predictions and per-fold scores are preserved. Means, sample SDs, and approximate 95% normal intervals across held-out groups are reported. With only five full-experiment seed folds these intervals are descriptive, and overlapping uncertainty is treated as a tie.

## 18. Failure cases

This section cannot be scientifically interpreted from a smoke run. For the full run it must identify per-condition confusions from `predictions.csv`, especially C0/C3/C4 separation and C0/C2 schedule separation.

## 19. Reasons This Hypothesis May Be Wrong

- Endpoint or summary statistics may match signatures within uncertainty.
- Seed-identity prediction may be strong while held-out-seed condition prediction is weak.
- Shuffled signatures may retain essentially all apparent signal, implicating value distributions rather than path order.
- Time augmentation may provide no repeatable benefit.
- Higher depths may add dimensionality without reliable accuracy or compression gains.
- Ordinary training curves may explain most apparent discrimination.

These are precommitted failure modes, not conclusions from the smoke run.

## 20. Limitations

The full Phase-0 contains only five seed identities, one architecture, one dataset, 30 epochs, and simple linear downstream models. Gradient statistics are from the last minibatch of each epoch, while update norms cover the whole epoch. Gini uses a deterministic bounded sample for large tensors. Results cannot support security, ownership, proof-of-training, or novelty claims.

## 21. KILL / UNCERTAIN / GO verdict

**NO SCIENTIFIC VERDICT ISSUED.** {'FULL PHASE-0 COMPUTE PENDING.' if is_smoke else 'The full outputs must be reviewed against the predeclared decision criteria before a verdict is written.'}

## 22. Evidence supporting that verdict

No evidence is used for a scientific verdict at this stage. The smoke experiment validates software flow only.

## 23. Recommended next action

Run the full 30-run GPU experiment and return the raw artifact archive for Stage 5 analysis. No mechanism-design recommendation is made before that analysis.
"""
    path = store.report_dir / ("SMOKE_REPORT.md" if is_smoke else "PHASE0_ANALYSIS_DRAFT.md")
    path.write_text(text, encoding="utf-8")
    return path
