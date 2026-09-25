from __future__ import annotations

import math
import os
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .cache import ArtifactStore
from .config import condition_config, derive_seeds, run_id, run_spec
from .data import build_datasets
from .model import build_model
from .telemetry import STAT_NAMES, collect_telemetry, snapshot_parameters
from .utils import atomic_json, seed_everything, software_environment, utc_now


METRIC_NAMES = ["train_loss", "train_accuracy", "test_accuracy", "learning_rate"]


def _atomic_torch_save(state: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _device() -> Any:
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _optimizer_and_scheduler(model: Any, cfg: dict[str, Any], condition: dict[str, Any]):
    import torch

    training = cfg["training"]
    common = {"lr": float(condition["learning_rate"]), "weight_decay": float(training["weight_decay"])}
    if condition["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), momentum=float(condition.get("momentum", 0.0)), **common)
    elif condition["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), **common)
    else:
        raise ValueError(f"Unknown optimizer: {condition['optimizer']}")
    if condition["schedule"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(training["epochs"]), eta_min=float(condition["learning_rate"]) * 0.01
        )
    elif condition["schedule"] == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(condition["schedule_milestones"]),
            gamma=float(condition["schedule_gamma"]),
        )
    else:
        raise ValueError(f"Unknown schedule: {condition['schedule']}")
    return optimizer, scheduler


def _worker_init(seed: int):
    def init(worker_id: int) -> None:
        import random
        import torch

        value = seed + worker_id
        random.seed(value)
        np.random.seed(value % (2**32 - 1))
        torch.manual_seed(value)

    return init


def _loader(
    dataset: Any,
    cfg: dict[str, Any],
    epoch: int,
    order_seed: int,
    train: bool,
    worker_seed: int | None = None,
):
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator().manual_seed(order_seed + epoch)
    batch_size = int(cfg["training"]["batch_size"] if train else cfg["training"]["eval_batch_size"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"] and torch.cuda.is_available()),
        generator=generator,
        worker_init_fn=_worker_init((worker_seed if worker_seed is not None else order_seed) + epoch * 1000),
        persistent_workers=False,
    )


def _train_epoch(
    model: Any,
    dataset: Any,
    optimizer: Any,
    scaler: Any,
    cfg: dict[str, Any],
    seeds: dict[str, int],
    epoch: int,
    device: Any,
    on_batch: Callable[[int, int], None] | None = None,
):
    import torch

    model.train()
    loader = _loader(
        dataset,
        cfg,
        epoch,
        seeds["data_order"],
        train=True,
        worker_seed=seeds["augmentation"],
    )
    iterator = iter(loader)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    steps = int(cfg["training"]["steps_per_epoch"])
    amp_enabled = bool(cfg["training"]["amp"] and device.type == "cuda")
    for step in range(1, steps + 1):
        try:
            inputs, labels = next(iterator)
        except StopIteration:
            # C5 has fewer unique examples. Cycling holds the number of optimizer
            # updates fixed, preventing trajectory identity via epoch update count.
            iterator = iter(loader)
            inputs, labels = next(iterator)
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()
        with autocast:
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, labels)
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        count = labels.numel()
        total_loss += float(loss.detach().item()) * count
        total_correct += int((logits.detach().argmax(1) == labels).sum().item())
        total_examples += count
        if on_batch is not None and (step == 1 or step % 50 == 0 or step == steps):
            on_batch(step, steps)
    return total_loss / total_examples, total_correct / total_examples


def _accuracy(model: Any, dataset: Any, cfg: dict[str, Any], device: Any) -> float:
    import torch

    model.eval()
    correct = 0
    count = 0
    loader = _loader(dataset, cfg, 0, 991_337, train=False)
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(inputs)
            correct += int((logits.argmax(1) == labels).sum().item())
            count += labels.numel()
    return correct / count


def train_one(
    config: dict[str, Any],
    output: str | Path,
    condition_id: str,
    seed_id: int,
    force: bool = False,
    run_position: int | None = None,
    total_runs: int | None = None,
    completed_before: int = 0,
) -> str:
    import torch

    store = ArtifactStore(output)
    identifier = run_id(config, condition_id, seed_id)
    epochs = int(config["training"]["epochs"])
    if store.is_complete(identifier, epochs) and not force:
        print(f"SKIP complete run {run_position or '?'} / {total_runs or '?'}: {condition_id} seed={seed_id}", flush=True)
        return identifier

    label = f"run {run_position or '?'} / {total_runs or '?'}: {condition_id} seed={seed_id}"
    print(f"SETUP {label}: loading data and building model", flush=True)
    condition = condition_config(config, condition_id)
    seeds = derive_seeds(seed_id)
    seed_everything(seeds["initialization"], bool(config["experiment"]["deterministic"]))
    bundle = build_datasets(config, condition, seeds)
    store.save_data_spec(identifier, bundle.data_spec, bundle.affected_indices, bundle.removed_indices)
    device = _device()
    model = build_model(config["training"]["architecture"], bundle.num_classes).to(device)
    # Initialization and training randomness are intentionally separated.
    seed_everything(seeds["training"], bool(config["experiment"]["deterministic"]))
    optimizer, scheduler = _optimizer_and_scheduler(model, config, condition)
    amp_enabled = bool(config["training"]["amp"] and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):  # PyTorch 2.2 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    checkpoint = store.checkpoint_path(identifier)
    start_epoch = 0
    telemetry_rows: list[np.ndarray] = []
    metric_rows: list[list[float]] = []
    layer_names: list[str] = []
    parameter_counts = np.array([], dtype=np.int64)
    started_at = utc_now()
    elapsed_before = 0.0
    if checkpoint.exists() and not force:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["next_epoch"])
        telemetry_rows = [np.asarray(row, dtype=np.float64) for row in state["telemetry_rows"]]
        metric_rows = [list(map(float, row)) for row in state["metric_rows"]]
        layer_names = list(state["layer_names"])
        parameter_counts = np.asarray(state["parameter_counts"], dtype=np.int64)
        started_at = state.get("started_at", started_at)
        elapsed_before = float(state.get("elapsed_seconds_accumulated", 0.0))
        print(f"RESUME {identifier} at epoch {start_epoch + 1}", flush=True)

    started_clock = time.perf_counter()
    total_epochs = (total_runs or 1) * epochs
    progress_path = store.root / "progress.json"
    atomic_json(
        progress_path,
        {
            "status": "training",
            "updated_at": utc_now(),
            "current_run": run_position,
            "total_runs": total_runs,
            "condition": condition_id,
            "seed_id": int(seed_id),
            "current_epoch": start_epoch + 1,
            "epochs_per_run": epochs,
            "completed_runs": completed_before,
            "completed_epochs": completed_before * epochs + start_epoch,
            "total_epochs": total_epochs,
            "estimated_hours_remaining": None,
        },
    )
    print(f"START {label}: epoch {start_epoch + 1}/{epochs} on {device}", flush=True)
    for epoch in range(start_epoch, epochs):
        # Makes epoch-boundary resume reproduce model-side stochasticity.
        seed_everything(seeds["training"] + epoch, bool(config["experiment"]["deterministic"]))
        epoch_start = snapshot_parameters(model, int(config["telemetry"]["parameter_min_ndim"]))
        learning_rate = float(optimizer.param_groups[0]["lr"])
        epoch_clock = time.perf_counter()

        def on_batch(step: int, steps: int) -> None:
            print(
                f"BATCH {label} epoch {epoch + 1}/{epochs} "
                f"step {step}/{steps} elapsed={time.perf_counter() - epoch_clock:.0f}s",
                flush=True,
            )

        train_loss, train_accuracy = _train_epoch(
            model, bundle.train, optimizer, scaler, config, seeds, epoch, device, on_batch=on_batch
        )
        test_accuracy = _accuracy(model, bundle.test, config, device)
        snap = collect_telemetry(
            model,
            epoch_start,
            minimum_ndim=int(config["telemetry"]["parameter_min_ndim"]),
            near_zero_threshold=float(config["telemetry"]["near_zero_threshold"]),
            gini_max_samples=int(config["telemetry"]["gini_max_samples"]),
        )
        if layer_names and snap.layer_names != layer_names:
            raise RuntimeError("Selected telemetry layers changed during training")
        layer_names = snap.layer_names
        parameter_counts = snap.parameter_counts
        telemetry_rows.append(snap.values)
        metric_rows.append([train_loss, train_accuracy, test_accuracy, learning_rate])
        scheduler.step()
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "next_epoch": epoch + 1,
            "telemetry_rows": telemetry_rows,
            "metric_rows": metric_rows,
            "layer_names": layer_names,
            "parameter_counts": parameter_counts,
            "started_at": started_at,
            "elapsed_seconds_accumulated": elapsed_before + (time.perf_counter() - started_clock),
        }
        _atomic_torch_save(state, checkpoint)
        completed_epochs = completed_before * epochs + epoch + 1
        elapsed_epoch = time.perf_counter() - epoch_clock
        estimated_hours = elapsed_epoch * max(total_epochs - completed_epochs, 0) / 3600
        atomic_json(
            progress_path,
            {
                "status": "training",
                "updated_at": utc_now(),
                "current_run": run_position,
                "total_runs": total_runs,
                "condition": condition_id,
                "seed_id": int(seed_id),
                "current_epoch": epoch + 1,
                "epochs_per_run": epochs,
                "completed_runs": completed_before,
                "completed_epochs": completed_epochs,
                "total_epochs": total_epochs,
                "last_epoch_seconds": round(elapsed_epoch, 1),
                "estimated_hours_remaining": round(estimated_hours, 1),
            },
        )
        print(
            f"PROGRESS {completed_epochs}/{total_epochs} epochs ({completed_epochs / total_epochs:.1%}); "
            f"{label} epoch {epoch + 1}/{epochs} saved; "
            f"loss={train_loss:.4f} train_acc={train_accuracy:.3f} test_acc={test_accuracy:.3f}; "
            f"rough ETA={estimated_hours:.1f}h",
            flush=True,
        )

    layer_array = np.stack(telemetry_rows)
    metric_array = np.asarray(metric_rows, dtype=np.float64)
    manifest = {
        "schema_version": 1,
        "run_id": identifier,
        "status": "complete",
        "completed_epochs": epochs,
        "condition": condition_id,
        "condition_description": condition["description"],
        "run_spec": run_spec(config, condition_id, seed_id),
        "data_spec_file": f"data_specs/{identifier}.json",
        "data_indices_file": f"data_specs/{identifier}.npz",
        "trajectory_shape": list(layer_array.shape),
        "ordinary_metrics_shape": list(metric_array.shape),
        "started_at": started_at,
        "completed_at": utc_now(),
        "training_elapsed_seconds": elapsed_before + (time.perf_counter() - started_clock),
        "environment": software_environment(),
    }
    store.save_run(
        identifier,
        manifest,
        layer_array,
        metric_array,
        layer_names,
        STAT_NAMES,
        METRIC_NAMES,
        parameter_counts,
    )
    if checkpoint.exists():
        checkpoint.unlink()
    return identifier


def train_all(config: dict[str, Any], output: str | Path, force: bool = False) -> list[str]:
    store = ArtifactStore(output)
    atomic_json(store.root / "experiment_config.json", {k: v for k, v in config.items() if not k.startswith("_")})
    identifiers: list[str] = []
    planned = [
        (condition["id"], int(seed_id))
        for condition in config["conditions"]
        for seed_id in config["experiment"]["seed_ids"]
    ]
    total_runs = len(planned)
    completed = sum(
        store.is_complete(run_id(config, condition_id, seed_id), int(config["training"]["epochs"]))
        for condition_id, seed_id in planned
    ) if not force else 0
    print(f"EXPERIMENT {completed}/{total_runs} runs complete; output={store.root.resolve()}", flush=True)
    for position, (condition_id, seed_id) in enumerate(planned, start=1):
        identifier = run_id(config, condition_id, seed_id)
        was_complete = store.is_complete(identifier, int(config["training"]["epochs"])) and not force
        identifiers.append(
            train_one(
                config,
                output,
                condition_id,
                seed_id,
                force=force,
                run_position=position,
                total_runs=total_runs,
                completed_before=completed,
            )
        )
        if not was_complete:
            completed += 1
    atomic_json(
        store.root / "progress.json",
        {
            "status": "training_complete",
            "updated_at": utc_now(),
            "completed_runs": total_runs,
            "total_runs": total_runs,
            "completed_epochs": total_runs * int(config["training"]["epochs"]),
            "total_epochs": total_runs * int(config["training"]["epochs"]),
            "estimated_hours_remaining": 0,
        },
    )
    print(f"TRAINING COMPLETE: {total_runs}/{total_runs} runs", flush=True)
    return identifiers
