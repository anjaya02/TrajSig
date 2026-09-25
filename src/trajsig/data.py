from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def deterministic_corruption(
    labels: np.ndarray, fraction: float, seed: int, num_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("corruption fraction must lie in [0, 1]")
    count = int(np.floor(len(labels) * fraction))
    rng = np.random.default_rng(seed)
    affected = np.sort(rng.permutation(len(labels))[:count])
    out = labels.copy()
    if count:
        # Offset by 1..K-1 guarantees every selected label actually changes.
        offsets = rng.integers(1, num_classes, size=count)
        out[affected] = (out[affected] + offsets) % num_classes
    return out, affected


def deterministic_subset(
    labels: np.ndarray,
    affected_classes: list[int],
    keep_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must lie in (0, 1]")
    rng = np.random.default_rng(seed)
    keep = np.ones(len(labels), dtype=bool)
    removed: list[np.ndarray] = []
    for class_id in affected_classes:
        indices = np.flatnonzero(labels == class_id)
        shuffled = rng.permutation(indices)
        n_keep = int(np.floor(len(indices) * keep_fraction))
        class_removed = np.sort(shuffled[n_keep:])
        keep[class_removed] = False
        removed.append(class_removed)
    kept = np.flatnonzero(keep)
    removed_all = np.sort(np.concatenate(removed)) if removed else np.array([], dtype=np.int64)
    return kept, removed_all


class RelabeledSubset:
    def __init__(self, dataset: Any, indices: np.ndarray, labels: np.ndarray):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        image, _ = self.dataset[int(self.indices[item])]
        return image, int(self.labels[int(self.indices[item])])


@dataclass
class DatasetBundle:
    train: Any
    test: Any
    num_classes: int
    data_spec: dict[str, Any]
    affected_indices: np.ndarray
    removed_indices: np.ndarray


def _synthetic_dataset(config: dict[str, Any]):
    import torch
    from torch.utils.data import TensorDataset

    data_cfg = config["data"]
    generator = torch.Generator().manual_seed(int(data_cfg["dataset_seed"]))
    n_train = int(data_cfg["train_size"])
    n_test = int(data_cfg["test_size"])
    size = int(data_cfg["image_size"])
    classes = int(data_cfg["num_classes"])
    # Fixed class prototypes plus noise create a learnable, deterministic smoke dataset.
    prototypes = torch.randn(classes, 3, size, size, generator=generator)
    train_y = torch.arange(n_train) % classes
    test_y = torch.arange(n_test) % classes
    train_x = prototypes[train_y] + 0.35 * torch.randn(n_train, 3, size, size, generator=generator)
    test_x = prototypes[test_y] + 0.35 * torch.randn(n_test, 3, size, size, generator=generator)
    return TensorDataset(train_x, train_y), TensorDataset(test_x, test_y), classes


def build_datasets(config: dict[str, Any], condition: dict[str, Any], seeds: dict[str, int]) -> DatasetBundle:
    kind = config["data"]["kind"]
    if kind == "synthetic":
        train_base, test, num_classes = _synthetic_dataset(config)
        labels = train_base.tensors[1].numpy().copy()
    elif kind == "cifar10":
        from torchvision import datasets, transforms

        root = config["data"]["root"]
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        train_base = datasets.CIFAR10(root, train=True, download=config["data"]["download"], transform=train_transform)
        test = datasets.CIFAR10(root, train=False, download=config["data"]["download"], transform=test_transform)
        labels = np.asarray(train_base.targets, dtype=np.int64)
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset kind: {kind}")

    modified, affected = deterministic_corruption(
        labels,
        float(condition.get("label_corruption", 0.0)),
        seeds["corruption"],
        num_classes,
    )
    indices = np.arange(len(labels), dtype=np.int64)
    removed = np.array([], dtype=np.int64)
    if condition.get("data_composition") == "c5_subset":
        indices, removed = deterministic_subset(
            labels,
            list(config["data"]["c5_classes"]),
            float(config["data"]["c5_keep_fraction"]),
            seeds["subset"],
        )
    train = RelabeledSubset(train_base, indices, modified)
    data_spec = {
        "kind": kind,
        "original_train_size": int(len(labels)),
        "effective_unique_train_size": int(len(indices)),
        "test_size": int(len(test)),
        "num_classes": int(num_classes),
        "label_corruption_fraction": float(condition.get("label_corruption", 0.0)),
        "corrupted_count": int(len(affected)),
        "removed_count": int(len(removed)),
        "fixed_steps_per_epoch": int(config["training"]["steps_per_epoch"]),
    }
    return DatasetBundle(train, test, num_classes, data_spec, affected, removed)

