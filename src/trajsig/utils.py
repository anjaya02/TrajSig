from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, **arrays)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def software_environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ["numpy", "pandas", "sklearn", "torch", "torchvision", "esig"]:
        try:
            module = __import__(name)
            version = getattr(module, "__version__", None)
            if name == "esig" and version is None:
                version = module.get_version()
            packages[name] = str(version)
        except Exception as exc:  # environment recording must not abort a run
            packages[name] = f"unavailable:{type(exc).__name__}"
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }
    try:
        import torch

        info["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        info["cuda_version"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["device"] = "unavailable"
    return info


def assert_finite(name: str, array: np.ndarray) -> None:
    bad = ~np.isfinite(array)
    if bad.any():
        positions = np.argwhere(bad)[:10].tolist()
        raise ValueError(f"{name} contains non-finite values at {positions}; refusing silent repair")

