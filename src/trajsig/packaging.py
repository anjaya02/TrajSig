from __future__ import annotations

import zipfile
from pathlib import Path


TRAINING_ARTIFACT_DIRS = ["telemetry", "manifests", "data_specs", "checkpoints"]


def package_training_artifacts(output: str | Path, destination: str | Path | None = None) -> Path:
    root = Path(output)
    destination = Path(destination) if destination else root / "phase0_training_artifacts.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in TRAINING_ARTIFACT_DIRS:
            directory = root / name
            if directory.exists():
                for path in sorted(directory.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(root))
        for name in ["experiment_config.json", "audit.json"]:
            path = root / name
            if path.exists():
                archive.write(path, path.relative_to(root))
    return destination

