from __future__ import annotations

import numpy as np


class SignatureAdapter:
    """Narrow adapter around esig so the backend can be replaced independently."""

    def __init__(self, backend: str | None = None):
        try:
            import esig
        except ImportError as exc:
            raise RuntimeError("esig is required for signature extraction; install project dependencies") from exc
        self._esig = esig
        if backend is not None:
            esig.set_backend(backend)
        self.backend = str(esig.get_backend())

    def dimension(self, channels: int, depth: int, family: str) -> int:
        if family == "signature":
            # esig's public dimension helper inconsistently excludes level zero
            # at depth 1 and includes it at larger depths. Project dimensions
            # always exclude the constant level-zero scalar.
            return channels if depth == 1 else int(self._esig.sigdim(channels, depth)) - 1
        if family == "logsignature":
            # The same esig helper has a depth-1 corner case.
            return channels if depth == 1 else int(self._esig.logsigdim(channels, depth))
        raise ValueError(f"Unknown signature family: {family}")

    def transform(self, trajectory: np.ndarray, depth: int, family: str) -> np.ndarray:
        path = np.ascontiguousarray(trajectory, dtype=np.float64)
        if path.ndim != 2 or len(path) < 2:
            raise ValueError("Signature input must have shape (time>=2, channels)")
        if not np.isfinite(path).all():
            raise ValueError("Signature input contains non-finite values")
        if family == "signature":
            result = np.asarray(self._esig.stream2sig(path, int(depth)), dtype=np.float64)[1:]
        elif family == "logsignature":
            result = np.asarray(self._esig.stream2logsig(path, int(depth)), dtype=np.float64)
        else:
            raise ValueError(f"Unknown signature family: {family}")
        expected = self.dimension(path.shape[1], int(depth), family)
        if result.shape != (expected,):
            raise RuntimeError(f"Signature backend returned {result.shape}, expected {(expected,)}")
        if not np.isfinite(result).all():
            raise ValueError("Signature backend produced non-finite values")
        return result
