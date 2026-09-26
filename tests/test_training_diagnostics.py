from __future__ import annotations

import numpy as np

from scripts.evaluate_training_diagnostics import feature_sets, fit_and_score, prefix_summary


def test_prefix_features_never_read_future_epochs() -> None:
    telemetry = np.arange(2 * 30 * 8, dtype=float).reshape(2, 30, 8)
    ordinary = np.arange(2 * 30 * 3, dtype=float).reshape(2, 30, 3)
    original = feature_sets({"telemetry": telemetry, "ordinary": ordinary}, 5)
    telemetry[:, 5:, :] = 1e9
    ordinary[:, 5:, :] = -1e9
    changed = feature_sets({"telemetry": telemetry, "ordinary": ordinary}, 5)
    for method in original:
        np.testing.assert_array_equal(original[method], changed[method])


def test_prefix_summary_mean_and_last() -> None:
    path = np.array([[[1.0, 2.0], [3.0, 4.0], [50.0, 60.0]]])
    np.testing.assert_array_equal(prefix_summary(path, 2), [[2.0, 3.0, 3.0, 4.0]])


def test_envelope_uses_benign_prototypes_without_test_condition() -> None:
    x = np.array([[0.0], [0.1], [10.0], [10.1], [20.0], [20.1], [30.0], [31.0]])
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    conditions = np.array(["C0", "C0", "C1", "C1", "C2", "C2", "C3", "C4"])
    scores = fit_and_score("telemetry_envelope", x, y, conditions, np.array([[0.05], [10.05], [20.05], [30.5]]))
    assert np.isfinite(scores).all()
    assert scores[-1] > max(scores[:-1])

