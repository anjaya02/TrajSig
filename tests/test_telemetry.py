import numpy as np

from trajsig.telemetry import compute_update_norm, gradient_gini


def test_update_norm():
    assert np.isclose(compute_update_norm(np.array([3.0, 4.0]), np.zeros(2)), 5.0)


def test_gini_edge_cases_and_concentration():
    assert gradient_gini(np.zeros(10)) == 0.0
    assert np.isclose(gradient_gini(np.ones(10)), 0.0)
    assert gradient_gini(np.array([0.0, 0.0, 0.0, 4.0])) > 0.7
    assert np.isclose(gradient_gini(np.array([-1.0, 1.0])), 0.0)

