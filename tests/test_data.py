import numpy as np

from trajsig.data import deterministic_corruption, deterministic_subset


def test_deterministic_label_corruption_changes_exact_indices():
    labels = np.arange(100) % 10
    first, indices_first = deterministic_corruption(labels, 0.1, 55, 10)
    second, indices_second = deterministic_corruption(labels, 0.1, 55, 10)
    assert np.array_equal(first, second)
    assert np.array_equal(indices_first, indices_second)
    assert len(indices_first) == 10
    assert np.all(first[indices_first] != labels[indices_first])
    unaffected = np.setdiff1d(np.arange(100), indices_first)
    assert np.array_equal(first[unaffected], labels[unaffected])


def test_corruption_is_nested_between_ten_and_twenty_percent():
    labels = np.arange(100) % 10
    _, ten = deterministic_corruption(labels, 0.1, 8, 10)
    _, twenty = deterministic_corruption(labels, 0.2, 8, 10)
    assert set(ten).issubset(set(twenty))


def test_deterministic_subset():
    labels = np.repeat(np.arange(4), 20)
    kept_a, removed_a = deterministic_subset(labels, [0, 1], 0.5, 77)
    kept_b, removed_b = deterministic_subset(labels, [0, 1], 0.5, 77)
    assert np.array_equal(kept_a, kept_b)
    assert np.array_equal(removed_a, removed_b)
    assert len(kept_a) == 60
    assert len(removed_a) == 20
    assert set(labels[removed_a]) == {0, 1}

