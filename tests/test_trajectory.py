import numpy as np

from trajsig.cache import RunArtifact
from trajsig.trajectory import PathStandardizer, augment_time, construct_trajectory, reorder_trajectory


def fake_artifact() -> RunArtifact:
    values = np.zeros((3, 2, 8), dtype=float)
    values[:, :, 0] = [[3, 4], [4, 3], [5, 12]]
    values[:, :, 1] = 1
    values[:, :, 2] = 0.5
    values[:, :, 3] = 0.1
    values[:, :, 4] = 0.01
    values[:, :, 5] = 0.2
    values[:, :, 6] = 0.3
    values[:, :, 7] = 0.4
    return RunArtifact(
        run_id="opaque",
        manifest={},
        layer_values=values,
        ordinary_metrics=np.ones((3, 4)),
        layer_names=["layer1.0.weight", "fc.weight"],
        stat_names=[str(i) for i in range(8)],
        metric_names=["loss", "train_acc", "test_acc", "lr"],
        parameter_counts=np.array([10, 20]),
    )


def test_trajectory_shape_consistency():
    internal, names = construct_trajectory(fake_artifact(), "global", "internal")
    combined, combined_names = construct_trajectory(fake_artifact(), "global", "internal_plus_metrics")
    assert internal.shape == (3, 8)
    assert combined.shape == (3, 12)
    assert len(names) == 8
    assert len(combined_names) == 12


def test_temporal_reversal_and_deterministic_shuffle():
    path = np.arange(30).reshape(10, 3)
    assert np.array_equal(reorder_trajectory(path, "reversed"), path[::-1])
    assert np.array_equal(reorder_trajectory(path, "shuffle_2"), reorder_trajectory(path, "shuffle_2"))
    assert not np.array_equal(reorder_trajectory(path, "shuffle_1"), reorder_trajectory(path, "shuffle_2"))


def test_time_augmentation():
    path = np.ones((4, 2))
    timed = augment_time(path)
    assert timed.shape == (4, 3)
    assert np.allclose(timed[:, 0], [0, 1 / 3, 2 / 3, 1])
    assert np.array_equal(timed[:, 1:], path)


def test_normalizer_does_not_fit_on_test():
    normalizer = PathStandardizer().fit([np.array([[0.0], [2.0]])])
    before = (normalizer.mean_.copy(), normalizer.scale_.copy())
    transformed = normalizer.transform(np.array([[1000.0], [2000.0]]))
    assert np.array_equal(normalizer.mean_, before[0])
    assert np.array_equal(normalizer.scale_, before[1])
    assert transformed.min() > 900

