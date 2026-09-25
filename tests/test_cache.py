import numpy as np

from trajsig.cache import ArtifactStore


def test_cache_roundtrip(tmp_path):
    store = ArtifactStore(tmp_path)
    store.save_run(
        "abcdef",
        {"status": "complete", "completed_epochs": 2},
        np.ones((2, 1, 8)),
        np.ones((2, 4)),
        ["layer.weight"],
        [f"s{i}" for i in range(8)],
        [f"m{i}" for i in range(4)],
        np.array([12]),
    )
    loaded = store.load_run("abcdef")
    assert loaded.layer_values.shape == (2, 1, 8)
    assert loaded.layer_names == ["layer.weight"]
    assert store.is_complete("abcdef", 2)

