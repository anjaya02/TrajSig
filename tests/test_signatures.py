import numpy as np
import pytest

from trajsig.signatures import SignatureAdapter


@pytest.fixture(scope="module")
def adapter():
    return SignatureAdapter()


def test_signature_adapter_dimensions(adapter):
    path = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]])
    for depth in [1, 2, 3]:
        signature = adapter.transform(path, depth, "signature")
        logsignature = adapter.transform(path, depth, "logsignature")
        assert len(signature) == adapter.dimension(2, depth, "signature")
        assert len(logsignature) == adapter.dimension(2, depth, "logsignature")
        assert np.isfinite(signature).all()
        assert np.isfinite(logsignature).all()


def test_signature_depth_one_is_path_increment(adapter):
    path = np.array([[1.0, 4.0], [3.0, 1.0], [6.0, 2.0]])
    assert np.allclose(adapter.transform(path, 1, "signature"), path[-1] - path[0])

