from __future__ import annotations

import numpy as np
import pytest

from extreme_price_movements.features_gmm_ae import (
    _gmm_distances,
    _gmm_predict_proba,
)


@pytest.mark.parametrize("covariance_type", ["diag", "tied", "full"])
def test_serialized_gmm_covariance_transform_is_finite(covariance_type: str) -> None:
    z = np.array([[0.0, 0.0], [1.0, -1.0], [3.0, 2.0]], dtype=np.float32)
    base = np.array([[1.0, 0.1], [0.1, 1.5]], dtype=np.float32)
    covariances = {
        "diag": np.array([[1.0, 1.5], [0.8, 1.2]], dtype=np.float32),
        "tied": base,
        "full": np.stack([base, 1.2 * base]),
    }[covariance_type]
    state = {
        "gmm_covariance_type": covariance_type,
        "gmm_weights": [0.55, 0.45],
        "gmm_means": [[0.0, 0.0], [2.0, 2.0]],
        "gmm_covariances": covariances,
    }

    probability = _gmm_predict_proba(z, state)
    distance, mahalanobis = _gmm_distances(z, state)

    assert probability.shape == (3, 2)
    assert np.allclose(probability.sum(axis=1), 1.0, atol=1e-5)
    assert np.isfinite(probability).all()
    assert np.isfinite(distance).all()
    assert np.isfinite(mahalanobis).all()
