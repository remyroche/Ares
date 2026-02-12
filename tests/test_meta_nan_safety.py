import numpy as np

from extreme_price_movements.meta_model import MetaModel


def test_signed_log_demeaned_handles_nan_without_propagation():
    m = MetaModel()
    y = np.array([0.1, np.nan, -0.2, 0.3], dtype=float)
    out = m._signed_log_demeaned(y)
    assert out.shape == y.shape
    assert np.all(np.isfinite(out))


def test_candidate_target_and_weight_sanitizes_nan_target_and_weights():
    m = MetaModel()
    y = np.array([0.1, np.nan, -0.2, 0.3], dtype=float)
    sw = np.array([1.0, np.nan, 0.5, 2.0], dtype=float)

    y_fit, sw_fit = m._candidate_target_and_weight(y, sw, "ridge_tailweighted_l1")

    assert np.all(np.isfinite(y_fit))
    assert sw_fit is not None
    assert np.all(np.isfinite(sw_fit))
