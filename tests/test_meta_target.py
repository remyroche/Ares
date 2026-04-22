import numpy as np

from extreme_price_movements.training import compute_meta_target


def test_compute_meta_target_uses_weighted_log_returns_globally():
    ret1 = np.array([0.01, -0.02, 0.03], dtype=float)
    ret2 = np.array([0.02, -0.01, 0.01], dtype=float)
    ret4 = np.array([0.00, 0.01, -0.02], dtype=float)

    y = compute_meta_target(ret1, ret2, ret4, vol_proxy=None, groups=np.array([1, 1, 2]))

    expected = np.array([ 0.01028626, -0.00539043,  0.00429887], dtype=np.float32)
    np.testing.assert_allclose(y, expected, atol=1e-7)


def test_compute_meta_target_groups_ignored_and_handles_edge_values():
    # Includes <-100% clip edge case and NaN; groups should be ignored.
    ret1 = np.array([0.0, -1.5, np.nan, 0.5], dtype=float)
    ret2 = np.array([0.1, -0.2, 0.3, 0.4], dtype=float)
    ret4 = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

    y_with_groups = compute_meta_target(ret1, ret2, ret4, vol_proxy=None, groups=np.array([1, 1, 2, 2]))
    y_no_groups = compute_meta_target(ret1, ret2, ret4, vol_proxy=None, groups=None)

    assert np.all(np.isfinite(y_with_groups))
    np.testing.assert_allclose(y_with_groups, y_no_groups, atol=1e-7)
