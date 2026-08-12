import numpy as np
import pandas as pd

from extreme_price_movements.base_error_tercile_meta import (
    expected_base_error_bps,
    fit_base_error_tercile_map,
    labels_from_base_error,
)


def _frame() -> pd.DataFrame:
    residual = np.array([-300, -180, -90, -20, 20, 60, 130, 250] * 2, dtype=float)
    return pd.DataFrame({
        "side_name": ["long"] * 8 + ["short"] * 8,
        "prequential_base_expected_net_bps": 10.0,
        "net_bps": 10.0 + residual,
    })


def test_tercile_mapping_is_side_local_and_ordered():
    frame = _frame()
    mapping = fit_base_error_tercile_map(frame, shrinkage_support=1.0)
    label = labels_from_base_error(frame, mapping)
    assert set(label) == {0, 1, 2}
    for side in ("long", "short"):
        mean = mapping.side_class_mean_bps[side]
        assert mean[0] < mean[1] < mean[2]


def test_probability_reconstruction_preserves_over_under_semantics():
    frame = _frame()
    mapping = fit_base_error_tercile_map(frame, shrinkage_support=1.0)
    p = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    correction = expected_base_error_bps(p, ["long", "long", "long"], mapping)
    assert correction[0] < correction[1] < correction[2]
