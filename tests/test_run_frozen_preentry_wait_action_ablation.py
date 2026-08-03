from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_frozen_preentry_wait_action_ablation import (
    route_wait,
    wait_slice,
    weighted_mean,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pred_direct_delta": [-0.01, 0.02],
            "pred_decomposed_delta": [0.03, -0.01],
            "pred_soft_score": [0.4, 0.8],
        }
    )


def test_predeclared_wait_routes_do_not_change_row_count() -> None:
    prediction = _predictions()
    delta = np.array([0.01, -0.02])
    assert route_wait("enter_now", prediction, delta).tolist() == [False, False]
    assert route_wait("always_wait10", prediction, delta).tolist() == [True, True]
    assert route_wait("oracle_wait10", prediction, delta).tolist() == [True, False]
    assert route_wait("direct", prediction, delta).tolist() == [False, True]
    assert route_wait("decomposed", prediction, delta).tolist() == [True, False]
    assert route_wait("soft", prediction, delta).tolist() == [False, True]


def test_fractional_global_weighting_is_exact() -> None:
    assert weighted_mean([0.01, -0.02], [1.0, 0.5]) == 0.0
    assert np.isnan(weighted_mean([0.01], [0.0]))


def test_wait_slice_excludes_every_pre_entry_bar_and_starts_at_index_10() -> None:
    timestamp = np.arange(720, dtype=np.int64)[None, :] * 60_000_000_000
    arrays = tuple(
        (np.arange(720, dtype=np.float32) + offset)[None, :]
        for offset in (0.0, 1.0, -1.0, 0.5)
    )
    sliced_timestamp, sliced_arrays = wait_slice(
        timestamp, arrays, wait_minutes=10
    )
    assert sliced_timestamp.shape == (1, 710)
    assert sliced_timestamp[0, 0] == timestamp[0, 10]
    assert all(array.shape == (1, 710) for array in sliced_arrays)
    assert sliced_arrays[0][0, 0] == arrays[0][0, 10]
