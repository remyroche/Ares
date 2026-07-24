from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_residual_hard_period_cnn import (
    SmallCausalCNN,
    SmallCausalTCN,
    _sequence_bundle,
    _subsample_indices,
)


def test_sequence_target_uses_only_future_event_onsets() -> None:
    days = pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC")
    local = pd.DataFrame(
        {
            "day": days,
            "state_a": np.arange(8, dtype=np.float32),
            "event_start": [False, False, False, True, False, False, False, False],
        }
    )
    bundle = _sequence_bundle(local, ["state_a"], window=4, horizon=2)
    # The model can only warn on the two days before the onset.  It must not
    # label the onset itself positive for an early-warning target.
    assert bundle.y.tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    assert bundle.x.shape == (8, 1, 4)
    assert bundle.x[3, 0, -1] == 3.0


def test_negative_sampling_is_bounded_and_retains_every_positive() -> None:
    y = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=np.int8)
    selected = _subsample_indices(y, maximum_negative_ratio=2, seed=1)
    assert set(np.flatnonzero(y)).issubset(set(selected))
    assert len(selected) <= 2 * int(y.sum()) + int(y.sum())


def test_cnn_has_causal_batch_shape() -> None:
    model = SmallCausalCNN(feature_count=3)
    output = model(__import__("torch").zeros((5, 3, 16)))
    assert tuple(output.shape) == (5,)


def test_tcn_has_causal_batch_shape() -> None:
    model = SmallCausalTCN(feature_count=3)
    output = model(__import__("torch").zeros((5, 3, 32)))
    assert tuple(output.shape) == (5,)
