from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_top40_reliability_costaware import (
    _choose_correction,
    _reliability_grade,
    _top40,
)


def _frame(n: int = 20) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({
        "candidate_id": [f"x{i}" for i in range(n)],
        "__ts__": ts,
        "side_name": ["long"] * n,
        "base_score": np.arange(n, dtype=float),
    })


def test_top40_is_query_local_and_uses_no_outcome_fields() -> None:
    frame = _frame()
    admitted = _top40(frame)
    # The first 4-hour query contributes seven rows and the second contributes
    # two rows after the per-query 40% ceiling.
    assert int(admitted.sum()) == 9
    assert admitted.iloc[-1]
    assert not admitted.iloc[0]


def test_reliability_classes_are_ordered_by_conversion_error() -> None:
    residual = np.array([-100.0, -50.0, 0.0, 50.0, 100.0])
    np.testing.assert_array_equal(_reliability_grade(residual), [0, 1, 1, 1, 2])


def test_correction_is_rejected_when_full_population_noop_is_better() -> None:
    n = 100
    base = np.linspace(-100.0, 100.0, n)
    frame = pd.DataFrame({
        "causal_base_map_bps": base,
        "net_bps": -base,
        "admitted_top40": np.ones(n, dtype=bool),
    })
    # A signed correction that worsens the global ordering must be disabled.
    params = _choose_correction(frame, base / 100.0)
    assert params["lambda"] == 0.0
    assert not params["beats_noop"]
