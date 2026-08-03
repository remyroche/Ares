from __future__ import annotations

import numpy as np

from extreme_price_movements.meaningful_mfe_label_grid import (
    MeaningfulMFEGridSpec,
    build_meaningful_mfe_grid_labels,
)


def _labels(high: np.ndarray, low: np.ndarray, close: np.ndarray):
    return build_meaningful_mfe_grid_labels(
        entry_price=np.array([100.0] * len(high)),
        future_high=high,
        future_low=low,
        future_close=close,
        atr_fraction=np.array([0.01] * len(high)),
        side_sign=np.array([1.0] * len(high)),
        spec=MeaningfulMFEGridSpec(horizon_hours=4, upper_atr=1.5),
    )


def test_favorable_first_and_supporting_labels() -> None:
    labels = _labels(
        np.array([[101.0, 102.0, 102.2, 102.0]]),
        np.array([[99.8, 99.5, 99.2, 99.0]]),
        np.array([[100.5, 101.7, 101.9, 101.8]]),
    ).iloc[0]
    assert labels["favorable_first"] == 1.0
    assert labels["first_favorable_hour"] == 2.0
    assert labels["soft_label"] >= 0.75
    assert labels["favorable_barrier_net_of_cost"] == np.float32(0.005)
    assert np.isfinite(labels["time_to_80pct_mfe_hours"])
    assert labels["reaches_80pct_economic_barrier"] == 1.0
    assert labels["time_to_80pct_economic_barrier_hours"] == 2.0
    assert 0.0 <= labels["economic_barrier_time_quality"] <= 1.0
    assert np.isfinite(labels["future_close_slope_atr_per_hour"])
    assert abs(labels["future_close_slope_atr_per_hour_clip_10"]) <= 10.0


def test_same_bar_conflict_is_adverse() -> None:
    labels = _labels(
        np.array([[102.0, 102.1, 102.1, 102.1]]),
        np.array([[99.0, 99.0, 99.0, 99.0]]),
        np.array([[100.0, 100.0, 100.0, 100.0]]),
    ).iloc[0]
    assert labels["favorable_first"] == 0.0
    assert labels["adverse_first"] == 1.0
    assert labels["outcome"] == "adverse_first_or_conflict"


def test_short_side_is_normalized() -> None:
    labels = build_meaningful_mfe_grid_labels(
        entry_price=np.array([100.0]),
        future_high=np.array([[100.2, 100.3, 100.4, 100.5]]),
        future_low=np.array([[99.5, 98.0, 97.5, 97.0]]),
        future_close=np.array([[99.7, 98.5, 98.0, 97.5]]),
        atr_fraction=np.array([0.01]),
        side_sign=np.array([-1.0]),
        spec=MeaningfulMFEGridSpec(horizon_hours=4, upper_atr=2.0),
    ).iloc[0]
    assert labels["favorable_first"] == 1.0
    assert labels["first_favorable_hour"] == 2.0
