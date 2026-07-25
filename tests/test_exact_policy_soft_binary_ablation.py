from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.exact_policy_soft_binary_ablation import (
    auxiliary_soft_targets,
    economic_metrics,
    execution_ev_soft_target,
    expanding_month_folds,
    top_fraction_mask,
)


def _targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__path_auxiliary_atr_fraction__": [0.01, 0.005],
            "__peak_mfe_atr_12h__": [3.0, 0.5],
            "__time_to_first_meaningful_mfe_hours_12h__": [1.0, 12.0],
            "__mae_before_meaningful_mfe_atr_12h__": [0.1, 1.5],
            "__bars_to_confirmed_adverse_trough__": [1.0, 10.0],
            "__future_slope_atr_per_hour_12h__": [0.3, -0.3],
            "__meaningful_mfe_reached_12h__": [1.0, 0.0],
        }
    )


def test_auxiliary_soft_targets_have_economic_direction() -> None:
    targets = auxiliary_soft_targets(_targets())
    assert list(targets) == [
        "peak_mfe_12h_atr",
        "time_to_first_meaningful_mfe",
        "mae_before_meaningful_mfe_atr",
        "bars_before_price_stops_decreasing",
        "future_slope_atr_per_hour",
    ]
    assert ((targets >= 0.0) & (targets <= 1.0)).all().all()
    assert (targets.iloc[0] > targets.iloc[1]).all()


def test_execution_ev_soft_target_respects_threshold_and_temperature() -> None:
    values = execution_ev_soft_target(
        [-0.01, 0.005, 0.02], threshold=0.005, temperature=0.005
    )
    assert values[0] < 0.5
    assert values[1] == pytest.approx(0.5)
    assert values[2] > 0.5
    with pytest.raises(ValueError, match="temperature"):
        execution_ev_soft_target([0.0], threshold=0.0, temperature=0.0)


def test_expanding_folds_train_strictly_before_purged_validation() -> None:
    timestamps = pd.date_range("2026-05-01", "2026-07-31 23:00", freq="h", tz="UTC")
    folds = expanding_month_folds(timestamps)
    assert [fold["month"] for fold in folds] == ["2026-06", "2026-07"]
    for fold in folds:
        train = timestamps[np.asarray(fold["train_indices"])]
        valid = timestamps[np.asarray(fold["validation_indices"])]
        assert train.max() < fold["train_cutoff"]
        assert valid.min() == fold["validation_start"]


def test_top_fraction_is_timestamp_side_local_and_high_is_best() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-06-01"] * 20, utc=True),
            "side_name": ["long"] * 10 + ["short"] * 10,
        }
    )
    score = np.r_[np.arange(10), np.arange(10)]
    selected = top_fraction_mask(frame, score)
    assert selected.sum() == 2
    assert selected[9] and selected[19]


def test_economic_metrics_use_exact_return_column() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-06-01"] * 10, utc=True),
            "side_name": ["long"] * 10,
            "execution_net_ev_12h": np.arange(10) / 100.0,
        }
    )
    metrics = economic_metrics(frame, np.arange(10))
    assert metrics["global_top10_rows"] == 1
    assert metrics["global_top10_mean_net_return"] == pytest.approx(0.09)
    assert metrics["timestamp_side_top10_mean_net_return"] == pytest.approx(0.09)
