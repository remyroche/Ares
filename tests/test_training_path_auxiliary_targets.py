import numpy as np
import pandas as pd

from extreme_price_movements.training import _build_hourly_path_auxiliary_targets


def test_training_path_targets_are_policy_independent_and_require_contiguous_hours():
    index = pd.date_range("2026-01-01", periods=14, freq="h", tz="UTC")
    panel = {
        "open": pd.DataFrame({"X": np.full(14, 100.0)}, index=index),
        "high": pd.DataFrame(
            {"X": [100.0, 101.0, 104.0] + [103.0] * 11}, index=index
        ),
        "low": pd.DataFrame({"X": np.full(14, 99.0)}, index=index),
    }
    feats = {"atr_pct": pd.DataFrame({"X": np.full(14, 0.02)}, index=index)}
    out = _build_hourly_path_auxiliary_targets(
        panel,
        feats,
        pd.DatetimeIndex([index[0]]),
        np.array(["X"]),
        side="long",
    )
    assert out["valid"].tolist() == [True]
    assert out["timing_valid"].tolist() == [True]
    np.testing.assert_allclose(out["peak_mfe_return"], [0.04])
    np.testing.assert_allclose(out["peak_mfe_atr"], [2.0])
    np.testing.assert_allclose(out["time_to_first_meaningful_mfe_hours"], [3.0])
    np.testing.assert_allclose(out["atr_fraction"], [0.02])
    np.testing.assert_allclose(out["mae_before_meaningful_mfe_atr"], [0.5])
    np.testing.assert_allclose(out["bars_before_price_stops_decreasing"], [1.0])
    np.testing.assert_allclose(out["future_slope_atr_per_hour"], [1.6 / 3.0])


def test_training_path_targets_include_decision_entry_bar_without_second_offset():
    index = pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC")
    panel = {
        "open": pd.DataFrame({"X": np.full(12, 100.0)}, index=index),
        "high": pd.DataFrame({"X": [105.0] + [101.0] * 11}, index=index),
        "low": pd.DataFrame({"X": np.full(12, 99.0)}, index=index),
    }
    feats = {"atr_pct": pd.DataFrame({"X": np.full(12, 0.02)}, index=index)}
    out = _build_hourly_path_auxiliary_targets(
        panel,
        feats,
        pd.DatetimeIndex([index[0]]),
        np.array(["X"]),
        side="long",
    )
    assert out["valid"].tolist() == [True]
    np.testing.assert_allclose(out["peak_mfe_return"], [0.05])
    np.testing.assert_allclose(out["time_to_first_meaningful_mfe_hours"], [1.0])
    np.testing.assert_allclose(out["mae_before_meaningful_mfe_atr"], [0.5])
    np.testing.assert_allclose(out["bars_before_price_stops_decreasing"], [0.0])
    np.testing.assert_allclose(out["future_slope_atr_per_hour"], [2.0])


def test_training_path_targets_reject_internal_hourly_gap():
    index = pd.date_range("2026-01-01", periods=14, freq="h", tz="UTC").delete(5)
    panel = {
        name: pd.DataFrame({"X": np.full(len(index), value)}, index=index)
        for name, value in (("open", 100.0), ("high", 101.0), ("low", 99.0))
    }
    feats = {
        "atr_pct": pd.DataFrame({"X": np.full(len(index), 0.02)}, index=index)
    }
    out = _build_hourly_path_auxiliary_targets(
        panel,
        feats,
        pd.DatetimeIndex([index[0]]),
        np.array(["X"]),
        side="long",
    )
    assert out["valid"].tolist() == [False]
