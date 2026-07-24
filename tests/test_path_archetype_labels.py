from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.path_archetype_labels import (
    CATBOOST_ARCHETYPE_COST_RETURN,
    PATH_ARCHETYPE_RULE_VERSION,
    deterministic_combined_path_archetype,
    deterministic_path_archetype,
    deterministic_path_realization_strength,
    materialize_path_archetypes,
    path_summary_columns,
    summarize_side_relative_path,
)


def _bars() -> pd.DataFrame:
    ts = pd.date_range("2026-01-01 01:00:00", periods=24, freq="h", tz="UTC")
    future = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": "BTC",
            "high": np.linspace(101, 124, 24),
            "low": np.linspace(99, 122, 24),
            "close": np.linspace(100.5, 123, 24),
        }
    )
    pre_decision = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01", tz="UTC")],
            "symbol": ["BTC"],
            "high": [1_000.0],
            "low": [1.0],
            "close": [500.0],
        }
    )
    return pd.concat([pre_decision, future], ignore_index=True)


def test_materializes_long_short_side_relative_and_decision_path_timing() -> None:
    candidates = pd.DataFrame(
        {
            "__ts__": [pd.Timestamp("2026-01-01", tz="UTC")] * 2,
            "__symbol__": ["BTC", "BTC"],
            "side": ["long", "short"],
            "entry_price": [100.0, 100.0],
            "risk_distance": [10.0, 10.0],
            "atr_fraction": [0.02, 0.02],
        }
    )
    out = materialize_path_archetypes(candidates, _bars())
    assert (
        out["__decision_ts__"] == pd.Timestamp("2026-01-01 01:00:00", tz="UTC")
    ).all()
    assert (
        out["__label_end_ts__"] == pd.Timestamp("2026-01-02 00:00:00", tz="UTC")
    ).all()
    assert out["path_arch_complete_24h"].tolist() == [1, 1]
    assert out.loc[0, "path_arch_peak_mfe_r"] == pytest.approx(2.4)
    assert out.loc[1, "path_arch_mae_24h_r"] == pytest.approx(-2.4)
    assert out["path_archetype_rule_version"].eq(PATH_ARCHETYPE_RULE_VERSION).all()


def test_barrier_only_candidate_materializes_default_one_r_stop_timing() -> None:
    candidates = pd.DataFrame(
        {
            "__ts__": [pd.Timestamp("2026-01-01", tz="UTC")],
            "__symbol__": ["BTC"],
            "side": ["long"],
            "__barrier_pct__": [0.02],
            "__path_auxiliary_atr_fraction__": [0.02],
        }
    )
    bars = _bars().copy()
    bars["open"] = 100.0
    bars.loc[bars["timestamp"] == pd.Timestamp("2026-01-01 01:00:00", tz="UTC"), "low"] = 97.0
    out = materialize_path_archetypes(candidates, bars)
    assert out.loc[0, "path_arch_time_to_stop_h"] == 1.0


def test_deterministic_rules_are_frozen() -> None:
    summary = {
        "path_arch_peak_mfe_r": 1.2,
        "path_arch_mfe_4h_r": 1.1,
        "path_arch_mfe_12h_r": 1.2,
        "path_arch_mae_4h_r": -0.3,
        "path_arch_final_return_r": 0.9,
        "path_arch_efficiency": 0.4,
        "path_arch_time_to_1r_h": 2.0,
        "path_arch_time_to_stop_h": np.nan,
        "path_arch_time_to_first_meaningful_mfe_h": 2.0,
        "path_arch_time_to_90pct_peak_mfe_h": 3.0,
    }
    assert deterministic_path_archetype(summary) == "fast_clean_winner"
    assert deterministic_path_archetype(summary) == "fast_clean_winner"
    assert (
        deterministic_path_archetype(
            {
                **summary,
                "path_arch_time_to_1r_h": np.nan,
                "path_arch_time_to_stop_h": 1.0,
            }
        )
        == "immediate_adverse_path"
    )


def test_incomplete_or_gapped_paths_are_not_labelled() -> None:
    candidates = pd.DataFrame(
        {
            "__ts__": [pd.Timestamp("2026-01-01", tz="UTC")],
            "__symbol__": ["BTC"],
            "side": ["long"],
            "entry_price": [100.0],
            "risk_distance": [10.0],
            "atr_fraction": [0.02],
        }
    )
    bars = _bars().iloc[:-1].copy()
    out = materialize_path_archetypes(candidates, bars)
    assert out.loc[0, "path_arch_complete_24h"] == 0
    assert pd.isna(out.loc[0, "path_archetype"])
    assert np.isnan(out.loc[0, "path_arch_mfe_24h_r"])


def test_vectorized_materialization_matches_scalar_oracle_for_sides_and_gaps() -> None:
    timestamps = pd.date_range("2026-01-01 01:00:00", periods=24, freq="h", tz="UTC")
    btc = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "BTC",
            "open": np.linspace(100, 123, 24),
            "high": np.linspace(101, 124, 24),
            "low": np.linspace(99, 122, 24),
            "close": np.linspace(100.5, 123, 24),
        }
    )
    eth = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "ETH",
            "open": np.linspace(200, 177, 24),
            "high": np.linspace(201, 178, 24),
            "low": np.linspace(199, 176, 24),
            "close": np.linspace(199.5, 176.5, 24),
        }
    )
    # One missing canonical UTC hour must invalidate the entire XRP label path.
    xrp = pd.DataFrame(
        {
            "timestamp": timestamps.delete(8),
            "symbol": "XRP",
            "open": np.linspace(10, 32, 23),
            "high": np.linspace(11, 33, 23),
            "low": np.linspace(9, 31, 23),
            "close": np.linspace(10.5, 32.5, 23),
        }
    )
    candidates = pd.DataFrame(
        {
            "__ts__": [pd.Timestamp("2026-01-01", tz="UTC")] * 4,
            "__symbol__": ["BTC", "BTC", "ETH", "XRP"],
            "side": ["long", "short", "short", "long"],
            "entry_price": [100.0, 100.0, np.nan, 10.0],
            "risk_distance": [10.0, 10.0, np.nan, 1.0],
            "atr_fraction": [0.02, 0.02, 0.02, 0.02],
            "barrier_pct": [np.nan, np.nan, 0.10, np.nan],
            "take_profit_r": [1.25, 1.25, np.nan, 1.0],
            "trailing_trigger_r": [0.75, 0.75, np.nan, 0.5],
            "stop_r": [1.0, 1.0, 1.0, 1.0],
            "path_cost_return": [0.004, 0.006, 0.008, 0.010],
            "activation_distance_return": [0.03, 0.04, 0.05, 0.06],
        }
    )
    bars = pd.concat((btc, eth, xrp), ignore_index=True)
    out = materialize_path_archetypes(candidates, bars)

    expected_specs = (
        (btc, 100.0, 10.0, 0.02, 1.0, 1.25, 0.75, 1.0, 0.004, 0.03),
        (btc, 100.0, 10.0, 0.02, -1.0, 1.25, 0.75, 1.0, 0.006, 0.04),
        # Missing candidate geometry uses the decision bar open and barrier risk.
        (eth, 200.0, 20.0, 0.02, -1.0, np.nan, np.nan, 1.0, 0.008, 0.05),
    )
    summary_columns = path_summary_columns()
    for row, (
        path,
        entry,
        risk,
        atr,
        sign,
        tp_r,
        trail_r,
        stop_r,
        cost_return,
        activation_distance_return,
    ) in enumerate(expected_specs):
        expected = summarize_side_relative_path(
            path["high"].to_numpy(dtype=np.float32),
            path["low"].to_numpy(dtype=np.float32),
            path["close"].to_numpy(dtype=np.float32),
            entry_price=entry,
            risk_distance=risk,
            atr_fraction=atr,
            side_sign=sign,
            take_profit_r=tp_r,
            trailing_trigger_r=trail_r,
            stop_r=stop_r,
            cost_return=cost_return,
            activation_distance_return=activation_distance_return,
        )
        np.testing.assert_allclose(
            out.loc[row, list(summary_columns)].to_numpy(dtype=np.float32),
            np.asarray([expected[column] for column in summary_columns], dtype=np.float32),
            equal_nan=True,
        )
        assert out.loc[row, "path_archetype"] == (
            deterministic_combined_path_archetype(expected)
        )

    assert out.loc[3, "path_arch_complete_24h"] == 0
    assert out.loc[3, list(summary_columns)].isna().all()
    assert pd.isna(out.loc[3, "path_archetype"])


def test_bulk_materialization_uses_bounded_vector_batches(monkeypatch) -> None:
    import extreme_price_movements.path_archetype_labels as labels

    def scalar_oracle_called(*args, **kwargs):
        raise AssertionError("candidate paths must not use the scalar row oracle")

    monkeypatch.setattr(labels, "summarize_side_relative_path", scalar_oracle_called)
    rows = 33_000  # Exceeds the implementation's fixed extraction batch size.
    candidates = pd.DataFrame(
        {
            "__ts__": pd.Timestamp("2026-01-01", tz="UTC"),
            "__symbol__": "BTC",
            "side": np.where(np.arange(rows) % 2, "long", "short"),
            "entry_price": 100.0,
            "risk_distance": 10.0,
            "atr_fraction": 0.02,
        }
    )
    out = materialize_path_archetypes(candidates, _bars())
    assert out["path_arch_complete_24h"].sum() == rows
    assert out["path_archetype"].notna().all()


@pytest.mark.parametrize(
    "atr_column",
    (
        "atr_fraction",
        "__atr_fraction__",
        "__path_auxiliary_atr_fraction__",
        "atr_pct",
        "atr_pct_base",
    ),
)
def test_atr_aliases_apply_usable_mfe_floor(atr_column: str) -> None:
    timestamps = pd.date_range("2026-01-01 01:00:00", periods=24, freq="h", tz="UTC")
    bars = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "BTC",
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
        }
    )
    candidates = pd.DataFrame(
        {
            "__ts__": [pd.Timestamp("2026-01-01", tz="UTC")],
            "__symbol__": ["BTC"],
            "side": ["long"],
            "entry_price": [100.0],
            "risk_distance": [10.0],
            atr_column: [0.04],
        }
    )
    out = materialize_path_archetypes(candidates, bars)
    expected = summarize_side_relative_path(
        bars["high"].to_numpy(dtype=np.float32),
        bars["low"].to_numpy(dtype=np.float32),
        bars["close"].to_numpy(dtype=np.float32),
        entry_price=100.0,
        risk_distance=10.0,
        atr_fraction=0.04,
        side_sign=1.0,
    )
    np.testing.assert_allclose(
        out.loc[0, list(path_summary_columns())].to_numpy(dtype=np.float32),
        np.asarray(
            [expected[column] for column in path_summary_columns()], dtype=np.float32
        ),
        equal_nan=True,
    )
    assert out.loc[0, "path_arch_complete_24h"] == 1
    assert out.loc[0, "path_arch_raw_peak_mfe_r"] == pytest.approx(0.1)
    assert out.loc[0, "path_arch_usable_mfe_floor_return"] == pytest.approx(0.06)
    assert out.loc[0, "path_arch_usable_mfe_threshold_r"] == pytest.approx(0.6)
    assert out.loc[0, "path_arch_peak_mfe_r"] == 0.0
    assert out.loc[0, "path_arch_mfe_24h_r"] == 0.0
    assert out.loc[0, "path_arch_mfe_to_cost"] == pytest.approx(
        0.01 / CATBOOST_ARCHETYPE_COST_RETURN
    )
    assert out.loc[0, "path_arch_mfe_to_activation_distance"] == pytest.approx(0.1)
    assert out.loc[0, "path_arch_time_to_first_meaningful_mfe_h"] == 24.0
    assert out.loc[0, "path_arch_time_to_90pct_peak_mfe_h"] == 24.0
    assert out.loc[0, "path_shape_archetype"] == "dead_timeout"
    assert out.loc[0, "path_realization_strength"] == "below_150atr"
    assert out.loc[0, "path_archetype"] == "dead_timeout__below_150atr"


def test_cost_aware_geometry_support_labels_are_materialized() -> None:
    candidates = pd.DataFrame(
        {
            "__ts__": [pd.Timestamp("2026-01-01", tz="UTC")],
            "__symbol__": ["BTC"],
            "side": ["long"],
            "entry_price": [100.0],
            "risk_distance": [10.0],
            "atr_fraction": [0.01],
        }
    )
    out = materialize_path_archetypes(candidates, _bars())
    expected = {
        "path_arch_cost_atr",
        "path_arch_meaningful_mfe_threshold_atr",
        "path_arch_peak_mfe_atr",
        "path_arch_peak_mfe_r",
        "path_arch_peak_mfe_minus_cost_atr",
        "path_arch_peak_mfe_div_cost",
        "path_arch_reaches_meaningful_mfe",
        "path_arch_bars_to_meaningful_mfe",
        "path_arch_bars_to_80pct_peak",
        "path_arch_bars_to_90pct_peak",
        "path_arch_mfe_2h_over_mfe_12h",
        "path_arch_mfe_4h_over_mfe_12h",
        "path_arch_mfe_8h_over_mfe_12h",
        "path_arch_bars_to_stop",
        "path_arch_stop_before_meaningful_mfe",
        "path_arch_mfe_before_stop_r",
        "path_arch_mae_2h_r",
        "path_arch_mae_4h_r",
        "path_arch_mae_before_meaningful_mfe_r",
        "path_arch_bars_below_entry_before_meaningful_mfe",
        "path_arch_adverse_area_before_meaningful_mfe_r",
        "path_arch_path_efficiency_to_meaningful_mfe",
        "path_arch_path_efficiency_to_90pct_peak",
        "path_arch_future_slope_atr_per_hour_4h",
        "path_arch_future_slope_atr_per_hour_12h",
        "path_arch_late_minus_early_slope",
        "path_arch_reversal_count",
        "path_arch_final_return_net_1pct",
        "path_arch_peak_retention_ratio",
        "path_arch_fraction_bars_above_50pct_peak",
    }
    assert expected.issubset(out.columns)
    assert out.loc[0, "path_arch_cost_atr"] == pytest.approx(1.0)
    assert out.loc[0, "path_arch_meaningful_mfe_threshold_atr"] == pytest.approx(
        1.5
    )
    assert out.loc[0, "path_arch_final_return_net_1pct"] == pytest.approx(0.22)
    assert out.loc[0, "path_arch_raw_mfe_atr_1h"] == pytest.approx(1.0)
    assert out.loc[0, "path_arch_raw_mfe_atr_12h"] == pytest.approx(12.0)
    assert out.loc[0, "path_arch_raw_mae_r_1h"] == pytest.approx(-0.1)
    assert out.loc[0, "path_arch_close_return_r_12h"] == pytest.approx(
        (_bars().loc[12, "close"] - 100.0) / 10.0
    )
    assert 0.0 <= out.loc[0, "path_arch_path_efficiency_to_90pct_peak"] <= 1.0

    legacy_cost = summarize_side_relative_path(
        highs=np.array([102.0, 103.0]),
        lows=np.array([99.0, 99.0]),
        closes=np.array([101.0, 102.0]),
        entry_price=100.0,
        risk_distance=10.0,
        atr_fraction=0.01,
        side_sign=1.0,
        cost_return=0.003,
        horizons_hours=(1, 2),
    )
    assert legacy_cost["path_arch_mfe_to_cost"] == pytest.approx(10.0)
    assert legacy_cost["path_arch_cost_atr"] == pytest.approx(1.0)
    assert legacy_cost["path_arch_peak_mfe_div_cost"] == pytest.approx(3.0)


def test_realization_strength_thresholds_and_efficiency_ratios() -> None:
    summary = summarize_side_relative_path(
        highs=np.array([101.0, 103.6, 106.0]),
        lows=np.array([99.0, 99.0, 99.0]),
        closes=np.array([100.5, 103.0, 105.0]),
        entry_price=100.0,
        risk_distance=10.0,
        atr_fraction=0.04,
        side_sign=1.0,
        cost_return=0.01,
        activation_distance_return=0.03,
        horizons_hours=(1, 2),
    )
    assert summary["path_arch_reached_150atr"] == 1.0
    assert summary["path_arch_reached_200atr"] == 0.0
    assert summary["path_arch_reached_300atr"] == 0.0
    assert summary["path_arch_reached_500atr"] == 0.0
    assert summary["path_arch_time_to_150atr_h"] == 3.0
    assert np.isnan(summary["path_arch_time_to_200atr_h"])
    assert np.isnan(summary["path_arch_time_to_300atr_h"])
    assert np.isnan(summary["path_arch_time_to_500atr_h"])
    assert summary["path_arch_peak_mfe_atr"] == pytest.approx(1.5)
    assert summary["path_arch_mfe_to_cost"] == pytest.approx(6.0)
    assert summary["path_arch_mfe_to_activation_distance"] == pytest.approx(2.0)
    assert deterministic_path_realization_strength(summary) == "atr150_200"
