import json

import numpy as np
import pandas as pd

from scripts.materialize_execution_action_target_pack import (
    compute_action_targets,
)


def _payload(close):
    close = np.asarray(close, dtype=float)
    start = pd.Timestamp("2025-03-01T00:00:00Z").value
    timestamp = start + np.arange(720, dtype=np.int64) * 60_000_000_000
    return json.dumps(
        {
            "timestamp": timestamp.tolist(),
            "open": close.tolist(),
            "high": (close + 0.1).tolist(),
            "low": (close - 0.1).tolist(),
            "close": close.tolist(),
        }
    )


def test_fixed_horizons_are_side_signed_and_cost_applied_once():
    payload = _payload(np.linspace(100.0, 112.0, 720))
    long = compute_action_targets(
        payload,
        decision_price=100.0,
        side_name="long",
        cost_return=0.01,
        atr_1h=1.0,
    )
    short = compute_action_targets(
        payload,
        decision_price=100.0,
        side_name="short",
        cost_return=0.01,
        atr_1h=1.0,
    )
    assert np.isclose(long["target_fixed_12h_gross_return"], 0.12)
    assert np.isclose(long["target_fixed_12h_net_return"], 0.11)
    assert np.isclose(short["target_fixed_12h_gross_return"], -0.12)
    assert np.isclose(short["target_fixed_12h_net_return"], -0.13)


def test_opportunity_buffers_are_nested_and_time_is_from_decision():
    close = np.full(720, 100.0)
    close[119:] = 101.3
    result = compute_action_targets(
        _payload(close),
        decision_price=100.0,
        side_name="long",
        cost_return=0.01,
        atr_1h=1.0,
    )
    assert result["target_cost_clear_opportunity_0bps"] == 1
    assert result["target_cost_clear_opportunity_25bps"] == 1
    assert result["target_cost_clear_opportunity_50bps"] == 0
    assert result["target_time_to_cost_clear_0bps_minutes"] == 120
    assert result["target_time_to_cost_clear_25bps_minutes"] == 120
    assert result["target_time_to_cost_clear_50bps_censored_hours"] == 12.0


def test_early_clean_nonflat_balances_flatness_and_adverse_path():
    clean = np.linspace(100.0, 101.0, 720)
    result = compute_action_targets(
        _payload(clean),
        decision_price=100.0,
        side_name="long",
        cost_return=0.001,
        atr_1h=1.0,
    )
    assert result["target_early_2h_clean_nonflat"] == 0
    stronger = np.concatenate(
        [np.linspace(100.0, 101.0, 120), np.full(600, 101.0)]
    )
    result = compute_action_targets(
        _payload(stronger),
        decision_price=100.0,
        side_name="long",
        cost_return=0.001,
        atr_1h=1.0,
    )
    assert result["target_early_2h_clean_nonflat"] == 1


def test_giveback_and_underwater_targets_are_economically_directional():
    close = np.concatenate(
        [
            np.linspace(100.0, 105.0, 360),
            np.linspace(105.0, 98.0, 360),
        ]
    )
    result = compute_action_targets(
        _payload(close),
        decision_price=100.0,
        side_name="long",
        cost_return=0.01,
        atr_1h=1.0,
    )
    assert result["target_final_close_giveback_from_peak_return"] > 0.06
    assert result["target_worst_post_peak_close_giveback_ratio"] > 1.0
    assert result["target_underwater_fraction_12h"] > 0.0


def test_zero_peak_timing_is_censored_and_giveback_ratio_is_undefined():
    close = np.linspace(100.0, 95.0, 720)
    payload = json.dumps(
        {
            "timestamp": (
                pd.Timestamp("2025-03-01T00:00:00Z").value
                + np.arange(720, dtype=np.int64) * 60_000_000_000
            ).tolist(),
            "open": close.tolist(),
            "high": close.tolist(),
            "low": (close - 0.1).tolist(),
            "close": close.tolist(),
        }
    )
    result = compute_action_targets(
        payload,
        decision_price=100.0,
        side_name="long",
        cost_return=0.01,
        atr_1h=1.0,
    )
    assert result["target_peak_mfe_timing_valid_12h"] == 0
    assert np.isnan(result["target_time_to_80pct_mfe_hours_12h"])
    assert result["target_time_to_80pct_mfe_censored_hours_12h"] == 12.0
    assert result["target_underwater_minutes_before_80pct_mfe"] == 719
    assert np.isnan(result["target_max_close_giveback_after_80pct_mfe_ratio"])
    assert np.isnan(result["target_final_close_giveback_from_peak_return"])
    assert np.isnan(result["target_worst_post_peak_close_giveback_return"])
