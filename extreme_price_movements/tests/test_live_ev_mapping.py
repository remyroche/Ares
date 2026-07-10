import pytest

from extreme_price_movements.inference.run_inference import (
    _estimated_ev_from_strategy_prediction,
    _ev_adjusted_prediction_after_entry_friction,
)


def test_live_ev_estimate_prefers_side_archetype_curve():
    calibration = {
        "long_strategy": [
            {
                "mean_score": 0.9,
                "mean_gross_return": -0.01,
                "mean_net_return": -0.01,
                "mean_cost_bps": 100.0,
                "hit_rate": 0.1,
                "count": 100,
            }
        ],
        "long|long__clean_continuation": [
            {
                "mean_score": 0.9,
                "mean_gross_return": 0.02,
                "mean_net_return": 0.015,
                "mean_cost_bps": 100.0,
                "hit_rate": 0.8,
                "count": 100,
            }
        ],
    }

    result = _estimated_ev_from_strategy_prediction(
        0.9,
        "long_strategy",
        calibration,
        side="long",
        policy_archetype="long__clean_continuation",
    )

    assert result["estimated_ev_net_return"] == pytest.approx(0.015)
    assert result["estimated_ev_hit_rate"] == pytest.approx(0.8)
    assert "long|long__clean_continuation" in result["estimated_ev_source"]


def test_live_ev_haircut_uses_entry_half_spread_excess_and_inverse_score_mapping():
    calibration = {
        "short|short__breakout": [
            {
                "mean_score": 0.5,
                "mean_net_return": 0.01,
                "count": 100,
            },
            {
                "mean_score": 1.0,
                "mean_net_return": 0.02,
                "count": 100,
            },
        ]
    }

    result = _ev_adjusted_prediction_after_entry_friction(
        calibrated_score=1.0,
        strategy_id="short_strategy",
        side="short",
        policy_archetype="short__breakout",
        calibration=calibration,
        live_entry_friction_bps=0.0,
        observed_spread_bps=150.0,
        orderbook_slippage_bps=0.0,
        adverse_signal_gap_bps=0.0,
        spread_baseline_bps=100.0,
        delay_slippage_baseline_bps=0.0,
        expected_stop_exit_friction_bps=0.0,
        stop_exit_baseline_bps=0.0,
    )

    assert result["ev_haircut_spread_excess_bps"] == pytest.approx(25.0)
    assert result["ev_haircut_bps"] == pytest.approx(25.0)
    assert result["ev_adjusted_net_return_before_friction"] == pytest.approx(0.02)
    assert result["ev_adjusted_net_return_after_friction"] == pytest.approx(0.0175)
    assert result["ev_adjusted_calibrated_score"] == pytest.approx(0.875)
    assert result["ev_adjusted_curve_key"] == "short|short__breakout"


def test_live_ev_rebase_uses_inference_only_low_fixed_cost_and_inflated_live_spread():
    calibration = {
        "long|long__breakout": [
            {
                "mean_score": 0.5,
                "mean_gross_return": 0.020,
                "mean_net_return": 0.010,
                "count": 100,
            },
            {
                "mean_score": 1.0,
                "mean_gross_return": 0.040,
                "mean_net_return": 0.030,
                "count": 100,
            },
        ]
    }

    result = _ev_adjusted_prediction_after_entry_friction(
        calibrated_score=0.5,
        strategy_id="long_strategy",
        side="long",
        policy_archetype="long__breakout",
        calibration=calibration,
        live_entry_friction_bps=0.0,
        observed_spread_bps=20.0,
        orderbook_slippage_bps=0.0,
        adverse_signal_gap_bps=0.0,
        spread_baseline_bps=100.0,
        delay_slippage_baseline_bps=0.0,
        expected_stop_exit_friction_bps=0.0,
        stop_exit_baseline_bps=0.0,
        inference_cost_rebase_enabled=True,
        inference_fixed_round_trip_cost_bps=20.0,
        inference_spread_multiplier=1.5,
    )

    # Inference-only cost = 20 bps fixed round trip + 1.5 * 20 bps live spread.
    assert result["ev_inference_total_cost_bps"] == pytest.approx(50.0)
    assert result["ev_adjusted_historical_net_return_before_rebase"] == pytest.approx(
        0.010
    )
    assert result["ev_adjusted_net_return_after_friction"] == pytest.approx(0.015)
    # Mapping the rebased EV back through the historical net curve gives low-spread
    # rows a rank-equivalent credit.
    assert result["ev_adjusted_calibrated_score"] > 0.5
    assert result["ev_inference_cost_rebase_applied"] is True
