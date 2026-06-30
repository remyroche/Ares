from argparse import Namespace

import pandas as pd
import pytest

from scripts.compare_dynamic_hr_surprise_threshold import (
    ROBUST_SUBWINDOWS_V2_PRESET,
    ROBUST_SUBWINDOWS_V3_PRESET,
    ROBUST_SUBWINDOWS_V4_PRESET,
    ROBUST_SUBWINDOWS_V5_PRESET,
    ROBUST_SUBWINDOWS_V6_PRESET,
    _apply_policy_preset,
    _inactive_params_for_heads,
    _recent_quantile_day_weights,
    _spread_net_return_from_columns,
    _weighted_quantile_by_column,
)


def test_calendar_inactive_params_can_fallback_to_deployed_threshold():
    args = Namespace(
        x_min_days=1.0,
        fallback_rejected_heads_to_deployed=True,
    )

    params = _inactive_params_for_heads(
        {"short_asset"},
        args,
        deployed_thresholds={"short_asset": 0.94},
    )

    param = params["short_asset"]
    assert param.guarded_y == 0.94
    assert param.deactivated is False
    assert param.dynamic_rejected is True
    assert param.fallback_to_deployed is True
    assert param.fallback_threshold == 0.94


def test_robust_subwindows_v2_preset_uses_calendar_replay_without_deployed_floor():
    args = Namespace(
        policy_preset=ROBUST_SUBWINDOWS_V2_PRESET,
        calendar_only=False,
        calendar_replay=False,
        use_deployed_threshold_floor=True,
        fallback_rejected_heads_to_deployed=True,
        require_dynamic_head_improvement_over_deployed=True,
        require_dynamic_head_tail_not_worse_than_deployed=True,
    )

    _apply_policy_preset(args)

    assert args.calendar_only is True
    assert args.calendar_replay is True
    assert args.calendar_xw_min_train_days == 90.0
    assert args.calendar_xw_max_train_days == 183.0
    assert args.calendar_y_train_days == 28.0
    assert args.use_deployed_threshold_floor is False
    assert args.fallback_rejected_heads_to_deployed is False
    assert args.head_optimization_mode == "independent"
    assert args.threshold_refresh_mode == "grid"
    assert args.subwindow_days == 7.0
    assert args.min_positive_objective_fraction == 0.55
    assert args.subwindow_q15_floor == -0.25
    assert args.min_threshold_selected_count == 15
    assert args.min_threshold_active_subwindows == 2


def test_robust_subwindows_v3_preset_adds_soft_deployed_threshold_prior():
    args = Namespace(
        policy_preset=ROBUST_SUBWINDOWS_V3_PRESET,
        calendar_only=False,
        calendar_replay=False,
        use_deployed_threshold_floor=True,
        fallback_rejected_heads_to_deployed=True,
        require_dynamic_head_improvement_over_deployed=True,
        require_dynamic_head_tail_not_worse_than_deployed=True,
    )

    _apply_policy_preset(args)

    assert args.calendar_only is True
    assert args.use_deployed_threshold_floor is False
    assert args.deployed_threshold_soft_prior_strength > 0.0
    assert args.deployed_threshold_soft_prior_deadband == pytest.approx(0.03)
    assert args.deployed_threshold_soft_prior_activity_weight == pytest.approx(0.50)


def test_robust_subwindows_v4_uses_subwindow_penalty_not_hard_gate():
    args = Namespace(
        policy_preset=ROBUST_SUBWINDOWS_V4_PRESET,
        calendar_only=False,
        calendar_replay=False,
        use_deployed_threshold_floor=True,
        fallback_rejected_heads_to_deployed=True,
        require_dynamic_head_improvement_over_deployed=True,
        require_dynamic_head_tail_not_worse_than_deployed=True,
    )

    _apply_policy_preset(args)

    assert args.calendar_only is True
    assert args.use_deployed_threshold_floor is False
    assert args.calendar_y_train_days == pytest.approx(20.0)
    assert args.subwindow_days == pytest.approx(5.0)
    assert args.min_subwindows == 4
    assert args.subwindow_constraints_mode == "penalty"
    assert args.min_positive_objective_fraction == pytest.approx(0.25)
    assert args.subwindow_q15_floor == pytest.approx(-1.00)
    assert args.subwindow_drawdown_floor == pytest.approx(-3.00)
    assert args.subwindow_constraint_penalty == pytest.approx(10.0)
    assert args.min_threshold_selected_count == 0
    assert args.min_threshold_active_subwindows == 0
    assert args.per_head_min_objective < -1e12


def test_robust_subwindows_v5_enables_recent_validation_guard():
    args = Namespace(
        policy_preset=ROBUST_SUBWINDOWS_V5_PRESET,
        calendar_only=False,
        calendar_replay=False,
        use_deployed_threshold_floor=True,
        fallback_rejected_heads_to_deployed=True,
        require_dynamic_head_improvement_over_deployed=True,
        require_dynamic_head_tail_not_worse_than_deployed=True,
        recent_validation_guard=False,
    )

    _apply_policy_preset(args)

    assert args.calendar_y_train_days == pytest.approx(20.0)
    assert args.subwindow_days == pytest.approx(5.0)
    assert args.subwindow_constraints_mode == "penalty"
    assert args.recent_validation_guard is True
    assert args.recent_validation_days == pytest.approx(5.0)
    assert args.recent_validation_min_count == 20
    assert args.recent_validation_min_total_pnl == pytest.approx(0.0)


def test_robust_subwindows_v6_uses_recent_quantile_threshold_objective():
    args = Namespace(
        policy_preset=ROBUST_SUBWINDOWS_V6_PRESET,
        calendar_only=False,
        calendar_replay=False,
        use_deployed_threshold_floor=True,
        fallback_rejected_heads_to_deployed=True,
        require_dynamic_head_improvement_over_deployed=True,
        require_dynamic_head_tail_not_worse_than_deployed=True,
        threshold_selection_objective="subwindow",
        recent_quantile_days=28.0,
    )

    _apply_policy_preset(args)

    assert args.calendar_y_train_days == pytest.approx(20.0)
    assert args.threshold_selection_objective == "recent_daily_quantile"
    assert args.recent_quantile_days == pytest.approx(20.0)
    assert args.subwindow_constraints_mode == "penalty"


def test_recent_quantile_bucket_weights_are_vectorized_by_day_bucket():
    args = Namespace(
        recent_quantile_weight_mode="bucket",
        recent_quantile_weight_last_7=1.0,
        recent_quantile_weight_prev_7=0.3,
        recent_quantile_weight_older=0.5,
    )
    days = pd.DatetimeIndex(
        [
            "2026-06-02 00:00:00+00:00",
            "2026-06-15 00:00:00+00:00",
            "2026-06-23 00:00:00+00:00",
        ]
    )

    weights = _recent_quantile_day_weights(days, pd.Timestamp("2026-06-25 12:00:00+00:00"), args)

    assert weights.tolist() == pytest.approx([0.5, 0.3, 1.0])


def test_weighted_quantile_by_column_uses_day_weights():
    values = pd.DataFrame(
        {
            "strict": [-10.0, 1.0, 2.0],
            "loose": [-2.0, 1.0, 10.0],
        }
    ).to_numpy(dtype=float)
    weights = pd.Series([10.0, 1.0, 1.0]).to_numpy(dtype=float)

    q50 = _weighted_quantile_by_column(values, weights, 0.50)

    assert q50.tolist() == pytest.approx([-10.0, -2.0])


def test_spread_net_return_uses_full_spread_when_half_spread_columns_are_zero():
    frame = pd.DataFrame(
        {
            "net_return": [0.010, -0.002],
            "net_return_before_spread": [0.010, -0.002],
            "expected_spread_bps": [40.0, 100.0],
            "expected_half_spread_bps": [0.0, 0.0],
            "spread_cost_bps": [0.0, 0.0],
            "exit_spread_cost_bps": [0.0, 0.0],
        }
    )

    returns, diagnostics = _spread_net_return_from_columns(frame, return_col="net_return")

    assert returns.tolist() == pytest.approx([0.006, -0.012])
    assert diagnostics["spread_adjustment_uses_full_spread_fallback"] is True
    assert diagnostics["spread_adjustment_bps_mean"] == pytest.approx(70.0)
