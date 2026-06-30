import numpy as np
import pandas as pd
from pathlib import Path

from scripts import run_recent_head_activation_optuna as mod


def _params() -> mod.RecentHeadParams:
    return mod.RecentHeadParams(
        lookback_hours=48,
        embargo_hours=12,
        min_samples=10,
        shrink_samples=30.0,
        decay_halflife_hours=48,
        health_clip=0.5,
        net_weight=1.0,
        hr_weight=0.5,
        weighted_hr_weight=0.5,
        ic_weight=0.3,
        full_sl_weight=1.0,
        cost_drag_weight=0.4,
        worst_return_weight=0.4,
        weighted_hr_power=1.5,
        head_control_strength=0.5,
        threshold_start=0.2,
        threshold_scale=0.1,
        threshold_power=1.2,
        max_threshold_shift=0.08,
        size_start=0.1,
        size_scale=0.4,
        min_size_multiplier=0.5,
        cap_start=0.6,
        cap_scale=0.5,
        hard_stop_health=-3.0,
        hard_stop_threshold=0.99,
        objective_q_low_weight=0.1,
        objective_q_mid_deterioration_weight=0.2,
        objective_defensive_success_weight=0.1,
        objective_full_sl_penalty=50.0,
        objective_q_low=0.05,
        objective_q_mid=0.15,
        objective_short_horizon_hours=48,
        objective_long_horizon_hours=120,
        objective_hard_stop_start=0.25,
        objective_hard_stop_weight=10_000.0,
        objective_head_action_weight=10_000.0,
        objective_head_hard_stop_start=0.45,
        objective_head_threshold_start=0.20,
        objective_head_size_floor=0.35,
        objective_head_capacity_start=0.75,
        objective_max_head_trade_share=0.75,
        objective_global_balance_weight=100.0,
        objective_weekly_balance_weight=100.0,
    )


def _candidate_frame() -> pd.DataFrame:
    n = 4
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-06-01T00:00:00Z", "2026-06-01T00:00:00Z"] * 2,
                utc=True,
            ),
            "symbol": ["A", "B", "C", "D"],
            "side": ["short"] * n,
            "strategy_id": ["short_boll_s1", "short_boll_s1", "short_asset_s1", "short_asset_s1"],
            "head": ["short_boll", "short_boll", "short_asset", "short_asset"],
            "base_strategy_threshold": [0.70] * n,
            "calibrated_score": [0.10, 0.90, 0.20, 0.80],
            "normalized_rank_score": [0.11, 0.22, 0.33, 0.44],
            "strategy_rank_pct": [0.11, 0.22, 0.33, 0.44],
            "policy_rank_pct": [0.11, 0.22, 0.33, 0.44],
            "rank_pct": [0.11, 0.22, 0.33, 0.44],
            "entry_price": [100.0] * n,
            "exit_timestamp": pd.date_range("2026-06-01T01:00:00Z", periods=n, freq="h"),
            "exit_price": [101.0] * n,
            "net_return": [0.01] * n,
            "gross_return": [0.012] * n,
            "holding_bars": [1] * n,
            "simple_policy_exit_reason": ["tp"] * n,
        }
    )


def test_objective_contract_does_not_allow_named_head_terms() -> None:
    contract = mod._objective_contract()
    allowed_text = " ".join(str(v) for v in contract.get("allowed_head_terms", [])).lower()
    for head in mod.HEADS:
        assert head.lower() not in allowed_text

    assert contract["head_identity_invariant"] is True
    assert contract["no_named_head_reward"] is True
    assert contract["no_named_head_suppression_penalty"] is True


def test_arg_parser_defaults_to_active_t1_rank_contract() -> None:
    args = mod._build_arg_parser().parse_args([])

    assert args.rank_contract == "short_boll_timestamp_rank"
    assert mod.DEFAULT_RANK_CONTRACT == "short_boll_timestamp_rank"


def test_global_reference_remains_explicit_research_challenger() -> None:
    args = mod._build_arg_parser().parse_args(
        ["--rank-contract", "anchor_global_policy_rank_reference"]
    )

    assert args.rank_contract == "anchor_global_policy_rank_reference"
    assert mod._rank_scope("anchor_global_policy_rank_reference") == "global_over_time"
    assert mod._ranked_candidate_label("anchor_global_policy_rank_reference") == "global_rank"


def test_t1_timestamp_rank_mode_repairs_only_short_boll() -> None:
    frame = _candidate_frame()

    ranked, diag = mod._apply_ablation_rank_contract(
        frame,
        rank_contract="short_boll_timestamp_rank",
        data_root=Path("."),
        rank_reference_run_id="unused",
    )

    short_boll = ranked.loc[ranked["head"].eq("short_boll")].sort_values("calibrated_score")
    short_asset = ranked.loc[ranked["head"].eq("short_asset")].sort_values("calibrated_score")

    np.testing.assert_allclose(short_boll["policy_rank_pct"].to_numpy(), [0.5, 1.0])
    np.testing.assert_allclose(short_asset["policy_rank_pct"].to_numpy(), [0.33, 0.44])
    assert diag["rank_contract"] == "short_boll_timestamp_rank"
    assert diag["rank_scope"] == "within_timestamp"
    assert diag["rank_reference_run_id"] is None
    assert diag["missing_rank_rows"] == 0


def test_recent_health_schedule_uses_outcome_availability_not_entry_time() -> None:
    reference = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-06-01T00:00:00Z", "2026-06-01T01:00:00Z"],
                utc=True,
            ),
            "exit_timestamp": pd.to_datetime(
                ["2026-06-01T04:00:00Z", "2026-06-03T00:00:00Z"],
                utc=True,
            ),
            "head": ["short_asset", "short_asset"],
            "normalized_rank_score": [0.90, 0.95],
            "net_return": [0.01, -0.10],
            "gross_return": [0.012, -0.098],
            "simple_policy_exit_reason": ["tp", "sl"],
        }
    )
    params = _params()

    schedule = mod._recent_health_schedule(
        reference,
        pd.Series(pd.to_datetime(["2026-06-02T00:00:00Z"], utc=True)),
        params=params,
        baselines={"short_asset": {}},
    )

    short_asset = schedule.loc[schedule["head"].eq("short_asset")].iloc[0]
    assert int(short_asset["recent_rows"]) == 1
    assert float(short_asset["recent_net_mean"]) > 0.0


def test_period_payload_records_bounds_and_head_counts() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-05-01T00:00:00Z", "2026-05-01T01:00:00Z", None],
                utc=True,
                errors="coerce",
            ),
            "head": ["short_asset", "short_boll", "short_asset"],
        }
    )

    payload = mod._period_payload(frame)

    assert payload["row_count"] == 3
    assert payload["timestamp_min"] == "2026-05-01T00:00:00+00:00"
    assert payload["timestamp_max"] == "2026-05-01T01:00:00+00:00"
    assert payload["timestamp_count"] == 2
    assert payload["head_counts"] == {"short_asset": 1, "short_boll": 1}


def test_chronological_selection_split_uses_complete_timestamps() -> None:
    timestamps = pd.date_range("2026-05-01", periods=6, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": np.repeat(timestamps, 2),
            "head": ["short_asset", "short_boll"] * 6,
            "net_return": np.arange(12, dtype=float),
        }
    )

    reference, objective, diag = mod._chronological_selection_split(
        frame,
        validation_frac=0.50,
        min_validation_timestamps=2,
    )

    ref_ts = pd.to_datetime(reference["timestamp"], utc=True)
    obj_ts = pd.to_datetime(objective["timestamp"], utc=True)
    assert diag["mode"] == "chronological_holdout"
    assert ref_ts.max() < obj_ts.min()
    assert ref_ts.nunique() == 3
    assert obj_ts.nunique() == 3
    assert set(frame.loc[frame["timestamp"].isin(obj_ts), "timestamp"]) == set(objective["timestamp"])
    assert diag["complete_timestamp_split"] is True


def test_chronological_selection_split_zero_fraction_uses_full_reference_replay() -> None:
    frame = _candidate_frame()

    reference, objective, diag = mod._chronological_selection_split(
        frame,
        validation_frac=0.0,
        min_validation_timestamps=2,
    )

    assert diag["mode"] == "full_reference_replay"
    assert len(reference) == len(frame)
    assert len(objective) == len(frame)


def test_selection_ev_reference_excludes_selection_objective_timestamps() -> None:
    timestamps = pd.date_range("2026-05-01", periods=6, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": np.repeat(timestamps, 2),
            "head": ["short_asset", "short_boll"] * 6,
            "net_return": np.arange(12, dtype=float),
        }
    )
    _, objective, split = mod._chronological_selection_split(
        frame,
        validation_frac=0.50,
        min_validation_timestamps=2,
    )

    ev_reference, diag = mod._selection_ev_reference(frame, split)

    ref_ts = pd.to_datetime(ev_reference["timestamp"], utc=True)
    obj_ts = pd.to_datetime(objective["timestamp"], utc=True)
    assert diag["mode"] == "chronological_pre_selection_objective_ev_reference"
    assert ref_ts.max() < obj_ts.min()
    assert set(ref_ts.unique()).isdisjoint(set(obj_ts.unique()))
    assert diag["complete_timestamp_split"] is True


def test_head_concentration_penalty_ignores_per_head_pnl() -> None:
    timestamps = pd.date_range("2026-05-01", periods=10, freq="h", tz="UTC")
    base = pd.DataFrame(
        {
            "timestamp": timestamps,
            "head": ["short_asset"] * 8 + ["short_boll"] * 2,
            "net_pnl": [100.0] * 8 + [-1000.0] * 2,
        }
    )
    flipped = base.copy()
    flipped["net_pnl"] = [-1000.0] * 8 + [100.0] * 2

    base_penalty = mod._head_activity_concentration_penalty(base, max_head_share=0.70)
    flipped_penalty = mod._head_activity_concentration_penalty(flipped, max_head_share=0.70)

    np.testing.assert_allclose(base_penalty, flipped_penalty)
    np.testing.assert_allclose(base_penalty[0], (0.80 - 0.70) ** 2)


def test_head_concentration_penalty_does_not_name_winning_head() -> None:
    timestamps = pd.date_range("2026-05-01", periods=12, freq="h", tz="UTC")
    frame_a = pd.DataFrame(
        {
            "timestamp": timestamps,
            "head": ["short_asset"] * 9 + ["short_boll"] * 3,
        }
    )
    frame_b = pd.DataFrame(
        {
            "timestamp": timestamps,
            "head": ["short_boll"] * 9 + ["short_asset"] * 3,
        }
    )

    penalty_a = mod._head_activity_concentration_penalty(frame_a, max_head_share=0.75)
    penalty_b = mod._head_activity_concentration_penalty(frame_b, max_head_share=0.75)

    np.testing.assert_allclose(penalty_a, penalty_b)


def test_head_concentration_penalty_counts_unknown_head_labels() -> None:
    timestamps = pd.date_range("2026-05-01", periods=10, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "head": ["future_head_a"] * 8 + ["future_head_b"] * 2,
        }
    )

    penalty = mod._head_activity_concentration_penalty(frame, max_head_share=0.70)

    np.testing.assert_allclose(penalty[0], (0.80 - 0.70) ** 2)


def test_head_concentration_cap_uses_observed_head_count() -> None:
    timestamps = pd.date_range("2026-05-01", periods=10, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "head": ["future_head_a"] * 8 + ["future_head_b"] * 2,
        }
    )

    penalty = mod._head_activity_concentration_penalty(frame, max_head_share=0.20)

    # With only two observed heads, the smallest feasible max-share cap is 0.50,
    # not 0.25 from the current production HEADS tuple.
    np.testing.assert_allclose(penalty[0], (0.80 - 0.50) ** 2)


def test_overlay_metrics_emit_unknown_head_action_columns() -> None:
    candidates = pd.DataFrame(
        {
            "head": ["future_head_a", "future_head_a", "future_head_b"],
            "recent_head_threshold_delta": [0.30, 0.10, 0.0],
            "portfolio_size_multiplier": [0.50, 0.70, 1.0],
            "portfolio_max_new_entries_per_strategy_per_bar": [1, 2, 2],
            "recent_head_hard_stop": [True, False, False],
        }
    )

    metrics = mod._overlay_metrics(candidates, pd.DataFrame())

    assert metrics["future_head_a_candidate_share_hard_stop"] == 0.5
    np.testing.assert_allclose(metrics["future_head_a_candidate_mean_threshold_delta"], 0.20)
    assert metrics["future_head_b_candidate_mean_size_multiplier"] == 1.0


def test_candidate_action_metric_heads_are_data_driven() -> None:
    summary = pd.Series(
        {
            "future_head_a_candidate_share_hard_stop": 0.25,
            "future_head_a_candidate_mean_threshold_delta": 0.30,
            "future_head_a_candidate_mean_size_multiplier": 0.20,
            "future_head_a_candidate_share_capacity_reduced": 0.90,
            "candidate_share_hard_stop": 0.25,
        }
    )

    assert mod._candidate_action_metric_heads(summary) == ["future_head_a"]


def test_objective_action_penalty_includes_unknown_head_metrics() -> None:
    summary = pd.Series(
        {
            "net_pnl": 100.0,
            "full_sl_rate": 0.1,
            "candidate_share_hard_stop": 0.0,
            "future_head_a_candidate_share_hard_stop": 0.10,
            "future_head_a_candidate_mean_threshold_delta": 0.25,
            "future_head_a_candidate_mean_size_multiplier": 0.20,
            "future_head_a_candidate_share_capacity_reduced": 0.90,
        }
    )

    common = {
        "accepted": pd.DataFrame(),
        "baseline_summary": pd.Series({"net_pnl": 0.0, "full_sl_rate": 0.1}),
        "baseline_accepted": pd.DataFrame(),
        "min_trades": 0,
        "params": _params(),
    }
    components = mod._objective_components(summary, **common)

    assert components["head_action_concentration_penalty"] > 0.0


def test_objective_penalizes_trade_count_below_floor() -> None:
    params = _params()
    common = {
        "accepted": pd.DataFrame(),
        "baseline_summary": pd.Series({"net_pnl": 0.0, "full_sl_rate": 0.1}),
        "baseline_accepted": pd.DataFrame(),
        "params": params,
    }
    low_trade_summary = pd.Series(
        {
            "net_pnl": 100.0,
            "trade_count": 6.0,
            "full_sl_rate": 0.1,
            "candidate_share_hard_stop": 0.0,
        }
    )
    enough_trade_summary = low_trade_summary.copy()
    enough_trade_summary["trade_count"] = 10.0

    low = mod._objective_components(low_trade_summary, min_trades=10, **common)
    enough = mod._objective_components(enough_trade_summary, min_trades=10, **common)

    assert low["min_trade_shortfall"] == 4.0
    assert low["min_trade_penalty"] == 16.0
    assert enough["min_trade_shortfall"] == 0.0
    assert enough["min_trade_penalty"] == 0.0
    np.testing.assert_allclose(enough["objective"] - low["objective"], 16.0)


def test_portfolio_promotion_gate_requires_defensive_success_and_no_degradation() -> None:
    baseline = pd.Series(
        {
            "trade_count": 100,
            "net_pnl": 100.0,
            "robust_downside_pnl": -50.0,
            "full_sl_rate": 0.20,
            "worst_24h_net_pnl": -25.0,
        }
    )
    passing = pd.Series(
        {
            "trade_count": 90,
            "net_pnl": 120.0,
            "robust_downside_pnl": -40.0,
            "full_sl_rate": 0.18,
            "worst_24h_net_pnl": -20.0,
            "defensive_success_pnl": 5.0,
            "loss_avoided_pnl": 15.0,
            "winner_pnl_sacrificed": 10.0,
        }
    )
    failing = passing.copy()
    failing["net_pnl"] = 95.0
    failing["defensive_success_pnl"] = -1.0
    failing["loss_avoided_pnl"] = 8.0
    failing["winner_pnl_sacrificed"] = 12.0

    pass_gate = mod._portfolio_promotion_gate(baseline, passing)
    fail_gate = mod._portfolio_promotion_gate(baseline, failing)

    assert pass_gate["passed"] is True
    assert fail_gate["passed"] is False
    assert fail_gate["gates"]["net_pnl_improved"] is False
    assert fail_gate["gates"]["defensive_success_positive"] is False
    assert fail_gate["gates"]["loss_avoided_exceeds_winner_sacrificed"] is False


def test_objective_action_penalty_is_head_identity_invariant() -> None:
    summary_a = pd.Series(
        {
            "net_pnl": 100.0,
            "full_sl_rate": 0.1,
            "candidate_share_hard_stop": 0.0,
            "short_asset_candidate_share_hard_stop": 0.10,
            "short_asset_candidate_mean_threshold_delta": 0.25,
            "short_asset_candidate_mean_size_multiplier": 0.20,
            "short_asset_candidate_share_capacity_reduced": 0.90,
            "short_boll_candidate_share_hard_stop": 0.00,
            "short_boll_candidate_mean_threshold_delta": 0.00,
            "short_boll_candidate_mean_size_multiplier": 1.00,
            "short_boll_candidate_share_capacity_reduced": 0.00,
        }
    )
    summary_b = summary_a.copy()
    for suffix in (
        "candidate_share_hard_stop",
        "candidate_mean_threshold_delta",
        "candidate_mean_size_multiplier",
        "candidate_share_capacity_reduced",
    ):
        a_key = f"short_asset_{suffix}"
        b_key = f"short_boll_{suffix}"
        summary_b[a_key], summary_b[b_key] = summary_a[b_key], summary_a[a_key]

    common = {
        "accepted": pd.DataFrame(),
        "baseline_summary": pd.Series({"net_pnl": 0.0, "full_sl_rate": 0.1}),
        "baseline_accepted": pd.DataFrame(),
        "min_trades": 0,
        "params": _params(),
    }
    obj_a = mod._objective_components(summary_a, **common)["objective"]
    obj_b = mod._objective_components(summary_b, **common)["objective"]

    np.testing.assert_allclose(obj_a, obj_b)


def test_objective_full_head_swap_is_identity_invariant() -> None:
    timestamps = pd.date_range("2026-05-01", periods=12, freq="h", tz="UTC")
    accepted_a = pd.DataFrame(
        {
            "timestamp": timestamps,
            "head": ["short_asset"] * 9 + ["short_boll"] * 3,
            "net_return": [0.020, 0.010, -0.015, 0.018, -0.004, 0.012, 0.009, -0.008, 0.006, 0.011, -0.010, 0.007],
            "gross_return": [0.022, 0.012, -0.013, 0.020, -0.002, 0.014, 0.011, -0.006, 0.008, 0.013, -0.008, 0.009],
            "simple_policy_exit_reason": ["tp", "tp", "sl", "tp", "timeout", "tp", "tp", "sl", "timeout", "tp", "sl", "tp"],
        }
    )
    accepted_b = accepted_a.copy()
    accepted_b["head"] = accepted_b["head"].replace({"short_asset": "short_boll", "short_boll": "short_asset"})

    baseline = pd.DataFrame(
        {
            "timestamp": timestamps,
            "head": ["short_asset", "short_boll"] * 6,
            "net_return": [0.0] * 12,
            "gross_return": [0.0] * 12,
            "simple_policy_exit_reason": ["timeout"] * 12,
        }
    )
    baseline_swapped = baseline.copy()
    baseline_swapped["head"] = baseline_swapped["head"].replace(
        {"short_asset": "short_boll", "short_boll": "short_asset"}
    )

    summary_a = pd.Series(
        {
            "net_pnl": 120.0,
            "full_sl_rate": 0.12,
            "candidate_share_hard_stop": 0.06,
            "short_asset_candidate_share_hard_stop": 0.12,
            "short_asset_candidate_mean_threshold_delta": 0.22,
            "short_asset_candidate_mean_size_multiplier": 0.42,
            "short_asset_candidate_share_capacity_reduced": 0.70,
            "short_boll_candidate_share_hard_stop": 0.03,
            "short_boll_candidate_mean_threshold_delta": 0.05,
            "short_boll_candidate_mean_size_multiplier": 0.90,
            "short_boll_candidate_share_capacity_reduced": 0.10,
        }
    )
    summary_b = summary_a.copy()
    for suffix in (
        "candidate_share_hard_stop",
        "candidate_mean_threshold_delta",
        "candidate_mean_size_multiplier",
        "candidate_share_capacity_reduced",
    ):
        a_key = f"short_asset_{suffix}"
        b_key = f"short_boll_{suffix}"
        summary_b[a_key], summary_b[b_key] = summary_a[b_key], summary_a[a_key]

    common = {
        "baseline_summary": pd.Series({"net_pnl": 0.0, "full_sl_rate": 0.10}),
        "min_trades": 0,
        "params": _params(),
    }
    obj_a = mod._objective_components(
        summary_a,
        accepted_a,
        baseline_accepted=baseline,
        **common,
    )["objective"]
    obj_b = mod._objective_components(
        summary_b,
        accepted_b,
        baseline_accepted=baseline_swapped,
        **common,
    )["objective"]

    np.testing.assert_allclose(obj_a, obj_b)
