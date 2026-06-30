from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.portfolio_policy_replay import (
    PortfolioPolicyParams,
    replay_candidates,
)
from scripts.run_market_state_head_priority_modulation import (
    apply_head_priority_schedule,
    build_head_priority_schedule,
)


def _candidate_rows() -> pd.DataFrame:
    ts = pd.Timestamp("2026-06-20T00:00:00Z")
    return pd.DataFrame(
        {
            "timestamp": [ts, ts],
            "symbol": ["AAA/USD:USD", "BBB/USD:USD"],
            "side": ["short", "short"],
            "strategy_id": ["short_asset_s1", "short_boll_s1"],
            "head": ["short_asset", "short_boll"],
            "strategy_rank_pct": [0.80, 0.80],
            "normalized_rank_score": [0.80, 0.80],
            "base_strategy_threshold": [0.70, 0.70],
            "calibrated_score": [0.60, 0.60],
            "entry_price": [100.0, 100.0],
            "exit_timestamp": [ts + pd.Timedelta(hours=1), ts + pd.Timedelta(hours=1)],
            "exit_price": [99.0, 99.0],
            "net_return": [0.01, 0.02],
            "gross_return": [0.012, 0.022],
            "holding_bars": [4, 4],
            "simple_policy_exit_reason": ["tp", "tp"],
            "fees_bps": [1.0, 1.0],
            "slippage_bps": [0.0, 0.0],
            "price_gap_bps": [0.0, 0.0],
            "expected_friction_bps": [0.0, 0.0],
        }
    )


def test_portfolio_priority_adjustment_changes_auction_order_only() -> None:
    candidates = _candidate_rows()
    adjusted = candidates.copy()
    adjusted["portfolio_priority_adjustment"] = [0.0, 0.50]
    params = PortfolioPolicyParams(
        max_concurrent_positions=1,
        max_concurrent_per_side=1,
        max_concurrent_per_strategy=1,
        max_new_entries_per_bar=1,
        global_threshold_floor=0.70,
        occupancy_threshold_alpha=0.0,
        cooldown_hours_after_loss=0.0,
    )

    decisions, _equity, _metrics = replay_candidates(adjusted, params, mode="global_auction")
    accepted = decisions.loc[decisions["accepted"]]

    assert len(accepted) == 1
    assert accepted.iloc[0]["strategy_id"] == "short_boll_s1"
    assert float(accepted.iloc[0]["normalized_rank_score"]) == pytest.approx(0.80)
    assert float(accepted.iloc[0]["dynamic_threshold"]) == pytest.approx(0.70)
    assert float(accepted.iloc[0]["portfolio_priority"]) > 0.50


def test_portfolio_priority_multiplier_changes_auction_order_only() -> None:
    candidates = _candidate_rows()
    adjusted = candidates.copy()
    adjusted["portfolio_priority_multiplier"] = [1.0, 2.0]
    params = PortfolioPolicyParams(
        max_concurrent_positions=1,
        max_concurrent_per_side=1,
        max_concurrent_per_strategy=1,
        max_new_entries_per_bar=1,
        global_threshold_floor=0.70,
        occupancy_threshold_alpha=0.0,
        cooldown_hours_after_loss=0.0,
    )

    decisions, _equity, _metrics = replay_candidates(adjusted, params, mode="global_auction")
    accepted = decisions.loc[decisions["accepted"]]

    assert len(accepted) == 1
    assert accepted.iloc[0]["strategy_id"] == "short_boll_s1"
    assert float(accepted.iloc[0]["normalized_rank_score"]) == pytest.approx(0.80)
    assert float(accepted.iloc[0]["dynamic_threshold"]) == pytest.approx(0.70)


def test_portfolio_rank_adjustment_changes_pre_filter_eligibility() -> None:
    candidates = _candidate_rows().iloc[[1]].copy()
    candidates["normalized_rank_score"] = [0.66]
    candidates["strategy_rank_pct"] = [0.66]
    params = PortfolioPolicyParams(
        max_concurrent_positions=1,
        max_concurrent_per_side=1,
        max_concurrent_per_strategy=1,
        max_new_entries_per_bar=1,
        global_threshold_floor=0.70,
        occupancy_threshold_alpha=0.0,
        cooldown_hours_after_loss=0.0,
    )

    baseline_decisions, _equity, _metrics = replay_candidates(candidates, params, mode="global_auction")
    assert not bool(baseline_decisions.iloc[0]["accepted"])
    assert baseline_decisions.iloc[0]["rejection_reason"] == "below_dynamic_threshold"

    adjusted = candidates.copy()
    adjusted["portfolio_rank_adjustment"] = [0.08]
    decisions, _equity, _metrics = replay_candidates(adjusted, params, mode="global_auction")

    assert bool(decisions.iloc[0]["accepted"])
    assert float(decisions.iloc[0]["normalized_rank_score"]) == pytest.approx(0.66)
    assert float(decisions.iloc[0]["effective_rank_score"]) == pytest.approx(0.74)
    assert float(decisions.iloc[0]["dynamic_threshold"]) == pytest.approx(0.70)


def test_build_head_priority_schedule_is_timestamp_centered() -> None:
    predictions = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-06-20T00:00:00Z",
                    "2026-06-20T00:00:00Z",
                    "2026-06-20T01:00:00Z",
                    "2026-06-20T01:00:00Z",
                ],
                utc=True,
            ),
            "head": ["short_asset", "short_boll", "short_asset", "short_boll"],
            "_rank": [0.9, 0.9, 0.9, 0.9],
            "pred_lcb_utility": [0.01, 0.03, 0.04, 0.02],
        }
    )

    schedule = build_head_priority_schedule(
        predictions,
        arm="P1_lcb_priority",
        min_rank=0.70,
        max_adjustment=0.20,
        max_priority_multiplier=1.5,
        max_rank_adjustment=0.06,
        priority_action="both",
    )

    assert set(schedule["head"]) == {"short_asset", "short_boll"}
    assert schedule["portfolio_priority_adjustment"].abs().max() <= 0.20 + 1e-12
    assert schedule["portfolio_priority_multiplier"].between(1.0 / 1.5, 1.5).all()
    assert schedule["portfolio_rank_adjustment"].abs().max() <= 0.06 + 1e-12
    centered_mean = schedule.groupby("timestamp")["centered_head_score"].mean()
    assert np.allclose(centered_mean.to_numpy(dtype=float), 0.0)
    first_ts = pd.Timestamp("2026-06-20T00:00:00Z")
    first = schedule.loc[schedule["timestamp"].eq(first_ts)].set_index("head")
    assert first.loc["short_boll", "portfolio_priority_adjustment"] > 0.0
    assert first.loc["short_asset", "portfolio_priority_adjustment"] < 0.0
    assert first.loc["short_boll", "portfolio_rank_adjustment"] > 0.0
    assert first.loc["short_asset", "portfolio_rank_adjustment"] < 0.0


def test_apply_head_priority_schedule_fails_closed_on_missing_head_timestamp() -> None:
    candidates = _candidate_rows()
    schedule = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-06-20T00:00:00Z")],
            "head": ["short_asset"],
            "portfolio_priority_adjustment": [0.1],
            "priority_arm": ["P1_lcb_priority"],
        }
    )

    with pytest.raises(ValueError, match="missing priority schedule values"):
        apply_head_priority_schedule(candidates, schedule, fail_closed=True)

    out, coverage = apply_head_priority_schedule(candidates, schedule, fail_closed=False)
    assert int(coverage["missing_rows"]) == 1
    assert out["portfolio_priority_adjustment"].tolist() == [0.1, 0.0]
    assert out["portfolio_rank_adjustment"].tolist() == [0.0, 0.0]
