from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import diagnose_market_state_no_backfill_failures as diagnostics
from scripts import score_market_state_controller_bundle as scorer


def test_direct_threshold_only_overlay_removes_only_rows_below_state_threshold() -> None:
    baseline_accepted = pd.DataFrame(
        [
            {
                "timestamp": "2026-06-25 12:00:00+00:00",
                "symbol": "WIN/USD:USD",
                "side": "short",
                "strategy_id": "short_boll_strategy",
                "head": "short_boll",
                "normalized_rank_score": 0.78,
                "dynamic_threshold": 0.77,
                "net_pnl": 20.0,
            },
            {
                "timestamp": "2026-06-25 16:00:00+00:00",
                "symbol": "LOSS/USD:USD",
                "side": "short",
                "strategy_id": "short_boll_strategy",
                "head": "short_boll",
                "normalized_rank_score": 0.74,
                "dynamic_threshold": 0.73,
                "net_pnl": -10.0,
            },
            {
                "timestamp": "2026-06-25 16:00:00+00:00",
                "symbol": "COMMON/USD:USD",
                "side": "short",
                "strategy_id": "short_boll_strategy",
                "head": "short_boll",
                "normalized_rank_score": 0.91,
                "dynamic_threshold": 0.73,
                "net_pnl": 3.0,
            },
        ]
    )
    schedule = pd.DataFrame(
        [
            {
                "timestamp": "2026-06-25 12:00:00+00:00",
                "strategy_id": "short_boll_strategy",
                "head": "short_boll",
                "base_threshold": 0.70,
                "state_threshold": 0.73,
            },
            {
                "timestamp": "2026-06-25 16:00:00+00:00",
                "strategy_id": "short_boll_strategy",
                "head": "short_boll",
                "base_threshold": 0.70,
                "state_threshold": 0.76,
            },
        ]
    )

    accepted, delta, summary = scorer._direct_threshold_only_overlay(
        baseline_accepted,
        schedule,
    )

    assert summary["direct_threshold_only"] is True
    assert summary["no_path_or_capacity_replay"] is True
    assert summary["direct_threshold_removed_count"] == 1
    assert summary["removed_loss_avoided"] == pytest.approx(10.0)
    assert summary["removed_winner_pnl_sacrificed"] == pytest.approx(0.0)
    assert summary["accepted_delta_defensive_success"] == pytest.approx(10.0)
    assert set(accepted["symbol"]) == {"WIN/USD:USD", "COMMON/USD:USD"}
    removed = delta.loc[delta["delta_action"].eq("removed_by_shadow_no_backfill")]
    assert removed["symbol"].tolist() == ["LOSS/USD:USD"]
    assert removed["direct_threshold_removed"].fillna(False).astype(bool).tolist() == [True]
    assert removed["direct_effective_threshold"].tolist() == pytest.approx([0.76])


def _write_score_dir(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    candidates = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-06-25 12:00:00+00:00",
                    "2026-06-25 16:00:00+00:00",
                ],
                utc=True,
            )
        }
    )
    candidates_path = root / "eval_candidates.parquet"
    candidates.to_parquet(candidates_path, index=False)

    pd.DataFrame(
        [
            {
                "timestamp": "2026-06-25 12:00:00+00:00",
                "symbol": "WIN/USD:USD",
                "side": "short",
                "strategy_id": "short_boll_strategy",
                "head": "short_boll",
                "normalized_rank_score": 0.78,
                "base_threshold": 0.70,
                "dynamic_threshold": 0.77,
                "net_pnl": 20.0,
                "simple_policy_exit_reason": "trailing",
                "delta_action": "removed_by_shadow_no_backfill",
            },
            {
                "timestamp": "2026-06-25 16:00:00+00:00",
                "symbol": "LOSS/USD:USD",
                "side": "short",
                "strategy_id": "short_boll_strategy",
                "head": "short_boll",
                "normalized_rank_score": 0.74,
                "base_threshold": 0.70,
                "dynamic_threshold": 0.73,
                "net_pnl": -10.0,
                "simple_policy_exit_reason": "full_sl",
                "delta_action": "removed_by_shadow_no_backfill",
            },
            {
                "timestamp": "2026-06-25 16:00:00+00:00",
                "symbol": "COMMON/USD:USD",
                "side": "short",
                "strategy_id": "short_boll_strategy",
                "head": "short_boll",
                "normalized_rank_score": 0.91,
                "base_threshold": 0.70,
                "dynamic_threshold": 0.73,
                "net_pnl": 3.0,
                "simple_policy_exit_reason": "trailing",
                "delta_action": "common_accepted",
            },
        ]
    ).to_csv(root / "shadow_no_backfill_accepted_trade_delta.csv", index=False)

    pd.DataFrame(
        [
            {
                "timestamp": "2026-06-25 12:00:00+00:00",
                "strategy_id": "short_boll_strategy",
                "head": "short_boll",
                "base_threshold": 0.70,
                "state_threshold": 0.73,
                "risk_severity": 0.5,
                "controller_reason": "rank_grid_scaled_no_feasible",
                "predicted_action_edge": 0.0,
                "suppressed_candidate_count": 0,
                "frontier_candidate_count": 3,
            },
            {
                "timestamp": "2026-06-25 16:00:00+00:00",
                "strategy_id": "short_boll_strategy",
                "head": "short_boll",
                "base_threshold": 0.70,
                "state_threshold": 0.76,
                "risk_severity": 1.0,
                "controller_reason": "rank_grid_scaled_no_feasible",
                "predicted_action_edge": 0.05,
                "suppressed_candidate_count": 2,
                "frontier_candidate_count": 2,
            },
        ]
    ).to_csv(root / "shadow_controller_proposed_schedule.csv", index=False)

    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-06-25 12:00:00+00:00",
                    "2026-06-25 16:00:00+00:00",
                ],
                utc=True,
            ),
            "state_shock_down": [0.2, 0.9],
            "state_liquidity_stress_proxy": [0.3, 0.8],
        }
    ).to_parquet(root / "market_state_timestamp_panel.parquet", index=False)

    manifest = {
        "eval_candidates": str(candidates_path),
        "selected_arm": "S1_observed_axes_shared_response__post_selection_overlay",
        "rank_contract": "anchor_global_policy_rank_reference",
        "source_contract_audit": {"overall_passed": True},
        "shadow_no_backfill_replay_summary": {
            "trade_count": 1,
            "net_pnl": -7.0,
            "full_sl_rate": 1.0,
            "timeout_rate": 0.0,
        },
        "shadow_no_backfill_accepted_delta_summary": {
            "baseline_trade_count": 3,
            "shadow_trade_count": 1,
            "removed_trade_count": 2,
            "added_trade_count": 0,
            "baseline_net_pnl": 13.0,
            "shadow_net_pnl": -7.0,
            "total_net_pnl_delta": -20.0,
            "common_net_pnl_delta": 0.0,
            "removed_loss_avoided": 10.0,
            "removed_winner_pnl_sacrificed": 20.0,
            "accepted_delta_defensive_success": -10.0,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_failure_diagnostics_separates_direct_and_indirect_suppression(
    tmp_path: Path,
) -> None:
    score_dir = tmp_path / "score"
    out = tmp_path / "diagnostics"
    _write_score_dir(score_dir)

    summary = diagnostics.build_diagnostics([score_dir], out)

    assert summary["removed_trade_count"] == 2
    assert summary["direct_removed_count"] == 1
    assert summary["indirect_removed_count"] == 1
    assert summary["direct_loss_avoided"] == pytest.approx(10.0)
    assert summary["direct_defensive_success"] == pytest.approx(10.0)
    assert summary["indirect_winner_pnl_sacrificed"] == pytest.approx(20.0)
    assert summary["indirect_defensive_success"] == pytest.approx(-20.0)
    assert summary["promotion_safe_subset_found"] is False
    assert "direct_threshold_counterfactual_positive_but_full_replay_negative" in summary[
        "failure_modes"
    ]
    assert "indirect_suppression_removed_winners" in summary["failure_modes"]

    removed = pd.read_csv(out / "no_backfill_removed_trade_diagnostics.csv")
    assert set(removed["direct_threshold_suppression"].astype(bool)) == {True, False}
    assert (out / "no_backfill_failure_diagnostics_summary.json").exists()
    assert (out / "no_backfill_failure_diagnostics_report.md").exists()
