from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.build_market_state_direct_suppression_ledger import (
    BASELINE_ARM,
    build_direct_suppression_ledger,
    write_direct_suppression_ledger,
)


def _write_source(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp("2026-06-24T00:00:00Z")
    accepted_rows = [
        {
            "arm": BASELINE_ARM,
            "timestamp": timestamp,
            "symbol": "BTC-PERP",
            "side": "short",
            "strategy_id": "short_asset",
            "head": "short_asset",
            "normalized_rank_score": 0.72,
            "effective_rank_score": 0.72,
            "base_strategy_threshold": 0.70,
            "deployment_rank_threshold": 0.70,
            "net_return": -0.020,
            "gross_return": -0.018,
            "net_pnl": -20.0,
            "gross_pnl": -18.0,
            "cost_pnl": 2.0,
            "simple_policy_exit_reason": "full_sl",
            "position_size": 1000.0,
        },
        {
            "arm": BASELINE_ARM,
            "timestamp": timestamp,
            "symbol": "ETH-PERP",
            "side": "short",
            "strategy_id": "short_asset",
            "head": "short_asset",
            "normalized_rank_score": 0.76,
            "effective_rank_score": 0.76,
            "base_strategy_threshold": 0.70,
            "deployment_rank_threshold": 0.70,
            "net_return": 0.010,
            "gross_return": 0.012,
            "net_pnl": 10.0,
            "gross_pnl": 12.0,
            "cost_pnl": 2.0,
            "simple_policy_exit_reason": "timeout",
            "position_size": 1000.0,
        },
        {
            "arm": BASELINE_ARM,
            "timestamp": timestamp,
            "symbol": "SOL-PERP",
            "side": "short",
            "strategy_id": "short_asset",
            "head": "short_asset",
            "normalized_rank_score": 0.95,
            "effective_rank_score": 0.95,
            "base_strategy_threshold": 0.70,
            "deployment_rank_threshold": 0.70,
            "net_return": -0.030,
            "gross_return": -0.028,
            "net_pnl": -30.0,
            "gross_pnl": -28.0,
            "cost_pnl": 2.0,
            "simple_policy_exit_reason": "full_sl",
            "position_size": 1000.0,
        },
    ]
    schedule_rows = [
        {
            "arm": "S1_observed_axes_shared_response__post_selection_overlay",
            "timestamp": timestamp,
            "strategy_id": "short_asset",
            "head": "short_asset",
            "base_threshold": 0.70,
            "state_threshold": 0.75,
            "raw_state_threshold": 0.75,
            "controller_mode": "threshold_raise_only",
            "threshold_action_enabled": True,
            "force_base_threshold": False,
            "risk_severity": 0.8,
            "controller_reason": "synthetic_test",
            "prediction_coverage": 1.0,
            "state_ood_share": 0.0,
            "mean_pred_utility": -0.02,
            "mean_pred_full_sl": 0.5,
            "mean_pred_timeout": 0.1,
            "base_candidate_count": 3,
            "frontier_candidate_count": 2,
            "frontier_upper_rank": 0.80,
            "accepted_frontier_key_filter_active": True,
            "accepted_frontier_candidate_count": 2,
            "accepted_frontier_suppressed_count": 1,
            "predicted_removed_loss_avoided": 0.02,
            "predicted_removed_winner_sacrificed": 0.0,
            "predicted_action_edge": 0.02,
            "fold": 0,
        }
    ]
    pd.DataFrame(accepted_rows).to_parquet(root / "accepted_trades.parquet", index=False)
    pd.DataFrame(schedule_rows).to_parquet(
        root / "strategy_threshold_schedule.parquet",
        index=False,
    )


def _write_later_shadow_score_source(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp("2026-06-25T00:00:00Z")
    accepted_rows = [
        {
            "arm": "S1_observed_axes_shared_response__post_selection_overlay",
            "timestamp": timestamp,
            "symbol": "BTC-PERP",
            "side": "short",
            "strategy_id": "short_boll",
            "head": "short_boll",
            "normalized_rank_score": 0.72,
            "effective_rank_score": 0.72,
            "net_return": -0.020,
            "gross_return": -0.018,
            "net_pnl": -20.0,
            "gross_pnl": -18.0,
            "cost_pnl": 2.0,
            "simple_policy_exit_reason": "full_sl",
            "position_size": 1000.0,
        },
        {
            "arm": "S1_observed_axes_shared_response__post_selection_overlay",
            "timestamp": timestamp,
            "symbol": "ETH-PERP",
            "side": "short",
            "strategy_id": "short_boll",
            "head": "short_boll",
            "normalized_rank_score": 0.77,
            "effective_rank_score": 0.77,
            "net_return": 0.012,
            "gross_return": 0.014,
            "net_pnl": 12.0,
            "gross_pnl": 14.0,
            "cost_pnl": 2.0,
            "simple_policy_exit_reason": "trailing",
            "position_size": 1000.0,
        },
    ]
    schedule_rows = [
        {
            "timestamp": timestamp,
            "strategy_id": "short_boll",
            "head": "short_boll",
            "base_threshold": 0.70,
            "state_threshold": 0.75,
            "raw_state_threshold": 0.75,
            "controller_mode": "threshold_raise_only",
            "threshold_action_enabled": True,
            "force_base_threshold": False,
            "risk_severity": 0.9,
            "controller_reason": "single_arm_later_window",
            "prediction_coverage": 1.0,
            "state_ood_share": 0.0,
            "mean_pred_utility": -0.02,
            "mean_pred_full_sl": 0.5,
            "mean_pred_timeout": 0.1,
            "base_candidate_count": 2,
            "frontier_candidate_count": 2,
            "frontier_upper_rank": 0.80,
            "predicted_removed_loss_avoided": 0.02,
            "predicted_removed_winner_sacrificed": 0.0,
            "predicted_action_edge": 0.02,
        }
    ]
    pd.DataFrame(accepted_rows).to_parquet(root / "accepted_trades.parquet", index=False)
    pd.DataFrame(schedule_rows).to_parquet(
        root / "strategy_threshold_schedule.parquet",
        index=False,
    )


def test_direct_suppression_ledger_uses_baseline_accepted_frontier_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_source(source)

    ledger, by_group, summary = build_direct_suppression_ledger(source)

    assert summary["artifact_contract"] == "direct_accepted_frontier_training_ledger_v1"
    assert summary["row_count"] == 2
    assert summary["baseline_accepted_rows"] == 3
    assert summary["current_schedule_suppressed_rows"] == 1
    assert abs(summary["current_schedule_defensive_utility"] - 0.020) <= 1e-12
    assert set(ledger["symbol"]) == {"BTC-PERP", "ETH-PERP"}

    loser = ledger.loc[ledger["symbol"].eq("BTC-PERP")].iloc[0]
    winner = ledger.loc[ledger["symbol"].eq("ETH-PERP")].iloc[0]
    assert bool(loser["direct_suppression_profitable"])
    assert bool(loser["would_suppress_at_state_threshold"])
    assert bool(loser["direct_suppression_full_sl"])
    assert abs(float(loser["loss_avoided_if_suppressed"]) - 0.020) <= 1e-12
    assert not bool(winner["direct_suppression_profitable"])
    assert not bool(winner["would_suppress_at_state_threshold"])
    assert bool(winner["direct_suppression_timeout"])
    assert abs(float(winner["winner_pnl_sacrificed_if_suppressed"]) - 0.010) <= 1e-12

    grouped = by_group.iloc[0]
    assert int(grouped["frontier_rows"]) == 2
    assert int(grouped["current_schedule_suppressed_rows"]) == 1
    assert abs(float(grouped["current_schedule_defensive_utility"]) - 0.020) <= 1e-12


def test_direct_suppression_ledger_writes_auditable_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    _write_source(source)
    ledger, by_group, summary = build_direct_suppression_ledger(source)

    outputs = write_direct_suppression_ledger(ledger, by_group, summary, output)

    for path in outputs.values():
        assert Path(path).exists()
    payload = json.loads(Path(outputs["summary_json"]).read_text())
    assert payload["artifact_contract"] == "direct_accepted_frontier_training_ledger_v1"
    assert payload["outputs"]["ledger_parquet"].endswith(
        "direct_accepted_frontier_training_ledger.parquet"
    )
    report = Path(outputs["report_md"]).read_text()
    assert "baseline-accepted frontier rows" in report
    assert "Does not change scores, ranks, thresholds" in report


def test_direct_suppression_ledger_supports_later_single_arm_score_dirs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "later_score"
    _write_later_shadow_score_source(source)

    ledger, by_group, summary = build_direct_suppression_ledger(
        source,
        accepted_arm_mode="all_accepted_as_baseline",
        controller_arm_fallback="S1_observed_axes_shared_response__post_selection_overlay",
        source_kind="later_shadow",
        source_window_id="jun24_09_jun25_08",
    )

    assert summary["accepted_arm_mode"] == "all_accepted_as_baseline"
    assert summary["controller_arm_source"] == "fallback_controller_arm"
    assert summary["source_kind"] == "later_shadow"
    assert summary["source_window_id"] == "jun24_09_jun25_08"
    assert summary["row_count"] == 2
    assert set(ledger["controller_arm"]) == {
        "S1_observed_axes_shared_response__post_selection_overlay"
    }
    assert set(ledger["source_accepted_arm"]) == {
        "S1_observed_axes_shared_response__post_selection_overlay"
    }
    assert set(ledger["source_kind"]) == {"later_shadow"}
    assert int(ledger["would_suppress_at_state_threshold"].sum()) == 1
    assert int(by_group["current_schedule_suppressed_rows"].sum()) == 1
