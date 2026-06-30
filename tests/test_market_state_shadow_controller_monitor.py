from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import report_market_state_shadow_controller_monitor as monitor


def _write_shadow_bundle(
    root: Path,
    *,
    applied_raise: bool = False,
    defensive_success: float = 0.8,
    loss_avoided: float = 1.2,
    winner_sacrificed: float = 0.4,
    include_suppression: bool = True,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    ts = pd.to_datetime(["2026-06-15 01:00:00+00:00", "2026-06-15 02:00:00+00:00"])
    manifest = {
        "generated_by": "score_market_state_controller_bundle",
        "selected_arm": "S2_observed_forecast_shared_response",
        "controller_execution_enabled": False,
        "controller_enabled_heads": [],
        "controller_enabled_scope": "disabled_by_activation_registry",
        "shadow_controller_only": True,
        "shadow_controller_enabled_heads": ["short_asset", "short_boll"],
        "shadow_controller_enabled_scope": "all_active_heads",
        "controller": {
            "execution_enabled": False,
            "controller_execution_enabled": False,
            "shadow_controller_only": True,
            "changes_scores_or_ranks": False,
            "changes_auction_ordering": False,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pd.DataFrame(
        {
            "split": ["score", "score"],
            "state_level": ["observed", "observed"],
            "timestamp": ts,
            "state_shock": [0.1, 0.2],
        }
    ).to_parquet(root / "market_state_timestamp_panel.parquet", index=False)
    schedule = pd.DataFrame(
        {
            "timestamp": ts,
            "strategy_id": ["s_asset", "s_boll"],
            "head": ["short_asset", "short_boll"],
            "base_threshold": [0.70, 0.70],
            "state_threshold": [0.74 if applied_raise else 0.70, 0.70],
            "raw_state_threshold": [0.74 if applied_raise else 0.70, 0.70],
        }
    )
    schedule.to_parquet(root / "strategy_threshold_schedule.parquet", index=False)
    schedule.to_csv(root / "controller_schedule.csv", index=False)
    proposed = schedule.copy()
    proposed["state_threshold"] = [0.76, 0.78]
    proposed["raw_state_threshold"] = [0.76, 0.78]
    proposed["arm"] = "S2_observed_forecast_shared_response__shadow_proposed"
    proposed.to_parquet(root / "shadow_controller_proposed_schedule.parquet", index=False)
    proposed.to_csv(root / "shadow_controller_proposed_schedule.csv", index=False)
    pd.DataFrame(
        {
            "scope": ["all"],
            "scope_value": ["all"],
            "schedule_rows": [2],
            "threshold_raised_count": [1 if applied_raise else 0],
            "mean_threshold_delta": [0.02 if applied_raise else 0.0],
            "max_threshold_delta": [0.04 if applied_raise else 0.0],
            "force_base_count": [1 if applied_raise else 2],
            "force_base_share": [0.5 if applied_raise else 1.0],
        }
    ).to_csv(root / "strategy_threshold_action_audit.csv", index=False)
    pd.DataFrame(
        {
            "scope": ["all"],
            "scope_value": ["all"],
            "schedule_rows": [2],
            "threshold_raised_count": [2],
            "mean_threshold_delta": [0.07],
            "max_threshold_delta": [0.08],
            "force_base_count": [0],
            "force_base_share": [0.0],
        }
    ).to_csv(root / "shadow_threshold_action_audit.csv", index=False)
    if include_suppression:
        pd.DataFrame(
            {
                "arm": [
                    "S2_observed_forecast_shared_response__shadow_proposed",
                    "S2_observed_forecast_shared_response__shadow_proposed",
                ],
                "scope": ["all", "head"],
                "scope_value": ["all", "short_boll"],
                "suppressed_candidates": [3, 3],
                "raised_schedule_count": [2, 2],
                "mean_threshold_delta": [0.07, 0.07],
                "suppressed_loss_avoided": [loss_avoided, loss_avoided],
                "suppressed_winner_pnl_sacrificed": [winner_sacrificed, winner_sacrificed],
                "realized_defensive_success": [defensive_success, defensive_success],
                "realized_defensive_success_per_candidate": [defensive_success / 3.0, defensive_success / 3.0],
                "suppressed_win_rate": [0.33, 0.33],
                "suppressed_full_sl_rate": [0.67, 0.67],
                "suppressed_timeout_rate": [0.0, 0.0],
            }
        ).to_csv(root / "shadow_threshold_candidate_suppression_utility.csv", index=False)
    pd.DataFrame(
        {
            "arm": ["S2_observed_forecast_shared_response"],
            "trade_count": [2],
            "net_pnl": [0.8],
            "gross_pnl": [1.0],
            "cost_pnl": [0.2],
            "full_sl_rate": [0.5],
            "timeout_rate": [0.0],
            "mean_threshold_delta": [0.0],
            "max_threshold_delta": [0.0],
            "share_threshold_raised": [0.0],
        }
    ).to_csv(root / "controller_replay_summary.csv", index=False)
    pd.DataFrame(
        {
            "arm": ["S2_observed_forecast_shared_response"],
            "head": ["short_boll"],
            "trade_count": [2],
            "win_rate": [0.5],
            "net_pnl": [0.8],
            "gross_pnl": [1.0],
            "cost_pnl": [0.2],
            "full_sl_rate": [0.5],
            "timeout_rate": [0.0],
        }
    ).to_csv(root / "controller_replay_by_head.csv", index=False)
    pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": ["BTC_USD", "ETH_USD"],
            "side": ["short", "short"],
            "strategy_id": ["s_asset", "s_boll"],
            "head": ["short_asset", "short_boll"],
            "net_pnl": [1.0, -0.2],
        }
    ).to_parquet(root / "accepted_trades.parquet", index=False)


def test_shadow_monitor_aggregates_positive_defensive_success(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_shadow_bundle(bundle)

    result = monitor.aggregate_shadow_bundles([bundle], run_artifact_audit=False)
    row = result["bundles"].iloc[0]

    assert bool(row["applied_noop_pass"]) is True
    assert bool(row["coverage_ok"]) is True
    assert bool(row["defensive_positive"]) is True
    assert result["rollup"]["total_shadow_suppressed_candidates"] == 3.0
    assert result["rollup"]["total_shadow_realized_defensive_success"] == 0.8
    assert result["rollup"]["weighted_shadow_defensive_success_per_candidate"] == 0.8 / 3.0
    assert result["rollup"]["shadow_promotion_gate_passed"] is True
    assert result["rollup"]["shadow_promotion_failures"] == []
    assert result["rollup"]["controller_should_remain_disabled"] is True
    assert result["by_head"].loc[0, "shadow_realized_defensive_success"] == 0.8


def test_shadow_monitor_flags_applied_raise_in_disabled_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_shadow_bundle(bundle, applied_raise=True)

    result = monitor.aggregate_shadow_bundles([bundle], run_artifact_audit=False)
    row = result["bundles"].iloc[0]

    assert bool(row["applied_noop_pass"]) is False
    assert result["rollup"]["applied_parity_failures"] == 1
    assert result["rollup"]["shadow_promotion_gate_passed"] is False
    assert "applied_noop_parity_failed" in result["rollup"]["shadow_promotion_failures"]


def test_shadow_monitor_flags_negative_defensive_success(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_shadow_bundle(
        bundle,
        defensive_success=-0.1,
        loss_avoided=0.3,
        winner_sacrificed=0.4,
    )

    result = monitor.aggregate_shadow_bundles([bundle], run_artifact_audit=False)
    row = result["bundles"].iloc[0]

    assert bool(row["coverage_ok"]) is True
    assert bool(row["defensive_positive"]) is False
    assert result["rollup"]["defensive_positive_bundle_share"] == 0.0
    assert result["rollup"]["shadow_promotion_gate_passed"] is False
    assert "defensive_success_not_positive" in result["rollup"]["shadow_promotion_failures"]
    assert (
        "loss_avoided_not_greater_than_winner_pnl_sacrificed"
        in result["rollup"]["shadow_promotion_failures"]
    )
    assert "insufficient_defensive_positive_bundle_share" in result["rollup"]["shadow_promotion_failures"]


def test_shadow_monitor_missing_suppression_utility_is_coverage_failure(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_shadow_bundle(bundle, include_suppression=False)

    result = monitor.aggregate_shadow_bundles([bundle], run_artifact_audit=False)
    row = result["bundles"].iloc[0]

    assert bool(row["coverage_ok"]) is False
    assert bool(row["defensive_positive"]) is False
    assert result["rollup"]["coverage_failures"] == 1
    assert result["rollup"]["shadow_promotion_gate_passed"] is False
    assert "coverage_failed" in result["rollup"]["shadow_promotion_failures"]
    assert "no_shadow_suppression" in result["rollup"]["shadow_promotion_failures"]


def test_shadow_monitor_writes_report_files(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    out = tmp_path / "report"
    _write_shadow_bundle(bundle)
    result = monitor.aggregate_shadow_bundles([bundle], run_artifact_audit=False)

    monitor.write_report(result, out)

    assert (out / "shadow_controller_monitor_summary.json").exists()
    assert (out / "shadow_controller_monitor_bundles.csv").exists()
    assert (out / "shadow_controller_monitor_by_head.csv").exists()
    report = (out / "shadow_controller_monitor_report.md").read_text(encoding="utf-8")
    assert "monitoring-only" in report
    assert "Shadow promotion gate passed" in report
