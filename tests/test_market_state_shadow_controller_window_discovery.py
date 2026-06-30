from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.discover_market_state_shadow_controller_windows import (
    discover_bundle_dirs,
    discover_shadow_controller_windows,
    readiness_for_bundle,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_bundle(
    root: Path,
    *,
    rank_contract: str = "short_boll_timestamp_rank",
    generated_by: str = "score_market_state_controller_bundle",
    execution_enabled: bool = False,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    ts = pd.to_datetime(
        [
            "2026-06-24T09:00:00Z",
            "2026-06-24T10:00:00Z",
            "2026-06-24T11:00:00Z",
        ],
        utc=True,
    )
    _write_json(
        root / "manifest.json",
        {
            "generated_by": generated_by,
            "selected_arm": "S2_observed_forecast_shared_response",
            "controller_execution_enabled": execution_enabled,
            "shadow_controller_only": True,
            "controller_enabled_heads": [],
            "shadow_controller_enabled_heads": ["short_asset", "short_boll"],
            "controller": {
                "execution_enabled": execution_enabled,
                "controller_execution_enabled": execution_enabled,
                "shadow_controller_only": True,
                "changes_scores_or_ranks": False,
                "changes_auction_ordering": False,
            },
        },
    )
    _write_json(
        root / "market_state_feature_contract.json",
        {
            "rank_contract": rank_contract,
            "active_heads": ["short_asset", "short_boll"],
            "disabled_heads": ["long_bars", "long_dist"],
            "invariants": {
                "one_market_state_row_per_timestamp": True,
                "state_join_timestamp_constant": True,
                "market_state_uses_strategy_ids": False,
                "market_state_uses_model_predictions": False,
                "market_state_uses_ranks": False,
                "market_state_uses_candidate_counts": False,
                "market_state_uses_portfolio_pnl": False,
                "market_state_uses_realized_strategy_outcomes": False,
                "actual_order_book_features_allowed": False,
                "controller_changes_scores_or_ranks": False,
                "controller_changes_auction_ordering": False,
                "controller_can_lower_thresholds": False,
            },
        },
    )
    pd.DataFrame(
        {
            "timestamp": ts,
            "split": ["score", "score", "score"],
            "state_level": ["observed", "observed", "observed"],
            "state_shock": [0.1, 0.2, 0.3],
        }
    ).to_parquet(root / "market_state_timestamp_panel.parquet", index=False)
    schedule = pd.DataFrame(
        {
            "timestamp": ts,
            "strategy_id": ["short_asset_s1", "short_boll_s1", "short_boll_s1"],
            "head": ["short_asset", "short_boll", "short_boll"],
            "base_threshold": [0.70, 0.70, 0.70],
            "state_threshold": [0.70, 0.70, 0.70],
        }
    )
    schedule.to_parquet(root / "strategy_threshold_schedule.parquet", index=False)
    schedule.to_csv(root / "controller_schedule.csv", index=False)
    proposed = schedule.copy()
    proposed["state_threshold"] = [0.72, 0.74, 0.76]
    proposed.to_parquet(root / "shadow_controller_proposed_schedule.parquet", index=False)
    proposed.to_csv(root / "shadow_controller_proposed_schedule.csv", index=False)
    pd.DataFrame(
        {
            "scope": ["all"],
            "scope_value": ["all"],
            "schedule_rows": [3],
            "threshold_raised_count": [0],
            "mean_threshold_delta": [0.0],
            "max_threshold_delta": [0.0],
        }
    ).to_csv(root / "strategy_threshold_action_audit.csv", index=False)
    pd.DataFrame(
        {
            "scope": ["all"],
            "scope_value": ["all"],
            "schedule_rows": [3],
            "threshold_raised_count": [3],
            "mean_threshold_delta": [0.04],
            "max_threshold_delta": [0.06],
        }
    ).to_csv(root / "shadow_threshold_action_audit.csv", index=False)
    pd.DataFrame(
        {
            "scope": ["all"],
            "scope_value": ["all"],
            "suppressed_candidates": [2],
            "suppressed_loss_avoided": [1.2],
            "suppressed_winner_pnl_sacrificed": [0.4],
            "realized_defensive_success": [0.8],
            "suppressed_win_rate": [0.25],
            "suppressed_full_sl_rate": [0.5],
            "suppressed_timeout_rate": [0.0],
        }
    ).to_csv(root / "shadow_threshold_candidate_suppression_utility.csv", index=False)
    pd.DataFrame(
        {
            "arm": ["S2_observed_forecast_shared_response"],
            "trade_count": [3],
            "net_pnl": [1.1],
            "gross_pnl": [1.3],
            "cost_pnl": [0.2],
            "full_sl_rate": [0.33],
            "timeout_rate": [0.0],
        }
    ).to_csv(root / "controller_replay_summary.csv", index=False)
    pd.DataFrame(
        {
            "arm": ["S2_observed_forecast_shared_response"],
            "head": ["short_boll"],
            "trade_count": [3],
            "win_rate": [0.67],
            "net_pnl": [1.1],
            "gross_pnl": [1.3],
            "cost_pnl": [0.2],
            "full_sl_rate": [0.33],
            "timeout_rate": [0.0],
        }
    ).to_csv(root / "controller_replay_by_head.csv", index=False)
    pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": ["BTC", "ETH", "SOL"],
            "side": ["short", "short", "short"],
            "strategy_id": ["short_asset_s1", "short_boll_s1", "short_boll_s1"],
            "head": ["short_asset", "short_boll", "short_boll"],
            "net_return": [0.01, -0.01, 0.02],
        }
    ).to_parquet(root / "accepted_trades.parquet", index=False)


def test_discover_bundle_dirs_filters_scored_controller_bundles(tmp_path: Path) -> None:
    good = tmp_path / "market_state_controller_bundle_score_good"
    other = tmp_path / "other"
    _write_bundle(good)
    _write_bundle(other, generated_by="other")

    found = discover_bundle_dirs([tmp_path], include_regex="bundle")

    assert found == [good]


def test_readiness_passes_valid_shadow_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)

    row = readiness_for_bundle(
        bundle,
        existing_dirs=set(),
        min_timestamp_count=3,
        run_artifact_audit=False,
    )

    assert row["status"] == "appendable"
    assert row["failures"] == []
    assert row["timestamp_count"] == 3
    assert row["applied_noop_pass"] is True
    assert row["shadow_schedule_safe"] is True
    assert row["coverage_ok"] is True


def test_readiness_marks_existing_bundle_already_monitored(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)

    row = readiness_for_bundle(
        bundle,
        existing_dirs={str(bundle.resolve())},
        min_timestamp_count=3,
        run_artifact_audit=False,
    )

    assert row["status"] == "already_monitored"
    assert row["already_monitored"] is True


def test_discovery_reports_appendable_and_failed_candidates(tmp_path: Path) -> None:
    good = tmp_path / "market_state_controller_bundle_score_good"
    bad = tmp_path / "market_state_controller_bundle_score_bad_contract"
    existing = tmp_path / "market_state_controller_bundle_score_existing"
    _write_bundle(good)
    _write_bundle(bad, rank_contract="anchor_global_policy_rank_reference")
    _write_bundle(existing)
    monitor_dir = tmp_path / "monitor"
    monitor_dir.mkdir()
    pd.DataFrame({"bundle_dir": [str(existing)]}).to_csv(
        monitor_dir / "shadow_controller_monitor_bundles.csv",
        index=False,
    )

    summary = discover_shadow_controller_windows(
        roots=[tmp_path],
        existing_monitor_dir=monitor_dir,
        output_dir=tmp_path / "out",
        include_regex="market_state_controller_bundle_score",
        min_timestamp_count=3,
        run_artifact_audit=False,
    )

    assert summary["discovered_candidate_count"] == 3
    assert summary["appendable_candidate_count"] == 1
    assert summary["already_monitored_count"] == 1
    assert summary["failed_candidate_count"] == 1
    appendable = pd.read_csv(summary["appendable_csv"])
    assert appendable["bundle_dir"].tolist() == [str(good)]
    readiness = pd.read_csv(summary["readiness_csv"])
    bad_row = readiness.loc[readiness["bundle_dir"].eq(str(bad))].iloc[0]
    assert "rank_contract_mismatch" in bad_row["failures"]
    assert (tmp_path / "out" / "market_state_shadow_controller_window_discovery_report.md").exists()


def test_discovery_can_exclude_development_windows_by_start_cutoff(tmp_path: Path) -> None:
    old = tmp_path / "market_state_controller_bundle_score_old"
    new = tmp_path / "market_state_controller_bundle_score_new"
    _write_bundle(old)
    _write_bundle(new)
    # Move the old bundle's timestamp panel before the cutoff while preserving
    # all other valid shadow-controller files.
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-06-15T17:00:00Z",
                    "2026-06-15T18:00:00Z",
                    "2026-06-15T19:00:00Z",
                ],
                utc=True,
            ),
            "split": ["score", "score", "score"],
            "state_level": ["observed", "observed", "observed"],
            "state_shock": [0.1, 0.2, 0.3],
        }
    ).to_parquet(old / "market_state_timestamp_panel.parquet", index=False)

    summary = discover_shadow_controller_windows(
        roots=[tmp_path],
        output_dir=tmp_path / "out_cutoff",
        include_regex="market_state_controller_bundle_score",
        min_timestamp_count=3,
        min_start_after="2026-06-23T00:00:00Z",
        run_artifact_audit=False,
    )

    assert summary["appendable_candidate_count"] == 1
    assert summary["excluded_candidate_count"] == 1
    readiness = pd.read_csv(summary["readiness_csv"])
    old_row = readiness.loc[readiness["bundle_dir"].eq(str(old))].iloc[0]
    assert old_row["status"] == "excluded"
    assert "before_min_start_after" in old_row["failures"]
