from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.discover_market_state_no_backfill_shadow_windows import (
    discover_no_backfill_shadow_windows,
    readiness_for_score_dir,
)


HEX = "a" * 64
ARM = "S1_observed_axes_shared_response__post_selection_overlay"


def _write_csv(path: Path, rows: list[dict[str, object]] | None = None) -> None:
    pd.DataFrame(rows or [{}]).to_csv(path, index=False)


def _write_score_dir(root: Path, *, period_start: str = "2026-06-26T09:00:00Z") -> None:
    root.mkdir(parents=True, exist_ok=True)
    timestamps = pd.date_range(period_start, periods=4, freq="h", tz="UTC")
    eval_candidates = root / "eval_candidates.parquet"
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["BTC-PERP", "ETH-PERP", "SOL-PERP", "XRP-PERP"],
        }
    ).to_parquet(eval_candidates, index=False)
    for name in (
        "market_state_timestamp_panel.parquet",
        "accepted_trades.parquet",
        "shadow_no_backfill_scored_candidates.parquet",
        "shadow_no_backfill_decisions.parquet",
        "shadow_no_backfill_accepted_trades.parquet",
    ):
        pd.DataFrame({"timestamp": timestamps[:1]}).to_parquet(root / name, index=False)
    for name in (
        "market_state_feature_coverage.csv",
        "strategy_threshold_action_audit.csv",
        "controller_replay_by_head.csv",
        "shadow_no_backfill_replay_by_head.csv",
        "shadow_no_backfill_accepted_trade_delta.csv",
        "shadow_direct_threshold_only_summary.csv",
        "shadow_direct_threshold_only_accepted_trade_delta.csv",
    ):
        _write_csv(root / name)
    _write_csv(
        root / "controller_replay_summary.csv",
        [
            {
                "net_pnl": 10.0,
                "max_drawdown": -0.01,
                "worst_24h_net_pnl": -5.0,
                "full_sl_rate": 0.2,
                "timeout_rate": 0.1,
            }
        ],
    )
    _write_csv(
        root / "shadow_no_backfill_replay_summary.csv",
        [
            {
                "net_pnl": 11.0,
                "max_drawdown": -0.01,
                "worst_24h_net_pnl": -4.0,
                "full_sl_rate": 0.2,
                "timeout_rate": 0.1,
            }
        ],
    )
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "strategy_id": ["short_asset"] * len(timestamps),
            "base_threshold": [0.70] * len(timestamps),
            "state_threshold": [0.72] * len(timestamps),
        }
    ).to_parquet(root / "strategy_threshold_schedule.parquet", index=False)
    (root / "strategy_threshold_controller_config.json").write_text("{}\n", encoding="utf-8")
    (root / "market_state_feature_contract.json").write_text(
        json.dumps({"rank_contract": "anchor_global_policy_rank_reference"}) + "\n",
        encoding="utf-8",
    )
    output_keys = [
        "controller_scored_candidates",
        "controller_predictions",
        "market_state_timestamp_panel",
        "market_state_feature_coverage",
        "strategy_threshold_schedule",
        "strategy_threshold_action_audit",
        "strategy_threshold_controller_config",
        "market_state_feature_contract",
        "controller_replay_summary",
        "controller_replay_by_head",
        "shadow_no_backfill_scored_candidates",
        "shadow_no_backfill_decisions",
        "shadow_no_backfill_accepted_trades",
        "shadow_no_backfill_replay_summary",
        "shadow_no_backfill_replay_by_head",
        "shadow_no_backfill_accepted_trade_delta",
    ]
    manifest = {
        "generated_by": "score_market_state_controller_bundle",
        "score_manifest_contract_version": "market_state_controller_score_manifest_v2",
        "selected_arm": ARM,
        "rank_contract": "anchor_global_policy_rank_reference",
        "active_heads": ["short_asset", "short_boll"],
        "disabled_heads": ["long_bars", "long_dist"],
        "controller_execution_enabled": False,
        "shadow_controller_only": True,
        "controller": {
            "changes_scores_or_ranks": False,
            "changes_auction_ordering": False,
        },
        "shadow_no_backfill_replay_available": True,
        "eval_candidates": str(eval_candidates),
        "bundle_sha256": HEX,
        "policy_manifest_sha256": HEX,
        "eval_candidates_sha256": HEX,
        "train_deployable_candidates_sha256": HEX,
        "output_sha256": {key: HEX for key in output_keys},
        "shadow_no_backfill_accepted_delta_summary": {
            "available": True,
            "baseline_trade_count": 4,
            "shadow_trade_count": 3,
            "total_net_pnl_delta": 1.0,
            "action_only_fixed_common_size_net_pnl_delta": 1.0,
            "path_dependent_common_trade_net_pnl_delta": 0.0,
            "removed_trade_count": 1,
            "added_trade_count": 0,
            "removed_loss_avoided": 1.0,
            "removed_winner_pnl_sacrificed": 0.0,
            "accepted_delta_defensive_success": 1.0,
            "key_columns": ["timestamp", "symbol", "strategy_id", "head", "side"],
        },
        "shadow_direct_threshold_only_delta_summary": {
            "available": True,
            "removed_trade_count": 1,
            "total_net_pnl_delta": 1.0,
            "removed_loss_avoided": 1.0,
            "winner_pnl_sacrificed": 0.0,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def test_no_backfill_shadow_window_discovery_marks_appendable(tmp_path: Path) -> None:
    score_dir = tmp_path / "market_state_controller_bundle_score_globalrank_no_backfill_test"
    _write_score_dir(score_dir)

    row = readiness_for_score_dir(
        score_dir,
        already_monitored=set(),
        expected_rank_contract="anchor_global_policy_rank_reference",
        expected_selected_arm=ARM,
        expected_active_heads=["short_asset", "short_boll"],
        expected_disabled_heads=["long_bars", "long_dist"],
        min_timestamp_count=3,
    )

    assert row["status"] == "appendable"
    assert row["timestamp_count"] == 4
    assert row["period_start"] == "2026-06-26T09:00:00+00:00"
    assert row["total_net_pnl_delta"] == 1.0


def test_no_backfill_shadow_window_discovery_uses_config_monitored_set(
    tmp_path: Path,
) -> None:
    score_dir = tmp_path / "market_state_controller_bundle_score_globalrank_no_backfill_test"
    out = tmp_path / "out"
    config = tmp_path / "config.json"
    _write_score_dir(score_dir)
    config.write_text(
        json.dumps(
            {
                "market_state_controller_validation": {
                    "global_rank_threshold_controller_no_backfill_shadow_monitor": {
                        "windows": [{"score_dir": str(score_dir)}]
                    }
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = discover_no_backfill_shadow_windows(
        roots=[tmp_path],
        output_dir=out,
        config=config,
        include_regex="globalrank_no_backfill",
    )

    assert summary["discovered_candidate_count"] == 1
    assert summary["appendable_candidate_count"] == 0
    assert summary["already_monitored_count"] == 1
    readiness = pd.read_csv(summary["readiness_csv"])
    assert readiness.iloc[0]["status"] == "already_monitored"
