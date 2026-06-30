from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import report_market_state_no_backfill_shadow_monitor as monitor


def _fake_sha(seed: str) -> str:
    return (seed * 64)[:64]


def _write_no_backfill_score_dir(
    root: Path,
    *,
    start: str,
    end: str,
    baseline_net_pnl: float,
    shadow_net_pnl: float,
    total_delta: float,
    action_delta: float,
    common_delta: float,
    baseline_trades: int,
    shadow_trades: int,
    removed_trades: int,
    direct_removed_trades: int | None = None,
    direct_removed_loss_avoided: float | None = None,
    direct_removed_winner_pnl_sacrificed: float | None = None,
    direct_accepted_delta_defensive_success: float | None = None,
    shadow_controller_only: bool = True,
    baseline_max_drawdown: float = -0.010,
    shadow_max_drawdown: float = -0.009,
    baseline_worst_24h_net_pnl: float = -10.0,
    shadow_worst_24h_net_pnl: float = -9.0,
    baseline_full_sl_rate: float = 0.40,
    shadow_full_sl_rate: float = 0.39,
    baseline_timeout_rate: float = 0.15,
    shadow_timeout_rate: float = 0.14,
    removed_loss_avoided: float | None = None,
    removed_winner_pnl_sacrificed: float = 0.0,
    accepted_delta_defensive_success: float | None = None,
    rank_contract: str = "anchor_global_policy_rank_reference",
    selected_arm: str = "S1_observed_axes_shared_response__post_selection_overlay",
    key_columns: list[str] | None = None,
    include_artifact_hashes: bool = True,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    candidates_path = root / "eval_candidates.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime([start, end], utc=True),
            "symbol": ["BTC_USD", "ETH_USD"],
        }
    ).to_parquet(candidates_path, index=False)
    pd.DataFrame(
        {
            "head": ["short_asset", "short_boll"],
            "baseline_trade_count": [baseline_trades - 1, 1],
            "shadow_trade_count": [shadow_trades - 1, 1],
            "net_pnl_delta": [total_delta, 0.0],
        }
    ).to_csv(root / "shadow_no_backfill_replay_by_head.csv", index=False)
    pd.DataFrame(
        {
            "arm": ["S1_observed_axes_shared_response__post_selection_overlay"],
            "trade_count": [baseline_trades],
            "net_pnl": [baseline_net_pnl],
            "max_drawdown": [baseline_max_drawdown],
            "worst_24h_net_pnl": [baseline_worst_24h_net_pnl],
            "full_sl_rate": [baseline_full_sl_rate],
            "timeout_rate": [baseline_timeout_rate],
        }
    ).to_csv(root / "controller_replay_summary.csv", index=False)
    key_columns = key_columns or ["timestamp", "symbol", "strategy_id", "head", "side"]
    direct_loss_avoided = action_delta if removed_loss_avoided is None else removed_loss_avoided
    direct_defensive_success = (
        action_delta
        if accepted_delta_defensive_success is None
        else accepted_delta_defensive_success
    )
    direct_removed_trades = removed_trades if direct_removed_trades is None else direct_removed_trades
    direct_loss_avoided = (
        direct_loss_avoided
        if direct_removed_loss_avoided is None
        else direct_removed_loss_avoided
    )
    direct_winner_sacrificed = (
        removed_winner_pnl_sacrificed
        if direct_removed_winner_pnl_sacrificed is None
        else direct_removed_winner_pnl_sacrificed
    )
    direct_defensive_success = (
        direct_defensive_success
        if direct_accepted_delta_defensive_success is None
        else direct_accepted_delta_defensive_success
    )

    manifest = {
        "score_manifest_contract_version": "market_state_controller_score_manifest_v2",
        "selected_arm": selected_arm,
        "rank_contract": rank_contract,
        "controller_execution_enabled": not shadow_controller_only,
        "shadow_controller_only": shadow_controller_only,
        "controller": {
            "controller_execution_enabled": not shadow_controller_only,
            "execution_enabled": not shadow_controller_only,
            "shadow_controller_only": shadow_controller_only,
        },
        "eval_candidates": str(candidates_path),
        "shadow_no_backfill_replay_available": True,
        "shadow_no_backfill_replay_summary": {
            "share_threshold_raised": 1.0,
            "mean_threshold_delta": 0.07,
            "trade_count": shadow_trades,
            "net_pnl": shadow_net_pnl,
            "max_drawdown": shadow_max_drawdown,
            "worst_24h_net_pnl": shadow_worst_24h_net_pnl,
            "full_sl_rate": shadow_full_sl_rate,
            "timeout_rate": shadow_timeout_rate,
        },
        "shadow_no_backfill_accepted_delta_summary": {
            "available": True,
            "key_columns": key_columns,
            "baseline_net_pnl": baseline_net_pnl,
            "shadow_net_pnl": shadow_net_pnl,
            "total_net_pnl_delta": total_delta,
            "full_path_replay_net_pnl_delta": total_delta,
            "action_only_fixed_common_size_net_pnl_delta": action_delta,
            "path_dependent_common_trade_net_pnl_delta": common_delta,
            "baseline_trade_count": baseline_trades,
            "shadow_trade_count": shadow_trades,
            "removed_trade_count": removed_trades,
            "added_trade_count": 0,
            "common_trade_count": shadow_trades,
            "removed_net_pnl": -action_delta,
            "removed_loss_avoided": action_delta
            if removed_loss_avoided is None
            else removed_loss_avoided,
            "removed_winner_pnl_sacrificed": removed_winner_pnl_sacrificed,
            "accepted_delta_defensive_success": action_delta
            if accepted_delta_defensive_success is None
            else accepted_delta_defensive_success,
            "common_net_pnl_delta": common_delta,
            "shadow_subset_of_baseline": True,
        },
        "shadow_direct_threshold_only_available": True,
        "shadow_direct_threshold_only_delta_summary": {
            "available": True,
            "key_columns": key_columns,
            "direct_threshold_only": True,
            "no_path_or_capacity_replay": True,
            "baseline_net_pnl": baseline_net_pnl,
            "shadow_net_pnl": baseline_net_pnl + direct_defensive_success,
            "total_net_pnl_delta": direct_defensive_success,
            "full_path_replay_net_pnl_delta": direct_defensive_success,
            "action_only_fixed_common_size_net_pnl_delta": direct_defensive_success,
            "path_dependent_common_trade_net_pnl_delta": 0.0,
            "baseline_trade_count": baseline_trades,
            "shadow_trade_count": max(0, baseline_trades - direct_removed_trades),
            "removed_trade_count": direct_removed_trades,
            "added_trade_count": 0,
            "common_trade_count": max(0, baseline_trades - direct_removed_trades),
            "removed_loss_avoided": direct_loss_avoided,
            "removed_winner_pnl_sacrificed": direct_winner_sacrificed,
            "accepted_delta_defensive_success": direct_defensive_success,
            "shadow_subset_of_baseline": True,
        },
        "source_contract_audit": {
            "overall_passed": True,
            "splits": {
                "eval": {
                    "feature_store_timestamp_coverage": 1.0,
                    "feature_count": 1632,
                    "feature_store_symbols_read": 184,
                }
            },
        },
    }
    if include_artifact_hashes:
        manifest.update(
            {
                "bundle_sha256": _fake_sha("a"),
                "policy_manifest_sha256": _fake_sha("b"),
                "eval_candidates_sha256": _fake_sha("c"),
                "train_deployable_candidates_sha256": _fake_sha("d"),
                "output_sha256": {
                    key: _fake_sha(format(index + 1, "x"))
                    for index, key in enumerate(monitor.REQUIRED_SCORE_OUTPUT_HASH_KEYS)
                },
            }
        )
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_disabled_noop_score_dir(
    root: Path,
    *,
    start: str = "2026-06-26 20:00:00+00:00",
    end: str = "2026-06-26 22:00:00+00:00",
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    candidates_path = root / "eval_candidates.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime([start, end], utc=True),
            "symbol": ["BTC_USD", "ETH_USD"],
        }
    ).to_parquet(candidates_path, index=False)
    pd.DataFrame(
        {
            "arm": ["S1_observed_axes_shared_response__post_selection_overlay"],
            "trade_count": [7],
            "net_pnl": [12.5],
            "max_drawdown": [-0.01],
            "worst_24h_net_pnl": [-2.0],
            "full_sl_rate": [0.2],
            "timeout_rate": [0.1],
        }
    ).to_csv(root / "controller_replay_summary.csv", index=False)
    pd.DataFrame(
        {
            "head": ["short_asset", "short_boll"],
            "trade_count": [4, 3],
            "net_pnl": [8.0, 4.5],
        }
    ).to_csv(root / "controller_replay_by_head.csv", index=False)
    manifest = {
        "score_manifest_contract_version": "market_state_controller_score_manifest_v2",
        "selected_arm": "S1_observed_axes_shared_response__post_selection_overlay",
        "rank_contract": "anchor_global_policy_rank_reference",
        "controller_execution_enabled": False,
        "shadow_controller_only": False,
        "controller": {
            "controller_execution_enabled": False,
            "execution_enabled": False,
            "shadow_controller_only": False,
        },
        "eval_candidates": str(candidates_path),
        "shadow_no_backfill_replay_available": False,
        "shadow_no_backfill_replay_summary": {},
        "shadow_no_backfill_accepted_delta_summary": {},
        "shadow_direct_threshold_only_available": False,
        "shadow_locked_accepted_overlay_available": False,
        "source_contract_audit": {
            "overall_passed": True,
            "splits": {
                "eval": {
                    "feature_store_timestamp_coverage": 0.0,
                    "feature_count": 0,
                    "feature_store_symbols_read": 181,
                }
            },
        },
        "bundle_sha256": _fake_sha("a"),
        "policy_manifest_sha256": _fake_sha("b"),
        "eval_candidates_sha256": _fake_sha("c"),
        "train_deployable_candidates_sha256": _fake_sha("d"),
        "output_sha256": {
            key: _fake_sha(format(index + 1, "x"))
            for index, key in enumerate(
                key
                for key in monitor.REQUIRED_SCORE_OUTPUT_HASH_KEYS
                if not key.startswith("shadow_no_backfill")
            )
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_empty_eval_score_dir(
    root: Path,
    *,
    start: str = "2026-06-27 13:00:00+00:00",
    end: str = "2026-06-27 15:00:00+00:00",
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    candidates_path = root / "eval_candidates.parquet"
    pd.DataFrame(
        columns=[
            "timestamp",
            "symbol",
            "strategy_id",
            "head",
            "entry_price",
            "exit_price",
            "exit_timestamp",
            "holding_bars",
        ]
    ).to_parquet(candidates_path, index=False)
    manifest = {
        "score_manifest_contract_version": "market_state_controller_score_manifest_v2",
        "selected_arm": "S1_observed_axes_shared_response__post_selection_overlay",
        "rank_contract": "anchor_global_policy_rank_reference",
        "controller_execution_enabled": False,
        "shadow_controller_only": True,
        "controller": {
            "controller_execution_enabled": False,
            "execution_enabled": False,
            "shadow_controller_only": True,
        },
        "period_start": start,
        "period_end": end,
        "window_start": start,
        "window_end": end,
        "eval_candidates": str(candidates_path),
        "shadow_no_backfill_replay_available": False,
        "shadow_no_backfill_replay_summary": {},
        "shadow_no_backfill_accepted_delta_summary": {},
        "shadow_direct_threshold_only_available": False,
        "shadow_locked_accepted_overlay_available": False,
        "score_report": {
            "empty_eval_candidates": True,
            "empty_eval_reason": "no_candidate_rows_after_rank_contract_and_disabled_heads",
        },
        "source_contract_audit": {
            "overall_passed": True,
            "splits": {
                "eval": {
                    "feature_store_timestamp_coverage": 1.0,
                    "feature_count": 1632,
                    "feature_store_symbols_read": 181,
                }
            },
        },
        "bundle_sha256": _fake_sha("a"),
        "policy_manifest_sha256": _fake_sha("b"),
        "eval_candidates_sha256": _fake_sha("c"),
        "train_deployable_candidates_sha256": _fake_sha("d"),
        "output_sha256": {
            key: _fake_sha(format(index + 1, "x"))
            for index, key in enumerate(
                key
                for key in monitor.REQUIRED_SCORE_OUTPUT_HASH_KEYS
                if not key.startswith("shadow_no_backfill")
            )
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_no_backfill_shadow_monitor_records_disabled_noop_scores_without_crashing(
    tmp_path: Path,
) -> None:
    score_dir = tmp_path / "disabled_noop_score"
    out = tmp_path / "monitor"
    _write_disabled_noop_score_dir(score_dir)

    summary = monitor.build_monitor([score_dir], out, min_later_window_count=1)

    assert summary["window_count"] == 1
    assert summary["status"] == "not_promoted_contract_or_scope_failure"
    assert summary["sum_total_net_pnl_delta"] == pytest.approx(0.0)
    assert summary["sum_removed_trade_count"] == 0
    assert summary["direct_threshold_only_available_window_count"] == 0
    assert "shadow_no_backfill_replay_not_available" in summary["promotion_gate_failures"]
    assert "accepted_delta_not_available" in summary["promotion_gate_failures"]
    assert "eval_feature_store_timestamp_coverage_below_gate" in summary["promotion_gate_failures"]
    assert "eval_source_feature_count_nonpositive" in summary["promotion_gate_failures"]
    windows = pd.read_csv(out / "no_backfill_shadow_window_metrics.csv")
    assert bool(windows["shadow_no_backfill_replay_available"].iloc[0]) is False
    assert windows["total_net_pnl_delta"].iloc[0] == pytest.approx(0.0)


def test_no_backfill_shadow_monitor_ignores_empty_eval_windows(
    tmp_path: Path,
) -> None:
    score_real = tmp_path / "score_real"
    score_empty = tmp_path / "score_empty"
    out = tmp_path / "monitor"
    _write_no_backfill_score_dir(
        score_real,
        start="2026-06-26 09:00:00+00:00",
        end="2026-06-27 08:00:00+00:00",
        baseline_net_pnl=-10.0,
        shadow_net_pnl=-5.0,
        total_delta=5.0,
        action_delta=2.0,
        common_delta=3.0,
        baseline_trades=10,
        shadow_trades=9,
        removed_trades=1,
    )
    _write_empty_eval_score_dir(score_empty)

    summary = monitor.build_monitor([score_real, score_empty], out, min_later_window_count=1)

    assert summary["window_count"] == 1
    assert summary["ignored_empty_eval_window_count"] == 1
    windows = pd.read_csv(out / "no_backfill_shadow_window_metrics.csv")
    ignored = pd.read_csv(out / "ignored_empty_eval_windows.csv")
    assert len(windows) == 1
    assert len(ignored) == 1
    assert ignored["period_start"].iloc[0] == "2026-06-27 13:00:00+00:00"


def test_no_backfill_shadow_monitor_separates_action_only_from_full_replay(
    tmp_path: Path,
) -> None:
    score_a = tmp_path / "score_a"
    score_b = tmp_path / "score_b"
    out = tmp_path / "monitor"
    _write_no_backfill_score_dir(
        score_a,
        start="2026-06-23 09:00:00+00:00",
        end="2026-06-24 08:00:00+00:00",
        baseline_net_pnl=-51.0,
        shadow_net_pnl=-56.0,
        total_delta=-5.0,
        action_delta=9.0,
        common_delta=-14.0,
        baseline_trades=49,
        shadow_trades=46,
        removed_trades=3,
    )
    _write_no_backfill_score_dir(
        score_b,
        start="2026-06-24 09:00:00+00:00",
        end="2026-06-25 08:00:00+00:00",
        baseline_net_pnl=-92.0,
        shadow_net_pnl=-98.0,
        total_delta=-6.0,
        action_delta=6.0,
        common_delta=-12.0,
        baseline_trades=66,
        shadow_trades=56,
        removed_trades=10,
    )

    summary = monitor.build_monitor([score_a, score_b], out)

    assert summary["window_count"] == 2
    assert summary["positive_delta_window_share"] == 0.0
    assert summary["action_only_positive_window_share"] == 1.0
    assert summary["direct_threshold_only_available_window_count"] == 2
    assert summary["direct_threshold_only_positive_window_share"] == 1.0
    assert summary["direct_threshold_only_suppression_window_share"] == 1.0
    assert summary["locked_accepted_overlay_available_window_count"] == 2
    assert summary["locked_accepted_overlay_positive_window_share"] == 1.0
    assert summary["locked_accepted_overlay_suppression_window_share"] == 1.0
    assert summary["locked_accepted_overlay_promotion_gate_passed"] is False
    assert summary["locked_accepted_overlay_promotion_gate_failures"] == [
        "locked_accepted_overlay_insufficient_later_window_count"
    ]
    assert summary["direct_threshold_only_promotion_gate_passed"] is False
    assert (
        "direct_threshold_only_insufficient_later_window_count"
        in summary["direct_threshold_only_promotion_gate_failures"]
    )
    assert summary["median_total_net_pnl_delta"] == pytest.approx(-5.5)
    assert summary["q25_total_net_pnl_delta"] == pytest.approx(-5.75)
    assert summary["sum_full_path_replay_net_pnl_delta"] == pytest.approx(-11.0)
    assert summary["sum_action_only_fixed_common_size_net_pnl_delta"] == pytest.approx(15.0)
    assert summary["sum_path_dependent_common_trade_net_pnl_delta"] == pytest.approx(-26.0)
    assert summary["sum_direct_threshold_only_net_pnl_delta"] == pytest.approx(15.0)
    assert summary["sum_direct_threshold_only_removed_trade_count"] == 13
    assert summary["sum_direct_threshold_only_defensive_success"] == pytest.approx(15.0)
    assert summary["sum_locked_accepted_overlay_net_pnl_delta"] == pytest.approx(15.0)
    assert summary["sum_locked_accepted_overlay_removed_trade_count"] == 13
    assert summary["sum_locked_accepted_overlay_defensive_success"] == pytest.approx(15.0)
    assert summary["sum_indirect_path_or_capacity_net_pnl_delta"] == pytest.approx(-26.0)
    assert summary["sum_indirect_path_or_capacity_removed_trade_count"] == 0
    assert summary["sum_indirect_path_or_capacity_defensive_success"] == pytest.approx(0.0)
    assert summary["sum_added_trade_count"] == 0
    assert summary["sum_removed_trade_count"] == 13
    assert summary["min_trade_retention"] == pytest.approx(56 / 66)
    assert summary["min_max_drawdown_delta"] > 0.0
    assert summary["min_worst_24h_net_pnl_delta"] > 0.0
    assert summary["max_full_sl_rate_delta"] < 0.0
    assert summary["max_timeout_rate_delta"] < 0.0
    assert summary["sum_removed_loss_avoided"] == pytest.approx(15.0)
    assert summary["sum_removed_winner_pnl_sacrificed"] == 0.0
    assert summary["all_shadow_subset_of_baseline"] is True
    assert summary["all_source_contracts_passed"] is True
    assert summary["all_score_manifest_artifact_hashes_complete"] is True
    assert summary["score_manifest_contract_versions"] == [
        "market_state_controller_score_manifest_v2"
    ]
    assert summary["windows_missing_score_input_hash_fields"] == 0
    assert summary["windows_missing_required_output_hashes"] == 0
    assert summary["min_eval_feature_store_timestamp_coverage"] == 1.0
    assert summary["min_eval_source_feature_count"] == 1632
    assert summary["promotion_gate_passed"] is False
    assert summary["expected_rank_contract"] == "anchor_global_policy_rank_reference"
    assert summary["rank_contracts"] == ["anchor_global_policy_rank_reference"]
    assert summary["selected_arms"] == ["S1_observed_axes_shared_response__post_selection_overlay"]
    assert (
        "full_path_replay_negative_despite_positive_action_only_counterfactual"
        in summary["promotion_gate_failures"]
    )
    assert (
        "full_path_replay_negative_despite_positive_direct_threshold_only_counterfactual"
        in summary["promotion_gate_failures"]
    )
    assert "indirect_path_or_capacity_delta_negative" in summary["promotion_gate_failures"]
    assert (
        "indirect_path_or_capacity_drag_overwhelms_direct_threshold_benefit"
        in summary["promotion_gate_failures"]
    )
    assert "controller_execution_disabled_shadow_only" in summary["promotion_gate_failures"]

    assert (out / "no_backfill_shadow_window_metrics.csv").exists()
    assert (out / "no_backfill_shadow_by_head.csv").exists()
    assert (out / "no_backfill_shadow_monitor_summary.json").exists()
    assert (out / "no_backfill_shadow_monitor_report.md").exists()
    windows = pd.read_csv(out / "no_backfill_shadow_window_metrics.csv")
    assert list(windows["period_start"]) == [
        "2026-06-23T09:00:00+00:00",
        "2026-06-24T09:00:00+00:00",
    ]
    by_head = pd.read_csv(out / "no_backfill_shadow_by_head.csv")
    assert len(by_head) == 4


def test_no_backfill_shadow_monitor_does_not_hardcode_negative_failures(
    tmp_path: Path,
) -> None:
    score_a = tmp_path / "score_positive_a"
    score_b = tmp_path / "score_positive_b"
    out = tmp_path / "monitor"
    _write_no_backfill_score_dir(
        score_a,
        start="2026-06-26 09:00:00+00:00",
        end="2026-06-27 08:00:00+00:00",
        baseline_net_pnl=-10.0,
        shadow_net_pnl=5.0,
        total_delta=15.0,
        action_delta=8.0,
        common_delta=7.0,
        baseline_trades=50,
        shadow_trades=45,
        removed_trades=5,
    )
    _write_no_backfill_score_dir(
        score_b,
        start="2026-06-27 09:00:00+00:00",
        end="2026-06-28 08:00:00+00:00",
        baseline_net_pnl=-3.0,
        shadow_net_pnl=2.0,
        total_delta=5.0,
        action_delta=3.0,
        common_delta=2.0,
        baseline_trades=40,
        shadow_trades=38,
        removed_trades=2,
    )

    summary = monitor.build_monitor([score_a, score_b], out)

    assert summary["status"] == "not_promoted_contract_or_scope_failure"
    assert summary["positive_delta_window_share"] == 1.0
    assert summary["median_total_net_pnl_delta"] > 0.0
    assert summary["q25_total_net_pnl_delta"] >= 0.0
    assert summary["sum_full_path_replay_net_pnl_delta"] > 0.0
    assert summary["sum_action_only_fixed_common_size_net_pnl_delta"] > 0.0
    assert summary["sum_path_dependent_common_trade_net_pnl_delta"] > 0.0
    assert summary["sum_indirect_path_or_capacity_net_pnl_delta"] > 0.0
    assert summary["promotion_gate_passed"] is False
    assert summary["promotion_gate_failures"] == [
        "insufficient_later_window_count",
        "controller_execution_disabled_shadow_only",
    ]
    assert summary["locked_accepted_overlay_promotion_gate_passed"] is False
    assert summary["locked_accepted_overlay_promotion_gate_failures"] == [
        "locked_accepted_overlay_insufficient_later_window_count"
    ]


def test_no_backfill_shadow_monitor_can_pass_with_enough_clean_later_windows(
    tmp_path: Path,
) -> None:
    score_dirs = []
    for index, total_delta in enumerate([5.0, 6.0, 7.0], start=1):
        score_dir = tmp_path / f"score_clean_{index}"
        _write_no_backfill_score_dir(
            score_dir,
            start=f"2026-07-0{index} 09:00:00+00:00",
            end=f"2026-07-0{index + 1} 08:00:00+00:00",
            baseline_net_pnl=-2.0,
            shadow_net_pnl=total_delta - 2.0,
            total_delta=total_delta,
            action_delta=2.0,
            common_delta=total_delta - 2.0,
            baseline_trades=50,
            shadow_trades=48,
            removed_trades=2,
            shadow_controller_only=False,
        )
        score_dirs.append(score_dir)

    summary = monitor.build_monitor(score_dirs, tmp_path / "monitor")

    assert summary["window_count"] == 3
    assert summary["positive_delta_window_share"] == 1.0
    assert summary["promotion_gate_passed"] is True
    assert summary["promotion_gate_failures"] == []
    assert summary["locked_accepted_overlay_promotion_gate_passed"] is True
    assert summary["locked_accepted_overlay_promotion_gate_failures"] == []
    assert summary["status"] == "promotion_gate_passed"


def test_no_backfill_shadow_monitor_rejects_missing_artifact_hashes(
    tmp_path: Path,
) -> None:
    score_dirs = []
    for index, total_delta in enumerate([5.0, 6.0, 7.0], start=1):
        score_dir = tmp_path / f"score_unhashed_{index}"
        _write_no_backfill_score_dir(
            score_dir,
            start=f"2026-07-0{index} 09:00:00+00:00",
            end=f"2026-07-0{index + 1} 08:00:00+00:00",
            baseline_net_pnl=-2.0,
            shadow_net_pnl=total_delta - 2.0,
            total_delta=total_delta,
            action_delta=2.0,
            common_delta=total_delta - 2.0,
            baseline_trades=50,
            shadow_trades=48,
            removed_trades=2,
            shadow_controller_only=False,
            include_artifact_hashes=False,
        )
        score_dirs.append(score_dir)

    summary = monitor.build_monitor(score_dirs, tmp_path / "monitor")

    assert summary["sum_total_net_pnl_delta"] > 0.0
    assert summary["all_score_manifest_artifact_hashes_complete"] is False
    assert summary["windows_missing_score_input_hash_fields"] == 3
    assert summary["windows_missing_required_output_hashes"] == 3
    assert summary["promotion_gate_passed"] is False
    assert "score_manifest_artifact_hashes_missing" in summary["promotion_gate_failures"]
    assert summary["status"] == "not_promoted_contract_or_scope_failure"


def test_no_backfill_shadow_monitor_rejects_positive_pnl_with_worse_risk(
    tmp_path: Path,
) -> None:
    score_dir = tmp_path / "score_risk_worse"
    _write_no_backfill_score_dir(
        score_dir,
        start="2026-06-29 09:00:00+00:00",
        end="2026-06-30 08:00:00+00:00",
        baseline_net_pnl=2.0,
        shadow_net_pnl=10.0,
        total_delta=8.0,
        action_delta=4.0,
        common_delta=4.0,
        baseline_trades=50,
        shadow_trades=45,
        removed_trades=5,
        shadow_controller_only=False,
        baseline_max_drawdown=-0.010,
        shadow_max_drawdown=-0.030,
        baseline_worst_24h_net_pnl=-10.0,
        shadow_worst_24h_net_pnl=-30.0,
        baseline_full_sl_rate=0.20,
        shadow_full_sl_rate=0.35,
        baseline_timeout_rate=0.10,
        shadow_timeout_rate=0.22,
    )

    summary = monitor.build_monitor([score_dir], tmp_path / "monitor")

    assert summary["sum_total_net_pnl_delta"] > 0.0
    assert summary["promotion_gate_passed"] is False
    assert "max_drawdown_worsened" in summary["promotion_gate_failures"]
    assert "worst_24h_net_pnl_worsened" in summary["promotion_gate_failures"]
    assert "full_sl_rate_worsened" in summary["promotion_gate_failures"]
    assert "timeout_rate_worsened" in summary["promotion_gate_failures"]
    assert "controller_execution_disabled_shadow_only" not in summary["promotion_gate_failures"]


def test_no_backfill_shadow_monitor_rejects_negative_defensive_success_and_winner_sacrifice(
    tmp_path: Path,
) -> None:
    score_dir = tmp_path / "score_winner_sacrifice"
    _write_no_backfill_score_dir(
        score_dir,
        start="2026-07-01 09:00:00+00:00",
        end="2026-07-02 08:00:00+00:00",
        baseline_net_pnl=1.0,
        shadow_net_pnl=3.0,
        total_delta=2.0,
        action_delta=-1.0,
        common_delta=3.0,
        baseline_trades=50,
        shadow_trades=45,
        removed_trades=5,
        shadow_controller_only=False,
        removed_loss_avoided=1.0,
        removed_winner_pnl_sacrificed=2.0,
        accepted_delta_defensive_success=-1.0,
    )

    summary = monitor.build_monitor([score_dir], tmp_path / "monitor")

    assert summary["sum_total_net_pnl_delta"] > 0.0
    assert summary["sum_removed_loss_avoided"] == pytest.approx(1.0)
    assert summary["sum_removed_winner_pnl_sacrificed"] == pytest.approx(2.0)
    assert summary["sum_accepted_delta_defensive_success"] == pytest.approx(-1.0)
    assert summary["promotion_gate_passed"] is False
    assert "defensive_success_not_positive" in summary["promotion_gate_failures"]
    assert (
        "suppressed_loss_avoided_not_greater_than_winner_pnl_sacrificed"
        in summary["promotion_gate_failures"]
    )
    assert "controller_execution_disabled_shadow_only" not in summary["promotion_gate_failures"]


def test_no_backfill_shadow_monitor_rejects_harmful_indirect_path_suppression(
    tmp_path: Path,
) -> None:
    score_dir = tmp_path / "score_indirect_harm"
    _write_no_backfill_score_dir(
        score_dir,
        start="2026-07-02 09:00:00+00:00",
        end="2026-07-03 08:00:00+00:00",
        baseline_net_pnl=20.0,
        shadow_net_pnl=5.0,
        total_delta=-15.0,
        action_delta=3.0,
        common_delta=-18.0,
        baseline_trades=50,
        shadow_trades=45,
        removed_trades=5,
        direct_removed_trades=2,
        direct_removed_loss_avoided=4.0,
        direct_removed_winner_pnl_sacrificed=1.0,
        direct_accepted_delta_defensive_success=3.0,
        shadow_controller_only=False,
        removed_loss_avoided=3.0,
        removed_winner_pnl_sacrificed=20.0,
        accepted_delta_defensive_success=-17.0,
    )

    summary = monitor.build_monitor([score_dir], tmp_path / "monitor")

    assert summary["sum_full_path_replay_net_pnl_delta"] == pytest.approx(-15.0)
    assert summary["sum_direct_threshold_only_net_pnl_delta"] == pytest.approx(3.0)
    assert summary["sum_locked_accepted_overlay_net_pnl_delta"] == pytest.approx(3.0)
    assert summary["locked_accepted_overlay_promotion_gate_passed"] is False
    assert (
        "locked_accepted_overlay_insufficient_later_window_count"
        in summary["locked_accepted_overlay_promotion_gate_failures"]
    )
    assert summary["sum_indirect_path_or_capacity_net_pnl_delta"] == pytest.approx(-18.0)
    assert summary["sum_indirect_path_or_capacity_removed_trade_count"] == 3
    assert summary["sum_indirect_path_or_capacity_winner_pnl_sacrificed"] == pytest.approx(19.0)
    assert summary["promotion_gate_passed"] is False
    assert "indirect_path_or_capacity_delta_negative" in summary["promotion_gate_failures"]
    assert (
        "indirect_path_or_capacity_drag_overwhelms_direct_threshold_benefit"
        in summary["promotion_gate_failures"]
    )
    assert "harmful_indirect_path_or_capacity_suppression" in summary["promotion_gate_failures"]
    assert (
        "indirect_winner_pnl_sacrificed_exceeds_loss_avoided"
        in summary["promotion_gate_failures"]
    )
    assert "defensive_success_not_positive" in summary["promotion_gate_failures"]


def test_no_backfill_shadow_monitor_classifies_positive_replay_with_indirect_drag_as_negative(
    tmp_path: Path,
) -> None:
    score_dir = tmp_path / "score_positive_with_indirect_drag"
    _write_no_backfill_score_dir(
        score_dir,
        start="2026-07-02 09:00:00+00:00",
        end="2026-07-03 08:00:00+00:00",
        baseline_net_pnl=-10.0,
        shadow_net_pnl=-5.0,
        total_delta=5.0,
        action_delta=8.0,
        common_delta=-3.0,
        baseline_trades=50,
        shadow_trades=47,
        removed_trades=3,
        direct_removed_trades=2,
        direct_removed_loss_avoided=8.0,
        direct_removed_winner_pnl_sacrificed=0.0,
        direct_accepted_delta_defensive_success=8.0,
        shadow_controller_only=False,
        removed_loss_avoided=8.0,
        removed_winner_pnl_sacrificed=0.0,
        accepted_delta_defensive_success=8.0,
    )

    summary = monitor.build_monitor(
        [score_dir],
        tmp_path / "monitor",
        min_later_window_count=1,
    )

    assert summary["sum_full_path_replay_net_pnl_delta"] == pytest.approx(5.0)
    assert summary["sum_direct_threshold_only_net_pnl_delta"] == pytest.approx(8.0)
    assert summary["sum_indirect_path_or_capacity_net_pnl_delta"] == pytest.approx(-3.0)
    assert "indirect_path_or_capacity_delta_negative" in summary["promotion_gate_failures"]
    assert "harmful_indirect_path_or_capacity_suppression" in summary["promotion_gate_failures"]
    assert summary["status"] == "not_promoted_negative_later_windows"


def test_no_backfill_shadow_monitor_rejects_rank_or_matching_contract_change(
    tmp_path: Path,
) -> None:
    score_dir = tmp_path / "score_bad_contract"
    _write_no_backfill_score_dir(
        score_dir,
        start="2026-07-03 09:00:00+00:00",
        end="2026-07-04 08:00:00+00:00",
        baseline_net_pnl=-1.0,
        shadow_net_pnl=4.0,
        total_delta=5.0,
        action_delta=2.0,
        common_delta=3.0,
        baseline_trades=50,
        shadow_trades=48,
        removed_trades=2,
        shadow_controller_only=False,
        rank_contract="short_boll_timestamp_rank",
        key_columns=["timestamp", "symbol", "strategy_id"],
    )

    summary = monitor.build_monitor([score_dir], tmp_path / "monitor")

    assert summary["sum_total_net_pnl_delta"] > 0.0
    assert summary["promotion_gate_passed"] is False
    assert "rank_contract_changed_or_unexpected" in summary["promotion_gate_failures"]
    assert "accepted_delta_key_columns_mismatch" in summary["promotion_gate_failures"]


def test_no_backfill_shadow_monitor_fails_closed_without_replay(tmp_path: Path) -> None:
    score_dir = tmp_path / "score_without_replay"
    score_dir.mkdir()
    (score_dir / "manifest.json").write_text(
        json.dumps({"shadow_no_backfill_replay_available": False}),
        encoding="utf-8",
    )

    summary = monitor.build_monitor([score_dir], tmp_path / "monitor")

    assert summary["status"] == "not_promoted_contract_or_scope_failure"
    assert summary["sum_total_net_pnl_delta"] == pytest.approx(0.0)
    assert summary["min_eval_source_feature_count"] == 0
    assert "shadow_no_backfill_replay_not_available" in summary["promotion_gate_failures"]
