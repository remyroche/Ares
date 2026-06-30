#!/usr/bin/env python3
"""Aggregate no-backfill market-state shadow scores across later windows."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/market_state_no_backfill_shadow_monitor")
DEFAULT_EXPECTED_RANK_CONTRACT = "anchor_global_policy_rank_reference"
EXPECTED_ACCEPTED_DELTA_KEY_COLUMNS = ["timestamp", "symbol", "strategy_id", "head", "side"]
DEFAULT_MIN_LATER_WINDOW_COUNT = 3
MIN_TRADE_RETENTION = 0.70
REQUIRED_SCORE_INPUT_HASH_FIELDS = [
    "bundle_sha256",
    "policy_manifest_sha256",
    "eval_candidates_sha256",
    "train_deployable_candidates_sha256",
]
REQUIRED_SCORE_OUTPUT_HASH_KEYS = [
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```csv\n" + frame.to_csv(index=False) + "```"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _candidate_period(manifest: dict[str, Any]) -> tuple[str | None, str | None]:
    manifest_start = manifest.get("period_start") or manifest.get("window_start")
    manifest_end = manifest.get("period_end") or manifest.get("window_end")
    path_raw = manifest.get("eval_candidates")
    if not path_raw:
        return manifest_start, manifest_end
    path = Path(str(path_raw))
    if not path.exists():
        return manifest_start, manifest_end
    timestamps = pd.to_datetime(pd.read_parquet(path, columns=["timestamp"])["timestamp"], utc=True)
    timestamps = timestamps.dropna()
    if timestamps.empty:
        return manifest_start, manifest_end
    return timestamps.min().isoformat(), timestamps.max().isoformat()


def _csv_summary_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    return dict(frame.iloc[0].to_dict())


def _num(payload: dict[str, Any], key: str, default: float = np.nan) -> float:
    value = payload.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _valid_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if len(text) != 64:
        return False
    return all(char in "0123456789abcdef" for char in text)


def _missing_score_input_hash_fields(manifest: dict[str, Any]) -> list[str]:
    return [
        field
        for field in REQUIRED_SCORE_INPUT_HASH_FIELDS
        if not _valid_sha256(manifest.get(field))
    ]


def _missing_required_output_hashes(manifest: dict[str, Any]) -> list[str]:
    hashes = manifest.get("output_sha256")
    if not isinstance(hashes, dict):
        return list(REQUIRED_SCORE_OUTPUT_HASH_KEYS)
    return [
        key
        for key in REQUIRED_SCORE_OUTPUT_HASH_KEYS
        if not _valid_sha256(hashes.get(key))
    ]


def _window_row(score_dir: Path) -> dict[str, Any]:
    manifest = _load_json(score_dir / "manifest.json")
    has_shadow_replay = manifest.get("shadow_no_backfill_replay_available") is True
    delta = dict(manifest.get("shadow_no_backfill_accepted_delta_summary") or {})
    direct_delta = dict(
        manifest.get("shadow_direct_threshold_only_delta_summary")
        or manifest.get("shadow_direct_threshold_only_summary")
        or {}
    )
    locked_delta = dict(
        manifest.get("shadow_locked_accepted_overlay_delta_summary")
        or manifest.get("shadow_locked_accepted_overlay_summary")
        or direct_delta
        or {}
    )
    baseline_replay = _csv_summary_row(score_dir / "controller_replay_summary.csv")
    replay = dict(manifest.get("shadow_no_backfill_replay_summary") or {})
    if not replay:
        replay = _csv_summary_row(score_dir / "shadow_no_backfill_replay_summary.csv")
    if not has_shadow_replay:
        replay = dict(baseline_replay)
        delta = {
            "available": False,
            "key_columns": [],
            "baseline_net_pnl": _num(baseline_replay, "net_pnl"),
            "shadow_net_pnl": _num(baseline_replay, "net_pnl"),
            "total_net_pnl_delta": 0.0,
            "full_path_replay_net_pnl_delta": 0.0,
            "action_only_fixed_common_size_net_pnl_delta": 0.0,
            "path_dependent_common_trade_net_pnl_delta": 0.0,
            "baseline_trade_count": int(_num(baseline_replay, "trade_count", 0.0)),
            "shadow_trade_count": int(_num(baseline_replay, "trade_count", 0.0)),
            "removed_trade_count": 0,
            "added_trade_count": 0,
            "common_trade_count": int(_num(baseline_replay, "trade_count", 0.0)),
            "removed_net_pnl": 0.0,
            "removed_loss_avoided": 0.0,
            "removed_winner_pnl_sacrificed": 0.0,
            "accepted_delta_defensive_success": 0.0,
            "common_net_pnl_delta": 0.0,
            "shadow_subset_of_baseline": True,
        }
    source_eval = (
        ((manifest.get("source_contract_audit") or {}).get("splits") or {}).get("eval") or {}
    )
    accepted_delta_key_columns = _string_list(delta.get("key_columns"))
    period_start, period_end = _candidate_period(manifest)
    missing_input_hashes = _missing_score_input_hash_fields(manifest)
    missing_output_hashes = _missing_required_output_hashes(manifest)
    output_sha256 = manifest.get("output_sha256")
    output_hash_count = len(output_sha256) if isinstance(output_sha256, dict) else 0
    total_delta = float(delta.get("total_net_pnl_delta", np.nan))
    action_delta = float(
        delta.get(
            "action_only_fixed_common_size_net_pnl_delta",
            delta.get("accepted_delta_defensive_success", np.nan),
        )
    )
    common_delta = float(
        delta.get(
            "path_dependent_common_trade_net_pnl_delta",
            delta.get("common_net_pnl_delta", np.nan),
        )
    )
    baseline_trade_count = int(delta.get("baseline_trade_count", 0))
    shadow_trade_count = int(delta.get("shadow_trade_count", 0))
    baseline_max_drawdown = _num(baseline_replay, "max_drawdown")
    shadow_max_drawdown = _num(replay, "max_drawdown")
    baseline_worst_24h = _num(baseline_replay, "worst_24h_net_pnl")
    shadow_worst_24h = _num(replay, "worst_24h_net_pnl")
    baseline_full_sl_rate = _num(baseline_replay, "full_sl_rate")
    shadow_full_sl_rate = _num(replay, "full_sl_rate")
    baseline_timeout_rate = _num(baseline_replay, "timeout_rate")
    shadow_timeout_rate = _num(replay, "timeout_rate")
    return {
        "score_dir": str(score_dir),
        "empty_eval_candidates": bool(
            ((manifest.get("score_report") or {}).get("empty_eval_candidates") is True)
        ),
        "period_start": period_start,
        "period_end": period_end,
        "score_manifest_contract_version": manifest.get("score_manifest_contract_version"),
        "bundle_sha256": manifest.get("bundle_sha256"),
        "policy_manifest_sha256": manifest.get("policy_manifest_sha256"),
        "eval_candidates_sha256": manifest.get("eval_candidates_sha256"),
        "train_deployable_candidates_sha256": manifest.get(
            "train_deployable_candidates_sha256"
        ),
        "output_sha256_count": int(output_hash_count),
        "missing_score_input_hash_fields": "|".join(missing_input_hashes),
        "missing_required_output_hashes": "|".join(missing_output_hashes),
        "score_manifest_artifact_hashes_complete": bool(
            not missing_input_hashes and not missing_output_hashes
        ),
        "shadow_no_backfill_replay_available": bool(has_shadow_replay),
        "selected_arm": manifest.get("selected_arm"),
        "rank_contract": manifest.get("rank_contract"),
        "accepted_delta_available": bool(delta.get("available") is True),
        "accepted_delta_key_columns": "|".join(accepted_delta_key_columns),
        "baseline_net_pnl": float(delta.get("baseline_net_pnl", np.nan)),
        "shadow_net_pnl": float(delta.get("shadow_net_pnl", np.nan)),
        "total_net_pnl_delta": total_delta,
        "full_path_replay_net_pnl_delta": float(
            delta.get("full_path_replay_net_pnl_delta", total_delta)
        ),
        "action_only_fixed_common_size_net_pnl_delta": action_delta,
        "path_dependent_common_trade_net_pnl_delta": common_delta,
        "baseline_trade_count": baseline_trade_count,
        "shadow_trade_count": shadow_trade_count,
        "trade_retention": (
            float(shadow_trade_count / baseline_trade_count)
            if baseline_trade_count > 0
            else np.nan
        ),
        "removed_trade_count": int(delta.get("removed_trade_count", 0)),
        "added_trade_count": int(delta.get("added_trade_count", 0)),
        "common_trade_count": int(delta.get("common_trade_count", 0)),
        "removed_net_pnl": float(delta.get("removed_net_pnl", 0.0)),
        "removed_loss_avoided": float(delta.get("removed_loss_avoided", 0.0)),
        "removed_winner_pnl_sacrificed": float(
            delta.get("removed_winner_pnl_sacrificed", 0.0)
        ),
        "accepted_delta_defensive_success": float(
            delta.get("accepted_delta_defensive_success", action_delta)
        ),
        "direct_threshold_only_available": bool(
            manifest.get("shadow_direct_threshold_only_available")
            or direct_delta.get("direct_threshold_only")
            or direct_delta.get("available")
        ),
        "direct_threshold_only_total_net_pnl_delta": float(
            direct_delta.get("total_net_pnl_delta", np.nan)
        ),
        "direct_threshold_only_removed_trade_count": int(
            direct_delta.get("removed_trade_count", 0) or 0
        ),
        "direct_threshold_only_removed_loss_avoided": float(
            direct_delta.get("removed_loss_avoided", 0.0) or 0.0
        ),
        "direct_threshold_only_winner_pnl_sacrificed": float(
            direct_delta.get("removed_winner_pnl_sacrificed", 0.0) or 0.0
        ),
        "direct_threshold_only_defensive_success": float(
            direct_delta.get("accepted_delta_defensive_success", np.nan)
        ),
        "locked_accepted_overlay_available": bool(
            manifest.get("shadow_locked_accepted_overlay_available")
            or locked_delta.get("locked_accepted_overlay")
            or manifest.get("shadow_direct_threshold_only_available")
            or direct_delta.get("direct_threshold_only")
            or direct_delta.get("available")
        ),
        "locked_accepted_overlay_total_net_pnl_delta": float(
            locked_delta.get("total_net_pnl_delta", np.nan)
        ),
        "locked_accepted_overlay_removed_trade_count": int(
            locked_delta.get("removed_trade_count", 0) or 0
        ),
        "locked_accepted_overlay_removed_loss_avoided": float(
            locked_delta.get("removed_loss_avoided", 0.0) or 0.0
        ),
        "locked_accepted_overlay_winner_pnl_sacrificed": float(
            locked_delta.get("removed_winner_pnl_sacrificed", 0.0) or 0.0
        ),
        "locked_accepted_overlay_defensive_success": float(
            locked_delta.get("accepted_delta_defensive_success", np.nan)
        ),
        "common_net_pnl_delta": float(delta.get("common_net_pnl_delta", common_delta)),
        "baseline_max_drawdown": baseline_max_drawdown,
        "shadow_max_drawdown": shadow_max_drawdown,
        "max_drawdown_delta": (
            shadow_max_drawdown - baseline_max_drawdown
            if np.isfinite(shadow_max_drawdown) and np.isfinite(baseline_max_drawdown)
            else np.nan
        ),
        "baseline_worst_24h_net_pnl": baseline_worst_24h,
        "shadow_worst_24h_net_pnl": shadow_worst_24h,
        "worst_24h_net_pnl_delta": (
            shadow_worst_24h - baseline_worst_24h
            if np.isfinite(shadow_worst_24h) and np.isfinite(baseline_worst_24h)
            else np.nan
        ),
        "baseline_full_sl_rate": baseline_full_sl_rate,
        "shadow_full_sl_rate": shadow_full_sl_rate,
        "full_sl_rate_delta": (
            shadow_full_sl_rate - baseline_full_sl_rate
            if np.isfinite(shadow_full_sl_rate) and np.isfinite(baseline_full_sl_rate)
            else np.nan
        ),
        "baseline_timeout_rate": baseline_timeout_rate,
        "shadow_timeout_rate": shadow_timeout_rate,
        "timeout_rate_delta": (
            shadow_timeout_rate - baseline_timeout_rate
            if np.isfinite(shadow_timeout_rate) and np.isfinite(baseline_timeout_rate)
            else np.nan
        ),
        "shadow_share_threshold_raised": _num(replay, "share_threshold_raised"),
        "shadow_mean_threshold_delta": _num(replay, "mean_threshold_delta"),
        "shadow_subset_of_baseline": bool(delta.get("shadow_subset_of_baseline")),
        "eval_feature_store_timestamp_coverage": source_eval.get(
            "feature_store_timestamp_coverage"
        ),
        "eval_source_feature_count": source_eval.get("feature_count"),
        "eval_feature_store_symbols_read": source_eval.get("feature_store_symbols_read"),
        "source_contract_passed": bool(
            (manifest.get("source_contract_audit") or {}).get("overall_passed")
        ),
        "controller_execution_enabled": bool(
            manifest.get("controller_execution_enabled")
            or (manifest.get("controller") or {}).get("controller_execution_enabled")
            or (manifest.get("controller") or {}).get("execution_enabled")
        ),
        "shadow_controller_only": bool(
            manifest.get("shadow_controller_only")
            or (manifest.get("controller") or {}).get("shadow_controller_only")
        ),
    }


def _promotion_gate_failures(
    windows: pd.DataFrame,
    *,
    expected_rank_contract: str = DEFAULT_EXPECTED_RANK_CONTRACT,
    expected_selected_arm: str | None = None,
    min_later_window_count: int = DEFAULT_MIN_LATER_WINDOW_COUNT,
) -> list[str]:
    failures: list[str] = []
    if windows.empty:
        return ["no_later_windows"]

    median_delta = float(windows["total_net_pnl_delta"].median())
    q25_delta = float(windows["total_net_pnl_delta"].quantile(0.25))
    positive_share = float((windows["total_net_pnl_delta"] > 0.0).mean())
    full_delta = float(windows["full_path_replay_net_pnl_delta"].sum())
    action_delta = float(windows["action_only_fixed_common_size_net_pnl_delta"].sum())
    common_delta = float(windows["path_dependent_common_trade_net_pnl_delta"].sum())
    direct_available = bool(
        windows.get("direct_threshold_only_available", pd.Series(dtype=bool))
        .fillna(False)
        .astype(bool)
        .any()
    )
    direct_delta = (
        float(
            pd.to_numeric(
                windows.get(
                    "direct_threshold_only_total_net_pnl_delta",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            ).sum()
        )
        if direct_available
        else np.nan
    )
    direct_removed_total = int(
        pd.to_numeric(
            windows.get(
                "direct_threshold_only_removed_trade_count",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        ).fillna(0).sum()
    )
    added_trades = int(windows["added_trade_count"].sum())
    removed_trades = int(windows["removed_trade_count"].sum())
    indirect_removed_trades = max(0, removed_trades - direct_removed_total)
    defensive_success = float(windows["accepted_delta_defensive_success"].sum())
    removed_loss_avoided = float(windows["removed_loss_avoided"].sum())
    winner_pnl_sacrificed = float(windows["removed_winner_pnl_sacrificed"].sum())
    indirect_delta = float(full_delta - direct_delta) if direct_available else common_delta
    direct_loss_avoided = (
        float(
            pd.to_numeric(
                windows.get(
                    "direct_threshold_only_removed_loss_avoided",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            ).fillna(0.0).sum()
        )
        if direct_available
        else 0.0
    )
    direct_winner_sacrificed = (
        float(
            pd.to_numeric(
                windows.get(
                    "direct_threshold_only_winner_pnl_sacrificed",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            ).fillna(0.0).sum()
        )
        if direct_available
        else 0.0
    )
    indirect_loss_avoided = max(0.0, removed_loss_avoided - direct_loss_avoided)
    indirect_winner_sacrificed = max(
        0.0,
        winner_pnl_sacrificed - direct_winner_sacrificed,
    )
    min_trade_retention = float(
        pd.to_numeric(windows["trade_retention"], errors="coerce").min()
    )
    max_drawdown_delta = pd.to_numeric(windows["max_drawdown_delta"], errors="coerce")
    worst_24h_delta = pd.to_numeric(windows["worst_24h_net_pnl_delta"], errors="coerce")
    full_sl_delta = pd.to_numeric(windows["full_sl_rate_delta"], errors="coerce")
    timeout_delta = pd.to_numeric(windows["timeout_rate_delta"], errors="coerce")
    rank_contracts = set(windows["rank_contract"].dropna().astype(str))
    selected_arms = set(windows["selected_arm"].dropna().astype(str))
    expected_key_columns = "|".join(EXPECTED_ACCEPTED_DELTA_KEY_COLUMNS)

    if len(windows) < int(min_later_window_count):
        failures.append("insufficient_later_window_count")
    if median_delta <= 0.0:
        failures.append("negative_median_later_window_total_delta_net_pnl")
    if q25_delta < 0.0:
        failures.append("negative_q25_later_window_total_delta_net_pnl")
    if positive_share <= 0.5:
        failures.append("positive_later_window_share_not_above_chance")
    if positive_share == 0.0:
        failures.append("zero_positive_later_window_share")
    if full_delta < 0.0 and action_delta > 0.0:
        failures.append("full_path_replay_negative_despite_positive_action_only_counterfactual")
    if direct_available and full_delta < 0.0 and direct_delta > 0.0:
        failures.append(
            "full_path_replay_negative_despite_positive_direct_threshold_only_counterfactual"
        )
    if direct_available and indirect_delta < 0.0:
        failures.append("indirect_path_or_capacity_delta_negative")
    if (
        direct_available
        and direct_delta > 0.0
        and indirect_delta < 0.0
        and abs(indirect_delta) > direct_delta
    ):
        failures.append("indirect_path_or_capacity_drag_overwhelms_direct_threshold_benefit")
    if indirect_removed_trades > 0 and indirect_delta < 0.0:
        failures.append("harmful_indirect_path_or_capacity_suppression")
    if indirect_winner_sacrificed > indirect_loss_avoided:
        failures.append("indirect_winner_pnl_sacrificed_exceeds_loss_avoided")
    if action_delta > 0.0 and common_delta < 0.0 and abs(common_delta) > action_delta:
        failures.append("common_trade_sizing_drag_exceeds_removed_loss_avoided")
    if defensive_success <= 0.0:
        failures.append("defensive_success_not_positive")
    if removed_loss_avoided <= winner_pnl_sacrificed:
        failures.append("suppressed_loss_avoided_not_greater_than_winner_pnl_sacrificed")
    if added_trades != 0:
        failures.append("no_backfill_overlay_added_replacement_trades")
    if not bool(windows["shadow_subset_of_baseline"].all()):
        failures.append("shadow_trades_not_subset_of_baseline")
    if not bool(windows["source_contract_passed"].all()):
        failures.append("source_contract_failed")
    if (
        "score_manifest_artifact_hashes_complete" not in windows.columns
        or not bool(windows["score_manifest_artifact_hashes_complete"].fillna(False).astype(bool).all())
    ):
        failures.append("score_manifest_artifact_hashes_missing")
    if (
        "shadow_no_backfill_replay_available" not in windows.columns
        or not bool(windows["shadow_no_backfill_replay_available"].fillna(False).astype(bool).all())
    ):
        failures.append("shadow_no_backfill_replay_not_available")
    if np.isfinite(min_trade_retention) and min_trade_retention < MIN_TRADE_RETENTION:
        failures.append("trade_retention_below_gate")
    if max_drawdown_delta.notna().any() and bool((max_drawdown_delta < -1e-12).any()):
        failures.append("max_drawdown_worsened")
    if worst_24h_delta.notna().any() and bool((worst_24h_delta < -1e-12).any()):
        failures.append("worst_24h_net_pnl_worsened")
    if full_sl_delta.notna().any() and bool((full_sl_delta > 1e-12).any()):
        failures.append("full_sl_rate_worsened")
    if timeout_delta.notna().any() and bool((timeout_delta > 1e-12).any()):
        failures.append("timeout_rate_worsened")
    if not rank_contracts:
        failures.append("rank_contract_missing")
    elif rank_contracts != {expected_rank_contract}:
        failures.append("rank_contract_changed_or_unexpected")
    if not selected_arms:
        failures.append("selected_arm_missing")
    elif len(selected_arms) > 1:
        failures.append("selected_arm_changed_across_windows")
    elif expected_selected_arm is not None and selected_arms != {expected_selected_arm}:
        failures.append("selected_arm_not_expected")
    if not bool(windows["accepted_delta_available"].fillna(False).astype(bool).all()):
        failures.append("accepted_delta_not_available")
    if not bool(windows["accepted_delta_key_columns"].astype(str).eq(expected_key_columns).all()):
        failures.append("accepted_delta_key_columns_mismatch")

    coverage = pd.to_numeric(
        windows["eval_feature_store_timestamp_coverage"], errors="coerce"
    )
    if coverage.notna().any() and float(coverage.min()) < 0.999:
        failures.append("eval_feature_store_timestamp_coverage_below_gate")
    feature_counts = pd.to_numeric(windows["eval_source_feature_count"], errors="coerce")
    if feature_counts.notna().any() and int(feature_counts.min()) <= 0:
        failures.append("eval_source_feature_count_nonpositive")

    if (
        "controller_execution_enabled" in windows.columns
        and "shadow_controller_only" in windows.columns
        and not bool(windows["controller_execution_enabled"].fillna(False).astype(bool).any())
        and bool(windows["shadow_controller_only"].fillna(False).astype(bool).all())
    ):
        failures.append("controller_execution_disabled_shadow_only")

    return failures


def _accepted_overlay_gate_failures(
    prefix: str,
    *,
    available_count: int,
    removed_total: int,
    suppression_share: float,
    positive_share: float,
    success_total: float,
    loss_total: float,
    winner_total: float,
    min_later_window_count: int,
) -> list[str]:
    failures: list[str] = []
    if available_count == 0:
        failures.append(f"{prefix}_not_available")
    if available_count < int(min_later_window_count):
        failures.append(f"{prefix}_insufficient_later_window_count")
    if removed_total <= 0:
        failures.append(f"{prefix}_no_suppression")
    if suppression_share <= 0.5:
        failures.append(f"{prefix}_suppression_not_recurrent")
    if positive_share <= 0.5:
        failures.append(f"{prefix}_positive_window_share_not_above_chance")
    if success_total <= 0.0:
        failures.append(f"{prefix}_defensive_success_not_positive")
    if loss_total <= winner_total:
        failures.append(f"{prefix}_loss_avoided_not_greater_than_winner_sacrificed")
    return failures


def _monitor_status(failures: list[str]) -> str:
    if not failures:
        return "promotion_gate_passed"
    if "shadow_no_backfill_replay_not_available" in failures:
        return "not_promoted_contract_or_scope_failure"
    negative_failures = {
        "negative_median_later_window_total_delta_net_pnl",
        "negative_q25_later_window_total_delta_net_pnl",
        "positive_later_window_share_not_above_chance",
        "zero_positive_later_window_share",
        "full_path_replay_negative_despite_positive_action_only_counterfactual",
        "full_path_replay_negative_despite_positive_direct_threshold_only_counterfactual",
        "indirect_path_or_capacity_delta_negative",
        "indirect_path_or_capacity_drag_overwhelms_direct_threshold_benefit",
        "harmful_indirect_path_or_capacity_suppression",
        "indirect_winner_pnl_sacrificed_exceeds_loss_avoided",
        "common_trade_sizing_drag_exceeds_removed_loss_avoided",
        "defensive_success_not_positive",
        "suppressed_loss_avoided_not_greater_than_winner_pnl_sacrificed",
        "max_drawdown_worsened",
        "worst_24h_net_pnl_worsened",
        "full_sl_rate_worsened",
        "timeout_rate_worsened",
    }
    if any(failure in negative_failures for failure in failures):
        return "not_promoted_negative_later_windows"
    if failures == ["controller_execution_disabled_shadow_only"]:
        return "shadow_quality_passed_controller_execution_disabled"
    return "not_promoted_contract_or_scope_failure"


def _monitor_interpretation(status: str) -> str:
    if status == "not_promoted_negative_later_windows":
        return (
            "The frozen no-backfill threshold overlay removed only baseline-accepted trades "
            "and added no replacements. The action-only/fixed-common-size counterfactual "
            "was positive, but full path-dependent replay was negative in the scored later "
            "windows because common-trade sizing/path drag was larger. The controller remains "
            "shadow-only and should not be promoted without new positive full-replay "
            "later-window evidence."
        )
    if status == "shadow_quality_passed_controller_execution_disabled":
        return (
            "The no-backfill threshold overlay passed the later-window quality checks in this "
            "monitor, but the scored bundle is still explicitly shadow-only. Promotion still "
            "requires an intentional activation artifact and production governance review."
        )
    if status == "promotion_gate_passed":
        return (
            "The no-backfill threshold overlay passed this monitor's later-window gates. "
            "Before active use, verify the activation registry, production bundle, and "
            "deployment parity under the same rank and candidate contracts."
        )
    return (
        "The no-backfill threshold overlay did not pass monitor gates because one or more "
        "scope, source-contract, or no-backfill invariants failed."
    )


def build_monitor(
    score_dirs: list[Path],
    output_dir: Path,
    *,
    expected_rank_contract: str = DEFAULT_EXPECTED_RANK_CONTRACT,
    expected_selected_arm: str | None = None,
    min_later_window_count: int = DEFAULT_MIN_LATER_WINDOW_COUNT,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [_window_row(path) for path in score_dirs]
    ignored_empty_rows = [row for row in all_rows if bool(row.get("empty_eval_candidates"))]
    rows = [row for row in all_rows if not bool(row.get("empty_eval_candidates"))]
    windows = pd.DataFrame(rows)
    windows.to_csv(output_dir / "no_backfill_shadow_window_metrics.csv", index=False)
    pd.DataFrame(ignored_empty_rows).to_csv(
        output_dir / "ignored_empty_eval_windows.csv",
        index=False,
    )

    by_head_frames: list[pd.DataFrame] = []
    for path in score_dirs:
        by_head_path = path / "shadow_no_backfill_replay_by_head.csv"
        if not by_head_path.exists():
            continue
        by_head = pd.read_csv(by_head_path)
        row = next(row for row in rows if row["score_dir"] == str(path))
        by_head.insert(0, "score_dir", str(path))
        by_head.insert(1, "period_start", row["period_start"])
        by_head.insert(2, "period_end", row["period_end"])
        by_head_frames.append(by_head)
    by_head_all = pd.concat(by_head_frames, ignore_index=True) if by_head_frames else pd.DataFrame()
    by_head_all.to_csv(output_dir / "no_backfill_shadow_by_head.csv", index=False)

    positive_share = float((windows["total_net_pnl_delta"] > 0.0).mean()) if len(windows) else 0.0
    action_positive_share = (
        float((windows["action_only_fixed_common_size_net_pnl_delta"] > 0.0).mean())
        if len(windows)
        else 0.0
    )
    direct_available_mask = (
        windows["direct_threshold_only_available"].fillna(False).astype(bool)
        if "direct_threshold_only_available" in windows.columns
        else pd.Series(False, index=windows.index)
    )
    direct_deltas = pd.to_numeric(
        windows.get("direct_threshold_only_total_net_pnl_delta", pd.Series(index=windows.index)),
        errors="coerce",
    )
    direct_removed_counts = pd.to_numeric(
        windows.get(
            "direct_threshold_only_removed_trade_count",
            pd.Series(index=windows.index),
        ),
        errors="coerce",
    ).fillna(0)
    direct_loss_avoided = pd.to_numeric(
        windows.get(
            "direct_threshold_only_removed_loss_avoided",
            pd.Series(index=windows.index),
        ),
        errors="coerce",
    ).fillna(0.0)
    direct_winner_sacrificed = pd.to_numeric(
        windows.get(
            "direct_threshold_only_winner_pnl_sacrificed",
            pd.Series(index=windows.index),
        ),
        errors="coerce",
    ).fillna(0.0)
    direct_defensive_success = pd.to_numeric(
        windows.get(
            "direct_threshold_only_defensive_success",
            pd.Series(index=windows.index),
        ),
        errors="coerce",
    ).fillna(0.0)
    locked_available_mask = (
        windows.get("locked_accepted_overlay_available", pd.Series(dtype=bool))
        .fillna(False)
        .astype(bool)
        if "locked_accepted_overlay_available" in windows.columns
        else direct_available_mask
    )
    locked_deltas = pd.to_numeric(
        windows.get("locked_accepted_overlay_total_net_pnl_delta", direct_deltas),
        errors="coerce",
    ).fillna(0.0)
    locked_removed_counts = pd.to_numeric(
        windows.get(
            "locked_accepted_overlay_removed_trade_count",
            direct_removed_counts,
        ),
        errors="coerce",
    ).fillna(0)
    locked_loss_avoided = pd.to_numeric(
        windows.get(
            "locked_accepted_overlay_removed_loss_avoided",
            direct_loss_avoided,
        ),
        errors="coerce",
    ).fillna(0.0)
    locked_winner_sacrificed = pd.to_numeric(
        windows.get(
            "locked_accepted_overlay_winner_pnl_sacrificed",
            direct_winner_sacrificed,
        ),
        errors="coerce",
    ).fillna(0.0)
    locked_defensive_success = pd.to_numeric(
        windows.get(
            "locked_accepted_overlay_defensive_success",
            direct_defensive_success,
        ),
        errors="coerce",
    ).fillna(0.0)
    direct_positive_share = (
        float((direct_deltas.loc[direct_available_mask] > 0.0).mean())
        if bool(direct_available_mask.any())
        else 0.0
    )
    direct_suppression_share = (
        float((direct_removed_counts.loc[direct_available_mask] > 0).mean())
        if bool(direct_available_mask.any())
        else 0.0
    )
    locked_positive_share = (
        float((locked_deltas.loc[locked_available_mask] > 0.0).mean())
        if bool(locked_available_mask.any())
        else 0.0
    )
    locked_suppression_share = (
        float((locked_removed_counts.loc[locked_available_mask] > 0).mean())
        if bool(locked_available_mask.any())
        else 0.0
    )
    direct_available_count = int(direct_available_mask.sum())
    direct_removed_total = int(direct_removed_counts.sum()) if len(windows) else 0
    direct_loss_total = float(direct_loss_avoided.sum()) if len(windows) else 0.0
    direct_winner_total = float(direct_winner_sacrificed.sum()) if len(windows) else 0.0
    direct_success_total = float(direct_defensive_success.sum()) if len(windows) else 0.0
    locked_available_count = int(locked_available_mask.sum())
    locked_removed_total = int(locked_removed_counts.sum()) if len(windows) else 0
    locked_loss_total = float(locked_loss_avoided.sum()) if len(windows) else 0.0
    locked_winner_total = float(locked_winner_sacrificed.sum()) if len(windows) else 0.0
    locked_success_total = float(locked_defensive_success.sum()) if len(windows) else 0.0
    total_removed_total = int(windows["removed_trade_count"].sum()) if len(windows) else 0
    indirect_removed_total = max(0, total_removed_total - direct_removed_total)
    full_path_delta_total = (
        float(windows["full_path_replay_net_pnl_delta"].sum()) if len(windows) else 0.0
    )
    indirect_path_delta_total = (
        full_path_delta_total - float(direct_deltas.sum()) if bool(direct_available_mask.any()) else 0.0
    )
    indirect_loss_total = max(
        0.0,
        (float(windows["removed_loss_avoided"].sum()) if len(windows) else 0.0)
        - direct_loss_total,
    )
    indirect_winner_total = max(
        0.0,
        (
            float(windows["removed_winner_pnl_sacrificed"].sum())
            if len(windows)
            else 0.0
        )
        - direct_winner_total,
    )
    indirect_defensive_success_total = indirect_loss_total - indirect_winner_total
    direct_failures = _accepted_overlay_gate_failures(
        "direct_threshold_only",
        available_count=direct_available_count,
        removed_total=direct_removed_total,
        suppression_share=direct_suppression_share,
        positive_share=direct_positive_share,
        success_total=direct_success_total,
        loss_total=direct_loss_total,
        winner_total=direct_winner_total,
        min_later_window_count=int(min_later_window_count),
    )
    locked_failures = _accepted_overlay_gate_failures(
        "locked_accepted_overlay",
        available_count=locked_available_count,
        removed_total=locked_removed_total,
        suppression_share=locked_suppression_share,
        positive_share=locked_positive_share,
        success_total=locked_success_total,
        loss_total=locked_loss_total,
        winner_total=locked_winner_total,
        min_later_window_count=int(min_later_window_count),
    )
    promotion_failures = _promotion_gate_failures(
        windows,
        expected_rank_contract=expected_rank_contract,
        expected_selected_arm=expected_selected_arm,
        min_later_window_count=int(min_later_window_count),
    )
    status = _monitor_status(promotion_failures)
    summary = {
        "generated_by": "report_market_state_no_backfill_shadow_monitor",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "monitor_dir": str(output_dir),
        "window_metrics_csv": str(output_dir / "no_backfill_shadow_window_metrics.csv"),
        "by_head_csv": str(output_dir / "no_backfill_shadow_by_head.csv"),
        "expected_rank_contract": expected_rank_contract,
        "expected_selected_arm": expected_selected_arm,
        "min_later_window_count": int(min_later_window_count),
        "rank_contracts": sorted(
            windows["rank_contract"].dropna().astype(str).unique().tolist()
        )
        if len(windows)
        else [],
        "selected_arms": sorted(windows["selected_arm"].dropna().astype(str).unique().tolist())
        if len(windows)
        else [],
        "window_count": int(len(windows)),
        "ignored_empty_eval_window_count": int(len(ignored_empty_rows)),
        "positive_delta_window_share": positive_share,
        "action_only_positive_window_share": action_positive_share,
        "direct_threshold_only_available_window_count": int(direct_available_mask.sum()),
        "direct_threshold_only_positive_window_share": direct_positive_share,
        "direct_threshold_only_suppression_window_share": direct_suppression_share,
        "locked_accepted_overlay_available_window_count": locked_available_count,
        "locked_accepted_overlay_positive_window_share": locked_positive_share,
        "locked_accepted_overlay_suppression_window_share": locked_suppression_share,
        "median_total_net_pnl_delta": float(windows["total_net_pnl_delta"].median())
        if len(windows)
        else 0.0,
        "q25_total_net_pnl_delta": float(windows["total_net_pnl_delta"].quantile(0.25))
        if len(windows)
        else 0.0,
        "sum_total_net_pnl_delta": float(windows["total_net_pnl_delta"].sum())
        if len(windows)
        else 0.0,
        "sum_full_path_replay_net_pnl_delta": full_path_delta_total,
        "sum_action_only_fixed_common_size_net_pnl_delta": float(
            windows["action_only_fixed_common_size_net_pnl_delta"].sum()
        )
        if len(windows)
        else 0.0,
        "sum_path_dependent_common_trade_net_pnl_delta": float(
            windows["path_dependent_common_trade_net_pnl_delta"].sum()
        )
        if len(windows)
        else 0.0,
        "sum_direct_threshold_only_net_pnl_delta": float(direct_deltas.sum())
        if len(windows)
        else 0.0,
        "sum_direct_threshold_only_removed_trade_count": direct_removed_total,
        "sum_direct_threshold_only_removed_loss_avoided": direct_loss_total,
        "sum_direct_threshold_only_winner_pnl_sacrificed": direct_winner_total,
        "sum_direct_threshold_only_defensive_success": direct_success_total,
        "sum_locked_accepted_overlay_net_pnl_delta": float(locked_deltas.sum())
        if len(windows)
        else 0.0,
        "sum_locked_accepted_overlay_removed_trade_count": locked_removed_total,
        "sum_locked_accepted_overlay_loss_avoided": locked_loss_total,
        "sum_locked_accepted_overlay_winner_pnl_sacrificed": locked_winner_total,
        "sum_locked_accepted_overlay_defensive_success": locked_success_total,
        "sum_indirect_path_or_capacity_net_pnl_delta": indirect_path_delta_total,
        "sum_indirect_path_or_capacity_removed_trade_count": indirect_removed_total,
        "sum_indirect_path_or_capacity_loss_avoided": indirect_loss_total,
        "sum_indirect_path_or_capacity_winner_pnl_sacrificed": indirect_winner_total,
        "sum_indirect_path_or_capacity_defensive_success": indirect_defensive_success_total,
        "sum_baseline_net_pnl": float(windows["baseline_net_pnl"].sum()) if len(windows) else 0.0,
        "sum_shadow_net_pnl": float(windows["shadow_net_pnl"].sum()) if len(windows) else 0.0,
        "sum_baseline_trade_count": int(windows["baseline_trade_count"].sum())
        if len(windows)
        else 0,
        "sum_shadow_trade_count": int(windows["shadow_trade_count"].sum())
        if len(windows)
        else 0,
        "sum_removed_trade_count": total_removed_total,
        "sum_added_trade_count": int(windows["added_trade_count"].sum()) if len(windows) else 0,
        "sum_common_trade_count": int(windows["common_trade_count"].sum())
        if len(windows)
        else 0,
        "min_trade_retention": float(windows["trade_retention"].min()) if len(windows) else 0.0,
        "min_max_drawdown_delta": float(windows["max_drawdown_delta"].min())
        if len(windows)
        else 0.0,
        "min_worst_24h_net_pnl_delta": float(windows["worst_24h_net_pnl_delta"].min())
        if len(windows)
        else 0.0,
        "max_full_sl_rate_delta": float(windows["full_sl_rate_delta"].max())
        if len(windows)
        else 0.0,
        "max_timeout_rate_delta": float(windows["timeout_rate_delta"].max())
        if len(windows)
        else 0.0,
        "sum_removed_loss_avoided": float(windows["removed_loss_avoided"].sum())
        if len(windows)
        else 0.0,
        "sum_removed_winner_pnl_sacrificed": float(
            windows["removed_winner_pnl_sacrificed"].sum()
        )
        if len(windows)
        else 0.0,
        "sum_accepted_delta_defensive_success": float(
            windows["accepted_delta_defensive_success"].sum()
        )
        if len(windows)
        else 0.0,
        "sum_common_net_pnl_delta": float(windows["common_net_pnl_delta"].sum())
        if len(windows)
        else 0.0,
        "min_eval_feature_store_timestamp_coverage": float(
            windows["eval_feature_store_timestamp_coverage"].min()
        )
        if len(windows)
        else 0.0,
        "min_eval_source_feature_count": (
            int(pd.to_numeric(windows["eval_source_feature_count"], errors="coerce").min())
            if len(windows)
            and pd.to_numeric(windows["eval_source_feature_count"], errors="coerce").notna().any()
            else 0
        ),
        "all_shadow_subset_of_baseline": bool(windows["shadow_subset_of_baseline"].all())
        if len(windows)
        else False,
        "all_source_contracts_passed": bool(windows["source_contract_passed"].all())
        if len(windows)
        else False,
        "all_score_manifest_artifact_hashes_complete": bool(
            windows["score_manifest_artifact_hashes_complete"].all()
        )
        if len(windows)
        else False,
        "score_manifest_contract_versions": sorted(
            windows["score_manifest_contract_version"].dropna().astype(str).unique().tolist()
        )
        if len(windows)
        else [],
        "windows_missing_score_input_hash_fields": int(
            windows["missing_score_input_hash_fields"].astype(str).ne("").sum()
        )
        if len(windows)
        else 0,
        "windows_missing_required_output_hashes": int(
            windows["missing_required_output_hashes"].astype(str).ne("").sum()
        )
        if len(windows)
        else 0,
        "promotion_gate_passed": not promotion_failures,
        "promotion_gate_failures": promotion_failures,
        "direct_threshold_only_promotion_gate_passed": not direct_failures,
        "direct_threshold_only_promotion_gate_failures": direct_failures,
        "locked_accepted_overlay_promotion_gate_passed": not locked_failures,
        "locked_accepted_overlay_promotion_gate_failures": locked_failures,
        "interpretation": _monitor_interpretation(status),
        "windows": rows,
    }
    (output_dir / "no_backfill_shadow_monitor_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = [
        "# No-Backfill Shadow Monitor",
        "",
        f"- Status: `{summary['status']}`",
        f"- Window count: `{summary['window_count']}`",
        f"- Positive full-replay window share: `{summary['positive_delta_window_share']}`",
        f"- Positive action-only window share: `{summary['action_only_positive_window_share']}`",
        f"- Direct-threshold-only available windows: `{summary['direct_threshold_only_available_window_count']}`",
        f"- Positive direct-threshold-only window share: `{summary['direct_threshold_only_positive_window_share']}`",
        f"- Direct-threshold-only suppression window share: `{summary['direct_threshold_only_suppression_window_share']}`",
        f"- Median full-replay delta net PnL: `{summary['median_total_net_pnl_delta']}`",
        f"- Q25 full-replay delta net PnL: `{summary['q25_total_net_pnl_delta']}`",
        f"- Sum full-replay delta net PnL: `{summary['sum_total_net_pnl_delta']}`",
        f"- Sum action-only fixed-common-size delta: `{summary['sum_action_only_fixed_common_size_net_pnl_delta']}`",
        f"- Sum direct-threshold-only delta: `{summary['sum_direct_threshold_only_net_pnl_delta']}`",
        f"- Sum direct-threshold-only defensive success: `{summary['sum_direct_threshold_only_defensive_success']}`",
        f"- Locked accepted-overlay available windows: `{summary['locked_accepted_overlay_available_window_count']}`",
        f"- Positive locked accepted-overlay window share: `{summary['locked_accepted_overlay_positive_window_share']}`",
        f"- Locked accepted-overlay suppression window share: `{summary['locked_accepted_overlay_suppression_window_share']}`",
        f"- Sum locked accepted-overlay delta: `{summary['sum_locked_accepted_overlay_net_pnl_delta']}`",
        f"- Sum locked accepted-overlay removed trades: `{summary['sum_locked_accepted_overlay_removed_trade_count']}`",
        f"- Sum locked accepted-overlay defensive success: `{summary['sum_locked_accepted_overlay_defensive_success']}`",
        f"- Sum indirect path/capacity delta: `{summary['sum_indirect_path_or_capacity_net_pnl_delta']}`",
        f"- Sum indirect path/capacity removed trades: `{summary['sum_indirect_path_or_capacity_removed_trade_count']}`",
        f"- Sum indirect path/capacity defensive success: `{summary['sum_indirect_path_or_capacity_defensive_success']}`",
        f"- Direct-threshold-only promotion gate passed: `{summary['direct_threshold_only_promotion_gate_passed']}`",
        f"- Direct-threshold-only failures: `{', '.join(summary['direct_threshold_only_promotion_gate_failures'])}`",
        f"- Locked accepted-overlay promotion gate passed: `{summary['locked_accepted_overlay_promotion_gate_passed']}`",
        f"- Locked accepted-overlay failures: `{', '.join(summary['locked_accepted_overlay_promotion_gate_failures'])}`",
        f"- Sum path-dependent common-trade delta: `{summary['sum_path_dependent_common_trade_net_pnl_delta']}`",
        f"- Minimum trade retention: `{summary['min_trade_retention']}`",
        f"- Minimum max-drawdown delta: `{summary['min_max_drawdown_delta']}`",
        f"- Minimum worst-24h delta: `{summary['min_worst_24h_net_pnl_delta']}`",
        f"- Maximum full-SL delta: `{summary['max_full_sl_rate_delta']}`",
        f"- Maximum timeout delta: `{summary['max_timeout_rate_delta']}`",
        f"- Artifact hashes complete: `{summary['all_score_manifest_artifact_hashes_complete']}`",
        f"- Windows missing input hashes: `{summary['windows_missing_score_input_hash_fields']}`",
        f"- Windows missing required output hashes: `{summary['windows_missing_required_output_hashes']}`",
        "",
        "## Windows",
        "",
        _markdown_table(windows),
        "",
        "## Interpretation",
        "",
        str(summary["interpretation"]),
    ]
    (output_dir / "no_backfill_shadow_monitor_report.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-rank-contract", default=DEFAULT_EXPECTED_RANK_CONTRACT)
    parser.add_argument("--expected-selected-arm", default=None)
    parser.add_argument("--min-later-window-count", type=int, default=DEFAULT_MIN_LATER_WINDOW_COUNT)
    args = parser.parse_args()
    summary = build_monitor(
        list(args.score_dir),
        args.output_dir,
        expected_rank_contract=str(args.expected_rank_contract),
        expected_selected_arm=args.expected_selected_arm,
        min_later_window_count=int(args.min_later_window_count),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
