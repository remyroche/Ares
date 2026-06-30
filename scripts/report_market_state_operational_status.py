#!/usr/bin/env python3
"""Summarize market-state controller operational status from the stack config.

This report distinguishes structural completion from promotion readiness.  It
is deliberately read-only: it does not score candidates, change thresholds, or
select a controller.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CONFIG = Path("config/reliability_blend_production_stack.json")
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/market_state_operational_status")


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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, str) and not value.strip():
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def build_operational_status(config: dict[str, Any]) -> dict[str, Any]:
    active = dict(config.get("active_stack") or {})
    experiment = dict(config.get("market_state_experiment_contract") or {})
    controller = dict(config.get("market_state_controller_validation") or {})
    monitor = dict(controller.get("shadow_controller_monitor") or {})
    monitor_discovery = dict(monitor.get("append_window_discovery") or {})
    threshold_smoke = dict(controller.get("global_rank_threshold_controller_smoke") or {})
    threshold_walkforward = dict(controller.get("global_rank_threshold_controller_walkforward") or {})
    t1_timestamp_walkforward = dict(
        controller.get("t1_timestamp_accepted_frontier_walkforward") or {}
    )
    threshold_no_backfill = dict(
        controller.get("global_rank_threshold_controller_no_backfill_walkforward") or {}
    )
    threshold_direct_ledger_raw = dict(
        threshold_no_backfill.get("direct_suppression_training_ledger") or {}
    )
    threshold_direct_ledger_combined = dict(
        threshold_no_backfill.get("direct_suppression_combined_training_ledger") or {}
    )
    threshold_direct_ledger = threshold_direct_ledger_combined or threshold_direct_ledger_raw
    threshold_direct_ledger_variant = "combined" if threshold_direct_ledger_combined else "single_source"
    threshold_direct_training_raw = dict(
        threshold_no_backfill.get("direct_suppression_shadow_training") or {}
    )
    threshold_direct_training_combined = dict(
        threshold_no_backfill.get("direct_suppression_combined_shadow_training") or {}
    )
    threshold_direct_training = threshold_direct_training_combined or threshold_direct_training_raw
    threshold_direct_training_variant = (
        "combined" if threshold_direct_training_combined else "single_source"
    )
    threshold_no_backfill_shadow = dict(
        controller.get("global_rank_threshold_controller_no_backfill_shadow_score_latest") or {}
    )
    threshold_no_backfill_monitor = dict(
        controller.get("global_rank_threshold_controller_no_backfill_shadow_monitor") or {}
    )
    threshold_no_backfill_discovery = dict(
        controller.get("global_rank_threshold_controller_no_backfill_shadow_window_discovery") or {}
    )
    threshold_no_backfill_next_readiness = dict(
        controller.get("global_rank_threshold_controller_no_backfill_next_window_readiness")
        or {}
    )
    threshold_no_backfill_next_runner = dict(
        controller.get("global_rank_threshold_controller_no_backfill_next_window_runner")
        or {}
    )
    threshold_no_backfill_failure = dict(
        controller.get("global_rank_threshold_controller_no_backfill_failure_diagnostics") or {}
    )
    plan = dict(controller.get("plan_completion_audit") or {})
    global_response_gate = dict(controller.get("strategy_response_quality_gate") or {})
    timestamp_response_gate = dict(controller.get("t1_timestamp_strategy_response_quality_gate") or {})
    active_rank_contract = active.get("rank_contract")
    experiment_rank_contract = experiment.get("rank_contract")
    if active_rank_contract == "short_boll_timestamp_rank" or (
        not active_rank_contract and experiment_rank_contract == "short_boll_timestamp_rank"
    ):
        response_gate = timestamp_response_gate or global_response_gate
    else:
        response_gate = global_response_gate or timestamp_response_gate
    priority = dict(config.get("market_state_priority_modulation_validation") or {})
    exchangefixed_priority = dict(priority.get("exchangefixed_global_rank_shadow_validation") or {})
    priority_headmix = dict(priority.get("global_rank_opportunity_headmix_validation") or {})
    rank_scope = dict(priority.get("short_boll_state_rank_scope_switch_shadow") or {})
    rank_scope_validation = dict(rank_scope.get("aggregate_validation") or {})

    structural_complete = bool(plan.get("passed_structural_audit")) and int(
        plan.get("hard_failure_count") or 0
    ) == 0
    experiment_contract_matches_active_stack = bool(
        experiment
        and experiment.get("rank_contract") == active.get("rank_contract")
        and experiment.get("rank_scope") == active.get("rank_scope")
        and (experiment.get("enabled_heads") or []) == (active.get("enabled_heads") or [])
        and (experiment.get("disabled_heads") or []) == (active.get("disabled_heads") or [])
        and experiment.get("policy_variant") == active.get("policy_variant")
        and _bool(experiment.get("qfail_active")) == _bool(active.get("qfail_active"))
    )
    threshold_controller_promoted = (
        active.get("market_state_threshold_controller_active") is True
        and monitor.get("shadow_promotion_gate_passed") is True
    )
    priority_promoted = active.get("market_state_priority_modulation_active") is True
    rank_router_promoted = rank_scope_validation.get("shadow_promotion_gate_passed") is True
    no_backfill_monitor_window_count = _int_or_none(
        threshold_no_backfill_monitor.get("window_count")
    )
    no_backfill_monitor_min_windows = _int_or_none(
        threshold_no_backfill_monitor.get("min_later_window_count")
    )
    no_backfill_monitor_windows = threshold_no_backfill_monitor.get("windows")
    if not isinstance(no_backfill_monitor_windows, list):
        no_backfill_monitor_windows = []
    no_backfill_monitor_window_periods = [
        {
            "period_start": window.get("period_start"),
            "period_end": window.get("period_end"),
            "total_net_pnl_delta": window.get("total_net_pnl_delta"),
            "baseline_trade_count": window.get("baseline_trade_count"),
            "shadow_trade_count": window.get("shadow_trade_count"),
        }
        for window in no_backfill_monitor_windows
        if isinstance(window, dict)
    ]
    no_backfill_monitor_latest_end = max(
        (
            str(window.get("period_end"))
            for window in no_backfill_monitor_windows
            if isinstance(window, dict) and window.get("period_end")
        ),
        default=None,
    )
    no_backfill_discovery_appendable_count = _int_or_none(
        threshold_no_backfill_discovery.get("appendable_candidate_count")
    )
    no_backfill_discovery_latest_end = threshold_no_backfill_discovery.get(
        "latest_discovered_window_end"
    )
    no_backfill_monitor_additional_windows_needed = None
    if (
        no_backfill_monitor_window_count is not None
        and no_backfill_monitor_min_windows is not None
    ):
        no_backfill_monitor_additional_windows_needed = max(
            0,
            no_backfill_monitor_min_windows - no_backfill_monitor_window_count,
        )

    blockers: list[dict[str, Any]] = []
    if not structural_complete:
        blockers.append(
            {
                "component": "structural_audit",
                "severity": "hard",
                "reason": "plan_completion_audit_not_structurally_complete",
            }
        )
    if experiment and not experiment_contract_matches_active_stack:
        blockers.append(
            {
                "component": "experiment_contract",
                "severity": "hard",
                "reason": "market_state_experiment_contract_mismatch_active_stack",
                "failures": [
                    "rank_contract_mismatch"
                    if experiment.get("rank_contract") != active.get("rank_contract")
                    else None,
                    "rank_scope_mismatch"
                    if experiment.get("rank_scope") != active.get("rank_scope")
                    else None,
                    "enabled_heads_mismatch"
                    if (experiment.get("enabled_heads") or [])
                    != (active.get("enabled_heads") or [])
                    else None,
                    "disabled_heads_mismatch"
                    if (experiment.get("disabled_heads") or [])
                    != (active.get("disabled_heads") or [])
                    else None,
                    "policy_variant_mismatch"
                    if experiment.get("policy_variant") != active.get("policy_variant")
                    else None,
                    "qfail_active_mismatch"
                    if _bool(experiment.get("qfail_active")) != _bool(active.get("qfail_active"))
                    else None,
                ],
            }
        )
        blockers[-1]["failures"] = [
            item for item in blockers[-1]["failures"] if item is not None
        ]
    if controller.get("status") != "promoted" or not threshold_controller_promoted:
        threshold_failures: list[str] = []
        for source in (
            monitor.get("shadow_promotion_failures"),
            threshold_no_backfill_monitor.get("promotion_gate_failures"),
            threshold_no_backfill_failure.get("failure_modes"),
            controller.get("promotion_gate_failures"),
        ):
            if isinstance(source, list):
                threshold_failures.extend(str(item) for item in source)
            elif source:
                threshold_failures.append(str(source))
        threshold_failures = list(dict.fromkeys(threshold_failures))
        blockers.append(
            {
                "component": "threshold_controller",
                "severity": "promotion",
                "reason": "threshold_controller_not_promoted",
                "failures": threshold_failures,
            }
        )
    if rank_scope_validation and rank_scope_validation.get("shadow_promotion_gate_passed") is not True:
        blockers.append(
            {
                "component": "rank_scope_router",
                "severity": "promotion",
                "reason": "rank_scope_router_later_window_gate_failed",
                "failures": rank_scope_validation.get("shadow_promotion_failures"),
            }
        )
    if priority.get("status") != "promoted" or not priority_promoted:
        priority_failures = (
            priority_headmix.get("promotion_gate_failures")
            or exchangefixed_priority.get("promotion_gate_failures")
            or priority.get("promotion_gate_failures")
        )
        blockers.append(
            {
                "component": "head_priority_modulation",
                "severity": "promotion",
                "reason": "priority_modulation_shadow_only",
                "failures": priority_failures,
            }
        )

    next_actions: list[str] = []
    if int(monitor_discovery.get("appendable_candidate_count") or 0) > 0:
        next_actions.append("append_discovered_shadow_controller_windows_to_monitor")
    else:
        next_actions.append("keep_scoring_new_later_windows_until_appendable_shadow_bundles_exist")
    if rank_scope_validation:
        next_actions.append("keep_rank_scope_router_shadow_only_until_later_window_gate_passes")
    if response_gate.get("quality_gate_passed") is True:
        next_actions.append("do_not_reopen_response_quality_gate_unless_new_data_invalidates_it")
    if threshold_no_backfill.get("selected_arm"):
        next_actions.append("score_no_backfill_threshold_candidate_in_shadow_on_later_matured_windows")
    if threshold_no_backfill_shadow:
        latest_delta = threshold_no_backfill_shadow.get("total_net_pnl_delta")
        if latest_delta is not None and float(latest_delta) < 0.0:
            next_actions.append("keep_no_backfill_threshold_candidate_shadow_only_after_negative_later_score")
    if threshold_no_backfill_discovery:
        if (no_backfill_discovery_appendable_count or 0) > 0:
            next_actions.append("append_no_backfill_shadow_score_windows_to_monitor")
        else:
            if threshold_no_backfill_next_readiness.get("scoreable_min_window_now") is True:
                next_actions.append("materialize_or_score_next_no_backfill_shadow_window")
            elif threshold_no_backfill_next_readiness:
                next_actions.append("wait_for_or_generate_more_feature_history_before_next_shadow_score")
            else:
                next_actions.append("audit_next_no_backfill_shadow_window_readiness")
    elif threshold_no_backfill_next_readiness:
        if threshold_no_backfill_next_readiness.get("scoreable_min_window_now") is True:
            next_actions.append("materialize_or_score_next_no_backfill_shadow_window")
        else:
            next_actions.append("wait_for_or_generate_more_feature_history_before_next_shadow_score")
    if threshold_no_backfill_monitor:
        positive_share = threshold_no_backfill_monitor.get("positive_delta_window_share")
        if positive_share is not None and float(positive_share) <= 0.0:
            next_actions.append("do_not_promote_no_backfill_threshold_candidate_without_new_positive_windows")
        direct_gate_passed = threshold_no_backfill_monitor.get(
            "direct_threshold_only_promotion_gate_passed"
        )
        direct_failures = threshold_no_backfill_monitor.get(
            "direct_threshold_only_promotion_gate_failures"
        )
        if direct_gate_passed is False or bool(direct_failures):
            next_actions.append(
                "retrain_threshold_controller_for_recurrent_direct_accepted_trade_suppression"
            )
            if not threshold_direct_ledger:
                next_actions.append("build_direct_suppression_training_ledger")
            elif not threshold_direct_training:
                next_actions.append("train_threshold_controller_on_direct_suppression_training_ledger")
        locked_gate_passed = threshold_no_backfill_monitor.get(
            "locked_accepted_overlay_promotion_gate_passed"
        )
        locked_failures = threshold_no_backfill_monitor.get(
            "locked_accepted_overlay_promotion_gate_failures"
        )
        if locked_gate_passed is False or bool(locked_failures):
            next_actions.append(
                "keep_locked_accepted_overlay_shadow_only_until_suppression_is_recurrent"
            )
        if (
            no_backfill_monitor_additional_windows_needed is not None
            and no_backfill_monitor_additional_windows_needed > 0
        ):
            next_actions.append(
                f"score_{no_backfill_monitor_additional_windows_needed}_additional_no_backfill_shadow_window"
            )
    if threshold_direct_ledger and not threshold_direct_training:
        next_actions.append("train_threshold_controller_on_direct_suppression_training_ledger")
    elif threshold_no_backfill and not threshold_direct_ledger:
        next_actions.append("build_direct_suppression_training_ledger")
    if threshold_direct_training and not threshold_direct_training.get("selected_arm"):
        next_actions.append("accumulate_more_direct_suppression_support_before_shadow_policy_activation")
    if threshold_no_backfill_failure.get("indirect_removed_count"):
        next_actions.append("redesign_threshold_controller_to_constrain_indirect_path_suppression")
    if priority_headmix.get("recurrent_challenger_selected") is False:
        next_actions.append("keep_priority_modulation_shadow_only_until_recurrent_accepted_set_gate_passes")
    next_actions = list(dict.fromkeys(next_actions))

    status = {
        "generated_by": "report_market_state_operational_status",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "active_stack_name": active.get("name"),
        "active_score_column": active.get("active_score_column"),
        "active_policy_variant": active.get("policy_variant"),
        "active_rank_contract": active.get("rank_contract"),
        "active_rank_scope": active.get("rank_scope"),
        "active_heads": active.get("enabled_heads") or [],
        "disabled_heads": active.get("disabled_heads") or [],
        "qfail_active": _bool(active.get("qfail_active")),
        "native_reliability_blend_active": _bool(active.get("native_reliability_blend_active")),
        "market_state_shadow_logging_only": _bool(active.get("market_state_shadow_logging_only")),
        "threshold_controller_active": _bool(active.get("market_state_threshold_controller_active")),
        "priority_modulation_active": _bool(active.get("market_state_priority_modulation_active")),
        "experiment_baseline_name": experiment.get("name"),
        "experiment_rank_contract": experiment.get("rank_contract"),
        "experiment_rank_scope": experiment.get("rank_scope"),
        "experiment_policy_variant": experiment.get("policy_variant"),
        "experiment_active_heads": experiment.get("enabled_heads") or [],
        "experiment_disabled_heads": experiment.get("disabled_heads") or [],
        "experiment_controller_action_scope": experiment.get("controller_action_scope"),
        "experiment_controller_action_constraints": experiment.get("controller_action_constraints")
        or [],
        "experiment_attribution_note": experiment.get("attribution_note"),
        "experiment_contract_matches_active_stack": experiment_contract_matches_active_stack,
        "structural_complete": bool(structural_complete),
        "plan_status_counts": plan.get("status_counts") or {},
        "response_quality_gate_passed": response_gate.get("quality_gate_passed"),
        "response_quality_passing_heads": response_gate.get("quality_passing_heads") or [],
        "threshold_controller_status": controller.get("status"),
        "threshold_controller_global_rank_smoke_status": threshold_smoke.get("status"),
        "threshold_controller_global_rank_smoke_dir": threshold_smoke.get("artifact_dir"),
        "threshold_controller_global_rank_smoke_selected_arm": threshold_smoke.get("selected_arm"),
        "threshold_controller_global_rank_smoke_selection_reason": threshold_smoke.get(
            "selection_reason"
        ),
        "threshold_controller_global_rank_smoke_baseline_net_pnl": threshold_smoke.get(
            "baseline_net_pnl"
        ),
        "threshold_controller_global_rank_smoke_best_full_delta_net_pnl": threshold_smoke.get(
            "best_full_replay_delta_net_pnl"
        ),
        "threshold_controller_global_rank_smoke_best_overlay_delta_net_pnl": threshold_smoke.get(
            "best_post_selection_overlay_delta_net_pnl"
        ),
        "threshold_controller_global_rank_smoke_baseline_accepted_defensive_success": threshold_smoke.get(
            "baseline_accepted_defensive_success"
        ),
        "threshold_controller_global_rank_walkforward_status": threshold_walkforward.get("status"),
        "threshold_controller_global_rank_walkforward_dir": threshold_walkforward.get("artifact_dir"),
        "threshold_controller_global_rank_walkforward_selected_arm": threshold_walkforward.get(
            "selected_arm"
        ),
        "threshold_controller_global_rank_walkforward_selection_reason": threshold_walkforward.get(
            "selection_reason"
        ),
        "threshold_controller_global_rank_walkforward_best_full_median_delta_net_pnl": threshold_walkforward.get(
            "best_full_replay_median_delta_net_pnl"
        ),
        "threshold_controller_global_rank_walkforward_best_full_q25_delta_net_pnl": threshold_walkforward.get(
            "best_full_replay_q25_delta_net_pnl"
        ),
        "threshold_controller_global_rank_walkforward_best_full_positive_delta_share": threshold_walkforward.get(
            "best_full_replay_positive_delta_share"
        ),
        "threshold_controller_global_rank_walkforward_best_overlay_median_delta_net_pnl": threshold_walkforward.get(
            "best_post_selection_overlay_median_delta_net_pnl"
        ),
        "threshold_controller_global_rank_walkforward_best_overlay_q25_delta_net_pnl": threshold_walkforward.get(
            "best_post_selection_overlay_q25_delta_net_pnl"
        ),
        "threshold_controller_global_rank_walkforward_best_overlay_positive_delta_share": threshold_walkforward.get(
            "best_post_selection_overlay_positive_delta_share"
        ),
        "threshold_controller_global_rank_walkforward_gate_passed": threshold_walkforward.get(
            "promotion_gate_passed"
        ),
        "threshold_controller_global_rank_walkforward_failures": threshold_walkforward.get(
            "promotion_gate_failures"
        )
        or [],
        "threshold_controller_t1_timestamp_walkforward_status": t1_timestamp_walkforward.get(
            "status"
        ),
        "threshold_controller_t1_timestamp_walkforward_dir": t1_timestamp_walkforward.get(
            "artifact_dir"
        ),
        "threshold_controller_t1_timestamp_walkforward_selected_arm": t1_timestamp_walkforward.get(
            "selected_arm"
        ),
        "threshold_controller_t1_timestamp_walkforward_selection_reason": t1_timestamp_walkforward.get(
            "selection_reason"
        ),
        "threshold_controller_t1_timestamp_walkforward_rank_contract": t1_timestamp_walkforward.get(
            "rank_contract"
        ),
        "threshold_controller_t1_timestamp_walkforward_folds": t1_timestamp_walkforward.get(
            "folds"
        ),
        "threshold_controller_t1_timestamp_walkforward_median_delta_net_pnl": t1_timestamp_walkforward.get(
            "median_delta_net_pnl"
        ),
        "threshold_controller_t1_timestamp_walkforward_q25_delta_net_pnl": t1_timestamp_walkforward.get(
            "q25_delta_net_pnl"
        ),
        "threshold_controller_t1_timestamp_walkforward_positive_delta_share": t1_timestamp_walkforward.get(
            "positive_delta_share"
        ),
        "threshold_controller_t1_timestamp_walkforward_accepted_frontier_candidate_total": t1_timestamp_walkforward.get(
            "accepted_frontier_candidate_total"
        ),
        "threshold_controller_t1_timestamp_walkforward_accepted_frontier_suppressed_total": t1_timestamp_walkforward.get(
            "accepted_frontier_suppressed_total"
        ),
        "threshold_controller_t1_timestamp_walkforward_promotion_gate_passed": t1_timestamp_walkforward.get(
            "promotion_gate_passed"
        ),
        "threshold_controller_t1_timestamp_walkforward_failures": t1_timestamp_walkforward.get(
            "promotion_gate_failures"
        )
        or [],
        "threshold_controller_no_backfill_status": threshold_no_backfill.get("status"),
        "threshold_controller_no_backfill_dir": threshold_no_backfill.get("artifact_dir"),
        "threshold_controller_no_backfill_bundle_dir": threshold_no_backfill.get("bundle_dir"),
        "threshold_controller_no_backfill_selected_arm": threshold_no_backfill.get("selected_arm"),
        "threshold_controller_no_backfill_selected_median_delta_net_pnl": threshold_no_backfill.get(
            "selected_median_delta_net_pnl"
        ),
        "threshold_controller_no_backfill_selected_q25_delta_net_pnl": threshold_no_backfill.get(
            "selected_q25_delta_net_pnl"
        ),
        "threshold_controller_no_backfill_positive_delta_share": threshold_no_backfill.get(
            "selected_positive_delta_share"
        ),
        "threshold_controller_no_backfill_removed_trades": threshold_no_backfill.get(
            "selected_removed_trades"
        ),
        "threshold_controller_no_backfill_defensive_success": threshold_no_backfill.get(
            "selected_defensive_success"
        ),
        "threshold_controller_no_backfill_winner_pnl_sacrificed": threshold_no_backfill.get(
            "selected_winner_pnl_sacrificed"
        ),
        "threshold_controller_no_backfill_action_entrants": threshold_no_backfill.get(
            "selected_action_entrants"
        ),
        "threshold_controller_no_backfill_bundle_execution_enabled": threshold_no_backfill.get(
            "bundle_controller_execution_enabled"
        ),
        "threshold_controller_no_backfill_promotion_gate_passed": threshold_no_backfill.get(
            "promotion_gate_passed"
        ),
        "threshold_controller_no_backfill_failures": threshold_no_backfill.get(
            "promotion_gate_failures"
        )
        or [],
        "threshold_controller_direct_suppression_ledger_contract": threshold_direct_ledger.get(
            "artifact_contract"
        ),
        "threshold_controller_direct_suppression_ledger_variant": threshold_direct_ledger_variant,
        "threshold_controller_direct_suppression_ledger_aggregation_contract": threshold_direct_ledger.get(
            "aggregation_contract"
        ),
        "threshold_controller_direct_suppression_ledger_dir": threshold_direct_ledger.get(
            "ledger_dir"
        ),
        "threshold_controller_direct_suppression_ledger_parquet": threshold_direct_ledger.get(
            "ledger_parquet"
        ),
        "threshold_controller_direct_suppression_ledger_report": threshold_direct_ledger.get(
            "report_md"
        ),
        "threshold_controller_direct_suppression_ledger_rows": threshold_direct_ledger.get(
            "row_count"
        ),
        "threshold_controller_direct_suppression_ledger_unique_keys": threshold_direct_ledger.get(
            "unique_decision_key_count"
        ),
        "threshold_controller_direct_suppression_ledger_profitable_rate": threshold_direct_ledger.get(
            "direct_profitable_rate"
        ),
        "threshold_controller_direct_suppression_ledger_mean_utility": threshold_direct_ledger.get(
            "mean_direct_defensive_utility"
        ),
        "threshold_controller_direct_suppression_ledger_total_utility": threshold_direct_ledger.get(
            "total_direct_defensive_utility"
        ),
        "threshold_controller_direct_suppression_ledger_current_schedule_suppressed_rows": threshold_direct_ledger.get(
            "current_schedule_suppressed_rows"
        ),
        "threshold_controller_direct_suppression_ledger_current_schedule_utility": threshold_direct_ledger.get(
            "current_schedule_defensive_utility"
        ),
        "threshold_controller_direct_suppression_ledger_short_asset_mean_utility": (
            (threshold_direct_ledger.get("by_head") or {}).get("short_asset") or {}
        ).get("mean_direct_defensive_utility"),
        "threshold_controller_direct_suppression_ledger_short_boll_mean_utility": (
            (threshold_direct_ledger.get("by_head") or {}).get("short_boll") or {}
        ).get("mean_direct_defensive_utility"),
        "threshold_controller_direct_suppression_ledger_interpretation": threshold_direct_ledger.get(
            "interpretation"
        ),
        "threshold_controller_direct_suppression_training_contract": threshold_direct_training.get(
            "artifact_contract"
        ),
        "threshold_controller_direct_suppression_training_variant": threshold_direct_training_variant,
        "threshold_controller_direct_suppression_training_dir": threshold_direct_training.get(
            "training_dir"
        ),
        "threshold_controller_direct_suppression_training_summary": threshold_direct_training.get(
            "summary_json"
        ),
        "threshold_controller_direct_suppression_training_report": threshold_direct_training.get(
            "report_md"
        ),
        "threshold_controller_direct_suppression_training_model_path": threshold_direct_training.get(
            "model_path"
        ),
        "threshold_controller_direct_suppression_training_model_mode": threshold_direct_training.get(
            "model_mode"
        ),
        "threshold_controller_direct_suppression_training_oof_rows": threshold_direct_training.get(
            "oof_rows"
        ),
        "threshold_controller_direct_suppression_training_oof_unique_keys": threshold_direct_training.get(
            "oof_unique_decision_keys"
        ),
        "threshold_controller_direct_suppression_training_oof_auc": threshold_direct_training.get(
            "oof_profit_auc"
        ),
        "threshold_controller_direct_suppression_training_oof_ap": threshold_direct_training.get(
            "oof_average_precision"
        ),
        "threshold_controller_direct_suppression_training_utility_spearman": threshold_direct_training.get(
            "oof_utility_spearman"
        ),
        "threshold_controller_direct_suppression_training_selected_arm": threshold_direct_training.get(
            "selected_arm"
        ),
        "threshold_controller_direct_suppression_training_selection_reason": threshold_direct_training.get(
            "selection_reason"
        ),
        "threshold_controller_direct_suppression_training_best_attempt_arm": threshold_direct_training.get(
            "best_attempt_controller_arm"
        ),
        "threshold_controller_direct_suppression_training_best_attempt_rows": threshold_direct_training.get(
            "best_attempt_suppressed_rows"
        ),
        "threshold_controller_direct_suppression_training_best_attempt_success": threshold_direct_training.get(
            "best_attempt_defensive_success"
        ),
        "threshold_controller_direct_suppression_training_promotion_allowed": threshold_direct_training.get(
            "promotion_allowed"
        ),
        "threshold_controller_direct_suppression_training_interpretation": threshold_direct_training.get(
            "interpretation"
        ),
        "threshold_controller_no_backfill_shadow_status": threshold_no_backfill_shadow.get("status"),
        "threshold_controller_no_backfill_shadow_score_dir": threshold_no_backfill_shadow.get("score_dir"),
        "threshold_controller_no_backfill_shadow_bundle_dir": threshold_no_backfill_shadow.get("bundle_dir"),
        "threshold_controller_no_backfill_shadow_eval_feature_store_dir": threshold_no_backfill_shadow.get(
            "eval_feature_store_dir"
        ),
        "threshold_controller_no_backfill_shadow_period_start": threshold_no_backfill_shadow.get(
            "period_start"
        ),
        "threshold_controller_no_backfill_shadow_period_end": threshold_no_backfill_shadow.get(
            "period_end"
        ),
        "threshold_controller_no_backfill_shadow_baseline_net_pnl": threshold_no_backfill_shadow.get(
            "baseline_net_pnl"
        ),
        "threshold_controller_no_backfill_shadow_net_pnl": threshold_no_backfill_shadow.get(
            "shadow_net_pnl"
        ),
        "threshold_controller_no_backfill_shadow_total_delta_net_pnl": threshold_no_backfill_shadow.get(
            "total_net_pnl_delta"
        ),
        "threshold_controller_no_backfill_shadow_baseline_trades": threshold_no_backfill_shadow.get(
            "baseline_trade_count"
        ),
        "threshold_controller_no_backfill_shadow_trades": threshold_no_backfill_shadow.get(
            "shadow_trade_count"
        ),
        "threshold_controller_no_backfill_shadow_removed_trades": threshold_no_backfill_shadow.get(
            "removed_trade_count"
        ),
        "threshold_controller_no_backfill_shadow_added_trades": threshold_no_backfill_shadow.get(
            "added_trade_count"
        ),
        "threshold_controller_no_backfill_shadow_removed_loss_avoided": threshold_no_backfill_shadow.get(
            "removed_loss_avoided"
        ),
        "threshold_controller_no_backfill_shadow_winner_pnl_sacrificed": threshold_no_backfill_shadow.get(
            "removed_winner_pnl_sacrificed"
        ),
        "threshold_controller_no_backfill_shadow_common_net_pnl_delta": threshold_no_backfill_shadow.get(
            "common_net_pnl_delta"
        ),
        "threshold_controller_no_backfill_shadow_subset": threshold_no_backfill_shadow.get(
            "shadow_subset_of_baseline"
        ),
        "threshold_controller_no_backfill_shadow_feature_coverage": threshold_no_backfill_shadow.get(
            "eval_feature_store_timestamp_coverage"
        ),
        "threshold_controller_no_backfill_shadow_source_feature_count": threshold_no_backfill_shadow.get(
            "eval_source_feature_count"
        ),
        "threshold_controller_no_backfill_shadow_bundle_contract_audit_json": threshold_no_backfill_shadow.get(
            "bundle_contract_audit_json"
        ),
        "threshold_controller_no_backfill_shadow_bundle_contract_audit_expected_rank_contract": threshold_no_backfill_shadow.get(
            "bundle_contract_audit_expected_rank_contract"
        ),
        "threshold_controller_no_backfill_shadow_bundle_contract_audit_passed": threshold_no_backfill_shadow.get(
            "bundle_contract_audit_passed"
        ),
        "threshold_controller_no_backfill_shadow_bundle_contract_audit_completion_grade_passed": threshold_no_backfill_shadow.get(
            "bundle_contract_audit_completion_grade_passed"
        ),
        "threshold_controller_no_backfill_shadow_promotion_gate_passed": threshold_no_backfill_shadow.get(
            "promotion_gate_passed"
        ),
        "threshold_controller_no_backfill_shadow_failures": threshold_no_backfill_shadow.get(
            "promotion_gate_failures"
        )
        or [],
        "threshold_controller_no_backfill_monitor_status": threshold_no_backfill_monitor.get("status"),
        "threshold_controller_no_backfill_monitor_dir": threshold_no_backfill_monitor.get("monitor_dir"),
        "threshold_controller_no_backfill_monitor_window_count": threshold_no_backfill_monitor.get(
            "window_count"
        ),
        "threshold_controller_no_backfill_monitor_min_later_window_count": threshold_no_backfill_monitor.get(
            "min_later_window_count"
        ),
        "threshold_controller_no_backfill_monitor_additional_windows_needed": no_backfill_monitor_additional_windows_needed,
        "threshold_controller_no_backfill_monitor_latest_window_end": no_backfill_monitor_latest_end,
        "threshold_controller_no_backfill_monitor_window_periods": no_backfill_monitor_window_periods,
        "threshold_controller_no_backfill_window_discovery_dir": threshold_no_backfill_discovery.get(
            "discovery_dir"
        ),
        "threshold_controller_no_backfill_window_discovery_json": threshold_no_backfill_discovery.get(
            "summary_json"
        ),
        "threshold_controller_no_backfill_window_discovery_report": threshold_no_backfill_discovery.get(
            "report_md"
        ),
        "threshold_controller_no_backfill_window_discovery_discovered_count": threshold_no_backfill_discovery.get(
            "discovered_candidate_count"
        ),
        "threshold_controller_no_backfill_window_discovery_appendable_count": threshold_no_backfill_discovery.get(
            "appendable_candidate_count"
        ),
        "threshold_controller_no_backfill_window_discovery_already_monitored_count": threshold_no_backfill_discovery.get(
            "already_monitored_count"
        ),
        "threshold_controller_no_backfill_window_discovery_failed_count": threshold_no_backfill_discovery.get(
            "failed_candidate_count"
        ),
        "threshold_controller_no_backfill_window_discovery_latest_end": no_backfill_discovery_latest_end,
        "threshold_controller_no_backfill_window_discovery_readiness_csv": threshold_no_backfill_discovery.get(
            "readiness_csv"
        ),
        "threshold_controller_no_backfill_window_discovery_appendable_csv": threshold_no_backfill_discovery.get(
            "appendable_csv"
        ),
        "threshold_controller_no_backfill_next_readiness_dir": threshold_no_backfill_next_readiness.get(
            "readiness_dir"
        ),
        "threshold_controller_no_backfill_next_readiness_status": threshold_no_backfill_next_readiness.get(
            "status"
        ),
        "threshold_controller_no_backfill_next_readiness_report": threshold_no_backfill_next_readiness.get(
            "report_md"
        ),
        "threshold_controller_no_backfill_next_feature_store_dir": threshold_no_backfill_next_readiness.get(
            "feature_store_dir"
        ),
        "threshold_controller_no_backfill_next_feature_timestamp_max": threshold_no_backfill_next_readiness.get(
            "feature_timestamp_max"
        ),
        "threshold_controller_no_backfill_next_maturity_buffer_hours": threshold_no_backfill_next_readiness.get(
            "maturity_buffer_hours"
        ),
        "threshold_controller_no_backfill_next_maturity_cutoff": threshold_no_backfill_next_readiness.get(
            "maturity_cutoff"
        ),
        "threshold_controller_no_backfill_next_window_start": threshold_no_backfill_next_readiness.get(
            "next_window_start"
        ),
        "threshold_controller_no_backfill_next_target_window_end": threshold_no_backfill_next_readiness.get(
            "target_window_end"
        ),
        "threshold_controller_no_backfill_next_proposed_scoreable_window_end": threshold_no_backfill_next_readiness.get(
            "proposed_scoreable_window_end"
        ),
        "threshold_controller_no_backfill_next_mature_timestamp_count": threshold_no_backfill_next_readiness.get(
            "mature_timestamp_count_available"
        ),
        "threshold_controller_no_backfill_next_min_timestamp_count": threshold_no_backfill_next_readiness.get(
            "min_timestamp_count"
        ),
        "threshold_controller_no_backfill_next_target_window_hours": threshold_no_backfill_next_readiness.get(
            "target_window_hours"
        ),
        "threshold_controller_no_backfill_next_scoreable_min_window_now": threshold_no_backfill_next_readiness.get(
            "scoreable_min_window_now"
        ),
        "threshold_controller_no_backfill_next_scoreable_full_window_now": threshold_no_backfill_next_readiness.get(
            "scoreable_full_window_now"
        ),
        "threshold_controller_no_backfill_next_missing_feature_hours_min": threshold_no_backfill_next_readiness.get(
            "missing_feature_hours_for_min_window"
        ),
        "threshold_controller_no_backfill_next_missing_feature_hours_full": threshold_no_backfill_next_readiness.get(
            "missing_feature_hours_for_full_window"
        ),
        "threshold_controller_no_backfill_next_failures": threshold_no_backfill_next_readiness.get(
            "failures"
        )
        or [],
        "threshold_controller_no_backfill_next_action": threshold_no_backfill_next_readiness.get(
            "next_action"
        ),
        "threshold_controller_no_backfill_next_runner_status": threshold_no_backfill_next_runner.get(
            "status"
        ),
        "threshold_controller_no_backfill_next_runner_reason": threshold_no_backfill_next_runner.get(
            "reason"
        ),
        "threshold_controller_no_backfill_next_runner_dir": threshold_no_backfill_next_runner.get(
            "runner_dir"
        ),
        "threshold_controller_no_backfill_next_runner_manifest": threshold_no_backfill_next_runner.get(
            "runner_manifest"
        ),
        "threshold_controller_no_backfill_next_runner_planned_step_count": threshold_no_backfill_next_runner.get(
            "planned_step_count"
        ),
        "threshold_controller_no_backfill_next_runner_completed_steps": threshold_no_backfill_next_runner.get(
            "completed_steps"
        )
        or [],
        "threshold_controller_no_backfill_monitor_positive_share": threshold_no_backfill_monitor.get(
            "positive_delta_window_share"
        ),
        "threshold_controller_no_backfill_monitor_action_only_positive_share": threshold_no_backfill_monitor.get(
            "action_only_positive_window_share"
        ),
        "threshold_controller_no_backfill_monitor_direct_threshold_only_available_window_count": threshold_no_backfill_monitor.get(
            "direct_threshold_only_available_window_count"
        ),
        "threshold_controller_no_backfill_monitor_direct_threshold_only_positive_share": threshold_no_backfill_monitor.get(
            "direct_threshold_only_positive_window_share"
        ),
        "threshold_controller_no_backfill_monitor_direct_threshold_only_suppression_share": threshold_no_backfill_monitor.get(
            "direct_threshold_only_suppression_window_share"
        ),
        "threshold_controller_no_backfill_monitor_direct_threshold_only_promotion_gate_passed": threshold_no_backfill_monitor.get(
            "direct_threshold_only_promotion_gate_passed"
        ),
        "threshold_controller_no_backfill_monitor_direct_threshold_only_failures": threshold_no_backfill_monitor.get(
            "direct_threshold_only_promotion_gate_failures"
        )
        or [],
        "threshold_controller_no_backfill_monitor_median_delta": threshold_no_backfill_monitor.get(
            "median_total_net_pnl_delta"
        ),
        "threshold_controller_no_backfill_monitor_q25_delta": threshold_no_backfill_monitor.get(
            "q25_total_net_pnl_delta"
        ),
        "threshold_controller_no_backfill_monitor_total_delta": threshold_no_backfill_monitor.get(
            "sum_total_net_pnl_delta"
        ),
        "threshold_controller_no_backfill_monitor_full_path_delta": threshold_no_backfill_monitor.get(
            "sum_full_path_replay_net_pnl_delta"
        ),
        "threshold_controller_no_backfill_monitor_action_only_delta": threshold_no_backfill_monitor.get(
            "sum_action_only_fixed_common_size_net_pnl_delta"
        ),
        "threshold_controller_no_backfill_monitor_direct_threshold_only_delta": threshold_no_backfill_monitor.get(
            "sum_direct_threshold_only_net_pnl_delta"
        ),
        "threshold_controller_no_backfill_monitor_direct_threshold_only_removed_trades": threshold_no_backfill_monitor.get(
            "sum_direct_threshold_only_removed_trade_count"
        ),
        "threshold_controller_no_backfill_monitor_direct_threshold_only_defensive_success": threshold_no_backfill_monitor.get(
            "sum_direct_threshold_only_defensive_success"
        ),
        "threshold_controller_no_backfill_monitor_locked_overlay_available_windows": threshold_no_backfill_monitor.get(
            "locked_accepted_overlay_available_window_count"
        ),
        "threshold_controller_no_backfill_monitor_locked_overlay_positive_share": threshold_no_backfill_monitor.get(
            "locked_accepted_overlay_positive_window_share"
        ),
        "threshold_controller_no_backfill_monitor_locked_overlay_suppression_share": threshold_no_backfill_monitor.get(
            "locked_accepted_overlay_suppression_window_share"
        ),
        "threshold_controller_no_backfill_monitor_locked_overlay_delta": threshold_no_backfill_monitor.get(
            "sum_locked_accepted_overlay_net_pnl_delta"
        ),
        "threshold_controller_no_backfill_monitor_locked_overlay_removed_trades": threshold_no_backfill_monitor.get(
            "sum_locked_accepted_overlay_removed_trade_count"
        ),
        "threshold_controller_no_backfill_monitor_locked_overlay_defensive_success": threshold_no_backfill_monitor.get(
            "sum_locked_accepted_overlay_defensive_success"
        ),
        "threshold_controller_no_backfill_monitor_locked_overlay_promotion_gate_passed": threshold_no_backfill_monitor.get(
            "locked_accepted_overlay_promotion_gate_passed"
        ),
        "threshold_controller_no_backfill_monitor_locked_overlay_failures": threshold_no_backfill_monitor.get(
            "locked_accepted_overlay_promotion_gate_failures"
        )
        or [],
        "threshold_controller_no_backfill_monitor_indirect_path_delta": threshold_no_backfill_monitor.get(
            "sum_indirect_path_or_capacity_net_pnl_delta"
        ),
        "threshold_controller_no_backfill_monitor_indirect_removed_trades": threshold_no_backfill_monitor.get(
            "sum_indirect_path_or_capacity_removed_trade_count"
        ),
        "threshold_controller_no_backfill_monitor_indirect_defensive_success": threshold_no_backfill_monitor.get(
            "sum_indirect_path_or_capacity_defensive_success"
        ),
        "threshold_controller_no_backfill_monitor_indirect_winner_pnl_sacrificed": threshold_no_backfill_monitor.get(
            "sum_indirect_path_or_capacity_winner_pnl_sacrificed"
        ),
        "threshold_controller_no_backfill_monitor_path_dependent_delta": threshold_no_backfill_monitor.get(
            "sum_path_dependent_common_trade_net_pnl_delta"
        ),
        "threshold_controller_no_backfill_monitor_baseline_net_pnl": threshold_no_backfill_monitor.get(
            "sum_baseline_net_pnl"
        ),
        "threshold_controller_no_backfill_monitor_shadow_net_pnl": threshold_no_backfill_monitor.get(
            "sum_shadow_net_pnl"
        ),
        "threshold_controller_no_backfill_monitor_removed_trades": threshold_no_backfill_monitor.get(
            "sum_removed_trade_count"
        ),
        "threshold_controller_no_backfill_monitor_added_trades": threshold_no_backfill_monitor.get(
            "sum_added_trade_count"
        ),
        "threshold_controller_no_backfill_monitor_removed_loss_avoided": threshold_no_backfill_monitor.get(
            "sum_removed_loss_avoided"
        ),
        "threshold_controller_no_backfill_monitor_winner_pnl_sacrificed": threshold_no_backfill_monitor.get(
            "sum_removed_winner_pnl_sacrificed"
        ),
        "threshold_controller_no_backfill_monitor_common_net_pnl_delta": threshold_no_backfill_monitor.get(
            "sum_common_net_pnl_delta"
        ),
        "threshold_controller_no_backfill_monitor_min_feature_coverage": threshold_no_backfill_monitor.get(
            "min_eval_feature_store_timestamp_coverage"
        ),
        "threshold_controller_no_backfill_monitor_min_source_feature_count": threshold_no_backfill_monitor.get(
            "min_eval_source_feature_count"
        ),
        "threshold_controller_no_backfill_monitor_artifact_hashes_complete": threshold_no_backfill_monitor.get(
            "all_score_manifest_artifact_hashes_complete"
        ),
        "threshold_controller_no_backfill_monitor_score_manifest_contract_versions": threshold_no_backfill_monitor.get(
            "score_manifest_contract_versions"
        )
        or [],
        "threshold_controller_no_backfill_monitor_windows_missing_score_input_hash_fields": threshold_no_backfill_monitor.get(
            "windows_missing_score_input_hash_fields"
        ),
        "threshold_controller_no_backfill_monitor_windows_missing_required_output_hashes": threshold_no_backfill_monitor.get(
            "windows_missing_required_output_hashes"
        ),
        "threshold_controller_no_backfill_monitor_promotion_gate_passed": threshold_no_backfill_monitor.get(
            "promotion_gate_passed"
        ),
        "threshold_controller_no_backfill_monitor_failures": threshold_no_backfill_monitor.get(
            "promotion_gate_failures"
        )
        or [],
        "threshold_controller_no_backfill_failure_diagnostics_dir": threshold_no_backfill_failure.get(
            "diagnostics_dir"
        ),
        "threshold_controller_no_backfill_failure_removed_count": threshold_no_backfill_failure.get(
            "removed_trade_count"
        ),
        "threshold_controller_no_backfill_failure_direct_removed_count": threshold_no_backfill_failure.get(
            "direct_removed_count"
        ),
        "threshold_controller_no_backfill_failure_indirect_removed_count": threshold_no_backfill_failure.get(
            "indirect_removed_count"
        ),
        "threshold_controller_no_backfill_failure_direct_defensive_success": threshold_no_backfill_failure.get(
            "direct_defensive_success"
        ),
        "threshold_controller_no_backfill_failure_indirect_defensive_success": threshold_no_backfill_failure.get(
            "indirect_defensive_success"
        ),
        "threshold_controller_no_backfill_failure_removed_loss_avoided": threshold_no_backfill_failure.get(
            "removed_loss_avoided"
        ),
        "threshold_controller_no_backfill_failure_winner_pnl_sacrificed": threshold_no_backfill_failure.get(
            "removed_winner_pnl_sacrificed"
        ),
        "threshold_controller_no_backfill_failure_promotion_safe_subset_found": threshold_no_backfill_failure.get(
            "promotion_safe_subset_found"
        ),
        "threshold_controller_no_backfill_failure_modes": threshold_no_backfill_failure.get(
            "failure_modes"
        )
        or [],
        "threshold_controller_no_backfill_failure_report": threshold_no_backfill_failure.get(
            "report_md"
        ),
        "shadow_monitor_bundle_count": monitor.get("bundle_count"),
        "shadow_monitor_gate_passed": monitor.get("shadow_promotion_gate_passed"),
        "shadow_monitor_failures": monitor.get("shadow_promotion_failures") or [],
        "shadow_monitor_appendable_candidate_count": monitor_discovery.get("appendable_candidate_count"),
        "shadow_monitor_already_monitored_count": monitor_discovery.get("already_monitored_count"),
        "shadow_monitor_excluded_candidate_count": monitor_discovery.get("excluded_candidate_count"),
        "rank_scope_router_gate_passed": rank_scope_validation.get("shadow_promotion_gate_passed"),
        "rank_scope_router_failures": rank_scope_validation.get("shadow_promotion_failures") or [],
        "rank_scope_router_later_median_delta_net_pnl": rank_scope_validation.get(
            "later_blend_median_delta_net_pnl"
        ),
        "head_priority_modulation_status": priority.get("status"),
        "head_priority_modulation_failures": (
            exchangefixed_priority.get("promotion_gate_failures")
            or priority.get("promotion_gate_failures")
            or []
        ),
        "head_priority_exchangefixed_gate_passed": exchangefixed_priority.get(
            "promotion_gate_passed"
        ),
        "head_priority_exchangefixed_delta_net_pnl": exchangefixed_priority.get(
            "pruned_lgbm_delta_net_pnl"
        ),
        "head_priority_exchangefixed_accepted_jaccard": exchangefixed_priority.get(
            "pruned_lgbm_accepted_jaccard"
        ),
        "head_priority_exchangefixed_report_dir": exchangefixed_priority.get(
            "pruned_lgbm_cap_sweep_dir"
        )
        or exchangefixed_priority.get(
            "pruned_lgbm_xgb_dir"
        ),
        "head_priority_headmix_gate_passed": priority_headmix.get(
            "opportunity_routing_gate_passed"
        ),
        "head_priority_headmix_failures": priority_headmix.get("promotion_gate_failures") or [],
        "head_priority_headmix_window_count": priority_headmix.get("window_count"),
        "head_priority_headmix_median_delta_net_pnl": priority_headmix.get(
            "median_delta_net_pnl"
        ),
        "head_priority_headmix_q25_delta_net_pnl": priority_headmix.get("q25_delta_net_pnl"),
        "head_priority_headmix_positive_delta_share": priority_headmix.get(
            "positive_delta_window_share"
        ),
        "head_priority_headmix_min_shadow_active_head_count": priority_headmix.get(
            "min_shadow_active_head_count"
        ),
        "head_priority_headmix_max_shadow_dominant_head_share": priority_headmix.get(
            "max_shadow_dominant_head_share"
        ),
        "head_priority_headmix_median_head_trade_share_l1_delta": priority_headmix.get(
            "median_head_trade_share_l1_delta"
        ),
        "head_priority_headmix_max_head_trade_share_l1_delta": priority_headmix.get(
            "max_head_trade_share_l1_delta"
        ),
        "head_priority_headmix_report_dir": priority_headmix.get("audit_dir"),
        "head_priority_recurrent_challenger_selected": priority_headmix.get(
            "recurrent_challenger_selected"
        ),
        "head_priority_recurrent_selection_reason": priority_headmix.get(
            "recurrent_selection_reason"
        ),
        "head_priority_recurrent_best_candidate": priority_headmix.get(
            "recurrent_best_candidate"
        ),
        "head_priority_recurrent_best_candidate_failures": priority_headmix.get(
            "recurrent_best_candidate_failures"
        )
        or [],
        "head_priority_recurrent_challenger_json": priority_headmix.get(
            "recurrent_challenger_json"
        ),
        "production_ready": bool(
            structural_complete
            and experiment_contract_matches_active_stack
            and threshold_controller_promoted
            and not priority_promoted
            and not rank_router_promoted
        ),
        "blockers": blockers,
        "next_actions": next_actions,
        "interpretation": (
            "Market-state infrastructure is structurally complete, but active "
            "execution remains on the static T1 contract because threshold-controller, "
            "rank-router, and head-priority promotion evidence is not sufficient."
        ),
    }
    return status


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.fillna("").astype(str)
    columns = list(view.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(row[col] for col in columns) + " |")
    return "\n".join(lines)


def _render_report(status: dict[str, Any]) -> str:
    lines = [
        "# Market-State Operational Status",
        "",
        "This report separates implementation completeness from promotion readiness.",
        "",
        "## Active Stack",
        "",
        f"- Stack: `{status['active_stack_name']}`",
        f"- Score column: `{status['active_score_column']}`",
        f"- Policy variant: `{status['active_policy_variant']}`",
        f"- Rank contract: `{status['active_rank_contract']}`",
        f"- Rank scope: `{status['active_rank_scope']}`",
        f"- Active heads: `{', '.join(status['active_heads'])}`",
        f"- Disabled heads: `{', '.join(status['disabled_heads'])}`",
        f"- q-fail active: `{status['qfail_active']}`",
        f"- Native reliability blend active: `{status['native_reliability_blend_active']}`",
        f"- Market-state shadow logging only: `{status['market_state_shadow_logging_only']}`",
        f"- Threshold controller active: `{status['threshold_controller_active']}`",
        f"- Priority modulation active: `{status['priority_modulation_active']}`",
        "",
        "## Market-State Experiment Contract",
        "",
        f"- Baseline: `{status['experiment_baseline_name']}`",
        f"- Rank contract: `{status['experiment_rank_contract']}`",
        f"- Rank scope: `{status['experiment_rank_scope']}`",
        f"- Policy variant: `{status['experiment_policy_variant']}`",
        f"- Active heads: `{', '.join(status['experiment_active_heads'])}`",
        f"- Disabled heads: `{', '.join(status['experiment_disabled_heads'])}`",
        f"- Controller action scope: `{status['experiment_controller_action_scope']}`",
        f"- Controller constraints: `{', '.join(status['experiment_controller_action_constraints'])}`",
        f"- Matches active stack contract: `{status['experiment_contract_matches_active_stack']}`",
        f"- Attribution note: `{status['experiment_attribution_note']}`",
        "",
        "## Gates",
        "",
        f"- Structural complete: `{status['structural_complete']}`",
        f"- Response quality gate passed: `{status['response_quality_gate_passed']}`",
        f"- Shadow threshold-controller gate passed: `{status['shadow_monitor_gate_passed']}`",
        f"- Rank-scope router gate passed: `{status['rank_scope_router_gate_passed']}`",
        f"- Exchange-fixed head-priority gate passed: `{status['head_priority_exchangefixed_gate_passed']}`",
        f"- Head-mix opportunity priority gate passed: `{status['head_priority_headmix_gate_passed']}`",
        f"- Production ready: `{status['production_ready']}`",
        "",
        "## Latest Head-Priority Evidence",
        "",
        f"- Exchange-fixed delta net PnL: `{status['head_priority_exchangefixed_delta_net_pnl']}`",
        f"- Exchange-fixed accepted Jaccard: `{status['head_priority_exchangefixed_accepted_jaccard']}`",
        f"- Evidence dir: `{status['head_priority_exchangefixed_report_dir']}`",
        f"- Head-mix median delta net PnL: `{status['head_priority_headmix_median_delta_net_pnl']}`",
        f"- Head-mix q25 delta net PnL: `{status['head_priority_headmix_q25_delta_net_pnl']}`",
        f"- Head-mix positive delta share: `{status['head_priority_headmix_positive_delta_share']}`",
        f"- Head-mix min active heads: `{status['head_priority_headmix_min_shadow_active_head_count']}`",
        f"- Head-mix max dominant-head share: `{status['head_priority_headmix_max_shadow_dominant_head_share']}`",
        f"- Head-mix max accepted-share L1 movement: `{status['head_priority_headmix_max_head_trade_share_l1_delta']}`",
        f"- Head-mix failures: `{', '.join(status['head_priority_headmix_failures'])}`",
        f"- Head-mix evidence dir: `{status['head_priority_headmix_report_dir']}`",
        f"- Recurrent challenger selected: `{status['head_priority_recurrent_challenger_selected']}`",
        f"- Recurrent selection reason: `{status['head_priority_recurrent_selection_reason']}`",
        f"- Recurrent best candidate: `{status['head_priority_recurrent_best_candidate']}`",
        f"- Recurrent best candidate failures: `{', '.join(status['head_priority_recurrent_best_candidate_failures'])}`",
        f"- Recurrent challenger evidence: `{status['head_priority_recurrent_challenger_json']}`",
        "",
        "## Latest Threshold-Controller Smoke",
        "",
        f"- Status: `{status['threshold_controller_global_rank_smoke_status']}`",
        f"- Selected arm: `{status['threshold_controller_global_rank_smoke_selected_arm']}`",
        f"- Selection reason: `{status['threshold_controller_global_rank_smoke_selection_reason']}`",
        f"- Baseline net PnL: `{status['threshold_controller_global_rank_smoke_baseline_net_pnl']}`",
        f"- Best full-replay delta net PnL: `{status['threshold_controller_global_rank_smoke_best_full_delta_net_pnl']}`",
        f"- Best post-selection overlay delta net PnL: `{status['threshold_controller_global_rank_smoke_best_overlay_delta_net_pnl']}`",
        f"- Baseline-accepted defensive success: `{status['threshold_controller_global_rank_smoke_baseline_accepted_defensive_success']}`",
        f"- Evidence dir: `{status['threshold_controller_global_rank_smoke_dir']}`",
        "",
        "## Latest Threshold-Controller Walk-Forward",
        "",
        f"- Status: `{status['threshold_controller_global_rank_walkforward_status']}`",
        f"- Selected arm: `{status['threshold_controller_global_rank_walkforward_selected_arm']}`",
        f"- Selection reason: `{status['threshold_controller_global_rank_walkforward_selection_reason']}`",
        f"- Best full median delta net PnL: `{status['threshold_controller_global_rank_walkforward_best_full_median_delta_net_pnl']}`",
        f"- Best full q25 delta net PnL: `{status['threshold_controller_global_rank_walkforward_best_full_q25_delta_net_pnl']}`",
        f"- Best full positive-delta share: `{status['threshold_controller_global_rank_walkforward_best_full_positive_delta_share']}`",
        f"- Best overlay median delta net PnL: `{status['threshold_controller_global_rank_walkforward_best_overlay_median_delta_net_pnl']}`",
        f"- Best overlay q25 delta net PnL: `{status['threshold_controller_global_rank_walkforward_best_overlay_q25_delta_net_pnl']}`",
        f"- Best overlay positive-delta share: `{status['threshold_controller_global_rank_walkforward_best_overlay_positive_delta_share']}`",
        f"- Gate passed: `{status['threshold_controller_global_rank_walkforward_gate_passed']}`",
        f"- Failures: `{', '.join(status['threshold_controller_global_rank_walkforward_failures'])}`",
        f"- Evidence dir: `{status['threshold_controller_global_rank_walkforward_dir']}`",
        "",
        "## T1 Timestamp Accepted-Frontier Walk-Forward",
        "",
        f"- Status: `{status['threshold_controller_t1_timestamp_walkforward_status']}`",
        f"- Rank contract: `{status['threshold_controller_t1_timestamp_walkforward_rank_contract']}`",
        f"- Selected arm: `{status['threshold_controller_t1_timestamp_walkforward_selected_arm']}`",
        f"- Selection reason: `{status['threshold_controller_t1_timestamp_walkforward_selection_reason']}`",
        f"- Folds: `{status['threshold_controller_t1_timestamp_walkforward_folds']}`",
        f"- Median delta net PnL: `{status['threshold_controller_t1_timestamp_walkforward_median_delta_net_pnl']}`",
        f"- Q25 delta net PnL: `{status['threshold_controller_t1_timestamp_walkforward_q25_delta_net_pnl']}`",
        f"- Positive-delta share: `{status['threshold_controller_t1_timestamp_walkforward_positive_delta_share']}`",
        f"- Accepted-frontier candidates: `{status['threshold_controller_t1_timestamp_walkforward_accepted_frontier_candidate_total']}`",
        f"- Accepted-frontier suppressions: `{status['threshold_controller_t1_timestamp_walkforward_accepted_frontier_suppressed_total']}`",
        f"- Promotion gate passed: `{status['threshold_controller_t1_timestamp_walkforward_promotion_gate_passed']}`",
        f"- Failures: `{', '.join(status['threshold_controller_t1_timestamp_walkforward_failures'])}`",
        f"- Evidence dir: `{status['threshold_controller_t1_timestamp_walkforward_dir']}`",
        "",
        "## No-Backfill Threshold Candidate",
        "",
        f"- Status: `{status['threshold_controller_no_backfill_status']}`",
        f"- Selected arm: `{status['threshold_controller_no_backfill_selected_arm']}`",
        f"- Median delta net PnL: `{status['threshold_controller_no_backfill_selected_median_delta_net_pnl']}`",
        f"- Q25 delta net PnL: `{status['threshold_controller_no_backfill_selected_q25_delta_net_pnl']}`",
        f"- Positive-delta share: `{status['threshold_controller_no_backfill_positive_delta_share']}`",
        f"- Removed baseline trades: `{status['threshold_controller_no_backfill_removed_trades']}`",
        f"- Defensive success: `{status['threshold_controller_no_backfill_defensive_success']}`",
        f"- Winner PnL sacrificed: `{status['threshold_controller_no_backfill_winner_pnl_sacrificed']}`",
        f"- Replacement entrants: `{status['threshold_controller_no_backfill_action_entrants']}`",
        f"- Bundle execution enabled: `{status['threshold_controller_no_backfill_bundle_execution_enabled']}`",
        f"- Promotion gate passed: `{status['threshold_controller_no_backfill_promotion_gate_passed']}`",
        f"- Failures: `{', '.join(status['threshold_controller_no_backfill_failures'])}`",
        f"- Evidence dir: `{status['threshold_controller_no_backfill_dir']}`",
        f"- Bundle dir: `{status['threshold_controller_no_backfill_bundle_dir']}`",
        "",
        "## Direct Accepted-Frontier Training Ledger",
        "",
        f"- Contract: `{status['threshold_controller_direct_suppression_ledger_contract']}`",
        f"- Rows: `{status['threshold_controller_direct_suppression_ledger_rows']}`",
        f"- Unique decision keys: `{status['threshold_controller_direct_suppression_ledger_unique_keys']}`",
        f"- Direct profitable suppression rate: `{status['threshold_controller_direct_suppression_ledger_profitable_rate']}`",
        f"- Mean direct defensive utility: `{status['threshold_controller_direct_suppression_ledger_mean_utility']}`",
        f"- Total direct defensive utility: `{status['threshold_controller_direct_suppression_ledger_total_utility']}`",
        f"- Current schedule suppressed rows: `{status['threshold_controller_direct_suppression_ledger_current_schedule_suppressed_rows']}`",
        f"- Current schedule defensive utility: `{status['threshold_controller_direct_suppression_ledger_current_schedule_utility']}`",
        f"- short_asset mean direct utility: `{status['threshold_controller_direct_suppression_ledger_short_asset_mean_utility']}`",
        f"- short_boll mean direct utility: `{status['threshold_controller_direct_suppression_ledger_short_boll_mean_utility']}`",
        f"- Interpretation: `{status['threshold_controller_direct_suppression_ledger_interpretation']}`",
        f"- Ledger dir: `{status['threshold_controller_direct_suppression_ledger_dir']}`",
        f"- Report: `{status['threshold_controller_direct_suppression_ledger_report']}`",
        "",
        "## Direct Suppression Shadow Training",
        "",
        f"- Contract: `{status['threshold_controller_direct_suppression_training_contract']}`",
        f"- Model mode: `{status['threshold_controller_direct_suppression_training_model_mode']}`",
        f"- OOF rows: `{status['threshold_controller_direct_suppression_training_oof_rows']}`",
        f"- OOF unique decision keys: `{status['threshold_controller_direct_suppression_training_oof_unique_keys']}`",
        f"- OOF AUC: `{status['threshold_controller_direct_suppression_training_oof_auc']}`",
        f"- OOF average precision: `{status['threshold_controller_direct_suppression_training_oof_ap']}`",
        f"- OOF utility Spearman: `{status['threshold_controller_direct_suppression_training_utility_spearman']}`",
        f"- Selected shadow arm: `{status['threshold_controller_direct_suppression_training_selected_arm']}`",
        f"- Selection reason: `{status['threshold_controller_direct_suppression_training_selection_reason']}`",
        f"- Best attempt arm: `{status['threshold_controller_direct_suppression_training_best_attempt_arm']}`",
        f"- Best attempt suppressed rows: `{status['threshold_controller_direct_suppression_training_best_attempt_rows']}`",
        f"- Best attempt defensive success: `{status['threshold_controller_direct_suppression_training_best_attempt_success']}`",
        f"- Promotion allowed: `{status['threshold_controller_direct_suppression_training_promotion_allowed']}`",
        f"- Interpretation: `{status['threshold_controller_direct_suppression_training_interpretation']}`",
        f"- Training dir: `{status['threshold_controller_direct_suppression_training_dir']}`",
        f"- Report: `{status['threshold_controller_direct_suppression_training_report']}`",
        "",
        "## No-Backfill Latest Shadow Score",
        "",
        f"- Status: `{status['threshold_controller_no_backfill_shadow_status']}`",
        f"- Period: `{status['threshold_controller_no_backfill_shadow_period_start']}` to `{status['threshold_controller_no_backfill_shadow_period_end']}`",
        f"- Baseline net PnL: `{status['threshold_controller_no_backfill_shadow_baseline_net_pnl']}`",
        f"- Shadow net PnL: `{status['threshold_controller_no_backfill_shadow_net_pnl']}`",
        f"- Total delta net PnL: `{status['threshold_controller_no_backfill_shadow_total_delta_net_pnl']}`",
        f"- Baseline trades: `{status['threshold_controller_no_backfill_shadow_baseline_trades']}`",
        f"- Shadow trades: `{status['threshold_controller_no_backfill_shadow_trades']}`",
        f"- Removed trades: `{status['threshold_controller_no_backfill_shadow_removed_trades']}`",
        f"- Added trades: `{status['threshold_controller_no_backfill_shadow_added_trades']}`",
        f"- Removed loss avoided: `{status['threshold_controller_no_backfill_shadow_removed_loss_avoided']}`",
        f"- Winner PnL sacrificed: `{status['threshold_controller_no_backfill_shadow_winner_pnl_sacrificed']}`",
        f"- Common-trade net PnL delta: `{status['threshold_controller_no_backfill_shadow_common_net_pnl_delta']}`",
        f"- Shadow accepted subset of baseline: `{status['threshold_controller_no_backfill_shadow_subset']}`",
        f"- Eval feature-store coverage: `{status['threshold_controller_no_backfill_shadow_feature_coverage']}`",
        f"- Eval source feature count: `{status['threshold_controller_no_backfill_shadow_source_feature_count']}`",
        f"- Bundle contract audit expected rank contract: `{status['threshold_controller_no_backfill_shadow_bundle_contract_audit_expected_rank_contract']}`",
        f"- Bundle contract audit passed: `{status['threshold_controller_no_backfill_shadow_bundle_contract_audit_passed']}`",
        f"- Bundle contract audit completion-grade passed: `{status['threshold_controller_no_backfill_shadow_bundle_contract_audit_completion_grade_passed']}`",
        f"- Promotion gate passed: `{status['threshold_controller_no_backfill_shadow_promotion_gate_passed']}`",
        f"- Failures: `{', '.join(status['threshold_controller_no_backfill_shadow_failures'])}`",
        f"- Score dir: `{status['threshold_controller_no_backfill_shadow_score_dir']}`",
        f"- Bundle dir: `{status['threshold_controller_no_backfill_shadow_bundle_dir']}`",
        f"- Bundle contract audit: `{status['threshold_controller_no_backfill_shadow_bundle_contract_audit_json']}`",
        f"- Eval feature store: `{status['threshold_controller_no_backfill_shadow_eval_feature_store_dir']}`",
        "",
        "## No-Backfill Shadow Monitor",
        "",
        f"- Status: `{status['threshold_controller_no_backfill_monitor_status']}`",
        f"- Window count: `{status['threshold_controller_no_backfill_monitor_window_count']}`",
        f"- Minimum later-window count: `{status['threshold_controller_no_backfill_monitor_min_later_window_count']}`",
        f"- Additional windows needed: `{status['threshold_controller_no_backfill_monitor_additional_windows_needed']}`",
        f"- Latest scored window end: `{status['threshold_controller_no_backfill_monitor_latest_window_end']}`",
        f"- Positive-delta window share: `{status['threshold_controller_no_backfill_monitor_positive_share']}`",
        f"- Positive action-only window share: `{status['threshold_controller_no_backfill_monitor_action_only_positive_share']}`",
        f"- Direct-threshold-only available windows: `{status['threshold_controller_no_backfill_monitor_direct_threshold_only_available_window_count']}`",
        f"- Positive direct-threshold-only window share: `{status['threshold_controller_no_backfill_monitor_direct_threshold_only_positive_share']}`",
        f"- Direct-threshold-only suppression window share: `{status['threshold_controller_no_backfill_monitor_direct_threshold_only_suppression_share']}`",
        f"- Direct-threshold-only promotion gate passed: `{status['threshold_controller_no_backfill_monitor_direct_threshold_only_promotion_gate_passed']}`",
        f"- Direct-threshold-only failures: `{', '.join(status['threshold_controller_no_backfill_monitor_direct_threshold_only_failures'])}`",
        f"- Median delta net PnL: `{status['threshold_controller_no_backfill_monitor_median_delta']}`",
        f"- Q25 delta net PnL: `{status['threshold_controller_no_backfill_monitor_q25_delta']}`",
        f"- Sum delta net PnL: `{status['threshold_controller_no_backfill_monitor_total_delta']}`",
        f"- Sum full-path replay delta: `{status['threshold_controller_no_backfill_monitor_full_path_delta']}`",
        f"- Sum action-only fixed-common-size delta: `{status['threshold_controller_no_backfill_monitor_action_only_delta']}`",
        f"- Sum direct-threshold-only delta: `{status['threshold_controller_no_backfill_monitor_direct_threshold_only_delta']}`",
        f"- Direct-threshold-only removed trades: `{status['threshold_controller_no_backfill_monitor_direct_threshold_only_removed_trades']}`",
        f"- Direct-threshold-only defensive success: `{status['threshold_controller_no_backfill_monitor_direct_threshold_only_defensive_success']}`",
        f"- Locked overlay available windows: `{status['threshold_controller_no_backfill_monitor_locked_overlay_available_windows']}`",
        f"- Locked overlay positive window share: `{status['threshold_controller_no_backfill_monitor_locked_overlay_positive_share']}`",
        f"- Locked overlay suppression window share: `{status['threshold_controller_no_backfill_monitor_locked_overlay_suppression_share']}`",
        f"- Locked overlay delta: `{status['threshold_controller_no_backfill_monitor_locked_overlay_delta']}`",
        f"- Locked overlay removed trades: `{status['threshold_controller_no_backfill_monitor_locked_overlay_removed_trades']}`",
        f"- Locked overlay defensive success: `{status['threshold_controller_no_backfill_monitor_locked_overlay_defensive_success']}`",
        f"- Locked overlay promotion gate passed: `{status['threshold_controller_no_backfill_monitor_locked_overlay_promotion_gate_passed']}`",
        f"- Locked overlay failures: `{', '.join(status['threshold_controller_no_backfill_monitor_locked_overlay_failures'])}`",
        f"- Sum indirect path/capacity delta: `{status['threshold_controller_no_backfill_monitor_indirect_path_delta']}`",
        f"- Indirect path/capacity removed trades: `{status['threshold_controller_no_backfill_monitor_indirect_removed_trades']}`",
        f"- Indirect path/capacity defensive success: `{status['threshold_controller_no_backfill_monitor_indirect_defensive_success']}`",
        f"- Indirect winner PnL sacrificed: `{status['threshold_controller_no_backfill_monitor_indirect_winner_pnl_sacrificed']}`",
        f"- Sum path-dependent common-trade delta: `{status['threshold_controller_no_backfill_monitor_path_dependent_delta']}`",
        f"- Sum baseline net PnL: `{status['threshold_controller_no_backfill_monitor_baseline_net_pnl']}`",
        f"- Sum shadow net PnL: `{status['threshold_controller_no_backfill_monitor_shadow_net_pnl']}`",
        f"- Removed trades: `{status['threshold_controller_no_backfill_monitor_removed_trades']}`",
        f"- Added trades: `{status['threshold_controller_no_backfill_monitor_added_trades']}`",
        f"- Removed loss avoided: `{status['threshold_controller_no_backfill_monitor_removed_loss_avoided']}`",
        f"- Winner PnL sacrificed: `{status['threshold_controller_no_backfill_monitor_winner_pnl_sacrificed']}`",
        f"- Common-trade net PnL delta: `{status['threshold_controller_no_backfill_monitor_common_net_pnl_delta']}`",
        f"- Minimum eval feature-store coverage: `{status['threshold_controller_no_backfill_monitor_min_feature_coverage']}`",
        f"- Minimum eval source feature count: `{status['threshold_controller_no_backfill_monitor_min_source_feature_count']}`",
        f"- Artifact hashes complete: `{status['threshold_controller_no_backfill_monitor_artifact_hashes_complete']}`",
        f"- Score manifest contract versions: `{', '.join(status['threshold_controller_no_backfill_monitor_score_manifest_contract_versions'])}`",
        f"- Windows missing input hashes: `{status['threshold_controller_no_backfill_monitor_windows_missing_score_input_hash_fields']}`",
        f"- Windows missing required output hashes: `{status['threshold_controller_no_backfill_monitor_windows_missing_required_output_hashes']}`",
        f"- Promotion gate passed: `{status['threshold_controller_no_backfill_monitor_promotion_gate_passed']}`",
        f"- Failures: `{', '.join(status['threshold_controller_no_backfill_monitor_failures'])}`",
        f"- Monitor dir: `{status['threshold_controller_no_backfill_monitor_dir']}`",
        "",
        "## No-Backfill Shadow Window Discovery",
        "",
        f"- Discovered scored windows: `{status['threshold_controller_no_backfill_window_discovery_discovered_count']}`",
        f"- Appendable windows: `{status['threshold_controller_no_backfill_window_discovery_appendable_count']}`",
        f"- Already monitored windows: `{status['threshold_controller_no_backfill_window_discovery_already_monitored_count']}`",
        f"- Failed windows: `{status['threshold_controller_no_backfill_window_discovery_failed_count']}`",
        f"- Latest discovered window end: `{status['threshold_controller_no_backfill_window_discovery_latest_end']}`",
        f"- Discovery dir: `{status['threshold_controller_no_backfill_window_discovery_dir']}`",
        f"- Readiness CSV: `{status['threshold_controller_no_backfill_window_discovery_readiness_csv']}`",
        f"- Appendable CSV: `{status['threshold_controller_no_backfill_window_discovery_appendable_csv']}`",
        "",
        "## Next No-Backfill Shadow Window Readiness",
        "",
        f"- Status: `{status['threshold_controller_no_backfill_next_readiness_status']}`",
        f"- Feature store: `{status['threshold_controller_no_backfill_next_feature_store_dir']}`",
        f"- Feature timestamp max: `{status['threshold_controller_no_backfill_next_feature_timestamp_max']}`",
        f"- Maturity buffer hours: `{status['threshold_controller_no_backfill_next_maturity_buffer_hours']}`",
        f"- Maturity cutoff: `{status['threshold_controller_no_backfill_next_maturity_cutoff']}`",
        f"- Next window start: `{status['threshold_controller_no_backfill_next_window_start']}`",
        f"- Target window end: `{status['threshold_controller_no_backfill_next_target_window_end']}`",
        f"- Proposed scoreable window end: `{status['threshold_controller_no_backfill_next_proposed_scoreable_window_end']}`",
        f"- Mature timestamps available: `{status['threshold_controller_no_backfill_next_mature_timestamp_count']}`",
        f"- Minimum timestamps required: `{status['threshold_controller_no_backfill_next_min_timestamp_count']}`",
        f"- Target window hours: `{status['threshold_controller_no_backfill_next_target_window_hours']}`",
        f"- Scoreable minimum window now: `{status['threshold_controller_no_backfill_next_scoreable_min_window_now']}`",
        f"- Scoreable full window now: `{status['threshold_controller_no_backfill_next_scoreable_full_window_now']}`",
        f"- Missing feature hours for minimum window: `{status['threshold_controller_no_backfill_next_missing_feature_hours_min']}`",
        f"- Missing feature hours for full window: `{status['threshold_controller_no_backfill_next_missing_feature_hours_full']}`",
        f"- Failures: `{', '.join(status['threshold_controller_no_backfill_next_failures'])}`",
        f"- Next action: `{status['threshold_controller_no_backfill_next_action']}`",
        f"- Readiness report: `{status['threshold_controller_no_backfill_next_readiness_report']}`",
        "",
        "## Next No-Backfill Shadow Runner",
        "",
        f"- Status: `{status['threshold_controller_no_backfill_next_runner_status']}`",
        f"- Reason: `{status['threshold_controller_no_backfill_next_runner_reason']}`",
        f"- Planned steps: `{status['threshold_controller_no_backfill_next_runner_planned_step_count']}`",
        f"- Completed steps: `{len(status['threshold_controller_no_backfill_next_runner_completed_steps'])}`",
        f"- Runner dir: `{status['threshold_controller_no_backfill_next_runner_dir']}`",
        f"- Runner manifest: `{status['threshold_controller_no_backfill_next_runner_manifest']}`",
        "",
        "## No-Backfill Failure Diagnostics",
        "",
        f"- Removed trades: `{status['threshold_controller_no_backfill_failure_removed_count']}`",
        f"- Direct threshold removals: `{status['threshold_controller_no_backfill_failure_direct_removed_count']}`",
        f"- Indirect path/capacity removals: `{status['threshold_controller_no_backfill_failure_indirect_removed_count']}`",
        f"- Direct defensive success: `{status['threshold_controller_no_backfill_failure_direct_defensive_success']}`",
        f"- Indirect defensive success: `{status['threshold_controller_no_backfill_failure_indirect_defensive_success']}`",
        f"- Removed loss avoided: `{status['threshold_controller_no_backfill_failure_removed_loss_avoided']}`",
        f"- Winner PnL sacrificed: `{status['threshold_controller_no_backfill_failure_winner_pnl_sacrificed']}`",
        f"- Safe subset found: `{status['threshold_controller_no_backfill_failure_promotion_safe_subset_found']}`",
        f"- Failure modes: `{', '.join(status['threshold_controller_no_backfill_failure_modes'])}`",
        f"- Report: `{status['threshold_controller_no_backfill_failure_report']}`",
        f"- Diagnostics dir: `{status['threshold_controller_no_backfill_failure_diagnostics_dir']}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = pd.DataFrame(status["blockers"])
    if blockers.empty:
        lines.append("_No blockers._")
    else:
        lines.append(_markdown_table(blockers))
    lines.extend(["", "## Next Actions", ""])
    for item in status["next_actions"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Interpretation", "", str(status["interpretation"]), ""])
    return "\n".join(lines)


def write_operational_status(status: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "market_state_operational_status.json").write_text(
        json.dumps(_json_safe(status), indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(status["blockers"]).to_csv(
        output_dir / "market_state_operational_status_blockers.csv",
        index=False,
    )
    pd.DataFrame({"next_action": status["next_actions"]}).to_csv(
        output_dir / "market_state_operational_status_next_actions.csv",
        index=False,
    )
    (output_dir / "market_state_operational_status_report.md").write_text(
        _render_report(status),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    status = build_operational_status(_load_json(args.config))
    write_operational_status(status, args.output_dir)
    print(json.dumps(_json_safe(status), indent=2))


if __name__ == "__main__":
    main()
