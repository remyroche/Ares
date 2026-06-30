#!/usr/bin/env python3
"""Audit supervised market-state plan completion against persisted artifacts.

The goal is a conservative evidence map, not a promotion decision. Each row
maps a plan requirement to a concrete artifact or audit result and labels it as
complete, partial, missing, failed, shadow-only, or gate-blocked.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/market_state_plan_completion_audit_20260626")

EXPECTED_ACTIVE_HEADS = ["short_asset", "short_boll"]
EXPECTED_DISABLED_HEADS = ["long_bars", "long_dist"]
EXPECTED_RANK_CONTRACT = "anchor_global_policy_rank_reference"
BASELINE_ARM = "S0_baseline_static_thresholds"

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"
STATUS_MISSING = "missing"
STATUS_SHADOW = "shadow_only"
STATUS_GATE_BLOCKED = "gate_blocked"
STATUS_NOT_REQUIRED = "not_required_yet"


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_first_json(paths: list[Path]) -> dict[str, Any]:
    for path in paths:
        payload = _read_json(path)
        if payload:
            return payload
    return {}


def _read_shadow_controller_monitor(path: Path) -> dict[str, Any]:
    legacy = _read_json(path / "shadow_controller_monitor_summary.json")
    if legacy:
        payload = dict(legacy)
        payload.setdefault("monitor_contract", "shadow_controller_monitor_summary")
        return payload

    no_backfill = _read_json(path / "no_backfill_shadow_monitor_summary.json")
    if not no_backfill:
        return {}

    payload = dict(no_backfill)
    payload.setdefault("monitor_contract", "no_backfill_shadow_monitor_summary")
    if "shadow_promotion_gate_passed" not in payload:
        payload["shadow_promotion_gate_passed"] = payload.get("promotion_gate_passed")
    if "controller_should_remain_disabled" not in payload:
        payload["controller_should_remain_disabled"] = payload.get("promotion_gate_passed") is not True
    if "total_shadow_realized_defensive_success" not in payload:
        for key in (
            "sum_locked_accepted_overlay_defensive_success",
            "sum_direct_threshold_only_defensive_success",
            "sum_accepted_delta_defensive_success",
        ):
            if key in payload:
                payload["total_shadow_realized_defensive_success"] = payload.get(key)
                break
    return payload


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _num(value: Any, default: float = float("nan")) -> float:
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) else default


def _exists_all(root: Path, names: list[str]) -> tuple[bool, list[str]]:
    missing = [name for name in names if not (root / name).exists()]
    return not missing, missing


def _add(
    rows: list[dict[str, Any]],
    requirement_id: str,
    section: str,
    requirement: str,
    status: str,
    evidence: str,
    notes: str = "",
) -> None:
    rows.append(
        {
            "requirement_id": requirement_id,
            "section": section,
            "requirement": requirement,
            "status": status,
            "evidence": evidence,
            "notes": notes,
        }
    )


def _status_from_bool(condition: bool, *, missing: bool = False) -> str:
    if missing:
        return STATUS_MISSING
    return STATUS_COMPLETE if condition else STATUS_FAILED


def audit_market_state_plan_completion(
    artifact_dir: Path,
    output_dir: Path,
    *,
    expected_rank_contract: str = EXPECTED_RANK_CONTRACT,
    controller_promotion_audit_dir: Path | None = None,
    state_head_quality_dir: Path | None = None,
    strategy_response_quality_dir: Path | None = None,
    shadow_priority_audit_dir: Path | None = None,
    shadow_controller_monitor_dir: Path | None = None,
    direct_suppression_training_dir: Path | None = None,
    direct_suppression_actionability_audit_dir: Path | None = None,
    backend_comparison_dir: Path | None = None,
    ablation_matrix_evidence_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    direct_suppression_training_dir_provided = direct_suppression_training_dir is not None
    controller_promotion_audit_dir = controller_promotion_audit_dir or Path()
    state_head_quality_dir = state_head_quality_dir or Path()
    strategy_response_quality_dir = strategy_response_quality_dir or Path()
    shadow_priority_audit_dir = shadow_priority_audit_dir or Path()
    shadow_controller_monitor_dir = shadow_controller_monitor_dir or Path()
    direct_suppression_training_dir = direct_suppression_training_dir or Path()
    direct_suppression_actionability_audit_dir = direct_suppression_actionability_audit_dir or Path()
    backend_comparison_dir = backend_comparison_dir or Path()
    ablation_matrix_evidence_dir = ablation_matrix_evidence_dir or Path()

    manifest = _read_json(artifact_dir / "manifest.json")
    feature_contract = _read_json(artifact_dir / "market_state_feature_contract.json")
    universe_contract = _read_json(artifact_dir / "market_state_universe_contract.json")
    controller_config = _read_json(artifact_dir / "strategy_threshold_controller_config.json")
    selected_controller = _read_json(artifact_dir / "walkforward_selected_controller_candidate.json")
    contract_audit = _read_json(artifact_dir / "market_state_controller_contract_audit.json")
    promotion_audit = _read_first_json(
        [
            controller_promotion_audit_dir / "market_state_controller_promotion_gate_audit.json",
            artifact_dir / "promotion_gate_audit" / "market_state_controller_promotion_gate_audit.json",
            artifact_dir / "market_state_controller_promotion_gate_audit.json",
        ]
    )
    state_quality = _read_json(state_head_quality_dir / "market_state_head_quality_gate.json")
    response_quality = _read_json(strategy_response_quality_dir / "market_state_strategy_response_quality_gate.json")
    priority_shadow = _read_first_json(
        [
            shadow_priority_audit_dir / "market_state_priority_shadow_promotion_gate.json",
            shadow_priority_audit_dir / "market_state_head_priority_promotion_gate_audit.json",
        ]
    )
    priority_recurrent = _read_json(shadow_priority_audit_dir / "recurrent_shadow_challenger.json")
    shadow_controller_monitor = _read_shadow_controller_monitor(shadow_controller_monitor_dir)
    direct_suppression_training = _read_json(
        direct_suppression_training_dir / "direct_suppression_training_summary.json"
    )
    direct_suppression_actionability = _read_json(
        direct_suppression_actionability_audit_dir / "direct_suppression_actionability_audit.json"
    )

    state_panel = _read_parquet(artifact_dir / "market_state_timestamp_panel.parquet")
    state_oof = _read_parquet(artifact_dir / "market_state_oof_predictions.parquet")
    response_oof = _read_parquet(artifact_dir / "strategy_response_oof_predictions.parquet")
    residual_ledger = _read_parquet(artifact_dir / "strategy_residual_target_ledger.parquet")
    schedule = _read_parquet(artifact_dir / "strategy_threshold_schedule.parquet")
    accepted = _read_parquet(artifact_dir / "accepted_trades.parquet")
    head_diag = _read_csv(artifact_dir / "market_state_head_diagnostics.csv")
    activation_registry = _read_csv(artifact_dir / "market_state_activation_registry.csv")
    replay_summary = _read_csv(artifact_dir / "portfolio_replay_summary.csv")
    ablation_replay_summary = _read_csv(ablation_matrix_evidence_dir / "portfolio_replay_summary.csv")
    replay_by_head = _read_csv(artifact_dir / "portfolio_replay_by_head.csv")
    overlap = _read_csv(artifact_dir / "walkforward_overlap.csv")
    backend_comparison = _read_csv(backend_comparison_dir / "backend_metric_comparison.csv")

    rows: list[dict[str, Any]] = []

    active_heads_ok = manifest.get("active_heads") == EXPECTED_ACTIVE_HEADS
    disabled_ok = sorted(manifest.get("disabled_heads") or []) == EXPECTED_DISABLED_HEADS
    rank_ok = manifest.get("rank_contract") == expected_rank_contract
    config_baseline = controller_config.get("baseline_contract", {})
    qfail_off = config_baseline.get("q_fail_enabled") is False
    policy_ok = manifest.get("policy_variant") == "refit_bar4_strategy_bar2"
    _add(
        rows,
        "0.1",
        "baseline_scope",
        "Evaluate against exact promoted T1 repaired static baseline.",
        _status_from_bool(active_heads_ok and disabled_ok and rank_ok and qfail_off and policy_ok, missing=not manifest),
        "manifest.json; strategy_threshold_controller_config.json",
        f"rank={manifest.get('rank_contract')}; expected_rank={expected_rank_contract}; active={manifest.get('active_heads')}; disabled={manifest.get('disabled_heads')}; q_fail_off={qfail_off}",
    )

    selected_null = selected_controller.get("selected_arm") is None
    promotion_ready = promotion_audit.get("controller_promotion_ready")
    if promotion_ready is None:
        promotion_ready = promotion_audit.get("promotion_gate_passed")
    promotion_failed = promotion_ready is False
    selected_no_backfill_overlay = (
        isinstance(selected_controller.get("selected_arm"), str)
        and str(selected_controller.get("selected_arm")).endswith("__post_selection_overlay")
        and isinstance(selected_controller.get("selection_policy"), dict)
        and selected_controller["selection_policy"].get("select_no_backfill_overlay_only") is True
        and isinstance(selected_controller.get("selected_metrics"), dict)
        and float(selected_controller["selected_metrics"].get("action_entrants") or 0.0) == 0.0
    )
    selected_shadow_execution_disabled = (
        manifest.get("selected_arm") is not None
        and (
            manifest.get("controller_execution_enabled") is False
            or manifest.get("shadow_controller_only") is True
        )
        and promotion_failed
    )
    controller_inactive_gate_blocked = (
        (selected_null and promotion_failed)
        or selected_shadow_execution_disabled
        or (selected_no_backfill_overlay and promotion_failed)
    )
    shadow_monitor_available = bool(shadow_controller_monitor)
    shadow_monitor_blocks_activation = (
        shadow_controller_monitor.get("shadow_promotion_gate_passed") is False
        or shadow_controller_monitor.get("controller_should_remain_disabled") is True
    )
    controller_latest_gate_blocked = controller_inactive_gate_blocked or (
        shadow_monitor_available and shadow_monitor_blocks_activation
    )
    _add(
        rows,
        "0.2",
        "baseline_scope",
        "Market-state controller remains execution-disabled unless gates pass.",
        STATUS_GATE_BLOCKED
        if controller_latest_gate_blocked
        else STATUS_COMPLETE
        if promotion_ready is True
        else _status_from_bool(False, missing=not selected_controller),
        "walkforward_selected_controller_candidate.json; promotion_gate_audit",
        f"selected_arm={selected_controller.get('selected_arm')}; manifest_selected_arm={manifest.get('selected_arm')}; "
        f"selected_no_backfill_overlay={selected_no_backfill_overlay}; "
        f"controller_execution_enabled={manifest.get('controller_execution_enabled')}; "
        f"shadow_controller_only={manifest.get('shadow_controller_only')}; "
        f"promotion_gate_passed={promotion_audit.get('promotion_gate_passed')}; "
        f"controller_promotion_ready={promotion_audit.get('controller_promotion_ready')}; "
        f"action_attribution_gate_passed={(promotion_audit.get('action_attribution_gate') or {}).get('passed')}; "
        f"shadow_monitor_contract={shadow_controller_monitor.get('monitor_contract')}; "
        f"shadow_monitor_gate_passed={shadow_controller_monitor.get('shadow_promotion_gate_passed')}; "
        f"shadow_monitor_should_disable={shadow_controller_monitor.get('controller_should_remain_disabled')}; "
        f"direct_threshold_gate_passed={shadow_controller_monitor.get('direct_threshold_only_promotion_gate_passed')}; "
        f"locked_overlay_gate_passed={shadow_controller_monitor.get('locked_accepted_overlay_promotion_gate_passed')}",
    )

    invariants = feature_contract.get("invariants", {})
    no_mutation = (
        invariants.get("controller_changes_scores_or_ranks") is False
        and invariants.get("controller_changes_auction_ordering") is False
        and invariants.get("controller_can_lower_thresholds") is False
    )
    _add(
        rows,
        "1.5",
        "design_principles",
        "First executable controller can only raise thresholds and cannot alter score/rank/order.",
        _status_from_bool(no_mutation, missing=not feature_contract),
        "market_state_feature_contract.json",
        "Contract invariants assert no score/rank/auction mutation and no threshold lowering.",
    )

    continuous_state_cols = [
        col
        for col in state_oof.columns
        if str(col).startswith("state_") or str(col).startswith("forecast_")
    ]
    _add(
        rows,
        "2.1",
        "architecture",
        "Represent market state as overlapping continuous state severities.",
        _status_from_bool(len(continuous_state_cols) >= 10, missing=state_oof.empty),
        "market_state_oof_predictions.parquet",
        f"state/forecast columns={len(continuous_state_cols)}",
    )

    latent_disabled = invariants.get("latent_gmm_active_controller_input") is False
    _add(
        rows,
        "2.2",
        "architecture",
        "GMM latent states are removed from active production architecture.",
        _status_from_bool(latent_disabled, missing=not feature_contract),
        "market_state_feature_contract.json; market_state_activation_registry.csv",
        "Latent GMM may remain shadow/diagnostic only.",
    )

    source_audit = feature_contract.get("source_contract_audit", {})
    no_forbidden = (
        source_audit.get("overall_passed") is True
        and source_audit.get("actual_order_book_features_allowed") is False
        and source_audit.get("candidate_population_fallback_allowed_for_production") is False
    )
    _add(
        rows,
        "3.1",
        "market_universe",
        "Market-state encoder uses strategy-independent OHLCV/OI/funding feature-store aggregates only.",
        _status_from_bool(no_forbidden, missing=not feature_contract),
        "market_state_feature_contract.json source_contract_audit",
        f"source={source_audit.get('required_source')}; overall_passed={source_audit.get('overall_passed')}",
    )

    universe_ok = (
        universe_contract.get("strategy_independent") is True
        and universe_contract.get("candidate_independent") is True
        and isinstance(universe_contract.get("eligible_symbols"), list)
        and bool(universe_contract.get("eligible_symbols"))
        and bool(universe_contract.get("minimum_history"))
        and bool(universe_contract.get("minimum_volume"))
        and bool(universe_contract.get("oi_coverage_requirements"))
        and bool(universe_contract.get("funding_coverage_requirements"))
    )
    _add(
        rows,
        "3.2",
        "market_universe",
        "Persist fixed universe-selection contract with eligible symbols and coverage requirements.",
        _status_from_bool(universe_ok, missing=not universe_contract),
        "market_state_universe_contract.json",
        f"eligible_symbol_count={universe_contract.get('eligible_symbol_count')}",
    )

    if not state_panel.empty:
        key_cols = [col for col in ["fold", "split", "state_arm", "timestamp"] if col in state_panel.columns]
        duplicate_rows = int(state_panel.duplicated(key_cols).sum()) if key_cols else -1
    else:
        duplicate_rows = -1
    _add(
        rows,
        "3.3",
        "market_universe",
        "Exactly one market-state row per timestamp/fold/split/state arm.",
        _status_from_bool(duplicate_rows == 0, missing=state_panel.empty),
        "market_state_timestamp_panel.parquet",
        f"rows={len(state_panel)}; duplicate_rows={duplicate_rows}",
    )

    contract_complete = (
        contract_audit.get("completion_grade_passed") is True
        and contract_audit.get("passed") is True
        and not contract_audit.get("failures")
    )
    _add(
        rows,
        "3.4",
        "market_universe",
        "Causal normalization, fold-fitted references, and training-only CDFs are verified.",
        _status_from_bool(contract_complete, missing=not contract_audit),
        "market_state_controller_contract_audit.json",
        "Completion-grade audit covers fold-fitted references, target CDFs, OOF parity, fallback and static replay.",
    )

    registry_ok = not activation_registry.empty and {
        "state_head",
        "recommended_status",
        "activation_registry_version",
    }.issubset(activation_registry.columns)
    _add(
        rows,
        "4.1",
        "state_registry",
        "Semantic state-head registry with active/shadow/disabled statuses is persisted.",
        _status_from_bool(registry_ok, missing=activation_registry.empty),
        "market_state_activation_registry.csv",
        f"rows={len(activation_registry)}; statuses={activation_registry.get('recommended_status', pd.Series(dtype=str)).value_counts().to_dict()}",
    )

    channels = set(continuous_state_cols)
    ood_present = "state_ood_score" in channels or "state_novelty" in channels or "state_ood_score" in response_oof.columns
    reliability_present = ood_present and {
        "state_drift_score",
        "state_uncertainty",
        "state_input_coverage",
    }.issubset(channels)
    _add(
        rows,
        "4.2",
        "state_registry",
        "Reliability channels OOD, drift, uncertainty and coverage are present.",
        _status_from_bool(reliability_present, missing=state_oof.empty),
        "market_state_oof_predictions.parquet",
        "present="
        f"{sorted({'state_ood_score','state_novelty','state_drift_score','state_uncertainty','state_input_coverage'}.intersection(channels))}; "
        f"downstream_state_ood_score={'state_ood_score' in response_oof.columns}",
    )

    liquidity_proxy_ok = "state_liquidity_stress_proxy" in channels and "state_liquidity_stress" not in channels
    _add(
        rows,
        "5.3",
        "feature_whitelist",
        "Liquidity output is named proxy and actual order-book fields are excluded.",
        _status_from_bool(liquidity_proxy_ok and no_forbidden, missing=state_oof.empty or not feature_contract),
        "market_state_oof_predictions.parquet; market_state_feature_contract.json",
        "Uses state_liquidity_stress_proxy and contract forbids actual order-book fields.",
    )

    forecast_heads = head_diag.loc[head_diag.get("state_level", pd.Series(dtype=str)).astype(str).eq("forecast")]
    lgbm_present = (artifact_dir / "market_state_lgbm_models.joblib").exists()
    _add(
        rows,
        "8.1",
        "forecast_heads",
        "Prospective LGBM state-head pack exists as independent overlapping heads.",
        _status_from_bool(lgbm_present and len(forecast_heads) >= 5, missing=head_diag.empty),
        "market_state_lgbm_models.joblib; market_state_head_diagnostics.csv",
        f"forecast_heads={len(forecast_heads)}",
    )

    xgb_exists = backend_comparison_dir.exists() and not backend_comparison.empty
    _add(
        rows,
        "8.6",
        "forecast_heads",
        "XGBoost challenger is available as an out-of-family benchmark.",
        STATUS_COMPLETE if xgb_exists else STATUS_PARTIAL,
        "market_state_backend_comparison_*",
        f"backend_comparison_dir={backend_comparison_dir if backend_comparison_dir else ''}",
    )

    skill_cols = {
        "mean_tail_average_precision",
        "mean_tail_brier_p90",
        "mean_tail_ece_5bin",
        "mean_tail_recall_p90",
        "mean_tail_false_alarm_rate_p90",
        "collapsed_folds",
    }
    _add(
        rows,
        "8.5",
        "forecast_heads",
        "State-head diagnostics include soft-severity and hard-tail quality metrics.",
        _status_from_bool(skill_cols.issubset(head_diag.columns), missing=head_diag.empty),
        "market_state_head_diagnostics.csv",
        f"metric_columns_present={sorted(skill_cols.intersection(head_diag.columns))}",
    )

    rank_curves = (artifact_dir / "strategy_rank_outcome_curves.joblib").exists()
    residual_cols = {"resid_utility", "resid_full_sl", "resid_timeout", "_rank", "strategy_id"}
    _add(
        rows,
        "10.2",
        "strategy_response",
        "Rank-conditioned baselines and residual targets are persisted for executable candidates.",
        _status_from_bool(rank_curves and residual_cols.issubset(residual_ledger.columns), missing=residual_ledger.empty),
        "strategy_rank_outcome_curves.joblib; strategy_residual_target_ledger.parquet",
        f"residual_rows={len(residual_ledger)}; accepted_rows={len(accepted)}",
    )

    all_candidates_not_accepted_only = not residual_ledger.empty and len(residual_ledger) > max(len(accepted), 0)
    _add(
        rows,
        "10.4",
        "strategy_response",
        "Strategy-response training uses all executable candidates, not accepted trades only.",
        _status_from_bool(all_candidates_not_accepted_only, missing=residual_ledger.empty),
        "strategy_residual_target_ledger.parquet; accepted_trades.parquet",
        f"residual_rows={len(residual_ledger)}; accepted_rows={len(accepted)}",
    )

    response_models_ok = (artifact_dir / "strategy_response_models.joblib").exists() and (
        artifact_dir / "strategy_response_ebm_models.joblib"
    ).exists()
    response_contracts_ok = (
        not response_oof.empty
        and set(response_oof.get("state_prediction_contract", pd.Series(dtype=str)).dropna().astype(str).unique())
        == {"outer_fold_validation_state_scores"}
    )
    _add(
        rows,
        "11.2",
        "strategy_response",
        "Strategy-response models output residual utility, excess full-SL and excess-timeout OOF predictions.",
        _status_from_bool(response_models_ok and response_contracts_ok, missing=response_oof.empty),
        "strategy_response_models.joblib; strategy_response_oof_predictions.parquet",
        f"response_rows={len(response_oof)}",
    )

    response_quality_done = response_quality.get("passed") is True
    response_quality_passing_count = int(response_quality.get("quality_passing_arm_count") or 0)
    _add(
        rows,
        "16.2",
        "strategy_response_metrics",
        "Strategy-response metrics include residual IC, decile utility, calibration and support.",
        _status_from_bool(response_quality_done, missing=not response_quality),
        "market_state_strategy_response_quality_gate.json",
        f"quality_passing_arm_count={response_quality_passing_count}",
    )

    response_quality_gate_passed = response_quality.get("quality_gate_passed")
    if response_quality_gate_passed is None:
        response_quality_gate_passed = response_quality_done and response_quality_passing_count > 0
    if not response_quality:
        response_gate_status = STATUS_MISSING
    elif bool(response_quality_gate_passed):
        response_gate_status = STATUS_COMPLETE
    elif response_quality_done:
        response_gate_status = STATUS_GATE_BLOCKED
    else:
        response_gate_status = STATUS_FAILED
    _add(
        rows,
        "14.3",
        "strategy_response",
        "Response-model gate identifies at least one quality-passing strategy-response arm before controller activation.",
        response_gate_status,
        "market_state_strategy_response_quality_gate.json",
        "quality_passing_arm_count="
        f"{response_quality_passing_count}; structural_passed={response_quality.get('structural_passed', response_quality.get('passed'))}; "
        f"quality_gate_passed={response_quality_gate_passed}; "
        f"quality_passing_heads={response_quality.get('quality_passing_heads')}; "
        f"support_blocked_heads={response_quality.get('support_blocked_heads')}; "
        "signal_passing_but_support_blocked_heads="
        f"{response_quality.get('signal_passing_but_support_blocked_heads')}",
    )

    threshold_ok = not schedule.empty and (pd.to_numeric(schedule["state_threshold"], errors="coerce") >= pd.to_numeric(schedule["base_threshold"], errors="coerce") - 1e-12).all()
    _add(
        rows,
        "12.3",
        "threshold_controller",
        "Threshold controller is penalty-only and never lowers thresholds below base.",
        _status_from_bool(threshold_ok and controller_config.get("controller", {}).get("penalty_only") is True, missing=schedule.empty),
        "strategy_threshold_schedule.parquet; strategy_threshold_controller_config.json",
        f"schedule_rows={len(schedule)}",
    )

    direct_policy_grid = direct_suppression_training.get("policy_grid") or {}
    direct_selection = direct_suppression_training.get("selection") or {}
    direct_oof = direct_suppression_training.get("oof") or {}
    actionability_selected_arm = direct_suppression_actionability.get("selected_arm")
    actionability_selection_reason = direct_suppression_actionability.get("selection_reason")
    actionability_dominant_blocker = direct_suppression_actionability.get("dominant_blocker")
    actionability_passing_rows = direct_suppression_actionability.get("passing_policy_rows")
    actionability_recurrent_rows = direct_suppression_actionability.get("recurrent_support_policy_rows")
    actionability_max_recurrent_success = direct_suppression_actionability.get(
        "max_recurrent_defensive_success"
    )
    actionability_max_recurrent_positive_share = direct_suppression_actionability.get(
        "max_recurrent_positive_fold_share"
    )
    direct_scopes = set(str(scope) for scope in (direct_policy_grid.get("policy_scopes") or []))
    direct_strategy_scope_ok = {
        "controller_arm_strategy",
        "controller_arm_head_strategy",
    }.issubset(direct_scopes)
    direct_selected_arm = direct_selection.get("selected_arm")
    direct_selection_reason = direct_selection.get("reason")
    direct_best = direct_selection.get("best_attempt") or {}
    if direct_suppression_actionability:
        if actionability_selected_arm is None and actionability_passing_rows == 0:
            direct_status = STATUS_GATE_BLOCKED
        elif actionability_selected_arm is None:
            direct_status = STATUS_SHADOW
        else:
            direct_status = STATUS_COMPLETE
    elif not direct_suppression_training and not direct_suppression_training_dir_provided:
        direct_status = STATUS_NOT_REQUIRED
    elif not direct_suppression_training:
        direct_status = STATUS_MISSING
    elif not direct_strategy_scope_ok:
        direct_status = STATUS_FAILED
    elif direct_selected_arm is None and direct_selection_reason == "no_policy_grid_row_passed_diagnostic_gate":
        direct_status = STATUS_GATE_BLOCKED
    elif direct_suppression_training.get("promotion_allowed") is False:
        direct_status = STATUS_SHADOW
    else:
        direct_status = STATUS_COMPLETE
    _add(
        rows,
        "12.6",
        "threshold_controller",
        "Direct accepted-frontier suppression learner evaluates per-strategy threshold-only policies and blocks activation without recurrent evidence.",
        direct_status,
        "direct_suppression_training_summary.json; direct_suppression_policy_grid.csv; direct_suppression_actionability_audit.json",
        "policy_scopes="
        f"{sorted(direct_scopes)}; selected_arm={direct_selected_arm}; reason={direct_selection_reason}; "
        f"best_scope={direct_best.get('policy_scope')}; best_head={direct_best.get('target_head')}; "
        f"best_strategy={direct_best.get('target_strategy_id')}; best_suppressed_rows={direct_best.get('suppressed_rows')}; "
        f"best_suppressed_folds={direct_best.get('suppressed_folds')}; min_suppressed_folds={direct_policy_grid.get('min_suppressed_folds')}; "
        f"oof_auc={direct_oof.get('prob_auc')}; oof_ap={direct_oof.get('prob_average_precision')}; "
        f"oof_utility_spearman={direct_oof.get('utility_spearman')}; "
        f"actionability_selected_arm={actionability_selected_arm}; "
        f"actionability_reason={actionability_selection_reason}; "
        f"actionability_dominant_blocker={actionability_dominant_blocker}; "
        f"actionability_passing_rows={actionability_passing_rows}; "
        f"actionability_recurrent_support_rows={actionability_recurrent_rows}; "
        f"actionability_max_recurrent_defensive_success={actionability_max_recurrent_success}; "
        f"actionability_max_recurrent_positive_fold_share={actionability_max_recurrent_positive_share}",
    )

    fail_closed_check = "missing_or_ood_state_falls_back_to_base_threshold" in contract_audit.get("artifact_audit_checks", [])
    _add(
        rows,
        "9.4",
        "threshold_controller",
        "Missing/OOD state fails closed to base threshold.",
        _status_from_bool(fail_closed_check and contract_complete, missing=not contract_audit),
        "market_state_controller_contract_audit.json",
        "Completion audit includes missing/OOD fallback and force-base checks.",
    )

    _add(
        rows,
        "13.1",
        "rollout_phases",
        "Execution-changing modulation remains inactive until its own promotion gates pass.",
        STATUS_COMPLETE
        if controller_latest_gate_blocked or promotion_ready is True
        else STATUS_FAILED,
        "promotion_gate_audit; strategy_threshold_controller_config.json",
        "Threshold-only remains a defensive track; pre-filter head-prior/rank modulation and auction-priority modulation remain shadow-only unless separately promoted.",
    )

    state_quality_done = state_quality.get("passed") is True
    _add(
        rows,
        "14.1",
        "state_head_pruning",
        "Weak, redundant, unused or unsafe heads are audited and disabled/shadowed.",
        _status_from_bool(state_quality_done, missing=not state_quality),
        "market_state_head_quality_gate.json; market_state_activation_registry.csv",
        f"active_candidates={state_quality.get('active_candidates')}; grade_counts={state_quality.get('grade_counts')}",
    )

    loo_present = (artifact_dir / "market_state_leave_one_head_out_aggregate.csv").exists()
    _add(
        rows,
        "14.5",
        "state_head_pruning",
        "Portfolio leave-one-state-head-out increment tests are persisted.",
        _status_from_bool(loo_present, missing=False),
        "market_state_leave_one_head_out_aggregate.csv",
        "",
    )

    fold_ok = controller_config.get("validation", {}).get("chronological_complete_timestamp_folds") is True
    embargo_ok = _num(controller_config.get("validation", {}).get("embargo_hours"), 0.0) > 0
    _add(
        rows,
        "15.1",
        "validation_contract",
        "Nested chronological fold and embargo contract is persisted.",
        _status_from_bool(fold_ok and embargo_ok, missing=not controller_config),
        "strategy_threshold_controller_config.json; market_state_feature_contract.json",
        f"embargo_hours={controller_config.get('validation', {}).get('embargo_hours')}",
    )

    _add(
        rows,
        "15.3",
        "validation_contract",
        "June attribution windows do not promote controller execution.",
        STATUS_GATE_BLOCKED,
        "promotion_gate_audit; selected controller manifest",
        "June remains development/attribution evidence until later matured windows pass.",
    )

    state_metrics_done = state_quality_done and state_quality.get("state_heads", 0) > 0
    _add(
        rows,
        "16.1",
        "metrics",
        "State-head metric audit covers skill, calibration, recall, false alarms, support and collapse.",
        _status_from_bool(state_metrics_done, missing=not state_quality),
        "market_state_head_quality_gate.json; market_state_head_quality_by_head.csv",
        f"forecast_quality_failures={len(state_quality.get('forecast_quality_failure_heads') or [])}",
    )

    metric_tables = [
        "strategy_threshold_action_audit.csv",
        "walkforward_threshold_action_utility.csv",
        "walkforward_threshold_action_edge_validation.csv",
        "walkforward_threshold_candidate_suppression_aggregate.csv",
        "walkforward_threshold_baseline_accepted_suppression_aggregate.csv",
    ]
    metric_ok, missing_metric = _exists_all(artifact_dir, metric_tables)
    _add(
        rows,
        "16.3",
        "metrics",
        "Controller metrics include threshold action, suppression utility and defensive success.",
        _status_from_bool(metric_ok, missing=False),
        "; ".join(metric_tables),
        f"missing={missing_metric}",
    )

    portfolio_metric_ok = not replay_summary.empty and not replay_by_head.empty and not overlap.empty
    _add(
        rows,
        "16.4",
        "metrics",
        "Portfolio metrics are paired against exact T1 with by-head and accepted-overlap outputs.",
        _status_from_bool(portfolio_metric_ok, missing=False),
        "portfolio_replay_summary.csv; portfolio_replay_by_head.csv; walkforward_overlap.csv",
        f"summary_rows={len(replay_summary)}; by_head_rows={len(replay_by_head)}; overlap_rows={len(overlap)}",
    )

    ablation_arms = set(replay_summary.get("arm", pd.Series(dtype=str)).dropna().astype(str).unique())
    external_ablation_arms = set(
        ablation_replay_summary.get("arm", pd.Series(dtype=str)).dropna().astype(str).unique()
    )
    combined_ablation_arms = ablation_arms | external_ablation_arms
    has_static = BASELINE_ARM in ablation_arms
    has_state = bool({"S1_observed_axes_shared_response", "S2_observed_forecast_shared_response"}.intersection(ablation_arms))
    has_pruned = "S7_pruned_state_pack" in combined_ablation_arms
    _add(
        rows,
        "17.1",
        "ablation_matrix",
        "Static, observed, observed+forecast and pruned state-pack threshold arms are replayed.",
        _status_from_bool(has_static and has_state and has_pruned, missing=replay_summary.empty),
        "portfolio_replay_summary.csv; optional external ablation evidence",
        f"arms={sorted(ablation_arms)}; external_arms={sorted(external_ablation_arms)}; "
        f"external_dir={ablation_matrix_evidence_dir if ablation_matrix_evidence_dir else ''}",
    )

    required_feature = [
        "market_state_feature_contract.json",
        "market_state_training_reference.joblib",
        "market_state_timestamp_panel.parquet",
        "market_state_feature_coverage.csv",
    ]
    ok, missing = _exists_all(artifact_dir, required_feature)
    _add(rows, "18.1", "implementation_artifacts", "Market feature bundle artifacts are persisted.", _status_from_bool(ok), "; ".join(required_feature), f"missing={missing}")

    required_state = [
        "market_state_target_definitions.json",
        "market_state_target_cdfs.joblib",
        "market_state_oof_predictions.parquet",
        "market_state_head_diagnostics.csv",
    ]
    model_present = (artifact_dir / "market_state_lgbm_models.joblib").exists() or (artifact_dir / "market_state_xgb_models.joblib").exists()
    ok, missing = _exists_all(artifact_dir, required_state)
    _add(rows, "18.2", "implementation_artifacts", "State target/model/OOF diagnostic artifacts are persisted.", _status_from_bool(ok and model_present), "; ".join(required_state), f"missing={missing}; model_present={model_present}")

    required_response = [
        "strategy_rank_outcome_curves.joblib",
        "strategy_residual_target_ledger.parquet",
        "strategy_response_models.joblib",
        "strategy_response_ebm_models.joblib",
        "strategy_response_oof_predictions.parquet",
        "strategy_state_effect_matrix.csv",
    ]
    ok, missing = _exists_all(artifact_dir, required_response)
    _add(rows, "18.3", "implementation_artifacts", "Strategy response artifacts are persisted.", _status_from_bool(ok), "; ".join(required_response), f"missing={missing}")

    required_controller = [
        "strategy_threshold_schedule.parquet",
        "strategy_threshold_controller_config.json",
        "strategy_threshold_action_audit.csv",
        "portfolio_replay_summary.csv",
        "portfolio_replay_by_head.csv",
    ]
    ok, missing = _exists_all(artifact_dir, required_controller)
    _add(rows, "18.4", "implementation_artifacts", "Controller artifacts are persisted.", _status_from_bool(ok), "; ".join(required_controller), f"missing={missing}")

    hashes_ok = (artifact_dir / "artifact_hashes.json").exists() and contract_complete
    _add(
        rows,
        "18.5",
        "implementation_artifacts",
        "Feature order, source schema, fold definition, model parameters, transforms, hashes and registries are persisted.",
        _status_from_bool(hashes_ok, missing=not (artifact_dir / "artifact_hashes.json").exists()),
        "artifact_hashes.json; market_state_feature_contract.json; controller contract audit",
        "Completion-grade contract audit verifies metadata and hash coverage.",
    )

    parity_checks = [
        "market_state_one_row_per_fold_split_arm_timestamp",
        "no_forbidden_market_state_columns",
        "oof_state_values_match_timestamp_panel",
        "response_oof_uses_oof_state_scores",
        "state_threshold_never_below_base_threshold",
        "missing_or_ood_state_falls_back_to_base_threshold",
        "static_baseline_replay_parity",
        "accepted_decision_keys_unique_when_available",
    ]
    audit_checks = set(contract_audit.get("artifact_audit_checks") or [])
    _add(
        rows,
        "19.1",
        "tests_parity",
        "Required parity checks are covered by completion-grade controller contract audit.",
        _status_from_bool(set(parity_checks).issubset(audit_checks) and contract_complete, missing=not contract_audit),
        "market_state_controller_contract_audit.json",
        f"covered={sorted(set(parity_checks).intersection(audit_checks))}",
    )

    _add(
        rows,
        "20.1",
        "promotion_gates",
        "Controller promotion gates are evaluated and block activation when lower-quartile/defensive evidence fails.",
        STATUS_GATE_BLOCKED
        if controller_latest_gate_blocked
        else STATUS_COMPLETE
        if promotion_ready is True
        else STATUS_MISSING,
        "promotion_gate_audit/market_state_controller_promotion_gate_audit.json; shadow_controller_monitor_summary.json or no_backfill_shadow_monitor_summary.json",
        f"promotion_gate_passed={promotion_audit.get('promotion_gate_passed')}; "
        f"controller_promotion_ready={promotion_audit.get('controller_promotion_ready')}; "
        f"action_attribution_gate_passed={(promotion_audit.get('action_attribution_gate') or {}).get('passed')}; "
        f"controller_should_remain_disabled={promotion_audit.get('controller_should_remain_disabled')}; "
        f"shadow_monitor_contract={shadow_controller_monitor.get('monitor_contract')}; "
        f"shadow_monitor_gate_passed={shadow_controller_monitor.get('shadow_promotion_gate_passed')}; "
        f"shadow_monitor_should_disable={shadow_controller_monitor.get('controller_should_remain_disabled')}; "
        f"shadow_monitor_defensive_success={shadow_controller_monitor.get('total_shadow_realized_defensive_success')}; "
        f"direct_threshold_gate_passed={shadow_controller_monitor.get('direct_threshold_only_promotion_gate_passed')}; "
        f"locked_overlay_gate_passed={shadow_controller_monitor.get('locked_accepted_overlay_promotion_gate_passed')}",
    )

    priority_shadow_done = False
    priority_shadow_gate_value: Any = None
    priority_shadow_notes: str | None = None
    if priority_shadow:
        if "priority_should_remain_shadow" in priority_shadow:
            priority_shadow_gate_value = priority_shadow.get("production_passing_candidate_count")
            priority_shadow_done = bool(priority_shadow.get("priority_should_remain_shadow"))
            priority_shadow_notes = (
                f"single_window_replay_gate_passed={priority_shadow.get('single_window_replay_gate_passed')}; "
                f"production_passing_candidate_count={priority_shadow.get('production_passing_candidate_count')}; "
                f"passing_candidate_count={priority_shadow.get('passing_candidate_count')}; "
                f"production_blockers={priority_shadow.get('production_blockers')}"
            )
        elif "opportunity_should_remain_shadow" in priority_shadow:
            opportunity = priority_shadow.get("opportunity_routing_gate") or {}
            priority_shadow_gate_value = priority_shadow.get("opportunity_routing_passed")
            priority_shadow_done = bool(priority_shadow.get("opportunity_should_remain_shadow"))
            priority_shadow_notes = (
                f"opportunity_routing_passed={priority_shadow.get('opportunity_routing_passed')}; "
                f"opportunity_should_remain_shadow={priority_shadow.get('opportunity_should_remain_shadow')}; "
                f"opportunity_failures={opportunity.get('failures')}; "
                f"action_windows={opportunity.get('action_window_count')}; "
                f"positive_action_windows={opportunity.get('positive_action_window_count')}"
            )
        elif "promotion_gate_passed" in priority_shadow:
            priority_shadow_gate_value = priority_shadow.get("promotion_gate_passed")
            priority_shadow_done = priority_shadow_gate_value is False
            priority_shadow_notes = (
                f"legacy_promotion_gate_passed={priority_shadow_gate_value}; "
                f"failures={priority_shadow.get('failures')}"
            )
        else:
            priority_shadow_gate_value = priority_shadow.get("passed")
            priority_shadow_done = priority_shadow_gate_value is False and bool(priority_shadow.get("failures"))
            priority_shadow_notes = (
                f"legacy_passed={priority_shadow_gate_value}; "
                f"failures={priority_shadow.get('failures')}"
            )
    if priority_recurrent:
        recurrent_best = priority_recurrent.get("best_candidate") or {}
        recurrent_notes = (
            f"recurrent_selected={priority_recurrent.get('selected')}; "
            f"recurrent_reason={priority_recurrent.get('reason')}; "
            f"recurrent_best_candidate={recurrent_best.get('arm_selector')}; "
            f"recurrent_best_failures={recurrent_best.get('fail_reasons')}; "
            f"recurrent_action_windows={recurrent_best.get('action_window_count')}; "
            f"recurrent_positive_action_windows={recurrent_best.get('positive_action_window_count')}"
        )
        priority_shadow_notes = (
            f"{priority_shadow_notes}; {recurrent_notes}"
            if priority_shadow_notes
            else recurrent_notes
        )
        if priority_recurrent.get("selected") is False:
            priority_shadow_gate_value = False
            priority_shadow_done = True
    _add(
        rows,
        "21.2",
        "rollout_sequence",
        "Market-state priority modulation remains shadow-only and separately gated.",
        STATUS_SHADOW if priority_shadow_done else STATUS_PARTIAL,
        "market_state_priority_shadow_promotion_gate.json or market_state_head_priority_promotion_gate_audit.json; recurrent_shadow_challenger.json",
        priority_shadow_notes
        or f"priority_shadow_gate={priority_shadow_gate_value}; failures={priority_shadow.get('failures') if priority_shadow else None}",
    )

    checklist = pd.DataFrame(rows)
    checklist.to_csv(output_dir / "market_state_plan_completion_checklist.csv", index=False)
    status_counts = checklist["status"].value_counts().to_dict()
    hard_failures = checklist.loc[checklist["status"].isin([STATUS_FAILED, STATUS_MISSING])].copy()
    payload = {
        "generated_by": "audit_market_state_plan_completion",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": str(artifact_dir),
        "output_dir": str(output_dir),
        "expected_rank_contract": expected_rank_contract,
        "controller_promotion_audit_dir": (
            str(controller_promotion_audit_dir) if controller_promotion_audit_dir else None
        ),
        "state_head_quality_dir": str(state_head_quality_dir) if state_head_quality_dir else None,
        "strategy_response_quality_dir": str(strategy_response_quality_dir) if strategy_response_quality_dir else None,
        "shadow_priority_audit_dir": str(shadow_priority_audit_dir) if shadow_priority_audit_dir else None,
        "shadow_controller_monitor_dir": str(shadow_controller_monitor_dir)
        if shadow_controller_monitor_dir
        else None,
        "direct_suppression_training_dir": (
            str(direct_suppression_training_dir)
            if direct_suppression_training_dir
            else None
        ),
        "direct_suppression_actionability_audit_dir": (
            str(direct_suppression_actionability_audit_dir)
            if direct_suppression_actionability_audit_dir
            else None
        ),
        "direct_suppression_policy_scopes": sorted(direct_scopes),
        "direct_suppression_selected_arm": direct_selected_arm,
        "direct_suppression_selection_reason": direct_selection_reason,
        "direct_suppression_best_attempt_policy_scope": direct_best.get("policy_scope"),
        "direct_suppression_best_attempt_target_head": direct_best.get("target_head"),
        "direct_suppression_best_attempt_target_strategy_id": direct_best.get("target_strategy_id"),
        "direct_suppression_best_attempt_suppressed_rows": direct_best.get("suppressed_rows"),
        "direct_suppression_best_attempt_suppressed_folds": direct_best.get("suppressed_folds"),
        "direct_suppression_actionability_selected_arm": actionability_selected_arm,
        "direct_suppression_actionability_selection_reason": actionability_selection_reason,
        "direct_suppression_actionability_dominant_blocker": actionability_dominant_blocker,
        "direct_suppression_actionability_passing_rows": actionability_passing_rows,
        "direct_suppression_actionability_recurrent_support_rows": actionability_recurrent_rows,
        "direct_suppression_actionability_max_recurrent_defensive_success": actionability_max_recurrent_success,
        "direct_suppression_actionability_max_recurrent_positive_fold_share": (
            actionability_max_recurrent_positive_share
        ),
        "shadow_controller_monitor_contract": shadow_controller_monitor.get("monitor_contract"),
        "shadow_controller_monitor_gate_passed": shadow_controller_monitor.get("shadow_promotion_gate_passed"),
        "shadow_controller_monitor_should_disable": shadow_controller_monitor.get("controller_should_remain_disabled"),
        "controller_promotion_ready": promotion_audit.get("controller_promotion_ready"),
        "controller_action_attribution_gate_passed": (
            (promotion_audit.get("action_attribution_gate") or {}).get("passed")
        ),
        "shadow_controller_direct_threshold_gate_passed": shadow_controller_monitor.get(
            "direct_threshold_only_promotion_gate_passed"
        ),
        "shadow_controller_locked_overlay_gate_passed": shadow_controller_monitor.get(
            "locked_accepted_overlay_promotion_gate_passed"
        ),
        "shadow_priority_recurrent_challenger_selected": priority_recurrent.get("selected")
        if priority_recurrent
        else None,
        "shadow_priority_recurrent_selection_reason": priority_recurrent.get("reason")
        if priority_recurrent
        else None,
        "shadow_priority_recurrent_best_candidate": (
            (priority_recurrent.get("best_candidate") or {}).get("arm_selector")
            if priority_recurrent
            else None
        ),
        "backend_comparison_dir": str(backend_comparison_dir) if backend_comparison_dir else None,
        "ablation_matrix_evidence_dir": (
            str(ablation_matrix_evidence_dir) if ablation_matrix_evidence_dir else None
        ),
        "passed_structural_audit": bool(hard_failures.empty),
        "status_counts": status_counts,
        "hard_failure_count": int(len(hard_failures)),
        "hard_failures": hard_failures.to_dict("records"),
        "gate_blocked_requirements": checklist.loc[checklist["status"].eq(STATUS_GATE_BLOCKED)].to_dict("records"),
        "shadow_only_requirements": checklist.loc[checklist["status"].eq(STATUS_SHADOW)].to_dict("records"),
    }
    (output_dir / "market_state_plan_completion_audit.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(output_dir, payload, checklist)
    return payload


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 80) -> str:
    if frame.empty:
        return "_No rows._\n"
    view = frame.loc[:, [col for col in columns if col in frame.columns]].head(max_rows).copy()
    lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in view.columns) + " |")
    return "\n".join(lines) + "\n"


def _write_report(output_dir: Path, payload: dict[str, Any], checklist: pd.DataFrame) -> None:
    counts = pd.DataFrame(
        [{"status": key, "count": value} for key, value in Counter(checklist["status"]).items()]
    ).sort_values(["status"])
    lines = [
        "# Market-State Plan Completion Audit",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Summary",
        "",
        f"- Structural audit passed: `{payload['passed_structural_audit']}`",
        f"- Hard failure count: `{payload['hard_failure_count']}`",
        "",
        _markdown_table(counts, ["status", "count"]),
        "## Hard Failures",
        "",
        _markdown_table(
            pd.DataFrame(payload["hard_failures"]),
            ["requirement_id", "section", "requirement", "status", "evidence", "notes"],
        ),
        "## Gate-Blocked Requirements",
        "",
        _markdown_table(
            pd.DataFrame(payload["gate_blocked_requirements"]),
            ["requirement_id", "section", "requirement", "status", "evidence", "notes"],
        ),
        "## Shadow-Only Requirements",
        "",
        _markdown_table(
            pd.DataFrame(payload["shadow_only_requirements"]),
            ["requirement_id", "section", "requirement", "status", "evidence", "notes"],
        ),
        "## Checklist",
        "",
        _markdown_table(
            checklist,
            ["requirement_id", "section", "requirement", "status", "evidence", "notes"],
            max_rows=120,
        ),
    ]
    (output_dir / "market_state_plan_completion_audit.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--controller-promotion-audit-dir", type=Path)
    parser.add_argument("--state-head-quality-dir", type=Path)
    parser.add_argument("--strategy-response-quality-dir", type=Path)
    parser.add_argument("--shadow-priority-audit-dir", type=Path)
    parser.add_argument("--shadow-controller-monitor-dir", type=Path)
    parser.add_argument("--direct-suppression-training-dir", type=Path)
    parser.add_argument("--direct-suppression-actionability-audit-dir", type=Path)
    parser.add_argument("--backend-comparison-dir", type=Path)
    parser.add_argument("--ablation-matrix-evidence-dir", type=Path)
    parser.add_argument("--expected-rank-contract", default=EXPECTED_RANK_CONTRACT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = audit_market_state_plan_completion(
        args.artifact_dir,
        args.output_dir,
        expected_rank_contract=str(args.expected_rank_contract),
        controller_promotion_audit_dir=args.controller_promotion_audit_dir,
        state_head_quality_dir=args.state_head_quality_dir,
        strategy_response_quality_dir=args.strategy_response_quality_dir,
        shadow_priority_audit_dir=args.shadow_priority_audit_dir,
        shadow_controller_monitor_dir=args.shadow_controller_monitor_dir,
        direct_suppression_training_dir=args.direct_suppression_training_dir,
        direct_suppression_actionability_audit_dir=args.direct_suppression_actionability_audit_dir,
        backend_comparison_dir=args.backend_comparison_dir,
        ablation_matrix_evidence_dir=args.ablation_matrix_evidence_dir,
    )
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
