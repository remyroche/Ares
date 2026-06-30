#!/usr/bin/env python3
"""Audit market-state controller artifacts against the T1 contract.

The audit is intentionally structural: it checks the frozen scope and safety
contract before any economic interpretation. It does not promote a controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_ACTIVE_HEADS = ["short_asset", "short_boll"]
EXPECTED_DISABLED_HEADS = ["long_bars", "long_dist"]
EXPECTED_RANK_CONTRACT = "short_boll_timestamp_rank"
SUPPORTED_RANK_CONTRACTS = {
    "short_boll_timestamp_rank",
    "anchor_global_policy_rank_reference",
}
REQUIRED_ARTIFACTS = [
    "manifest.json",
    "market_state_feature_contract.json",
    "market_state_universe_contract.json",
    "market_state_training_reference.joblib",
    "market_state_timestamp_panel.parquet",
    "market_state_feature_coverage.csv",
    "market_state_target_definitions.json",
    "market_state_target_cdfs.joblib",
    "market_state_oof_predictions.parquet",
    "market_state_head_diagnostics.csv",
    "strategy_rank_outcome_curves.joblib",
    "strategy_residual_target_ledger.parquet",
    "strategy_response_models.joblib",
    "strategy_response_oof_predictions.parquet",
    "strategy_state_effect_matrix.csv",
    "strategy_threshold_schedule.parquet",
    "strategy_threshold_controller_config.json",
    "strategy_threshold_action_audit.csv",
    "portfolio_replay_summary.csv",
    "portfolio_replay_by_head.csv",
    "artifact_hashes.json",
    "market_state_activation_registry.csv",
    "walkforward_state_head_registry.csv",
    "walkforward_selected_controller_candidate.json",
    "walkforward_controller_state_diagnostics.csv",
    "walkforward_threshold_action_utility.csv",
    "walkforward_threshold_action_edge_validation.csv",
    "walkforward_threshold_action_edge_bucket_performance.csv",
    "walkforward_threshold_candidate_suppression_utility.csv",
    "walkforward_threshold_candidate_suppression_aggregate.csv",
    "walkforward_threshold_baseline_accepted_suppression_utility.csv",
    "walkforward_threshold_baseline_accepted_suppression_aggregate.csv",
    "market_state_leave_one_head_out_aggregate.csv",
]
MODEL_ARTIFACT_OPTIONS = [
    "market_state_lgbm_models.joblib",
    "market_state_xgb_models.joblib",
]
OPTIONAL_HASHED_ARTIFACTS = [
    "accepted_trades.parquet",
]
MATERIALIZED_BUNDLE_REQUIRED_ARTIFACTS = [
    "manifest.json",
    "market_state_controller_bundle.joblib",
    "market_state_feature_contract.json",
    "market_state_universe_contract.json",
    "market_state_training_reference.joblib",
    "market_state_timestamp_panel.parquet",
    "market_state_feature_coverage.csv",
    "train_market_state_features.csv",
    "strategy_threshold_controller_config.json",
    "artifact_hashes.json",
    "controller_predictions.parquet",
    "controller_schedule.csv",
    "strategy_threshold_schedule.parquet",
    "strategy_threshold_action_audit.csv",
    "controller_scored_candidates.parquet",
    "decisions.parquet",
    "accepted_trades.parquet",
    "controller_replay_summary.csv",
    "controller_replay_by_head.csv",
]
SCORED_BUNDLE_REQUIRED_ARTIFACTS = [
    "manifest.json",
    "market_state_feature_contract.json",
    "market_state_timestamp_panel.parquet",
    "market_state_feature_coverage.csv",
    "strategy_threshold_controller_config.json",
    "controller_predictions.parquet",
    "controller_schedule.csv",
    "strategy_threshold_schedule.parquet",
    "strategy_threshold_action_audit.csv",
    "controller_scored_candidates.parquet",
    "decisions.parquet",
    "accepted_trades.parquet",
    "controller_replay_summary.csv",
    "controller_replay_by_head.csv",
]
SHADOW_BUNDLE_REQUIRED_ARTIFACTS = [
    "shadow_controller_proposed_schedule.csv",
    "shadow_controller_proposed_schedule.parquet",
    "shadow_threshold_action_audit.csv",
    "shadow_threshold_candidate_suppression_utility.csv",
]
FORBIDDEN_MARKET_STATE_COLUMNS = {
    "strategy_id",
    "strategy",
    "head",
    "side",
    "symbol",
    "candidate_count",
    "accepted_trade_count",
    "accepted",
    "portfolio_pnl",
    "net_pnl",
    "gross_pnl",
    "net_return",
    "label",
    "target",
    "y",
    "rank",
    "rank_pct",
    "policy_rank_pct",
    "strategy_rank_pct",
    "score",
    "calibrated_score",
}
FORBIDDEN_MARKET_STATE_TOKENS = {
    "accepted",
    "anchor",
    "candidate",
    "confidence",
    "decision",
    "fail",
    "failure",
    "headhealth",
    "leaf",
    "ledger",
    "margin",
    "meta",
    "model",
    "pnl",
    "policy",
    "portfolio",
    "prediction",
    "qfail",
    "rank",
    "reliability",
    "score",
    "strategy",
    "symbol",
    "target",
    "trade",
    "y",
    "ybin",
}
STATE_JOIN_INVARIANT_COLUMNS = [
    "state_feature_coverage",
    "state_ood_score",
    "state_drift_score",
    "state_ood_cutoff",
    "state_ood_flag",
]
ALLOWED_MARKET_STATE_SEMANTIC_FEATURES = {
    # Plan-approved market-state reliability channels. Keep this explicit so the
    # semantic audit still rejects generic score/model/rank/performance fields.
    "state_drift_score",
    "state_ood_score",
    "state_ood_score_mean",
    "state_ood_score_max",
}
DECISION_KEY_COLS = ["timestamp", "symbol", "side", "strategy_id"]
ARTIFACT_AUDIT_CHECKS = [
    "required_files_present",
    "artifact_hashes_present_complete_and_verified",
    "required_payload_artifacts_hash_covered",
    "feature_contract_matches_t1_scope",
    "feature_contract_invariants_enforced",
    "feature_contract_source_audit_passed",
    "feature_contract_source_feature_names_safe",
    "feature_contract_activation_registry_filter_enforced_when_present",
    "market_state_universe_contract_persisted_and_verified",
    "chronological_fold_embargo_enforced",
    "training_outcome_maturity_contract_enforced",
    "fold_fitted_reference_and_model_bundles_persisted",
    "feature_store_tail_references_fold_fitted",
    "target_definition_and_cdfs_are_fold_fitted",
    "controller_config_matches_t1_scope",
    "selected_controller_is_null",
    "state_head_registries_present_and_versioned",
    "market_state_one_row_per_fold_split_arm_timestamp",
    "no_forbidden_market_state_columns",
    "oof_prediction_contract_present",
    "oof_state_values_match_timestamp_panel",
    "response_oof_uses_oof_state_scores",
    "state_join_diagnostics_constant_within_fold_arm_timestamp",
    "state_threshold_never_below_base_threshold",
    "force_base_rows_equal_base_threshold",
    "missing_or_ood_state_falls_back_to_base_threshold",
    "unique_strategy_threshold_schedule_rows",
    "static_baseline_replay_parity",
    "accepted_decision_keys_unique_when_available",
    "state_head_diagnostics_metric_coverage",
    "strategy_response_metric_coverage",
    "controller_metric_coverage",
    "portfolio_metric_coverage",
    "leave_one_state_head_out_metric_coverage",
]
BUNDLE_ARTIFACT_AUDIT_CHECKS = [
    "bundle_required_files_present",
    "bundle_hashes_present_complete_and_verified",
    "bundle_feature_contract_matches_t1_scope",
    "bundle_feature_contract_invariants_enforced",
    "bundle_feature_contract_source_audit_passed",
    "bundle_feature_contract_source_feature_names_safe",
    "bundle_feature_contract_activation_registry_filter_enforced_when_present",
    "bundle_universe_contract_persisted_and_verified_when_materialized",
    "bundle_market_state_one_row_per_split_state_level_timestamp",
    "bundle_no_forbidden_market_state_columns",
    "bundle_applied_schedule_never_lowers_thresholds",
    "bundle_disabled_controller_applied_schedule_is_noop",
    "bundle_shadow_schedule_never_lowers_thresholds",
    "bundle_shadow_action_audit_present",
    "bundle_shadow_suppression_utility_present",
    "bundle_replay_summary_metric_coverage",
    "bundle_accepted_decision_keys_unique",
]
FEATURE_CONTRACT_TRUE_INVARIANTS = [
    "one_market_state_row_per_timestamp",
    "state_join_timestamp_constant",
]
FEATURE_CONTRACT_FALSE_INVARIANTS = [
    "market_state_uses_strategy_ids",
    "market_state_uses_model_predictions",
    "market_state_uses_ranks",
    "market_state_uses_candidate_counts",
    "market_state_uses_portfolio_pnl",
    "market_state_uses_realized_strategy_outcomes",
    "actual_order_book_features_allowed",
    "candidate_population_fallback_enabled",
    "candidate_population_fallback_is_production_safe",
    "controller_changes_scores_or_ranks",
    "controller_changes_auction_ordering",
    "controller_can_lower_thresholds",
    "latent_gmm_active_controller_input",
]
ORDER_BOOK_FEATURE_TOKENS = {
    "ask",
    "bid",
    "book",
    "cancel",
    "cancellation",
    "depth",
    "imbalance",
    "microprice",
    "orderbook",
    "replenish",
    "replenishment",
}
SOURCE_SCHEMA_FEATURE_LIST_KEYS = [
    "feature_store_columns",
    "observed_axis_columns",
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} did not contain a JSON object")
    return payload


def _controller(manifest: dict[str, Any]) -> dict[str, Any]:
    value = manifest.get("controller", {})
    return value if isinstance(value, dict) else {}


def _contract_value(manifest: dict[str, Any], controller: dict[str, Any], key: str) -> Any:
    if key in controller:
        return controller.get(key)
    return manifest.get(key)


def _audit_source_contract_audit(source_audit: Any, *, prefix: str) -> list[str]:
    failures: list[str] = []
    if not isinstance(source_audit, dict):
        failures.append(f"{prefix} is missing")
        return failures

    if source_audit.get("overall_passed") is not True:
        failures.append(f"{prefix}.overall_passed is not true")
    if source_audit.get("actual_order_book_features_allowed") is not False:
        failures.append(f"{prefix} allows actual order-book features")
    if source_audit.get("candidate_population_fallback_allowed_for_production") is not False:
        failures.append(f"{prefix} allows production candidate fallback")
    splits = source_audit.get("splits")
    if not isinstance(splits, dict) or not splits:
        failures.append(f"{prefix}.splits is missing")
        return failures

    for split, payload in splits.items():
        if not isinstance(payload, dict):
            failures.append(f"{prefix}.{split} is not an object")
            continue
        if payload.get("passed") is not True:
            failures.append(f"{prefix}.{split}.passed is not true")
        if payload.get("production_safe") is not True:
            failures.append(f"{prefix}.{split}.production_safe is not true")
        if payload.get("candidate_fallback_enabled") is not False:
            failures.append(f"{prefix}.{split}.candidate_fallback_enabled is not false")
        if int(payload.get("validation_forbidden_column_count") or 0) != 0:
            failures.append(f"{prefix}.{split}.validation_forbidden_column_count != 0")
        if payload.get("timestamp_unique") is not True:
            failures.append(f"{prefix}.{split}.timestamp_unique is not true")
        if payload.get("market_wide_one_row_per_timestamp") is not True:
            failures.append(f"{prefix}.{split}.market_wide_one_row_per_timestamp is not true")
        source = payload.get("source")
        if source is not None and source != "feature_store_market_aggregates":
            failures.append(f"{prefix}.{split}.source is not feature_store_market_aggregates")
        if payload.get("feature_store_enabled") is False:
            failures.append(f"{prefix}.{split}.feature_store_enabled is false")
    return failures


def audit_manifest(
    manifest: dict[str, Any],
    *,
    require_null_selection: bool = False,
    allow_disabled_by_activation_registry: bool = False,
    expected_rank_contract: str = EXPECTED_RANK_CONTRACT,
) -> list[str]:
    failures: list[str] = []
    if manifest.get("rank_contract") != expected_rank_contract:
        failures.append(f"rank_contract != {expected_rank_contract}")
    if sorted(manifest.get("disabled_heads") or []) != EXPECTED_DISABLED_HEADS:
        failures.append(f"disabled_heads != {EXPECTED_DISABLED_HEADS}")
    if manifest.get("active_heads") != EXPECTED_ACTIVE_HEADS:
        failures.append(f"active_heads != {EXPECTED_ACTIVE_HEADS}")

    controller = _controller(manifest)
    if controller.get("penalty_only") is not True:
        failures.append("controller.penalty_only is not true")
    if controller.get("changes_scores_or_ranks") is not False:
        failures.append("controller.changes_scores_or_ranks is not false")
    if controller.get("changes_auction_ordering") is not False:
        failures.append("controller.changes_auction_ordering is not false")
    enabled_heads = _contract_value(manifest, controller, "controller_enabled_heads")
    enabled_scope = _contract_value(manifest, controller, "controller_enabled_scope")
    execution_enabled = _contract_value(manifest, controller, "controller_execution_enabled")
    if execution_enabled is None:
        execution_enabled = _contract_value(manifest, controller, "execution_enabled")
    shadow_controller_only = bool(
        _contract_value(manifest, controller, "shadow_controller_only")
        or manifest.get("shadow_controller_only")
    )
    shadow_enabled_heads = manifest.get("shadow_controller_enabled_heads")
    shadow_enabled_scope = manifest.get("shadow_controller_enabled_scope")
    disabled_by_registry = (
        allow_disabled_by_activation_registry
        and execution_enabled is False
        and enabled_heads == []
        and enabled_scope == "disabled_by_activation_registry"
    )
    shadow_only_valid = (
        shadow_controller_only
        and execution_enabled is False
        and enabled_heads == []
        and enabled_scope in {"disabled_by_activation_registry", None}
        and shadow_enabled_heads == EXPECTED_ACTIVE_HEADS
        and shadow_enabled_scope in {"all_active_heads", "explicit"}
    )
    if not disabled_by_registry and not shadow_only_valid:
        if enabled_heads != EXPECTED_ACTIVE_HEADS:
            failures.append(f"controller.controller_enabled_heads != {EXPECTED_ACTIVE_HEADS}")
        if enabled_scope != "all_active_heads":
            failures.append("controller.controller_enabled_scope != all_active_heads")
    candidate_fallback = _contract_value(manifest, controller, "allow_candidate_state_fallback")
    if candidate_fallback is not None and candidate_fallback is not False:
        failures.append("candidate-state fallback is enabled")
    latent_enabled = _contract_value(manifest, controller, "include_latent_shadow_arms")
    latent_report = manifest.get("latent_report") if isinstance(manifest.get("latent_report"), dict) else {}
    if latent_enabled is None and latent_report:
        latent_enabled = str(latent_report.get("mode", "")).startswith("latent")
    if latent_enabled is not None and latent_enabled is not False:
        failures.append("latent/GMM shadow arms are enabled")

    failures.extend(_audit_source_contract_audit(manifest.get("source_contract_audit"), prefix="source_contract_audit"))

    selected = manifest.get("selected_controller_candidate")
    if isinstance(selected, dict):
        selected_arm = selected.get("selected_arm")
    else:
        selected_arm = manifest.get("selected_arm")
    if require_null_selection and selected_arm is not None:
        failures.append("selected controller is not null despite require-null-selection")
    return failures


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    raise ValueError(f"unsupported tabular artifact: {path}")


def _load_artifact_json(artifact_dir: Path, name: str) -> dict[str, Any]:
    return _load_json(artifact_dir / name)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_artifact_path(path_value: Any, artifact_dir: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if path.exists():
        return path
    candidate = artifact_dir / path
    if candidate.exists():
        return candidate
    return path


def _missing_artifacts(artifact_dir: Path) -> list[str]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (artifact_dir / name).exists()]
    if not any((artifact_dir / name).exists() for name in MODEL_ARTIFACT_OPTIONS):
        missing.append("one_of:" + "|".join(MODEL_ARTIFACT_OPTIONS))
    return missing


def _artifact_bundle_kind(manifest: dict[str, Any]) -> str:
    generated_by = str(manifest.get("generated_by") or "")
    if generated_by == "materialize_market_state_controller_bundle":
        return "materialized_bundle"
    if generated_by == "score_market_state_controller_bundle":
        return "scored_bundle"
    return "walkforward"


def _missing_bundle_artifacts(artifact_dir: Path, manifest: dict[str, Any]) -> list[str]:
    kind = _artifact_bundle_kind(manifest)
    required = (
        MATERIALIZED_BUNDLE_REQUIRED_ARTIFACTS
        if kind == "materialized_bundle"
        else SCORED_BUNDLE_REQUIRED_ARTIFACTS
    )
    if bool(manifest.get("shadow_controller_only", False)):
        required = [*required, *SHADOW_BUNDLE_REQUIRED_ARTIFACTS]
    return [name for name in required if not (artifact_dir / name).exists()]


def _duplicate_count(frame: pd.DataFrame, keys: list[str]) -> int:
    if not all(key in frame.columns for key in keys):
        return -1
    return int(frame.duplicated(keys).sum())


def _forbidden_columns(frame: pd.DataFrame) -> list[str]:
    columns = set(map(str, frame.columns))
    return sorted(columns.intersection(FORBIDDEN_MARKET_STATE_COLUMNS))


def _unsafe_order_book_feature_names(names: list[str]) -> list[str]:
    unsafe: list[str] = []
    for name in names:
        feature = str(name)
        tokens = {token for token in feature.lower().replace("-", "_").split("_") if token}
        if tokens.intersection(ORDER_BOOK_FEATURE_TOKENS):
            unsafe.append(feature)
    return sorted(set(unsafe))


def _unsafe_market_state_semantic_feature_names(names: list[str]) -> list[str]:
    unsafe: list[str] = []
    for name in names:
        feature = str(name)
        normalized = feature.lower().replace("-", "_")
        if not (normalized.startswith("state_") or normalized.startswith("forecast_")):
            continue
        if normalized in ALLOWED_MARKET_STATE_SEMANTIC_FEATURES:
            continue
        tokens = {token for token in normalized.split("_") if token}
        suffix = normalized.removeprefix("state_").removeprefix("forecast_")
        if suffix in FORBIDDEN_MARKET_STATE_COLUMNS or tokens.intersection(FORBIDDEN_MARKET_STATE_TOKENS):
            unsafe.append(feature)
    return sorted(set(unsafe))


def _audit_market_state_frame(frame: pd.DataFrame, *, name: str, include_split: bool) -> list[str]:
    failures: list[str] = []
    required = ["fold", "state_arm", "timestamp"]
    if include_split:
        required.append("split")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        failures.append(f"{name} missing columns: {missing}")
        return failures

    duplicate_keys = required
    duplicates = _duplicate_count(frame, duplicate_keys)
    if duplicates > 0:
        failures.append(f"{name} has duplicate market-state rows for {duplicate_keys}: {duplicates}")

    forbidden = _forbidden_columns(frame)
    if forbidden:
        failures.append(f"{name} contains forbidden market-state columns: {forbidden}")
    unsafe_order_book = _unsafe_order_book_feature_names(list(map(str, frame.columns)))
    if unsafe_order_book:
        failures.append(f"{name} contains actual order-book-like columns: {unsafe_order_book[:10]}")
    unsafe_semantic = _unsafe_market_state_semantic_feature_names(list(map(str, frame.columns)))
    if unsafe_semantic:
        failures.append(
            f"{name} contains strategy/model/performance-like market-state columns: {unsafe_semantic[:10]}"
        )

    return failures


def _audit_state_join_invariance(frame: pd.DataFrame, *, name: str) -> list[str]:
    failures: list[str] = []
    keys = ["fold", "arm", "timestamp"]
    missing_keys = [column for column in keys if column not in frame.columns]
    if missing_keys:
        failures.append(f"{name} missing join-invariance keys: {missing_keys}")
        return failures

    invariant_columns = [column for column in STATE_JOIN_INVARIANT_COLUMNS if column in frame.columns]
    if not invariant_columns:
        failures.append(f"{name} has no state join invariant columns")
        return failures

    for column in invariant_columns:
        counts = frame.groupby(keys, dropna=False)[column].nunique(dropna=False)
        bad = int((counts > 1).sum())
        if bad:
            failures.append(f"{name}.{column} varies within fold/arm/timestamp groups: {bad}")
    return failures


def _state_value_columns(frame: pd.DataFrame) -> list[str]:
    key_columns = {"fold", "split", "state_arm", "arm", "timestamp"}
    return [
        column
        for column in map(str, frame.columns)
        if column not in key_columns and (column.startswith("state_") or column.startswith("forecast_"))
    ]


def _normalise_timestamp_key(frame: pd.DataFrame, timestamp_column: str = "timestamp") -> pd.DataFrame:
    out = frame.copy()
    out[timestamp_column] = pd.to_datetime(out[timestamp_column], utc=True, errors="coerce")
    return out


def _audit_oof_state_values_match_panel(panel: pd.DataFrame, state_oof: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    key = ["fold", "split", "state_arm", "timestamp"]
    missing_panel = [column for column in key if column not in panel.columns]
    missing_oof = [column for column in key if column not in state_oof.columns]
    if missing_panel or missing_oof:
        failures.append(
            "cannot compare market_state_oof_predictions to timestamp panel; "
            f"missing_panel={missing_panel}, missing_oof={missing_oof}"
        )
        return failures

    common_value_columns = sorted(set(_state_value_columns(panel)).intersection(_state_value_columns(state_oof)))
    if not common_value_columns:
        failures.append("no common state/forecast value columns for OOF state parity check")
        return failures

    panel_work = _normalise_timestamp_key(panel.loc[:, key + common_value_columns])
    oof_work = _normalise_timestamp_key(state_oof.loc[:, key + common_value_columns])
    merged = oof_work.merge(
        panel_work,
        on=key,
        how="left",
        suffixes=("_oof", "_panel"),
        indicator=True,
    )
    missing_rows = int(merged["_merge"].ne("both").sum())
    if missing_rows:
        failures.append(f"market_state_oof_predictions rows missing from timestamp panel: {missing_rows}")
        return failures

    mismatched_columns: list[str] = []
    for column in common_value_columns:
        left = merged[f"{column}_oof"]
        right = merged[f"{column}_panel"]
        both_na = left.isna() & right.isna()
        if pd.api.types.is_numeric_dtype(left) or pd.api.types.is_numeric_dtype(right):
            diff = (pd.to_numeric(left, errors="coerce") - pd.to_numeric(right, errors="coerce")).abs()
            bad = ~(both_na | (diff <= 1e-12))
        else:
            bad = ~(both_na | left.astype(str).eq(right.astype(str)))
        if bool(bad.any()):
            mismatched_columns.append(f"{column}:{int(bad.sum())}")
    if mismatched_columns:
        failures.append(
            "market_state_oof_predictions values differ from timestamp panel: "
            f"{mismatched_columns[:10]}"
        )
    return failures


def _audit_response_oof_uses_state_oof(response_oof: pd.DataFrame, state_oof: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    response_keys = ["fold", "arm", "timestamp"]
    state_keys = ["fold", "state_arm", "timestamp"]
    missing_response = [column for column in response_keys if column not in response_oof.columns]
    missing_state = [column for column in state_keys if column not in state_oof.columns]
    if missing_response or missing_state:
        failures.append(
            "cannot verify response OOF state coverage; "
            f"missing_response={missing_response}, missing_state={missing_state}"
        )
        return failures
    if response_oof.empty:
        failures.append("strategy_response_oof_predictions is empty")
        return failures

    response_keys_frame = _normalise_timestamp_key(
        response_oof.loc[:, response_keys].drop_duplicates().rename(columns={"arm": "state_arm"})
    )
    state_keys_frame = _normalise_timestamp_key(state_oof.loc[:, state_keys].drop_duplicates())
    merged = response_keys_frame.merge(state_keys_frame, on=state_keys, how="left", indicator=True)
    missing = int(merged["_merge"].ne("both").sum())
    if missing:
        failures.append(f"strategy_response_oof_predictions keys missing from market_state_oof_predictions: {missing}")
    return failures


def _audit_threshold_schedule(frame: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    required = [
        "timestamp",
        "strategy_id",
        "head",
        "fold",
        "arm",
        "base_threshold",
        "state_threshold",
        "raw_state_threshold",
        "threshold_action_enabled",
        "force_base_threshold",
        "controller_reason",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        failures.append(f"strategy_threshold_schedule missing columns: {missing}")
        return failures

    base = pd.to_numeric(frame["base_threshold"], errors="coerce")
    state = pd.to_numeric(frame["state_threshold"], errors="coerce")
    if bool(base.isna().any() or state.isna().any()):
        failures.append("strategy_threshold_schedule contains non-finite base/state thresholds")
    else:
        lowered = int((state < base - 1e-12).sum())
        if lowered:
            failures.append(f"strategy_threshold_schedule lowers thresholds below base: {lowered}")

    force_base = frame["force_base_threshold"].fillna(False).astype(bool)
    if force_base.any() and not bool(base.isna().any() or state.isna().any()):
        not_base = int(((state - base).abs() > 1e-12).loc[force_base].sum())
        if not_base:
            failures.append(f"force_base_threshold rows do not equal base threshold: {not_base}")

    fallback_mask = pd.Series(False, index=frame.index)
    if {"prediction_coverage", "min_prediction_coverage"}.issubset(frame.columns):
        coverage = pd.to_numeric(frame["prediction_coverage"], errors="coerce")
        min_coverage = pd.to_numeric(frame["min_prediction_coverage"], errors="coerce")
        coverage_bad = coverage.notna() & min_coverage.notna() & (coverage < min_coverage - 1e-12)
        fallback_mask |= coverage_bad
    if "state_ood_share" in frame.columns:
        ood_share = pd.to_numeric(frame["state_ood_share"], errors="coerce")
        fallback_mask |= ood_share.notna() & (ood_share >= 1.0 - 1e-12)
    if {"state_ood_score_max", "state_ood_cutoff", "prediction_coverage", "min_prediction_coverage"}.issubset(
        frame.columns
    ):
        ood_max = pd.to_numeric(frame["state_ood_score_max"], errors="coerce")
        ood_cutoff = pd.to_numeric(frame["state_ood_cutoff"], errors="coerce")
        coverage = pd.to_numeric(frame["prediction_coverage"], errors="coerce")
        min_coverage = pd.to_numeric(frame["min_prediction_coverage"], errors="coerce")
        fallback_mask |= (
            ood_max.notna()
            & ood_cutoff.notna()
            & (ood_max > ood_cutoff + 1e-12)
            & coverage.notna()
            & min_coverage.notna()
            & (coverage < min_coverage - 1e-12)
        )
    if fallback_mask.any():
        bad_force = int((~force_base).loc[fallback_mask].sum())
        if bad_force:
            failures.append(f"missing/OOD fallback rows are not force_base_threshold: {bad_force}")
        if not bool(base.isna().any() or state.isna().any()):
            bad_threshold = int(((state - base).abs() > 1e-12).loc[fallback_mask].sum())
            if bad_threshold:
                failures.append(f"missing/OOD fallback rows do not equal base threshold: {bad_threshold}")
        if "controller_reason" in frame.columns:
            reasons = frame["controller_reason"].astype(str)
            allowed_reasons = {"insufficient_prediction_coverage", "state_ood_fallback"}
            bad_reason = int((~reasons.isin(allowed_reasons)).loc[fallback_mask].sum())
            if bad_reason:
                failures.append(f"missing/OOD fallback rows have unexpected controller_reason: {bad_reason}")

    duplicate_keys = ["fold", "arm", "timestamp", "strategy_id"]
    duplicates = _duplicate_count(frame, duplicate_keys)
    if duplicates > 0:
        failures.append(f"strategy_threshold_schedule duplicate rows for {duplicate_keys}: {duplicates}")
    return failures


def _audit_bundle_threshold_schedule(
    frame: pd.DataFrame,
    *,
    name: str,
    require_noop: bool,
    require_shadow_arm: bool = False,
) -> list[str]:
    failures: list[str] = []
    required = [
        "timestamp",
        "strategy_id",
        "head",
        "base_threshold",
        "state_threshold",
        "raw_state_threshold",
        "threshold_action_enabled",
        "force_base_threshold",
        "controller_reason",
    ]
    if require_shadow_arm:
        required.append("arm")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        failures.append(f"{name} missing columns: {missing}")
        return failures
    if frame.empty:
        failures.append(f"{name} is empty")
        return failures

    base = pd.to_numeric(frame["base_threshold"], errors="coerce")
    state = pd.to_numeric(frame["state_threshold"], errors="coerce")
    raw_state = pd.to_numeric(frame["raw_state_threshold"], errors="coerce")
    if bool(base.isna().any() or state.isna().any() or raw_state.isna().any()):
        failures.append(f"{name} contains non-finite base/state thresholds")
        return failures
    lowered = int((state < base - 1e-12).sum())
    if lowered:
        failures.append(f"{name} lowers thresholds below base: {lowered}")
    raw_lowered = int((raw_state < base - 1e-12).sum())
    if raw_lowered:
        failures.append(f"{name} raw_state_threshold below base: {raw_lowered}")
    if require_noop:
        raised = int((state > base + 1e-12).sum())
        if raised:
            failures.append(f"{name} is expected to be no-op but raises thresholds: {raised}")
        enabled = frame["threshold_action_enabled"].fillna(False).astype(bool)
        if bool(enabled.any()):
            failures.append(f"{name} is expected to be no-op but has enabled threshold actions")
    force_base = frame["force_base_threshold"].fillna(False).astype(bool)
    if force_base.any():
        bad = int(((state - base).abs() > 1e-12).loc[force_base].sum())
        if bad:
            failures.append(f"{name} force_base_threshold rows do not equal base threshold: {bad}")
    duplicate_keys = ["timestamp", "strategy_id"]
    duplicates = _duplicate_count(frame, duplicate_keys)
    if duplicates > 0:
        failures.append(f"{name} duplicate rows for {duplicate_keys}: {duplicates}")
    return failures


def _audit_bundle_action_audit(frame: pd.DataFrame, *, name: str) -> list[str]:
    failures: list[str] = []
    required = [
        "scope",
        "scope_value",
        "schedule_rows",
        "threshold_raised_count",
        "threshold_raised_share",
        "force_base_count",
        "force_base_share",
        "mean_base_threshold",
        "mean_state_threshold",
        "mean_threshold_delta",
        "max_threshold_delta",
    ]
    failures.extend(_audit_required_columns(frame, name=name, required=required))
    failures.extend(_audit_nonempty(frame, name=name))
    if not frame.empty and not [column for column in required if column not in frame.columns]:
        failures.extend(
            _audit_numeric_finite(
                frame,
                name=name,
                columns=[
                    "schedule_rows",
                    "threshold_raised_count",
                    "threshold_raised_share",
                    "force_base_count",
                    "force_base_share",
                    "mean_base_threshold",
                    "mean_state_threshold",
                    "mean_threshold_delta",
                    "max_threshold_delta",
                ],
            )
        )
        failures.extend(
            _audit_ratio_bounds(
                frame,
                name=name,
                columns=["threshold_raised_share", "force_base_share"],
            )
        )
    return failures


def _audit_bundle_suppression_utility(frame: pd.DataFrame, *, name: str) -> list[str]:
    failures: list[str] = []
    required = [
        "arm",
        "scope",
        "scope_value",
        "suppressed_candidates",
        "raised_schedule_count",
        "mean_threshold_delta",
        "suppressed_loss_avoided",
        "suppressed_winner_pnl_sacrificed",
        "realized_defensive_success",
        "suppressed_full_sl_rate",
        "suppressed_timeout_rate",
    ]
    failures.extend(_audit_required_columns(frame, name=name, required=required))
    if failures:
        return failures
    if frame.empty:
        # Empty is valid when the shadow controller did not propose any raise.
        return failures
    failures.extend(
        _audit_numeric_finite(
            frame,
            name=name,
            columns=[
                "suppressed_candidates",
                "raised_schedule_count",
                "mean_threshold_delta",
                "suppressed_loss_avoided",
                "suppressed_winner_pnl_sacrificed",
                "realized_defensive_success",
                "suppressed_full_sl_rate",
                "suppressed_timeout_rate",
            ],
        )
    )
    failures.extend(
        _audit_ratio_bounds(
            frame,
            name=name,
            columns=["suppressed_full_sl_rate", "suppressed_timeout_rate"],
        )
    )
    return failures


def _audit_bundle_replay_summary(frame: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    required = [
        "arm",
        "trade_count",
        "net_pnl",
        "gross_pnl",
        "cost_pnl",
        "full_sl_rate",
        "timeout_rate",
        "mean_threshold_delta",
        "max_threshold_delta",
        "share_threshold_raised",
    ]
    failures.extend(_audit_required_columns(frame, name="controller_replay_summary", required=required))
    failures.extend(_audit_nonempty(frame, name="controller_replay_summary"))
    if not frame.empty and not [column for column in required if column not in frame.columns]:
        failures.extend(
            _audit_numeric_finite(
                frame,
                name="controller_replay_summary",
                columns=[
                    "trade_count",
                    "net_pnl",
                    "gross_pnl",
                    "cost_pnl",
                    "full_sl_rate",
                    "timeout_rate",
                    "mean_threshold_delta",
                    "max_threshold_delta",
                    "share_threshold_raised",
                ],
            )
        )
        failures.extend(
            _audit_ratio_bounds(
                frame,
                name="controller_replay_summary",
                columns=["full_sl_rate", "timeout_rate", "share_threshold_raised"],
            )
        )
    return failures


def _audit_bundle_accepted_trades(frame: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    required = DECISION_KEY_COLS
    missing = [column for column in required if column not in frame.columns]
    if missing:
        failures.append(f"accepted_trades missing decision key columns: {missing}")
        return failures
    keys = frame.loc[:, required].copy()
    keys["timestamp"] = pd.to_datetime(keys["timestamp"], utc=True, errors="coerce")
    if bool(keys["timestamp"].isna().any()):
        failures.append("accepted_trades contains non-finite decision timestamps")
    for column in ("symbol", "side", "strategy_id"):
        keys[column] = keys[column].astype(str)
    duplicates = int(keys.duplicated(required).sum())
    if duplicates:
        failures.append(f"accepted_trades duplicate decision keys: {duplicates}")
    return failures


def _audit_bundle_timestamp_panel(frame: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    required = ["split", "state_level", "timestamp"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        failures.append(f"market_state_timestamp_panel missing columns: {missing}")
        return failures
    if frame.empty:
        failures.append("market_state_timestamp_panel is empty")
        return failures
    duplicate_keys = required
    duplicates = _duplicate_count(frame, duplicate_keys)
    if duplicates > 0:
        failures.append(f"market_state_timestamp_panel duplicate rows for {duplicate_keys}: {duplicates}")
    forbidden = _forbidden_columns(frame)
    if forbidden:
        failures.append(f"market_state_timestamp_panel contains forbidden market-state columns: {forbidden}")
    unsafe = _unsafe_order_book_feature_names(list(map(str, frame.columns)))
    if unsafe:
        failures.append(f"market_state_timestamp_panel contains order-book-like columns: {unsafe[:10]}")
    unsafe_semantic = _unsafe_market_state_semantic_feature_names(list(map(str, frame.columns)))
    if unsafe_semantic:
        failures.append(
            "market_state_timestamp_panel contains strategy/model/performance-like columns: "
            f"{unsafe_semantic[:10]}"
        )
    value_cols = _state_value_columns(frame)
    if not value_cols:
        failures.append("market_state_timestamp_panel has no state/forecast value columns")
    return failures


def _audit_static_baseline_parity(summary: pd.DataFrame, overlap: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    baseline_arm = "S0_baseline_static_thresholds"
    if "arm" not in summary.columns:
        failures.append("portfolio_replay_summary missing arm column")
        return failures
    baseline = summary.loc[summary["arm"].astype(str).eq(baseline_arm)].copy()
    if baseline.empty:
        failures.append("portfolio_replay_summary missing S0_baseline_static_thresholds rows")
    else:
        for column in ("mean_threshold_delta", "p75_threshold_delta", "max_threshold_delta", "share_threshold_raised"):
            if column not in baseline.columns:
                failures.append(f"portfolio_replay_summary missing {column} for static baseline parity")
                continue
            values = pd.to_numeric(baseline[column], errors="coerce")
            if bool(values.isna().any()):
                failures.append(f"portfolio_replay_summary.{column} has non-finite static baseline values")
            elif bool((values.abs() > 1e-12).any()):
                failures.append(f"portfolio_replay_summary.{column} is nonzero for static baseline")

    if "arm" not in overlap.columns:
        failures.append("walkforward_overlap missing arm column")
        return failures
    base_overlap = overlap.loc[overlap["arm"].astype(str).eq(baseline_arm)].copy()
    if base_overlap.empty:
        failures.append("walkforward_overlap missing S0_baseline_static_thresholds rows")
        return failures
    expected_zero = ["new_vs_baseline", "removed_vs_baseline", "entrant_net_pnl", "removed_net_pnl", "defensive_success"]
    for column in expected_zero:
        if column not in base_overlap.columns:
            failures.append(f"walkforward_overlap missing {column} for static baseline parity")
            continue
        values = pd.to_numeric(base_overlap[column], errors="coerce")
        if bool(values.isna().any()):
            failures.append(f"walkforward_overlap.{column} has non-finite static baseline values")
        elif bool((values.abs() > 1e-12).any()):
            failures.append(f"walkforward_overlap.{column} is nonzero for static baseline")
    if {"accepted", "overlap_with_baseline"}.issubset(base_overlap.columns):
        accepted = pd.to_numeric(base_overlap["accepted"], errors="coerce")
        overlap_count = pd.to_numeric(base_overlap["overlap_with_baseline"], errors="coerce")
        if bool(accepted.isna().any() or overlap_count.isna().any()):
            failures.append("walkforward_overlap accepted/overlap counts are non-finite for static baseline")
        elif bool(((accepted - overlap_count).abs() > 1e-12).any()):
            failures.append("walkforward_overlap static baseline accepted != overlap_with_baseline")
    else:
        failures.append("walkforward_overlap missing accepted/overlap_with_baseline for static baseline parity")
    if "jaccard_vs_baseline" in base_overlap.columns:
        jaccard = pd.to_numeric(base_overlap["jaccard_vs_baseline"], errors="coerce")
        if bool(jaccard.isna().any()):
            failures.append("walkforward_overlap.jaccard_vs_baseline is non-finite for static baseline")
        elif bool(((jaccard - 1.0).abs() > 1e-12).any()):
            failures.append("walkforward_overlap.jaccard_vs_baseline != 1 for static baseline")
    else:
        failures.append("walkforward_overlap missing jaccard_vs_baseline for static baseline parity")
    return failures


def _audit_accepted_trades(frame: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    required = ["fold", "arm", *DECISION_KEY_COLS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        failures.append(f"accepted_trades missing decision key columns: {missing}")
        return failures
    keys = frame.loc[:, required].copy()
    keys["timestamp"] = pd.to_datetime(keys["timestamp"], utc=True, errors="coerce")
    if bool(keys["timestamp"].isna().any()):
        failures.append("accepted_trades contains non-finite decision timestamps")
    for column in ("arm", "symbol", "side", "strategy_id"):
        keys[column] = keys[column].astype(str)
    duplicates = int(keys.duplicated(required).sum())
    if duplicates:
        failures.append(f"accepted_trades duplicate decision keys by fold/arm: {duplicates}")
    return failures


def _audit_artifact_hashes(payload: dict[str, Any], *, artifact_dir: Path) -> list[str]:
    failures: list[str] = []
    if payload.get("hash_version") != "sha256_artifact_hashes_v1":
        failures.append("artifact_hashes.hash_version is not sha256_artifact_hashes_v1")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        failures.append("artifact_hashes.artifacts is missing or empty")
        return failures
    missing_hash = [
        name
        for name, item in artifacts.items()
        if not isinstance(item, dict) or item.get("exists") is not True or not item.get("sha256")
    ]
    if missing_hash:
        failures.append(f"artifact_hashes contains incomplete entries: {missing_hash[:10]}")
        return failures

    hashed_paths: set[Path] = set()
    mismatches: list[str] = []
    missing_files: list[str] = []
    for name, item in artifacts.items():
        path = _resolve_artifact_path(item.get("path"), artifact_dir)
        if path is None or not path.exists():
            missing_files.append(str(name))
            continue
        hashed_paths.add(path.resolve())
        expected_bytes = item.get("bytes")
        if expected_bytes is not None and int(expected_bytes) != path.stat().st_size:
            mismatches.append(f"{name}:bytes")
            continue
        expected_sha = str(item.get("sha256"))
        actual_sha = _sha256_file(path)
        if actual_sha != expected_sha:
            mismatches.append(f"{name}:sha256")
    if missing_files:
        failures.append(f"artifact_hashes references missing files: {missing_files[:10]}")
    if mismatches:
        failures.append(f"artifact_hashes mismatched files: {mismatches[:10]}")
    hash_excluded = {"artifact_hashes.json", "manifest.json"}
    required_paths = {(artifact_dir / name).resolve() for name in REQUIRED_ARTIFACTS if name not in hash_excluded}
    missing_required_hashes = sorted(
        str(path.relative_to(artifact_dir.resolve())) if path.is_relative_to(artifact_dir.resolve()) else str(path)
        for path in required_paths
        if path.exists() and path not in hashed_paths
    )
    if missing_required_hashes:
        failures.append(f"artifact_hashes missing required artifact coverage: {missing_required_hashes[:10]}")

    model_paths = {(artifact_dir / name).resolve() for name in MODEL_ARTIFACT_OPTIONS if (artifact_dir / name).exists()}
    if model_paths and not bool(model_paths.intersection(hashed_paths)):
        failures.append("artifact_hashes missing model artifact coverage")
    optional_missing_hashes = sorted(
        str(path.relative_to(artifact_dir.resolve())) if path.is_relative_to(artifact_dir.resolve()) else str(path)
        for path in ((artifact_dir / name).resolve() for name in OPTIONAL_HASHED_ARTIFACTS)
        if path.exists() and path not in hashed_paths
    )
    if optional_missing_hashes:
        failures.append(f"artifact_hashes missing optional artifact coverage: {optional_missing_hashes[:10]}")
    return failures


def _audit_manifest_output_hashes(manifest: dict[str, Any], *, artifact_dir: Path) -> list[str]:
    failures: list[str] = []
    outputs = manifest.get("outputs")
    hashes = manifest.get("output_sha256")
    if not isinstance(outputs, dict) or not outputs:
        failures.append("manifest.outputs is missing or empty")
        return failures
    if not isinstance(hashes, dict) or not hashes:
        failures.append("manifest.output_sha256 is missing or empty")
        return failures
    missing_hashes: list[str] = []
    mismatches: list[str] = []
    for name, raw_path in sorted(outputs.items()):
        path = _resolve_artifact_path(raw_path, artifact_dir)
        if path is None or not path.exists() or not path.is_file():
            failures.append(f"manifest.outputs.{name} references missing file")
            continue
        expected = hashes.get(name)
        if not expected:
            missing_hashes.append(str(name))
            continue
        actual = _sha256_file(path)
        if str(expected) != actual:
            mismatches.append(str(name))
    if missing_hashes:
        failures.append(f"manifest.output_sha256 missing output hashes: {missing_hashes[:10]}")
    if mismatches:
        failures.append(f"manifest.output_sha256 mismatched outputs: {mismatches[:10]}")
    return failures


def _audit_bundle_feature_contract(
    payload: dict[str, Any],
    *,
    expected_rank_contract: str = EXPECTED_RANK_CONTRACT,
) -> list[str]:
    failures: list[str] = []
    if payload.get("contract_version") != "market_state_feature_contract_v1":
        failures.append("market_state_feature_contract.contract_version is unexpected")
    if payload.get("rank_contract") != expected_rank_contract:
        failures.append(f"market_state_feature_contract.rank_contract != {expected_rank_contract}")
    if payload.get("active_heads") != EXPECTED_ACTIVE_HEADS:
        failures.append(f"market_state_feature_contract.active_heads != {EXPECTED_ACTIVE_HEADS}")
    if sorted(payload.get("disabled_heads") or []) != EXPECTED_DISABLED_HEADS:
        failures.append(f"market_state_feature_contract.disabled_heads != {EXPECTED_DISABLED_HEADS}")
    invariants = payload.get("invariants")
    if not isinstance(invariants, dict):
        failures.append("market_state_feature_contract.invariants is missing")
    else:
        for key in FEATURE_CONTRACT_TRUE_INVARIANTS:
            if invariants.get(key) is not True:
                failures.append(f"market_state_feature_contract.invariants.{key} is not true")
        for key in FEATURE_CONTRACT_FALSE_INVARIANTS:
            if invariants.get(key) is not False:
                failures.append(f"market_state_feature_contract.invariants.{key} is not false")
    source_schema = payload.get("source_schema")
    if not isinstance(source_schema, dict):
        failures.append("market_state_feature_contract.source_schema is missing")
    else:
        unsafe_features: list[str] = []
        for key in SOURCE_SCHEMA_FEATURE_LIST_KEYS:
            values = source_schema.get(key)
            if values is None:
                continue
            if not isinstance(values, list):
                failures.append(f"market_state_feature_contract.source_schema.{key} is not a list")
                continue
            unsafe_features.extend(_unsafe_order_book_feature_names([str(value) for value in values]))
            unsafe_features.extend(
                [str(value) for value in values if str(value) in FORBIDDEN_MARKET_STATE_COLUMNS]
            )
        if unsafe_features:
            failures.append(
                "market_state_feature_contract.source_schema contains unsafe feature names: "
                f"{sorted(set(unsafe_features))[:10]}"
            )
    failures.extend(
        _audit_source_contract_audit(
            payload.get("source_contract_audit"),
            prefix="market_state_feature_contract.source_contract_audit",
        )
    )
    join_validation = payload.get("state_join_validation")
    if not isinstance(join_validation, dict) or not join_validation:
        failures.append("market_state_feature_contract.state_join_validation is missing")
    else:
        for split, item in join_validation.items():
            if not isinstance(item, dict):
                failures.append(f"market_state_feature_contract.state_join_validation.{split} is not an object")
                continue
            if item.get("state_join_timestamp_constant") is not True:
                failures.append(
                    f"market_state_feature_contract.state_join_validation.{split}.state_join_timestamp_constant is not true"
                )
    return failures


def _audit_bundle_universe_contract(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if payload.get("contract_version") != "market_state_universe_contract_v1":
        failures.append("market_state_universe_contract.contract_version is not market_state_universe_contract_v1")
    if payload.get("required_source") != "feature_store_market_aggregates":
        failures.append("market_state_universe_contract.required_source is not feature_store_market_aggregates")
    if payload.get("strategy_independent") is not True:
        failures.append("market_state_universe_contract.strategy_independent is not true")
    if payload.get("candidate_independent") is not True:
        failures.append("market_state_universe_contract.candidate_independent is not true")
    if payload.get("actual_order_book_features_allowed") is not False:
        failures.append("market_state_universe_contract.actual_order_book_features_allowed is not false")
    if payload.get("candidate_population_fallback_enabled") is not False:
        failures.append("market_state_universe_contract.candidate_population_fallback_enabled is not false")
    validation = payload.get("validation")
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        failures.append("market_state_universe_contract.validation.passed is not true")
    eligible_symbols = payload.get("eligible_symbols")
    if not isinstance(eligible_symbols, list) or not eligible_symbols:
        failures.append("market_state_universe_contract.eligible_symbols is missing")
    return failures


def _audit_feature_contract(
    payload: dict[str, Any],
    *,
    expected_rank_contract: str = EXPECTED_RANK_CONTRACT,
) -> list[str]:
    failures: list[str] = []
    if payload.get("rank_contract") != expected_rank_contract:
        failures.append(f"market_state_feature_contract.rank_contract != {expected_rank_contract}")
    if payload.get("active_heads") != EXPECTED_ACTIVE_HEADS:
        failures.append(f"market_state_feature_contract.active_heads != {EXPECTED_ACTIVE_HEADS}")
    if sorted(payload.get("disabled_heads") or []) != EXPECTED_DISABLED_HEADS:
        failures.append(f"market_state_feature_contract.disabled_heads != {EXPECTED_DISABLED_HEADS}")
    invariants = payload.get("invariants")
    if not isinstance(invariants, dict):
        failures.append("market_state_feature_contract.invariants is missing")
    else:
        for key in FEATURE_CONTRACT_TRUE_INVARIANTS:
            if invariants.get(key) is not True:
                failures.append(f"market_state_feature_contract.invariants.{key} is not true")
        for key in FEATURE_CONTRACT_FALSE_INVARIANTS:
            if invariants.get(key) is not False:
                failures.append(f"market_state_feature_contract.invariants.{key} is not false")
        if invariants.get("controller_changes_scores_or_ranks") is not False:
            failures.append("market_state_feature_contract allows score/rank changes")
        if invariants.get("controller_changes_auction_ordering") is not False:
            failures.append("market_state_feature_contract allows auction-order changes")
    validation = payload.get("validation")
    if not isinstance(validation, dict):
        failures.append("market_state_feature_contract.validation is missing")
    else:
        if validation.get("passed") is not True:
            failures.append("market_state_feature_contract.validation.passed is not true")
        if validation.get("failures") not in ([], None):
            failures.append("market_state_feature_contract.validation.failures is not empty")
        if int(validation.get("fold_count") or 0) <= 0:
            failures.append("market_state_feature_contract.validation.fold_count <= 0")
        if int(validation.get("state_head_registry_rows") or 0) <= 0:
            failures.append("market_state_feature_contract.validation.state_head_registry_rows <= 0")
        if validation.get("training_outcome_maturity_contract_passed") is not True:
            failures.append(
                "market_state_feature_contract.validation.training_outcome_maturity_contract_passed is not true"
            )
        maturity_failures = validation.get("training_outcome_maturity_failures")
        if maturity_failures not in ([], None):
            failures.append(
                "market_state_feature_contract.validation.training_outcome_maturity_failures is not empty"
            )
        if "training_immature_outcome_rows_dropped" not in validation:
            failures.append(
                "market_state_feature_contract.validation.training_immature_outcome_rows_dropped is missing"
            )
    fold_definition = payload.get("fold_definition")
    if not isinstance(fold_definition, dict):
        failures.append("market_state_feature_contract.fold_definition is missing")
    else:
        failures.extend(_audit_fold_definition(fold_definition, prefix="market_state_feature_contract.fold_definition"))
    source_schema = payload.get("source_schema")
    if not isinstance(source_schema, dict):
        failures.append("market_state_feature_contract.source_schema is missing")
    else:
        unsafe_features: list[str] = []
        for key in SOURCE_SCHEMA_FEATURE_LIST_KEYS:
            values = source_schema.get(key)
            if values is None:
                continue
            if not isinstance(values, list):
                failures.append(f"market_state_feature_contract.source_schema.{key} is not a list")
                continue
            unsafe_features.extend(_unsafe_order_book_feature_names([str(value) for value in values]))
            for value in values:
                feature = str(value)
                if feature in FORBIDDEN_MARKET_STATE_COLUMNS:
                    unsafe_features.append(feature)
        if unsafe_features:
            failures.append(
                "market_state_feature_contract.source_schema contains unsafe feature names: "
                f"{sorted(set(unsafe_features))[:10]}"
            )
        activation_filter = payload.get("state_activation_filter")
        if isinstance(activation_filter, dict) and activation_filter.get("enforced") is True:
            activation_reason = str(activation_filter.get("reason") or "")
            active_cols = set(map(str, activation_filter.get("active_state_feature_columns") or []))
            dropped_cols = set(map(str, activation_filter.get("dropped_state_feature_columns") or []))
            state_cols = set(map(str, source_schema.get("state_feature_columns") or []))
            response_cols = set(map(str, source_schema.get("response_feature_columns") or []))
            allowed_empty_reasons = {
                "selected_controller_rejected_noop",
                "activation_registry_unavailable_fail_closed",
                "activation_registry_active_candidate_filter",
            }
            if not active_cols and activation_reason not in allowed_empty_reasons:
                failures.append(
                    "market_state_feature_contract.state_activation_filter is enforced with no active state features"
                )
            outside_active = sorted(state_cols.difference(active_cols))
            if outside_active:
                failures.append(
                    "market_state_feature_contract.source_schema.state_feature_columns outside "
                    f"activation filter: {outside_active[:10]}"
                )
            leaked_dropped = sorted(dropped_cols.intersection(state_cols.union(response_cols)))
            if leaked_dropped:
                failures.append(
                    "market_state_feature_contract.source_schema contains dropped activation-registry "
                    f"state features: {leaked_dropped[:10]}"
                )
        elif activation_filter is not None and not isinstance(activation_filter, dict):
            failures.append("market_state_feature_contract.state_activation_filter is not an object")
    failures.extend(
        _audit_source_contract_audit(
            payload.get("source_contract_audit"),
            prefix="market_state_feature_contract.source_contract_audit",
        )
    )
    universe_contract = payload.get("universe_contract")
    if not isinstance(universe_contract, dict):
        failures.append("market_state_feature_contract.universe_contract is missing")
    else:
        if universe_contract.get("contract_version") != "market_state_universe_contract_v1":
            failures.append("market_state_feature_contract.universe_contract has unexpected version")
        validation = universe_contract.get("validation")
        if not isinstance(validation, dict) or validation.get("passed") is not True:
            failures.append("market_state_feature_contract.universe_contract.validation.passed is not true")
    return failures


def _audit_universe_contract(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if payload.get("contract_version") != "market_state_universe_contract_v1":
        failures.append("market_state_universe_contract.contract_version is not market_state_universe_contract_v1")
    if payload.get("required_source") != "feature_store_market_aggregates":
        failures.append("market_state_universe_contract.required_source is not feature_store_market_aggregates")
    if payload.get("strategy_independent") is not True:
        failures.append("market_state_universe_contract.strategy_independent is not true")
    if payload.get("candidate_independent") is not True:
        failures.append("market_state_universe_contract.candidate_independent is not true")
    if payload.get("actual_order_book_features_allowed") is not False:
        failures.append("market_state_universe_contract.actual_order_book_features_allowed is not false")
    if payload.get("candidate_population_fallback_enabled") is not False:
        failures.append("market_state_universe_contract.candidate_population_fallback_enabled is not false")
    validation = payload.get("validation")
    if not isinstance(validation, dict):
        failures.append("market_state_universe_contract.validation is missing")
    else:
        if validation.get("passed") is not True:
            failures.append("market_state_universe_contract.validation.passed is not true")
        if validation.get("failures") not in ([], None):
            failures.append("market_state_universe_contract.validation.failures is not empty")
        if int(validation.get("fold_split_count") or 0) <= 0:
            failures.append("market_state_universe_contract.validation.fold_split_count <= 0")
        if validation.get("eligible_symbol_list_constant") is not True:
            failures.append("market_state_universe_contract.validation.eligible_symbol_list_constant is not true")
    eligible_symbols = payload.get("eligible_symbols")
    if not isinstance(eligible_symbols, list) or not eligible_symbols:
        failures.append("market_state_universe_contract.eligible_symbols is missing")
    else:
        symbols = [str(symbol) for symbol in eligible_symbols]
        if len(symbols) != len(set(symbols)):
            failures.append("market_state_universe_contract.eligible_symbols contains duplicates")
        if int(payload.get("eligible_symbol_count") or 0) != len(symbols):
            failures.append("market_state_universe_contract.eligible_symbol_count mismatch")
    if not payload.get("minimum_history"):
        failures.append("market_state_universe_contract.minimum_history is missing")
    if not payload.get("minimum_volume"):
        failures.append("market_state_universe_contract.minimum_volume is missing")
    if not payload.get("oi_coverage_requirements"):
        failures.append("market_state_universe_contract.oi_coverage_requirements is missing")
    if not payload.get("funding_coverage_requirements"):
        failures.append("market_state_universe_contract.funding_coverage_requirements is missing")
    if not isinstance(payload.get("excluded_symbols_and_reasons"), dict):
        failures.append("market_state_universe_contract.excluded_symbols_and_reasons is missing")
    splits = payload.get("fold_split_contracts")
    if not isinstance(splits, dict) or not splits:
        failures.append("market_state_universe_contract.fold_split_contracts is missing")
        return failures
    for key, row_raw in splits.items():
        row = dict(row_raw or {})
        prefix = f"market_state_universe_contract.fold_split_contracts.{key}"
        if row.get("source") != "feature_store_market_aggregates":
            failures.append(f"{prefix}.source is not feature_store_market_aggregates")
        if row.get("production_safe") is not True:
            failures.append(f"{prefix}.production_safe is not true")
        if row.get("candidate_fallback_enabled") is not False:
            failures.append(f"{prefix}.candidate_fallback_enabled is not false")
        if row.get("strategy_independent") is not True:
            failures.append(f"{prefix}.strategy_independent is not true")
        if row.get("candidate_independent") is not True:
            failures.append(f"{prefix}.candidate_independent is not true")
        if row.get("universe_definition_version") != "feature_store_timestamp_market_state_v1":
            failures.append(f"{prefix}.universe_definition_version is unexpected")
        row_symbols = row.get("eligible_symbols")
        if not isinstance(row_symbols, list) or not row_symbols:
            failures.append(f"{prefix}.eligible_symbols is missing")
            continue
        if len(row_symbols) != len(set(map(str, row_symbols))):
            failures.append(f"{prefix}.eligible_symbols contains duplicates")
        if int(row.get("eligible_symbol_count") or 0) != len(row_symbols):
            failures.append(f"{prefix}.eligible_symbol_count mismatch")
        if int(row.get("available_symbol_count") or 0) < len(row_symbols):
            failures.append(f"{prefix}.available_symbol_count < eligible_symbol_count")
        excluded_symbols = [str(symbol) for symbol in row.get("excluded_symbols") or []]
        excluded_reasons = dict(row.get("excluded_symbols_and_reasons") or {})
        if any(symbol not in excluded_reasons for symbol in excluded_symbols):
            failures.append(f"{prefix}.excluded_symbols missing reasons")
        for required in (
            "minimum_history",
            "minimum_volume",
            "oi_coverage_requirements",
            "funding_coverage_requirements",
            "selection_reason",
        ):
            if not row.get(required):
                failures.append(f"{prefix}.{required} is missing")
    return failures


def _audit_fold_definition(payload: dict[str, Any], *, prefix: str) -> list[str]:
    failures: list[str] = []
    folds = payload.get("folds_built")
    if not isinstance(folds, list) or not folds:
        failures.append(f"{prefix}.folds_built is missing")
        return failures
    try:
        embargo_hours = float(payload.get("embargo_hours"))
    except (TypeError, ValueError):
        embargo_hours = float("nan")
    if not pd.notna(embargo_hours) or embargo_hours <= 0:
        failures.append(f"{prefix}.embargo_hours <= 0 or missing")
        embargo_hours = 0.0
    min_valid_rows = int(payload.get("min_valid_rows") or 0)
    min_valid_timestamps = int(payload.get("min_valid_timestamps") or 0)

    seen_folds: set[int] = set()
    previous_valid_start = None
    for item in folds:
        if not isinstance(item, dict):
            failures.append(f"{prefix}.folds_built contains non-object fold")
            continue
        missing = [
            key
            for key in ("fold", "train_start", "train_end", "valid_start", "valid_end")
            if key not in item
        ]
        if missing:
            failures.append(f"{prefix}.fold_{item.get('fold', '?')} missing keys: {missing}")
            continue
        fold_id = int(item.get("fold"))
        if fold_id in seen_folds:
            failures.append(f"{prefix}.fold {fold_id} is duplicated")
        seen_folds.add(fold_id)
        train_start = pd.to_datetime(item.get("train_start"), utc=True, errors="coerce")
        train_end = pd.to_datetime(item.get("train_end"), utc=True, errors="coerce")
        valid_start = pd.to_datetime(item.get("valid_start"), utc=True, errors="coerce")
        valid_end = pd.to_datetime(item.get("valid_end"), utc=True, errors="coerce")
        if any(pd.isna(value) for value in (train_start, train_end, valid_start, valid_end)):
            failures.append(f"{prefix}.fold {fold_id} has non-parseable timestamps")
            continue
        if train_start > train_end:
            failures.append(f"{prefix}.fold {fold_id} train_start > train_end")
        if train_end >= valid_start:
            failures.append(f"{prefix}.fold {fold_id} train_end >= valid_start")
        if valid_start > valid_end:
            failures.append(f"{prefix}.fold {fold_id} valid_start > valid_end")
        actual_embargo_hours = (valid_start - train_end).total_seconds() / 3600.0
        if actual_embargo_hours + 1e-12 < embargo_hours:
            failures.append(
                f"{prefix}.fold {fold_id} embargo {actual_embargo_hours:.6g}h < required {embargo_hours:.6g}h"
            )
        valid_rows = int(item.get("valid_rows_available") or 0)
        if min_valid_rows > 0 and valid_rows < min_valid_rows:
            failures.append(f"{prefix}.fold {fold_id} valid_rows_available < min_valid_rows")
        valid_timestamps = int(item.get("valid_timestamps_available") or 0)
        if min_valid_timestamps > 0 and valid_timestamps < min_valid_timestamps:
            failures.append(f"{prefix}.fold {fold_id} valid_timestamps_available < min_valid_timestamps")
        if previous_valid_start is not None and valid_start <= previous_valid_start:
            failures.append(f"{prefix}.fold {fold_id} valid_start is not strictly increasing")
        previous_valid_start = valid_start
    return failures


def _audit_controller_config(
    payload: dict[str, Any],
    *,
    expected_rank_contract: str = EXPECTED_RANK_CONTRACT,
) -> list[str]:
    failures: list[str] = []
    baseline = payload.get("baseline_contract")
    if not isinstance(baseline, dict):
        failures.append("strategy_threshold_controller_config.baseline_contract is missing")
    else:
        if baseline.get("rank_contract") != expected_rank_contract:
            failures.append(f"strategy_threshold_controller_config.rank_contract != {expected_rank_contract}")
        if baseline.get("active_heads") != EXPECTED_ACTIVE_HEADS:
            failures.append(f"strategy_threshold_controller_config.active_heads != {EXPECTED_ACTIVE_HEADS}")
        if sorted(baseline.get("disabled_heads") or []) != EXPECTED_DISABLED_HEADS:
            failures.append(f"strategy_threshold_controller_config.disabled_heads != {EXPECTED_DISABLED_HEADS}")
        if baseline.get("q_fail_enabled") is not False:
            failures.append("strategy_threshold_controller_config q_fail_enabled is not false")
        if baseline.get("changes_scores_or_ranks") is not False:
            failures.append("strategy_threshold_controller_config allows score/rank changes")
        if baseline.get("changes_auction_ordering") is not False:
            failures.append("strategy_threshold_controller_config allows auction-order changes")
    controller = payload.get("controller")
    if not isinstance(controller, dict):
        failures.append("strategy_threshold_controller_config.controller is missing")
    else:
        if controller.get("penalty_only") is not True:
            failures.append("strategy_threshold_controller_config.controller.penalty_only is not true")
    validation = payload.get("validation")
    if not isinstance(validation, dict):
        failures.append("strategy_threshold_controller_config.validation is missing")
    else:
        if validation.get("chronological_complete_timestamp_folds") is not True:
            failures.append("strategy_threshold_controller_config.validation.chronological_complete_timestamp_folds is not true")
        if int(validation.get("embargo_hours") or 0) <= 0:
            failures.append("strategy_threshold_controller_config.validation.embargo_hours <= 0")
        no_backfill_overlay_selection = (
            isinstance(controller, dict)
            and controller.get("select_no_backfill_overlay_only") is True
            and controller.get("include_post_selection_overlay_arms") is True
            and isinstance(controller.get("post_selection_overlay_contract"), str)
            and "no freed-capacity backfill" in controller.get("post_selection_overlay_contract", "")
        )
        if (
            validation.get("selected_controller_is_null") is not True
            and not no_backfill_overlay_selection
        ):
            failures.append("strategy_threshold_controller_config.validation.selected_controller_is_null is not true")
    return failures


def _expected_fold_keys(feature_contract: dict[str, Any]) -> tuple[set[str], set[int]]:
    fold_definition = feature_contract.get("fold_definition")
    if not isinstance(fold_definition, dict):
        return set(), set()
    folds = fold_definition.get("folds_built")
    if not isinstance(folds, list):
        return set(), set()
    fold_ids = {
        int(item.get("fold"))
        for item in folds
        if isinstance(item, dict) and item.get("fold") is not None
    }
    return {f"fold_{fold_id}" for fold_id in fold_ids}, fold_ids


def _load_joblib_dict(path: Path, *, name: str) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        payload = joblib.load(path)
    except Exception as exc:  # pragma: no cover - defensive path gives actionable CLI failures.
        return None, [f"{name} failed to load as joblib: {exc}"]
    if not isinstance(payload, dict):
        return None, [f"{name} is not a dict artifact"]
    return payload, []


def _audit_training_reference(payload: dict[str, Any], expected_fold_keys: set[str]) -> list[str]:
    failures: list[str] = []
    if payload.get("generated_by") != "run_market_state_threshold_controller_walkforward":
        failures.append("market_state_training_reference.generated_by is unexpected")
    if payload.get("reference_version") != "market_state_training_reference_bundle_v1":
        failures.append("market_state_training_reference.reference_version is unexpected")
    folds = payload.get("fold_references")
    if not isinstance(folds, dict) or not folds:
        failures.append("market_state_training_reference.fold_references is missing")
        return failures
    if expected_fold_keys and set(folds) != expected_fold_keys:
        failures.append("market_state_training_reference.fold_references keys do not match fold definition")
    for fold_key, fold_ref in folds.items():
        prefix = f"market_state_training_reference.fold_references.{fold_key}"
        if not isinstance(fold_ref, dict):
            failures.append(f"{prefix} is not an object")
            continue
        encoder = fold_ref.get("observed_axis_encoder")
        if not isinstance(encoder, dict):
            failures.append(f"{prefix}.observed_axis_encoder is missing")
            continue
        if encoder.get("mode") != "observed_axis_robust_z_v1":
            failures.append(f"{prefix}.observed_axis_encoder.mode is unexpected")
        try:
            min_input_coverage = float(encoder.get("minimum_input_coverage"))
        except (TypeError, ValueError):
            min_input_coverage = float("nan")
        if not pd.notna(min_input_coverage) or not (0.0 < min_input_coverage <= 1.0):
            failures.append(f"{prefix}.observed_axis_encoder.minimum_input_coverage is invalid")

        column_refs = encoder.get("column_refs")
        if not isinstance(column_refs, dict) or not column_refs:
            failures.append(f"{prefix}.observed_axis_encoder.column_refs is missing")
        else:
            unsafe = _unsafe_order_book_feature_names([str(col) for col in column_refs])
            if unsafe:
                failures.append(f"{prefix}.observed_axis_encoder.column_refs contains unsafe feature names: {unsafe[:10]}")
            for col, ref in list(column_refs.items()):
                if not isinstance(ref, dict):
                    failures.append(f"{prefix}.observed_axis_encoder.column_refs.{col} is not an object")
                    continue
                for key in ("median", "scale", "q05", "q95"):
                    try:
                        value = float(ref.get(key))
                    except (TypeError, ValueError):
                        value = float("nan")
                    if not pd.notna(value):
                        failures.append(f"{prefix}.observed_axis_encoder.column_refs.{col}.{key} is non-finite")
                        break
                try:
                    scale = float(ref.get("scale"))
                except (TypeError, ValueError):
                    scale = 0.0
                if not pd.notna(scale) or scale <= 0.0:
                    failures.append(f"{prefix}.observed_axis_encoder.column_refs.{col}.scale <= 0")

        axes = encoder.get("axes")
        if not isinstance(axes, dict) or not axes:
            failures.append(f"{prefix}.observed_axis_encoder.axes is missing")
        else:
            for axis, cols in axes.items():
                if not isinstance(cols, list):
                    failures.append(f"{prefix}.observed_axis_encoder.axes.{axis} is not a list")
                    continue
                unsafe = _unsafe_order_book_feature_names([str(col) for col in cols])
                if unsafe:
                    failures.append(f"{prefix}.observed_axis_encoder.axes.{axis} contains unsafe feature names: {unsafe[:10]}")

        axis_sources = encoder.get("axis_sources")
        if not isinstance(axis_sources, dict) or not axis_sources:
            failures.append(f"{prefix}.observed_axis_encoder.axis_sources is missing")
        else:
            for required in (
                "state_input_coverage",
                "state_uncertainty",
                "state_low_input_coverage",
            ):
                if required not in axis_sources:
                    failures.append(f"{prefix}.observed_axis_encoder.axis_sources missing {required}")
        source_validation = encoder.get("source_validation")
        if not isinstance(source_validation, dict):
            failures.append(f"{prefix}.observed_axis_encoder.source_validation is missing")
        elif "train" not in source_validation:
            failures.append(f"{prefix}.observed_axis_encoder.source_validation.train is missing")

        feature_store_columns = [str(col) for col in list(fold_ref.get("feature_store_columns") or [])]
        feature_store_reports = fold_ref.get("feature_store_reports")
        if not isinstance(feature_store_reports, dict):
            failures.append(f"{prefix}.feature_store_reports is missing")
            continue
        train_fs = feature_store_reports.get("train")
        if not isinstance(train_fs, dict):
            failures.append(f"{prefix}.feature_store_reports.train is missing")
            continue
        valid_fs = (
            feature_store_reports.get("valid")
            if "valid" in feature_store_reports
            else feature_store_reports.get("eval")
        )
        if feature_store_columns:
            unsafe = _unsafe_order_book_feature_names(feature_store_columns)
            if unsafe:
                failures.append(f"{prefix}.feature_store_columns contains unsafe feature names: {unsafe[:10]}")
            if train_fs.get("tail_reference_source") != "self_window_reference":
                failures.append(f"{prefix}.feature_store_reports.train.tail_reference_source is not self_window_reference")
            if train_fs.get("tail_reference_role") != "fit_on_training_timestamps":
                failures.append(f"{prefix}.feature_store_reports.train.tail_reference_role is not fit_on_training_timestamps")
            refs = fold_ref.get("feature_store_tail_reference_quantiles")
            if refs is None:
                refs = train_fs.get("tail_reference_quantiles")
            if not isinstance(refs, dict) or not refs:
                failures.append(f"{prefix}.feature_store_tail_reference_quantiles is missing")
            else:
                unsafe_ref_names = _unsafe_order_book_feature_names([str(col) for col in refs])
                if unsafe_ref_names:
                    failures.append(
                        f"{prefix}.feature_store_tail_reference_quantiles contains unsafe feature names: "
                        f"{unsafe_ref_names[:10]}"
                    )
                for col, ref_raw in list(refs.items()):
                    if not isinstance(ref_raw, dict):
                        failures.append(f"{prefix}.feature_store_tail_reference_quantiles.{col} is not an object")
                        continue
                    for key in ("q10", "q90"):
                        try:
                            value = float(ref_raw.get(key))
                        except (TypeError, ValueError):
                            value = float("nan")
                        if not pd.notna(value):
                            failures.append(
                                f"{prefix}.feature_store_tail_reference_quantiles.{col}.{key} is non-finite"
                            )
                            break
            if isinstance(valid_fs, dict):
                if valid_fs.get("tail_reference_source") != "provided_train_reference":
                    failures.append(
                        f"{prefix}.feature_store_reports.valid.tail_reference_source is not provided_train_reference"
                    )
                valid_role = valid_fs.get("tail_reference_role")
                if valid_role not in {
                    "transformed_with_training_timestamp_reference",
                    "transformed_with_bundle_training_reference",
                }:
                    failures.append(
                        f"{prefix}.feature_store_reports.valid.tail_reference_role is not training-reference transform"
                    )
    return failures


def _audit_target_definitions(payload: dict[str, Any], expected_fold_ids: set[int]) -> list[str]:
    failures: list[str] = []
    if payload.get("contract_version") != "market_state_target_definitions_v1":
        failures.append("market_state_target_definitions.contract_version is unexpected")
    if payload.get("target_type") != "training_cdf_normalized_future_market_geometry_soft_severity":
        failures.append("market_state_target_definitions.target_type is unexpected")
    targets = payload.get("forecast_targets")
    if not isinstance(targets, dict) or not targets:
        failures.append("market_state_target_definitions.forecast_targets is missing")
        return failures
    for target, report in list(targets.items()):
        if not isinstance(report, dict):
            failures.append(f"market_state_target_definitions.{target} is not an object")
            continue
        if int(report.get("fold_count") or 0) <= 0:
            failures.append(f"market_state_target_definitions.{target}.fold_count <= 0")
        folds = report.get("folds")
        if not isinstance(folds, list) or not folds:
            failures.append(f"market_state_target_definitions.{target}.folds is missing")
            continue
        actual_fold_ids = {
            int(item.get("fold"))
            for item in folds
            if isinstance(item, dict) and item.get("fold") is not None
        }
        if expected_fold_ids and actual_fold_ids != expected_fold_ids:
            failures.append(f"market_state_target_definitions.{target}.folds do not match fold definition")
        for item in folds:
            if not isinstance(item, dict):
                failures.append(f"market_state_target_definitions.{target}.fold item is not an object")
                continue
            train_mode = item.get("train_prediction_mode")
            if train_mode == "chronological_expanding_oof_or_fallback":
                if float(item.get("oof_coverage") or 0.0) <= 0.0:
                    failures.append(f"market_state_target_definitions.{target}.oof_coverage <= 0")
            elif train_mode == "bounded_current_axis_fallback":
                if item.get("mode") != "current_axis_fallback":
                    failures.append(f"market_state_target_definitions.{target}.fallback mode is unexpected")
                if not item.get("fallback_axis"):
                    failures.append(f"market_state_target_definitions.{target}.fallback_axis is missing")
            else:
                failures.append(f"market_state_target_definitions.{target}.train_prediction_mode is unexpected")
    return failures


def _audit_target_cdfs(payload: dict[str, Any], expected_fold_keys: set[str]) -> list[str]:
    failures: list[str] = []
    if payload.get("artifact_version") != "market_state_target_cdfs_v1":
        failures.append("market_state_target_cdfs.artifact_version is unexpected")
    if payload.get("normalization") != "training_fold_empirical_cdf_raw_future_market_geometry_targets":
        failures.append("market_state_target_cdfs.normalization is unexpected")
    if int(payload.get("target_count") or 0) <= 0:
        failures.append("market_state_target_cdfs.target_count <= 0")
    if int(payload.get("missing_reference_count") or 0) != 0:
        failures.append("market_state_target_cdfs.missing_reference_count != 0")
    folds = payload.get("folds")
    if not isinstance(folds, dict) or not folds:
        failures.append("market_state_target_cdfs.folds is missing")
    elif expected_fold_keys and set(folds) != expected_fold_keys:
        failures.append("market_state_target_cdfs.folds keys do not match fold definition")
    return failures


def _audit_forecast_model_bundle(payload: dict[str, Any], expected_fold_keys: set[str], *, expected_kind: str) -> list[str]:
    failures: list[str] = []
    if payload.get("generated_by") != "run_market_state_threshold_controller_walkforward":
        failures.append("market_state_forecast_models.generated_by is unexpected")
    if payload.get("artifact_version") != "market_state_forecast_models_v1":
        failures.append("market_state_forecast_models.artifact_version is unexpected")
    if payload.get("forecast_model_kind") != expected_kind:
        failures.append(f"market_state_forecast_models.forecast_model_kind != {expected_kind}")
    folds = payload.get("fold_forecast_artifacts")
    if not isinstance(folds, dict) or not folds:
        failures.append("market_state_forecast_models.fold_forecast_artifacts is missing")
    elif expected_fold_keys and set(folds) != expected_fold_keys:
        failures.append("market_state_forecast_models.fold_forecast_artifacts keys do not match fold definition")
    return failures


def _audit_rank_curve_bundle(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if payload.get("generated_by") != "run_market_state_threshold_controller_walkforward":
        failures.append("strategy_rank_outcome_curves.generated_by is unexpected")
    if payload.get("artifact_version") != "strategy_rank_outcome_curves_v1":
        failures.append("strategy_rank_outcome_curves.artifact_version is unexpected")
    table = payload.get("rank_curve_table")
    if not isinstance(table, pd.DataFrame) or table.empty:
        failures.append("strategy_rank_outcome_curves.rank_curve_table is missing or empty")
    return failures


def _audit_response_model_bundle(payload: dict[str, Any], expected_fold_ids: set[int]) -> list[str]:
    failures: list[str] = []
    if payload.get("generated_by") != "run_market_state_threshold_controller_walkforward":
        failures.append("strategy_response_models.generated_by is unexpected")
    if payload.get("response_model_kind") not in {"additive_ebm", "hist_gradient_boosting", "xgboost"}:
        failures.append("strategy_response_models.response_model_kind is unexpected")
    fold_models = payload.get("fold_models")
    if not isinstance(fold_models, dict) or not fold_models:
        failures.append("strategy_response_models.fold_models is missing")
        return failures
    seen_fold_ids: set[int] = set()
    for name, bundle in fold_models.items():
        if not isinstance(bundle, dict):
            failures.append(f"strategy_response_models.{name} is not an object")
            continue
        fold_id = int(bundle.get("fold") or 0)
        if fold_id <= 0:
            failures.append(f"strategy_response_models.{name}.fold <= 0")
        else:
            seen_fold_ids.add(fold_id)
        if not bundle.get("arm"):
            failures.append(f"strategy_response_models.{name}.arm is missing")
        for key in ("state_columns", "response_feature_columns"):
            values = bundle.get(key)
            if not isinstance(values, list) or not values:
                failures.append(f"strategy_response_models.{name}.{key} is missing")
        if not isinstance(bundle.get("model_report"), dict) or not bundle.get("model_report"):
            failures.append(f"strategy_response_models.{name}.model_report is missing")
        else:
            model_report = dict(bundle.get("model_report") or {})
            if not model_report.get("state_training_input_contract"):
                failures.append(f"strategy_response_models.{name}.state_training_input_contract is missing")
            if model_report.get("response_training_uses_oof_state_scores") is not True:
                failures.append(f"strategy_response_models.{name}.response_training_uses_oof_state_scores is not true")
            if model_report.get("response_training_state_contract_passed") is not True:
                failures.append(f"strategy_response_models.{name}.response_training_state_contract_passed is not true")
            non_oof = model_report.get("learned_state_non_oof_columns")
            if isinstance(non_oof, list) and non_oof:
                failures.append(f"strategy_response_models.{name}.learned_state_non_oof_columns is not empty")
        if not isinstance(bundle.get("models"), dict) or not bundle.get("models"):
            failures.append(f"strategy_response_models.{name}.models is missing")
    if expected_fold_ids and not expected_fold_ids.issubset(seen_fold_ids):
        failures.append("strategy_response_models.fold_models do not cover all fold definitions")
    return failures


def _audit_joblib_bundles(artifact_dir: Path, feature_contract: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    expected_fold_keys, expected_fold_ids = _expected_fold_keys(feature_contract)

    training_reference, load_failures = _load_joblib_dict(
        artifact_dir / "market_state_training_reference.joblib",
        name="market_state_training_reference",
    )
    failures.extend(load_failures)
    if training_reference is not None:
        failures.extend(_audit_training_reference(training_reference, expected_fold_keys))

    target_cdfs, load_failures = _load_joblib_dict(
        artifact_dir / "market_state_target_cdfs.joblib",
        name="market_state_target_cdfs",
    )
    failures.extend(load_failures)
    if target_cdfs is not None:
        failures.extend(_audit_target_cdfs(target_cdfs, expected_fold_keys))

    expected_kind = "xgboost" if (artifact_dir / "market_state_xgb_models.joblib").exists() else "lightgbm"
    model_path = artifact_dir / (
        "market_state_xgb_models.joblib" if expected_kind == "xgboost" else "market_state_lgbm_models.joblib"
    )
    forecast_models, load_failures = _load_joblib_dict(model_path, name="market_state_forecast_models")
    failures.extend(load_failures)
    if forecast_models is not None:
        failures.extend(_audit_forecast_model_bundle(forecast_models, expected_fold_keys, expected_kind=expected_kind))

    rank_curves, load_failures = _load_joblib_dict(
        artifact_dir / "strategy_rank_outcome_curves.joblib",
        name="strategy_rank_outcome_curves",
    )
    failures.extend(load_failures)
    if rank_curves is not None:
        failures.extend(_audit_rank_curve_bundle(rank_curves))

    response_models, load_failures = _load_joblib_dict(
        artifact_dir / "strategy_response_models.joblib",
        name="strategy_response_models",
    )
    failures.extend(load_failures)
    if response_models is not None:
        failures.extend(_audit_response_model_bundle(response_models, expected_fold_ids))

    return failures


def _audit_selected_controller(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    selected_arm = payload.get("selected_arm")
    selection_policy = payload.get("selection_policy")
    selected_metrics = payload.get("selected_metrics")
    no_backfill_overlay_selection = (
        isinstance(selected_arm, str)
        and selected_arm.endswith("__post_selection_overlay")
        and isinstance(selection_policy, dict)
        and selection_policy.get("select_no_backfill_overlay_only") is True
        and isinstance(selected_metrics, dict)
        and float(selected_metrics.get("action_entrants") or 0.0) == 0.0
        and selected_metrics.get("passed_selection_gates") is True
    )
    if selected_arm is not None and not no_backfill_overlay_selection:
        failures.append("walkforward_selected_controller_candidate.selected_arm is not null")
    if (
        payload.get("reason") != "no_arm_passed_selection_gates"
        and not no_backfill_overlay_selection
    ):
        failures.append("walkforward_selected_controller_candidate.reason is not no_arm_passed_selection_gates")
    if not isinstance(payload.get("selection_policy"), dict):
        failures.append("walkforward_selected_controller_candidate.selection_policy is missing")
    return failures


def _audit_state_registry(frame: pd.DataFrame, *, name: str, require_activation_fields: bool) -> list[str]:
    failures: list[str] = []
    required = [
        "state_level",
        "state_head",
        "component_group",
        "aggregate_status",
        "folds_seen",
        "trained_folds",
        "mean_oof_coverage",
        "status_counts",
    ]
    if require_activation_fields:
        required.extend(["recommended_status", "activation_registry_version"])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        failures.append(f"{name} missing columns: {missing}")
        return failures
    if frame.empty:
        failures.append(f"{name} is empty")
    if "activation_registry_version" in frame.columns:
        versions = set(frame["activation_registry_version"].dropna().astype(str).unique())
        if versions != {"market_state_activation_registry_v1"}:
            failures.append(f"{name} has unexpected activation_registry_version: {sorted(versions)}")
    if "recommended_status" in frame.columns:
        statuses = set(frame["recommended_status"].dropna().astype(str).unique())
        allowed = {"active_candidate", "disabled_candidate", "shadow_candidate", "shadow"}
        unknown = sorted(statuses - allowed)
        if unknown:
            failures.append(f"{name} has unexpected recommended_status values: {unknown}")
    return failures


def _audit_required_columns(frame: pd.DataFrame, *, name: str, required: list[str]) -> list[str]:
    missing = [column for column in required if column not in frame.columns]
    return [f"{name} missing columns: {missing}"] if missing else []


def _audit_nonempty(frame: pd.DataFrame, *, name: str) -> list[str]:
    return [f"{name} is empty"] if frame.empty else []


def _audit_numeric_finite(
    frame: pd.DataFrame,
    *,
    name: str,
    columns: list[str],
    mask: pd.Series | None = None,
    allow_empty_mask: bool = False,
) -> list[str]:
    failures: list[str] = []
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        failures.append(f"{name} missing numeric metric columns: {missing}")
        return failures
    if mask is None:
        work_mask = pd.Series(True, index=frame.index)
    else:
        work_mask = mask.reindex(frame.index).fillna(False).astype(bool)
    if not bool(work_mask.any()):
        if not allow_empty_mask:
            failures.append(f"{name} has no applicable rows for numeric metric audit")
        return failures
    for column in columns:
        values = pd.to_numeric(frame.loc[work_mask, column], errors="coerce")
        bad = values.isna() | values.isin([float("inf"), float("-inf")])
        if bool(bad.any()):
            failures.append(f"{name}.{column} has non-finite metric values: {int(bad.sum())}")
    return failures


def _audit_ratio_bounds(
    frame: pd.DataFrame,
    *,
    name: str,
    columns: list[str],
    mask: pd.Series | None = None,
) -> list[str]:
    failures: list[str] = []
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        failures.append(f"{name} missing bounded metric columns: {missing}")
        return failures
    if mask is None:
        work_mask = pd.Series(True, index=frame.index)
    else:
        work_mask = mask.reindex(frame.index).fillna(False).astype(bool)
    if not bool(work_mask.any()):
        return failures
    for column in columns:
        values = pd.to_numeric(frame.loc[work_mask, column], errors="coerce")
        finite_bad = values.isna() | values.isin([float("inf"), float("-inf")])
        if bool(finite_bad.any()):
            failures.append(f"{name}.{column} has non-finite bounded metric values: {int(finite_bad.sum())}")
            continue
        out_of_range = (values < -1e-12) | (values > 1.0 + 1e-12)
        if bool(out_of_range.any()):
            failures.append(f"{name}.{column} outside [0, 1]: {int(out_of_range.sum())}")
    return failures


def _audit_state_head_diagnostics(frame: pd.DataFrame) -> list[str]:
    name = "market_state_head_diagnostics"
    failures: list[str] = []
    required = [
        "state_level",
        "state_head",
        "component_group",
        "aggregate_status",
        "folds_seen",
        "trained_folds",
        "fallback_folds",
        "shadow_disabled_folds",
        "active_fold_share",
        "fallback_fold_share",
        "mean_source_count",
        "mean_validation_rows",
        "collapsed_folds",
        "status_counts",
    ]
    failures.extend(_audit_required_columns(frame, name=name, required=required))
    failures.extend(_audit_nonempty(frame, name=name))
    if failures:
        return failures

    failures.extend(
        _audit_numeric_finite(
            frame,
            name=name,
            columns=[
                "folds_seen",
                "trained_folds",
                "fallback_folds",
                "shadow_disabled_folds",
                "active_fold_share",
                "fallback_fold_share",
                "mean_source_count",
                "collapsed_folds",
            ],
        )
    )
    failures.extend(_audit_ratio_bounds(frame, name=name, columns=["active_fold_share", "fallback_fold_share"]))

    observed_mask = frame["state_level"].astype(str).eq("observed_axis")
    forecast_mask = frame["state_level"].astype(str).eq("forecast") & (
        pd.to_numeric(frame["trained_folds"], errors="coerce") > 0
    )
    if not bool(observed_mask.any()):
        failures.append(f"{name} has no observed_axis state rows")
    if not bool(forecast_mask.any()):
        failures.append(f"{name} has no trained forecast state rows")

    forecast_metrics = [
        "mean_validation_top_decile_lift",
        "mean_validation_rows",
        "mean_tail_average_precision",
        "mean_tail_ap_lift_p90",
        "mean_tail_brier_p90",
        "mean_tail_ece_5bin",
        "mean_tail_false_alarm_rate_p90",
        "mean_tail_recall_p90",
        "positive_validation_lift_share",
        "mean_oof_coverage",
        "min_oof_coverage",
        "mean_target_rows",
        "mean_target_std",
    ]
    failures.extend(
        _audit_numeric_finite(
            frame,
            name=name,
            columns=forecast_metrics,
            mask=forecast_mask,
            allow_empty_mask=False,
        )
    )
    failures.extend(
        _audit_ratio_bounds(
            frame,
            name=name,
            columns=[
                "mean_tail_average_precision",
                "mean_tail_brier_p90",
                "mean_tail_ece_5bin",
                "mean_tail_false_alarm_rate_p90",
                "mean_tail_recall_p90",
                "positive_validation_lift_share",
                "mean_oof_coverage",
                "min_oof_coverage",
            ],
            mask=forecast_mask,
        )
    )
    return failures


def _audit_strategy_response_metrics(response_oof: pd.DataFrame, effect_matrix: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    response_name = "strategy_response_oof_predictions"
    response_required = [
        "timestamp",
        "strategy_id",
        "head",
        "state_feature_coverage",
        "response_feature_coverage",
        "state_ood_score",
        "state_ood_cutoff",
        "state_ood_flag",
        "base_mu",
        "base_psl",
        "base_pto",
        "pred_resid_utility",
        "pred_resid_utility_lcb",
        "pred_resid_full_sl",
        "pred_resid_timeout",
        "actual_resid_utility",
        "actual_resid_full_sl",
        "actual_resid_timeout",
        "fold",
        "arm",
        "state_prediction_contract",
    ]
    failures.extend(_audit_required_columns(response_oof, name=response_name, required=response_required))
    failures.extend(_audit_nonempty(response_oof, name=response_name))
    if not failures:
        failures.extend(
            _audit_numeric_finite(
                response_oof,
                name=response_name,
                columns=[
                    "state_feature_coverage",
                    "response_feature_coverage",
                    "state_ood_score",
                    "state_ood_cutoff",
                    "base_mu",
                    "base_psl",
                    "base_pto",
                    "pred_resid_utility",
                    "pred_resid_utility_lcb",
                    "pred_resid_full_sl",
                    "pred_resid_timeout",
                    "actual_resid_utility",
                    "actual_resid_full_sl",
                    "actual_resid_timeout",
                ],
            )
        )
        failures.extend(
            _audit_ratio_bounds(
                response_oof,
                name=response_name,
                columns=[
                    "state_feature_coverage",
                    "response_feature_coverage",
                    "base_psl",
                    "base_pto",
                ],
            )
        )
        contracts = set(response_oof["state_prediction_contract"].dropna().astype(str).unique())
        if contracts != {"outer_fold_validation_state_scores"}:
            failures.append(
                "strategy_response_oof_predictions state_prediction_contract is not "
                f"outer_fold_validation_state_scores: {sorted(contracts)}"
            )
        heads = set(response_oof["head"].dropna().astype(str).unique())
        missing_heads = sorted(set(EXPECTED_ACTIVE_HEADS) - heads)
        if missing_heads:
            failures.append(f"{response_name} missing active heads: {missing_heads}")

    effect_name = "strategy_state_effect_matrix"
    effect_required = [
        "fold",
        "arm",
        "scope",
        "scope_value",
        "state_feature",
        "target",
        "rows",
        "state_q10",
        "state_q90",
        "target_mean_state_q10",
        "target_mean_state_q90",
        "target_q90_minus_q10",
        "pearson",
        "spearman",
    ]
    failures.extend(_audit_required_columns(effect_matrix, name=effect_name, required=effect_required))
    failures.extend(_audit_nonempty(effect_matrix, name=effect_name))
    if not effect_matrix.empty and not [column for column in effect_required if column not in effect_matrix.columns]:
        failures.extend(
            _audit_numeric_finite(
                effect_matrix,
                name=effect_name,
                columns=[
                    "rows",
                    "state_q10",
                    "state_q90",
                    "target_mean_state_q10",
                    "target_mean_state_q90",
                    "target_q90_minus_q10",
                    "pearson",
                    "spearman",
                ],
            )
        )
        expected_targets = {
            "pred_resid_utility",
            "pred_resid_utility_lcb",
            "pred_resid_full_sl",
            "pred_resid_timeout",
        }
        actual_targets = set(effect_matrix["target"].dropna().astype(str).unique())
        missing_targets = sorted(expected_targets - actual_targets)
        if missing_targets:
            failures.append(f"{effect_name} missing response targets: {missing_targets}")
        if int((pd.to_numeric(effect_matrix["rows"], errors="coerce") <= 0).sum()):
            failures.append(f"{effect_name}.rows contains non-positive support")
    return failures


def _audit_controller_metric_tables(
    *,
    schedule: pd.DataFrame,
    action_audit: pd.DataFrame,
    controller_diag: pd.DataFrame,
    action_utility: pd.DataFrame,
    edge_validation: pd.DataFrame,
    edge_bucket: pd.DataFrame,
    suppression_utility: pd.DataFrame,
    suppression_aggregate: pd.DataFrame,
    baseline_suppression_utility: pd.DataFrame,
    baseline_suppression_aggregate: pd.DataFrame,
) -> list[str]:
    failures: list[str] = []
    schedule_metric_cols = [
        "risk_severity",
        "prediction_coverage",
        "min_prediction_coverage",
        "state_ood_score_mean",
        "state_ood_score_max",
        "state_ood_cutoff",
        "state_ood_share",
        "mean_pred_utility",
        "mean_pred_lcb",
        "mean_pred_full_sl",
        "mean_pred_timeout",
        "tail_candidate_count",
        "suppressed_candidate_count",
        "predicted_removed_loss_avoided",
        "predicted_removed_winner_sacrificed",
        "predicted_action_edge",
        "action_edge_per_suppressed",
    ]
    failures.extend(
        _audit_required_columns(
            schedule,
            name="strategy_threshold_schedule",
            required=["fold", "arm", "timestamp", "strategy_id", "head", *schedule_metric_cols],
        )
    )
    if not schedule.empty and not [column for column in schedule_metric_cols if column not in schedule.columns]:
        failures.extend(
            _audit_numeric_finite(
                schedule,
                name="strategy_threshold_schedule",
                columns=schedule_metric_cols,
            )
        )
        failures.extend(
            _audit_ratio_bounds(
                schedule,
                name="strategy_threshold_schedule",
                columns=["prediction_coverage", "min_prediction_coverage", "state_ood_share"],
            )
        )

    action_cols = [
        "baseline_accepted",
        "current_accepted",
        "overlap",
        "entrants",
        "removed",
        "entrant_net_pnl",
        "removed_net_pnl",
        "net_replacement_pnl",
        "same_key_net_pnl_delta",
        "net_action_pnl_delta",
        "removed_loss_avoided",
        "removed_winner_pnl_sacrificed",
        "defensive_success",
    ]
    for frame, name in (
        (action_audit, "strategy_threshold_action_audit"),
        (edge_validation, "walkforward_threshold_action_edge_validation"),
    ):
        failures.extend(_audit_required_columns(frame, name=name, required=["fold", "arm", "timestamp", "strategy_id", *action_cols]))
        failures.extend(_audit_nonempty(frame, name=name))
        if not frame.empty and not [column for column in action_cols if column not in frame.columns]:
            failures.extend(_audit_numeric_finite(frame, name=name, columns=action_cols))

    diag_base_cols = [
        "trade_count",
        "net_pnl",
        "gross_pnl",
        "cost_pnl",
        "full_sl_rate",
        "timeout_rate",
        "mean_threshold_delta",
    ]
    diag_controller_cols = [
        "force_base_share",
        "mean_prediction_coverage",
        "mean_state_ood_score",
        "max_state_ood_score",
        "mean_state_ood_cutoff",
        "mean_state_ood_share",
        "mean_pred_utility",
        "mean_pred_lcb",
        "mean_pred_full_sl",
        "mean_pred_timeout",
        "mean_predicted_removed_loss_avoided",
        "mean_predicted_removed_winner_sacrificed",
        "mean_predicted_action_edge",
    ]
    failures.extend(
        _audit_required_columns(
            controller_diag,
            name="walkforward_controller_state_diagnostics",
            required=["fold", "arm", "head", *diag_base_cols, *diag_controller_cols],
        )
    )
    failures.extend(_audit_nonempty(controller_diag, name="walkforward_controller_state_diagnostics"))
    if not controller_diag.empty and not [column for column in diag_base_cols if column not in controller_diag.columns]:
        failures.extend(_audit_numeric_finite(controller_diag, name="walkforward_controller_state_diagnostics", columns=diag_base_cols))
    non_baseline = (
        controller_diag["arm"].astype(str).ne("S0_baseline_static_thresholds")
        if "arm" in controller_diag.columns
        else pd.Series(False, index=controller_diag.index)
    )
    if not controller_diag.empty and not [column for column in diag_controller_cols if column not in controller_diag.columns]:
        failures.extend(
            _audit_numeric_finite(
                controller_diag,
                name="walkforward_controller_state_diagnostics",
                columns=diag_controller_cols,
                mask=non_baseline,
            )
        )
        failures.extend(
            _audit_ratio_bounds(
                controller_diag,
                name="walkforward_controller_state_diagnostics",
                columns=["force_base_share", "mean_prediction_coverage", "mean_state_ood_share"],
                mask=non_baseline,
            )
        )

    utility_cols = [
        "baseline_accepted",
        "current_accepted",
        "overlap",
        "entrants",
        "removed",
        "net_action_pnl_delta",
        "removed_loss_avoided",
        "removed_winner_pnl_sacrificed",
        "defensive_success",
    ]
    failures.extend(_audit_required_columns(action_utility, name="walkforward_threshold_action_utility", required=["fold", "arm", "scope", "scope_value", *utility_cols]))
    failures.extend(_audit_nonempty(action_utility, name="walkforward_threshold_action_utility"))
    if not action_utility.empty and not [column for column in utility_cols if column not in action_utility.columns]:
        failures.extend(_audit_numeric_finite(action_utility, name="walkforward_threshold_action_utility", columns=utility_cols))

    edge_bucket_cols = [
        "schedule_rows",
        "baseline_accepted",
        "current_accepted",
        "entrants",
        "removed",
        "mean_threshold_delta",
        "mean_predicted_action_edge",
        "sum_predicted_action_edge",
        "net_action_pnl_delta",
        "removed_loss_avoided",
        "removed_winner_pnl_sacrificed",
        "defensive_success",
        "realized_minus_predicted_action_edge",
    ]
    failures.extend(_audit_required_columns(edge_bucket, name="walkforward_threshold_action_edge_bucket_performance", required=["fold", "arm", "predicted_action_edge_bucket", *edge_bucket_cols]))
    failures.extend(_audit_nonempty(edge_bucket, name="walkforward_threshold_action_edge_bucket_performance"))
    if not edge_bucket.empty and not [column for column in edge_bucket_cols if column not in edge_bucket.columns]:
        failures.extend(_audit_numeric_finite(edge_bucket, name="walkforward_threshold_action_edge_bucket_performance", columns=edge_bucket_cols))

    suppression_cols = [
        "suppressed_candidates",
        "raised_schedule_count",
        "mean_threshold_delta",
        "suppressed_net_return_sum",
        "suppressed_loss_avoided",
        "suppressed_winner_pnl_sacrificed",
        "realized_defensive_success",
        "suppressed_full_sl_rate",
        "suppressed_timeout_rate",
        "mean_predicted_action_edge",
        "sum_predicted_action_edge",
    ]
    for frame, name in (
        (suppression_utility, "walkforward_threshold_candidate_suppression_utility"),
        (baseline_suppression_utility, "walkforward_threshold_baseline_accepted_suppression_utility"),
    ):
        failures.extend(_audit_required_columns(frame, name=name, required=["fold", "arm", "scope", "scope_value", *suppression_cols]))
        if not frame.empty and not [column for column in suppression_cols if column not in frame.columns]:
            failures.extend(_audit_numeric_finite(frame, name=name, columns=suppression_cols))
            failures.extend(_audit_ratio_bounds(frame, name=name, columns=["suppressed_full_sl_rate", "suppressed_timeout_rate"]))

    suppression_agg_cols = [
        "folds_with_suppression",
        "suppressed_candidates",
        "suppressed_loss_avoided",
        "suppressed_winner_pnl_sacrificed",
        "realized_defensive_success",
        "positive_suppression_fold_share",
        "mean_suppressed_full_sl_rate",
        "mean_suppressed_timeout_rate",
    ]
    for frame, name in (
        (suppression_aggregate, "walkforward_threshold_candidate_suppression_aggregate"),
        (baseline_suppression_aggregate, "walkforward_threshold_baseline_accepted_suppression_aggregate"),
    ):
        failures.extend(_audit_required_columns(frame, name=name, required=["arm", "scope", "scope_value", *suppression_agg_cols]))
        if not frame.empty and not [column for column in suppression_agg_cols if column not in frame.columns]:
            failures.extend(_audit_numeric_finite(frame, name=name, columns=suppression_agg_cols))
            failures.extend(
                _audit_ratio_bounds(
                    frame,
                    name=name,
                    columns=[
                        "positive_suppression_fold_share",
                        "mean_suppressed_full_sl_rate",
                        "mean_suppressed_timeout_rate",
                    ],
                )
            )
    return failures


def _audit_portfolio_metric_tables(summary: pd.DataFrame, by_head: pd.DataFrame, overlap: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    summary_cols = [
        "trade_count",
        "net_pnl",
        "gross_pnl",
        "cost_pnl",
        "cost_to_abs_gross",
        "compounded_return",
        "max_drawdown",
        "worst_24h_net_pnl",
        "full_sl_rate",
        "timeout_rate",
        "avg_open_positions",
        "mean_threshold_delta",
        "p75_threshold_delta",
        "max_threshold_delta",
        "share_threshold_raised",
    ]
    failures.extend(_audit_required_columns(summary, name="portfolio_replay_summary", required=["fold", "arm", *summary_cols]))
    failures.extend(_audit_nonempty(summary, name="portfolio_replay_summary"))
    if not summary.empty and not [column for column in summary_cols if column not in summary.columns]:
        failures.extend(_audit_numeric_finite(summary, name="portfolio_replay_summary", columns=summary_cols))
        failures.extend(
            _audit_ratio_bounds(
                summary,
                name="portfolio_replay_summary",
                columns=["cost_to_abs_gross", "full_sl_rate", "timeout_rate", "share_threshold_raised"],
            )
        )

    by_head_cols = [
        "trade_count",
        "win_rate",
        "net_pnl",
        "gross_pnl",
        "cost_pnl",
        "mean_net_return",
        "q05_net_return",
        "full_sl_rate",
        "timeout_rate",
    ]
    failures.extend(_audit_required_columns(by_head, name="portfolio_replay_by_head", required=["fold", "arm", "head", *by_head_cols]))
    failures.extend(_audit_nonempty(by_head, name="portfolio_replay_by_head"))
    if not by_head.empty and not [column for column in by_head_cols if column not in by_head.columns]:
        failures.extend(_audit_numeric_finite(by_head, name="portfolio_replay_by_head", columns=by_head_cols))
        failures.extend(_audit_ratio_bounds(by_head, name="portfolio_replay_by_head", columns=["win_rate", "full_sl_rate", "timeout_rate"]))
        baseline_heads = set(
            by_head.loc[by_head["arm"].astype(str).eq("S0_baseline_static_thresholds"), "head"]
            .dropna()
            .astype(str)
            .unique()
        )
        missing_heads = sorted(set(EXPECTED_ACTIVE_HEADS) - baseline_heads)
        if missing_heads:
            failures.append(f"portfolio_replay_by_head baseline missing active heads: {missing_heads}")

    overlap_cols = [
        "accepted",
        "overlap_with_baseline",
        "new_vs_baseline",
        "removed_vs_baseline",
        "jaccard_vs_baseline",
        "position_size_sum",
        "position_size_mean",
        "entrant_net_pnl",
        "removed_net_pnl",
        "net_replacement_pnl",
        "removed_loss_avoided",
        "removed_winner_pnl_sacrificed",
        "defensive_success",
    ]
    failures.extend(_audit_required_columns(overlap, name="walkforward_overlap", required=["fold", "arm", *overlap_cols]))
    failures.extend(_audit_nonempty(overlap, name="walkforward_overlap"))
    if not overlap.empty and not [column for column in overlap_cols if column not in overlap.columns]:
        failures.extend(_audit_numeric_finite(overlap, name="walkforward_overlap", columns=overlap_cols))
        failures.extend(_audit_ratio_bounds(overlap, name="walkforward_overlap", columns=["jaccard_vs_baseline"]))
    return failures


def _audit_leave_one_head_out_metrics(frame: pd.DataFrame) -> list[str]:
    name = "market_state_leave_one_head_out_aggregate"
    failures: list[str] = []
    required = [
        "state_head",
        "action_arm_hint",
        "loo_replay_folds",
        "loo_mode",
        "loo_median_increment_net_pnl",
        "loo_mean_increment_net_pnl",
        "loo_q25_increment_net_pnl",
        "loo_positive_increment_share",
        "loo_mean_accepted_jaccard",
        "loo_mean_delta_trade_count",
        "loo_mean_threshold_raise_delta",
        "loo_state_head_defensive_success",
        "loo_state_head_median_defensive_success",
        "loo_state_head_positive_defensive_share",
        "loo_state_head_loss_avoided",
        "loo_state_head_winner_pnl_sacrificed",
        "loo_state_head_net_action_pnl_delta",
    ]
    failures.extend(_audit_required_columns(frame, name=name, required=required))
    failures.extend(_audit_nonempty(frame, name=name))
    if not frame.empty and not [column for column in required if column not in frame.columns]:
        failures.extend(
            _audit_numeric_finite(
                frame,
                name=name,
                columns=[
                    "loo_replay_folds",
                    "loo_median_increment_net_pnl",
                    "loo_mean_increment_net_pnl",
                    "loo_q25_increment_net_pnl",
                    "loo_positive_increment_share",
                    "loo_mean_accepted_jaccard",
                    "loo_mean_delta_trade_count",
                    "loo_mean_threshold_raise_delta",
                    "loo_state_head_defensive_success",
                    "loo_state_head_median_defensive_success",
                    "loo_state_head_positive_defensive_share",
                    "loo_state_head_loss_avoided",
                    "loo_state_head_winner_pnl_sacrificed",
                    "loo_state_head_net_action_pnl_delta",
                ],
            )
        )
        failures.extend(
            _audit_ratio_bounds(
                frame,
                name=name,
                columns=[
                    "loo_positive_increment_share",
                    "loo_mean_accepted_jaccard",
                    "loo_state_head_positive_defensive_share",
                ],
            )
        )
    return failures


def audit_artifacts(
    artifact_dir: Path,
    *,
    expected_rank_contract: str = EXPECTED_RANK_CONTRACT,
) -> list[str]:
    """Audit persisted walk-forward artifacts, not only manifest declarations."""

    manifest_path = artifact_dir / "manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.exists() else {}
    if _artifact_bundle_kind(manifest) in {"materialized_bundle", "scored_bundle"}:
        return audit_bundle_artifacts(
            artifact_dir,
            manifest,
            expected_rank_contract=expected_rank_contract,
        )

    failures: list[str] = []
    missing = _missing_artifacts(artifact_dir)
    if missing:
        failures.append(f"missing required artifacts: {missing}")
        return failures

    failures.extend(
        _audit_artifact_hashes(_load_artifact_json(artifact_dir, "artifact_hashes.json"), artifact_dir=artifact_dir)
    )
    feature_contract = _load_artifact_json(artifact_dir, "market_state_feature_contract.json")
    failures.extend(
        _audit_feature_contract(
            feature_contract,
            expected_rank_contract=expected_rank_contract,
        )
    )
    failures.extend(_audit_universe_contract(_load_artifact_json(artifact_dir, "market_state_universe_contract.json")))
    _, expected_fold_ids = _expected_fold_keys(feature_contract)
    failures.extend(_audit_target_definitions(_load_artifact_json(artifact_dir, "market_state_target_definitions.json"), expected_fold_ids))
    failures.extend(_audit_joblib_bundles(artifact_dir, feature_contract))
    failures.extend(
        _audit_controller_config(
            _load_artifact_json(artifact_dir, "strategy_threshold_controller_config.json"),
            expected_rank_contract=expected_rank_contract,
        )
    )
    failures.extend(
        _audit_selected_controller(_load_artifact_json(artifact_dir, "walkforward_selected_controller_candidate.json"))
    )
    activation_registry = _read_frame(artifact_dir / "market_state_activation_registry.csv")
    failures.extend(
        _audit_state_registry(
            activation_registry,
            name="market_state_activation_registry",
            require_activation_fields=True,
        )
    )
    state_head_registry = _read_frame(artifact_dir / "walkforward_state_head_registry.csv")
    failures.extend(
        _audit_state_registry(
            state_head_registry,
            name="walkforward_state_head_registry",
            require_activation_fields=False,
        )
    )
    state_head_diagnostics = _read_frame(artifact_dir / "market_state_head_diagnostics.csv")
    failures.extend(_audit_state_head_diagnostics(state_head_diagnostics))

    timestamp_panel = _read_frame(artifact_dir / "market_state_timestamp_panel.parquet")
    failures.extend(_audit_market_state_frame(timestamp_panel, name="market_state_timestamp_panel", include_split=True))

    state_oof = _read_frame(artifact_dir / "market_state_oof_predictions.parquet")
    failures.extend(_audit_market_state_frame(state_oof, name="market_state_oof_predictions", include_split=True))
    if "prediction_contract" not in state_oof.columns:
        failures.append("market_state_oof_predictions missing prediction_contract column")
    failures.extend(_audit_oof_state_values_match_panel(timestamp_panel, state_oof))

    response_oof = _read_frame(artifact_dir / "strategy_response_oof_predictions.parquet")
    failures.extend(_audit_state_join_invariance(response_oof, name="strategy_response_oof_predictions"))
    failures.extend(_audit_response_oof_uses_state_oof(response_oof, state_oof))
    state_effect_matrix = _read_frame(artifact_dir / "strategy_state_effect_matrix.csv")
    failures.extend(_audit_strategy_response_metrics(response_oof, state_effect_matrix))

    schedule = _read_frame(artifact_dir / "strategy_threshold_schedule.parquet")
    failures.extend(_audit_threshold_schedule(schedule))
    failures.extend(
        _audit_controller_metric_tables(
            schedule=schedule,
            action_audit=_read_frame(artifact_dir / "strategy_threshold_action_audit.csv"),
            controller_diag=_read_frame(artifact_dir / "walkforward_controller_state_diagnostics.csv"),
            action_utility=_read_frame(artifact_dir / "walkforward_threshold_action_utility.csv"),
            edge_validation=_read_frame(artifact_dir / "walkforward_threshold_action_edge_validation.csv"),
            edge_bucket=_read_frame(artifact_dir / "walkforward_threshold_action_edge_bucket_performance.csv"),
            suppression_utility=_read_frame(artifact_dir / "walkforward_threshold_candidate_suppression_utility.csv"),
            suppression_aggregate=_read_frame(artifact_dir / "walkforward_threshold_candidate_suppression_aggregate.csv"),
            baseline_suppression_utility=_read_frame(
                artifact_dir / "walkforward_threshold_baseline_accepted_suppression_utility.csv"
            ),
            baseline_suppression_aggregate=_read_frame(
                artifact_dir / "walkforward_threshold_baseline_accepted_suppression_aggregate.csv"
            ),
        )
    )

    summary = _read_frame(artifact_dir / "portfolio_replay_summary.csv")
    if "arm" not in summary.columns or "fold" not in summary.columns:
        failures.append("portfolio_replay_summary missing arm/fold columns")
    overlap_path = artifact_dir / "walkforward_overlap.csv"
    if overlap_path.exists():
        overlap = _read_frame(overlap_path)
        failures.extend(_audit_static_baseline_parity(summary, overlap))
        failures.extend(_audit_portfolio_metric_tables(summary, _read_frame(artifact_dir / "portfolio_replay_by_head.csv"), overlap))
    else:
        failures.append("walkforward_overlap missing; cannot verify static baseline parity")
    failures.extend(_audit_leave_one_head_out_metrics(_read_frame(artifact_dir / "market_state_leave_one_head_out_aggregate.csv")))
    accepted_path = artifact_dir / "accepted_trades.parquet"
    if accepted_path.exists():
        failures.extend(_audit_accepted_trades(_read_frame(accepted_path)))
    return failures


def audit_bundle_artifacts(
    artifact_dir: Path,
    manifest: dict[str, Any] | None = None,
    *,
    expected_rank_contract: str = EXPECTED_RANK_CONTRACT,
) -> list[str]:
    """Audit materialized/scored deployment bundles.

    These directories intentionally do not contain the full walk-forward OOF
    research ledger.  They must instead prove deployable parity: frozen source
    contracts, one-row-per-timestamp state panels, no-op execution when the
    controller is disabled, separate shadow proposals, and replay/decision-key
    integrity.
    """

    manifest = manifest or _load_json(artifact_dir / "manifest.json")
    failures: list[str] = []
    missing = _missing_bundle_artifacts(artifact_dir, manifest)
    if missing:
        failures.append(f"missing required bundle artifacts: {missing}")
        return failures

    if _artifact_bundle_kind(manifest) == "materialized_bundle":
        failures.extend(
            _audit_artifact_hashes(
                _load_artifact_json(artifact_dir, "artifact_hashes.json"),
                artifact_dir=artifact_dir,
            )
        )
        failures.extend(_audit_bundle_universe_contract(_load_artifact_json(artifact_dir, "market_state_universe_contract.json")))
    else:
        failures.extend(_audit_manifest_output_hashes(manifest, artifact_dir=artifact_dir))

    feature_contract = _load_artifact_json(artifact_dir, "market_state_feature_contract.json")
    failures.extend(
        _audit_bundle_feature_contract(
            feature_contract,
            expected_rank_contract=expected_rank_contract,
        )
    )
    failures.extend(_audit_bundle_timestamp_panel(_read_frame(artifact_dir / "market_state_timestamp_panel.parquet")))

    execution_enabled = bool(
        manifest.get(
            "controller_execution_enabled",
            _controller(manifest).get("execution_enabled", True),
        )
    )
    shadow_controller_only = bool(manifest.get("shadow_controller_only", False))
    schedule = _read_frame(artifact_dir / "strategy_threshold_schedule.parquet")
    failures.extend(
        _audit_bundle_threshold_schedule(
            schedule,
            name="strategy_threshold_schedule",
            require_noop=not execution_enabled,
        )
    )
    failures.extend(
        _audit_bundle_action_audit(
            _read_frame(artifact_dir / "strategy_threshold_action_audit.csv"),
            name="strategy_threshold_action_audit",
        )
    )
    if shadow_controller_only:
        proposed = _read_frame(artifact_dir / "shadow_controller_proposed_schedule.parquet")
        failures.extend(
            _audit_bundle_threshold_schedule(
                proposed,
                name="shadow_controller_proposed_schedule",
                require_noop=False,
                require_shadow_arm=True,
            )
        )
        failures.extend(
            _audit_bundle_action_audit(
                _read_frame(artifact_dir / "shadow_threshold_action_audit.csv"),
                name="shadow_threshold_action_audit",
            )
        )
        failures.extend(
            _audit_bundle_suppression_utility(
                _read_frame(artifact_dir / "shadow_threshold_candidate_suppression_utility.csv"),
                name="shadow_threshold_candidate_suppression_utility",
            )
        )

    failures.extend(_audit_bundle_replay_summary(_read_frame(artifact_dir / "controller_replay_summary.csv")))
    failures.extend(_audit_bundle_accepted_trades(_read_frame(artifact_dir / "accepted_trades.parquet")))
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--require-null-selection", action="store_true", default=False)
    parser.add_argument("--audit-artifacts", action="store_true", default=False)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path where the audit result JSON should be persisted.",
    )
    parser.add_argument(
        "--allow-disabled-by-activation-registry",
        action="store_true",
        default=False,
        help=(
            "Accept a fail-closed materialized bundle whose activation registry "
            "disabled all controller state heads."
        ),
    )
    parser.add_argument(
        "--expected-rank-contract",
        choices=sorted(SUPPORTED_RANK_CONTRACTS),
        default=EXPECTED_RANK_CONTRACT,
        help=(
            "Rank contract expected in the controller manifest and feature "
            "contracts. Defaults to the original timestamp-rank T1 contract; "
            "use anchor_global_policy_rank_reference for current global-rank "
            "T1 artifacts."
        ),
    )
    args = parser.parse_args()

    manifest = _load_json(args.artifact_dir / "manifest.json")
    artifact_kind = _artifact_bundle_kind(manifest)
    failures = audit_manifest(
        manifest,
        require_null_selection=bool(args.require_null_selection),
        allow_disabled_by_activation_registry=bool(args.allow_disabled_by_activation_registry),
        expected_rank_contract=str(args.expected_rank_contract),
    )
    if bool(args.audit_artifacts):
        failures.extend(
            audit_artifacts(
                args.artifact_dir,
                expected_rank_contract=str(args.expected_rank_contract),
            )
        )
    warnings = []
    if not bool(args.audit_artifacts):
        warnings.append(
            "artifact audit not run; pass --audit-artifacts for completion-grade "
            "market-state controller audit"
        )
    completion_grade_audit = bool(args.audit_artifacts)
    result = {
        "artifact_dir": str(args.artifact_dir),
        "artifact_kind": artifact_kind,
        "audit_scope": "manifest_and_artifacts" if completion_grade_audit else "manifest_only",
        "manifest_audit_enabled": True,
        "artifact_audit_enabled": bool(args.audit_artifacts),
        "artifact_audit_required_for_completion": True,
        "artifact_audit_checks": (
            (BUNDLE_ARTIFACT_AUDIT_CHECKS if artifact_kind in {"materialized_bundle", "scored_bundle"} else ARTIFACT_AUDIT_CHECKS)
            if bool(args.audit_artifacts)
            else []
        ),
        "expected_rank_contract": str(args.expected_rank_contract),
        "require_null_selection": bool(args.require_null_selection),
        "passed": not failures,
        "completion_grade_audit": completion_grade_audit,
        "completion_grade_passed": completion_grade_audit and not failures,
        "warnings": warnings,
        "failures": failures,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
