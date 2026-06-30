"""Quant-style stage gates for the performance-regime pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QuantStageGateConfig:
    enabled: bool = True
    fail_fast: bool = True
    min_outer_folds: int = 1
    min_strategy_count: int = 1
    min_train_rows: int = 100
    min_valid_rows: int = 20
    min_train_timestamps: int = 20
    min_valid_timestamps: int = 5
    min_feature_count: int = 10
    max_missing_family_share: float = 0.50
    min_label_std: float = 1e-4
    max_first_stage_oof_brier: float = 0.35
    min_first_stage_prediction_std: float = 1e-5
    min_pruned_leaf_count: int = 4
    min_mean_leaf_stability: float = 0.0
    min_interaction_candidate_count: int = 1
    min_archetype_count: int = 2
    min_archetype_compression_source_coverage: float = 0.999
    min_archetype_compression_silhouette_mean: float = -0.25
    max_archetype_compression_member_cov: float = 2.50
    max_archetype_compression_distance_to_seed_p95: float = 1.50
    max_archetype_expert_oof_brier: float = 0.35
    min_archetype_expert_prediction_std: float = 1e-5
    min_archetype_expert_predictive_fold_share: float = 0.25
    min_portfolio_action_prediction_std: float = 1e-6
    min_activation_deactivation_share: float = 0.01


@dataclass(frozen=True)
class StageGateDecision:
    stage: str
    passed: bool
    failures: tuple[str, ...]
    warnings: tuple[str, ...]
    metrics: dict[str, Any]

    def to_row(self, *, fold: int | None = None) -> dict[str, Any]:
        row: dict[str, Any] = {
            "stage": self.stage,
            "gate_passed": bool(self.passed),
            "failure_count": int(len(self.failures)),
            "warning_count": int(len(self.warnings)),
            "failures": "; ".join(self.failures),
            "warnings": "; ".join(self.warnings),
        }
        if fold is not None:
            row["fold"] = int(fold)
        row.update({str(k): _safe_metric(v) for k, v in self.metrics.items()})
        return row


class StageGateError(RuntimeError):
    def __init__(self, decision: StageGateDecision):
        super().__init__(
            f"Stage gate failed for {decision.stage}: "
            + ("; ".join(decision.failures) if decision.failures else "unknown failure")
        )
        self.decision = decision


def gate_config_for_profile(
    profile: str,
    *,
    enabled: bool = True,
    fail_fast: bool = True,
) -> QuantStageGateConfig:
    profile = str(profile or "standard").lower()
    if profile == "smoke":
        return QuantStageGateConfig(
            enabled=enabled,
            fail_fast=fail_fast,
            min_train_rows=1,
            min_valid_rows=1,
            min_train_timestamps=2,
            min_valid_timestamps=1,
            min_feature_count=1,
            max_missing_family_share=1.0,
            min_label_std=0.0,
            max_first_stage_oof_brier=1.0,
            min_first_stage_prediction_std=0.0,
            min_pruned_leaf_count=1,
            min_interaction_candidate_count=0,
            min_archetype_count=1,
            min_archetype_compression_source_coverage=0.0,
            min_archetype_compression_silhouette_mean=-1.0,
            max_archetype_compression_member_cov=1e6,
            max_archetype_compression_distance_to_seed_p95=1e6,
            max_archetype_expert_oof_brier=1.0,
            min_archetype_expert_prediction_std=0.0,
            min_archetype_expert_predictive_fold_share=0.0,
            min_portfolio_action_prediction_std=0.0,
            min_activation_deactivation_share=0.0,
        )
    if profile == "lenient":
        return QuantStageGateConfig(
            enabled=enabled,
            fail_fast=fail_fast,
            min_train_rows=20,
            min_valid_rows=5,
            min_train_timestamps=8,
            min_valid_timestamps=2,
            min_feature_count=3,
            max_missing_family_share=0.75,
            max_first_stage_oof_brier=0.45,
            min_pruned_leaf_count=1,
            min_interaction_candidate_count=0,
            min_archetype_count=1,
            min_archetype_compression_source_coverage=0.95,
            min_archetype_compression_silhouette_mean=-0.50,
            max_archetype_compression_member_cov=5.0,
            max_archetype_compression_distance_to_seed_p95=3.0,
            max_archetype_expert_oof_brier=0.45,
            min_archetype_expert_predictive_fold_share=0.10,
            min_activation_deactivation_share=0.0,
        )
    return QuantStageGateConfig(enabled=enabled, fail_fast=fail_fast)


def _num(metrics: dict[str, Any], key: str, default: float = np.nan) -> float:
    try:
        return float(metrics.get(key, default))
    except Exception:
        return float(default)


def _safe_metric(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else np.nan
    if isinstance(value, float):
        return value if np.isfinite(value) else np.nan
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return "; ".join(f"{k}={v}" for k, v in value.items())
    return str(value)


def _require(
    failures: list[str],
    metrics: dict[str, Any],
    key: str,
    op: str,
    threshold: float,
) -> None:
    value = _num(metrics, key)
    if op == ">=" and not (np.isfinite(value) and value >= threshold):
        failures.append(f"{key}={value:.6g} < required {threshold:.6g}")
    elif op == "<=" and not (np.isfinite(value) and value <= threshold):
        failures.append(f"{key}={value:.6g} > allowed {threshold:.6g}")


def evaluate_stage_gate(
    stage: str,
    metrics: dict[str, Any],
    config: QuantStageGateConfig,
) -> StageGateDecision:
    failures: list[str] = []
    warnings: list[str] = []
    stage = str(stage)
    if not config.enabled:
        return StageGateDecision(stage, True, (), ("stage gates disabled",), dict(metrics))

    if stage == "load_input":
        _require(failures, metrics, "row_count", ">=", 1)
        _require(failures, metrics, "column_count", ">=", 4)
    elif stage == "resolve_strategies_and_features":
        _require(failures, metrics, "strategy_count", ">=", config.min_strategy_count)
        _require(failures, metrics, "requested_feature_count", ">=", config.min_feature_count)
        if "remaining_model_prediction_feature_count" in metrics:
            _require(failures, metrics, "remaining_model_prediction_feature_count", "<=", 0.0)
        if "remaining_qfail_feature_count" in metrics:
            _require(failures, metrics, "remaining_qfail_feature_count", "<=", 0.0)
    elif stage == "build_outer_folds":
        _require(failures, metrics, "outer_fold_count", ">=", config.min_outer_folds)
        _require(failures, metrics, "timestamp_count", ">=", config.min_train_timestamps + config.min_valid_timestamps)
    elif stage == "prepare_fold":
        _require(failures, metrics, "train_rows", ">=", config.min_train_rows)
        _require(failures, metrics, "valid_rows", ">=", config.min_valid_rows)
        _require(failures, metrics, "train_timestamp_count", ">=", config.min_train_timestamps)
        _require(failures, metrics, "valid_timestamp_count", ">=", config.min_valid_timestamps)
    elif stage == "build_labels":
        _require(failures, metrics, "label_rows", ">=", 1)
        _require(failures, metrics, "anchor_rows", ">=", max(1, _num(metrics, "strategy_count", 1)))
        _require(failures, metrics, "min_bad_label_std", ">=", config.min_label_std)
    elif stage == "build_feature_matrices":
        _require(failures, metrics, "train_feature_count", ">=", config.min_feature_count)
        _require(failures, metrics, "valid_feature_count", ">=", config.min_feature_count)
        _require(failures, metrics, "missing_family_share", "<=", config.max_missing_family_share)
    elif stage == "train_first_stage":
        _require(failures, metrics, "model_count", ">=", 2)
        _require(failures, metrics, "mean_oof_weighted_brier", "<=", config.max_first_stage_oof_brier)
        _require(failures, metrics, "median_prediction_std", ">=", config.min_first_stage_prediction_std)
    elif stage == "extract_score_prune_leaves":
        _require(failures, metrics, "extracted_leaf_count", ">=", 1)
        _require(failures, metrics, "pruned_leaf_count", ">=", config.min_pruned_leaf_count)
        _require(failures, metrics, "mean_pruned_leaf_stability", ">=", config.min_mean_leaf_stability)
    elif stage == "extract_leaf_guided_interactions":
        if bool(metrics.get("interaction_gate_required", True)):
            total = _num(metrics, "pair_count", 0.0) + _num(metrics, "triple_count", 0.0)
            if total < config.min_interaction_candidate_count:
                failures.append(
                    f"interaction_candidate_count={total:.6g} < required {config.min_interaction_candidate_count}"
                )
    elif stage == "feedback_operator_generation_and_second_pass":
        if bool(metrics.get("feedback_run", False)) and bool(metrics.get("require_oof_improvement", True)):
            if not bool(metrics.get("second_pass_accepted", False)):
                failures.append("second feedback pass did not improve OOF and was rejected")
    elif stage == "cluster_archetypes":
        _require(failures, metrics, "archetype_count", ">=", config.min_archetype_count)
        raw_count = _num(metrics, "raw_archetype_count", 0.0)
        compressed_count = _num(metrics, "archetype_count", 0.0)
        if np.isfinite(raw_count) and np.isfinite(compressed_count) and raw_count > compressed_count:
            _require(
                failures,
                metrics,
                "compression_source_coverage_min",
                ">=",
                config.min_archetype_compression_source_coverage,
            )
            _require(
                failures,
                metrics,
                "compression_silhouette_mean",
                ">=",
                config.min_archetype_compression_silhouette_mean,
            )
            _require(
                failures,
                metrics,
                "compression_member_count_cov_max",
                "<=",
                config.max_archetype_compression_member_cov,
            )
            _require(
                failures,
                metrics,
                "compression_distance_to_seed_p95",
                "<=",
                config.max_archetype_compression_distance_to_seed_p95,
            )
    elif stage == "train_archetype_experts":
        if bool(metrics.get("skipped", False)):
            failures.append(str(metrics.get("skip_reason", "archetype expert stage skipped")))
        _require(failures, metrics, "expert_count", ">=", config.min_archetype_count)
        _require(failures, metrics, "predictive_expert_count", ">=", config.min_archetype_count)
        _require(failures, metrics, "mean_oof_weighted_brier", "<=", config.max_archetype_expert_oof_brier)
        _require(failures, metrics, "mean_prediction_std", ">=", config.min_archetype_expert_prediction_std)
        _require(
            failures,
            metrics,
            "predictive_expert_fold_share",
            ">=",
            config.min_archetype_expert_predictive_fold_share,
        )
    elif stage == "train_portfolio_calibrator":
        if bool(metrics.get("skipped", False)):
            failures.append(str(metrics.get("skip_reason", "portfolio calibrator stage skipped")))
        _require(failures, metrics, "diagnostic_rows", ">=", 1)
        _require(failures, metrics, "mean_action_prediction_std", ">=", config.min_portfolio_action_prediction_std)
        _require(failures, metrics, "activation_target_deactivation_share", ">=", config.min_activation_deactivation_share)

    return StageGateDecision(stage, len(failures) == 0, tuple(failures), tuple(warnings), dict(metrics))
