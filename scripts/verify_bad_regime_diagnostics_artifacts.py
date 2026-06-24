#!/usr/bin/env python3
"""Verify bad-regime diagnostic artifacts against the corrected goal contract.

This is an artifact-level guardrail.  Unit tests prove that individual code
paths work; this script checks that the retained report folders actually expose
the required diagnostics, metrics, clean feature contract, and training
intervention evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


EXPECTED_HEADS = ("long_bars", "long_dist", "short_asset", "short_boll")
EXPECTED_TARGETS = (
    "high_conf_miss",
    "high_conf_negative_net_pnl",
    "high_conf_tail_loss",
    "prediction_minus_outcome",
    "continuous_net_utility",
)
EXPECTED_MODEL_FAMILIES = (
    "prediction_controls_only",
    "nuisance_controls_only",
    "prediction_plus_nuisance",
    "clean_reconstructed_features",
    "model_support_variables",
    "market_state_archetypes",
    "prediction_plus_archetype",
    "support_x_market_interactions",
)
EXPECTED_CANONICAL_VARIABLES = (
    "prediction_support_quality",
    "prediction_reconstruction_anomaly",
    "prediction_path_instability",
    "regime_similarity_or_novelty",
    "leverage_funding_crowding",
    "liquidity_participation_stress",
    "tail_volatility_stress",
    "relative_value_dislocation",
    "breadth_market_state",
    "network_concentration",
)
DIRECT_TIME_IDENTIFIERS = {
    "timestamp",
    "datetime",
    "date",
    "week",
    "bad_week",
    "row_id",
    "__row_pos",
}
FORBIDDEN_EXACT_NAMES = {
    "y_move",
    "y_move_soft",
    "y_bin",
    "target",
    "return",
    "barrier_pct",
    "mae_ret",
    "mfe_ret",
    "mae",
    "mfe",
    "bars_to_mfe",
    "is_timeout",
    "exit_code",
    "label_code",
}
FORBIDDEN_FEATURE_TOKENS = (
    "leaf_target",
    "barrier_pct",
    "rank_bin_",
    "rank_bin_win_rate",
    "rank_bin_lift",
    "rank_bin_net_ret",
    "rank_bin_se",
    "net_ret_oof",
    "policy_result",
    "post_trade",
)
KEY_COLUMNS = {"timestamp", "symbol", "index", "row_index", "strategy_id"}


@dataclass
class Finding:
    severity: str
    check: str
    path: str
    detail: str


def _path_label(path: Path | str) -> str:
    return str(path)


def _add(findings: list[Finding], severity: str, check: str, path: Path | str, detail: str) -> None:
    findings.append(Finding(severity=severity, check=check, path=_path_label(path), detail=detail))


def _read_csv(path: Path, findings: list[Finding], *, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            _add(findings, "ERROR", "file_exists", path, "required CSV is missing")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        _add(findings, "ERROR", "read_csv", path, f"failed to read CSV: {exc}")
        return pd.DataFrame()


def _read_text(path: Path, findings: list[Finding], *, required: bool = True) -> str:
    if not path.exists():
        if required:
            _add(findings, "ERROR", "file_exists", path, "required report is missing")
        return ""
    try:
        return path.read_text()
    except Exception as exc:
        _add(findings, "ERROR", "read_text", path, f"failed to read text: {exc}")
        return ""


def _require_columns(
    df: pd.DataFrame,
    required: Iterable[str],
    findings: list[Finding],
    *,
    path: Path,
    check: str,
) -> bool:
    missing = [col for col in required if col not in df.columns]
    if missing:
        _add(findings, "ERROR", check, path, f"missing columns: {', '.join(missing)}")
        return False
    return True


def _require_values(
    values: Iterable[Any],
    expected: Iterable[str],
    findings: list[Finding],
    *,
    path: Path,
    check: str,
) -> None:
    actual = {str(v) for v in values if pd.notna(v)}
    missing = [v for v in expected if v not in actual]
    if missing:
        _add(findings, "ERROR", check, path, f"missing expected values: {', '.join(missing)}")


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    lowered = series.astype(str).str.lower()
    return lowered.isin({"true", "1", "yes", "y"})


def _is_forbidden_feature_name(name: str) -> bool:
    lowered = str(name).lower()
    if lowered in KEY_COLUMNS:
        return False
    if lowered in FORBIDDEN_EXACT_NAMES:
        return True
    return any(token in lowered for token in FORBIDDEN_FEATURE_TOKENS)


def _check_report_sections(path: Path, sections: Iterable[str], findings: list[Finding]) -> None:
    text = _read_text(path, findings)
    if not text:
        return
    for section in sections:
        if section not in text:
            _add(findings, "ERROR", "report_section", path, f"missing section text: {section}")


def _check_candidate_contracts(directory: Path, findings: list[Finding]) -> None:
    files = sorted(directory.glob("*_candidate_feature_contract.csv"))
    if not files:
        _add(findings, "ERROR", "candidate_contracts", directory, "no candidate feature contract files found")
        return
    required_cols = (
        "head",
        "strategy_id",
        "feature",
        "source_family",
        "allowed_by_clean_contract",
        "available_before_trade",
        "outcome_independent",
        "fold_fitted",
        "live_equivalent",
        "train_live_parity_validated",
        "causal_availability",
    )
    heads_seen: set[str] = set()
    for path in files:
        df = _read_csv(path, findings)
        if df.empty or not _require_columns(df, required_cols, findings, path=path, check="candidate_contract_schema"):
            continue
        heads_seen.update(df["head"].dropna().astype(str).unique())
        features = df["feature"].astype(str)
        allowed = _bool_series(df["allowed_by_clean_contract"])
        available = _bool_series(df["available_before_trade"])
        independent = _bool_series(df["outcome_independent"])
        forbidden = features.map(_is_forbidden_feature_name)
        direct_time = features.str.lower().isin(DIRECT_TIME_IDENTIFIERS)
        bad_allowed = df.loc[(forbidden | direct_time) & (allowed | available | independent), "feature"].astype(str).head(20)
        if not bad_allowed.empty:
            _add(
                findings,
                "ERROR",
                "candidate_contract_leakage",
                path,
                "forbidden or direct-time features marked usable: " + ", ".join(bad_allowed.tolist()),
            )
        if not df["causal_availability"].notna().all():
            _add(findings, "ERROR", "candidate_contract_causal_availability", path, "blank causal availability entries")
    missing_heads = [head for head in EXPECTED_HEADS if head not in heads_seen]
    if missing_heads:
        _add(findings, "ERROR", "candidate_contract_heads", directory, f"missing heads: {', '.join(missing_heads)}")


def _check_classifier_dir(directory: Path, findings: list[Finding]) -> None:
    _check_report_sections(
        directory / "diagnostic_report.md",
        (
            "High-confidence failure classifier",
            "Adversarial validation",
            "Residualized adversarial validation",
            "Feature reconstruction coverage",
        ),
        findings,
    )
    _check_candidate_contracts(directory, findings)

    high_conf = _read_csv(directory / "high_conf_failure_summary.csv", findings)
    if not high_conf.empty and _require_columns(
        high_conf,
        (
            "head",
            "auc_mean",
            "folds",
            "rows",
            "positive_rate",
            "rank_threshold",
            "candidate_feature_count",
            "date_min",
            "date_max",
        ),
        findings,
        path=directory / "high_conf_failure_summary.csv",
        check="high_conf_schema",
    ):
        _require_values(high_conf["head"], EXPECTED_HEADS, findings, path=directory, check="high_conf_heads")
        bad_threshold = high_conf.loc[
            pd.to_numeric(high_conf["rank_threshold"], errors="coerce").sub(0.70).abs() > 1e-6
        ]
        if not bad_threshold.empty:
            _add(findings, "ERROR", "high_conf_rank_threshold", directory, "rank_threshold is not consistently 0.70")

    adversarial = _read_csv(directory / "adversarial_validation_summary.csv", findings)
    if not adversarial.empty and _require_columns(
        adversarial,
        (
            "head",
            "diagnostic",
            "auc_mean",
            "bad_rows",
            "normal_rows",
            "bad_week",
            "baseline_weeks",
            "baseline_match_method",
            "baseline_match_score_mean",
        ),
        findings,
        path=directory / "adversarial_validation_summary.csv",
        check="adversarial_schema",
    ):
        diagnostics = set(adversarial["diagnostic"].dropna().astype(str))
        for required in ("adversarial_global_bad_weeks", "adversarial_local_bad_week"):
            if required not in diagnostics:
                _add(findings, "ERROR", "adversarial_diagnostic", directory, f"missing diagnostic={required}")

    residualized = _read_csv(directory / "adversarial_residualized_validation_summary.csv", findings)
    if not residualized.empty and _require_columns(
        residualized,
        (
            "head",
            "diagnostic",
            "raw_auc",
            "nuisance_auc",
            "residualized_auc",
            "incremental_auc_beyond_nuisance",
            "bad_week",
            "baseline_weeks",
            "baseline_match_method",
            "baseline_match_score_mean",
        ),
        findings,
        path=directory / "adversarial_residualized_validation_summary.csv",
        check="residualized_adversarial_schema",
    ):
        diagnostics = set(residualized["diagnostic"].dropna().astype(str))
        for required in ("residualized_adversarial_global_bad_weeks", "residualized_adversarial_local_bad_week"):
            if required not in diagnostics:
                _add(findings, "ERROR", "residualized_adversarial_diagnostic", directory, f"missing diagnostic={required}")


def _check_leaf_dir(directory: Path, findings: list[Finding]) -> None:
    _check_report_sections(
        directory / "diagnostic_report.md",
        (
            "Leaf instability",
            "Top Base/Meta Leaf Shifts",
            "Top Residual x Archetype Leaf Interactions",
        ),
        findings,
    )
    manifest = _read_csv(directory / "leaf_instability_manifest.csv", findings)
    required_manifest_cols = (
        "head",
        "meta_leaf_rows",
        "base_leaf_rows",
        "meta_top_instability_score",
        "meta_top_occupancy_shift",
        "meta_top_outcome_shift",
        "meta_top_calibration_shift",
        "base_top_instability_score",
        "base_top_occupancy_shift",
        "base_top_outcome_shift",
        "base_top_calibration_shift",
        "meta_leaf_archetype_interaction_rows",
        "base_leaf_archetype_interaction_rows",
    )
    if not manifest.empty and _require_columns(
        manifest,
        required_manifest_cols,
        findings,
        path=directory / "leaf_instability_manifest.csv",
        check="leaf_manifest_schema",
    ):
        _require_values(manifest["head"], EXPECTED_HEADS, findings, path=directory, check="leaf_manifest_heads")
        if pd.to_numeric(manifest["meta_leaf_rows"], errors="coerce").fillna(0).sum() <= 0:
            _add(findings, "ERROR", "leaf_manifest_meta_rows", directory, "no meta leaf rows recorded")

    interaction_files = sorted(directory.glob("*_leaf_archetype_interactions.csv"))
    if not interaction_files:
        _add(findings, "ERROR", "leaf_interaction_files", directory, "no residual x archetype leaf interaction files found")
        return
    required_cols = (
        "head",
        "model_kind",
        "tree_id",
        "leaf_id",
        "historical_support",
        "recent_support",
        "global_slope",
        "leaf_slope",
        "leaf_x_archetype_slope",
        "leaf_x_period_state_slope",
        "leaf_x_within_timestamp_slope",
        "episode_sign_stability",
        "economic_effect",
        "interaction_score",
    )
    for path in interaction_files:
        df = _read_csv(path, findings)
        if df.empty:
            _add(findings, "ERROR", "leaf_interaction_nonempty", path, "interaction file is empty")
            continue
        _require_columns(df, required_cols, findings, path=path, check="leaf_interaction_schema")
        if not any(str(col).startswith("context_leaf_mean__") for col in df.columns):
            _add(findings, "ERROR", "leaf_interaction_context", path, "missing context mean columns")


def _check_usefulness_dir(directory: Path, findings: list[Finding]) -> None:
    _check_report_sections(
        directory / "bad_regime_archetype_usefulness_report.md",
        (
            "Incremental Binary Classifier Signal",
            "Failure-gate acceptance metrics",
            "Period Versus Within-Period Decomposition",
            "Alias / Duplicate Audit",
            "Canonical Reduction",
            "Leave-One-Episode-Out Transfer",
            "Shadow Failure-Risk Sizing",
            "Training Intervention Recommendations",
        ),
        findings,
    )
    _check_candidate_contracts(directory, findings)

    model_summary = _read_csv(directory / "archetype_model_auc_summary.csv", findings)
    if not model_summary.empty and _require_columns(
        model_summary,
        (
            "head",
            "target",
            "target_kind",
            "model",
            "auc_mean",
            "roc_auc",
            "pr_auc",
            "log_loss",
            "brier",
            "failure_capture_at_5pct_abstain",
            "failure_capture_at_10pct_abstain",
            "failure_capture_at_20pct_abstain",
            "retained_return_mean_at_10pct_abstain",
            "tail_loss_avoided_at_10pct_abstain",
            "rejected_winner_cost_at_10pct_abstain",
        ),
        findings,
        path=directory / "archetype_model_auc_summary.csv",
        check="archetype_model_summary_schema",
    ):
        _require_values(model_summary["head"], EXPECTED_HEADS, findings, path=directory, check="model_summary_heads")
        _require_values(model_summary["target"], EXPECTED_TARGETS, findings, path=directory, check="model_summary_targets")
        _require_values(model_summary["model"], EXPECTED_MODEL_FAMILIES, findings, path=directory, check="model_summary_models")

    decomposition = _read_csv(directory / "archetype_period_within_decomposition.csv", findings)
    _require_columns(
        decomposition,
        (
            "head",
            "feature",
            "prediction_only_auc",
            "period_state_auc",
            "within_timestamp_state_auc",
            "period_x_prediction_auc",
            "full_period_within_interaction_auc",
            "delta_period_vs_prediction",
            "delta_within_vs_prediction",
            "delta_interaction_vs_prediction",
            "delta_full_vs_prediction",
        ),
        findings,
        path=directory / "archetype_period_within_decomposition.csv",
        check="period_within_schema",
    )

    audit = _read_csv(directory / "archetype_alias_resolution_audit.csv", findings)
    if not audit.empty and _require_columns(
        audit,
        (
            "head",
            "output_feature",
            "resolved_parents",
            "source_archetype",
            "requested_features",
            "resolved_features",
            "resolved_fraction",
            "fallback_fraction",
            "active_archetype",
            "exact_duplicate_group",
            "pearson_duplicate_group",
            "spearman_duplicate_group",
            "unique_values",
            "variance",
            "heads_available",
            "available_before_trade",
            "outcome_independent",
            "fold_fitted",
            "live_equivalent",
            "train_live_parity_validated",
        ),
        findings,
        path=directory / "archetype_alias_resolution_audit.csv",
        check="alias_audit_schema",
    ):
        if pd.to_numeric(audit["fallback_fraction"], errors="coerce").isna().all():
            _add(findings, "ERROR", "alias_audit_fallback", directory, "fallback_fraction is entirely blank")

    canonical = _read_csv(directory / "canonical_archetype_reduction.csv", findings)
    if not canonical.empty and _require_columns(
        canonical,
        (
            "canonical_variable",
            "state_family",
            "source_archetype",
            "mechanism_channel",
            "deployable_aliases",
            "top_parent_features",
            "recommended_for_training",
        ),
        findings,
        path=directory / "canonical_archetype_reduction.csv",
        check="canonical_schema",
    ):
        _require_values(
            canonical["canonical_variable"],
            EXPECTED_CANONICAL_VARIABLES,
            findings,
            path=directory,
            check="canonical_variables",
        )
        state_families = set(canonical["state_family"].dropna().astype(str))
        for family in ("model_state", "market_state"):
            if family not in state_families:
                _add(findings, "ERROR", "canonical_state_family", directory, f"missing state_family={family}")

    for filename, columns in {
        "archetype_leave_one_episode_transfer.csv": (
            "head",
            "target",
            "model",
            "heldout_episode",
            "transfer_roc_auc",
            "transfer_failure_capture_at_10pct_abstain",
        ),
        "shadow_failure_risk_head_summary.csv": (
            "head",
            "target",
            "model",
            "shadow_roc_auc",
            "shadow_failure_capture_at_10pct_abstain",
        ),
        "shadow_failure_risk_policy_eval.csv": (
            "head",
            "target",
            "model",
            "policy",
            "avg_size",
            "success_minus_failure_exposure",
            "tail_loss_delta_q05",
            "winner_haircut_mean",
            "loser_loss_reduction_mean",
            "risk_sizing_score",
        ),
        "training_intervention_recommendations.csv": (
            "head",
            "target",
            "action",
            "recommendation",
            "decision_reason",
            "recurrence_pass",
            "incremental_lift_pass",
            "economic_pass",
        ),
    }.items():
        path = directory / filename
        df = _read_csv(path, findings)
        if not df.empty:
            _require_columns(df, columns, findings, path=path, check=f"{filename}_schema")


def _write_json(path: Path, findings: list[Finding]) -> None:
    payload = [finding.__dict__ for finding in findings]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classifier-dir",
        default="data_perp/reports/meta_recent_failure_diagnostics_20260622_clean_contract_v1",
        help="Directory containing clean classifier/adversarial diagnostics.",
    )
    parser.add_argument(
        "--leaf-dir",
        default="data_perp/reports/meta_recent_failure_diagnostics_20260622_leaf_archetype_decomp_v1",
        help="Directory containing base/meta leaf instability and interaction diagnostics.",
    )
    parser.add_argument(
        "--usefulness-dir",
        default="data_perp/reports/meta_recent_failure_diagnostics_20260622_archetype_usefulness_multitarget_clean_contract_v1",
        help="Directory containing archetype usefulness, transfer, shadow-risk, and intervention diagnostics.",
    )
    parser.add_argument("--json-output", default="", help="Optional JSON file for detailed findings.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    findings: list[Finding] = []
    classifier_dir = Path(args.classifier_dir)
    leaf_dir = Path(args.leaf_dir)
    usefulness_dir = Path(args.usefulness_dir)

    _check_classifier_dir(classifier_dir, findings)
    _check_leaf_dir(leaf_dir, findings)
    _check_usefulness_dir(usefulness_dir, findings)

    error_count = sum(1 for finding in findings if finding.severity == "ERROR")
    warning_count = sum(1 for finding in findings if finding.severity == "WARNING")
    print(
        f"[bad_regime_artifact_verify] errors={error_count} warnings={warning_count} "
        f"checks_with_findings={len(findings)}"
    )
    for finding in findings:
        print(f"{finding.severity}\t{finding.check}\t{finding.path}\t{finding.detail}")
    if args.json_output:
        _write_json(Path(args.json_output), findings)
    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
