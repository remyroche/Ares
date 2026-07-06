#!/usr/bin/env python3
"""Summarize current source-quality diagnostic artifacts.

This script is intentionally diagnostic-only. It does not train models or
promote labels. Its job is to make the current source-archetype, quality-label,
and path-risk state reproducible from existing artifacts, with a loud alignment
status before any training ablation uses the rows.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SOURCE_DIR = Path(
    "data_perp/reports/source_tags_s10_policy_net_v17_proxy_alignment_diagnostic_july_refresh_basegateoff"
)
DEFAULT_V2_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "source_archetypes_v2_july_source_refresh_basegateoff_v1"
)
DEFAULT_TIMEOUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "timeout_holding_risk_stage1_weekaware_july_source_refresh_basegateoff_v1"
)
DEFAULT_RECOVERY_SUPPORT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "utility_path_timeout_recovery_support_gates_badmae_splitmetrics_july_source_refresh_basegateoff_v1"
)
DEFAULT_RECOVERY_FAILURE_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "utility_path_timeout_recovery_failure_head_july_source_refresh_basegateoff_v1"
)
DEFAULT_BAD_MAE_GAP_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "bad_mae_recovery_feature_gap_july_source_refresh_basegateoff_v1"
)
DEFAULT_CLEAN_SUBSET_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "source_quality_clean_joined_subset_july_refresh_basegateoff_v1"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "source_quality_current_state_july_refresh_basegateoff_v1"
)

LABEL_COLS = [
    "quality_label_v0",
    "quality_label_source_rank_v1",
    "quality_label_source_wf_v1",
    "quality_label_clean_path_v2",
    "quality_label_recoverable_opportunity_v2",
    "quality_label_opportunity_capture_v3",
    "quality_label_economic_capture_v4",
]

RISK_SUMMARY_COLS = [
    "artifact",
    "decision",
    "feature_set",
    "source_bucket",
    "causal_gate",
    "selection",
    "mean_u",
    "worst_month_u",
    "positive_weeks",
    "weeks",
    "q25_week_u",
    "bad_mae_1r_rate",
    "bad_mae_negative_rate",
    "bad_mae_recovered_rate",
    "fast_bad_mae_negative_rate",
    "timeout_rate",
    "wide_barrier_25bps_rate",
    "utility_without_top_symbol",
]


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _table(frame: pd.DataFrame, cols: list[str] | None = None, limit: int | None = None) -> str:
    if frame.empty:
        return "No rows."
    view = frame.copy()
    if cols is not None:
        view = view[[col for col in cols if col in view.columns]]
    if limit is not None:
        view = view.head(limit)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
    try:
        return view.to_markdown(index=False)
    except Exception:
        return view.to_string(index=False)


def _status_from_failures(failures: list[str], warnings: list[str]) -> str:
    if failures:
        return "fail"
    if warnings:
        return "warning"
    return "pass"


def _artifact_inventory(paths: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, path in paths.items():
        rows.append(
            {
                "artifact": name,
                "path": str(path),
                "exists": bool(path.exists()),
                "bytes": int(path.stat().st_size) if path.exists() and path.is_file() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _alignment_summary(
    source_dir: Path,
    *,
    min_outcome_match_rate: float,
    min_prediction_match_rate: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    audit = _read_csv(source_dir / "row_alignment_audit.csv")
    if audit.empty:
        return audit, {
            "status": "fail",
            "failures": ["missing_row_alignment_audit"],
            "warnings": [],
            "min_outcome_match_rate": min_outcome_match_rate,
            "min_prediction_match_rate": min_prediction_match_rate,
        }

    row = audit.iloc[0].to_dict()
    failures: list[str] = []
    warnings: list[str] = []
    outcome_rate = _finite_float(row.get("outcome_match_rate"), 0.0)
    prediction_rate = _finite_float(row.get("prediction_match_rate"), 1.0)
    prediction_rows = int(_finite_float(row.get("prediction_rows"), 0.0))
    if outcome_rate < min_outcome_match_rate:
        failures.append(
            f"outcome_match_rate {outcome_rate:.4f} < required {min_outcome_match_rate:.4f}"
        )
    if prediction_rows > 0 and prediction_rate < min_prediction_match_rate:
        failures.append(
            f"prediction_match_rate {prediction_rate:.4f} < required {min_prediction_match_rate:.4f}"
        )
    for col in [
        "duplicate_candidate_id_rows",
        "duplicate_timestamp_symbol_rows",
        "duplicate_timestamp_symbol_side_rows",
        "rows_with_multiple_outcomes_joined",
        "rows_with_multiple_predictions_joined",
        "label_duplicate_keys",
        "prediction_duplicate_keys",
    ]:
        value = int(_finite_float(row.get(col), 0.0))
        if value:
            failures.append(f"{col}={value}")
    if int(_finite_float(row.get("metadata_columns_preserved"), 0.0)) != 1:
        failures.append("metadata_columns_not_preserved")
    reported_quality = str(row.get("alignment_quality", ""))
    if reported_quality and reported_quality not in {"pass", "ok", "clean"}:
        warnings.append(f"reported_alignment_quality={reported_quality}")
    reported_warnings = str(row.get("alignment_warnings", ""))
    if reported_warnings and reported_warnings.lower() not in {"nan", "none", ""}:
        warnings.append(f"reported_alignment_warnings={reported_warnings}")
    return audit, {
        "status": _status_from_failures(failures, warnings),
        "failures": failures,
        "warnings": warnings,
        "min_outcome_match_rate": min_outcome_match_rate,
        "min_prediction_match_rate": min_prediction_match_rate,
        "outcome_match_rate": outcome_rate,
        "prediction_match_rate": prediction_rate,
        "prediction_rows": prediction_rows,
    }


def _source_coverage(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    tag_cols = [col for col in source.columns if col.startswith("tag_")]
    rows: list[dict[str, Any]] = []
    total = max(len(source), 1)
    for col in tag_cols:
        count = int(source[col].fillna(False).astype(bool).sum())
        rows.append(
            {
                "scope": "multi_tag",
                "source": col.removeprefix("tag_"),
                "rows": count,
                "coverage_pct": count / total,
            }
        )
    if "primary_source_tag" in source.columns:
        for source_name, count in source["primary_source_tag"].value_counts(dropna=False).items():
            rows.append(
                {
                    "scope": "primary_source_tag",
                    "source": str(source_name),
                    "rows": int(count),
                    "coverage_pct": int(count) / total,
                }
            )
    ts_col = "__ts__" if "__ts__" in source.columns else "timestamp" if "timestamp" in source.columns else None
    symbol_col = "__symbol__" if "__symbol__" in source.columns else "symbol" if "symbol" in source.columns else None
    ts = pd.to_datetime(source[ts_col], utc=True, errors="coerce") if ts_col else pd.Series(dtype="datetime64[ns, UTC]")
    summary = {
        "rows": int(len(source)),
        "cols": int(source.shape[1]),
        "timestamp_col": ts_col,
        "symbol_col": symbol_col,
        "date_min": ts.min().isoformat() if len(ts.dropna()) else None,
        "date_max": ts.max().isoformat() if len(ts.dropna()) else None,
        "symbols": int(source[symbol_col].nunique()) if symbol_col else None,
        "tag_count": int(len(tag_cols)),
    }
    coverage = pd.DataFrame(rows).sort_values(["scope", "coverage_pct"], ascending=[True, False])
    return coverage, summary


def _label_distribution(labels: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = max(len(labels), 1)
    for col in LABEL_COLS:
        if col not in labels.columns:
            rows.append(
                {
                    "label_col": col,
                    "label_value": "missing",
                    "rows": 0,
                    "pct_of_all_rows": 0.0,
                }
            )
            continue
        counts = labels[col].value_counts(dropna=False).sort_index()
        for label_value, count in counts.items():
            rows.append(
                {
                    "label_col": col,
                    "label_value": str(label_value),
                    "rows": int(count),
                    "pct_of_all_rows": int(count) / total,
                }
            )
    return pd.DataFrame(rows)


def _compact_existing_table(path: Path, cols: list[str], *, sort_col: str | None = None, ascending: bool = False) -> pd.DataFrame:
    frame = _read_csv(path)
    if frame.empty:
        return frame
    use = [col for col in cols if col in frame.columns]
    out = frame.loc[:, use].copy()
    if sort_col and sort_col in out.columns:
        out = out.sort_values(sort_col, ascending=ascending, kind="mergesort")
    return out


def _risk_candidate_summary(risk_dirs: dict[str, Path]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for name, directory in risk_dirs.items():
        path = directory / "source_utility_path_timeout_risk_aggregate.csv"
        frame = _read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["artifact"] = name
        top = frame.sort_values("mean_u", ascending=False, kind="mergesort").head(12)
        if "bad_mae_negative_rate" in frame.columns:
            low_negative = frame[frame["mean_u"].gt(0.0)].sort_values(
                ["bad_mae_negative_rate", "mean_u"],
                ascending=[True, False],
                kind="mergesort",
            ).head(8)
            top = pd.concat([top, low_negative], ignore_index=True).drop_duplicates()
        parts.append(top)
    if not parts:
        return pd.DataFrame(columns=RISK_SUMMARY_COLS)
    out = pd.concat(parts, ignore_index=True)
    return out[[col for col in RISK_SUMMARY_COLS if col in out.columns]].sort_values(
        ["artifact", "mean_u"], ascending=[True, False], kind="mergesort"
    )


def _v2_summary(v2_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = _read_json(v2_dir / "manifest.json")
    scorecard = _compact_existing_table(
        v2_dir / "source_archetypes_v2_scorecard.csv",
        [
            "archetype",
            "rows",
            "coverage",
            "mean_utility",
            "median_utility",
            "p25_utility",
            "bad_mae_1r_rate",
            "timeout_rate",
            "decision",
        ],
        sort_col="mean_utility",
    )
    join_report = manifest.get("join_report", {}) if isinstance(manifest, dict) else {}
    status = "missing"
    failures: list[str] = []
    if manifest:
        q_rate = _finite_float(join_report.get("join_match_rate_vs_quality"), 0.0)
        l_rate = _finite_float(join_report.get("join_match_rate_vs_labels"), 0.0)
        status = "pass" if q_rate >= 0.999 and l_rate >= 0.999 else "fail"
        if status == "fail":
            failures.append(f"v2 join rates quality={q_rate:.4f} labels={l_rate:.4f}")
    return scorecard, {
        "status": status,
        "failures": failures,
        "rows": manifest.get("rows") if manifest else None,
        "date_min": manifest.get("date_min") if manifest else None,
        "date_max": manifest.get("date_max") if manifest else None,
        "symbols": manifest.get("symbols") if manifest else None,
        "join_report": join_report,
    }


def _clean_subset_summary(clean_subset_dir: Path | None) -> dict[str, Any]:
    if clean_subset_dir is None:
        return {"enabled": False, "status": "missing", "reason": "not_configured"}
    manifest = _read_json(clean_subset_dir / "manifest.json")
    if not manifest:
        return {
            "enabled": False,
            "status": "missing",
            "reason": "missing_manifest",
            "manifest_path": str(clean_subset_dir / "manifest.json"),
        }
    return {
        "enabled": True,
        "status": manifest.get("subset_status", manifest.get("overall_status", "unknown")),
        "overall_status": manifest.get("overall_status"),
        "rows": manifest.get("rows"),
        "columns": manifest.get("columns"),
        "date_min": manifest.get("date_min"),
        "date_max": manifest.get("date_max"),
        "symbols": manifest.get("symbols"),
        "duplicate_candidate_id_rows": manifest.get("duplicate_candidate_id_rows"),
        "join_report": manifest.get("join_report", {}),
        "v2_merge_report": manifest.get("v2_merge_report", {}),
        "warnings": manifest.get("warnings", []),
        "failures": manifest.get("failures", []),
        "manifest_path": str(clean_subset_dir / "manifest.json"),
    }


def _overall_status(parts: list[str]) -> str:
    if any(part in {"fail", "missing"} for part in parts):
        return "fail"
    if "warning" in parts:
        return "warning"
    return "pass"


def build_current_state_report(
    *,
    source_dir: Path,
    v2_dir: Path,
    timeout_dir: Path,
    recovery_support_dir: Path,
    recovery_failure_dir: Path,
    bad_mae_gap_dir: Path,
    clean_subset_dir: Path | None,
    output_dir: Path,
    min_outcome_match_rate: float,
    min_prediction_match_rate: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "candidate_source_tags": source_dir / "candidate_source_tags.parquet",
        "quality_label_candidates": source_dir / "quality_label_candidates.parquet",
        "row_alignment_audit": source_dir / "row_alignment_audit.csv",
        "label_ablation_manifest": source_dir / "label_ablation_manifest.json",
        "source_failure_modes": source_dir / "failure_mode_by_source.csv",
        "opportunity_capture": source_dir / "opportunity_capture_by_source.csv",
        "v2_manifest": v2_dir / "manifest.json",
        "v2_scorecard": v2_dir / "source_archetypes_v2_scorecard.csv",
        "timeout_aggregate": timeout_dir / "timeout_holding_risk_label_aggregate.csv",
        "recovery_support_aggregate": recovery_support_dir / "source_utility_path_timeout_risk_aggregate.csv",
        "recovery_failure_aggregate": recovery_failure_dir / "source_utility_path_timeout_risk_aggregate.csv",
        "bad_mae_feature_gap": bad_mae_gap_dir / "bad_mae_recovery_feature_gap_summary.csv",
    }
    optional_artifact_paths = {
        "clean_subset_manifest": clean_subset_dir / "manifest.json" if clean_subset_dir else Path("__not_configured__"),
        "clean_subset_parquet": (
            clean_subset_dir / "source_quality_clean_joined_subset.parquet"
            if clean_subset_dir
            else Path("__not_configured__")
        ),
    }
    inventory = _artifact_inventory(artifact_paths)
    optional_inventory = _artifact_inventory(optional_artifact_paths)
    if not optional_inventory.empty:
        optional_inventory["optional"] = True
        inventory["optional"] = False
        inventory = pd.concat([inventory, optional_inventory], ignore_index=True)
    missing_artifacts = inventory.loc[~inventory["exists"], "artifact"].astype(str).tolist()
    required_missing = [
        artifact
        for artifact in missing_artifacts
        if artifact not in {"clean_subset_manifest", "clean_subset_parquet"}
    ]

    source = pd.read_parquet(artifact_paths["candidate_source_tags"]) if artifact_paths["candidate_source_tags"].exists() else pd.DataFrame()
    labels = (
        pd.read_parquet(artifact_paths["quality_label_candidates"])
        if artifact_paths["quality_label_candidates"].exists()
        else pd.DataFrame()
    )
    coverage, source_summary = _source_coverage(source) if not source.empty else (pd.DataFrame(), {"rows": 0})
    label_dist = _label_distribution(labels) if not labels.empty else pd.DataFrame()
    alignment_audit, alignment = _alignment_summary(
        source_dir,
        min_outcome_match_rate=min_outcome_match_rate,
        min_prediction_match_rate=min_prediction_match_rate,
    )
    v2_scorecard, v2 = _v2_summary(v2_dir)
    clean_subset = _clean_subset_summary(clean_subset_dir)
    failure_modes = _compact_existing_table(
        source_dir / "failure_mode_by_source.csv",
        [
            "bucket",
            "rows",
            "coverage_pct",
            "mean_net_utility",
            "positive_utility_rate",
            "clean_win_rate",
            "path_failure_rate",
            "timeout_failure_rate",
            "recoverable_opportunity_rate",
            "missed_opportunity_rate",
            "no_edge_rate",
            "opportunity_captured_rate",
            "economic_capture_rate",
            "p90_mae",
        ],
        sort_col="mean_net_utility",
    )
    opportunity = _compact_existing_table(
        source_dir / "opportunity_capture_by_source.csv",
        [
            "bucket",
            "outcome_rows",
            "opportunity_rows",
            "opportunity_coverage_pct",
            "capture_rate",
            "economic_capture_rate",
            "capture_loss_rate",
            "missed_opportunity_rate",
            "mean_net_utility",
            "captured_mean_utility",
            "economic_capture_mean_utility",
            "loss_mean_utility",
            "mean_capture_efficiency",
            "p90_mae",
        ],
        sort_col="mean_net_utility",
    )
    timeout = _compact_existing_table(
        timeout_dir / "timeout_holding_risk_label_aggregate.csv",
        [
            "decision",
            "label",
            "feature_set",
            "selector",
            "fraction",
            "target_auc",
            "target_pr_auc_lift",
            "score_ic_timeout",
            "mean_u",
            "delta_mean_u_vs_valid",
            "timeout_rate",
            "valid_timeout_rate",
            "bad_mae_1r_rate",
            "wide_barrier_25bps_rate",
            "positive_weeks",
            "weeks",
            "q25_week_u",
        ],
        sort_col="target_auc",
    )
    risk = _risk_candidate_summary(
        {
            "recovery_support_split": recovery_support_dir,
            "recovery_failure_head": recovery_failure_dir,
        }
    )
    bad_mae_gap = _compact_existing_table(
        bad_mae_gap_dir / "bad_mae_recovery_feature_gap_summary.csv",
        [
            "scope",
            "contrast",
            "selection",
            "feature_set",
            "source_bucket",
            "causal_gate",
            "features_tested",
            "strong_z_features",
            "strong_auc_features",
            "top_feature",
            "top_best_auc",
            "top_direction",
            "diagnosis",
            "rows",
            "bad_mae_negative_rows",
            "bad_mae_recovered_rows",
            "negative_mean_u",
            "recovered_mean_u",
        ],
        sort_col="top_best_auc",
    )

    output_paths = {
        "inventory": output_dir / "source_quality_current_state_inventory.csv",
        "alignment_audit": output_dir / "source_quality_current_state_alignment_audit.csv",
        "source_coverage": output_dir / "source_quality_current_state_source_coverage.csv",
        "label_distribution": output_dir / "source_quality_current_state_label_distribution.csv",
        "failure_modes": output_dir / "source_quality_current_state_failure_modes.csv",
        "opportunity_capture": output_dir / "source_quality_current_state_opportunity_capture.csv",
        "v2_scorecard": output_dir / "source_quality_current_state_v2_scorecard.csv",
        "timeout_summary": output_dir / "source_quality_current_state_timeout_summary.csv",
        "risk_candidates": output_dir / "source_quality_current_state_risk_candidates.csv",
        "bad_mae_gap": output_dir / "source_quality_current_state_bad_mae_gap.csv",
        "report": output_dir / "source_quality_current_state_report.md",
        "manifest": output_dir / "manifest.json",
    }
    inventory.to_csv(output_paths["inventory"], index=False)
    alignment_audit.to_csv(output_paths["alignment_audit"], index=False)
    coverage.to_csv(output_paths["source_coverage"], index=False)
    label_dist.to_csv(output_paths["label_distribution"], index=False)
    failure_modes.to_csv(output_paths["failure_modes"], index=False)
    opportunity.to_csv(output_paths["opportunity_capture"], index=False)
    v2_scorecard.to_csv(output_paths["v2_scorecard"], index=False)
    timeout.to_csv(output_paths["timeout_summary"], index=False)
    risk.to_csv(output_paths["risk_candidates"], index=False)
    bad_mae_gap.to_csv(output_paths["bad_mae_gap"], index=False)

    label_manifest = _read_json(source_dir / "label_ablation_manifest.json")
    entries = label_manifest.get("experiments") or label_manifest.get("ablation_experiments") or []
    artifact_status = "fail" if required_missing else "pass"
    overall = _overall_status([artifact_status, alignment["status"], v2["status"]])
    manifest = {
        "status": overall,
        "artifact_status": artifact_status,
        "missing_artifacts": missing_artifacts,
        "required_missing_artifacts": required_missing,
        "source_summary": source_summary,
        "alignment": alignment,
        "v2_summary": v2,
        "clean_subset_summary": clean_subset,
        "label_ablation_experiment_count": int(len(entries)),
        "outputs": {name: str(path) for name, path in output_paths.items()},
    }
    _write_report(
        output_paths["report"],
        manifest=manifest,
        coverage=coverage,
        label_dist=label_dist,
        failure_modes=failure_modes,
        opportunity=opportunity,
        v2_scorecard=v2_scorecard,
        timeout=timeout,
        risk=risk,
        bad_mae_gap=bad_mae_gap,
    )
    output_paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def _write_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    coverage: pd.DataFrame,
    label_dist: pd.DataFrame,
    failure_modes: pd.DataFrame,
    opportunity: pd.DataFrame,
    v2_scorecard: pd.DataFrame,
    timeout: pd.DataFrame,
    risk: pd.DataFrame,
    bad_mae_gap: pd.DataFrame,
) -> None:
    alignment = manifest["alignment"]
    source_summary = manifest["source_summary"]
    lines = [
        "# Source Quality Current State",
        "",
        "Scope: diagnostic-only source archetypes, quality labels, regime/source matrices, and path-risk screens.",
        "",
        "## Executive Status",
        "",
        f"- Overall status: `{manifest['status']}`",
        f"- Artifact status: `{manifest['artifact_status']}`",
        f"- Broad source rows: `{source_summary.get('rows', 0)}`",
        f"- Broad date range: `{source_summary.get('date_min')}` to `{source_summary.get('date_max')}`",
        f"- Symbols: `{source_summary.get('symbols')}`",
        f"- V2 joined subset status: `{manifest['v2_summary'].get('status')}`",
        f"- V2 joined rows: `{manifest['v2_summary'].get('rows')}`",
        f"- Clean joined subset status: `{manifest['clean_subset_summary'].get('status')}`",
        f"- Clean joined subset rows: `{manifest['clean_subset_summary'].get('rows')}`",
        f"- Label ablation entries defined: `{manifest['label_ablation_experiment_count']}`",
        "",
        "## Alignment Gate",
        "",
        f"- Alignment status: `{alignment['status']}`",
        f"- Outcome match rate: `{alignment.get('outcome_match_rate')}`",
        f"- Prediction match rate: `{alignment.get('prediction_match_rate')}`",
        f"- Required outcome match rate: `{alignment['min_outcome_match_rate']}`",
        f"- Required prediction match rate: `{alignment['min_prediction_match_rate']}`",
        f"- Failures: `{'; '.join(alignment.get('failures') or ['none'])}`",
        f"- Warnings: `{'; '.join(alignment.get('warnings') or ['none'])}`",
        "",
        "Interpretation: `fail` means the broad materialization is not safe as a direct training-ablation join without a stricter row contract. It can still be used for diagnostic coverage.",
        "",
        "## Clean Joined Subset",
        "",
        f"- Enabled: `{manifest['clean_subset_summary'].get('enabled')}`",
        f"- Subset status: `{manifest['clean_subset_summary'].get('status')}`",
        f"- Overall status: `{manifest['clean_subset_summary'].get('overall_status')}`",
        f"- Rows: `{manifest['clean_subset_summary'].get('rows')}`",
        f"- Duplicate candidate IDs: `{manifest['clean_subset_summary'].get('duplicate_candidate_id_rows')}`",
        f"- Match vs labels: `{manifest['clean_subset_summary'].get('join_report', {}).get('join_match_rate_vs_labels')}`",
        f"- V2 match rate: `{manifest['clean_subset_summary'].get('v2_merge_report', {}).get('match_rate')}`",
        f"- Manifest: `{manifest['clean_subset_summary'].get('manifest_path')}`",
        "",
        "## Source Coverage",
        "",
        _table(coverage, ["scope", "source", "rows", "coverage_pct"], limit=35),
        "",
        "## Label Distribution",
        "",
        _table(label_dist, ["label_col", "label_value", "rows", "pct_of_all_rows"], limit=60),
        "",
        "## Primary Source Failure Modes",
        "",
        _table(failure_modes, limit=30),
        "",
        "## Opportunity Capture By Source",
        "",
        _table(opportunity, limit=30),
        "",
        "## V2 Archetype Scorecard",
        "",
        _table(v2_scorecard, limit=20),
        "",
        "## Timeout Learnability",
        "",
        _table(timeout, limit=20),
        "",
        "## Risk Candidate Summary",
        "",
        _table(risk, limit=40),
        "",
        "## Bad-MAE Recovery Feature Gap",
        "",
        _table(bad_mae_gap, limit=20),
        "",
        "## Next Steps",
        "",
        "1. Fix row alignment before using the broad source-quality rows in training ablations.",
        "2. Prefer the clean V2 joined subset for path-risk analysis until the broad join contract is repaired.",
        "3. Keep split bad-MAE metrics as first-class diagnostics; raw bad-MAE alone is too blunt.",
        "4. Run label ablations only after the alignment gate is `pass` or after explicitly selecting a clean joined subset.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--v2-dir", type=Path, default=DEFAULT_V2_DIR)
    parser.add_argument("--timeout-dir", type=Path, default=DEFAULT_TIMEOUT_DIR)
    parser.add_argument("--recovery-support-dir", type=Path, default=DEFAULT_RECOVERY_SUPPORT_DIR)
    parser.add_argument("--recovery-failure-dir", type=Path, default=DEFAULT_RECOVERY_FAILURE_DIR)
    parser.add_argument("--bad-mae-gap-dir", type=Path, default=DEFAULT_BAD_MAE_GAP_DIR)
    parser.add_argument("--clean-subset-dir", type=Path, default=DEFAULT_CLEAN_SUBSET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-outcome-match-rate", type=float, default=0.80)
    parser.add_argument("--min-prediction-match-rate", type=float, default=0.80)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when the current-state gate is not pass.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_current_state_report(
        source_dir=args.source_dir,
        v2_dir=args.v2_dir,
        timeout_dir=args.timeout_dir,
        recovery_support_dir=args.recovery_support_dir,
        recovery_failure_dir=args.recovery_failure_dir,
        bad_mae_gap_dir=args.bad_mae_gap_dir,
        clean_subset_dir=args.clean_subset_dir,
        output_dir=args.output_dir,
        min_outcome_match_rate=args.min_outcome_match_rate,
        min_prediction_match_rate=args.min_prediction_match_rate,
    )
    print(f"wrote current-state report: {manifest['outputs']['report']}")
    print(f"status: {manifest['status']}")
    if args.strict and manifest["status"] != "pass":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
