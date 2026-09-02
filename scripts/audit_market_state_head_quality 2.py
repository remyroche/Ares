#!/usr/bin/env python3
"""Audit market-state head quality and activation evidence.

This is a diagnostic audit, not a controller promotion gate. It makes the
state-head layer inspectable by combining forecast skill, collapse/coverage,
redundancy, response effects, action effects, and leave-one-out economics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/market_state_head_quality_20260626")

REQUIRED_DIAGNOSTIC_COLUMNS = {
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
    "mean_oof_coverage",
    "min_oof_coverage",
    "status_counts",
}

FORECAST_SKILL_COLUMNS = {
    "mean_validation_top_decile_lift",
    "mean_tail_average_precision",
    "mean_tail_ap_lift_p90",
    "mean_tail_brier_p90",
    "mean_tail_ece_5bin",
    "mean_tail_false_alarm_rate_p90",
    "mean_tail_recall_p90",
    "positive_validation_lift_share",
    "mean_target_rows",
    "mean_target_std",
}

DEFAULT_POLICY = {
    "min_trained_folds": 2,
    "min_active_fold_share": 0.67,
    "max_fallback_fold_share": 0.34,
    "max_collapsed_folds": 0,
    "min_oof_coverage": 0.80,
    "min_target_rows": 100,
    "min_target_std": 1e-6,
    "min_validation_rows": 30,
    "min_top_decile_lift": 0.0,
    "min_tail_ap_lift": 0.0,
    "min_positive_lift_share": 0.50,
    "max_tail_brier": 0.35,
    "max_tail_ece": 0.50,
    "max_tail_false_alarm": 0.50,
    "min_tail_recall": 0.0,
    "min_response_mean_abs_spearman": 0.03,
    "min_response_sign_stability": 0.50,
    "min_threshold_raise_share": 0.01,
    "min_loo_q25_increment": 0.0,
    "min_loo_positive_share": 0.50,
    "min_defensive_success": 0.0,
    "redundancy_abs_corr_warn": 0.95,
}


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


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or path.is_dir():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _num(value: Any, default: float = float("nan")) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _split_reasons(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return []
    return [part for part in str(value).split(";") if part and part.lower() != "nan"]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _finite_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out


def _merge_activation(diagnostics: pd.DataFrame, activation: pd.DataFrame) -> pd.DataFrame:
    if activation.empty:
        out = diagnostics.copy()
        for column in [
            "recommended_status",
            "activation_disable_reason",
            "forecast_skill_gate_pass",
            "response_gate_pass",
            "action_gate_pass",
            "leave_one_out_gate_pass",
            "defensive_action_gate_pass",
            "max_abs_spearman_corr",
            "redundant_with",
            "redundancy_flag",
            "response_mean_abs_spearman",
            "response_sign_stability",
            "threshold_raise_share",
            "suppressed_candidate_count",
            "loo_q25_increment_net_pnl",
            "loo_positive_increment_share",
            "loo_state_head_defensive_success",
            "loo_state_head_loss_avoided",
            "loo_state_head_winner_pnl_sacrificed",
        ]:
            out[column] = np.nan
        return out
    extra_cols = [
        col
        for col in [
            "state_head",
            "recommended_status",
            "activation_disable_reason",
            "forecast_skill_gate_pass",
            "response_gate_pass",
            "action_gate_pass",
            "leave_one_out_gate_pass",
            "defensive_action_gate_pass",
            "max_abs_spearman_corr",
            "redundant_with",
            "redundancy_group",
            "redundancy_flag",
            "response_mean_abs_q90_q10",
            "response_max_abs_q90_q10",
            "response_mean_abs_spearman",
            "response_sign_stability",
            "threshold_raise_share",
            "suppressed_candidate_count",
            "mean_state_ood_share",
            "loo_median_increment_net_pnl",
            "loo_q25_increment_net_pnl",
            "loo_positive_increment_share",
            "loo_state_head_defensive_success",
            "loo_state_head_loss_avoided",
            "loo_state_head_winner_pnl_sacrificed",
            "loo_state_head_net_action_pnl_delta",
        ]
        if col in activation.columns
    ]
    return diagnostics.merge(
        activation.loc[:, extra_cols].drop_duplicates("state_head"),
        on="state_head",
        how="left",
        suffixes=("", "_activation"),
    )


def _quality_reasons(row: pd.Series, policy: dict[str, Any]) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    state_level = str(row.get("state_level"))
    aggregate_status = str(row.get("aggregate_status"))

    if aggregate_status != "active":
        failures.append(f"aggregate_status_{aggregate_status or 'missing'}")
    if _num(row.get("folds_seen"), 0.0) <= 0:
        failures.append("no_folds_seen")
    if _num(row.get("active_fold_share"), 0.0) < float(policy["min_active_fold_share"]):
        failures.append("low_active_fold_share")
    if _num(row.get("fallback_fold_share"), 0.0) > float(policy["max_fallback_fold_share"]):
        failures.append("high_fallback_fold_share")
    if _num(row.get("collapsed_folds"), 0.0) > float(policy["max_collapsed_folds"]):
        failures.append("collapsed_output")
    min_oof_coverage = _num(row.get("min_oof_coverage"), np.nan)
    if state_level == "forecast":
        if not np.isfinite(min_oof_coverage) or min_oof_coverage < float(policy["min_oof_coverage"]):
            failures.append("low_oof_coverage")
    elif np.isfinite(min_oof_coverage) and min_oof_coverage < float(policy["min_oof_coverage"]):
        warnings.append("low_state_output_coverage")

    if state_level == "forecast":
        if _num(row.get("trained_folds"), 0.0) < float(policy["min_trained_folds"]):
            failures.append("insufficient_trained_folds")
        if _num(row.get("mean_validation_rows"), 0.0) < float(policy["min_validation_rows"]):
            failures.append("low_validation_rows")
        if _num(row.get("mean_target_rows"), 0.0) < float(policy["min_target_rows"]):
            failures.append("low_target_rows")
        if _num(row.get("mean_target_std"), 0.0) <= float(policy["min_target_std"]):
            failures.append("collapsed_target")
        if _num(row.get("mean_validation_top_decile_lift"), -np.inf) <= float(policy["min_top_decile_lift"]):
            failures.append("nonpositive_top_decile_lift")
        if _num(row.get("mean_tail_ap_lift_p90"), -np.inf) <= float(policy["min_tail_ap_lift"]):
            failures.append("nonpositive_tail_ap_lift")
        if _num(row.get("positive_validation_lift_share"), 0.0) < float(policy["min_positive_lift_share"]):
            failures.append("forecast_lift_not_recurrent")
        if _num(row.get("mean_tail_brier_p90"), np.inf) > float(policy["max_tail_brier"]):
            warnings.append("tail_brier_high")
        if _num(row.get("mean_tail_ece_5bin"), np.inf) > float(policy["max_tail_ece"]):
            warnings.append("tail_ece_high")
        if _num(row.get("mean_tail_false_alarm_rate_p90"), np.inf) > float(policy["max_tail_false_alarm"]):
            warnings.append("tail_false_alarm_high")
        if _num(row.get("mean_tail_recall_p90"), 0.0) <= float(policy["min_tail_recall"]):
            warnings.append("tail_recall_zero")

    max_corr = _num(row.get("max_abs_spearman_corr"), np.nan)
    if np.isfinite(max_corr) and max_corr >= float(policy["redundancy_abs_corr_warn"]):
        warnings.append("high_redundancy_correlation")
    if _bool(row.get("redundancy_flag")) and not _bool(row.get("response_gate_pass")):
        failures.append("redundant_without_response_effect")

    response_effect = _num(row.get("response_mean_abs_spearman"), np.nan)
    if np.isfinite(response_effect) and response_effect < float(policy["min_response_mean_abs_spearman"]):
        warnings.append("weak_response_effect")
    sign_stability = _num(row.get("response_sign_stability"), np.nan)
    if np.isfinite(sign_stability) and sign_stability < float(policy["min_response_sign_stability"]):
        warnings.append("unstable_response_sign")

    raise_share = _num(row.get("threshold_raise_share"), np.nan)
    if np.isfinite(raise_share) and raise_share < float(policy["min_threshold_raise_share"]):
        warnings.append("no_material_threshold_action")
    q25 = _num(row.get("loo_q25_increment_net_pnl"), np.nan)
    if np.isfinite(q25) and q25 < float(policy["min_loo_q25_increment"]):
        failures.append("negative_loo_q25_increment")
    positive_share = _num(row.get("loo_positive_increment_share"), np.nan)
    if np.isfinite(positive_share) and positive_share < float(policy["min_loo_positive_share"]):
        failures.append("loo_increment_not_recurrent")
    defensive = _num(row.get("loo_state_head_defensive_success"), np.nan)
    if np.isfinite(defensive) and defensive <= float(policy["min_defensive_success"]):
        failures.append("defensive_success_not_positive")
    winner_sacrificed = _num(row.get("loo_state_head_winner_pnl_sacrificed"), np.nan)
    loss_avoided = _num(row.get("loo_state_head_loss_avoided"), np.nan)
    if np.isfinite(winner_sacrificed) and np.isfinite(loss_avoided) and winner_sacrificed > loss_avoided + 1e-12:
        failures.append("winner_sacrifice_exceeds_loss_avoided")

    return list(dict.fromkeys(failures)), list(dict.fromkeys(warnings))


def _grade_from_reasons(failures: list[str], warnings: list[str], row: pd.Series) -> str:
    if failures:
        return "fail"
    status = str(row.get("recommended_status", ""))
    if status == "active_candidate" and not warnings:
        return "execution_candidate"
    if status == "active_candidate":
        return "watch_active_candidate"
    if warnings:
        return "shadow_watch"
    return "diagnostic_pass"


def _quality_table(diagnostics: pd.DataFrame, activation: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    numeric_cols = [
        "folds_seen",
        "trained_folds",
        "fallback_folds",
        "shadow_disabled_folds",
        "active_fold_share",
        "fallback_fold_share",
        "mean_source_count",
        "mean_validation_rows",
        "mean_validation_top_decile_lift",
        "mean_tail_average_precision",
        "mean_tail_ap_lift_p90",
        "mean_tail_brier_p90",
        "mean_tail_ece_5bin",
        "mean_tail_false_alarm_rate_p90",
        "mean_tail_recall_p90",
        "collapsed_folds",
        "positive_validation_lift_share",
        "mean_oof_coverage",
        "min_oof_coverage",
        "mean_target_rows",
        "mean_target_std",
    ]
    diagnostics = _finite_frame(diagnostics, [col for col in numeric_cols if col in diagnostics.columns])
    merged = _merge_activation(diagnostics, activation)
    merged = _finite_frame(
        merged,
        [
            "max_abs_spearman_corr",
            "response_mean_abs_q90_q10",
            "response_max_abs_q90_q10",
            "response_mean_abs_spearman",
            "response_sign_stability",
            "threshold_raise_share",
            "suppressed_candidate_count",
            "mean_state_ood_share",
            "loo_median_increment_net_pnl",
            "loo_q25_increment_net_pnl",
            "loo_positive_increment_share",
            "loo_state_head_defensive_success",
            "loo_state_head_loss_avoided",
            "loo_state_head_winner_pnl_sacrificed",
            "loo_state_head_net_action_pnl_delta",
        ],
    )
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        failures, warnings = _quality_reasons(row, policy)
        record = row.to_dict()
        record["state_head_quality_grade"] = _grade_from_reasons(failures, warnings, row)
        record["state_head_quality_passed"] = not failures
        record["state_head_quality_fail_reasons"] = ";".join(failures)
        record["state_head_quality_warnings"] = ";".join(warnings)
        record["activation_disable_reason_list"] = ";".join(_split_reasons(row.get("activation_disable_reason")))
        rows.append(record)
    return pd.DataFrame(rows)


def _group_summary(quality: pd.DataFrame) -> pd.DataFrame:
    if quality.empty:
        return pd.DataFrame()
    return (
        quality.groupby(["state_level", "component_group"], dropna=False, sort=False)
        .agg(
            heads=("state_head", "nunique"),
            quality_passed=("state_head_quality_passed", "sum"),
            active_candidates=("recommended_status", lambda x: int((x.astype(str) == "active_candidate").sum())),
            median_top_decile_lift=("mean_validation_top_decile_lift", "median"),
            median_tail_ap_lift=("mean_tail_ap_lift_p90", "median"),
            median_tail_brier=("mean_tail_brier_p90", "median"),
            median_oof_coverage=("mean_oof_coverage", "median"),
            median_loo_q25_increment=("loo_q25_increment_net_pnl", "median"),
            median_defensive_success=("loo_state_head_defensive_success", "median"),
        )
        .reset_index()
    )


def _reason_counts(quality: pd.DataFrame, column: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    if column not in quality.columns:
        return {}
    for value in quality[column].dropna().astype(str):
        for reason in value.split(";"):
            if reason:
                counter[reason] += 1
    return dict(counter)


def audit_market_state_head_quality(
    artifact_dir: Path,
    output_dir: Path,
    *,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = {**DEFAULT_POLICY, **(policy or {})}
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = artifact_dir / "market_state_head_diagnostics.csv"
    activation_path = artifact_dir / "market_state_activation_registry.csv"
    diagnostics = _read_csv(diagnostics_path)
    activation = _read_csv(activation_path)
    failures: list[str] = []
    if diagnostics.empty:
        failures.append(f"{diagnostics_path} is missing or empty")
        quality = pd.DataFrame()
    else:
        missing = sorted(REQUIRED_DIAGNOSTIC_COLUMNS.difference(diagnostics.columns))
        if missing:
            failures.append(f"market_state_head_diagnostics missing columns: {missing}")
        forecast_rows = diagnostics.loc[diagnostics.get("state_level", pd.Series(dtype=str)).astype(str).eq("forecast")]
        missing_forecast = sorted(FORECAST_SKILL_COLUMNS.difference(forecast_rows.columns))
        if not forecast_rows.empty and missing_forecast:
            failures.append(f"forecast diagnostics missing skill columns: {missing_forecast}")
        quality = _quality_table(diagnostics, activation, policy)

    group_summary = _group_summary(quality)
    quality.to_csv(output_dir / "market_state_head_quality_by_head.csv", index=False)
    group_summary.to_csv(output_dir / "market_state_head_quality_by_group.csv", index=False)

    grade_counts = quality["state_head_quality_grade"].astype(str).value_counts().to_dict() if not quality.empty else {}
    status_counts = quality["recommended_status"].astype(str).value_counts().to_dict() if "recommended_status" in quality.columns else {}
    active_candidates = (
        quality.loc[quality.get("recommended_status", pd.Series(dtype=str)).astype(str).eq("active_candidate"), "state_head"]
        .dropna()
        .astype(str)
        .tolist()
        if not quality.empty and "recommended_status" in quality.columns
        else []
    )
    forecast_quality_failures = (
        quality.loc[
            quality.get("state_level", pd.Series(dtype=str)).astype(str).eq("forecast")
            & ~quality.get("state_head_quality_passed", pd.Series(dtype=bool)).fillna(False),
            "state_head",
        ]
        .dropna()
        .astype(str)
        .tolist()
        if not quality.empty
        else []
    )

    payload = {
        "generated_by": "audit_market_state_head_quality",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": str(artifact_dir),
        "output_dir": str(output_dir),
        "passed": not failures,
        "failures": failures,
        "policy": policy,
        "diagnostics_sha256": _sha256(diagnostics_path),
        "activation_registry_sha256": _sha256(activation_path),
        "diagnostic_rows": int(len(diagnostics)),
        "activation_rows": int(len(activation)),
        "state_heads": int(quality["state_head"].nunique(dropna=True)) if not quality.empty else 0,
        "forecast_heads": int((quality["state_level"].astype(str) == "forecast").sum()) if not quality.empty else 0,
        "observed_heads": int((quality["state_level"].astype(str) == "observed_axis").sum()) if not quality.empty else 0,
        "grade_counts": grade_counts,
        "recommended_status_counts": status_counts,
        "active_candidates": active_candidates,
        "forecast_quality_failure_heads": forecast_quality_failures,
        "quality_fail_reason_counts": _reason_counts(quality, "state_head_quality_fail_reasons"),
        "quality_warning_counts": _reason_counts(quality, "state_head_quality_warnings"),
        "activation_disable_reason_counts": _reason_counts(quality, "activation_disable_reason_list"),
    }
    (output_dir / "market_state_head_quality_gate.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(output_dir, payload, quality, group_summary)
    return payload


def _format_float(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except Exception:
        return ""
    if not np.isfinite(val):
        return ""
    return f"{val:.{digits}f}"


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._\n"
    view = frame.loc[:, [col for col in columns if col in frame.columns]].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: _format_float(x))
    lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in view.columns) + " |")
    return "\n".join(lines) + "\n"


def _write_report(
    output_dir: Path,
    payload: dict[str, Any],
    quality: pd.DataFrame,
    group_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Market-State Head Quality Audit",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Summary",
        "",
        f"- Structural audit passed: `{payload['passed']}`",
        f"- State heads: `{payload['state_heads']}`",
        f"- Forecast heads: `{payload['forecast_heads']}`",
        f"- Observed heads: `{payload['observed_heads']}`",
        f"- Active candidates: `{len(payload['active_candidates'])}`",
        "",
        "## Grade Counts",
        "",
        _markdown_table(pd.DataFrame([{"grade": k, "count": v} for k, v in payload["grade_counts"].items()]), ["grade", "count"]),
        "## Active Candidates",
        "",
    ]
    if payload["active_candidates"]:
        active = quality.loc[quality["state_head"].astype(str).isin(payload["active_candidates"])].copy()
        lines.append(
            _markdown_table(
                active,
                [
                    "state_head",
                    "component_group",
                    "mean_validation_top_decile_lift",
                    "mean_tail_ap_lift_p90",
                    "mean_tail_brier_p90",
                    "threshold_raise_share",
                    "loo_q25_increment_net_pnl",
                    "loo_state_head_defensive_success",
                    "state_head_quality_grade",
                    "state_head_quality_warnings",
                ],
            )
        )
    else:
        lines.append("_None._\n")

    lines.extend(["## Group Summary", ""])
    lines.append(
        _markdown_table(
            group_summary.sort_values(["quality_passed", "active_candidates"], ascending=[False, False])
            if not group_summary.empty
            else group_summary,
            [
                "state_level",
                "component_group",
                "heads",
                "quality_passed",
                "active_candidates",
                "median_top_decile_lift",
                "median_tail_ap_lift",
                "median_tail_brier",
                "median_oof_coverage",
                "median_loo_q25_increment",
                "median_defensive_success",
            ],
            max_rows=40,
        )
    )

    forecast = quality.loc[quality["state_level"].astype(str).eq("forecast")].copy() if not quality.empty else quality
    if not forecast.empty:
        lines.extend(["## Forecast Heads", ""])
        lines.append(
            _markdown_table(
                forecast.sort_values(
                    ["state_head_quality_passed", "mean_tail_ap_lift_p90", "mean_validation_top_decile_lift"],
                    ascending=[False, False, False],
                ),
                [
                    "state_head",
                    "component_group",
                    "mean_validation_top_decile_lift",
                    "mean_tail_ap_lift_p90",
                    "mean_tail_brier_p90",
                    "mean_tail_ece_5bin",
                    "mean_tail_recall_p90",
                    "positive_validation_lift_share",
                    "min_oof_coverage",
                    "state_head_quality_grade",
                    "state_head_quality_fail_reasons",
                    "state_head_quality_warnings",
                    "recommended_status",
                ],
                max_rows=40,
            )
        )

    for title, key in [
        ("Quality Fail Reasons", "quality_fail_reason_counts"),
        ("Quality Warnings", "quality_warning_counts"),
        ("Activation Disable Reasons", "activation_disable_reason_counts"),
    ]:
        counts = payload.get(key) or {}
        if counts:
            lines.extend([f"## {title}", ""])
            rows = pd.DataFrame([{"reason": reason, "count": count} for reason, count in counts.items()])
            lines.append(_markdown_table(rows.sort_values(["count", "reason"], ascending=[False, True]), ["reason", "count"]))

    lines.extend(
        [
            "## Interpretation",
            "",
            "This audit should not promote a controller by itself. It identifies which state heads have enough statistical quality and economic support to remain in the executable response stack. Controller activation still requires paired T1 replay gates and later untouched-window confirmation.",
            "",
        ]
    )
    (output_dir / "market_state_head_quality_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-tail-ap-lift", type=float, default=DEFAULT_POLICY["min_tail_ap_lift"])
    parser.add_argument("--max-tail-brier", type=float, default=DEFAULT_POLICY["max_tail_brier"])
    parser.add_argument("--max-tail-ece", type=float, default=DEFAULT_POLICY["max_tail_ece"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = {
        "min_tail_ap_lift": args.min_tail_ap_lift,
        "max_tail_brier": args.max_tail_brier,
        "max_tail_ece": args.max_tail_ece,
    }
    payload = audit_market_state_head_quality(args.artifact_dir, args.output_dir, policy=policy)
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
