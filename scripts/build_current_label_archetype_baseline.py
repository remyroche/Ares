#!/usr/bin/env python3
"""Build the current label/archetype Stage 0 baseline report.

The report is intentionally read-only over completed diagnostic artifacts. It
does not train, tune, or mutate any source data. Its job is to answer the
promotion gate questions from the label/archetype roadmap:

- What is the current best candidate?
- Why is it not training-ready?
- Which failure mode dominates?
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("data_perp/reports/source_quality_label_walkforward_ablation_v1_sideaware_20260702")
DEFAULT_OUTPUT = Path("reports/current_label_archetype_baseline.md")
SIDE_COVERAGE_AUDIT_JSON = Path("side_coverage_audit/label_side_coverage_audit.json")
UTILITY_HORIZON_NOTE = (
    "Corrected roadmap utility window: 1-6 hours, not 1-60 minutes. "
    "The current Stage 0/1 bundle remains diagnostic evidence, but its label "
    "inputs must be horizon-checked before promotion: the legacy `s10_policy_net` "
    "utility comes from a simple-policy replay path, while the newer "
    "`first_touch_c0_fast6` materialization encodes `fast6` as six 15-minute bars "
    "plus a 96-bar timeout path."
)


@dataclass(frozen=True)
class InputSpec:
    key: str
    relpath: str
    description: str


REQUIRED_INPUTS = (
    InputSpec(
        "source_quality_aggregate",
        "source_quality_label_walkforward_aggregate.csv",
        "source-quality walk-forward aggregate",
    ),
    InputSpec(
        "source_quality_monthly",
        "source_quality_label_walkforward_monthly.csv",
        "source-quality walk-forward monthly rows",
    ),
    InputSpec(
        "feature_learnability",
        "feature_learnability/source_tag_feature_learnability_aggregate.csv",
        "source-tag feature learnability aggregate",
    ),
    InputSpec(
        "utility_label_rework",
        "utility_label_rework/source_utility_label_rework_aggregate.csv",
        "utility-label rework aggregate",
    ),
    InputSpec(
        "utility_risk_gate",
        "utility_risk_gate/source_utility_risk_gate_aggregate.csv",
        "utility risk-gate aggregate",
    ),
    InputSpec(
        "candidate_weekly_aggregate",
        "utility_risk_gate_candidate_weekly/candidate_weekly_aggregate.csv",
        "shortlisted risk-gate weekly aggregate",
    ),
    InputSpec(
        "candidate_weekly_metrics",
        "utility_risk_gate_candidate_weekly/candidate_weekly_metrics.csv",
        "shortlisted risk-gate weekly metrics",
    ),
    InputSpec(
        "path_risk_aggregate",
        "utility_path_risk_dual_head/source_utility_path_risk_dual_head_aggregate.csv",
        "utility/path-risk dual-head aggregate",
    ),
    InputSpec(
        "path_risk_weekly",
        "utility_path_risk_dual_head/source_utility_path_risk_dual_head_weekly.csv",
        "utility/path-risk dual-head weekly rows",
    ),
)

OPTIONAL_INPUTS = (
    InputSpec(
        "timeout_holding_aggregate",
        "timeout_holding_risk/timeout_holding_risk_label_aggregate.csv",
        "timeout/holding-risk label aggregate",
    ),
    InputSpec(
        "timeout_stage1_aggregate",
        "timeout_holding_risk_stage1_metrics_v1/timeout_holding_risk_label_aggregate.csv",
        "timeout/holding-risk Stage 1 metrics aggregate",
    ),
    InputSpec(
        "timeout_stage1_calibration",
        "timeout_holding_risk_stage1_metrics_v1/timeout_holding_risk_calibration_deciles.csv",
        "timeout/holding-risk Stage 1 calibration deciles",
    ),
    InputSpec(
        "timeout_stage1_weekaware_aggregate",
        "timeout_holding_risk_stage1_weekaware_v1/timeout_holding_risk_label_aggregate.csv",
        "timeout/holding-risk Stage 1 week-aware aggregate",
    ),
    InputSpec(
        "timeout_stage1_weekaware_calibration",
        "timeout_holding_risk_stage1_weekaware_v1/timeout_holding_risk_calibration_deciles.csv",
        "timeout/holding-risk Stage 1 week-aware calibration deciles",
    ),
    InputSpec(
        "joint_path_timeout_aggregate",
        "utility_path_timeout_joint_risk/source_utility_path_timeout_risk_aggregate.csv",
        "joint utility/path/timeout-risk aggregate",
    ),
    InputSpec(
        "archetype_scorecard",
        "source_archetypes_v2/source_archetypes_v2_scorecard.csv",
        "source archetype v2 scorecard",
    ),
    InputSpec(
        "archetype_quality",
        "source_archetypes_v2/source_archetypes_v2_quality.csv",
        "source archetype v2 quality table",
    ),
    InputSpec(
        "candidate_selected_rows",
        "utility_risk_gate_candidate_weekly/candidate_selected_rows.csv",
        "shortlisted risk-gate selected rows",
    ),
    InputSpec(
        "path_risk_selected_rows",
        "utility_path_risk_dual_head/source_utility_path_risk_dual_head_selected_rows.csv",
        "utility/path-risk selected rows",
    ),
    InputSpec(
        "joint_path_timeout_selected_rows",
        "utility_path_timeout_joint_risk/source_utility_path_timeout_risk_selected_rows.csv",
        "joint utility/path/timeout-risk selected rows",
    ),
    InputSpec(
        "conditional_gmm_selected_features",
        "conditional_gmm_feature_selection/conditional_selected_features.csv",
        "conditional GMM selected features",
    ),
    InputSpec(
        "conditional_gmm_selected_pairs",
        "conditional_gmm_feature_selection/conditional_selected_feature_target_pairs.csv",
        "conditional GMM selected feature-target pairs",
    ),
    InputSpec(
        "conditional_gmm_signature_columns",
        "conditional_gmm_feature_selection/conditional_gmm_signature_columns.csv",
        "conditional GMM signature columns",
    ),
)

ALT_INPUT_RELPATHS = {
    "source_quality_aggregate": (
        "source_quality_label_walkforward_ablation/source_quality_label_walkforward_aggregate.csv",
    ),
    "source_quality_monthly": (
        "source_quality_label_walkforward_ablation/source_quality_label_walkforward_monthly.csv",
    ),
}


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _finite(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    number = _finite(value)
    if not math.isfinite(number):
        return ""
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    headers = [label for _, label in columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        values = [_fmt(row.get(key)) for key, _ in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _read_inputs(input_dir: Path) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    frames: dict[str, pd.DataFrame] = {}
    manifest_rows: list[dict[str, Any]] = []
    missing: list[Path] = []
    for spec in (*REQUIRED_INPUTS, *OPTIONAL_INPUTS):
        candidates = [input_dir / spec.relpath]
        candidates.extend(input_dir / relpath for relpath in ALT_INPUT_RELPATHS.get(spec.key, ()))
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        exists = path.exists()
        row: dict[str, Any] = {
            "artifact": spec.description,
            "path": str(path),
            "status": "present" if exists else "missing",
            "rows": "",
            "columns": "",
        }
        if exists:
            frame = pd.read_csv(path)
            frames[spec.key] = frame
            row["rows"] = len(frame)
            row["columns"] = len(frame.columns)
        elif spec in REQUIRED_INPUTS:
            missing.append(path)
        manifest_rows.append(row)
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(str(p) for p in missing))
    return frames, manifest_rows


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _format_side_counts(counts: Any) -> str:
    if not isinstance(counts, dict):
        return ""
    ordered_keys = [key for key in ("long", "short") if key in counts]
    ordered_keys.extend(sorted(key for key in counts if key not in {"long", "short"}))
    return ", ".join(f"{key}: {_fmt(counts.get(key))}" for key in ordered_keys)


def _side_coverage_blocker_sentence(audit: dict[str, Any] | None) -> str:
    if not audit or bool(audit.get("bidirectional_evidence_ready")):
        return ""
    registry = audit.get("registry_summary", {})
    if not isinstance(registry, dict):
        registry = {}
    long_rows = _fmt(registry.get("total_long_rows", ""))
    short_rows = _fmt(registry.get("total_short_rows", ""))
    bidirectional_registries = _fmt(registry.get("bidirectional_registries", ""))
    blocking = audit.get("blocking_artifacts") or []
    blocking_roles = [
        str(item.get("role"))
        for item in blocking
        if isinstance(item, dict) and str(item.get("role", "")).strip()
    ]
    if blocking_roles:
        role_text = ", ".join(f"`{role}`" for role in blocking_roles[:5])
        more = len(blocking_roles) - 5
        if more > 0:
            role_text += f", plus {more} more"
        return (
            " Bidirectional side evidence is also incomplete: strategy-registry scaffolding "
            f"now has {bidirectional_registries or '0'} bidirectional registries "
            f"({long_rows or '0'} long rows, {short_rows or '0'} short rows), but required "
            f"diagnostic ledgers remain long-only: {role_text}. Short-side performance must be "
            "materialized and propagated into the label/report bundle before promotion."
        )
    return (
        " Bidirectional side evidence is also missing: scanned strategy registries "
        f"currently contain {long_rows or '0'} long rows and {short_rows or '0'} short rows, "
        "so short-side performance must be materialized from real short candidates before promotion."
    )


def _render_side_coverage_section(audit: dict[str, Any] | None) -> list[str]:
    if not audit:
        return []
    registry = audit.get("registry_summary", {})
    if not isinstance(registry, dict):
        registry = {}
    artifact_rows: list[dict[str, Any]] = []
    for artifact in audit.get("artifacts", []):
        if not isinstance(artifact, dict):
            continue
        artifact_rows.append(
            {
                "role": artifact.get("role", ""),
                "status": artifact.get("status", ""),
                "rows": artifact.get("rows", ""),
                "side_counts": _format_side_counts(artifact.get("side_counts")),
                "top_side_share": artifact.get("top_side_share", ""),
                "bidirectional": "yes" if artifact.get("bidirectional") else "no",
                "failures": ", ".join(str(item) for item in artifact.get("failures", [])),
            }
        )
    return [
        "## Side Coverage Audit",
        "",
        f"Decision: `{audit.get('decision', '')}`. Bidirectional evidence ready: `{bool(audit.get('bidirectional_evidence_ready'))}`.",
        (
            "Strategy registries scanned: "
            f"`{_fmt(registry.get('registries', ''))}`; "
            f"with long: `{_fmt(registry.get('registries_with_long', ''))}`; "
            f"with short: `{_fmt(registry.get('registries_with_short', ''))}`; "
            f"bidirectional: `{_fmt(registry.get('bidirectional_registries', ''))}`."
        ),
        "",
        _markdown_table(
            artifact_rows,
            [
                ("role", "role"),
                ("status", "status"),
                ("rows", "rows"),
                ("side_counts", "side_counts"),
                ("top_side_share", "top_side_share"),
                ("bidirectional", "bidirectional"),
                ("failures", "failures"),
            ],
        ),
        "",
    ]


def _decision_counts(frame: pd.DataFrame) -> str:
    if "decision" not in frame.columns:
        return ""
    counts = frame["decision"].fillna("NA").astype(str).value_counts().sort_index()
    return ", ".join(f"{idx}: {int(value)}" for idx, value in counts.items())


def _top_row(frame: pd.DataFrame, sort_cols: list[str], ascending: list[bool] | None = None) -> pd.Series:
    existing = [col for col in sort_cols if col in frame.columns]
    if not existing or frame.empty:
        return pd.Series(dtype=object)
    if ascending is None:
        ascending = [False] * len(existing)
    return frame.sort_values(existing, ascending=ascending[: len(existing)], na_position="last").iloc[0]


def _source_concentration(selected_rows: pd.DataFrame | None, candidate: str) -> dict[str, Any]:
    if selected_rows is None or selected_rows.empty or "candidate" not in selected_rows.columns:
        return {}
    subset = selected_rows[selected_rows["candidate"].astype(str).eq(str(candidate))]
    if subset.empty or "primary_source_tag" not in subset.columns:
        return {}
    source_counts = subset["primary_source_tag"].fillna("NA").astype(str).value_counts()
    top_source = str(source_counts.index[0]) if len(source_counts) else ""
    top_source_share = float(source_counts.iloc[0] / len(subset)) if len(source_counts) else float("nan")
    max_week_top_source = float("nan")
    if "week_start" in subset.columns:
        week_shares = []
        for _, group in subset.groupby("week_start", dropna=False):
            vc = group["primary_source_tag"].fillna("NA").astype(str).value_counts(normalize=True)
            if len(vc):
                week_shares.append(float(vc.iloc[0]))
        if week_shares:
            max_week_top_source = max(week_shares)
    return {
        "top_source": top_source,
        "top_source_share": top_source_share,
        "max_week_top_source_share": max_week_top_source,
    }


def _side_name_series(frame: pd.DataFrame) -> pd.Series:
    if "side_name" in frame.columns:
        values = frame["side_name"].fillna("").astype(str).str.strip().str.lower()
        mapped = values.where(values.isin({"long", "short"}), "")
        if mapped.ne("").any():
            return mapped[mapped.ne("")]
    if "side" in frame.columns:
        numeric = _safe_numeric(frame["side"])
    elif "__side__" in frame.columns:
        numeric = _safe_numeric(frame["__side__"])
    else:
        return pd.Series(dtype=object)
    numeric = numeric.dropna()
    if numeric.empty:
        return pd.Series(dtype=object)
    return pd.Series(np.where(numeric < 0.0, "short", "long"), index=numeric.index, dtype=object)


def _side_concentration(selected_rows: pd.DataFrame | None, candidate: str) -> dict[str, Any]:
    if selected_rows is None or selected_rows.empty or "candidate" not in selected_rows.columns:
        return {}
    subset = selected_rows[selected_rows["candidate"].astype(str).eq(str(candidate))]
    if subset.empty:
        return {}
    side_names = _side_name_series(subset)
    if side_names.empty:
        return {}
    side_counts = side_names.value_counts()
    top_side = str(side_counts.index[0]) if len(side_counts) else ""
    top_side_share = float(side_counts.iloc[0] / len(side_names)) if len(side_counts) else float("nan")
    max_week_top_side = float("nan")
    if "week_start" in subset.columns:
        week_shares = []
        aligned = pd.DataFrame({"week_start": subset.loc[side_names.index, "week_start"], "side_name": side_names})
        for _, group in aligned.groupby("week_start", dropna=False):
            vc = group["side_name"].value_counts(normalize=True)
            if len(vc):
                week_shares.append(float(vc.iloc[0]))
        if week_shares:
            max_week_top_side = max(week_shares)
    return {
        "top_side": top_side,
        "top_side_share": top_side_share,
        "max_week_top_side_share_from_rows": max_week_top_side,
    }


def _first_available(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key, "")
        if isinstance(value, str):
            if value.strip():
                return value
        elif value is not None and math.isfinite(_finite(value)):
            return value
    return ""


def _candidate_key_from_row(row: pd.Series) -> str:
    if "candidate" in row.index and str(row.get("candidate", "")):
        return str(row["candidate"])
    if "risk_heads" in row.index:
        return (
            f"{row.get('label')}__{row.get('risk_heads')}__{row.get('feature_set')}__"
            f"{row.get('source_bucket')}__{row.get('causal_gate')}__{row.get('selection')}__"
            f"top{float(row.get('top_frac')):g}"
        )
    if "risk_target" in row.index:
        return (
            f"{row.get('label')}__{row.get('risk_target')}__{row.get('feature_set')}__"
            f"{row.get('source_bucket')}__{row.get('causal_gate')}__{row.get('selection')}__"
            f"top{float(row.get('top_frac')):g}"
        )
    return ""


def _summarize_candidates(frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    sq = frames["source_quality_aggregate"]
    util = frames["utility_label_rework"]
    risk = frames["utility_risk_gate"]
    weekly = frames["candidate_weekly_aggregate"]
    path = frames["path_risk_aggregate"]
    timeout = frames.get(
        "timeout_stage1_weekaware_aggregate",
        frames.get("timeout_stage1_aggregate", frames.get("timeout_holding_aggregate", pd.DataFrame())),
    )
    joint = frames.get("joint_path_timeout_aggregate", pd.DataFrame())

    best_source_quality = _top_row(sq, ["mean_u", "worst_month_mean_u"])
    best_utility = _top_row(util, ["mean_model_u", "worst_model_month_u"])
    economic = risk[risk["decision"].astype(str).eq("candidate_gate_within_economic_limits")]
    best_economic_gate = _top_row(economic, ["mean_u", "worst_month_u"])
    best_weekly = _top_row(weekly, ["q25_week_u", "positive_weeks", "mean_u"])
    path_robust = path[_safe_numeric(path.get("min_selected_rows", pd.Series(dtype=float))).fillna(0) > 0]
    best_path = _top_row(path_robust, ["q25_week_u", "positive_weeks", "mean_u"])
    best_joint = pd.Series(dtype=object)
    if not joint.empty:
        joint_robust = joint[_safe_numeric(joint.get("min_selected_rows", pd.Series(dtype=float))).fillna(0) > 0]
        best_joint = _top_row(joint_robust, ["q25_week_u", "positive_weeks", "mean_u"])
    best_timeout = pd.Series(dtype=object)
    if not timeout.empty:
        low_risk = timeout[timeout["decision"].astype(str).eq("candidate_timeout_filter")]
        best_timeout = _top_row(low_risk, ["timeout_reduction_frac_vs_valid", "score_ic_timeout"])

    candidate_selected = frames.get("candidate_selected_rows")
    path_selected = frames.get("path_risk_selected_rows")
    joint_selected = frames.get("joint_path_timeout_selected_rows")

    out = {
        "best_source_quality": best_source_quality.to_dict(),
        "best_utility": best_utility.to_dict(),
        "best_economic_gate": best_economic_gate.to_dict(),
        "best_weekly": best_weekly.to_dict(),
        "best_path": best_path.to_dict(),
        "best_joint": best_joint.to_dict(),
        "best_timeout": best_timeout.to_dict(),
    }
    for name, selected in (
        ("best_weekly", candidate_selected),
        ("best_path", path_selected),
        ("best_joint", joint_selected),
    ):
        candidate = _candidate_key_from_row(pd.Series(out[name]))
        out[name].update(_source_concentration(selected, candidate))
        out[name].update(_side_concentration(selected, candidate))
    return out


def _summary_rows(candidates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    best_weekly = candidates["best_weekly"]
    best_path = candidates["best_path"]
    best_joint = candidates["best_joint"]
    best_economic = candidates["best_economic_gate"]
    rows: list[dict[str, Any]] = [
        {
            "scope": "Best monthly/economic gate",
            "label": best_economic.get("label", ""),
            "bucket": best_economic.get("source_bucket", ""),
            "gate": best_economic.get("risk_gate", ""),
            "top_frac": best_economic.get("top_frac", ""),
            "mean_u": best_economic.get("mean_u", ""),
            "worst_month": best_economic.get("worst_month_u", ""),
            "bad_mae": best_economic.get("bad_mae_1r_rate", ""),
            "timeout": best_economic.get("timeout_rate", ""),
            "wide": best_economic.get("wide_barrier_25bps_rate", ""),
            "rows": best_economic.get("mean_selected_rows", ""),
            "decision": best_economic.get("decision", ""),
        },
        {
            "scope": "Best weekly candidate",
            "label": best_weekly.get("label", ""),
            "bucket": best_weekly.get("source_bucket", ""),
            "gate": best_weekly.get("risk_gate", ""),
            "top_frac": best_weekly.get("top_frac", ""),
            "mean_u": best_weekly.get("mean_u", ""),
            "worst_week": best_weekly.get("worst_week_u", ""),
            "q25_week": best_weekly.get("q25_week_u", ""),
            "pos_weeks": f"{int(best_weekly.get('positive_weeks', 0))}/{int(best_weekly.get('weeks', 0))}",
            "bad_mae": best_weekly.get("mean_bad_mae_1r_rate", ""),
            "timeout": best_weekly.get("mean_timeout_rate", ""),
            "max_symbol": best_weekly.get("max_top_symbol_share", ""),
            "max_side": _first_available(
                best_weekly,
                ("max_side_top_share", "max_week_side_top_share", "max_week_top_side_share_from_rows", "top_side_share"),
            ),
            "top_side": best_weekly.get("top_side", ""),
            "top_side_share": best_weekly.get("top_side_share", ""),
            "top_source": best_weekly.get("top_source", ""),
            "top_source_share": best_weekly.get("top_source_share", ""),
        },
        {
            "scope": "Best path-risk candidate",
            "label": best_path.get("label", ""),
            "bucket": best_path.get("source_bucket", ""),
            "gate": best_path.get("causal_gate", ""),
            "top_frac": best_path.get("top_frac", ""),
            "mean_u": best_path.get("mean_u", ""),
            "worst_week": best_path.get("worst_week_u", ""),
            "q25_week": best_path.get("q25_week_u", ""),
            "pos_weeks": f"{int(best_path.get('positive_weeks', 0))}/{int(best_path.get('weeks', 0))}",
            "bad_mae": best_path.get("bad_mae_1r_rate", ""),
            "timeout": best_path.get("timeout_rate", ""),
            "max_symbol": best_path.get("max_top_symbol_share", ""),
            "max_side": _first_available(
                best_path,
                ("max_week_side_top_share", "max_side_top_share", "max_week_top_side_share_from_rows", "top_side_share"),
            ),
            "top_side": best_path.get("top_side", ""),
            "top_side_share": best_path.get("top_side_share", ""),
            "top_source": best_path.get("top_source", ""),
            "top_source_share": best_path.get("top_source_share", ""),
            "decision": best_path.get("decision", ""),
        },
        {
            "scope": "Best joint path/timeout candidate",
            "label": best_joint.get("label", ""),
            "bucket": best_joint.get("source_bucket", ""),
            "gate": best_joint.get("causal_gate", ""),
            "top_frac": best_joint.get("top_frac", ""),
            "mean_u": best_joint.get("mean_u", ""),
            "worst_month": best_joint.get("worst_month_u", ""),
            "worst_week": best_joint.get("worst_week_u", ""),
            "q25_week": best_joint.get("q25_week_u", ""),
            "pos_weeks": f"{int(best_joint.get('positive_weeks', 0))}/{int(best_joint.get('weeks', 0))}",
            "bad_mae": best_joint.get("bad_mae_1r_rate", ""),
            "timeout": best_joint.get("timeout_rate", ""),
            "wide": best_joint.get("wide_barrier_25bps_rate", ""),
            "max_symbol": best_joint.get("max_week_top_symbol_share", ""),
            "max_side": _first_available(
                best_joint,
                ("max_week_side_top_share", "overall_side_top_share", "max_week_top_side_share_from_rows", "top_side_share"),
            ),
            "top_side": best_joint.get("top_side", ""),
            "top_side_share": best_joint.get("top_side_share", ""),
            "top_source": best_joint.get("top_source", ""),
            "top_source_share": best_joint.get("top_source_share", ""),
            "decision": best_joint.get("decision", ""),
        },
    ]
    return rows


def _timeout_stage1_rows(frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    stage1 = frames.get("timeout_stage1_weekaware_aggregate", frames.get("timeout_stage1_aggregate"))
    if stage1 is None or stage1.empty:
        return []
    candidates = stage1[
        stage1["selector"].astype(str).isin(["low_risk_keep", "low_risk_keep_weekly"])
    ].copy()
    if "fraction" in candidates.columns:
        half = candidates[_safe_numeric(candidates["fraction"]).eq(0.5)]
        if not half.empty:
            candidates = half
    if candidates.empty:
        return []
    candidates = candidates.sort_values(
        ["timeout_reduction_frac_vs_valid", "top_risk_decile_timeout_lift", "target_auc"],
        ascending=False,
        na_position="last",
        kind="mergesort",
    ).head(6)
    rows: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        checks = {
            "top_risk_lift": _finite(row.get("top_risk_decile_timeout_lift")) >= 1.5,
            "low_timeout": _finite(row.get("timeout_rate")) <= 0.20,
            "utility": _finite(row.get("delta_mean_u_vs_valid")) >= -0.001,
            "q25_week": _finite(row.get("q25_week_u_delta_vs_valid")) > 0.0,
            "positive_weeks": int(_finite(row.get("positive_weeks")) if math.isfinite(_finite(row.get("positive_weeks"))) else -1)
            >= int(_finite(row.get("valid_positive_weeks")) if math.isfinite(_finite(row.get("valid_positive_weeks"))) else 10**9),
            "rows_week": _finite(row.get("min_week_selected_rows")) >= 5.0,
        }
        fail_reasons = [name for name, ok in checks.items() if not ok]
        rows.append(
            {
                "gate": "pass" if not fail_reasons else "fail",
                "fail_reasons": ", ".join(fail_reasons),
                "label": row.get("label", ""),
                "feature_set": row.get("feature_set", ""),
                "selector": row.get("selector", ""),
                "fraction": row.get("fraction", ""),
                "AUC": row.get("target_auc", ""),
                "Brier": row.get("target_brier_score", ""),
                "top_decile_timeout_lift": row.get("top_risk_decile_timeout_lift", ""),
                "timeout_reduction": row.get("timeout_reduction_frac_vs_valid", ""),
                "timeout": row.get("timeout_rate", ""),
                "valid_timeout": row.get("valid_timeout_rate", ""),
                "delta_u": row.get("delta_mean_u_vs_valid", ""),
                "q25_week_delta": row.get("q25_week_u_delta_vs_valid", ""),
                "positive_weeks": row.get("positive_weeks", ""),
                "valid_positive_weeks": row.get("valid_positive_weeks", ""),
                "min_week_rows": row.get("min_week_selected_rows", ""),
                "max_side": _first_available(
                    row.to_dict(),
                    ("max_week_side_top_share", "side_top_share", "overall_side_top_share"),
                ),
            }
        )
    return rows


def _feature_selection_rows(frames: dict[str, pd.DataFrame], *, limit: int = 12) -> list[dict[str, Any]]:
    selected = frames.get("conditional_gmm_selected_features")
    if selected is None or selected.empty:
        return []
    sort_cols = [col for col in ("max_pair_score", "selected_pair_count") if col in selected.columns]
    if sort_cols:
        selected = selected.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
    keep = [
        "feature",
        "family",
        "max_pair_score",
        "selected_pair_count",
        "targets",
        "primary_categories",
        "lookback_hours",
        "horizon_relevance",
    ]
    return selected.head(int(limit))[[col for col in keep if col in selected.columns]].to_dict("records")


def _render_report(input_dir: Path, frames: dict[str, pd.DataFrame], manifest_rows: list[dict[str, Any]]) -> str:
    candidates = _summarize_candidates(frames)
    side_audit = _read_json_if_exists(input_dir / SIDE_COVERAGE_AUDIT_JSON)
    side_blocker = _side_coverage_blocker_sentence(side_audit)
    decision_rows = []
    for key in (
        "source_quality_aggregate",
        "utility_label_rework",
        "utility_risk_gate",
        "candidate_weekly_aggregate",
        "path_risk_aggregate",
        "timeout_holding_aggregate",
        "timeout_stage1_aggregate",
        "timeout_stage1_weekaware_aggregate",
        "joint_path_timeout_aggregate",
    ):
        frame = frames.get(key)
        if frame is not None:
            decision_rows.append({"artifact": key, "decision_counts": _decision_counts(frame) or "NA"})

    learn = frames["feature_learnability"]
    learn_counts = learn["dominant_diagnosis"].fillna("NA").astype(str).value_counts().sort_index()
    learn_top = _top_row(learn, ["mean_model_top_u", "mean_model_ic_u"]).to_dict()

    archetype_rows: list[dict[str, Any]] = []
    if "archetype_scorecard" in frames:
        arch = frames["archetype_scorecard"].sort_values(
            ["economic_distinction_score", "mean_utility"],
            ascending=False,
            na_position="last",
        )
        keep = [
            "decision",
            "archetype",
            "rows",
            "coverage_joined",
            "mean_utility",
            "bad_mae_1r_rate",
            "timeout_rate",
            "wide_barrier_25bps_rate",
            "top_symbol_share",
            "top_side_share",
            "economic_distinction_score",
        ]
        archetype_rows = arch.head(7)[[c for c in keep if c in arch.columns]].to_dict("records")

    best_timeout = candidates["best_timeout"]
    timeout_stage1_rows = _timeout_stage1_rows(frames)
    feature_selection_rows = _feature_selection_rows(frames)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Current Label / Archetype Baseline",
        "",
        f"Generated: `{generated}`",
        f"Input root: `{input_dir}`",
        "",
        f"Utility horizon note: {UTILITY_HORIZON_NOTE}",
        "",
        "## Stage 0 Gate Answer",
        "",
        "Current status: diagnostic-only. The source/archetype and quality-label work has produced useful labels, gates, and risk heads, but no candidate is training-ready for `train_base`, `train_meta`, policy selection, or production inference.",
        "",
        "Current best candidate: `utility_linear_source_q80_v1` with `base_plus_source` features remains the leading family. The best monthly/economic gate is an all-rows low-barrier/high-barrier-relief gate at top 1%, while the best weekly shortlist is the risk-adjusted-capture top 10% gate-relative variant.",
        "",
        "Why it is not training-ready: the monthly/economic gate still carries high bad-MAE and timeout rates, and the weekly shortlist still has negative worst-week utility plus high concentration risk. The stronger path/timeout variants reduce some risk, but still fail on worst month/week, timeout or wide-barrier risk, and symbol/source/side concentration."
        + side_blocker,
        "",
        "Dominant failure modes: weekly instability, timeout / holding-risk, concentration, and residual bad-MAE / path risk. Concentration is tracked across symbols, sources, and side when side-aware ledgers are available.",
        "",
        *_render_side_coverage_section(side_audit),
        "## Key Candidates",
        "",
        _markdown_table(
            _summary_rows(candidates),
            [
                ("scope", "scope"),
                ("label", "label"),
                ("bucket", "bucket"),
                ("gate", "gate"),
                ("top_frac", "top_frac"),
                ("mean_u", "mean_u"),
                ("worst_month", "worst_month"),
                ("worst_week", "worst_week"),
                ("q25_week", "q25_week"),
                ("pos_weeks", "pos_weeks"),
                ("bad_mae", "bad_MAE"),
                ("timeout", "timeout"),
                ("wide", "wide25"),
                ("max_symbol", "max_symbol"),
                ("max_side", "max_side"),
                ("top_side", "top_side"),
                ("top_side_share", "top_side_share"),
                ("top_source", "top_source"),
                ("top_source_share", "top_source_share"),
                ("rows", "mean_rows"),
                ("decision", "decision"),
            ],
        ),
        "",
        "## Timeout / Holding Head Status",
        "",
        "The separate timeout / holding-risk head exists and is useful diagnostically. The strongest low-risk filters cut timeout materially and clear the top-risk timeout-lift check, but the stricter Stage 1 readiness gate still fails weekly lower-tail and/or positive-week checks.",
        "",
        _markdown_table(
            [
                {
                    "label": best_timeout.get("label", ""),
                    "kind": best_timeout.get("label_kind", ""),
                    "feature_set": best_timeout.get("feature_set", ""),
                    "selector": best_timeout.get("selector", ""),
                    "fraction": best_timeout.get("fraction", ""),
                    "score_ic_timeout": best_timeout.get("score_ic_timeout", ""),
                    "target_auc": best_timeout.get("target_auc", ""),
                    "target_brier_score": best_timeout.get("target_brier_score", ""),
                    "top_risk_decile_timeout_lift": best_timeout.get("top_risk_decile_timeout_lift", ""),
                    "timeout_reduction": best_timeout.get("timeout_reduction_frac_vs_valid", ""),
                    "mean_u": best_timeout.get("mean_u", ""),
                    "delta_mean_u": best_timeout.get("delta_mean_u_vs_valid", ""),
                    "timeout_rate": best_timeout.get("timeout_rate", ""),
                    "valid_timeout_rate": best_timeout.get("valid_timeout_rate", ""),
                    "q25_week": best_timeout.get("q25_week_u", ""),
                    "q25_week_u_delta_vs_valid": best_timeout.get("q25_week_u_delta_vs_valid", ""),
                    "worst_week": best_timeout.get("worst_week_u", ""),
                    "max_side": _first_available(
                        best_timeout,
                        ("max_week_side_top_share", "side_top_share", "overall_side_top_share"),
                    ),
                    "decision": best_timeout.get("decision", ""),
                }
            ],
            [
                ("label", "label"),
                ("kind", "kind"),
                ("feature_set", "features"),
                ("selector", "selector"),
                ("fraction", "fraction"),
                ("score_ic_timeout", "timeout_IC"),
                ("target_auc", "AUC"),
                ("target_brier_score", "Brier"),
                ("top_risk_decile_timeout_lift", "top_decile_lift"),
                ("timeout_reduction", "timeout_reduction"),
                ("mean_u", "mean_u"),
                ("delta_mean_u", "delta_u"),
                ("timeout_rate", "timeout"),
                ("valid_timeout_rate", "valid_timeout"),
                ("q25_week", "q25_week"),
                ("q25_week_u_delta_vs_valid", "q25_delta"),
                ("worst_week", "worst_week"),
                ("max_side", "max_side"),
                ("decision", "decision"),
            ],
        ),
        "",
        "## Stage 1 Timeout Readiness",
        "",
        _markdown_table(
            timeout_stage1_rows,
            [
                ("gate", "gate"),
                ("fail_reasons", "fail_reasons"),
                ("label", "label"),
                ("feature_set", "features"),
                ("selector", "selector"),
                ("fraction", "fraction"),
                ("AUC", "AUC"),
                ("Brier", "Brier"),
                ("top_decile_timeout_lift", "top_decile_lift"),
                ("timeout_reduction", "timeout_reduction"),
                ("timeout", "timeout"),
                ("valid_timeout", "valid_timeout"),
                ("delta_u", "delta_u"),
                ("q25_week_delta", "q25_delta"),
                ("positive_weeks", "positive_weeks"),
                ("valid_positive_weeks", "valid_positive_weeks"),
                ("min_week_rows", "min_week_rows"),
                ("max_side", "max_side"),
            ],
        ),
        "",
        "## Learnability",
        "",
        "Feature and target learnability remains the main blocker: labels are often learnable, but the learned ranking does not reliably select utility-positive rows.",
        "",
        "Diagnosis counts: "
        + ", ".join(f"`{idx}`: {int(value)}" for idx, value in learn_counts.items()),
        "",
        _markdown_table(
            [
                {
                    "recommendation": learn_top.get("recommendation", ""),
                    "ablation": learn_top.get("ablation", ""),
                    "source_bucket": learn_top.get("source_bucket", ""),
                    "months": learn_top.get("months", ""),
                    "mean_valid_rows": learn_top.get("mean_valid_rows", ""),
                    "target_ic_pos": learn_top.get("target_ic_positive_months", ""),
                    "oracle_pos": learn_top.get("oracle_top_positive_months", ""),
                    "model_pos": learn_top.get("model_top_positive_months", ""),
                    "mean_target_ic_u": learn_top.get("mean_target_ic_u", ""),
                    "mean_model_ic_u": learn_top.get("mean_model_ic_u", ""),
                    "oracle_top_u": learn_top.get("mean_oracle_top_u", ""),
                    "model_top_u": learn_top.get("mean_model_top_u", ""),
                    "gap": learn_top.get("mean_model_vs_oracle_top_u_gap", ""),
                    "diagnosis": learn_top.get("dominant_diagnosis", ""),
                }
            ],
            [
                ("recommendation", "recommendation"),
                ("ablation", "ablation"),
                ("source_bucket", "source_bucket"),
                ("months", "months"),
                ("mean_valid_rows", "valid_rows"),
                ("target_ic_pos", "target_IC_pos"),
                ("oracle_pos", "oracle_pos"),
                ("model_pos", "model_pos"),
                ("mean_target_ic_u", "target_IC_u"),
                ("mean_model_ic_u", "model_IC_u"),
                ("oracle_top_u", "oracle_top_u"),
                ("model_top_u", "model_top_u"),
                ("gap", "model_oracle_gap"),
                ("diagnosis", "diagnosis"),
            ],
        ),
        "",
        "## Conditional GMM Feature Selection",
        "",
        "Dedicated feature selection is present and uses existing columns only. It scores feature-target pairs for 3-7h side-aware candidates while allowing causal trailing lookbacks longer than the target horizon.",
        "",
        _markdown_table(
            feature_selection_rows,
            [
                ("feature", "feature"),
                ("family", "family"),
                ("max_pair_score", "score"),
                ("selected_pair_count", "pairs"),
                ("targets", "targets"),
                ("primary_categories", "categories"),
                ("lookback_hours", "lookback_h"),
                ("horizon_relevance", "horizon_rel"),
            ],
        ),
        "",
        "## Archetype Scorecard",
        "",
        _markdown_table(
            archetype_rows,
            [
                ("decision", "decision"),
                ("archetype", "archetype"),
                ("rows", "rows"),
                ("coverage_joined", "coverage"),
                ("mean_utility", "mean_u"),
                ("bad_mae_1r_rate", "bad_MAE"),
                ("timeout_rate", "timeout"),
                ("wide_barrier_25bps_rate", "wide25"),
                ("top_symbol_share", "top_symbol"),
                ("top_side_share", "top_side"),
                ("economic_distinction_score", "distinction"),
            ],
        ),
        "",
        "## Evidence Inventory",
        "",
        _markdown_table(
            manifest_rows,
            [
                ("artifact", "artifact"),
                ("status", "status"),
                ("rows", "rows"),
                ("columns", "columns"),
                ("path", "path"),
            ],
        ),
        "",
        "## Decision Counts",
        "",
        _markdown_table(
            decision_rows,
            [("artifact", "artifact"), ("decision_counts", "decision_counts")],
        ),
        "",
        "## Stage 1 Direction",
        "",
        "Continue Stage 1 work, but keep it diagnostic-only until a candidate simultaneously clears weekly lower-tail utility, timeout/holding risk, bad-MAE/path risk, symbol/source/side concentration, positive-week retention, and minimum weekly-row depth. The current timeout head proves the risk signal is learnable; the missing piece is using it without worsening weekly stability or thinning selected weeks.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames, manifest_rows = _read_inputs(args.input_dir)
    report = _render_report(args.input_dir, frames, manifest_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
