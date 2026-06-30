#!/usr/bin/env python3
"""Synthesize C3el diagnostics into a head-specific next-ablation plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HEAD_ORDER = {"short_asset": 0, "short_boll": 1, "long_bars": 2, "long_dist": 3}


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def _num(row: pd.Series, col: str, default: float = np.nan) -> float:
    try:
        value = float(row.get(col, default))
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _best_rule(rule_candidates: pd.DataFrame) -> dict[str, Any]:
    if rule_candidates.empty:
        return {}
    work = rule_candidates.copy()
    if "passes_min_rows" in work.columns:
        passes = work["passes_min_rows"]
        if passes.dtype != bool:
            passes = passes.astype(str).str.lower().isin({"true", "1", "yes"})
        work = work.loc[passes].copy()
    if work.empty:
        return {}
    sort_cols = [col for col in ["score", "sum_delta_full_J", "positive_share"] if col in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    row = work.iloc[0]
    return {
        "rule": str(row.get("rule", "")),
        "rows": int(_num(row, "rows", 0.0)),
        "positive_share": _num(row, "positive_share"),
        "positive_day_share": _num(row, "positive_day_share"),
        "sum_delta_full_J": _num(row, "sum_delta_full_J"),
        "worst_delta_full_J": _num(row, "worst_delta_full_J"),
        "coverage_of_strict": _num(row, "coverage_of_strict"),
    }


def _best_filter(filter_trials: pd.DataFrame, loo: pd.DataFrame) -> dict[str, Any]:
    if filter_trials.empty:
        return {}
    work = filter_trials.copy()
    if "rule_name" in work.columns:
        work = work.loc[~work["rule_name"].astype(str).eq("no_filter")].copy()
    if "objective" in work.columns:
        obj = pd.to_numeric(work["objective"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        work = work.loc[obj.notna()].assign(_objective=obj.loc[obj.notna()])
    if work.empty:
        return {}
    sort_cols = [col for col in ["_objective", "delta_full_J_sum", "positive_e50_rate"] if col in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    row = work.iloc[0]
    loo_delta = np.nan
    loo_positive_day_share = np.nan
    if not loo.empty and "delta_full_J_sum" in loo.columns:
        day_delta = pd.to_numeric(loo["delta_full_J_sum"], errors="coerce").fillna(0.0)
        loo_delta = float(day_delta.sum())
        loo_positive_day_share = float(day_delta.gt(0.0).mean()) if len(day_delta) else np.nan
    return {
        "rule_family": str(row.get("rule_family", "")),
        "rule": str(row.get("rule", "")),
        "keep_count": int(_num(row, "keep_count", 0.0)),
        "positive_e50_rate": _num(row, "positive_e50_rate"),
        "delta_full_J_sum": _num(row, "delta_full_J_sum"),
        "delta_full_J_worst": _num(row, "delta_full_J_worst"),
        "loo_delta_full_J_sum": loo_delta,
        "loo_positive_day_share": loo_positive_day_share,
    }


def _threshold_row(thresholds: pd.DataFrame, head: str) -> pd.Series | None:
    if thresholds.empty or "head" not in thresholds.columns:
        return None
    rows = thresholds.loc[thresholds["head"].astype(str).eq(str(head))].copy()
    if rows.empty:
        return None
    diagnosis_priority = {
        "holdout_positive": 0,
        "holdout_selection_negative": 1,
        "no_positive_holdout_threshold_trial": 2,
        "no_eligible_threshold_trials": 3,
        "missing_threshold_trial_artifact": 4,
    }
    rows["_diagnosis_priority"] = rows.get("diagnosis", "").astype(str).map(diagnosis_priority).fillna(9)
    rows["_eligible_trials"] = pd.to_numeric(rows.get("threshold_trial_eligible_count", 0.0), errors="coerce").fillna(0.0)
    rows["_candidate_rank"] = rows.get("candidate", "").astype(str).str.contains("default", case=False, regex=False).astype(int)
    return rows.sort_values(
        ["_diagnosis_priority", "_eligible_trials", "_candidate_rank"],
        ascending=[True, False, False],
    ).iloc[0]


def _score_gate_row(score_gates: pd.DataFrame, head: str) -> pd.Series | None:
    if score_gates.empty or "head" not in score_gates.columns:
        return None
    rows = score_gates.loc[score_gates["head"].astype(str).eq(str(head))].copy()
    if rows.empty:
        return None
    if "week_start" in rows.columns:
        all_rows = rows.loc[rows["week_start"].astype(str).eq("ALL")].copy()
        if not all_rows.empty:
            return all_rows.iloc[0]
    numeric_cols = [
        "rows",
        "score_eligible_groups",
        "guard_action_feature_min_groups",
        "gate_kept_groups",
        "max_eval_keep",
    ]
    agg: dict[str, Any] = {"head": head, "week_start": "ALL"}
    for col in numeric_cols:
        agg[col] = pd.to_numeric(rows.get(col, 0.0), errors="coerce").fillna(0.0).sum()
    agg["score_eligible_share"] = float(agg["score_eligible_groups"] / max(agg["rows"], 1.0))
    agg["gate_keep_share"] = float(agg["gate_kept_groups"] / max(agg["rows"], 1.0))
    diagnoses = rows.get("diagnosis", pd.Series("", index=rows.index)).astype(str)
    agg["diagnosis"] = diagnoses.mode().iloc[0] if not diagnoses.empty else ""
    return pd.Series(agg)


def _conditioning_row(conditioning: pd.DataFrame, head: str) -> pd.Series | None:
    if conditioning.empty:
        return None
    rows = conditioning.copy()
    if "head" in rows.columns:
        rows = rows.loc[rows["head"].astype(str).eq(str(head))].copy()
    if rows.empty:
        return None
    sort_cols = [
        col
        for col in [
            "selected_sum_delta_full_J",
            "selected_positive_week_share",
            "selected_mean_delta_full_J",
        ]
        if col in rows.columns
    ]
    if sort_cols:
        for col in sort_cols:
            rows[col] = pd.to_numeric(rows[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(-np.inf)
        rows = rows.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return rows.iloc[0]


def _conditioning_ablation_row(folds: pd.DataFrame, head: str) -> pd.Series | None:
    if folds.empty or "head" not in folds.columns:
        return None
    rows = folds.loc[folds["head"].astype(str).eq(str(head))].copy()
    if rows.empty:
        return None
    out = {
        "head": head,
        "conditioning_ablation_weeks": int(len(rows)),
        "conditioning_ablation_kept_groups": pd.to_numeric(
            rows.get("kept_eval_groups", 0.0), errors="coerce"
        ).fillna(0.0).sum(),
        "conditioning_ablation_feature_max_guarded_groups": pd.to_numeric(
            rows.get("action_feature_max_guarded_eval_groups", 0.0), errors="coerce"
        ).fillna(0.0).sum(),
        "conditioning_ablation_feature_min_guarded_groups": pd.to_numeric(
            rows.get("action_feature_min_guarded_eval_groups", 0.0), errors="coerce"
        ).fillna(0.0).sum(),
    }
    total_guarded = (
        out["conditioning_ablation_feature_max_guarded_groups"]
        + out["conditioning_ablation_feature_min_guarded_groups"]
    )
    if out["conditioning_ablation_kept_groups"] > 0 and total_guarded <= 0:
        out["conditioning_ablation_status"] = "nonbinding_feature_guard"
    elif (
        out["conditioning_ablation_feature_max_guarded_groups"] > 0
        and out["conditioning_ablation_feature_min_guarded_groups"] <= 0
    ):
        out["conditioning_ablation_status"] = "feature_max_guard_bound"
    elif (
        out["conditioning_ablation_feature_min_guarded_groups"] > 0
        and out["conditioning_ablation_feature_max_guarded_groups"] <= 0
    ):
        out["conditioning_ablation_status"] = "feature_min_guard_bound"
    elif total_guarded > 0:
        out["conditioning_ablation_status"] = "feature_guard_bound"
    else:
        out["conditioning_ablation_status"] = "no_conditioned_actions"
    out["conditioning_ablation_feature_guarded_groups"] = total_guarded
    return pd.Series(out)


def _weekly_condition_row(weekly_conditions: pd.DataFrame, head: str) -> pd.Series | None:
    if weekly_conditions.empty:
        return None
    rows = weekly_conditions.copy()
    if "head" in rows.columns:
        rows = rows.loc[rows["head"].astype(str).eq(str(head))].copy()
    if rows.empty:
        return None
    sort_cols = [
        col
        for col in [
            "selected_delta_net_pnl_sum",
            "selected_positive_week_share",
            "selected_worst_delta_net_pnl",
        ]
        if col in rows.columns
    ]
    if sort_cols:
        for col in sort_cols:
            rows[col] = pd.to_numeric(rows[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(-np.inf)
        rows = rows.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return rows.iloc[0]


def _recommend_head(
    *,
    head: str,
    label_row: pd.Series,
    threshold_row: pd.Series | None,
    score_gate_row: pd.Series | None,
    conditioning_row: pd.Series | None,
    conditioning_ablation_row: pd.Series | None,
    weekly_condition_row: pd.Series | None,
    best_rule: dict[str, Any],
    best_filter: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    label_diag = str(label_row.get("diagnosis", ""))
    threshold_diag = str(threshold_row.get("diagnosis", "")) if threshold_row is not None else "missing_threshold_diagnostic"
    score_gate_diag = str(score_gate_row.get("diagnosis", "")) if score_gate_row is not None else "missing_score_gate_diagnostic"
    current_rate = _num(label_row, "current_positive_rate")
    relaxed_rate = _num(label_row, "relaxed_full_positive_rate")
    headroom_ratio = _num(label_row, "full_gain_to_worst_abs_ratio")
    score_eligible = int(_num(score_gate_row, "score_eligible_groups", 0.0)) if score_gate_row is not None else 0
    guard_blocked = int(_num(score_gate_row, "guard_action_feature_min_groups", 0.0)) if score_gate_row is not None else 0
    gate_kept = int(_num(score_gate_row, "gate_kept_groups", 0.0)) if score_gate_row is not None else 0
    guard_block_share = float(guard_blocked / max(score_eligible, 1))
    kept_per_score_eligible = float(gate_kept / max(score_eligible, 1))
    conditioning_feature = str(conditioning_row.get("feature", "")) if conditioning_row is not None else ""
    conditioning_direction = str(conditioning_row.get("direction", "")) if conditioning_row is not None else ""
    conditioning_quantile = _num(conditioning_row, "quantile") if conditioning_row is not None else np.nan
    conditioning_sum_delta = _num(conditioning_row, "selected_sum_delta_full_J") if conditioning_row is not None else np.nan
    conditioning_positive_week_share = (
        _num(conditioning_row, "selected_positive_week_share") if conditioning_row is not None else np.nan
    )
    conditioning_ablation_status = (
        str(conditioning_ablation_row.get("conditioning_ablation_status", ""))
        if conditioning_ablation_row is not None
        else ""
    )
    conditioning_ablation_kept = (
        int(_num(conditioning_ablation_row, "conditioning_ablation_kept_groups", 0.0))
        if conditioning_ablation_row is not None
        else 0
    )
    conditioning_ablation_guarded = (
        int(_num(conditioning_ablation_row, "conditioning_ablation_feature_guarded_groups", 0.0))
        if conditioning_ablation_row is not None
        else 0
    )
    weekly_condition_feature = str(weekly_condition_row.get("feature", "")) if weekly_condition_row is not None else ""
    weekly_condition_direction = str(weekly_condition_row.get("direction", "")) if weekly_condition_row is not None else ""
    weekly_condition_threshold = _num(weekly_condition_row, "threshold") if weekly_condition_row is not None else np.nan
    weekly_condition_delta = (
        _num(weekly_condition_row, "selected_delta_net_pnl_sum") if weekly_condition_row is not None else np.nan
    )
    weekly_condition_positive_share = (
        _num(weekly_condition_row, "selected_positive_week_share") if weekly_condition_row is not None else np.nan
    )
    target_rows = int(readiness.get("unlabeled_target_rows", 0) or 0)

    priority = "P3"
    action = "monitor_only"
    rationale = []

    if label_diag == "negative_oracle_headroom":
        priority = "P9"
        action = "disable_size_action_learning_keep_diagnostic"
        rationale.append("exact-state oracle headroom is negative")
    elif head == "short_asset":
        priority = "P0"
        action = "collect_forward_labels_for_guarded_strict_rule"
        rationale.append("monitored default replay improves, but threshold holdout is negative/fallback-only")
        if best_rule:
            rationale.append(f"best strict guard `{best_rule.get('rule')}` has positive_share={best_rule.get('positive_share'):.2%}")
        if best_filter:
            rationale.append(
                f"fallback filter LOO delta={best_filter.get('loo_delta_full_J_sum'):.2f}, positive_day_share={best_filter.get('loo_positive_day_share'):.2%}"
            )
        if target_rows <= 0:
            rationale.append("forward target backlog is zero, so exact-state replay should wait")
    elif head == "short_boll" and label_diag == "usable_label_support":
        priority = "P1"
        action = "rerun_head_native_threshold_trials_short_boll_only"
        rationale.append("label support is usable, but current challenger lacks threshold-trial evidence")
        rationale.append("require non-fallback holdout-positive threshold evidence before replay/promotion")
        if threshold_diag == "holdout_selection_negative":
            action = "redesign_short_boll_action_label_or_regime_conditioning"
            rationale.append("an existing head-specific threshold-trial run has eligible trials but zero positive holdout value")
            if conditioning_feature:
                rationale.append(
                    f"best conditioning hypothesis `{conditioning_feature} {conditioning_direction} q{conditioning_quantile:.2f}` has in-sample sum_delta={conditioning_sum_delta:.2f}"
                )
            if conditioning_ablation_status == "nonbinding_feature_guard":
                rationale.append(
                    f"conditioned feature guard was non-binding ({conditioning_ablation_guarded} guarded / {conditioning_ablation_kept} kept); condition threshold/objective instead"
                )
            elif conditioning_ablation_status in {"feature_min_guard_bound", "feature_max_guard_bound", "feature_guard_bound"}:
                rationale.append(
                    f"conditioned feature guard bound ({conditioning_ablation_guarded} guarded); compare economics before moving it into threshold/objective"
                )
            if weekly_condition_feature:
                rationale.append(
                    f"best weekly condition `{weekly_condition_feature} {weekly_condition_direction} {weekly_condition_threshold:.4g}` selects weeks with delta_net_pnl={weekly_condition_delta:.2f}"
                )
        elif score_eligible > 0 and guard_block_share >= 0.5:
            action = "rerun_short_boll_threshold_trials_with_guard_grid"
            rationale.append(
                f"score gate found {score_eligible} candidates, but feature guard blocked {guard_blocked} ({guard_block_share:.2%})"
            )
        elif score_eligible > 0 and kept_per_score_eligible <= 0.15:
            rationale.append(
                f"only {gate_kept}/{score_eligible} score-qualified groups survived final gates; inspect cap/guard settings"
            )
    elif head == "long_bars" and label_diag == "usable_label_support":
        priority = "P2"
        action = "diagnostic_threshold_trial_only_until_portfolio_case_exists"
        rationale.append("label support is usable, but portfolio/head activation case is not established")
    else:
        priority = "P4"
        action = "shadow_monitor_only"
        rationale.append(f"label diagnosis is `{label_diag}`")

    if label_diag == "sparse_low_precision_headroom":
        rationale.append("do not broadly relax labels; positive headroom is small versus negative tail")
    if threshold_diag in {"holdout_selection_negative", "missing_threshold_trial_artifact"}:
        rationale.append(f"threshold diagnosis is `{threshold_diag}`")

    return {
        "head": head,
        "priority": priority,
        "recommended_action": action,
        "label_diagnosis": label_diag,
        "threshold_diagnosis": threshold_diag,
        "score_gate_diagnosis": score_gate_diag,
        "score_eligible_groups": score_eligible,
        "feature_guard_blocked_groups": guard_blocked,
        "gate_kept_groups": gate_kept,
        "feature_guard_block_share": guard_block_share,
        "kept_per_score_eligible": kept_per_score_eligible,
        "best_conditioning_feature": conditioning_feature,
        "best_conditioning_direction": conditioning_direction,
        "best_conditioning_quantile": conditioning_quantile,
        "best_conditioning_sum_delta_full_J": conditioning_sum_delta,
        "best_conditioning_positive_week_share": conditioning_positive_week_share,
        "conditioning_ablation_status": conditioning_ablation_status,
        "conditioning_ablation_kept_groups": conditioning_ablation_kept,
        "conditioning_ablation_guarded_groups": conditioning_ablation_guarded,
        "best_weekly_condition_feature": weekly_condition_feature,
        "best_weekly_condition_direction": weekly_condition_direction,
        "best_weekly_condition_threshold": weekly_condition_threshold,
        "best_weekly_condition_delta_net_pnl": weekly_condition_delta,
        "best_weekly_condition_positive_share": weekly_condition_positive_share,
        "current_positive_rate": current_rate,
        "relaxed_full_positive_rate": relaxed_rate,
        "full_gain_to_worst_abs_ratio": headroom_ratio,
        "best_guard_rule": best_rule.get("rule", "") if head == "short_asset" else "",
        "best_guard_rows": best_rule.get("rows", np.nan) if head == "short_asset" else np.nan,
        "best_guard_sum_delta_full_J": best_rule.get("sum_delta_full_J", np.nan) if head == "short_asset" else np.nan,
        "best_filter_loo_delta_full_J": best_filter.get("loo_delta_full_J_sum", np.nan) if head == "short_asset" else np.nan,
        "readiness_unlabeled_target_rows": target_rows,
        "rationale": " | ".join(rationale),
    }


def build_plan(
    *,
    label_objectives: Path,
    threshold_diagnostics: Path,
    score_gate_diagnostics: Path | None = None,
    conditioning_slices: Path | None = None,
    conditioning_ablation_folds: Path | None = None,
    weekly_conditions: Path | None = None,
    rule_candidates: Path | None = None,
    fallback_filter_dir: Path | None = None,
    readiness_manifest: Path | None = None,
) -> pd.DataFrame:
    labels = _read_frame(label_objectives)
    thresholds = _read_frame(threshold_diagnostics) if threshold_diagnostics.exists() else pd.DataFrame()
    score_gates = _read_frame(score_gate_diagnostics) if score_gate_diagnostics and score_gate_diagnostics.exists() else pd.DataFrame()
    conditioning = _read_frame(conditioning_slices) if conditioning_slices and conditioning_slices.exists() else pd.DataFrame()
    conditioning_ablation = (
        _read_frame(conditioning_ablation_folds)
        if conditioning_ablation_folds and conditioning_ablation_folds.exists()
        else pd.DataFrame()
    )
    weekly_conditions_frame = (
        _read_frame(weekly_conditions) if weekly_conditions and weekly_conditions.exists() else pd.DataFrame()
    )
    rules = _read_frame(rule_candidates) if rule_candidates and rule_candidates.exists() else pd.DataFrame()
    best_rule = _best_rule(rules)
    best_filter: dict[str, Any] = {}
    if fallback_filter_dir is not None and fallback_filter_dir.exists():
        all_trials = fallback_filter_dir / "all_filter_trials.csv"
        loo = fallback_filter_dir / "leave_one_day_filter_validation.csv"
        best_filter = _best_filter(
            _read_frame(all_trials) if all_trials.exists() else pd.DataFrame(),
            _read_frame(loo) if loo.exists() else pd.DataFrame(),
        )
    readiness = _read_json(readiness_manifest)
    rows = []
    for label_row in labels.itertuples(index=False):
        row = pd.Series(label_row._asdict())
        head = str(row.get("head", ""))
        rows.append(
            _recommend_head(
                head=head,
                label_row=row,
                threshold_row=_threshold_row(thresholds, head),
                score_gate_row=_score_gate_row(score_gates, head),
                conditioning_row=_conditioning_row(conditioning, head),
                conditioning_ablation_row=_conditioning_ablation_row(conditioning_ablation, head),
                weekly_condition_row=_weekly_condition_row(weekly_conditions_frame, head),
                best_rule=best_rule,
                best_filter=best_filter,
                readiness=readiness,
            )
        )
    out = pd.DataFrame(rows)
    out["_priority_order"] = out["priority"].str.extract(r"P(\\d+)").astype(float).fillna(99.0)
    out["_head_order"] = out["head"].map(HEAD_ORDER).fillna(99).astype(int)
    return out.sort_values(["_priority_order", "_head_order"]).drop(columns=["_priority_order", "_head_order"]).reset_index(drop=True)


def _write_markdown(path: Path, plan: pd.DataFrame) -> None:
    lines = [
        "# C3el next-ablation plan",
        "",
        "This report consolidates label-objective, threshold-selection, strict-rule, fallback-filter, and readiness diagnostics into one head-specific plan.",
        "",
    ]
    if plan.empty:
        lines.append("No plan rows.")
    else:
        display_cols = [
            "head",
            "priority",
            "recommended_action",
            "label_diagnosis",
            "threshold_diagnosis",
            "score_gate_diagnosis",
            "score_eligible_groups",
            "feature_guard_blocked_groups",
            "gate_kept_groups",
            "best_conditioning_feature",
            "best_conditioning_direction",
            "best_conditioning_quantile",
            "best_conditioning_sum_delta_full_J",
            "best_conditioning_positive_week_share",
            "conditioning_ablation_status",
            "conditioning_ablation_kept_groups",
            "conditioning_ablation_guarded_groups",
            "best_weekly_condition_feature",
            "best_weekly_condition_direction",
            "best_weekly_condition_threshold",
            "best_weekly_condition_delta_net_pnl",
            "best_weekly_condition_positive_share",
            "current_positive_rate",
            "relaxed_full_positive_rate",
            "full_gain_to_worst_abs_ratio",
            "best_guard_rule",
            "best_guard_sum_delta_full_J",
            "best_filter_loo_delta_full_J",
            "readiness_unlabeled_target_rows",
            "rationale",
        ]
        lines.append(plan[display_cols].to_markdown(index=False, floatfmt=".4f"))
    lines.extend(
        [
            "",
            "## Decision Rules",
            "",
            "- P0: collect or monitor exact-state labels for the guarded short_asset rule; do not replay until forward targets exist.",
            "- P1: if eligible holdout threshold trials are missing, materialize them; if they are holdout-negative, redesign the action label or condition it on observable regime state before replay.",
            "- P2: diagnostic only until the portfolio case for the head is re-established.",
            "- P9: disable size-action learning for this head unless new labels overturn negative oracle headroom.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-objectives", type=Path, required=True)
    parser.add_argument("--threshold-diagnostics", type=Path, required=True)
    parser.add_argument("--score-gate-diagnostics", type=Path)
    parser.add_argument("--conditioning-slices", type=Path)
    parser.add_argument("--conditioning-ablation-folds", type=Path)
    parser.add_argument("--weekly-conditions", type=Path)
    parser.add_argument("--rule-candidates", type=Path)
    parser.add_argument("--fallback-filter-dir", type=Path)
    parser.add_argument("--readiness-manifest", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(
        label_objectives=args.label_objectives,
        threshold_diagnostics=args.threshold_diagnostics,
        score_gate_diagnostics=args.score_gate_diagnostics,
        conditioning_slices=args.conditioning_slices,
        conditioning_ablation_folds=args.conditioning_ablation_folds,
        weekly_conditions=args.weekly_conditions,
        rule_candidates=args.rule_candidates,
        fallback_filter_dir=args.fallback_filter_dir,
        readiness_manifest=args.readiness_manifest,
    )
    plan.to_csv(args.out_dir / "c3el_next_ablation_plan.csv", index=False)
    _write_markdown(args.out_dir / "summary.md", plan)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "generated_by": "report_c3el_next_ablation_plan",
                "label_objectives": str(args.label_objectives),
                "threshold_diagnostics": str(args.threshold_diagnostics),
                "score_gate_diagnostics": str(args.score_gate_diagnostics) if args.score_gate_diagnostics else None,
                "conditioning_slices": str(args.conditioning_slices) if args.conditioning_slices else None,
                "conditioning_ablation_folds": str(args.conditioning_ablation_folds)
                if args.conditioning_ablation_folds
                else None,
                "weekly_conditions": str(args.weekly_conditions) if args.weekly_conditions else None,
                "rule_candidates": str(args.rule_candidates) if args.rule_candidates else None,
                "fallback_filter_dir": str(args.fallback_filter_dir) if args.fallback_filter_dir else None,
                "readiness_manifest": str(args.readiness_manifest) if args.readiness_manifest else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(plan.to_string(index=False))


if __name__ == "__main__":
    main()
