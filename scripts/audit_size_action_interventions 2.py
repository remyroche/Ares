#!/usr/bin/env python3
"""Audit selected size-action interventions for sparse policy learners."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_BASE_ARM = "C3ed_bagged_safety_c3ea_or_pressure_opportunity_action_tail_veto_union_gate"
DEFAULT_CANDIDATE_ARM = "C3ej_bagged_safety_c3ed_or_high_value_zero_classifier_union_gate"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _active_schedule_rows(schedules: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    if schedules.empty:
        return pd.DataFrame()
    work = schedules.loc[schedules["arm"].isin(arms)].copy()
    work["multiplier_num"] = pd.to_numeric(work.get("multiplier"), errors="coerce").fillna(1.0)
    work = work.loc[work["multiplier_num"] < 1.0].copy()
    if work.empty:
        return work
    work["intervention_key"] = work["timestamp"].astype(str) + "|" + work["strategy_id"].astype(str)
    return work


def _selected_action_labels(run_dir: Path, active: pd.DataFrame) -> pd.DataFrame:
    if active.empty:
        return active.copy()
    label_cols = [
        "timestamp",
        "strategy_id",
        "multiplier",
        "fold_id",
        "split",
        "delta_full_J",
        "delta_immediate_J",
        "delta_full_net_pnl",
        "delta_full_cost_pnl",
        "delta_full_turnover",
        "best_multiplier",
        "best_gain",
        "best_margin",
        "best_gain_per_notional",
        "best_margin_per_notional",
        "best_nonbaseline_gain",
        "worst_nonbaseline_gain",
        "group_can_bind",
        "y_intervene",
    ]
    panel_path = run_dir / "size_action_exact_panel.csv"
    if not panel_path.exists():
        return active.copy()
    panel = pd.read_csv(panel_path, usecols=lambda col: col in set(label_cols))
    if panel.empty:
        return active.copy()
    panel["multiplier_num"] = pd.to_numeric(panel.get("multiplier"), errors="coerce")
    return active.merge(
        panel,
        on=["timestamp", "strategy_id", "fold_id", "multiplier_num"],
        how="left",
        suffixes=("", "_label"),
    )


def _build_selected_action_audit(run_dir: Path, base_arm: str, candidate_arm: str) -> pd.DataFrame:
    schedules = _read_csv(run_dir / "size_action_schedules.csv")
    active = _active_schedule_rows(schedules, [base_arm, candidate_arm])
    if active.empty:
        return active
    base_keys = set(active.loc[active["arm"].eq(base_arm), "intervention_key"])
    active["selection_class"] = np.where(
        active["arm"].eq(base_arm),
        "base_active",
        np.where(active["intervention_key"].isin(base_keys), "shared_with_base", "candidate_added"),
    )
    merged = _selected_action_labels(run_dir, active)
    ordered_cols = [
        "arm",
        "selection_class",
        "fold_id",
        "timestamp",
        "strategy_id",
        "multiplier_num",
        "union_preferred_source",
        "selection_score",
        "zero_cut_classifier_score",
        "p_intervene",
        "pred_delta_J",
        "cal_q25_delta_J",
        "cal_positive_rate",
        "p_action_positive",
        "eligible_action",
        "secondary_acceptance_pass",
        "nonoverlap_suppressed",
        "delta_full_J",
        "delta_full_net_pnl",
        "delta_immediate_J",
        "delta_full_cost_pnl",
        "delta_full_turnover",
        "best_multiplier",
        "best_gain",
        "best_margin",
        "best_nonbaseline_gain",
        "worst_nonbaseline_gain",
        "y_intervene",
    ]
    return merged[[col for col in ordered_cols if col in merged.columns]].sort_values(
        ["arm", "fold_id", "timestamp", "strategy_id"]
    )


def _summarize_selected_actions(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    work = selected.copy()
    for col in ["delta_full_J", "delta_full_net_pnl", "delta_immediate_J"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    group_cols = ["arm", "selection_class"]
    if "union_preferred_source" in work.columns:
        group_cols.append("union_preferred_source")
    return (
        work.groupby(group_cols, dropna=False)
        .agg(
            actions=("timestamp", "size"),
            folds=("fold_id", "nunique"),
            positive_label=("delta_full_J", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) > 0).sum())),
            delta_full_J_sum=("delta_full_J", "sum"),
            delta_net_pnl_sum=("delta_full_net_pnl", "sum"),
            delta_immediate_J_sum=("delta_immediate_J", "sum"),
            median_delta_net=("delta_full_net_pnl", "median"),
            q25_delta_net=("delta_full_net_pnl", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.25))),
            min_delta_net=("delta_full_net_pnl", "min"),
        )
        .reset_index()
    )


def _candidate_fold_quality(run_dir: Path, candidate_arm: str) -> pd.DataFrame:
    quality = _read_csv(run_dir / "size_action_action_quality.csv")
    if quality.empty or "arm" not in quality.columns:
        return pd.DataFrame()
    return quality.loc[quality["arm"].eq(candidate_arm)].copy()


def _candidate_replay_label_audit(run_dir: Path, candidate_arm: str) -> pd.DataFrame:
    audit = _read_csv(run_dir / "size_action_replay_vs_label_audit.csv")
    if audit.empty or "arm" not in audit.columns:
        return pd.DataFrame()
    return audit.loc[audit["arm"].eq(candidate_arm)].copy()


def _promotion_snapshot(comparison_dir: Path, base_arm: str, candidate_arm: str) -> pd.DataFrame:
    if comparison_dir is None:
        return pd.DataFrame()
    path = comparison_dir / "size_action_promotion_gate_comparison.csv"
    gates = _read_csv(path)
    if gates.empty or "arm" not in gates.columns:
        return pd.DataFrame()
    return gates.loc[gates["arm"].isin([base_arm, candidate_arm])].copy()


def _write_markdown(
    out_path: Path,
    promotion: pd.DataFrame,
    selected_summary: pd.DataFrame,
    fold_quality: pd.DataFrame,
    replay_audit: pd.DataFrame,
) -> None:
    lines: list[str] = ["# Size-Action Intervention Audit", ""]
    if not promotion.empty:
        promotion = promotion.copy()
        if "failed_gates" in promotion.columns:
            promotion["failed_gates"] = promotion["failed_gates"].fillna("")
        cols = [
            "run",
            "arm",
            "interventions",
            "positive_actions",
            "precision_total",
            "realized_delta_full_J_sum",
            "delta_net_pnl_sum",
            "positive_delta_folds",
            "selected_false_groups",
            "selected_false_delta_sum",
            "binding_intervention_rate_total",
            "mean_oracle_capture",
            "promotion_ready",
            "failed_gates",
        ]
        lines.extend(["## Promotion Gate Snapshot", "", promotion[[c for c in cols if c in promotion.columns]].to_markdown(index=False), ""])
    if not selected_summary.empty:
        lines.extend(["## Selected Action Summary", "", selected_summary.to_markdown(index=False), ""])
    if not fold_quality.empty:
        cols = [
            "fold_id",
            "scheduled_groups",
            "intervention_count",
            "positive_action_count",
            "positive_action_rate",
            "realized_delta_full_J_sum",
            "realized_delta_full_net_pnl_sum",
            "oracle_positive_group_capture_rate",
            "oracle_gain_capture_ratio",
        ]
        lines.extend(["## Candidate Fold Quality", "", fold_quality[[c for c in cols if c in fold_quality.columns]].to_markdown(index=False), ""])
    if not replay_audit.empty:
        cols = [
            "fold_id",
            "delta_net_pnl",
            "realized_delta_full_J_sum",
            "realized_delta_full_net_pnl_sum",
            "replay_minus_label_net_pnl",
            "sequential_replay_positive",
            "independent_label_positive",
            "sequential_replay_disagrees_with_label",
        ]
        lines.extend(["## Replay vs Independent Label", "", replay_audit[[c for c in cols if c in replay_audit.columns]].to_markdown(index=False), ""])
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-arm", default=DEFAULT_BASE_ARM)
    parser.add_argument("--candidate-arm", default=DEFAULT_CANDIDATE_ARM)
    parser.add_argument("--comparison-dir", type=Path, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = _build_selected_action_audit(args.run_dir, args.base_arm, args.candidate_arm)
    selected_summary = _summarize_selected_actions(selected)
    fold_quality = _candidate_fold_quality(args.run_dir, args.candidate_arm)
    replay_audit = _candidate_replay_label_audit(args.run_dir, args.candidate_arm)
    promotion = _promotion_snapshot(args.comparison_dir, args.base_arm, args.candidate_arm) if args.comparison_dir else pd.DataFrame()

    selected.to_csv(args.out_dir / "selected_actions.csv", index=False)
    selected_summary.to_csv(args.out_dir / "selected_action_summary.csv", index=False)
    fold_quality.to_csv(args.out_dir / "candidate_fold_quality.csv", index=False)
    replay_audit.to_csv(args.out_dir / "candidate_replay_vs_label_audit.csv", index=False)
    if not promotion.empty:
        promotion.to_csv(args.out_dir / "promotion_gate_snapshot.csv", index=False)
    _write_markdown(args.out_dir / "size_action_intervention_audit.md", promotion, selected_summary, fold_quality, replay_audit)

    print(
        {
            "out_dir": str(args.out_dir),
            "selected_rows": int(len(selected)),
            "summary_rows": int(len(selected_summary)),
            "fold_quality_rows": int(len(fold_quality)),
            "replay_audit_rows": int(len(replay_audit)),
        }
    )


if __name__ == "__main__":
    main()
