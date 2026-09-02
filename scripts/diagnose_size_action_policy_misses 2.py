#!/usr/bin/env python3
"""Diagnose exact-state size-action policy misses.

The report compares three group-level cohorts:

* selected_positive: selected non-baseline actions with positive realized full-path value
* selected_false: selected non-baseline actions with non-positive realized full-path value
* missed_oracle: confidently actionable groups that the arm left at baseline

It uses only persisted run artifacts, so it is cheap and does not rerun the
counterfactual replay engine.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["timestamp", "strategy_id", "fold_id"]
DEFAULT_ARM = "C3dv_bagged_safety_c3db_or_zero_harm_mean_recall_zero_union_gate"
MATERIAL_EPS = 1e-6


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def _numeric_columns(df: pd.DataFrame, excluded: set[str]) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            finite = np.isfinite(pd.to_numeric(df[col], errors="coerce")).sum()
            if finite:
                cols.append(col)
    return cols


def _cohort_summary(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for cohort, sub in df.groupby("cohort", dropna=False):
        row = {"cohort": cohort, "rows": int(len(sub))}
        for col in numeric_cols:
            values = pd.to_numeric(sub[col], errors="coerce")
            row[f"{col}__mean"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{col}__median"] = float(values.median()) if values.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cohort")


def _feature_diff(
    df: pd.DataFrame,
    numeric_cols: list[str],
    left: str,
    right: str,
) -> pd.DataFrame:
    left_df = df[df["cohort"] == left]
    right_df = df[df["cohort"] == right]
    rows = []
    for col in numeric_cols:
        l = pd.to_numeric(left_df[col], errors="coerce")
        r = pd.to_numeric(right_df[col], errors="coerce")
        if not l.notna().any() or not r.notna().any():
            continue
        l_mean = float(l.mean())
        r_mean = float(r.mean())
        pooled = float(np.nanstd(pd.concat([l, r], ignore_index=True), ddof=0))
        diff = l_mean - r_mean
        rows.append(
            {
                "left": left,
                "right": right,
                "feature": col,
                "left_mean": l_mean,
                "right_mean": r_mean,
                "mean_diff": diff,
                "abs_mean_diff": abs(diff),
                "std_mean_diff": diff / pooled if pooled > 0 else np.nan,
                "left_median": float(l.median()),
                "right_median": float(r.median()),
                "left_non_null": int(l.notna().sum()),
                "right_non_null": int(r.notna().sum()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["abs_mean_diff", "feature"], ascending=[False, True])


def _selected_action_values(run_dir: Path) -> pd.DataFrame:
    action_scores = _read_csv(run_dir / "size_action_eval_action_scores.csv")
    keep_cols = KEYS + [
        "multiplier",
        "delta_full_J",
        "delta_immediate_J",
        "delta_full_net_pnl",
        "delta_full_cost_pnl",
        "delta_full_J_per_notional",
        "pred_delta_J",
        "ranker_score",
        "cal_mean_delta_J",
        "cal_lcb_mean_delta_J",
        "cal_q10_delta_J",
        "cal_q25_delta_J",
        "cal_positive_rate",
        "p_action_positive",
        "p_action_economic_positive",
        "best_multiplier",
        "best_gain_x",
        "best_gain_y",
        "best_margin",
        "group_affected_notional",
        "y_intervene_x",
        "y_intervene_y",
    ]
    keep_cols = [c for c in keep_cols if c in action_scores.columns]
    return action_scores[keep_cols].copy()


def build_report(run_dir: Path, arm: str, out_dir: Path) -> dict[str, float | int | str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    group_scores = _read_csv(run_dir / "size_action_stage1_group_scores.csv")
    schedules = _read_csv(run_dir / "size_action_schedules.csv")
    action_values = _selected_action_values(run_dir)

    if "score_source" in group_scores.columns:
        group_scores["_score_source_priority"] = group_scores["score_source"].eq("bagged_stage1").astype(int)
        group_scores = (
            group_scores.sort_values(KEYS + ["_score_source_priority"], ascending=[True, True, True, False])
            .drop_duplicates(KEYS, keep="first")
            .drop(columns=["_score_source_priority"])
        )
    else:
        group_scores = group_scores.drop_duplicates(KEYS, keep="first")

    arm_schedule = schedules[schedules["arm"] == arm].copy()
    if arm_schedule.empty:
        raise ValueError(f"No schedule rows found for arm: {arm}")

    selected = arm_schedule[pd.to_numeric(arm_schedule["multiplier"], errors="coerce") < 1.0].copy()
    selected = selected.merge(
        action_values,
        on=KEYS + ["multiplier"],
        how="left",
        suffixes=("", "_action"),
    )
    selected_keys = selected[KEYS].drop_duplicates()

    group_scores = group_scores.merge(
        selected_keys.assign(selected_by_arm=True),
        on=KEYS,
        how="left",
    )
    group_scores["selected_by_arm"] = group_scores["selected_by_arm"].eq(True)

    actionable = (
        (pd.to_numeric(group_scores.get("y_intervene", 0), errors="coerce") > 0)
        & (pd.to_numeric(group_scores.get("group_can_bind", 0), errors="coerce") > 0)
    )
    missed = group_scores[actionable & ~group_scores["selected_by_arm"]].copy()
    missed["cohort"] = "missed_oracle"

    selected["selected_positive"] = pd.to_numeric(selected.get("delta_full_J", 0), errors="coerce") > MATERIAL_EPS
    selected["selected_confident_actionable"] = (
        pd.to_numeric(selected.get("y_intervene_x", selected.get("y_intervene_y", 0)), errors="coerce") > 0
    )
    selected_pos = selected[selected["selected_positive"]].copy()
    selected_pos["cohort"] = "selected_positive"
    selected_false = selected[~selected["selected_positive"]].copy()
    selected_false["cohort"] = "selected_false"

    # Align group-level fields onto selected cohorts so feature diffs compare the same columns.
    group_feature_cols = [c for c in group_scores.columns if c not in {"selected_by_arm", "cohort"}]
    selected_group = pd.concat([selected_pos, selected_false], ignore_index=True)
    selected_group = selected_group[KEYS + ["cohort", "multiplier", "delta_full_J", "delta_full_net_pnl"]].merge(
        group_scores[group_feature_cols],
        on=KEYS,
        how="left",
        suffixes=("", "_group"),
    )
    missed_group = missed.copy()
    missed_group["multiplier"] = np.nan
    missed_group["delta_full_J"] = np.nan
    missed_group["delta_full_net_pnl"] = np.nan

    cohorts = pd.concat([selected_group, missed_group], ignore_index=True, sort=False)
    numeric_cols = _numeric_columns(cohorts, excluded=set(KEYS + ["selected_by_arm"]))

    summary = _cohort_summary(cohorts, numeric_cols)
    summary.to_csv(out_dir / "size_action_policy_miss_cohort_summary.csv", index=False)

    diffs = []
    for left, right in [
        ("selected_positive", "missed_oracle"),
        ("selected_false", "selected_positive"),
        ("selected_false", "missed_oracle"),
    ]:
        diff = _feature_diff(cohorts, numeric_cols, left, right)
        if not diff.empty:
            diffs.append(diff.head(80))
    diff_df = pd.concat(diffs, ignore_index=True) if diffs else pd.DataFrame()
    diff_df.to_csv(out_dir / "size_action_policy_miss_feature_diffs.csv", index=False)

    selected.to_csv(out_dir / "size_action_policy_selected_actions.csv", index=False)
    missed.to_csv(out_dir / "size_action_policy_missed_oracle_groups.csv", index=False)

    metrics = {
        "run_dir": str(run_dir),
        "arm": arm,
        "selected_actions": int(len(selected)),
        "selected_positive_actions": int(len(selected_pos)),
        "selected_false_actions": int(len(selected_false)),
        "selected_confident_actionable": int(selected["selected_confident_actionable"].sum()),
        "missed_oracle_groups": int(len(missed)),
        "selected_precision": float(len(selected_pos) / len(selected)) if len(selected) else 0.0,
        "selected_delta_full_J_sum": float(pd.to_numeric(selected.get("delta_full_J", 0), errors="coerce").sum()),
        "selected_false_delta_full_J_sum": float(
            pd.to_numeric(selected_false.get("delta_full_J", 0), errors="coerce").sum()
        ),
        "missed_oracle_gain_sum": float(pd.to_numeric(missed.get("best_gain", 0), errors="coerce").sum()),
        "missed_oracle_gain_median": float(pd.to_numeric(missed.get("best_gain", np.nan), errors="coerce").median()),
    }
    pd.DataFrame([metrics]).to_csv(out_dir / "size_action_policy_miss_metrics.csv", index=False)

    md = [
        "# Size Action Policy Miss Diagnostic",
        "",
        f"Run: `{run_dir}`",
        f"Arm: `{arm}`",
        "",
        "## Metrics",
        "",
    ]
    for key, value in metrics.items():
        md.append(f"- `{key}`: `{value}`")
    md.extend(["", "## Largest Feature Differences", ""])
    if diff_df.empty:
        md.append("No feature differences were available.")
    else:
        show = diff_df.head(30).copy()
        md.append(show.to_markdown(index=False))
    (out_dir / "size_action_policy_miss_diagnostic.md").write_text("\n".join(md) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--arm", default=DEFAULT_ARM)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    out_dir = args.out_dir or (args.run_dir / "policy_miss_diagnostic")
    metrics = build_report(args.run_dir, args.arm, out_dir)
    print(metrics)


if __name__ == "__main__":
    main()
