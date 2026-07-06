#!/usr/bin/env python3
"""Expanding monthly selector for contextual TP/SL variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


PROFILE_RULES: Dict[str, Dict[str, float]] = {
    "raw_objective": {},
    "balanced_tail": {
        "min_positive_week_share": 0.55,
        "max_mean_day_full_sl_delta": 0.0,
        "max_mean_week_full_sl_delta": 0.0,
        "min_q20_day_delta_net_pnl": -250.0,
    },
    "production_pre_oos": {
        "min_positive_week_share": 0.60,
        "max_mean_day_full_sl_delta": 0.0,
        "max_mean_week_full_sl_delta": 0.0,
        "min_q20_day_delta_net_pnl": -250.0,
    },
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _prepare_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["day_ts"] = pd.to_datetime(df["day"], utc=True)
    df["month"] = df["day_ts"].dt.to_period("M").astype(str)
    return df


def _prepare_weekly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ends = df["week"].astype(str).str.split("/", expand=True)[1]
    df["week_end_ts"] = pd.to_datetime(ends, utc=True)
    df["month"] = df["week_end_ts"].dt.to_period("M").astype(str)
    return df


def _objective(daily: pd.DataFrame, weekly: pd.DataFrame) -> Dict[str, Any]:
    if daily.empty or weekly.empty:
        return {
            "daily_weekly_objective": -np.inf,
            "avg_week_delta_net_pnl": np.nan,
            "q35_day_delta_net_pnl": np.nan,
            "q20_day_delta_net_pnl": np.nan,
            "sum_delta_net_pnl": np.nan,
            "positive_week_share": 0.0,
            "positive_week_count": 0,
            "positive_day_share": 0.0,
            "positive_day_count": 0,
            "mean_day_full_sl_delta": np.nan,
            "mean_week_full_sl_delta": np.nan,
            "mean_day_hit_rate_delta": np.nan,
            "mean_week_hit_rate_delta": np.nan,
            "sum_delta_trades": 0,
        }
    day_delta = daily["delta_net_pnl"].astype(float)
    week_delta = weekly["delta_net_pnl"].astype(float)
    q35 = float(day_delta.quantile(0.35))
    q20 = float(day_delta.quantile(0.20))
    avg_week = float(week_delta.mean())
    return {
        "daily_weekly_objective": float(avg_week + 0.7 * q35 + 0.3 * q20),
        "avg_week_delta_net_pnl": avg_week,
        "q35_day_delta_net_pnl": q35,
        "q20_day_delta_net_pnl": q20,
        "sum_delta_net_pnl": float(day_delta.sum()),
        "positive_week_share": float((week_delta > 0).mean()),
        "positive_week_count": int((week_delta > 0).sum()),
        "positive_day_share": float((day_delta > 0).mean()),
        "positive_day_count": int((day_delta > 0).sum()),
        "mean_day_full_sl_delta": float(daily["delta_full_sl_rate"].mean()),
        "mean_week_full_sl_delta": float(weekly["delta_full_sl_rate"].mean()),
        "mean_day_hit_rate_delta": float(daily["delta_hit_rate"].mean()),
        "mean_week_hit_rate_delta": float(weekly["delta_hit_rate"].mean()),
        "sum_delta_trades": int(daily["delta_trades"].sum()),
    }


def _summarize(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    labels: Iterable[str],
    months: List[str],
    suffix: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for label in labels:
        day_part = daily[(daily["label"] == label) & (daily["month"].isin(months))]
        week_part = weekly[(weekly["label"] == label) & (weekly["month"].isin(months))]
        row = {"label": label, f"months_{suffix}": ",".join(months)}
        row.update({f"{k}_{suffix}": v for k, v in _objective(day_part, week_part).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _passes(row: pd.Series, rules: Dict[str, float]) -> Tuple[bool, str]:
    failures: List[str] = []
    checks = {
        "positive_week_share": (row["positive_week_share"], ">=", rules.get("min_positive_week_share", -np.inf)),
        "mean_day_full_sl_delta": (row["mean_day_full_sl_delta"], "<=", rules.get("max_mean_day_full_sl_delta", np.inf)),
        "mean_week_full_sl_delta": (row["mean_week_full_sl_delta"], "<=", rules.get("max_mean_week_full_sl_delta", np.inf)),
        "q20_day_delta_net_pnl": (row["q20_day_delta_net_pnl"], ">=", rules.get("min_q20_day_delta_net_pnl", -np.inf)),
    }
    for name, (value, op, threshold) in checks.items():
        if not np.isfinite(float(threshold)):
            continue
        ok = value >= threshold if op == ">=" else value <= threshold
        if not ok:
            failures.append(name)
    return not failures, ",".join(failures)


def _select_candidate(train_summary: pd.DataFrame, profile: str, fallback_label: str) -> Tuple[str, str]:
    rules = PROFILE_RULES[profile]
    ranked = train_summary.copy()
    pass_values: List[bool] = []
    fail_values: List[str] = []
    for _, row in ranked.iterrows():
        normalized = row.rename(lambda c: c.removesuffix("_train"))
        ok, failures = _passes(normalized, rules)
        pass_values.append(ok)
        fail_values.append(failures)
    ranked["passes"] = pass_values
    ranked["failures"] = fail_values
    ranked = ranked.sort_values(
        ["passes", "daily_weekly_objective_train", "sum_delta_net_pnl_train"],
        ascending=[False, False, False],
    )
    passed = ranked[ranked["passes"]]
    if passed.empty:
        return fallback_label, "fallback_no_train_candidate_passed"
    row = passed.iloc[0]
    return str(row["label"]), str(row["failures"])


def _month_sequence(daily: pd.DataFrame) -> List[str]:
    return sorted(daily["month"].dropna().unique())


def _markdown_table(df: pd.DataFrame, columns: List[str]) -> str:
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df[columns].iterrows():
        vals: List[str] = []
        for value in row:
            if isinstance(value, float):
                vals.append(f"{value:.6g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-csv", required=True)
    parser.add_argument("--weekly-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--baseline-label", default="wf_recent")
    parser.add_argument(
        "--fallback-label",
        default="baseline",
        help="Candidate to use when a profile has no passing train candidate. Use 'baseline' to do nothing.",
    )
    parser.add_argument("--min-train-months", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    daily = _prepare_daily(Path(args.daily_csv))
    weekly = _prepare_weekly(Path(args.weekly_csv))
    labels = sorted(set(daily["label"]) - {args.baseline_label})
    months = _month_sequence(daily)

    eval_rows: List[Dict[str, Any]] = []
    static_rows: List[Dict[str, Any]] = []
    for eval_idx in range(args.min_train_months, len(months)):
        train_months = months[:eval_idx]
        eval_month = months[eval_idx]
        train_summary = _summarize(daily, weekly, labels, train_months, "train")
        eval_summary = _summarize(daily, weekly, labels, [eval_month], "eval")
        eval_best = eval_summary.sort_values(
            ["daily_weekly_objective_eval", "sum_delta_net_pnl_eval"], ascending=False
        ).iloc[0]
        for label in labels:
            static_row = eval_summary[eval_summary["label"] == label].iloc[0].to_dict()
            static_row["eval_month"] = eval_month
            static_rows.append(static_row)
        for profile in PROFILE_RULES:
            selected_label, reason = _select_candidate(train_summary, profile, args.fallback_label)
            if selected_label == "baseline":
                eval_rows.append(
                    {
                        "profile": profile,
                        "eval_month": eval_month,
                        "train_months": ",".join(train_months),
                        "selected_label": selected_label,
                        "selection_reason": reason,
                        "fallback_label": args.fallback_label,
                        "eval_objective": 0.0,
                        "eval_sum_delta_net_pnl": 0.0,
                        "eval_positive_week_share": np.nan,
                        "eval_q20_day_delta_net_pnl": 0.0,
                        "eval_mean_day_full_sl_delta": 0.0,
                        "eval_best_label": eval_best["label"],
                        "eval_best_objective": eval_best["daily_weekly_objective_eval"],
                        "eval_regret": eval_best["daily_weekly_objective_eval"],
                    }
                )
                continue
            result = eval_summary[eval_summary["label"] == selected_label].iloc[0]
            eval_rows.append(
                {
                    "profile": profile,
                    "eval_month": eval_month,
                    "train_months": ",".join(train_months),
                    "selected_label": selected_label,
                    "selection_reason": reason,
                    "fallback_label": args.fallback_label,
                    "eval_objective": result["daily_weekly_objective_eval"],
                    "eval_sum_delta_net_pnl": result["sum_delta_net_pnl_eval"],
                    "eval_positive_week_share": result["positive_week_share_eval"],
                    "eval_q20_day_delta_net_pnl": result["q20_day_delta_net_pnl_eval"],
                    "eval_mean_day_full_sl_delta": result["mean_day_full_sl_delta_eval"],
                    "eval_best_label": eval_best["label"],
                    "eval_best_objective": eval_best["daily_weekly_objective_eval"],
                    "eval_regret": eval_best["daily_weekly_objective_eval"] - result["daily_weekly_objective_eval"],
                }
            )

    eval_df = pd.DataFrame(eval_rows)
    static_df = pd.DataFrame(static_rows)
    profile_summary = (
        eval_df.groupby("profile", as_index=False)
        .agg(
            eval_months=("eval_month", "count"),
            selected_labels=("selected_label", lambda s: ",".join(s.astype(str))),
            positive_eval_months=("eval_sum_delta_net_pnl", lambda s: int((s > 0).sum())),
            sum_eval_delta_net_pnl=("eval_sum_delta_net_pnl", "sum"),
            mean_eval_objective=("eval_objective", "mean"),
            q20_eval_objective=("eval_objective", lambda s: float(s.quantile(0.20))),
            mean_eval_regret=("eval_regret", "mean"),
            mean_eval_day_full_sl_delta=("eval_mean_day_full_sl_delta", "mean"),
        )
    )
    static_summary = (
        static_df.groupby("label", as_index=False)
        .agg(
            eval_months=("eval_month", "count"),
            positive_eval_months=("sum_delta_net_pnl_eval", lambda s: int((s > 0).sum())),
            sum_eval_delta_net_pnl=("sum_delta_net_pnl_eval", "sum"),
            mean_eval_objective=("daily_weekly_objective_eval", "mean"),
            q20_eval_objective=("daily_weekly_objective_eval", lambda s: float(s.quantile(0.20))),
            mean_eval_day_full_sl_delta=("mean_day_full_sl_delta_eval", "mean"),
        )
        .sort_values(["mean_eval_objective", "sum_eval_delta_net_pnl"], ascending=False)
    )

    eval_df.to_csv(out_dir / "monthly_selector_results.csv", index=False)
    profile_summary.to_csv(out_dir / "monthly_selector_profile_summary.csv", index=False)
    static_df.to_csv(out_dir / "monthly_selector_static_variant_results.csv", index=False)
    static_summary.to_csv(out_dir / "monthly_selector_static_variant_summary.csv", index=False)
    (out_dir / "monthly_selector_manifest.json").write_text(
        json.dumps(
            _json_safe(
                {
                    "daily_csv": str(args.daily_csv),
                    "weekly_csv": str(args.weekly_csv),
                    "min_train_months": args.min_train_months,
                    "fallback_label": args.fallback_label,
                    "profiles": PROFILE_RULES,
                    "months": months,
                    "objective": "avg_week_delta_net_pnl + 0.7*q35_day_delta_net_pnl + 0.3*q20_day_delta_net_pnl",
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    md = [
        "# Contextual TP/SL Monthly Selector Validation",
        "",
        "This is an expanding monthly selector over existing Jan-Jun replay metrics. It selects from prior months only, then evaluates the next month. It does not rerun portfolio simulation and is not untouched OOS.",
        "",
        "Objective: `avg_week_delta_net_pnl + 0.7*q35_day_delta_net_pnl + 0.3*q20_day_delta_net_pnl`.",
        "",
        "## Dynamic Profile Summary",
        "",
        _markdown_table(
            profile_summary,
            [
                "profile",
                "eval_months",
                "selected_labels",
                "positive_eval_months",
                "sum_eval_delta_net_pnl",
                "mean_eval_objective",
                "q20_eval_objective",
                "mean_eval_regret",
                "mean_eval_day_full_sl_delta",
            ],
        ),
        "",
        "## Static Variant Summary On Same Eval Months",
        "",
        _markdown_table(
            static_summary,
            [
                "label",
                "eval_months",
                "positive_eval_months",
                "sum_eval_delta_net_pnl",
                "mean_eval_objective",
                "q20_eval_objective",
                "mean_eval_day_full_sl_delta",
            ],
        ),
        "",
        "## Monthly Decisions",
        "",
        _markdown_table(
            eval_df,
            [
                "profile",
                "eval_month",
                "selected_label",
                "eval_objective",
                "eval_sum_delta_net_pnl",
                "eval_q20_day_delta_net_pnl",
                "eval_mean_day_full_sl_delta",
                "eval_best_label",
                "eval_regret",
            ],
        ),
        "",
    ]
    (out_dir / "monthly_selector_report.md").write_text("\n".join(md), encoding="utf-8")
    print(out_dir / "monthly_selector_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
