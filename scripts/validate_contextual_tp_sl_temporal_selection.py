#!/usr/bin/env python3
"""Temporal selection validation for contextual TP/SL variants.

Select variants on earlier months, then evaluate the selected variant on later
months using the daily-tail / weekly-PnL objective.
"""

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


def _month_range(start: str, end: str) -> List[str]:
    return [str(period) for period in pd.period_range(start=start, end=end, freq="M")]


def _parse_splits(spec: str) -> List[Tuple[str, str, str, str]]:
    splits: List[Tuple[str, str, str, str]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        train_spec, eval_spec = item.split(":", 1)
        if len(train_spec) != 15 or train_spec[7] != "-":
            raise ValueError(f"Expected YYYY-MM-YYYY-MM train range, got {train_spec!r}")
        if len(eval_spec) != 15 or eval_spec[7] != "-":
            raise ValueError(f"Expected YYYY-MM-YYYY-MM eval range, got {eval_spec!r}")
        train_start, train_end = train_spec[:7], train_spec[8:]
        eval_start, eval_end = eval_spec[:7], eval_spec[8:]
        splits.append((train_start, train_end, eval_start, eval_end))
    return splits


def _prepare_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["day_ts"] = pd.to_datetime(df["day"], utc=True)
    df["month"] = df["day_ts"].dt.to_period("M").astype(str)
    return df


def _prepare_weekly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    starts = df["week"].astype(str).str.split("/", expand=True)[0]
    ends = df["week"].astype(str).str.split("/", expand=True)[1]
    df["week_start_ts"] = pd.to_datetime(starts, utc=True)
    df["week_end_ts"] = pd.to_datetime(ends, utc=True)
    # Assign cross-month weeks to their ending month because most realized PnL
    # in this replay period belongs to the post-start days.
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


def _summarize_window(
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


def _select_for_profile(train_summary: pd.DataFrame, profile: str) -> Tuple[pd.Series | None, pd.DataFrame]:
    rules = PROFILE_RULES[profile]
    ranked = train_summary.copy()
    pass_values: List[bool] = []
    fail_values: List[str] = []
    for _, row in ranked.iterrows():
        # Strip train suffix for profile checks.
        normalized = row.rename(lambda c: c.removesuffix("_train"))
        ok, failures = _passes(normalized, rules)
        pass_values.append(ok)
        fail_values.append(failures)
    ranked["profile"] = profile
    ranked["passes_train_profile"] = pass_values
    ranked["train_fail_reasons"] = fail_values
    ranked = ranked.sort_values(
        ["passes_train_profile", "daily_weekly_objective_train", "sum_delta_net_pnl_train"],
        ascending=[False, False, False],
    )
    passed = ranked[ranked["passes_train_profile"]]
    return (passed.iloc[0] if len(passed) else None), ranked


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
        "--splits",
        default="2026-01-2026-03:2026-04-2026-06,2026-01-2026-04:2026-05-2026-06,2026-01-2026-05:2026-06-2026-06",
        help="Comma-separated train_start-train_end:eval_start-eval_end month ranges.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    daily = _prepare_daily(Path(args.daily_csv))
    weekly = _prepare_weekly(Path(args.weekly_csv))
    labels = sorted(set(daily["label"]) - {args.baseline_label})

    selection_rows: List[Dict[str, Any]] = []
    ranking_rows: List[Dict[str, Any]] = []
    for train_start, train_end, eval_start, eval_end in _parse_splits(args.splits):
        train_months = _month_range(train_start, train_end)
        eval_months = _month_range(eval_start, eval_end)
        train_summary = _summarize_window(daily, weekly, labels, train_months, "train")
        eval_summary = _summarize_window(daily, weekly, labels, eval_months, "eval")
        merged = train_summary.merge(eval_summary, on="label", how="left")
        eval_best = merged.sort_values(["daily_weekly_objective_eval", "sum_delta_net_pnl_eval"], ascending=False).iloc[0]
        for profile in PROFILE_RULES:
            selected, ranked = _select_for_profile(train_summary, profile)
            ranked["train_window"] = f"{train_start}:{train_end}"
            ranked["eval_window"] = f"{eval_start}:{eval_end}"
            ranking_rows.extend(ranked.to_dict("records"))
            if selected is None:
                selection_rows.append(
                    {
                        "profile": profile,
                        "train_window": f"{train_start}:{train_end}",
                        "eval_window": f"{eval_start}:{eval_end}",
                        "selected_label": "baseline",
                        "selection_reason": "no_train_candidate_passed",
                    }
                )
                continue
            selected_label = str(selected["label"])
            result = merged[merged["label"] == selected_label].iloc[0].to_dict()
            row = {
                "profile": profile,
                "train_window": f"{train_start}:{train_end}",
                "eval_window": f"{eval_start}:{eval_end}",
                "selected_label": selected_label,
                "train_objective": result["daily_weekly_objective_train"],
                "train_sum_delta_net_pnl": result["sum_delta_net_pnl_train"],
                "train_positive_week_share": result["positive_week_share_train"],
                "eval_objective": result["daily_weekly_objective_eval"],
                "eval_sum_delta_net_pnl": result["sum_delta_net_pnl_eval"],
                "eval_positive_week_share": result["positive_week_share_eval"],
                "eval_q20_day_delta_net_pnl": result["q20_day_delta_net_pnl_eval"],
                "eval_mean_day_full_sl_delta": result["mean_day_full_sl_delta_eval"],
                "eval_mean_week_full_sl_delta": result["mean_week_full_sl_delta_eval"],
                "eval_best_label": eval_best["label"],
                "eval_best_objective": eval_best["daily_weekly_objective_eval"],
                "eval_best_sum_delta_net_pnl": eval_best["sum_delta_net_pnl_eval"],
                "eval_objective_regret": eval_best["daily_weekly_objective_eval"] - result["daily_weekly_objective_eval"],
            }
            selection_rows.append(row)

    selections = pd.DataFrame(selection_rows)
    rankings = pd.DataFrame(ranking_rows)
    selections.to_csv(out_dir / "temporal_selection_results.csv", index=False)
    rankings.to_csv(out_dir / "temporal_train_rankings.csv", index=False)
    summary = (
        selections[selections["selected_label"] != "baseline"]
        .groupby("profile", as_index=False)
        .agg(
            splits=("selected_label", "count"),
            positive_eval_splits=("eval_sum_delta_net_pnl", lambda s: int((s > 0).sum())),
            mean_eval_objective=("eval_objective", "mean"),
            mean_eval_sum_delta_net_pnl=("eval_sum_delta_net_pnl", "mean"),
            mean_eval_regret=("eval_objective_regret", "mean"),
            selected_labels=("selected_label", lambda s: ",".join(s.astype(str))),
        )
    )
    summary.to_csv(out_dir / "temporal_selection_profile_summary.csv", index=False)
    (out_dir / "temporal_selection_manifest.json").write_text(
        json.dumps(
            _json_safe(
                {
                    "daily_csv": str(args.daily_csv),
                    "weekly_csv": str(args.weekly_csv),
                    "splits": args.splits,
                    "profiles": PROFILE_RULES,
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
        "# Contextual TP/SL Temporal Selection Validation",
        "",
        "This validates selection stability using existing Jan-Jun development replay metrics. It selects variants on earlier months and evaluates on later months. It does not rerun portfolio simulation and is not untouched OOS.",
        "",
        "Objective: `avg_week_delta_net_pnl + 0.7*q35_day_delta_net_pnl + 0.3*q20_day_delta_net_pnl`.",
        "",
        "## Profile Summary",
        "",
        _markdown_table(
            summary,
            [
                "profile",
                "splits",
                "positive_eval_splits",
                "mean_eval_objective",
                "mean_eval_sum_delta_net_pnl",
                "mean_eval_regret",
                "selected_labels",
            ],
        ),
        "",
        "## Split Results",
        "",
        _markdown_table(
            selections,
            [
                "profile",
                "train_window",
                "eval_window",
                "selected_label",
                "train_objective",
                "eval_objective",
                "eval_sum_delta_net_pnl",
                "eval_positive_week_share",
                "eval_q20_day_delta_net_pnl",
                "eval_best_label",
                "eval_objective_regret",
            ],
        ),
        "",
    ]
    (out_dir / "temporal_selection_report.md").write_text("\n".join(md), encoding="utf-8")
    print(out_dir / "temporal_selection_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
