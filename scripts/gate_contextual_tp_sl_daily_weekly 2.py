#!/usr/bin/env python3
"""Gate contextual TP/SL variants with weekly PnL and daily-tail objective."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


FAMILY_BY_LABEL = {
    "shortasset_uncertainty_only": "uncertainty",
    "shortasset_drift_only": "drift",
    "shortasset_ood_only": "ood",
    "longbars_uncertainty_only": "uncertainty",
    "longbars_drift_only": "drift",
    "longbars_ood_only": "ood",
    "longbars_weekgate_only": "recent_hr_surprise",
    "combined": "uncertainty_plus_recent_hr_surprise",
}


PROFILE_RULES: Dict[str, Dict[str, float]] = {
    "pnl_tail_objective": {
        "min_daily_weekly_objective": 0.0,
        "min_positive_week_share": 0.55,
    },
    "balanced_tail": {
        "min_daily_weekly_objective": 0.0,
        "min_positive_week_share": 0.55,
        "max_mean_day_full_sl_delta": 0.0,
        "max_mean_week_full_sl_delta": 0.0,
        "min_q20_day_delta_net_pnl": -250.0,
    },
    "production_pre_oos": {
        "min_daily_weekly_objective": 0.0,
        "min_positive_week_share": 0.60,
        "max_mean_day_full_sl_delta": 0.0,
        "max_mean_week_full_sl_delta": 0.0,
        "min_q20_day_delta_net_pnl": -250.0,
        "min_june_net_delta": 0.0,
        "max_june_full_sl_delta": 0.0,
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


def _weighted_rate(group: pd.DataFrame, column: str) -> float:
    trades = group["trades"].astype(float).to_numpy()
    values = group[column].astype(float).to_numpy()
    mask = np.isfinite(trades) & np.isfinite(values) & (trades > 0)
    denom = trades[mask].sum()
    if denom <= 0:
        return np.nan
    return float(np.sum(values[mask] * trades[mask]) / denom)


def _read_period(path: Path, label: str, frequency: str) -> pd.DataFrame:
    file_name = f"combo_replay_{frequency}_metrics.csv"
    period_col = "day" if frequency == "daily" else "week"
    period_type = "day" if frequency == "daily" else "week"
    metrics_path = path / file_name
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {frequency} metrics: {metrics_path}")
    df = pd.read_csv(metrics_path)
    df = df[(df["period_type"] == period_type) & (df["head"].isna())].copy()
    df["label"] = label
    return df[["label", period_col, "net_pnl", "gross_pnl", "trades", "hit_rate", "full_sl_rate", "timeout_rate"]]


def _load_periods(global_df: pd.DataFrame, labels: Iterable[str], frequency: str) -> pd.DataFrame:
    period_col = "day" if frequency == "daily" else "week"
    rows: List[pd.DataFrame] = []
    wanted = set(labels)
    for _, record in global_df[global_df["label"].isin(wanted)].iterrows():
        rows.append(_read_period(Path(str(record["path"])), str(record["label"]), frequency))
    raw = pd.concat(rows, ignore_index=True)
    grouped_rows: List[Dict[str, Any]] = []
    for (label, period), group in raw.groupby(["label", period_col], sort=True):
        grouped_rows.append(
            {
                "label": label,
                period_col: period,
                "net_pnl": float(group["net_pnl"].sum()),
                "gross_pnl": float(group["gross_pnl"].sum()),
                "trades": int(group["trades"].sum()),
                "hit_rate": _weighted_rate(group, "hit_rate"),
                "full_sl_rate": _weighted_rate(group, "full_sl_rate"),
                "timeout_rate": _weighted_rate(group, "timeout_rate"),
            }
        )
    return pd.DataFrame(grouped_rows)


def _complete_period_grid(df: pd.DataFrame, labels: List[str], period_col: str) -> pd.DataFrame:
    periods = sorted(df[period_col].dropna().unique())
    grid = pd.MultiIndex.from_product([labels, periods], names=["label", period_col]).to_frame(index=False)
    merged = grid.merge(df, on=["label", period_col], how="left")
    for column in ["net_pnl", "gross_pnl", "trades"]:
        merged[column] = merged[column].fillna(0.0)
    return merged


def _add_deltas(df: pd.DataFrame, period_col: str, baseline_label: str) -> pd.DataFrame:
    baseline = df[df["label"] == baseline_label].rename(
        columns={
            "net_pnl": "net_pnl_baseline",
            "gross_pnl": "gross_pnl_baseline",
            "trades": "trades_baseline",
            "hit_rate": "hit_rate_baseline",
            "full_sl_rate": "full_sl_rate_baseline",
            "timeout_rate": "timeout_rate_baseline",
        }
    )
    merged = df.merge(
        baseline[
            [
                period_col,
                "net_pnl_baseline",
                "gross_pnl_baseline",
                "trades_baseline",
                "hit_rate_baseline",
                "full_sl_rate_baseline",
                "timeout_rate_baseline",
            ]
        ],
        on=period_col,
        how="left",
    )
    for column in ["net_pnl", "gross_pnl", "trades", "hit_rate", "full_sl_rate", "timeout_rate"]:
        merged[f"delta_{column}"] = merged[column] - merged[f"{column}_baseline"]
    return merged


def _summarize(global_df: pd.DataFrame, daily: pd.DataFrame, weekly: pd.DataFrame, baseline_label: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for label in sorted(set(global_df["label"]) - {baseline_label}):
        day_group = daily[daily["label"] == label]
        week_group = weekly[weekly["label"] == label]
        month_group = global_df[global_df["label"] == label]
        day_delta = day_group["delta_net_pnl"].astype(float)
        week_delta = week_group["delta_net_pnl"].astype(float)
        june = month_group[month_group["month"].astype(str) == "2026-06"]
        objective = float(week_delta.mean() + 0.7 * day_delta.quantile(0.35) + 0.3 * day_delta.quantile(0.20))
        rows.append(
            {
                "label": label,
                "diagnostic_family": FAMILY_BY_LABEL.get(label, "unknown"),
                "daily_weekly_objective": objective,
                "avg_week_delta_net_pnl": float(week_delta.mean()),
                "q35_day_delta_net_pnl": float(day_delta.quantile(0.35)),
                "q20_day_delta_net_pnl": float(day_delta.quantile(0.20)),
                "sum_delta_net_pnl": float(month_group["delta_net_pnl"].sum()),
                "positive_week_share": float((week_delta > 0).mean()),
                "positive_week_count": int((week_delta > 0).sum()),
                "positive_day_share": float((day_delta > 0).mean()),
                "positive_day_count": int((day_delta > 0).sum()),
                "mean_day_full_sl_delta": float(day_group["delta_full_sl_rate"].mean()),
                "mean_week_full_sl_delta": float(week_group["delta_full_sl_rate"].mean()),
                "mean_day_hit_rate_delta": float(day_group["delta_hit_rate"].mean()),
                "mean_week_hit_rate_delta": float(week_group["delta_hit_rate"].mean()),
                "june_net_delta": float(june["delta_net_pnl"].iloc[0]) if len(june) else np.nan,
                "june_full_sl_delta": float(june["delta_full_sl_rate"].iloc[0]) if len(june) else np.nan,
                "june_drawdown_delta": float(june["delta_max_drawdown"].iloc[0]) if len(june) else np.nan,
                "sum_delta_trades": int(month_group["delta_trade_count"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _passes(row: pd.Series, rules: Dict[str, float]) -> tuple[bool, str]:
    failures: List[str] = []
    checks = {
        "daily_weekly_objective": (row["daily_weekly_objective"], ">=", rules.get("min_daily_weekly_objective", -np.inf)),
        "positive_week_share": (row["positive_week_share"], ">=", rules.get("min_positive_week_share", -np.inf)),
        "mean_day_full_sl_delta": (row["mean_day_full_sl_delta"], "<=", rules.get("max_mean_day_full_sl_delta", np.inf)),
        "mean_week_full_sl_delta": (row["mean_week_full_sl_delta"], "<=", rules.get("max_mean_week_full_sl_delta", np.inf)),
        "q20_day_delta_net_pnl": (row["q20_day_delta_net_pnl"], ">=", rules.get("min_q20_day_delta_net_pnl", -np.inf)),
        "june_net_delta": (row["june_net_delta"], ">=", rules.get("min_june_net_delta", -np.inf)),
        "june_full_sl_delta": (row["june_full_sl_delta"], "<=", rules.get("max_june_full_sl_delta", np.inf)),
    }
    for name, (value, op, threshold) in checks.items():
        if not np.isfinite(float(threshold)):
            continue
        ok = value >= threshold if op == ">=" else value <= threshold
        if not ok:
            failures.append(name)
    return not failures, ",".join(failures)


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
    parser.add_argument("--global-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--baseline-label", default="wf_recent")
    args = parser.parse_args()

    global_df = pd.read_csv(args.global_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted(global_df["label"].unique())
    daily_raw = _load_periods(global_df, labels, "daily")
    weekly_raw = _load_periods(global_df, labels, "weekly")
    daily = _add_deltas(_complete_period_grid(daily_raw, labels, "day"), "day", args.baseline_label)
    weekly = _add_deltas(_complete_period_grid(weekly_raw, labels, "week"), "week", args.baseline_label)
    summary = _summarize(global_df, daily, weekly, args.baseline_label)
    summary = summary.sort_values(["daily_weekly_objective", "sum_delta_net_pnl"], ascending=False).reset_index(drop=True)
    summary["objective_rank"] = np.arange(1, len(summary) + 1)

    profile_rows: List[Dict[str, Any]] = []
    champions: Dict[str, Any] = {}
    for profile, rules in PROFILE_RULES.items():
        ranked = summary.copy()
        pass_values: List[bool] = []
        fail_values: List[str] = []
        for _, row in ranked.iterrows():
            ok, failures = _passes(row, rules)
            pass_values.append(ok)
            fail_values.append(failures)
        ranked["profile"] = profile
        ranked["passes_profile"] = pass_values
        ranked["profile_fail_reasons"] = fail_values
        ranked = ranked.sort_values(
            ["passes_profile", "daily_weekly_objective", "sum_delta_net_pnl"], ascending=[False, False, False]
        )
        profile_rows.extend(ranked.to_dict("records"))
        passed = ranked[ranked["passes_profile"]]
        champions[profile] = passed.iloc[0].to_dict() if len(passed) else None

    profile_df = pd.DataFrame(profile_rows)
    daily.to_csv(out_dir / "daily_all_variant_metrics.csv", index=False)
    weekly.to_csv(out_dir / "weekly_all_variant_metrics.csv", index=False)
    summary.to_csv(out_dir / "daily_weekly_objective_summary.csv", index=False)
    profile_df.to_csv(out_dir / "daily_weekly_profile_ranking.csv", index=False)
    (out_dir / "daily_weekly_gate_manifest.json").write_text(
        json.dumps(
            _json_safe(
                {
                    "source_global_csv": str(args.global_csv),
                    "baseline_label": args.baseline_label,
                    "objective": "avg_week_delta_net_pnl + 0.7*q35_day_delta_net_pnl + 0.3*q20_day_delta_net_pnl",
                    "profiles": PROFILE_RULES,
                    "champions": champions,
                    "notes": "Development walk-forward gate over existing replay artifacts; not untouched OOS.",
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary_cols = [
        "objective_rank",
        "label",
        "diagnostic_family",
        "daily_weekly_objective",
        "avg_week_delta_net_pnl",
        "q35_day_delta_net_pnl",
        "q20_day_delta_net_pnl",
        "sum_delta_net_pnl",
        "positive_week_count",
        "positive_day_count",
        "mean_day_full_sl_delta",
        "june_net_delta",
        "june_full_sl_delta",
    ]
    profile_cols = [
        "profile",
        "label",
        "passes_profile",
        "profile_fail_reasons",
        "daily_weekly_objective",
        "positive_week_share",
        "q20_day_delta_net_pnl",
        "mean_day_full_sl_delta",
        "june_net_delta",
    ]
    md = [
        "# Contextual TP/SL Daily-Weekly Objective Gate",
        "",
        "This is a development walk-forward gate over existing replay artifacts. It does not rerun portfolio simulation and is not untouched OOS.",
        "",
        "Objective: `avg_week_delta_net_pnl + 0.7*q35_day_delta_net_pnl + 0.3*q20_day_delta_net_pnl`.",
        "",
        "## Overall Ranking",
        "",
        _markdown_table(summary, summary_cols),
        "",
        "## Profile Champions",
        "",
    ]
    for profile in PROFILE_RULES:
        champion = champions.get(profile)
        if champion is None:
            md.append(f"- `{profile}`: no variant passed all gates; keep baseline.")
        else:
            md.append(
                f"- `{profile}`: `{champion['label']}` "
                f"(objective={champion['daily_weekly_objective']:.2f}, delta_net_pnl={champion['sum_delta_net_pnl']:.2f})."
            )
    md.extend(["", "## Profile Ranking", "", _markdown_table(profile_df, profile_cols), ""])
    (out_dir / "daily_weekly_gate_report.md").write_text("\n".join(md), encoding="utf-8")
    print(out_dir / "daily_weekly_gate_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
