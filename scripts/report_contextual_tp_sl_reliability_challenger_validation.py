#!/usr/bin/env python3
"""Validate reliability-rule challengers against a no-rule baseline.

The report is intentionally replay-folder driven: it consumes the daily,
weekly, and accepted-decision artifacts already produced by
``ablate_contextual_tp_sl_conditional_head_filters.py``. It does not rerun the
portfolio engine and does not refit thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd


OBJECTIVE_COL = "objective_avgweek_0p7dayq35_0p3dayq20"


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
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
        return None
    return value


def _objective(daily_pnl: Iterable[float], weekly_pnl: Iterable[float]) -> Dict[str, float]:
    daily = pd.to_numeric(pd.Series(daily_pnl), errors="coerce").dropna()
    weekly = pd.to_numeric(pd.Series(weekly_pnl), errors="coerce").dropna()
    avg_week = float(weekly.mean()) if not weekly.empty else 0.0
    daily_q20 = float(daily.quantile(0.20)) if not daily.empty else 0.0
    daily_q35 = float(daily.quantile(0.35)) if not daily.empty else 0.0
    weekly_q20 = float(weekly.quantile(0.20)) if not weekly.empty else 0.0
    return {
        OBJECTIVE_COL: avg_week + 0.7 * daily_q35 + 0.3 * daily_q20,
        "avg_week_pnl": avg_week,
        "net_pnl": float(daily.sum()) if not daily.empty else float(weekly.sum()),
        "daily_q20_pnl": daily_q20,
        "daily_q35_pnl": daily_q35,
        "weekly_q20_pnl": weekly_q20,
        "weeks": int(len(weekly)),
        "days": int(len(daily)),
    }


def _head_name(strategy_id: str) -> str:
    value = str(strategy_id)
    if "boll" in value:
        return "short_bollinger"
    if "asset" in value:
        return "short_asset"
    if "dist" in value:
        return "long_dist"
    if "bars" in value:
        return "long_bars"
    return value


def _period_metrics(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    *,
    baseline_combo: str,
    baseline_rule: str,
    windows: Mapping[str, str | None],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for window, start in windows.items():
        cur_daily = daily
        cur_weekly = weekly
        if start is not None:
            start_ts = pd.Timestamp(start, tz="UTC")
            cur_daily = cur_daily.loc[cur_daily["day_start"].ge(start_ts)]
            cur_weekly = cur_weekly.loc[cur_weekly["week_start"].ge(start_ts)]
        base_daily = cur_daily.loc[
            cur_daily["combo_id"].astype(str).eq(baseline_combo)
            & cur_daily["rule_id"].astype(str).eq(baseline_rule)
        ]
        base_weekly = cur_weekly.loc[
            cur_weekly["combo_id"].astype(str).eq(baseline_combo)
            & cur_weekly["rule_id"].astype(str).eq(baseline_rule)
        ]
        baseline = _objective(base_daily.get("net_pnl", []), base_weekly.get("net_pnl", []))
        for (combo_id, rule_id), group in cur_daily.groupby(["combo_id", "rule_id"], sort=False):
            group_weekly = cur_weekly.loc[
                cur_weekly["combo_id"].astype(str).eq(str(combo_id))
                & cur_weekly["rule_id"].astype(str).eq(str(rule_id))
            ]
            selected = _objective(group.get("net_pnl", []), group_weekly.get("net_pnl", []))
            rec: Dict[str, Any] = {
                "window": window,
                "combo_id": combo_id,
                "rule_id": rule_id,
            }
            for key, value in selected.items():
                rec[key] = value
                rec[f"delta_{key}"] = value - baseline[key]
            rec["pass_pnl_tail_gate"] = bool(
                rec[f"delta_{OBJECTIVE_COL}"] >= 0.0
                and rec["delta_net_pnl"] >= 0.0
                and rec["delta_weekly_q20_pnl"] >= 0.0
                and rec["delta_daily_q20_pnl"] >= 0.0
            )
            rows.append(rec)
    return pd.DataFrame(rows)


def _accepted_metrics(accepted: pd.DataFrame, keys: List[str], baseline_rule: str) -> pd.DataFrame:
    grouped = (
        accepted.groupby(["rule_id", *keys], dropna=False)
        .agg(
            trades=("net_pnl", "size"),
            net_pnl=("net_pnl", "sum"),
            gross_pnl=("gross_pnl", "sum"),
            cost_pnl=("cost_pnl", "sum"),
            hit_rate=("hit", "mean"),
            full_sl_rate=("full_sl", "mean"),
        )
        .reset_index()
    )
    baseline = grouped.loc[grouped["rule_id"].astype(str).eq(baseline_rule)].drop(columns=["rule_id"]).set_index(keys)
    rows: List[Dict[str, Any]] = []
    for rule_id, group in grouped.groupby("rule_id", sort=False):
        current = group.drop(columns=["rule_id"]).set_index(keys)
        for idx in sorted(set(baseline.index).union(current.index)):
            base_row = baseline.loc[idx] if idx in baseline.index else None
            cur_row = current.loc[idx] if idx in current.index else None
            rec: Dict[str, Any] = {"rule_id": rule_id}
            idx_values = idx if isinstance(idx, tuple) else (idx,)
            for key, value in zip(keys, idx_values):
                rec[key] = value
            for col in ("trades", "net_pnl", "gross_pnl", "cost_pnl", "hit_rate", "full_sl_rate"):
                base_value = float(base_row[col]) if base_row is not None else 0.0
                cur_value = float(cur_row[col]) if cur_row is not None else 0.0
                rec[col] = cur_value
                rec[f"baseline_{col}"] = base_value
                rec[f"delta_{col}"] = cur_value - base_value
            rows.append(rec)
    return pd.DataFrame(rows)


def _replacement_quality(accepted: pd.DataFrame, baseline_rule: str, challenger_rule: str) -> pd.DataFrame:
    keys = ["timestamp", "strategy_id", "symbol"]
    baseline = accepted.loc[accepted["rule_id"].astype(str).eq(baseline_rule)].copy()
    challenger = accepted.loc[accepted["rule_id"].astype(str).eq(challenger_rule)].copy()
    baseline_keys = set(map(tuple, baseline[keys].astype(str).to_numpy()))
    challenger_keys = set(map(tuple, challenger[keys].astype(str).to_numpy()))
    common = baseline_keys & challenger_keys
    frames = {
        "entrants": challenger.loc[~challenger[keys].astype(str).apply(tuple, axis=1).isin(baseline_keys)],
        "removed": baseline.loc[~baseline[keys].astype(str).apply(tuple, axis=1).isin(challenger_keys)],
        "common_challenger": challenger.loc[challenger[keys].astype(str).apply(tuple, axis=1).isin(common)],
        "common_baseline": baseline.loc[baseline[keys].astype(str).apply(tuple, axis=1).isin(common)],
    }
    rows: List[Dict[str, Any]] = []
    for bucket, frame in frames.items():
        rows.append(
            {
                "bucket": bucket,
                "trades": int(len(frame)),
                "net_pnl": float(frame["net_pnl"].sum()),
                "hit_rate": float(frame["hit"].mean()) if not frame.empty else 0.0,
                "avg_net_pnl": float(frame["net_pnl"].mean()) if not frame.empty else 0.0,
                "full_sl_rate": float(frame["full_sl"].mean()) if not frame.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline-combo", default="long_bars:S_long_dist:R_short_asset:R_short_bollinger:R")
    parser.add_argument("--baseline-rule", default="none")
    parser.add_argument("--champion-rule", default="recent_hr_two_signal_rank_m010")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(args.attribution_dir / "conditional_filter_daily.csv")
    weekly = pd.read_csv(args.attribution_dir / "conditional_filter_weekly.csv")
    daily["day_start"] = pd.to_datetime(daily["day"], utc=True, errors="coerce")
    weekly["week_start"] = pd.to_datetime(
        weekly["week"].astype(str).str.split("/", n=1).str[0],
        utc=True,
        errors="coerce",
    )
    windows = {
        "full": None,
        "pre_may": None,
        "may_june": "2026-05-01",
        "june": "2026-06-01",
    }
    periods = _period_metrics(
        daily,
        weekly,
        baseline_combo=args.baseline_combo,
        baseline_rule=args.baseline_rule,
        windows=windows,
    )
    if not periods.empty:
        pre_may_mask = periods["window"].eq("pre_may")
        if pre_may_mask.any():
            daily_cut = daily.loc[daily["day_start"].lt(pd.Timestamp("2026-05-01", tz="UTC"))]
            weekly_cut = weekly.loc[weekly["week_start"].lt(pd.Timestamp("2026-05-01", tz="UTC"))]
            pre_may = _period_metrics(
                daily_cut,
                weekly_cut,
                baseline_combo=args.baseline_combo,
                baseline_rule=args.baseline_rule,
                windows={"pre_may": None},
            )
            periods = pd.concat([periods.loc[~pre_may_mask], pre_may], ignore_index=True)
    periods = periods.sort_values(["window", f"delta_{OBJECTIVE_COL}"], ascending=[True, False])
    periods.to_csv(args.out_dir / "period_validation_vs_baseline.csv", index=False)

    accepted_path = args.attribution_dir / "conditional_filter_accepted_decisions.parquet"
    accepted = pd.read_parquet(accepted_path)
    accepted["timestamp"] = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
    accepted["week_start"] = accepted["timestamp"].dt.tz_convert(None).dt.to_period("W").dt.start_time
    accepted["month"] = accepted["timestamp"].dt.strftime("%Y-%m")
    accepted["head"] = accepted["strategy_id"].astype(str).map(_head_name)
    size = pd.to_numeric(accepted["position_size"], errors="coerce").fillna(0.0)
    net_return = pd.to_numeric(accepted["position_net_return"], errors="coerce").fillna(0.0)
    gross_return = pd.to_numeric(accepted["position_gross_return"], errors="coerce").fillna(0.0)
    accepted["net_pnl"] = size * net_return
    accepted["gross_pnl"] = size * gross_return
    accepted["cost_pnl"] = accepted["gross_pnl"] - accepted["net_pnl"]
    accepted["hit"] = net_return.gt(0.0)
    accepted["full_sl"] = accepted["position_exit_reason"].astype(str).eq("full_sl")

    head = _accepted_metrics(accepted, ["head"], args.baseline_rule)
    month = _accepted_metrics(accepted, ["month"], args.baseline_rule)
    week = _accepted_metrics(accepted, ["week_start"], args.baseline_rule)
    head_month = _accepted_metrics(accepted, ["head", "month"], args.baseline_rule)
    replacement = _replacement_quality(accepted, args.baseline_rule, args.champion_rule)
    head.to_csv(args.out_dir / "head_attribution_vs_baseline.csv", index=False)
    month.to_csv(args.out_dir / "month_attribution_vs_baseline.csv", index=False)
    week.to_csv(args.out_dir / "week_attribution_vs_baseline.csv", index=False)
    head_month.to_csv(args.out_dir / "head_month_attribution_vs_baseline.csv", index=False)
    replacement.to_csv(args.out_dir / "replacement_quality.csv", index=False)

    champion_periods = periods.loc[periods["rule_id"].astype(str).eq(args.champion_rule)]
    champion_month = month.loc[month["rule_id"].astype(str).eq(args.champion_rule)]
    positive_month_share = float(champion_month["delta_net_pnl"].gt(0.0).mean()) if not champion_month.empty else 0.0
    negative_months = champion_month.loc[champion_month["delta_net_pnl"].lt(0.0), "month"].astype(str).tolist()
    payload = {
        "attribution_dir": str(args.attribution_dir),
        "out_dir": str(args.out_dir),
        "baseline_combo": args.baseline_combo,
        "baseline_rule": args.baseline_rule,
        "champion_rule": args.champion_rule,
        "daily_start": str(daily["day_start"].min()),
        "daily_end": str(daily["day_start"].max()),
        "weekly_start": str(weekly["week_start"].min()),
        "weekly_end": str(weekly["week_start"].max()),
        "champion_positive_month_share": positive_month_share,
        "champion_negative_months": negative_months,
    }
    (args.out_dir / "validation_summary.json").write_text(json.dumps(_json_safe(payload), indent=2))

    show_cols = [
        "window",
        "rule_id",
        f"delta_{OBJECTIVE_COL}",
        "delta_net_pnl",
        "delta_weekly_q20_pnl",
        "delta_daily_q20_pnl",
        "delta_daily_q35_pnl",
        "pass_pnl_tail_gate",
    ]
    head_cols = ["head", "rule_id", "delta_net_pnl", "delta_trades", "delta_hit_rate", "delta_full_sl_rate"]
    lines = [
        "# Reliability Challenger Validation",
        "",
        f"Attribution source: `{args.attribution_dir}`",
        f"Period: `{payload['daily_start']}` to `{payload['daily_end']}`",
        "Costs are included via replayed accepted decisions.",
        "",
        "## Period Deltas",
        "",
        champion_periods[[c for c in show_cols if c in champion_periods.columns]].round(6).to_markdown(index=False)
        if not champion_periods.empty
        else "_No champion rows._",
        "",
        "## Head Deltas",
        "",
        head.loc[head["rule_id"].astype(str).eq(args.champion_rule), [c for c in head_cols if c in head.columns]]
        .round(6)
        .to_markdown(index=False),
        "",
        "## Monthly Deltas",
        "",
        champion_month[
            ["month", "delta_net_pnl", "delta_trades", "delta_hit_rate", "delta_full_sl_rate"]
        ]
        .round(6)
        .to_markdown(index=False),
        "",
        "## Replacement Quality",
        "",
        replacement.round(6).to_markdown(index=False),
    ]
    (args.out_dir / "reliability_challenger_validation_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(_json_safe(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
