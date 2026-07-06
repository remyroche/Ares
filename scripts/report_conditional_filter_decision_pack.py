#!/usr/bin/env python3
"""Build a compact decision pack for conditional-filter replay challengers.

The script consumes outputs from ``ablate_contextual_tp_sl_conditional_head_filters.py``.
It does not replay candidates. It compares selected rules to a baseline over
daily, weekly, monthly, head, and accepted-trade replacement views.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


OBJECTIVE_COL = "objective_avgweek_0p7dayq35_0p3dayq20"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
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


def _head_name(strategy_id: str) -> str:
    text = str(strategy_id)
    if text.startswith("short_bollinger"):
        return "short_bollinger"
    parts = text.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else text


def _period_start(frame: pd.DataFrame, col: str) -> pd.Series:
    if col == "day":
        return pd.to_datetime(frame[col], utc=True, errors="coerce")
    if col == "week":
        periods = pd.PeriodIndex(frame[col].astype(str), freq="W")
        return pd.Series(periods.start_time.tz_localize("UTC"), index=frame.index)
    raise ValueError(f"Unsupported period column `{col}`")


def _quantiles(values: pd.Series, prefix: str) -> Dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {f"{prefix}_q{q:02d}": 0.0 for q in (5, 10, 20, 35, 50)}
    return {f"{prefix}_q{q:02d}": float(clean.quantile(q / 100.0)) for q in (5, 10, 20, 35, 50)}


def _pivot_delta(frame: pd.DataFrame, period_col: str, rule_id: str, baseline_rule: str) -> pd.DataFrame:
    pivot = (
        frame.pivot_table(index=period_col, columns="rule_id", values="net_pnl", aggfunc="sum")
        .fillna(0.0)
        .sort_index()
    )
    if baseline_rule not in pivot.columns:
        raise ValueError(f"Missing baseline rule `{baseline_rule}` in {period_col} table")
    if rule_id not in pivot.columns:
        return pd.DataFrame(columns=[period_col, "baseline_net_pnl", "candidate_net_pnl", "delta_net_pnl"])
    out = pd.DataFrame(
        {
            period_col: pivot.index,
            "baseline_net_pnl": pivot[baseline_rule].to_numpy(dtype=float),
            "candidate_net_pnl": pivot[rule_id].to_numpy(dtype=float),
        }
    )
    out["rule_id"] = rule_id
    out["delta_net_pnl"] = out["candidate_net_pnl"] - out["baseline_net_pnl"]
    return out


def _period_summary(daily: pd.DataFrame, weekly: pd.DataFrame, rule_id: str, baseline_rule: str) -> Dict[str, Any]:
    day = _pivot_delta(daily, "day", rule_id, baseline_rule)
    week = _pivot_delta(weekly, "week", rule_id, baseline_rule)
    day_delta = day["delta_net_pnl"] if not day.empty else pd.Series(dtype=float)
    week_delta = week["delta_net_pnl"] if not week.empty else pd.Series(dtype=float)
    avg_week = float(week_delta.mean()) if len(week_delta) else 0.0
    daily_q20 = float(day_delta.quantile(0.20)) if len(day_delta) else 0.0
    daily_q35 = float(day_delta.quantile(0.35)) if len(day_delta) else 0.0
    weighted_tail = 0.7 * daily_q35 + 0.3 * daily_q20
    rec: Dict[str, Any] = {
        "rule_id": rule_id,
        "days": int(len(day_delta)),
        "weeks": int(len(week_delta)),
        "active_days": int(day_delta.ne(0.0).sum()) if len(day_delta) else 0,
        "active_weeks": int(week_delta.ne(0.0).sum()) if len(week_delta) else 0,
        "delta_net_pnl": float(day_delta.sum()),
        "delta_avg_week_pnl": avg_week,
        "delta_weighted_daily_tail": weighted_tail,
        "delta_objective": avg_week + weighted_tail,
        "positive_day_share": float((day_delta > 0.0).mean()) if len(day_delta) else 0.0,
        "positive_week_share": float((week_delta > 0.0).mean()) if len(week_delta) else 0.0,
        "active_positive_day_share": float((day_delta[day_delta.ne(0.0)] > 0.0).mean())
        if day_delta.ne(0.0).any()
        else 0.0,
        "active_positive_week_share": float((week_delta[week_delta.ne(0.0)] > 0.0).mean())
        if week_delta.ne(0.0).any()
        else 0.0,
        "worst_day_delta": float(day_delta.min()) if len(day_delta) else 0.0,
        "worst_week_delta": float(week_delta.min()) if len(week_delta) else 0.0,
    }
    rec.update(_quantiles(day_delta, "delta_daily"))
    rec.update(_quantiles(week_delta, "delta_weekly"))
    return rec


def _accepted_with_pnl(accepted: pd.DataFrame) -> pd.DataFrame:
    out = accepted.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["month"] = out["timestamp"].dt.strftime("%Y-%m")
    out["week"] = out["timestamp"].dt.tz_convert(None).dt.to_period("W").astype(str)
    out["head"] = out["strategy_id"].astype(str).map(_head_name)
    size = pd.to_numeric(out.get("position_size"), errors="coerce").fillna(0.0)
    net_return = pd.to_numeric(out.get("position_net_return"), errors="coerce").fillna(0.0)
    gross_return = pd.to_numeric(out.get("position_gross_return"), errors="coerce").fillna(0.0)
    out["net_pnl_amount"] = size * net_return
    out["gross_pnl_amount"] = size * gross_return
    out["cost_pnl_amount"] = out["gross_pnl_amount"] - out["net_pnl_amount"]
    out["hit"] = net_return.gt(0.0)
    out["full_sl"] = out.get("position_exit_reason", "").astype(str).eq("full_sl")
    return out


def _accepted_group_delta(accepted: pd.DataFrame, keys: Sequence[str], rules: Sequence[str], baseline_rule: str) -> pd.DataFrame:
    grouped = (
        accepted.groupby(["rule_id", *keys], dropna=False)
        .agg(
            trades=("net_pnl_amount", "size"),
            net_pnl=("net_pnl_amount", "sum"),
            gross_pnl=("gross_pnl_amount", "sum"),
            costs=("cost_pnl_amount", "sum"),
            hit_rate=("hit", "mean"),
            full_sl_rate=("full_sl", "mean"),
        )
        .reset_index()
    )
    baseline = grouped.loc[grouped["rule_id"].astype(str).eq(baseline_rule)].drop(columns=["rule_id"]).set_index(list(keys))
    rows: List[Dict[str, Any]] = []
    for rule_id in rules:
        current = grouped.loc[grouped["rule_id"].astype(str).eq(rule_id)].drop(columns=["rule_id"]).set_index(list(keys))
        idx_values = sorted(set(baseline.index).union(current.index))
        for idx in idx_values:
            base = baseline.loc[idx] if idx in baseline.index else None
            cur = current.loc[idx] if idx in current.index else None
            idx_tuple = idx if isinstance(idx, tuple) else (idx,)
            rec: Dict[str, Any] = {"rule_id": rule_id}
            for key, value in zip(keys, idx_tuple):
                rec[key] = value
            for col in ("trades", "net_pnl", "gross_pnl", "costs", "hit_rate", "full_sl_rate"):
                base_val = float(base[col]) if base is not None else 0.0
                cur_val = float(cur[col]) if cur is not None else 0.0
                rec[f"baseline_{col}"] = base_val
                rec[col] = cur_val
                rec[f"delta_{col}"] = cur_val - base_val
            rows.append(rec)
    return pd.DataFrame(rows)


def _replacement_quality(accepted: pd.DataFrame, rule_id: str, baseline_rule: str) -> Dict[str, Any]:
    keys = ["timestamp", "symbol", "side", "strategy_id"]
    baseline = accepted.loc[accepted["rule_id"].astype(str).eq(baseline_rule)].copy()
    challenger = accepted.loc[accepted["rule_id"].astype(str).eq(rule_id)].copy()
    if baseline.empty or challenger.empty:
        return {}
    base_keys = set(map(tuple, baseline[keys].astype(str).to_numpy()))
    cand_keys = set(map(tuple, challenger[keys].astype(str).to_numpy()))
    entrants = challenger.loc[[tuple(row) in cand_keys - base_keys for row in challenger[keys].astype(str).to_numpy()]]
    removed = baseline.loc[[tuple(row) in base_keys - cand_keys for row in baseline[keys].astype(str).to_numpy()]]

    def summarize(prefix: str, frame: pd.DataFrame) -> Dict[str, Any]:
        return {
            f"{prefix}_trades": int(len(frame)),
            f"{prefix}_net_pnl": float(frame["net_pnl_amount"].sum()),
            f"{prefix}_hit_rate": float(frame["hit"].mean()) if len(frame) else 0.0,
            f"{prefix}_full_sl_rate": float(frame["full_sl"].mean()) if len(frame) else 0.0,
        }

    out: Dict[str, Any] = {"rule_id": rule_id}
    out.update(summarize("entrant", entrants))
    out.update(summarize("removed", removed))
    out["entrant_minus_removed_net_pnl"] = out["entrant_net_pnl"] - out["removed_net_pnl"]
    out["entrant_minus_removed_hit_rate"] = out["entrant_hit_rate"] - out["removed_hit_rate"]
    out["entrant_minus_removed_full_sl_rate"] = out["entrant_full_sl_rate"] - out["removed_full_sl_rate"]
    return out


def _markdown_table(frame: pd.DataFrame, cols: Sequence[str], max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    use_cols = [c for c in cols if c in frame.columns]
    out = frame[use_cols]
    if max_rows is not None:
        out = out.head(max_rows)
    return out.round(6).to_markdown(index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline-rule", default="none")
    parser.add_argument("--rule", action="append", required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(args.attribution_dir / "conditional_filter_daily.csv")
    weekly = pd.read_csv(args.attribution_dir / "conditional_filter_weekly.csv")
    daily["day_start"] = _period_start(daily, "day")
    weekly["week_start"] = _period_start(weekly, "week")
    accepted = _accepted_with_pnl(pd.read_parquet(args.attribution_dir / "conditional_filter_accepted_decisions.parquet"))

    rules = [str(r) for r in args.rule]
    summaries: List[Dict[str, Any]] = []
    replacements: List[Dict[str, Any]] = []
    for rule_id in rules:
        rec = _period_summary(daily, weekly, rule_id, args.baseline_rule)
        replacements.append(_replacement_quality(accepted, rule_id, args.baseline_rule))
        summaries.append(rec)

    summary = pd.DataFrame(summaries)
    replacement = pd.DataFrame(replacements)
    head_delta = _accepted_group_delta(accepted, ["head"], rules, args.baseline_rule)
    month_delta = _accepted_group_delta(accepted, ["month"], rules, args.baseline_rule)
    week_delta = _accepted_group_delta(accepted, ["week"], rules, args.baseline_rule)
    summary = summary.merge(replacement, on="rule_id", how="left")
    summary = summary.sort_values(["delta_objective", "delta_net_pnl"], ascending=[False, False])

    summary.to_csv(args.out_dir / "decision_pack_summary.csv", index=False)
    head_delta.to_csv(args.out_dir / "decision_pack_head_deltas.csv", index=False)
    month_delta.to_csv(args.out_dir / "decision_pack_month_deltas.csv", index=False)
    week_delta.to_csv(args.out_dir / "decision_pack_week_deltas.csv", index=False)
    replacement.to_csv(args.out_dir / "decision_pack_replacement_quality.csv", index=False)

    payload = {
        "attribution_dir": str(args.attribution_dir),
        "out_dir": str(args.out_dir),
        "baseline_rule": args.baseline_rule,
        "rules": rules,
        "daily_start": str(daily["day_start"].min()),
        "daily_end": str(daily["day_start"].max()),
        "weekly_start": str(weekly["week_start"].min()),
        "weekly_end": str(weekly["week_start"].max()),
        "accepted_start": str(accepted["timestamp"].min()),
        "accepted_end": str(accepted["timestamp"].max()),
        "winner_by_objective": str(summary.iloc[0]["rule_id"]) if not summary.empty else None,
    }
    (args.out_dir / "decision_pack_manifest.json").write_text(json.dumps(_json_safe(payload), indent=2))

    summary_cols = [
        "rule_id",
        "delta_net_pnl",
        "delta_objective",
        "active_days",
        "active_weeks",
        "delta_daily_q05",
        "delta_daily_q10",
        "delta_daily_q20",
        "delta_daily_q35",
        "delta_weekly_q05",
        "delta_weekly_q10",
        "delta_weekly_q20",
        "delta_weekly_q35",
        "positive_week_share",
        "active_positive_week_share",
        "entrant_trades",
        "entrant_net_pnl",
        "removed_trades",
        "removed_net_pnl",
        "entrant_minus_removed_net_pnl",
    ]
    head_cols = ["rule_id", "head", "delta_trades", "delta_net_pnl", "delta_hit_rate", "delta_full_sl_rate"]
    month_cols = ["rule_id", "month", "delta_trades", "delta_net_pnl", "delta_hit_rate", "delta_full_sl_rate"]
    lines = [
        "# Conditional Filter Decision Pack",
        "",
        f"Attribution source: `{args.attribution_dir}`",
        f"Daily range: `{payload['daily_start']}` to `{payload['daily_end']}`",
        f"Accepted-decision range: `{payload['accepted_start']}` to `{payload['accepted_end']}`",
        "Costs are included via replayed accepted decisions.",
        "",
        "## Summary",
        "",
        _markdown_table(summary, summary_cols),
        "",
        "## Head Deltas",
        "",
        _markdown_table(head_delta.sort_values(["rule_id", "head"]), head_cols),
        "",
        "## Monthly Deltas",
        "",
        _markdown_table(month_delta.sort_values(["rule_id", "month"]), month_cols),
        "",
        "## Worst Weekly Deltas",
        "",
        _markdown_table(
            pd.concat(
                [
                    group.sort_values("delta_net_pnl").head(8)
                    for _, group in week_delta.groupby("rule_id", sort=False)
                ],
                ignore_index=True,
            ),
            ["rule_id", "week", "delta_trades", "delta_net_pnl", "delta_hit_rate", "delta_full_sl_rate"],
        ),
    ]
    (args.out_dir / "decision_pack_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(_json_safe(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
