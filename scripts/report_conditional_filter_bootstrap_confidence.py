#!/usr/bin/env python3
"""Block-bootstrap confidence for conditional-filter replay challengers.

This consumes replay outputs from ``ablate_contextual_tp_sl_conditional_head_filters.py``.
It does not rerun the portfolio. Weekly blocks are sampled with replacement and
daily deltas inside each sampled week are kept together.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


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


def _week_start_from_day(day: pd.Series) -> pd.Series:
    ts = pd.to_datetime(day, utc=True, errors="coerce").dt.tz_convert(None)
    return ts.dt.to_period("W").astype(str)


def _pivot_delta(frame: pd.DataFrame, index_col: str, rule_id: str, baseline_rule: str) -> pd.Series:
    pivot = (
        frame.pivot_table(index=index_col, columns="rule_id", values="net_pnl", aggfunc="sum")
        .fillna(0.0)
        .sort_index()
    )
    if baseline_rule not in pivot.columns:
        raise ValueError(f"Missing baseline rule `{baseline_rule}` in {index_col} table")
    if rule_id not in pivot.columns:
        return pd.Series(dtype=float)
    return (pivot[rule_id].astype(float) - pivot[baseline_rule].astype(float)).rename("delta_net_pnl")


def _score_sample(week_delta: np.ndarray, day_delta: np.ndarray) -> Dict[str, float]:
    if week_delta.size == 0:
        avg_week = weekly_q20 = weekly_q05 = weekly_q10 = 0.0
    else:
        avg_week = float(np.mean(week_delta))
        weekly_q05 = float(np.quantile(week_delta, 0.05))
        weekly_q10 = float(np.quantile(week_delta, 0.10))
        weekly_q20 = float(np.quantile(week_delta, 0.20))
    if day_delta.size == 0:
        daily_q20 = daily_q35 = 0.0
        net = float(np.sum(week_delta))
    else:
        daily_q20 = float(np.quantile(day_delta, 0.20))
        daily_q35 = float(np.quantile(day_delta, 0.35))
        net = float(np.sum(day_delta))
    weighted_tail = 0.7 * daily_q35 + 0.3 * daily_q20
    return {
        "delta_net_pnl": net,
        "delta_avg_week_pnl": avg_week,
        "delta_objective": avg_week + weighted_tail,
        "delta_weighted_daily_tail": weighted_tail,
        "delta_daily_q20": daily_q20,
        "delta_daily_q35": daily_q35,
        "delta_weekly_q05": weekly_q05,
        "delta_weekly_q10": weekly_q10,
        "delta_weekly_q20": weekly_q20,
    }


def _bootstrap_rule(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    rule_id: str,
    baseline_rule: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[Dict[str, float], pd.DataFrame]:
    daily_work = daily.copy()
    daily_work["week"] = _week_start_from_day(daily_work["day"])
    day_delta = _pivot_delta(daily_work, "day", rule_id, baseline_rule)
    week_delta = _pivot_delta(weekly, "week", rule_id, baseline_rule)
    if week_delta.empty:
        observed = _score_sample(np.array([], dtype=float), day_delta.to_numpy(dtype=float))
        return observed, pd.DataFrame()

    week_values = week_delta.to_dict()
    day_to_week = daily_work.drop_duplicates("day").set_index("day")["week"].to_dict()
    day_by_week = {
        week: day_delta.loc[[day for day in day_delta.index if day_to_week.get(day) == week]].to_numpy(dtype=float)
        for week in week_delta.index
    }

    observed = _score_sample(week_delta.to_numpy(dtype=float), day_delta.to_numpy(dtype=float))
    rng = np.random.default_rng(int(seed))
    weeks = np.array(list(week_delta.index), dtype=object)
    rows: List[Dict[str, float]] = []
    for i in range(int(n_bootstrap)):
        sampled = rng.choice(weeks, size=len(weeks), replace=True)
        sample_week = np.array([week_values[w] for w in sampled], dtype=float)
        sample_days = np.concatenate([day_by_week.get(w, np.array([], dtype=float)) for w in sampled])
        rec = _score_sample(sample_week, sample_days)
        rec["bootstrap_id"] = float(i)
        rows.append(rec)
    return observed, pd.DataFrame(rows)


def _summarize_bootstrap(rule_id: str, observed: Dict[str, float], samples: pd.DataFrame) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"rule_id": rule_id}
    for key, value in observed.items():
        rec[f"observed_{key}"] = value
        if samples.empty or key not in samples.columns:
            rec[f"{key}_p05"] = np.nan
            rec[f"{key}_p50"] = np.nan
            rec[f"{key}_p95"] = np.nan
            rec[f"prob_{key}_positive"] = np.nan
            continue
        vals = pd.to_numeric(samples[key], errors="coerce").dropna()
        rec[f"{key}_p05"] = float(vals.quantile(0.05)) if not vals.empty else np.nan
        rec[f"{key}_p50"] = float(vals.quantile(0.50)) if not vals.empty else np.nan
        rec[f"{key}_p95"] = float(vals.quantile(0.95)) if not vals.empty else np.nan
        rec[f"prob_{key}_positive"] = float(vals.gt(0.0).mean()) if not vals.empty else np.nan
    return rec


def _markdown_table(frame: pd.DataFrame, cols: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    use_cols = [c for c in cols if c in frame.columns]
    return frame[use_cols].round(6).to_markdown(index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline-rule", default="none")
    parser.add_argument("--rule", action="append", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(args.attribution_dir / "conditional_filter_daily.csv")
    weekly = pd.read_csv(args.attribution_dir / "conditional_filter_weekly.csv")
    rows: List[Dict[str, Any]] = []
    for offset, rule_id in enumerate(args.rule):
        observed, samples = _bootstrap_rule(
            daily,
            weekly,
            str(rule_id),
            args.baseline_rule,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + offset,
        )
        samples.to_csv(args.out_dir / f"bootstrap_samples_{rule_id}.csv", index=False)
        rows.append(_summarize_bootstrap(str(rule_id), observed, samples))
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "bootstrap_confidence_summary.csv", index=False)

    payload = {
        "attribution_dir": str(args.attribution_dir),
        "out_dir": str(args.out_dir),
        "baseline_rule": args.baseline_rule,
        "rules": [str(r) for r in args.rule],
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
    }
    (args.out_dir / "bootstrap_confidence_manifest.json").write_text(json.dumps(_json_safe(payload), indent=2))

    cols = [
        "rule_id",
        "observed_delta_net_pnl",
        "delta_net_pnl_p05",
        "delta_net_pnl_p50",
        "delta_net_pnl_p95",
        "prob_delta_net_pnl_positive",
        "observed_delta_objective",
        "delta_objective_p05",
        "delta_objective_p50",
        "delta_objective_p95",
        "prob_delta_objective_positive",
        "observed_delta_weekly_q20",
        "delta_weekly_q20_p05",
        "prob_delta_weekly_q20_positive",
        "observed_delta_daily_q20",
        "delta_daily_q20_p05",
        "prob_delta_daily_q20_positive",
    ]
    lines = [
        "# Conditional Filter Bootstrap Confidence",
        "",
        f"Attribution source: `{args.attribution_dir}`",
        f"Baseline rule: `{args.baseline_rule}`",
        f"Bootstrap samples: `{args.n_bootstrap}`",
        "Weekly blocks are sampled with replacement; daily rows inside sampled weeks are kept together.",
        "",
        _markdown_table(summary, cols),
    ]
    (args.out_dir / "bootstrap_confidence_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(_json_safe(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
