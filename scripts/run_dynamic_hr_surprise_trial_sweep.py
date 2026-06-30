#!/usr/bin/env python3
"""Run and ledger dynamic HR-surprise threshold trial sweeps."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CANDIDATES = Path(
    "data_perp/reports/finalfit_broad_candidate_regen_20260627/"
    "dynamic_hr_surprise_robust_subwindows_walkforward_to_jun25_20260627/"
    "inputs/simple_policy_candidates_broad_through_20260625.parquet"
)
DEFAULT_POLICY_PARAMS = Path(
    "data_perp/artifacts/20260620_185313_no_mkt4_evband002_policy_uncertainty_ev/"
    "best_policy_params.json"
)
DEFAULT_ROOT = Path(
    "data_perp/reports/finalfit_broad_candidate_regen_20260627/"
    "dynamic_hr_surprise_v4_v5_v6_comparison_20260627"
)
DEFAULT_SWEEP_DIR = DEFAULT_ROOT / "trial_sweep_20260627"
DEFAULT_LEDGER = DEFAULT_ROOT / "dynamic_hr_surprise_trial_ledger.md"


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    name: str
    description: str
    overrides: tuple[str, ...]


TRIALS: tuple[TrialSpec, ...] = (
    TrialSpec(
        "T01",
        "v5_guard_hr35",
        "v4-style 5-day subwindow objective plus recent validation guard with hit-rate floor raised to 35%.",
        ("--recent-validation-guard", "--recent-validation-min-hit-rate", "0.35"),
    ),
    TrialSpec(
        "T02",
        "v5_guard_10d_hr30",
        "v4-style 5-day subwindow objective plus a slower 10-day validation guard at 30% hit-rate floor.",
        ("--recent-validation-guard", "--recent-validation-days", "10.0", "--recent-validation-min-hit-rate", "0.30"),
    ),
    TrialSpec(
        "T03",
        "v5_guard_avg_nonnegative",
        "v4-style 5-day subwindow objective plus recent guard requiring non-negative average PnL.",
        (
            "--recent-validation-guard",
            "--recent-validation-min-hit-rate",
            "0.30",
            "--recent-validation-min-avg-pnl",
            "0.0",
        ),
    ),
    TrialSpec(
        "T04",
        "v6_q35_weighted_1_03_05",
        "Recent daily Q35 Y objective with bucket weights: last 7d=1.0, days 8-14=0.3, older lookback=0.5.",
        (
            "--threshold-selection-objective",
            "recent_daily_quantile",
            "--recent-quantile-days",
            "30.0",
            "--recent-quantile-level",
            "0.35",
            "--recent-quantile-weight-mode",
            "bucket",
            "--recent-quantile-weight-last-7",
            "1.0",
            "--recent-quantile-weight-prev-7",
            "0.3",
            "--recent-quantile-weight-older",
            "0.5",
        ),
    ),
    TrialSpec(
        "T05",
        "v6_q30_weighted_15_05_02",
        "Recent daily Q30 objective with stronger recency: last 7d=1.5, days 8-14=0.5, older lookback=0.2.",
        (
            "--threshold-selection-objective",
            "recent_daily_quantile",
            "--recent-quantile-days",
            "30.0",
            "--recent-quantile-level",
            "0.30",
            "--recent-quantile-weight-mode",
            "bucket",
            "--recent-quantile-weight-last-7",
            "1.5",
            "--recent-quantile-weight-prev-7",
            "0.5",
            "--recent-quantile-weight-older",
            "0.2",
        ),
    ),
    TrialSpec(
        "T06",
        "v6_q40_weighted_guard",
        "Weighted Q40 recent daily objective plus the 5-day recent validation guard.",
        (
            "--threshold-selection-objective",
            "recent_daily_quantile",
            "--recent-quantile-days",
            "30.0",
            "--recent-quantile-level",
            "0.40",
            "--recent-quantile-weight-mode",
            "bucket",
            "--recent-quantile-weight-last-7",
            "1.0",
            "--recent-quantile-weight-prev-7",
            "0.3",
            "--recent-quantile-weight-older",
            "0.5",
            "--recent-validation-guard",
            "--recent-validation-min-hit-rate",
            "0.30",
        ),
    ),
    TrialSpec(
        "T07",
        "q40_weighted_guard_hr35",
        "T06 with recent validation hit-rate floor raised from 30% to 35%.",
        (
            "--threshold-selection-objective",
            "recent_daily_quantile",
            "--recent-quantile-days",
            "30.0",
            "--recent-quantile-level",
            "0.40",
            "--recent-quantile-weight-mode",
            "bucket",
            "--recent-quantile-weight-last-7",
            "1.0",
            "--recent-quantile-weight-prev-7",
            "0.3",
            "--recent-quantile-weight-older",
            "0.5",
            "--recent-validation-guard",
            "--recent-validation-min-hit-rate",
            "0.35",
        ),
    ),
    TrialSpec(
        "T08",
        "q40_weighted_guard_10d",
        "T06 with a slower 10-day recent validation guard.",
        (
            "--threshold-selection-objective",
            "recent_daily_quantile",
            "--recent-quantile-days",
            "30.0",
            "--recent-quantile-level",
            "0.40",
            "--recent-quantile-weight-mode",
            "bucket",
            "--recent-quantile-weight-last-7",
            "1.0",
            "--recent-quantile-weight-prev-7",
            "0.3",
            "--recent-quantile-weight-older",
            "0.5",
            "--recent-validation-guard",
            "--recent-validation-days",
            "10.0",
            "--recent-validation-min-hit-rate",
            "0.30",
        ),
    ),
    TrialSpec(
        "T09",
        "q40_weighted_guard_last7_12",
        "T06 with slightly stronger last-7d weight: last 7d=1.2, days 8-14=0.3, older=0.5.",
        (
            "--threshold-selection-objective",
            "recent_daily_quantile",
            "--recent-quantile-days",
            "30.0",
            "--recent-quantile-level",
            "0.40",
            "--recent-quantile-weight-mode",
            "bucket",
            "--recent-quantile-weight-last-7",
            "1.2",
            "--recent-quantile-weight-prev-7",
            "0.3",
            "--recent-quantile-weight-older",
            "0.5",
            "--recent-validation-guard",
            "--recent-validation-min-hit-rate",
            "0.30",
        ),
    ),
    TrialSpec(
        "T10",
        "q40_weighted_guard_last7_08",
        "T06 with softer last-7d weight: last 7d=0.8, days 8-14=0.4, older=0.6.",
        (
            "--threshold-selection-objective",
            "recent_daily_quantile",
            "--recent-quantile-days",
            "30.0",
            "--recent-quantile-level",
            "0.40",
            "--recent-quantile-weight-mode",
            "bucket",
            "--recent-quantile-weight-last-7",
            "0.8",
            "--recent-quantile-weight-prev-7",
            "0.4",
            "--recent-quantile-weight-older",
            "0.6",
            "--recent-validation-guard",
            "--recent-validation-min-hit-rate",
            "0.30",
        ),
    ),
    TrialSpec(
        "T11",
        "q45_weighted_guard",
        "T06 with a more permissive Q45 daily objective.",
        (
            "--threshold-selection-objective",
            "recent_daily_quantile",
            "--recent-quantile-days",
            "30.0",
            "--recent-quantile-level",
            "0.45",
            "--recent-quantile-weight-mode",
            "bucket",
            "--recent-quantile-weight-last-7",
            "1.0",
            "--recent-quantile-weight-prev-7",
            "0.3",
            "--recent-quantile-weight-older",
            "0.5",
            "--recent-validation-guard",
            "--recent-validation-min-hit-rate",
            "0.30",
        ),
    ),
    TrialSpec(
        "T12",
        "q35_weighted_guard",
        "T06 with a stricter Q35 daily objective.",
        (
            "--threshold-selection-objective",
            "recent_daily_quantile",
            "--recent-quantile-days",
            "30.0",
            "--recent-quantile-level",
            "0.35",
            "--recent-quantile-weight-mode",
            "bucket",
            "--recent-quantile-weight-last-7",
            "1.0",
            "--recent-quantile-weight-prev-7",
            "0.3",
            "--recent-quantile-weight-older",
            "0.5",
            "--recent-validation-guard",
            "--recent-validation-min-hit-rate",
            "0.30",
        ),
    ),
    TrialSpec(
        "T13",
        "q42_weighted_guard_hr35",
        "Interpolate T07 and T11: Q42 daily objective with 35% recent HR guard.",
        (
            "--threshold-selection-objective",
            "recent_daily_quantile",
            "--recent-quantile-days",
            "30.0",
            "--recent-quantile-level",
            "0.42",
            "--recent-quantile-weight-mode",
            "bucket",
            "--recent-quantile-weight-last-7",
            "1.0",
            "--recent-quantile-weight-prev-7",
            "0.3",
            "--recent-quantile-weight-older",
            "0.5",
            "--recent-validation-guard",
            "--recent-validation-min-hit-rate",
            "0.35",
        ),
    ),
    TrialSpec(
        "T14",
        "q45_weighted_guard_hr35",
        "Q45 daily objective with the stricter 35% recent HR guard.",
        (
            "--threshold-selection-objective",
            "recent_daily_quantile",
            "--recent-quantile-days",
            "30.0",
            "--recent-quantile-level",
            "0.45",
            "--recent-quantile-weight-mode",
            "bucket",
            "--recent-quantile-weight-last-7",
            "1.0",
            "--recent-quantile-weight-prev-7",
            "0.3",
            "--recent-quantile-weight-older",
            "0.5",
            "--recent-validation-guard",
            "--recent-validation-min-hit-rate",
            "0.35",
        ),
    ),
    TrialSpec(
        "T15",
        "q40_weighted_guard_hr40",
        "T07 with a stricter 40% recent HR guard.",
        (
            "--threshold-selection-objective",
            "recent_daily_quantile",
            "--recent-quantile-days",
            "30.0",
            "--recent-quantile-level",
            "0.40",
            "--recent-quantile-weight-mode",
            "bucket",
            "--recent-quantile-weight-last-7",
            "1.0",
            "--recent-quantile-weight-prev-7",
            "0.3",
            "--recent-quantile-weight-older",
            "0.5",
            "--recent-validation-guard",
            "--recent-validation-min-hit-rate",
            "0.40",
        ),
    ),
    TrialSpec(
        "T16",
        "q42_weighted_guard_hr35_last7_11",
        "Q42, 35% HR guard, and slightly stronger last-7d weight 1.1.",
        (
            "--threshold-selection-objective",
            "recent_daily_quantile",
            "--recent-quantile-days",
            "30.0",
            "--recent-quantile-level",
            "0.42",
            "--recent-quantile-weight-mode",
            "bucket",
            "--recent-quantile-weight-last-7",
            "1.1",
            "--recent-quantile-weight-prev-7",
            "0.3",
            "--recent-quantile-weight-older",
            "0.5",
            "--recent-validation-guard",
            "--recent-validation-min-hit-rate",
            "0.35",
        ),
    ),
    TrialSpec(
        "T17",
        "q45_weighted_guard_hr35_iqr20",
        "Q45 and 35% HR guard with a higher recent daily IQR penalty.",
        (
            "--threshold-selection-objective",
            "recent_daily_quantile",
            "--recent-quantile-days",
            "30.0",
            "--recent-quantile-level",
            "0.45",
            "--recent-quantile-weight-mode",
            "bucket",
            "--recent-quantile-weight-last-7",
            "1.0",
            "--recent-quantile-weight-prev-7",
            "0.3",
            "--recent-quantile-weight-older",
            "0.5",
            "--recent-quantile-iqr-penalty",
            "0.20",
            "--recent-validation-guard",
            "--recent-validation-min-hit-rate",
            "0.35",
        ),
    ),
)


def _common_compare_args(candidates: Path, policy_params: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        "scripts/compare_dynamic_hr_surprise_threshold.py",
        "--candidates",
        str(candidates),
        "--policy-params",
        str(policy_params),
        "--output-dir",
        str(output_dir),
        "--calendar-only",
        "--calendar-eval-start",
        "2026-05-01",
        "--calendar-eval-end",
        "2026-06-25T23:59:59Z",
        "--calendar-xw-min-train-days",
        "90.0",
        "--calendar-xw-max-train-days",
        "183.0",
        "--calendar-y-train-days",
        "20.0",
        "--disable-deployed-threshold-floor",
        "--head-optimization-mode",
        "independent",
        "--threshold-refresh-mode",
        "grid",
        "--top-rank-floor",
        "0.70",
        "--trials",
        "120",
        "--threshold-grid-size",
        "201",
        "--x-min-days",
        "1.0",
        "--x-max-days",
        "28.0",
        "--w-lower-min",
        "0.0",
        "--w-lower-max",
        "0.25",
        "--w-raise-min",
        "0.0",
        "--w-raise-max",
        "0.60",
        "--y-min",
        "-0.50",
        "--y-max",
        "1.50",
        "--z-clip",
        "5.0",
        "--subwindow-constraints-mode",
        "penalty",
        "--subwindow-days",
        "5.0",
        "--min-subwindows",
        "4",
        "--min-positive-objective-fraction",
        "0.25",
        "--subwindow-q15-floor",
        "-1.00",
        "--subwindow-drawdown-floor",
        "-3.00",
        "--lambda-iqr",
        "0.25",
        "--lambda-tail",
        "0.50",
        "--subwindow-constraint-penalty",
        "10.0",
        "--min-threshold-selected-count",
        "0",
        "--min-threshold-active-subwindows",
        "0",
        "--deployed-threshold-soft-prior-strength",
        "8.0",
        "--deployed-threshold-soft-prior-deadband",
        "0.03",
        "--deployed-threshold-soft-prior-power",
        "2.0",
        "--deployed-threshold-soft-prior-activity-weight",
        "0.25",
    ]


def _week_start(ts: pd.Series) -> pd.Series:
    dates = pd.to_datetime(ts, utc=True, errors="coerce").dt.floor("D")
    return dates - pd.to_timedelta(dates.dt.weekday, unit="D")


def _trial_metrics(output_dir: Path, trial: TrialSpec) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    selected_path = output_dir / "calendar_dynamic_hr_surprise_selected_rows.parquet"
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected rows: {selected_path}")
    selected = pd.read_parquet(selected_path).copy()
    selected["timestamp"] = pd.to_datetime(selected["timestamp"], utc=True, errors="coerce")
    selected["week_start"] = _week_start(selected["timestamp"])
    selected["hit"] = pd.to_numeric(selected["net_return"], errors="coerce").gt(0.0).astype(float)
    week_index = pd.date_range(
        pd.Timestamp("2026-05-01", tz="UTC") - pd.Timedelta(days=pd.Timestamp("2026-05-01").weekday()),
        pd.Timestamp("2026-06-25", tz="UTC"),
        freq="W-MON",
    )
    if len(selected):
        weekly = (
            selected.groupby("week_start", observed=True)
            .agg(
                pnl_net_spread=("net_return", "sum"),
                trades=("net_return", "size"),
                hits=("hit", "sum"),
            )
            .reindex(week_index, fill_value=0.0)
            .rename_axis("week_start")
            .reset_index()
        )
    else:
        weekly = pd.DataFrame({"week_start": week_index, "pnl_net_spread": 0.0, "trades": 0.0, "hits": 0.0})
    weekly["trial_id"] = trial.trial_id
    weekly["variant"] = trial.name
    weekly["hit_rate"] = np.divide(
        weekly["hits"].to_numpy(dtype=float),
        weekly["trades"].to_numpy(dtype=float),
        out=np.full(len(weekly), np.nan),
        where=weekly["trades"].to_numpy(dtype=float) > 0,
    )
    weekly["pnl_per_trade"] = np.divide(
        weekly["pnl_net_spread"].to_numpy(dtype=float),
        weekly["trades"].to_numpy(dtype=float),
        out=np.full(len(weekly), np.nan),
        where=weekly["trades"].to_numpy(dtype=float) > 0,
    )
    by_head = (
        selected.groupby("head", observed=True)
        .agg(
            pnl_net_spread=("net_return", "sum"),
            trades=("net_return", "size"),
            hits=("hit", "sum"),
        )
        .reset_index()
        if len(selected)
        else pd.DataFrame(columns=["head", "pnl_net_spread", "trades", "hits"])
    )
    by_head["trial_id"] = trial.trial_id
    by_head["variant"] = trial.name
    by_head["hit_rate"] = np.divide(
        by_head["hits"].to_numpy(dtype=float),
        by_head["trades"].to_numpy(dtype=float),
        out=np.full(len(by_head), np.nan),
        where=by_head["trades"].to_numpy(dtype=float) > 0,
    )
    by_head["pnl_per_trade"] = np.divide(
        by_head["pnl_net_spread"].to_numpy(dtype=float),
        by_head["trades"].to_numpy(dtype=float),
        out=np.full(len(by_head), np.nan),
        where=by_head["trades"].to_numpy(dtype=float) > 0,
    )
    total_pnl = float(weekly["pnl_net_spread"].sum())
    trades = int(weekly["trades"].sum())
    hits = float(weekly["hits"].sum())
    metrics = {
        "trial_id": trial.trial_id,
        "variant": trial.name,
        "description": trial.description,
        "total_pnl_net_spread": total_pnl,
        "trades": trades,
        "hit_rate": float(hits / trades) if trades else np.nan,
        "pnl_per_trade": float(total_pnl / trades) if trades else np.nan,
        "worst_week_pnl": float(weekly["pnl_net_spread"].min()) if len(weekly) else 0.0,
        "q15_week_pnl": float(weekly["pnl_net_spread"].quantile(0.15)) if len(weekly) else 0.0,
        "q05_week_pnl": float(weekly["pnl_net_spread"].quantile(0.05)) if len(weekly) else 0.0,
        "positive_week_fraction": float((weekly["pnl_net_spread"] > 0.0).mean()) if len(weekly) else 0.0,
        "output_dir": str(output_dir),
    }
    return metrics, weekly, by_head


def _format_pct(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{100.0 * value:.2f}%"


def _append_ledger(ledger: Path, metrics: dict[str, Any], weekly: pd.DataFrame, by_head: pd.DataFrame) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    week_rows = []
    for row in weekly.itertuples(index=False):
        week_rows.append(
            f"| {pd.Timestamp(row.week_start).date()} | {row.pnl_net_spread:.4f} | {int(row.trades)} | "
            f"{_format_pct(float(row.hit_rate))} | {row.pnl_per_trade:.5f} |"
        )
    head_rows = []
    for row in by_head.sort_values("pnl_net_spread", ascending=False).itertuples(index=False):
        head_rows.append(
            f"| {row.head} | {row.pnl_net_spread:.4f} | {int(row.trades)} | "
            f"{_format_pct(float(row.hit_rate))} | {row.pnl_per_trade:.5f} |"
        )
    block = f"""

### {metrics['trial_id']} `{metrics['variant']}`

Updated: {timestamp}

Change: {metrics['description']}

Summary:

| Total PnL | Worst Week | Q15 Week | Q05 Week | Positive Week Frac | Trades | Hit Rate | PnL/Trade |
|---:|---:|---:|---:|---:|---:|---:|---:|
| {metrics['total_pnl_net_spread']:.4f} | {metrics['worst_week_pnl']:.4f} | {metrics['q15_week_pnl']:.4f} | {metrics['q05_week_pnl']:.4f} | {_format_pct(metrics['positive_week_fraction'])} | {metrics['trades']} | {_format_pct(metrics['hit_rate'])} | {metrics['pnl_per_trade']:.5f} |

Week-by-week:

| Week Start | PnL | Trades | Hit Rate | PnL/Trade |
|---|---:|---:|---:|---:|
{chr(10).join(week_rows)}

Per-head:

| Head | PnL | Trades | Hit Rate | PnL/Trade |
|---|---:|---:|---:|---:|
{chr(10).join(head_rows)}

Output folder: `{metrics['output_dir']}`
"""
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(block)


def _write_sweep_outputs(output_root: Path, summaries: list[dict[str, Any]], weekly_parts: list[pd.DataFrame], head_parts: list[pd.DataFrame]) -> None:
    if summaries:
        pd.DataFrame(summaries).to_csv(output_root / "trial_sweep_summary.csv", index=False)
    if weekly_parts:
        pd.concat(weekly_parts, ignore_index=True).to_csv(output_root / "trial_sweep_weekly.csv", index=False)
    if head_parts:
        pd.concat(head_parts, ignore_index=True).to_csv(output_root / "trial_sweep_by_head.csv", index=False)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trials": [trial.__dict__ for trial in TRIALS],
    }
    (output_root / "trial_sweep_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--policy-params", type=Path, default=DEFAULT_POLICY_PARAMS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--trial", action="append", choices=[trial.trial_id for trial in TRIALS])
    args = parser.parse_args()

    selected_trials = [trial for trial in TRIALS if not args.trial or trial.trial_id in set(args.trial)]
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    weekly_parts: list[pd.DataFrame] = []
    head_parts: list[pd.DataFrame] = []
    env = os.environ.copy()
    env["PYTHONPATH"] = "." if not env.get("PYTHONPATH") else f".{os.pathsep}{env['PYTHONPATH']}"

    for trial in selected_trials:
        output_dir = args.output_root / f"{trial.trial_id}_{trial.name}"
        summary_path = output_dir / "calendar_dynamic_hr_surprise_policy_summary.csv"
        if not args.skip_existing or not summary_path.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            cmd = _common_compare_args(args.candidates, args.policy_params, output_dir) + list(trial.overrides)
            print(f"\nRunning {trial.trial_id} {trial.name}", flush=True)
            subprocess.run(cmd, check=True, env=env)
        else:
            print(f"\nSkipping existing {trial.trial_id} {trial.name}", flush=True)
        metrics, weekly, by_head = _trial_metrics(output_dir, trial)
        summaries.append(metrics)
        weekly_parts.append(weekly)
        head_parts.append(by_head)
        _write_sweep_outputs(args.output_root, summaries, weekly_parts, head_parts)
        _append_ledger(args.ledger, metrics, weekly, by_head)
        print(
            f"{trial.trial_id}: pnl={metrics['total_pnl_net_spread']:.4f} "
            f"worst_week={metrics['worst_week_pnl']:.4f} "
            f"trades={metrics['trades']} hr={_format_pct(metrics['hit_rate'])}",
            flush=True,
        )


if __name__ == "__main__":
    main()
