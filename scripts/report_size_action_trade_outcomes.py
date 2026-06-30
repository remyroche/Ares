#!/usr/bin/env python3
"""Replay size-action arms and report accepted-trade outcome metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.portfolio_policy_replay import fit_hierarchical_ev_curves, normalise_candidate_table
from scripts.run_exact_state_size_action_learning import (
    DEFAULT_POLICY_MANIFEST,
    DEFAULT_TRAIN_BROAD,
    DEFAULT_TRAIN_DEPLOYABLE,
    _replay_arm,
)
from scripts.run_global_portfolio_period_multiplier import _load_candidates, _load_policy_params
from scripts.run_global_portfolio_period_multiplier_walkforward import _build_folds, _timestamp_mask


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _split_arms(raw: str, schedules: pd.DataFrame) -> list[str]:
    if raw.strip().lower() == "all":
        return sorted(str(x) for x in schedules["arm"].dropna().unique())
    return [part.strip() for part in raw.split(",") if part.strip()]


def _accepted_with_outcomes(accepted: pd.DataFrame, *, arm: str, fold_id: int, eval_start: pd.Timestamp, eval_end: pd.Timestamp) -> pd.DataFrame:
    if accepted.empty:
        return pd.DataFrame()
    out = accepted.copy()
    out["arm"] = str(arm)
    out["fold_id"] = int(fold_id)
    out["eval_start"] = eval_start
    out["eval_end"] = eval_end
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if "exit_timestamp" in out.columns:
        out["exit_timestamp"] = pd.to_datetime(out["exit_timestamp"], utc=True, errors="coerce")
    for col in ["position_size", "net_return", "gross_return", "net_pnl", "gross_pnl", "cost_pnl"]:
        out[col] = pd.to_numeric(out.get(col), errors="coerce").fillna(0.0)
    out["net_win"] = out["net_pnl"] > 0.0
    out["gross_win"] = out["gross_pnl"] > 0.0
    reason = out.get("simple_policy_exit_reason", pd.Series("", index=out.index)).astype(str).str.lower()
    out["full_sl"] = reason.isin(["sl", "full_sl", "stop", "stop_loss"])
    out["timeout"] = reason.str.contains("timeout", regex=False)
    return out


def _summarise(group: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if group.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key_vals, g in group.groupby(keys, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        row = dict(zip(keys, key_vals))
        trade_count = int(len(g))
        turnover = float(g["position_size"].sum())
        net_pnl = float(g["net_pnl"].sum())
        gross_pnl = float(g["gross_pnl"].sum())
        cost_pnl = float(g["cost_pnl"].sum())
        gross_abs = float(g["gross_pnl"].abs().sum())
        row.update(
            {
                "trade_count": trade_count,
                "net_hit_rate": float(g["net_win"].mean()) if trade_count else 0.0,
                "gross_hit_rate": float(g["gross_win"].mean()) if trade_count else 0.0,
                "net_pnl": net_pnl,
                "gross_pnl": gross_pnl,
                "cost_pnl": cost_pnl,
                "cost_to_abs_gross": float(cost_pnl / max(gross_abs, 1e-9)),
                "notional_turnover": turnover,
                "net_ev_per_trade": float(net_pnl / max(trade_count, 1)),
                "gross_ev_per_trade": float(gross_pnl / max(trade_count, 1)),
                "cost_per_trade": float(cost_pnl / max(trade_count, 1)),
                "net_ev_bps_turnover": float(net_pnl / max(turnover, 1e-9) * 10000.0),
                "gross_ev_bps_turnover": float(gross_pnl / max(turnover, 1e-9) * 10000.0),
                "cost_bps_turnover": float(cost_pnl / max(turnover, 1e-9) * 10000.0),
                "full_sl_rate": float(g["full_sl"].mean()) if trade_count else 0.0,
                "timeout_rate": float(g["timeout"].mean()) if trade_count else 0.0,
                "mean_net_return": float(g["net_return"].mean()) if trade_count else 0.0,
                "median_net_return": float(g["net_return"].median()) if trade_count else 0.0,
                "q05_net_return": float(g["net_return"].quantile(0.05)) if trade_count else 0.0,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _write_markdown(path: Path, payload: dict[str, Any], summaries: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# Size-Action Accepted Trade Outcome Report",
        "",
        f"Run: `{payload['run_dir']}`",
        f"Arms: `{', '.join(payload['arms'])}`",
        f"Costs/spread: `net_pnl = gross_pnl - cost_pnl`; cost_pnl is from the portfolio replay candidate net/gross returns.",
        "",
    ]
    for name, frame in summaries.items():
        lines.extend([f"## {name}", ""])
        if frame.empty:
            lines.extend(["No rows.", ""])
            continue
        show = frame.copy()
        for col in show.columns:
            if pd.api.types.is_datetime64_any_dtype(show[col]):
                show[col] = show[col].astype(str)
        lines.extend(show.to_markdown(index=False, floatfmt=".6f").splitlines())
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arms", default="C0_baseline,C1_exact_state_oracle_full,C3el_bagged_safety_c3ed_or_high_value_zero_classifier_broad_union_gate")
    parser.add_argument("--broad-candidates", type=Path, default=DEFAULT_TRAIN_BROAD)
    parser.add_argument("--deployable-candidates", type=Path, default=DEFAULT_TRAIN_DEPLOYABLE)
    parser.add_argument("--policy-manifest", type=Path, default=DEFAULT_POLICY_MANIFEST)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--min-train-hours", type=int, default=None)
    parser.add_argument("--fold-hours", type=int, default=None)
    parser.add_argument("--embargo-hours", type=int, default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--market-mode", default="perps")
    args = parser.parse_args()

    manifest = _read_manifest(args.run_dir / "manifest.json")
    min_train_hours = int(args.min_train_hours if args.min_train_hours is not None else manifest.get("min_train_hours", 336))
    fold_hours = int(args.fold_hours if args.fold_hours is not None else manifest.get("fold_hours", 168))
    embargo_hours = int(args.embargo_hours if args.embargo_hours is not None else manifest.get("embargo_hours", 96))
    max_folds = int(args.max_folds if args.max_folds is not None else manifest.get("fold_count", 6))

    params, _policy_payload = _load_policy_params(args.policy_manifest, args.policy_variant)
    broad = normalise_candidate_table(_load_candidates(args.broad_candidates))
    deployable = normalise_candidate_table(_load_candidates(args.deployable_candidates))
    schedules = pd.read_csv(args.run_dir / "size_action_schedules.csv")
    schedules["timestamp"] = pd.to_datetime(schedules["timestamp"], utc=True, errors="coerce")
    arms = _split_arms(args.arms, schedules)

    folds = _build_folds(
        broad["timestamp"],
        min_train_hours=min_train_hours,
        fold_hours=fold_hours,
        embargo_hours=embargo_hours,
        max_folds=max_folds,
    )
    accepted_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_id = int(fold["fold_id"])
        train_end = pd.Timestamp(fold["train_end"])
        eval_start = pd.Timestamp(fold["eval_start"])
        eval_end = pd.Timestamp(fold["eval_end"]) + pd.Timedelta(nanoseconds=1)
        eval_candidates = broad.loc[_timestamp_mask(broad, start=eval_start, end=eval_end)].copy()
        if eval_candidates.empty:
            continue
        ev_curve = fit_hierarchical_ev_curves(
            deployable.loc[_timestamp_mask(deployable, end=train_end + pd.Timedelta(nanoseconds=1))].copy()
        )
        for arm in arms:
            schedule = schedules.loc[
                (schedules["fold_id"].eq(fold_id)) & schedules["arm"].astype(str).eq(str(arm))
            ].copy()
            row, accepted = _replay_arm(arm, eval_candidates, params, ev_curve, schedule, args.market_mode)
            row["fold_id"] = fold_id
            row["eval_start"] = eval_start
            row["eval_end"] = eval_end - pd.Timedelta(nanoseconds=1)
            fold_rows.append(row)
            acc = _accepted_with_outcomes(
                accepted,
                arm=arm,
                fold_id=fold_id,
                eval_start=eval_start,
                eval_end=eval_end - pd.Timedelta(nanoseconds=1),
            )
            if not acc.empty:
                accepted_frames.append(acc)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    accepted_all = pd.concat(accepted_frames, ignore_index=True) if accepted_frames else pd.DataFrame()
    fold_summary = pd.DataFrame(fold_rows)
    if not accepted_all.empty:
        accepted_all["week"] = accepted_all["timestamp"].dt.to_period("W").astype(str)
        accepted_all["month"] = accepted_all["timestamp"].dt.to_period("M").astype(str)
    overall = _summarise(accepted_all, ["arm"])
    by_fold = _summarise(accepted_all, ["arm", "fold_id", "eval_start", "eval_end"])
    by_week = _summarise(accepted_all, ["arm", "week"])
    by_month = _summarise(accepted_all, ["arm", "month"])
    by_strategy = _summarise(accepted_all, ["arm", "strategy_id"])
    accepted_all.to_csv(args.out_dir / "size_action_accepted_trades.csv", index=False)
    fold_summary.to_csv(args.out_dir / "size_action_replayed_fold_summary.csv", index=False)
    for name, frame in {
        "overall": overall,
        "by_fold": by_fold,
        "by_week": by_week,
        "by_month": by_month,
        "by_strategy": by_strategy,
    }.items():
        frame.to_csv(args.out_dir / f"size_action_trade_metrics_{name}.csv", index=False)
    payload = {
        "run_dir": str(args.run_dir),
        "arms": arms,
        "min_train_hours": min_train_hours,
        "fold_hours": fold_hours,
        "embargo_hours": embargo_hours,
        "max_folds": max_folds,
        "market_mode": args.market_mode,
    }
    (args.out_dir / "size_action_trade_outcome_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    _write_markdown(
        args.out_dir / "size_action_trade_outcome_report.md",
        payload,
        {
            "Overall": overall,
            "By Fold": by_fold,
            "By Week": by_week,
            "By Month": by_month,
            "By Strategy": by_strategy,
        },
    )
    print({"out_dir": str(args.out_dir), "accepted_trades": int(len(accepted_all)), "arms": arms})


if __name__ == "__main__":
    main()
