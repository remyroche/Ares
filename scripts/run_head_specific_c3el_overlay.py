#!/usr/bin/env python3
"""Legacy replay of C3el with independent per-head intervention strengths.

The frozen C3el scorer emits one size multiplier per timestamp/strategy group.
This script leaves the scorer untouched and adds an auditable policy layer:

    adjusted_multiplier = 1 - strength_head * (1 - scorer_multiplier)

where strength_head=0 disables C3el for that head and strength_head=1 keeps the
frozen scorer's original action. Intermediate values soften the size cut.

This is not the current head-specific C3el implementation. Use
``scripts/run_head_native_c3el_action_learner.py`` for head-native C3el: it
fits feature selection, classifier, action-value model, thresholds, caps, and
guards separately per head before assembling the final replay schedule.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.portfolio_policy_replay import (
    fit_hierarchical_ev_curves,
    normalise_candidate_table,
    replay_candidates,
)
from scripts.run_exact_state_size_action_learning import _accepted_trades, _apply_size_schedule
from scripts.run_global_portfolio_period_multiplier import _load_policy_params


HEADS = ("long_bars", "long_dist", "short_asset", "short_boll")


def _head_from_strategy(strategy_id: Any) -> str:
    text = str(strategy_id)
    for head in HEADS:
        if text.startswith(head):
            return head
    return text.split("_", 1)[0] if text else "unknown"


def _load_candidates(path: Path, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = normalise_candidate_table(pd.read_parquet(path))
    return frame.loc[frame["timestamp"].ge(start) & frame["timestamp"].lt(end)].copy()


def _summarise(group: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if group.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key_vals, g in group.groupby(keys, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        row = dict(zip(keys, key_vals))
        n = int(len(g))
        turnover = float(g["position_size"].sum())
        net = float(g["net_pnl"].sum())
        gross = float(g["gross_pnl"].sum())
        cost = float(g["cost_pnl"].sum())
        row.update(
            {
                "trade_count": n,
                "net_hit_rate_pct": float(g["net_win"].mean() * 100.0) if n else np.nan,
                "gross_hit_rate_pct": float(g["gross_win"].mean() * 100.0) if n else np.nan,
                "net_pnl": net,
                "gross_pnl": gross,
                "cost_pnl": cost,
                "notional_turnover": turnover,
                "net_ev_per_trade": float(net / max(n, 1)),
                "net_ev_bps_turnover": float(net / max(turnover, 1e-9) * 10000.0),
                "cost_bps_turnover": float(cost / max(turnover, 1e-9) * 10000.0),
                "full_sl_rate_pct": float(g["full_sl"].mean() * 100.0) if n else np.nan,
                "timeout_rate_pct": float(g["timeout"].mean() * 100.0) if n else np.nan,
                "mean_net_return_pct": float(g["net_return"].mean() * 100.0) if n else np.nan,
                "q05_net_return_pct": float(g["net_return"].quantile(0.05) * 100.0) if n else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _prepare_accepted(accepted: pd.DataFrame, arm: str) -> pd.DataFrame:
    out = accepted.copy()
    out["arm"] = str(arm)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if "exit_timestamp" in out.columns:
        out["exit_timestamp"] = pd.to_datetime(out["exit_timestamp"], utc=True, errors="coerce")
    for col in ("position_size", "net_return", "gross_return", "net_pnl", "gross_pnl", "cost_pnl"):
        out[col] = pd.to_numeric(out.get(col), errors="coerce").fillna(0.0)
    if "head" not in out.columns:
        out["head"] = out["strategy_id"].map(_head_from_strategy)
    out["net_win"] = out["net_pnl"] > 0.0
    out["gross_win"] = out["gross_pnl"] > 0.0
    reason = out.get("simple_policy_exit_reason", pd.Series("", index=out.index)).astype(str).str.lower()
    out["full_sl"] = reason.isin(["sl", "full_sl", "stop", "stop_loss"])
    out["timeout"] = reason.str.contains("timeout", regex=False)
    out["week_start"] = out["timestamp"].dt.to_period("W-SUN").dt.start_time.dt.tz_localize("UTC")
    out["month"] = out["timestamp"].dt.to_period("M").astype(str)
    return out


def _adjust_schedule(schedule: pd.DataFrame, strengths: dict[str, float]) -> pd.DataFrame:
    out = schedule.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["strategy_id"] = out["strategy_id"].astype(str)
    out["head"] = out["strategy_id"].map(_head_from_strategy)
    raw_multiplier = pd.to_numeric(out.get("multiplier"), errors="coerce")
    if raw_multiplier.isna().all() and "selected_multiplier" in out.columns:
        raw_multiplier = pd.to_numeric(out["selected_multiplier"], errors="coerce")
    raw_multiplier = raw_multiplier.fillna(1.0).clip(lower=0.0, upper=1.0)
    strength = out["head"].map(lambda head: float(strengths.get(str(head), 0.0))).astype(float).clip(0.0, 1.0)
    adjusted = 1.0 - strength * (1.0 - raw_multiplier)
    out["raw_multiplier"] = raw_multiplier
    out["head_strength"] = strength
    out["multiplier"] = adjusted.clip(lower=0.0, upper=1.0)
    return out


def _week_start(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True, errors="coerce").dt.to_period("W-SUN").dt.start_time.dt.tz_localize("UTC")


def _adjust_schedule_by_week(
    schedule: pd.DataFrame,
    weekly_strengths: dict[pd.Timestamp, dict[str, float]],
    *,
    default_strengths: dict[str, float],
) -> pd.DataFrame:
    out = schedule.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["strategy_id"] = out["strategy_id"].astype(str)
    out["head"] = out["strategy_id"].map(_head_from_strategy)
    out["week_start"] = _week_start(out["timestamp"])
    raw_multiplier = pd.to_numeric(out.get("multiplier"), errors="coerce")
    if raw_multiplier.isna().all() and "selected_multiplier" in out.columns:
        raw_multiplier = pd.to_numeric(out["selected_multiplier"], errors="coerce")
    raw_multiplier = raw_multiplier.fillna(1.0).clip(lower=0.0, upper=1.0)

    def row_strength(row: pd.Series) -> float:
        week_strengths = weekly_strengths.get(pd.Timestamp(row["week_start"]), default_strengths)
        return float(week_strengths.get(str(row["head"]), default_strengths.get(str(row["head"]), 0.0)))

    strength = out.apply(row_strength, axis=1).astype(float).clip(0.0, 1.0)
    out["raw_multiplier"] = raw_multiplier
    out["head_strength"] = strength
    out["multiplier"] = (1.0 - strength * (1.0 - raw_multiplier)).clip(lower=0.0, upper=1.0)
    return out


def _replay_with_strengths(
    candidates: pd.DataFrame,
    params: Any,
    ev_curve: dict[str, Any],
    schedule: pd.DataFrame,
    strengths: dict[str, float],
    *,
    arm: str,
    market_mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    adjusted = _adjust_schedule(schedule, strengths)
    arm_candidates = _apply_size_schedule(candidates, adjusted[["timestamp", "strategy_id", "multiplier"]])
    decisions, _equity, metrics = replay_candidates(
        arm_candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    accepted = _prepare_accepted(_accepted_trades(arm_candidates, decisions), arm)
    return accepted, metrics


def _objective(
    accepted: pd.DataFrame,
    *,
    worst_week_weight: float,
    full_sl_penalty: float,
) -> float:
    if accepted.empty:
        return float("-inf")
    overall = _summarise(accepted, ["arm"])
    net_pnl = float(overall["net_pnl"].iloc[0])
    full_sl = float(overall["full_sl_rate_pct"].iloc[0])
    weekly = _summarise(accepted, ["week_start"])
    worst_week = float(weekly["net_pnl"].min()) if not weekly.empty else 0.0
    return net_pnl + float(worst_week_weight) * worst_week - float(full_sl_penalty) * full_sl


def _grid_values(raw: str) -> list[float]:
    vals = sorted({float(x.strip()) for x in raw.split(",") if x.strip()})
    if not vals:
        raise ValueError("strength grid cannot be empty")
    for val in vals:
        if val < 0.0 or val > 1.0:
            raise ValueError(f"strength grid value outside [0, 1]: {val}")
    return vals


def _format_strengths(strengths: dict[str, float]) -> str:
    return ",".join(f"{head}={strengths.get(head, 0.0):.2f}" for head in HEADS)


def _select_walkforward_weekly_strengths(
    tried: dict[tuple[float, ...], tuple[float, pd.DataFrame]],
    weeks: list[pd.Timestamp],
    *,
    default_strengths: dict[str, float],
    worst_week_weight: float,
    full_sl_penalty: float,
) -> tuple[dict[pd.Timestamp, dict[str, float]], pd.DataFrame]:
    weekly_strengths: dict[pd.Timestamp, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    default_key = tuple(float(default_strengths.get(head, 0.0)) for head in HEADS)
    for week in weeks:
        week = pd.Timestamp(week)
        best_key = default_key
        best_score = float("-inf")
        train_weeks = 0
        for key, (_full_score, accepted) in tried.items():
            if accepted.empty or "week_start" not in accepted.columns:
                continue
            prior = accepted.loc[pd.to_datetime(accepted["week_start"], utc=True, errors="coerce").lt(week)].copy()
            train_weeks = max(train_weeks, int(prior["week_start"].nunique()) if not prior.empty else 0)
            if prior.empty:
                continue
            score = _objective(prior, worst_week_weight=worst_week_weight, full_sl_penalty=full_sl_penalty)
            if score > best_score:
                best_score = score
                best_key = key
        selected = {head: float(best_key[idx]) for idx, head in enumerate(HEADS)}
        weekly_strengths[week] = selected
        rows.append(
            {
                "week_start": week,
                "train_weeks": int(train_weeks),
                "prior_objective": float(best_score) if np.isfinite(best_score) else 0.0,
                "config": _format_strengths(selected),
                **{f"strength_{head}": selected[head] for head in HEADS},
            }
        )
    return weekly_strengths, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broad-candidates", type=Path, required=True)
    parser.add_argument("--deployable-candidates", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start", default="2026-05-29T00:00:00+00:00")
    parser.add_argument("--end", default="2026-06-26T00:00:00+00:00")
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--strength-grid", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--search", choices=["greedy", "exhaustive"], default="greedy")
    parser.add_argument("--max-greedy-passes", type=int, default=3)
    parser.add_argument("--worst-week-weight", type=float, default=0.0)
    parser.add_argument("--full-sl-penalty", type=float, default=0.0)
    parser.add_argument("--walkforward-strengths", action="store_true")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    params, _payload = _load_policy_params(args.policy_manifest, args.policy_variant)
    candidates = _load_candidates(args.broad_candidates, start=start, end=end)
    deployable = normalise_candidate_table(pd.read_parquet(args.deployable_candidates))
    deployable_train = deployable.loc[deployable["timestamp"].lt(start)].copy()
    ev_curve = fit_hierarchical_ev_curves(deployable_train if not deployable_train.empty else deployable)
    schedule = pd.read_csv(args.schedule)
    schedule["timestamp"] = pd.to_datetime(schedule["timestamp"], utc=True, errors="coerce")
    schedule = schedule.loc[schedule["timestamp"].ge(start) & schedule["timestamp"].lt(end)].copy()
    grid = _grid_values(args.strength_grid)

    tried: dict[tuple[float, ...], tuple[float, pd.DataFrame]] = {}
    trial_rows: list[dict[str, Any]] = []

    def evaluate(strengths: dict[str, float], name: str) -> float:
        key = tuple(float(strengths.get(head, 0.0)) for head in HEADS)
        if key in tried:
            return tried[key][0]
        accepted, _metrics = _replay_with_strengths(
            candidates,
            params,
            ev_curve,
            schedule,
            strengths,
            arm=name,
            market_mode=args.market_mode,
        )
        score = _objective(
            accepted,
            worst_week_weight=args.worst_week_weight,
            full_sl_penalty=args.full_sl_penalty,
        )
        tried[key] = (score, accepted)
        overall = _summarise(accepted, ["arm"])
        weekly = _summarise(accepted, ["week_start"])
        trial = {
            "config": _format_strengths(strengths),
            "objective": score,
            "worst_week_net_pnl": float(weekly["net_pnl"].min()) if not weekly.empty else np.nan,
            **{f"strength_{head}": strengths.get(head, 0.0) for head in HEADS},
        }
        if not overall.empty:
            for col in (
                "trade_count",
                "net_hit_rate_pct",
                "net_pnl",
                "notional_turnover",
                "net_ev_bps_turnover",
                "full_sl_rate_pct",
                "timeout_rate_pct",
            ):
                trial[col] = float(overall[col].iloc[0])
        trial_rows.append(trial)
        return score

    baseline_strengths = {head: 0.0 for head in HEADS}
    original_strengths = {head: 1.0 for head in HEADS}
    evaluate(baseline_strengths, "C0_baseline")
    evaluate(original_strengths, "C3el_original")

    if args.search == "exhaustive":
        for values in product(grid, repeat=len(HEADS)):
            evaluate(dict(zip(HEADS, values)), "candidate")
    else:
        current = original_strengths.copy()
        best_score = evaluate(current, "candidate")
        for _pass in range(max(int(args.max_greedy_passes), 1)):
            improved = False
            for head in HEADS:
                local_best = current.copy()
                local_score = best_score
                for value in grid:
                    candidate = current.copy()
                    candidate[head] = float(value)
                    score = evaluate(candidate, "candidate")
                    if score > local_score:
                        local_best = candidate
                        local_score = score
                if local_score > best_score:
                    current = local_best
                    best_score = local_score
                    improved = True
            if not improved:
                break

    trials = pd.DataFrame(trial_rows).drop_duplicates(subset=[f"strength_{head}" for head in HEADS])
    trials = trials.sort_values("objective", ascending=False).reset_index(drop=True)
    trials.to_csv(args.out_dir / "head_specific_c3el_trials.csv", index=False)

    best = trials.iloc[0].to_dict()
    best_strengths = {head: float(best[f"strength_{head}"]) for head in HEADS}
    best_key = tuple(best_strengths[head] for head in HEADS)
    best_accepted = tried[best_key][1].copy()
    best_accepted["arm"] = "C3el_head_specific_best"
    baseline = tried[tuple(0.0 for _ in HEADS)][1].copy()
    baseline["arm"] = "C0_baseline"
    original = tried[tuple(1.0 for _ in HEADS)][1].copy()
    original["arm"] = "C3el_original"
    selected_frames = [baseline, original, best_accepted]
    weekly_strength_frame = pd.DataFrame()
    if bool(args.walkforward_strengths):
        weeks = sorted(pd.Series(_week_start(schedule["timestamp"])).dropna().drop_duplicates())
        weekly_strengths, weekly_strength_frame = _select_walkforward_weekly_strengths(
            tried,
            weeks,
            default_strengths=baseline_strengths,
            worst_week_weight=float(args.worst_week_weight),
            full_sl_penalty=float(args.full_sl_penalty),
        )
        dynamic_schedule = _adjust_schedule_by_week(schedule, weekly_strengths, default_strengths=baseline_strengths)
        dynamic_candidates = _apply_size_schedule(candidates, dynamic_schedule[["timestamp", "strategy_id", "multiplier"]])
        dynamic_decisions, _dynamic_equity, _dynamic_metrics = replay_candidates(
            dynamic_candidates,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode=args.market_mode,
        )
        dynamic = _prepare_accepted(_accepted_trades(dynamic_candidates, dynamic_decisions), "C3el_walkforward_strength")
        selected_frames.append(dynamic)
        dynamic_schedule.to_csv(args.out_dir / "head_specific_c3el_walkforward_schedule.csv", index=False)
        weekly_strength_frame.to_csv(args.out_dir / "head_specific_c3el_weekly_strengths.csv", index=False)
    dedupe_subset = ["arm", "timestamp", "strategy_id"]
    if "symbol" in baseline.columns:
        dedupe_subset.append("symbol")
    accepted_all = pd.concat(selected_frames, ignore_index=True).drop_duplicates(subset=dedupe_subset, keep="first")
    accepted_all.to_csv(args.out_dir / "head_specific_c3el_accepted_trades.csv", index=False)

    summaries: dict[str, pd.DataFrame] = {}
    for keys, name in [
        (["arm"], "overall"),
        (["arm", "head"], "by_head"),
        (["arm", "week_start"], "weekly"),
        (["arm", "week_start", "head"], "weekly_by_head"),
        (["arm", "month"], "monthly"),
        (["arm", "month", "head"], "monthly_by_head"),
    ]:
        frame = _summarise(accepted_all, keys)
        summaries[name] = frame
        frame.to_csv(args.out_dir / f"head_specific_c3el_{name}.csv", index=False)

    weekly_head = summaries["weekly_by_head"]
    baseline_idx = weekly_head.loc[weekly_head["arm"].eq("C0_baseline")].set_index(["week_start", "head"])
    best_idx = weekly_head.loc[weekly_head["arm"].eq("C3el_head_specific_best")].set_index(["week_start", "head"])
    delta_rows: list[dict[str, Any]] = []
    for key in baseline_idx.index.union(best_idx.index):
        base = baseline_idx.loc[key] if key in baseline_idx.index else None
        best_row = best_idx.loc[key] if key in best_idx.index else None
        row = {"week_start": key[0], "head": key[1]}
        for prefix, record in [("baseline", base), ("head_specific", best_row)]:
            if record is None:
                row.update(
                    {
                        f"{prefix}_trade_count": 0.0,
                        f"{prefix}_net_hit_rate_pct": np.nan,
                        f"{prefix}_net_pnl": 0.0,
                        f"{prefix}_net_ev_bps_turnover": np.nan,
                        f"{prefix}_full_sl_rate_pct": np.nan,
                    }
                )
            else:
                row.update(
                    {
                        f"{prefix}_trade_count": float(record["trade_count"]),
                        f"{prefix}_net_hit_rate_pct": float(record["net_hit_rate_pct"]),
                        f"{prefix}_net_pnl": float(record["net_pnl"]),
                        f"{prefix}_net_ev_bps_turnover": float(record["net_ev_bps_turnover"]),
                        f"{prefix}_full_sl_rate_pct": float(record["full_sl_rate_pct"]),
                    }
                )
        row["delta_net_pnl"] = row["head_specific_net_pnl"] - row["baseline_net_pnl"]
        row["delta_net_hit_rate_pp"] = (
            row["head_specific_net_hit_rate_pct"] - row["baseline_net_hit_rate_pct"]
            if np.isfinite(row["head_specific_net_hit_rate_pct"]) and np.isfinite(row["baseline_net_hit_rate_pct"])
            else np.nan
        )
        delta_rows.append(row)
    deltas = pd.DataFrame(delta_rows).sort_values(["week_start", "head"])
    deltas.to_csv(args.out_dir / "head_specific_c3el_weekly_by_head_vs_baseline.csv", index=False)

    manifest = {
        "generated_by": "run_head_specific_c3el_overlay",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "search": args.search,
        "strength_grid": grid,
        "best_strengths": best_strengths,
        "best_objective": float(best["objective"]),
        "worst_week_weight": float(args.worst_week_weight),
        "full_sl_penalty": float(args.full_sl_penalty),
        "walkforward_strengths": bool(args.walkforward_strengths),
        "broad_candidates": str(args.broad_candidates),
        "deployable_candidates": str(args.deployable_candidates),
        "schedule": str(args.schedule),
        "policy_manifest": str(args.policy_manifest),
        "policy_variant": str(args.policy_variant),
        "market_mode": str(args.market_mode),
        "candidate_rows": int(len(candidates)),
        "schedule_rows": int(len(schedule)),
        "trial_count": int(len(trials)),
    }
    (args.out_dir / "head_specific_c3el_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("\nTop trials")
    print(trials.head(10).to_string(index=False))
    print("\nOverall")
    print(summaries["overall"].to_string(index=False))


if __name__ == "__main__":
    main()
