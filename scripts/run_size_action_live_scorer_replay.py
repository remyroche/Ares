#!/usr/bin/env python3
"""Replay a materialized size-action live scorer on a fixed candidate universe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.portfolio_policy_replay import (
    fit_hierarchical_ev_curves,
    normalise_candidate_table,
    replay_candidates,
)
from scripts.materialize_size_action_live_scorer import score_size_action_frame
from scripts.run_exact_state_size_action_learning import _accepted_trades, _apply_size_schedule
from scripts.run_global_portfolio_period_multiplier import _load_policy_params


HEADS = ("long_bars", "long_dist", "short_asset", "short_boll")
HEAD_ALIASES = {"short_bollinger": "short_boll"}


def _head_from_strategy(strategy_id: Any) -> str:
    text = str(strategy_id)
    for alias, head in HEAD_ALIASES.items():
        if text == alias or text.startswith(f"{alias}_"):
            return head
    for head in HEADS:
        if text == head or text.startswith(f"{head}_"):
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
    else:
        out["head"] = out["head"].fillna(out["strategy_id"].map(_head_from_strategy)).map(
            lambda value: HEAD_ALIASES.get(str(value), str(value))
        )
    out["net_win"] = out["net_pnl"] > 0.0
    out["gross_win"] = out["gross_pnl"] > 0.0
    reason = out.get("simple_policy_exit_reason", pd.Series("", index=out.index)).astype(str).str.lower()
    out["full_sl"] = reason.isin(["sl", "full_sl", "stop", "stop_loss"])
    out["timeout"] = reason.str.contains("timeout", regex=False)
    out["week_start"] = out["timestamp"].dt.to_period("W-SUN").dt.start_time.dt.tz_localize("UTC")
    out["month"] = out["timestamp"].dt.to_period("M").astype(str)
    return out


def _replay(
    candidates: pd.DataFrame,
    params: Any,
    ev_curve: dict[str, Any],
    *,
    market_mode: str,
    arm: str,
    schedule: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    arm_candidates = candidates
    if schedule is not None and not schedule.empty:
        arm_candidates = _apply_size_schedule(candidates, schedule[["timestamp", "strategy_id", "multiplier"]])
    decisions, _equity, metrics = replay_candidates(
        arm_candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    accepted = _prepare_accepted(_accepted_trades(arm_candidates, decisions), arm)
    return accepted, metrics


def _write_summary(accepted_all: pd.DataFrame, out_dir: Path) -> dict[str, pd.DataFrame]:
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
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broad-candidates", type=Path, required=True)
    parser.add_argument("--deployable-candidates", type=Path, required=True)
    parser.add_argument("--action-features", type=Path, required=True)
    parser.add_argument("--scorer-bundle", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start", default="2026-05-29T00:00:00+00:00")
    parser.add_argument("--end", default="2026-06-26T00:00:00+00:00")
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--scorer-arm-name", default="C3el_live_scorer")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    params, _payload = _load_policy_params(args.policy_manifest, args.policy_variant)
    candidates = _load_candidates(args.broad_candidates, start=start, end=end)
    deployable = normalise_candidate_table(pd.read_parquet(args.deployable_candidates))
    deployable_train = deployable.loc[deployable["timestamp"].lt(start)].copy()
    ev_curve = fit_hierarchical_ev_curves(deployable_train if not deployable_train.empty else deployable)

    action_features = pd.read_parquet(args.action_features)
    action_features["timestamp"] = pd.to_datetime(action_features["timestamp"], utc=True, errors="coerce")
    action_features = action_features.loc[action_features["timestamp"].ge(start) & action_features["timestamp"].lt(end)].copy()
    action_features.to_parquet(args.out_dir / "action_feature_rows.parquet", index=False)

    scored = score_size_action_frame(args.scorer_bundle, action_features)
    scored["timestamp"] = pd.to_datetime(scored["timestamp"], utc=True, errors="coerce")
    scored["multiplier"] = pd.to_numeric(scored.get("selected_multiplier"), errors="coerce").fillna(1.0).clip(0.0, 1.0)
    scored.to_csv(args.out_dir / "size_action_scores.csv", index=False)
    schedule = scored[["timestamp", "strategy_id", "multiplier"]].copy()
    schedule.to_csv(args.out_dir / "size_schedule.csv", index=False)

    baseline, baseline_metrics = _replay(
        candidates,
        params,
        ev_curve,
        market_mode=args.market_mode,
        arm="C0_baseline",
    )
    scorer_accepted, scorer_metrics = _replay(
        candidates,
        params,
        ev_curve,
        market_mode=args.market_mode,
        arm=args.scorer_arm_name,
        schedule=schedule,
    )
    accepted_all = pd.concat([baseline, scorer_accepted], ignore_index=True)
    accepted_all.to_csv(args.out_dir / "accepted_trades.csv", index=False)
    summaries = _write_summary(accepted_all, args.out_dir)

    manifest = {
        "generated_by": "run_size_action_live_scorer_replay",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "broad_candidates": str(args.broad_candidates),
        "deployable_candidates": str(args.deployable_candidates),
        "action_features": str(args.action_features),
        "scorer_bundle": str(args.scorer_bundle),
        "policy_manifest": str(args.policy_manifest),
        "policy_variant": str(args.policy_variant),
        "market_mode": str(args.market_mode),
        "candidate_rows": int(len(candidates)),
        "candidate_timestamps": int(candidates["timestamp"].nunique()) if "timestamp" in candidates.columns else 0,
        "action_feature_rows": int(len(action_features)),
        "schedule_groups": int(len(scored)),
        "interventions": int(pd.to_numeric(scored.get("selected_multiplier"), errors="coerce").fillna(1.0).lt(1.0).sum()),
        "accepted_baseline": int(len(baseline)),
        "accepted_scorer": int(len(scorer_accepted)),
        "baseline_replay_metrics": baseline_metrics,
        "scorer_replay_metrics": scorer_metrics,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    print("\nOverall")
    print(summaries["overall"].to_string(index=False))
    print("\nBy head")
    print(summaries["by_head"].to_string(index=False))


if __name__ == "__main__":
    main()
