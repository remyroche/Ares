#!/usr/bin/env python3
"""Ablate per-head score gates for a materialized size-action scorer schedule."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.portfolio_policy_replay import fit_hierarchical_ev_curves, normalise_candidate_table
from scripts.run_global_portfolio_period_multiplier import _load_policy_params
from scripts.run_size_action_live_scorer_replay import _head_from_strategy, _load_candidates, _replay, _summarise


HEADS = ("long_bars", "long_dist", "short_asset", "short_boll")


def _score_schedule(scores: pd.DataFrame, config: dict[str, float | None]) -> pd.DataFrame:
    out = scores.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["strategy_id"] = out["strategy_id"].astype(str)
    out["head"] = out["strategy_id"].map(_head_from_strategy)
    out["selected_multiplier"] = pd.to_numeric(out.get("selected_multiplier"), errors="coerce").fillna(1.0)
    out["pred_delta_J"] = pd.to_numeric(out.get("pred_delta_J"), errors="coerce").fillna(0.0)
    accepted = out["selected_multiplier"].lt(1.0)
    keep = pd.Series(False, index=out.index)
    for head, threshold in config.items():
        if threshold is None:
            continue
        keep = keep | (out["head"].eq(head) & accepted & out["pred_delta_J"].ge(float(threshold)))
    out["multiplier"] = np.where(keep, out["selected_multiplier"], 1.0)
    return out[["timestamp", "strategy_id", "multiplier"]]


def _threshold_grid(scores: pd.DataFrame, quantiles: list[float]) -> dict[str, dict[float, float]]:
    work = scores.copy()
    work["head"] = work["strategy_id"].astype(str).map(_head_from_strategy)
    work["selected_multiplier"] = pd.to_numeric(work.get("selected_multiplier"), errors="coerce").fillna(1.0)
    work["pred_delta_J"] = pd.to_numeric(work.get("pred_delta_J"), errors="coerce").fillna(0.0)
    work = work.loc[work["selected_multiplier"].lt(1.0)].copy()
    thresholds: dict[str, dict[float, float]] = {}
    for head, group in work.groupby("head"):
        thresholds[str(head)] = {}
        for q in quantiles:
            thresholds[str(head)][float(q)] = float(group["pred_delta_J"].quantile(float(q))) if not group.empty else float("inf")
    return thresholds


def _objective(accepted: pd.DataFrame, *, worst_week_weight: float, full_sl_penalty: float) -> tuple[float, float, float]:
    overall = _summarise(accepted, ["arm"])
    if overall.empty:
        return float("-inf"), 0.0, 0.0
    net_pnl = float(overall["net_pnl"].iloc[0])
    full_sl = float(overall["full_sl_rate_pct"].iloc[0])
    weekly = _summarise(accepted, ["week_start"])
    worst_week = float(weekly["net_pnl"].min()) if not weekly.empty else 0.0
    return net_pnl + float(worst_week_weight) * worst_week - float(full_sl_penalty) * full_sl, worst_week, full_sl


def _format_config(config: dict[str, float | None], thresholds: dict[str, dict[float, float]]) -> str:
    parts: list[str] = []
    for head in HEADS:
        value = config.get(head)
        if value is None:
            parts.append(f"{head}=off")
            continue
        label = None
        for q, threshold in thresholds.get(head, {}).items():
            if np.isclose(float(value), float(threshold), rtol=0.0, atol=1e-9):
                label = f"q{int(round(q * 100))}"
                break
        parts.append(f"{head}={label or f'{float(value):.2f}'}")
    return ",".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broad-candidates", type=Path, required=True)
    parser.add_argument("--deployable-candidates", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start", default="2026-05-29T00:00:00+00:00")
    parser.add_argument("--end", default="2026-06-26T00:00:00+00:00")
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--quantiles", default="0.50,0.75,0.90")
    parser.add_argument("--search", choices=["singles", "exhaustive"], default="exhaustive")
    parser.add_argument("--worst-week-weight", type=float, default=0.0)
    parser.add_argument("--full-sl-penalty", type=float, default=0.0)
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    quantiles = [float(x.strip()) for x in str(args.quantiles).split(",") if x.strip()]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.scores)
    scores["timestamp"] = pd.to_datetime(scores["timestamp"], utc=True, errors="coerce")
    scores = scores.loc[scores["timestamp"].ge(start) & scores["timestamp"].lt(end)].copy()
    thresholds = _threshold_grid(scores, quantiles)

    params, _payload = _load_policy_params(args.policy_manifest, args.policy_variant)
    candidates = _load_candidates(args.broad_candidates, start=start, end=end)
    deployable = normalise_candidate_table(pd.read_parquet(args.deployable_candidates))
    deployable_train = deployable.loc[deployable["timestamp"].lt(start)].copy()
    ev_curve = fit_hierarchical_ev_curves(deployable_train if not deployable_train.empty else deployable)

    tried: dict[str, pd.DataFrame] = {}
    trial_rows: list[dict[str, Any]] = []

    def evaluate(name: str, config: dict[str, float | None]) -> None:
        if name in tried:
            return
        if name == "C0_baseline":
            accepted, _metrics = _replay(candidates, params, ev_curve, market_mode=args.market_mode, arm=name)
        else:
            schedule = _score_schedule(scores, config)
            accepted, _metrics = _replay(candidates, params, ev_curve, market_mode=args.market_mode, arm=name, schedule=schedule)
        score, worst_week, full_sl = _objective(
            accepted,
            worst_week_weight=float(args.worst_week_weight),
            full_sl_penalty=float(args.full_sl_penalty),
        )
        tried[name] = accepted
        overall = _summarise(accepted, ["arm"])
        row = {
            "arm": name,
            "config": _format_config(config, thresholds) if name != "C0_baseline" else "baseline",
            "objective": float(score),
            "worst_week_net_pnl": float(worst_week),
            "full_sl_rate_pct": float(full_sl),
        }
        if not overall.empty:
            for col in ("trade_count", "net_hit_rate_pct", "net_pnl", "gross_pnl", "cost_pnl", "notional_turnover", "net_ev_bps_turnover"):
                row[col] = float(overall[col].iloc[0])
        trial_rows.append(row)

    evaluate("C0_baseline", {head: None for head in HEADS})
    evaluate("raw_scorer", {head: -float("inf") for head in HEADS})
    for head in HEADS:
        for q in quantiles:
            evaluate(f"single_{head}_q{int(round(q * 100))}", {h: (thresholds.get(head, {}).get(q) if h == head else None) for h in HEADS})
    for q in quantiles:
        evaluate(f"all_heads_q{int(round(q * 100))}", {head: thresholds.get(head, {}).get(q) for head in HEADS})

    if args.search == "exhaustive":
        options: list[list[tuple[str, float | None]]] = []
        for head in HEADS:
            head_options: list[tuple[str, float | None]] = [("off", None)]
            for q in quantiles:
                head_options.append((f"q{int(round(q * 100))}", thresholds.get(head, {}).get(q)))
            options.append(head_options)
        for combo in product(*options):
            labels = [label for label, _threshold in combo]
            config = {head: threshold for head, (_label, threshold) in zip(HEADS, combo)}
            evaluate("combo_" + "_".join(f"{head}-{label}" for head, label in zip(HEADS, labels)), config)

    trials = pd.DataFrame(trial_rows).drop_duplicates("arm").sort_values("objective", ascending=False).reset_index(drop=True)
    trials.to_csv(args.out_dir / "score_gate_trials.csv", index=False)
    best_arm = str(trials.iloc[0]["arm"])
    selected_frames: list[pd.DataFrame] = []
    for arm_name in ("C0_baseline", "raw_scorer", best_arm):
        frame = tried.get(arm_name)
        if frame is None or frame.empty:
            continue
        frame = frame.copy()
        frame["arm"] = str(arm_name)
        selected_frames.append(frame)
    dedupe_subset = ["arm", "timestamp", "strategy_id"]
    if selected_frames and "symbol" in selected_frames[0].columns:
        dedupe_subset.append("symbol")
    accepted_all = pd.concat(selected_frames, ignore_index=True).drop_duplicates(subset=dedupe_subset, keep="first")
    accepted_all.to_csv(args.out_dir / "accepted_trades.csv", index=False)
    for keys, name in [
        (["arm"], "overall"),
        (["arm", "head"], "by_head"),
        (["arm", "week_start"], "weekly"),
        (["arm", "week_start", "head"], "weekly_by_head"),
        (["arm", "month"], "monthly"),
        (["arm", "month", "head"], "monthly_by_head"),
    ]:
        _summarise(accepted_all, keys).to_csv(args.out_dir / f"{name}.csv", index=False)

    manifest = {
        "generated_by": "run_size_action_score_gate_ablation",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "scores": str(args.scores),
        "broad_candidates": str(args.broad_candidates),
        "deployable_candidates": str(args.deployable_candidates),
        "policy_manifest": str(args.policy_manifest),
        "policy_variant": str(args.policy_variant),
        "market_mode": str(args.market_mode),
        "search": str(args.search),
        "quantiles": quantiles,
        "thresholds": thresholds,
        "best_arm": best_arm,
        "best_config": str(trials.iloc[0]["config"]),
        "trial_count": int(len(trials)),
        "worst_week_weight": float(args.worst_week_weight),
        "full_sl_penalty": float(args.full_sl_penalty),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    print("\nTop trials")
    print(trials.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
