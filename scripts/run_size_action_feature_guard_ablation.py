#!/usr/bin/env python3
"""Replay size-action schedules with simple feature guards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.portfolio_policy_replay import fit_hierarchical_ev_curves, normalise_candidate_table
from scripts.run_global_portfolio_period_multiplier import _load_policy_params
from scripts.run_size_action_live_scorer_replay import _head_from_strategy, _load_candidates, _replay, _summarise


def _load_feature_frame(path: Path, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.loc[frame["timestamp"].ge(start) & frame["timestamp"].lt(end)].copy()
    frame["strategy_id"] = frame["strategy_id"].astype(str)
    keep = [
        "timestamp",
        "strategy_id",
        "head",
        "strategy_candidate_count",
        "timestamp_above_threshold_count",
        "strategy_above_threshold_count",
        "timestamp_symbol_count",
        "timestamp_candidate_count",
        "remaining_capital",
        "wallet",
        "affected_notional",
        "projected_removed_trade_count",
    ]
    return frame[[col for col in keep if col in frame.columns]].drop_duplicates(["timestamp", "strategy_id"])


def _guard_schedule(schedule: pd.DataFrame, features: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = schedule.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["strategy_id"] = out["strategy_id"].astype(str)
    out["head"] = out.get("head", out["strategy_id"].map(_head_from_strategy)).astype(str)
    out["multiplier"] = pd.to_numeric(out.get("multiplier"), errors="coerce").fillna(1.0).clip(0.0, 1.0)
    merged = out.merge(features, on=["timestamp", "strategy_id"], how="left", suffixes=("", "_feature"))
    guard_applies = pd.Series(False, index=merged.index)
    guard_pass = pd.Series(True, index=merged.index)
    for head, rules in (config.get("head_rules") or {}).items():
        head_mask = merged["head"].astype(str).eq(str(head)) & merged["multiplier"].lt(1.0)
        if not head_mask.any():
            continue
        guard_applies = guard_applies | head_mask
        for feature, threshold in (rules.get("min", {}) or {}).items():
            values = pd.to_numeric(merged.get(feature), errors="coerce")
            guard_pass = guard_pass & (~head_mask | values.ge(float(threshold)))
        for feature, threshold in (rules.get("max", {}) or {}).items():
            values = pd.to_numeric(merged.get(feature), errors="coerce")
            guard_pass = guard_pass & (~head_mask | values.le(float(threshold)))
    blocked = guard_applies & ~guard_pass
    merged["guard_applies"] = guard_applies
    merged["guard_pass"] = guard_pass
    merged["guard_blocked"] = blocked
    merged.loc[blocked, "multiplier"] = 1.0
    return merged


def _configs() -> dict[str, dict[str, Any]]:
    return {
        "C3el_shared_gate": {"head_rules": {}},
        "short_asset_breadth_medium": {
            "head_rules": {
                "short_asset": {
                    "min": {
                        "strategy_candidate_count": 25,
                        "timestamp_above_threshold_count": 22,
                        "strategy_above_threshold_count": 18,
                    }
                }
            }
        },
        "short_asset_breadth_strict": {
            "head_rules": {
                "short_asset": {
                    "min": {
                        "strategy_candidate_count": 30,
                        "timestamp_above_threshold_count": 25,
                        "strategy_above_threshold_count": 20,
                    }
                }
            }
        },
        "short_asset_breadth_very_strict": {
            "head_rules": {
                "short_asset": {
                    "min": {
                        "strategy_candidate_count": 35,
                        "timestamp_above_threshold_count": 30,
                        "strategy_above_threshold_count": 25,
                    }
                }
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broad-candidates", type=Path, required=True)
    parser.add_argument("--deployable-candidates", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--training-frame", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start", default="2026-05-29T00:00:00+00:00")
    parser.add_argument("--end", default="2026-06-26T00:00:00+00:00")
    parser.add_argument("--market-mode", default="perps")
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
    schedule = pd.read_csv(args.schedule)
    schedule["timestamp"] = pd.to_datetime(schedule["timestamp"], utc=True, errors="coerce")
    schedule = schedule.loc[schedule["timestamp"].ge(start) & schedule["timestamp"].lt(end)].copy()
    features = _load_feature_frame(args.training_frame, start=start, end=end)

    accepted_frames: list[pd.DataFrame] = []
    schedule_rows: list[pd.DataFrame] = []
    baseline, _baseline_metrics = _replay(candidates, params, ev_curve, market_mode=args.market_mode, arm="C0_baseline")
    accepted_frames.append(baseline)
    trial_rows: list[dict[str, Any]] = []
    for name, config in _configs().items():
        guarded = _guard_schedule(schedule, features, config)
        replay_schedule = guarded[["timestamp", "strategy_id", "multiplier"]]
        accepted, _metrics = _replay(candidates, params, ev_curve, market_mode=args.market_mode, arm=name, schedule=replay_schedule)
        accepted_frames.append(accepted)
        schedule_out = guarded[["timestamp", "strategy_id", "head", "multiplier", "guard_applies", "guard_pass", "guard_blocked"]].copy()
        schedule_out["arm"] = name
        schedule_rows.append(schedule_out)
        overall = _summarise(accepted, ["arm"])
        row = {"arm": name, "guarded_rows": int(guarded["guard_blocked"].sum())}
        if not overall.empty:
            for col in ("trade_count", "net_hit_rate_pct", "net_pnl", "net_ev_bps_turnover", "full_sl_rate_pct"):
                row[col] = float(overall[col].iloc[0])
        trial_rows.append(row)

    accepted_all = pd.concat(accepted_frames, ignore_index=True)
    accepted_all.to_csv(args.out_dir / "accepted_trades.csv", index=False)
    pd.concat(schedule_rows, ignore_index=True).to_csv(args.out_dir / "guarded_schedules.csv", index=False)
    trials = pd.DataFrame(trial_rows).sort_values("net_pnl", ascending=False).reset_index(drop=True)
    trials.to_csv(args.out_dir / "feature_guard_trials.csv", index=False)
    for keys, name in [
        (["arm"], "overall"),
        (["arm", "head"], "by_head"),
        (["arm", "week_start"], "weekly"),
        (["arm", "week_start", "head"], "weekly_by_head"),
    ]:
        _summarise(accepted_all, keys).to_csv(args.out_dir / f"{name}.csv", index=False)
    manifest = {
        "generated_by": "run_size_action_feature_guard_ablation",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "schedule": str(args.schedule),
        "training_frame": str(args.training_frame),
        "configs": _configs(),
        "best_arm": str(trials.iloc[0]["arm"]) if not trials.empty else None,
        "out_dir": str(args.out_dir),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    print(trials.to_string(index=False))


if __name__ == "__main__":
    main()
