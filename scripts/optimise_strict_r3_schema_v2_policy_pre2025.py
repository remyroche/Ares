#!/usr/bin/env python3
"""Freeze a long-only SimplePolicyOptimiser geometry on pre-2025 OOF rows.

The trading stack is not refit or selected here.  This action-layer search uses
only strict-prequential 2024 upstream scores and complete 15-minute paths whose
outcomes are fully resolved before 2025-01-01.  The resulting policy is frozen
before every reported 2025--2026 evaluation row.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The declared outcome contract has a flat cost deducted once below.  Disable
# simulator-side spread/gap proxies before importing the module constants.
os.environ.setdefault("EPM_SIMPLE_POLICY_STOP_EXIT_BASE_GAP_BPS", "0")
os.environ.setdefault("EPM_SIMPLE_POLICY_STOP_EXIT_MAX_GAP_BPS", "0")
os.environ.setdefault("EPM_SIMPLE_POLICY_SPREAD_MODEL_ENABLED", "0")

from extreme_price_movements.simple_policy_optimiser import simulate_and_score  # noqa: E402
from scripts.replay_strict_r3_simple_policy_15m import (  # noqa: E402
    COST_BPS,
    HORIZON_BARS,
    _load_15m,
    _paths_for_group,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prequential-ledger", type=Path, required=True)
    parser.add_argument("--policy-labels", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--per-month-cap", type=int, default=3500)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def _candidate_paths(
    ledger_path: Path,
    policy_path: Path,
    *,
    top_fraction: float,
    per_month_cap: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    cols = [
        "candidate_id", "__ts__", "__symbol__", "side_name",
        "prequential_upstream", "stack_is_prequential",
    ]
    ledger = pd.read_parquet(ledger_path, columns=cols)
    ledger["__ts__"] = pd.to_datetime(ledger["__ts__"], utc=True)
    ledger = ledger.loc[
        ledger["side_name"].astype(str).eq("long")
        & ledger["stack_is_prequential"].fillna(False).astype(bool)
        & ledger["__ts__"].ge(pd.Timestamp("2024-01-01", tz="UTC"))
        & ledger["__ts__"].lt(pd.Timestamp("2025-01-01", tz="UTC"))
        & np.isfinite(pd.to_numeric(ledger["prequential_upstream"], errors="coerce"))
    ].copy()
    if ledger.empty:
        raise ValueError("no strict-prequential long 2024 rows are available")
    # One pooled-global development tail; never a per-timestamp or held-month
    # percentile.  The threshold is computed once on pre-2025 OOF scores.
    cutoff = float(ledger["prequential_upstream"].quantile(1.0 - top_fraction))
    ledger = ledger.loc[ledger["prequential_upstream"].ge(cutoff)].copy()
    ledger["month"] = ledger["__ts__"].dt.strftime("%Y-%m")
    ledger["_stable_hash"] = pd.util.hash_pandas_object(
        ledger["candidate_id"].astype(str), index=False,
    ).astype("uint64")
    before_cap = ledger.groupby("month").size().astype(int).to_dict()
    ledger = (
        ledger.sort_values(["month", "_stable_hash", "candidate_id"], kind="stable")
        .groupby("month", group_keys=False)
        .head(int(per_month_cap))
        .reset_index(drop=True)
    )

    labels = pd.read_parquet(
        policy_path,
        columns=["candidate_id", "atr_1h", "policy_path_valid", "policy_label_available_ts"],
    )
    labels["policy_label_available_ts"] = pd.to_datetime(labels["policy_label_available_ts"], utc=True)
    labels = labels.loc[
        labels["policy_path_valid"].fillna(False).astype(bool)
        & labels["policy_label_available_ts"].lt(pd.Timestamp("2025-01-01", tz="UTC"))
    ].copy()
    sample = ledger.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    sample["atr_1h"] = pd.to_numeric(sample["atr_1h"], errors="coerce")
    sample = sample.loc[np.isfinite(sample["atr_1h"]) & sample["atr_1h"].gt(0)].copy()

    rows: list[pd.DataFrame] = []
    opens: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    lows: list[np.ndarray] = []
    closes: list[np.ndarray] = []
    for symbol, group in sample.groupby("__symbol__", sort=True):
        ts, op, hi, lo, cl = _load_15m(str(symbol))
        if not len(ts):
            continue
        valid, f_open, f_high, f_low, f_close = _paths_for_group(group, ts, op, hi, lo, cl)
        take = np.flatnonzero(valid)
        if not len(take):
            continue
        rows.append(group.iloc[take].copy())
        opens.append(f_open)
        highs.append(f_high)
        lows.append(f_low)
        closes.append(f_close)
    if not rows:
        raise ValueError("no complete pre-2025 15-minute paths are available")
    selected = pd.concat(rows, ignore_index=True)
    arrays = tuple(np.concatenate(part, axis=0) for part in (opens, highs, lows, closes))
    if any(len(value) != len(selected) for value in arrays):
        raise AssertionError("candidate/path alignment failed")
    coverage = {
        "pre_cap_rows_by_month": before_cap,
        "post_cap_rows": int(len(ledger)),
        "policy_label_eligible_rows": int(len(sample)),
        "complete_15m_path_rows": int(len(selected)),
        "complete_path_fraction_after_cap": float(len(selected) / max(len(ledger), 1)),
        "months": selected.groupby("month").size().astype(int).to_dict(),
        "symbols": int(selected["__symbol__"].nunique()),
        "pooled_global_upstream_cutoff": cutoff,
    }
    return selected, arrays[0], arrays[1], arrays[2], arrays[3], coverage


def main() -> None:
    args = _args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    rows, opens, highs, lows, closes, coverage = _candidate_paths(
        args.prequential_ledger,
        args.policy_labels,
        top_fraction=float(args.top_fraction),
        per_month_cap=int(args.per_month_cap),
    )
    entry = opens[:, 0].astype(np.float64)
    run = pd.DataFrame({
        "timestamp": rows["__ts__"].to_numpy(),
        "symbol": rows["__symbol__"].astype(str).to_numpy(),
        "side": np.ones(len(rows), dtype=np.float32),
        "rank_pct": rows["prequential_upstream"].to_numpy(np.float64),
        "barrier_pct": rows["atr_1h"].to_numpy(np.float64) / entry,
        "expected_half_spread_bps": np.zeros(len(rows)),
        "exit_quote_half_spread_bps": np.zeros(len(rows)),
        "entry_slippage_proxy_bps": np.zeros(len(rows)),
        "market_mode": "perps",
    })

    trial_rows: list[dict[str, object]] = []

    def evaluate(policy: dict[str, float], trial_number: int, name: str) -> float:
        result = simulate_and_score(
            run, opens, highs, lows, closes,
            cost_pct=0.0, size_power=1.0, replay_timeframe="15m", market_mode="perps",
            sl_mult=float(policy["sl_mult"]), sl_abs_cap_pct=0.0,
            trailing_activation_mult=float(policy["trailing_activation_mult"]),
            trailing_activation_cap_pct=0.0,
            trailing_activation_max_bars=HORIZON_BARS,
            fixed_trailing_gap_mult=float(policy["fixed_trailing_gap_mult"]),
            capital_protect_mfe_mult=0.0, adverse_exit_enabled=False,
            hard_tp_abs_pct=0.0, max_concurrent_trades=max(len(run), 1),
            max_concurrent_per_asset=max(len(run), 1),
            max_new_entries_per_bar=max(len(run), 1),
        )
        if not np.asarray(result["selected_mask"], dtype=bool).all():
            raise ValueError("candidate-local policy search unexpectedly applied portfolio limits")
        gross = np.asarray(result["gross_returns"], dtype=np.float64) * 10_000.0
        net = gross - COST_BPS
        monthly = pd.DataFrame({"month": rows["month"].to_numpy(), "net_bps": net}).groupby("month")["net_bps"].mean()
        median = float(monthly.median())
        mad = float((monthly - median).abs().median())
        objective = median - 0.5 * mad
        trial_rows.append({
            "trial": int(trial_number), "trial_name": name, **policy,
            "objective_bps": objective, "median_month_net_bps": median,
            "month_net_mad_bps": mad, "worst_month_net_bps": float(monthly.min()),
            "mean_net_bps": float(np.nanmean(net)), "mean_gross_bps": float(np.nanmean(gross)),
            "positive_months": int((monthly > 0).sum()), "months": int(len(monthly)),
        })
        return objective

    control = {
        "sl_mult": 3.0,
        "trailing_activation_mult": 0.5,
        "fixed_trailing_gap_mult": 0.25,
    }
    evaluate(control, -1, "frozen_canonical_control")

    def objective(trial: optuna.Trial) -> float:
        policy = {
            "sl_mult": trial.suggest_float("sl_mult", 1.0, 5.0),
            "trailing_activation_mult": trial.suggest_float("trailing_activation_mult", 0.25, 4.0),
            "fixed_trailing_gap_mult": trial.suggest_float("fixed_trailing_gap_mult", 0.10, 2.0),
        }
        return evaluate(policy, trial.number, "optuna")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(args.seed)),
    )
    study.optimize(objective, n_trials=int(args.trials), show_progress_bar=False)
    trials = pd.DataFrame(trial_rows).sort_values("objective_bps", ascending=False, kind="stable")
    trials.to_parquet(args.out_dir / "trials.parquet", index=False)
    best = trials.iloc[0]
    winner = {
        key: float(best[key])
        for key in ("sl_mult", "trailing_activation_mult", "fixed_trailing_gap_mult")
    }
    payload = {
        "schema": "strict_r3_schema_v2_long_policy_optimiser_pre2025_v1",
        "side": "long",
        "development_period": "2024-01-01 <= signal timestamp < 2025-01-01",
        "stack_score": "strict-prequential prequential_upstream (base + policy residual consensus)",
        "selection_population": f"one pooled-global top {100 * float(args.top_fraction):g}% tail; deterministic equal-month cap {args.per_month_cap}",
        "future_evaluation_period_used": False,
        "engine": "extreme_price_movements.simple_policy_optimiser.simulate_and_score",
        "entry": "first 15-minute open at signal close + 1h",
        "timeout": "12 hours / 48 complete 15-minute bars",
        "cost": f"{COST_BPS:g} bps deducted exactly once outside the simulator",
        "objective": "median monthly net bps/trade - 0.5 * monthly net MAD",
        "coverage": coverage,
        "winner": winner,
        "winner_trial": int(best["trial"]),
        "winner_trial_name": str(best["trial_name"]),
        "winner_objective_bps": float(best["objective_bps"]),
        "control_objective_bps": float(trials.loc[trials.trial.eq(-1), "objective_bps"].iloc[0]),
        "prequential_ledger": str(args.prequential_ledger),
        "policy_labels": str(args.policy_labels),
        "seed": int(args.seed),
        "trials": int(args.trials),
    }
    (args.out_dir / "winner.json").write_text(json.dumps(payload, indent=2))
    (args.out_dir / "run_manifest.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"event": "complete", "output": str(args.out_dir), **payload}, default=str))


if __name__ == "__main__":
    main()
