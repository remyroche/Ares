#!/usr/bin/env python3
"""Backfill strict-R3 *evaluation labels* from downloaded 15-minute OHLC.

This producer deliberately does not score candidates, filter their universe, or
rewrite a source panel.  It takes the target-free strict-R3 identities already
scored by a walk-forward ledger and produces a one-row-per-candidate policy
label overlay.  Existing valid outcomes retain precedence; only rows whose
primary path is invalid may be resolved from complete, non-stale 15-minute
bars.  There is intentionally no minute-data fallback.

The execution contract is the declared frozen SimplePolicyOptimiser geometry:
entry at the first 15-minute open at signal close + one hour, H12 timeout and
the configured SL/trailing parameters.  Gross is produced by the simulator;
the fixed 100-bps round-trip cost is deducted exactly once afterwards.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Match the frozen policy label contract: no hidden spread/gap adjustment.
os.environ.setdefault("EPM_SIMPLE_POLICY_STOP_EXIT_BASE_GAP_BPS", "0")
os.environ.setdefault("EPM_SIMPLE_POLICY_STOP_EXIT_MAX_GAP_BPS", "0")
os.environ.setdefault("EPM_SIMPLE_POLICY_SPREAD_MODEL_ENABLED", "0")
from extreme_price_movements.simple_policy_optimiser import simulate_and_score  # noqa: E402
from scripts.replay_strict_r3_simple_policy_15m import (  # noqa: E402
    COST_BPS,
    HORIZON_BARS,
    _coarse_causal_atr,
    _load_15m,
    _paths_for_group,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--policy-json", type=Path,
        default=ROOT / "config/strict_r3_frozen_15m_policy.json",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _month_prediction(month_dir: Path) -> Path:
    month = month_dir.name.removeprefix("month=")
    # The canonical walk-forward producer stores folds directly under
    # ``<run>/scores/month=YYYY-MM``.  Older producers used
    # ``<run>/month=YYYY-MM/scores/month=YYYY-MM``.  Labels are an evaluation
    # overlay, so supporting both layouts must not change identities or score
    # inputs.
    candidates = (
        month_dir / "predictions.parquet",
        month_dir / "scores" / f"month={month}" / "predictions.parquet",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no predictions.parquet for {month}: {candidates}")


def _policy(payload: dict[str, object]) -> dict[str, float]:
    winner = payload.get("winner")
    if not isinstance(winner, dict):
        raise ValueError("policy JSON needs a winner object")
    keys = ("sl_mult", "trailing_activation_mult", "fixed_trailing_gap_mult")
    output = {key: float(winner[key]) for key in keys}
    if not all(np.isfinite(value) and value >= 0.0 for value in output.values()):
        raise ValueError("policy geometry must be finite and non-negative")
    if float(payload.get("cost_bps_once", COST_BPS)) != COST_BPS:
        raise ValueError("the strict-R3 label overlay permits exactly the frozen 100-bps cost")
    if int(payload.get("timeout_hours", 12)) != 12:
        raise ValueError("the strict-R3 label overlay permits only H12")
    return output


def _empty(group: pd.DataFrame) -> pd.DataFrame:
    result = group.loc[:, ["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"]].copy()
    result["policy_path_valid"] = False
    result["policy_gross_bps"] = np.nan
    result["policy_net_bps"] = np.nan
    result["policy_exit_bar_15m"] = -1
    result["policy_exit_reason"] = "invalid_path"
    result["policy_entry_price"] = np.nan
    result["policy_exit_price"] = np.nan
    result["policy_atr"] = np.nan
    result["policy_atr_source"] = "unavailable"
    result["policy_outcome_source"] = "unavailable"
    result["policy_market_data_quality"] = "incomplete_15m_path"
    result["policy_label_available_ts"] = result["__decision_ts__"] + pd.Timedelta(hours=12)
    result["policy_cost_bps"] = np.nan
    return result


def _replay_coarse_symbol(group: pd.DataFrame, policy: dict[str, float]) -> pd.DataFrame:
    """Candidate-local replay using only local downloaded coarse 15-minute bars."""

    result = _empty(group).reset_index(drop=True)
    ts, opens, highs, lows, closes = _load_15m(str(group["__symbol__"].iloc[0]))
    if not len(ts):
        return result
    valid, f_open, f_high, f_low, f_close = _paths_for_group(group, ts, opens, highs, lows, closes)
    positions = np.flatnonzero(valid)
    if not len(positions):
        return result
    # All fallback ATR observations are available at the candidate decision
    # timestamp.  A missing provided ATR must never make the outcome a zero.
    fallback_atr = _coarse_causal_atr(ts, opens, highs, lows, closes)
    candidate_atr = pd.to_numeric(group.get("atr_1h"), errors="coerce")
    fallback = pd.to_datetime(group["__decision_ts__"], utc=True).map(fallback_atr)
    atr = candidate_atr.where(candidate_atr.gt(0.0), fallback).to_numpy(float)
    usable = np.isfinite(atr[positions]) & (atr[positions] > 0.0)
    positions = positions[usable]
    f_open, f_high, f_low, f_close = (
        f_open[usable], f_high[usable], f_low[usable], f_close[usable],
    )
    if not len(positions):
        return result
    entry = f_open[:, 0].astype(np.float64)
    run = pd.DataFrame({
        "timestamp": group.iloc[positions]["__ts__"].to_numpy(),
        "symbol": group.iloc[positions]["__symbol__"].astype(str).to_numpy(),
        "side": np.ones(len(positions), dtype=np.float32),
        "rank_pct": np.ones(len(positions), dtype=np.float32),
        "barrier_pct": atr[positions] / entry,
        "expected_half_spread_bps": np.zeros(len(positions)),
        "exit_quote_half_spread_bps": np.zeros(len(positions)),
        "entry_slippage_proxy_bps": np.zeros(len(positions)),
        "market_mode": "perps",
    })
    sim = simulate_and_score(
        run, f_open, f_high, f_low, f_close,
        cost_pct=0.0, size_power=1.0, replay_timeframe="15m", market_mode="perps",
        sl_mult=policy["sl_mult"], sl_abs_cap_pct=0.0,
        trailing_activation_mult=policy["trailing_activation_mult"],
        trailing_activation_cap_pct=0.0, trailing_activation_max_bars=HORIZON_BARS,
        fixed_trailing_gap_mult=policy["fixed_trailing_gap_mult"],
        capital_protect_mfe_mult=0.0, adverse_exit_enabled=False, hard_tp_abs_pct=0.0,
        max_concurrent_trades=max(len(run), 1), max_concurrent_per_asset=max(len(run), 1),
        max_new_entries_per_bar=max(len(run), 1),
    )
    if not np.asarray(sim["selected_mask"], dtype=bool).all():
        raise AssertionError("candidate-local label replay unexpectedly applied a portfolio limit")
    gross = np.asarray(sim["gross_returns"], dtype=np.float64) * 10_000.0
    result.loc[positions, "policy_path_valid"] = np.isfinite(gross)
    result.loc[positions, "policy_gross_bps"] = gross
    result.loc[positions, "policy_net_bps"] = gross - COST_BPS
    result.loc[positions, "policy_exit_bar_15m"] = np.asarray(sim["exit_bars"], dtype=np.int16)
    result.loc[positions, "policy_exit_reason"] = np.asarray(sim["exit_reason"], dtype=object)
    result.loc[positions, "policy_entry_price"] = np.asarray(sim["entry_prices"], dtype=np.float64)
    result.loc[positions, "policy_exit_price"] = np.asarray(sim["exit_prices"], dtype=np.float64)
    result.loc[positions, "policy_atr"] = atr[positions]
    result.loc[positions, "policy_atr_source"] = np.where(
        candidate_atr.iloc[positions].gt(0.0).to_numpy(),
        "source_decision_time_atr", "coarse_15m_wilder14",
    )
    good = result["policy_path_valid"].to_numpy(bool)
    result.loc[good, "policy_outcome_source"] = "coarse_15m"
    result.loc[good, "policy_market_data_quality"] = "complete_48x15m"
    result.loc[good, "policy_cost_bps"] = COST_BPS
    if good.any() and not np.allclose(
        result.loc[good, "policy_net_bps"], result.loc[good, "policy_gross_bps"] - COST_BPS,
        rtol=0.0, atol=1e-12,
    ):
        raise AssertionError("coarse policy overlay cost was not applied exactly once")
    return result


def _overlay(primary: pd.DataFrame, coarse: pd.DataFrame) -> pd.DataFrame:
    """Preserve valid primary outcomes; fill only invalid identities from coarse OHLC."""

    if not primary["candidate_id"].is_unique or not coarse["candidate_id"].is_unique:
        raise ValueError("overlay identities must be unique")
    primary = primary.copy()
    # Prediction ledgers predate some explicit label-lineage columns.  Seed
    # null primary columns so every coarse candidate column gets an unambiguous
    # ``__coarse`` partner after the one-to-one merge.
    for column in (
        "policy_entry_price", "policy_exit_price", "policy_cost_bps",
        "policy_atr_source", "policy_outcome_source", "policy_market_data_quality",
    ):
        if column not in primary:
            primary[column] = np.nan
    result = primary.merge(
        coarse.drop(columns=["__ts__", "__decision_ts__", "__symbol__", "side_name"]),
        on="candidate_id", how="left", suffixes=("", "__coarse"), validate="one_to_one",
    )
    invalid = ~result["policy_path_valid"].fillna(False).astype(bool)
    coarse_valid = result["policy_path_valid__coarse"].fillna(False).astype(bool)
    take = invalid & coarse_valid
    columns = (
        "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
        "policy_exit_reason", "policy_entry_price", "policy_exit_price", "policy_label_available_ts",
        "policy_cost_bps",
    )
    for column in columns:
        result.loc[take, column] = result.loc[take, f"{column}__coarse"]
    result["policy_outcome_source"] = np.where(
        take, "coarse_15m", np.where(
            result["policy_path_valid"].fillna(False).astype(bool), "primary_existing", "unavailable",
        ),
    )
    result["policy_market_data_quality"] = np.where(
        take, "complete_48x15m", np.where(
            result["policy_path_valid"].fillna(False).astype(bool), "primary_existing", "incomplete_15m_path",
        ),
    )
    result["policy_atr_source"] = np.where(
        take, result["policy_atr_source__coarse"], "primary_existing",
    )
    drop = [column for column in result if column.endswith("__coarse")]
    return result.drop(columns=drop)


def main() -> None:
    args = _args()
    policy_payload = json.loads(args.policy_json.read_text())
    policy = _policy(policy_payload)
    # See _month_prediction: accept both historical nested-fold layout and
    # the current canonical ``scores/month=YYYY-MM`` layout.
    months = sorted(path for path in args.fold_root.glob("month=20??-??") if path.is_dir())
    if not months:
        scores = args.fold_root / "scores"
        months = sorted(path for path in scores.glob("month=20??-??") if path.is_dir())
    if not months:
        raise FileNotFoundError("no month=YYYY-MM fold directories")
    if args.out_dir.exists() and not args.resume:
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=bool(args.resume))
    audits: list[dict[str, object]] = []
    for month_dir in months:
        month = month_dir.name.removeprefix("month=")
        output = args.out_dir / f"month={month}.parquet"
        if output.exists():
            audits.append(json.loads((args.out_dir / f"month={month}.json").read_text()))
            continue
        primary = pd.read_parquet(_month_prediction(month_dir))
        primary["__ts__"] = pd.to_datetime(primary["__ts__"], utc=True)
        primary["__decision_ts__"] = pd.to_datetime(primary["__decision_ts__"], utc=True)
        primary["atr_1h"] = np.nan  # Use only causal coarse ATR unless explicitly supplied in this ledger.
        primary_valid = primary["policy_path_valid"].fillna(False).astype(bool)
        checkpoints = args.out_dir / f"month={month}_symbol_parts"
        checkpoints.mkdir(exist_ok=True)
        pieces: list[Path] = []
        for ordinal, symbol in enumerate(sorted(primary["__symbol__"].astype(str).unique()), 1):
            group = primary.loc[primary["__symbol__"].astype(str).eq(symbol)].copy()
            checkpoint = checkpoints / f"{hashlib.sha256(symbol.encode()).hexdigest()[:20]}.parquet"
            if not checkpoint.exists():
                _replay_coarse_symbol(group, policy).to_parquet(checkpoint, index=False, compression="zstd")
            pieces.append(checkpoint)
            if ordinal % 20 == 0:
                gc.collect(); pa.default_memory_pool().release_unused()
        coarse = pd.concat([pd.read_parquet(path) for path in pieces], ignore_index=True)
        merged = _overlay(primary, coarse)
        if len(merged) != len(primary) or merged["candidate_id"].duplicated().any():
            raise AssertionError("coarse label overlay changed scored candidate identity")
        final_valid = merged["policy_path_valid"].fillna(False).astype(bool)
        audit = {
            "month": month, "rows": int(len(merged)), "primary_valid_rows": int(primary_valid.sum()),
            "coarse_filled_rows": int((~primary_valid & final_valid).sum()),
            "final_valid_rows": int(final_valid.sum()), "final_valid_rate": float(final_valid.mean()),
            "remaining_invalid_rows": int((~final_valid).sum()),
        }
        merged.to_parquet(output, index=False, compression="zstd")
        (args.out_dir / f"month={month}.json").write_text(json.dumps(audit, indent=2) + "\n")
        audits.append(audit)
        print(json.dumps({"event": "month_complete", **audit}), flush=True)
        del primary, coarse, merged
        gc.collect(); pa.default_memory_pool().release_unused()
    coverage = pd.DataFrame(audits).sort_values("month", kind="stable")
    coverage.to_parquet(args.out_dir / "coverage.parquet", index=False)
    manifest = {
        "schema": "strict_r3_policy_label_overlay_coarse15m_v1",
        "fold_root": str(args.fold_root), "policy_json": str(args.policy_json),
        "entry": "first 15-minute open at signal close + one hour",
        "outcome_usage": "evaluation and causally resolved historical state only; never scoring inputs",
        "minute_data": "prohibited", "coarse_source": "downloaded 15-minute OHLC",
        "timeout_hours": 12, "cost_bps_once": COST_BPS, "winner": policy,
        "months": coverage.to_dict("records"),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
