#!/usr/bin/env python3
"""Bounded portfolio-safety sweep over completed strict-R3 P3 router scores.

The script never refits a model or changes score/admission semantics.  It
replays the exact dual-MC1 >= 50-bps population with the canonical rich policy
labels, varying only predeclared concurrency and per-timestamp entry caps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_table(prediction: pd.DataFrame, threshold_bps: float) -> pd.DataFrame:
    from extreme_price_movements.portfolio_policy_replay import normalise_candidate_table

    required = {
        "candidate_id", "__decision_ts__", "__symbol__", "enhanced_base_routed",
        "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
        "policy_entry_price", "policy_exit_price", "policy_exit_reason",
        "current_mc1_expected_bps", "bcf_mc1_expected_bps",
    }
    missing = sorted(required.difference(prediction.columns))
    if missing:
        raise ValueError(f"dual-MC1 panel lacks {missing}")
    decision = pd.to_datetime(prediction["__decision_ts__"], utc=True, errors="raise")
    valid = (
        prediction["enhanced_base_routed"].fillna(False).astype(bool)
        & prediction["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(prediction["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(prediction["policy_exit_bar_15m"], errors="coerce"))
        & pd.to_numeric(prediction["current_mc1_expected_bps"], errors="coerce").ge(threshold_bps)
        & pd.to_numeric(prediction["bcf_mc1_expected_bps"], errors="coerce").ge(threshold_bps)
    )
    admitted = prediction.loc[valid].copy()
    if admitted.empty:
        raise ValueError("dual-MC1 gate produced no label-valid candidates")
    admitted["auction_rank"] = admitted.groupby("__decision_ts__", sort=False)["bcf_mc1_expected_bps"].rank(pct=True, method="average")
    exit_bar = pd.to_numeric(admitted["policy_exit_bar_15m"], errors="raise").astype(int)
    decision = pd.to_datetime(admitted["__decision_ts__"], utc=True, errors="raise")
    candidate = pd.DataFrame(
        {
            "timestamp": decision,
            "symbol": admitted["__symbol__"].astype(str),
            "side": "long",
            "strategy_id": "strict_r3_enhanced_live_stack_long",
            "policy_archetype": "strict_r3_enhanced_live_stack_long",
            "normalized_rank_score": admitted["auction_rank"].to_numpy(float),
            "strategy_rank_pct": admitted["auction_rank"].to_numpy(float),
            "base_strategy_threshold": 0.0,
            "calibrated_score": pd.to_numeric(admitted["bcf_mc1_expected_bps"], errors="raise").to_numpy(float),
            "entry_price": pd.to_numeric(admitted["policy_entry_price"], errors="raise"),
            "exit_timestamp": decision + pd.to_timedelta((exit_bar + 1) * 15, unit="min"),
            "exit_price": pd.to_numeric(admitted["policy_exit_price"], errors="raise"),
            "net_return": pd.to_numeric(admitted["policy_net_bps"], errors="raise") / 10_000.0,
            "gross_return": pd.to_numeric(admitted["policy_gross_bps"], errors="raise") / 10_000.0,
            "holding_bars": exit_bar + 1,
            "simple_policy_exit_reason": admitted["policy_exit_reason"].astype(str),
            "fees_bps": 100.0,
            "slippage_bps": 0.0,
            "expected_friction_bps": 100.0,
            "price_gap_bps": 0.0,
            "liquidity_capacity_weight": 1.0,
            "source_month": decision.dt.strftime("%Y-%m"),
            "candidate_id": admitted["candidate_id"].astype(str),
            "mapped_expected_net_bps": pd.to_numeric(admitted["bcf_mc1_expected_bps"], errors="raise"),
            "policy_outcome_available": np.ones(len(admitted), dtype=bool),
        }
    )
    return normalise_candidate_table(candidate)


def _risk_metrics(decisions: pd.DataFrame, equity: pd.DataFrame) -> dict[str, float]:
    """Use the established adaptive-exit trade-risk convention, labelled MTM."""
    mtm = pd.to_numeric(equity["mtm_equity"], errors="coerce").dropna()
    if len(mtm):
        drawdown = mtm / mtm.cummax() - 1.0
        mtm_max_drawdown = float(drawdown.min())
        ulcer = float(np.sqrt(np.mean(np.square(100.0 * drawdown))))
        growth = float(mtm.iloc[-1] / mtm.iloc[0]) if mtm.iloc[0] else np.nan
    else:
        mtm_max_drawdown = ulcer = np.nan
        growth = 1.0
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)]
    net = pd.to_numeric(accepted["position_net_return"], errors="coerce").dropna()
    downside = net.loc[net < 0.0]
    sortino = float(net.mean() / downside.std(ddof=0)) if len(downside) > 1 and downside.std(ddof=0) > 0.0 else np.nan
    return {
        "mtm_max_drawdown": mtm_max_drawdown,
        "portfolio_growth_multiple": growth,
        "growth_to_mtm_drawdown": math.log(growth) / abs(mtm_max_drawdown) if np.isfinite(growth) and growth > 0.0 and mtm_max_drawdown < 0.0 else np.nan,
        "sortino_trade": sortino,
        "ulcer_index_pct": ulcer,
    }


def run(*, prediction_path: Path, out: Path, threshold_bps: float, concurrency: list[int], entries_per_bar: list[int]) -> None:
    from scripts.report_strict_r3_mc1_d2_controlled_portfolio import CAUSAL_AUCTION_CURVE, _metrics, _params
    from extreme_price_movements.portfolio_policy_replay import replay_candidates

    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    prediction = pd.read_parquet(prediction_path)
    candidates = _candidate_table(prediction, threshold_bps)
    out.mkdir(parents=True)
    records: list[dict[str, object]] = []
    for maximum in concurrency:
        for per_bar in entries_per_bar:
            params = replace(
                _params(),
                max_concurrent_positions=int(maximum),
                max_concurrent_per_side=int(maximum),
                max_new_entries_per_bar=int(per_bar),
                max_new_entries_per_strategy_per_bar=int(per_bar),
                portfolio_policy_version=(
                    f"strict_r3_router_only_safety_c{maximum}_e{per_bar}_research"
                ),
            )
            decisions, equity, _ = replay_candidates(
                candidates,
                params,
                mode="global_auction",
                ev_curve=CAUSAL_AUCTION_CURVE,
                market_mode="perps",
                initial_wallet=1000.0,
            )
            if "policy_outcome_available" not in decisions.columns:
                decisions["policy_outcome_available"] = True
            label = f"c{maximum}_e{per_bar}"
            metric = _metrics(decisions, equity, label, "2026_febjul")
            metric.update(
                {
                    "max_concurrent_positions": maximum,
                    "max_new_entries_per_bar": per_bar,
                    "candidate_admitted_rows": int(len(candidates)),
                }
            )
            metric.update(_risk_metrics(decisions, equity))
            records.append(metric)
    pd.DataFrame(records).to_parquet(out / "portfolio_safety_metrics.parquet", index=False)
    manifest = {
        "schema": "strict_r3_router_only_portfolio_safety_sweep_v1",
        "scope": "offline research-only; fixed P3 router / T6-T9 consensus / dual-MC1 mappings / policy labels",
        "prediction_sha256": _sha256(prediction_path),
        "threshold_bps": threshold_bps,
        "admission": "both current and BCF MC1 mapped EV >= threshold; auction priority is BCF MC1 mapped EV",
        "changed_dimensions": {"max_concurrent_positions": concurrency, "max_new_entries_per_bar": entries_per_bar},
        "unchanged": "router, base coordinates, consensus heads, MC1 maps, policy labels, costs, exit policy, 80% wallet cap, 7x leverage, 10% margin slots",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold-bps", type=float, default=50.0)
    parser.add_argument("--concurrency", default="4,5,6,8")
    parser.add_argument("--entries-per-bar", default="1,2")
    args = parser.parse_args()
    run(
        prediction_path=args.prediction_path,
        out=args.out,
        threshold_bps=args.threshold_bps,
        concurrency=[int(value) for value in args.concurrency.split(",")],
        entries_per_bar=[int(value) for value in args.entries_per_bar.split(",")],
    )


if __name__ == "__main__":
    main()
