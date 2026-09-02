#!/usr/bin/env python3
"""Test alternate MC1-only scores as auction orders, not admission authority.

Frozen strict MC1_d2 expected value remains the sole +50-bps admission gate.
Each challenger score only replaces final_score's timestamp-local auction rank
after admission.  This isolates whether a target/loss variant adds selection
value at contested timestamps without weakening MC1 calibration.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_strict_r3_mc1_d2_controlled_portfolio import (  # noqa: E402
    _candidate_table,
    _metrics,
    _params,
    CAUSAL_AUCTION_CURVE,
)
from extreme_price_movements.portfolio_policy_replay import replay_candidates  # noqa: E402


def _attach_outcomes(decisions: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        output = decisions.copy()
        output["policy_outcome_available"] = pd.Series(dtype=bool)
        return output
    lookup = candidates.loc[:, ["candidate_id", "policy_outcome_available"]].reset_index(drop=True)
    lookup.index.name = "candidate_index"
    output = decisions.merge(lookup, on="candidate_index", how="left", validate="many_to_one")
    if output.policy_outcome_available.isna().any():
        raise ValueError("portfolio decision is missing candidate outcome provenance")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--challenger-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    control = pd.read_parquet(args.control)
    control["__decision_ts__"] = pd.to_datetime(control["__decision_ts__"], utc=True)
    control = control.loc[:, [
        "candidate_id", "__decision_ts__", "__symbol__", "final_score", "mc1_expected_bps",
    ]].rename(columns={"final_score": "frozen_final_score", "mc1_expected_bps": "frozen_mc1_expected_bps"})
    if control.candidate_id.duplicated().any():
        raise ValueError("frozen control identity is not unique")
    policy_columns = [
        "candidate_id", "policy_path_valid", "policy_net_bps", "policy_gross_bps",
        "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price", "policy_exit_reason",
    ]
    policy = pd.read_parquet(args.ledger, columns=policy_columns)
    if policy.candidate_id.duplicated().any():
        raise ValueError("policy ledger candidate identity is not unique")
    arms: dict[str, pd.DataFrame] = {"frozen_final_score": control.assign(final_score=control.frozen_final_score)}
    huber_frame: pd.DataFrame | None = None
    for path in sorted(args.challenger_dir.glob("predictions_*.parquet")):
        name = path.stem.removeprefix("predictions_")
        if name == "huber_clip":
            continue  # it has no +50 output scale; no meaningful auction order.
        challenger = pd.read_parquet(path, columns=["candidate_id", "mc1_expected_bps"])
        challenger = challenger.rename(columns={"mc1_expected_bps": "auction_score"})
        arm = control.merge(challenger, on="candidate_id", how="inner", validate="one_to_one")
        if len(arm) != len(control):
            raise ValueError(f"{name} identity does not match frozen control")
        arms[name] = arm.assign(final_score=arm.auction_score)
        if name == "huber_asin":
            huber_frame = arm
    if huber_frame is not None:
        # Blend rank coordinates, not incomparable raw outputs.  Both ranks
        # are target-free timestamp-local quantities computed before admission.
        base_rank = huber_frame.groupby("__decision_ts__", sort=False)["frozen_final_score"].rank(pct=True, method="average")
        huber_rank = huber_frame.groupby("__decision_ts__", sort=False)["auction_score"].rank(pct=True, method="average")
        for weight in (.25, .50, .75):
            arms[f"huber_rank_blend_{int(weight * 100):02d}"] = huber_frame.assign(
                final_score=(1.0 - weight) * base_rank + weight * huber_rank,
            )
        admitted_count = huber_frame.assign(
            __admitted__=huber_frame.frozen_mc1_expected_bps.ge(50.0),
        ).groupby("__decision_ts__", sort=False)["__admitted__"].transform("sum")
        for minimum in (3, 4, 5):
            arms[f"huber_contested_ge{minimum}"] = huber_frame.assign(
                final_score=huber_frame.frozen_final_score.where(admitted_count.lt(minimum), huber_frame.auction_score),
            )
        for weight in (.25, .50, .75):
            mixed = (1.0 - weight) * base_rank + weight * huber_rank
            arms[f"huber_contested_ge4_blend_{int(weight * 100):02d}"] = huber_frame.assign(
                final_score=huber_frame.frozen_final_score.where(admitted_count.lt(4), mixed),
            )
    metrics: list[dict[str, object]] = []
    for name, frame in arms.items():
        frame = frame.rename(columns={"frozen_mc1_expected_bps": "mc1_expected_bps"})
        for year in (2025, 2026):
            part = frame.loc[frame.__decision_ts__.dt.year.eq(year)].copy()
            candidates = _candidate_table(part, policy, 50.0)
            decisions, equity, _ = replay_candidates(
                candidates, _params(), mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE,
                market_mode="perps", initial_wallet=1000.0,
            )
            decisions = _attach_outcomes(decisions, candidates)
            metrics.append(_metrics(decisions, equity, name, str(year)))
        print(json.dumps({"event": "arm_complete", "arm": name}), flush=True)
        del frame
        gc.collect()
    pd.DataFrame(metrics).to_parquet(args.out_dir / "auction_score_metrics.parquet", index=False)
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_mc1_d2_auction_score_ablation_v1", "status": "complete",
        "purpose": "auction-only test; frozen MC1_d2 remains admission authority",
        "admission": "frozen strict MC1 expected policy net >= +50 bps",
        "auction": "timestamp-local rank of final_score, named challenger output, pre-admission rank-normalized Huber blend, or fractional Huber authority only at declared admitted-count contention within frozen admitted cohort",
        "portfolio": "long-only, 7x, 10%-margin slots, 2 entries per timestamp, 8 concurrent",
        "exclusions": ["live state", "exchange I/O", "admission target changes"],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
