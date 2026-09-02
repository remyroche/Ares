#!/usr/bin/env python3
"""Replay a phase's two independently prequential MC1 maps through one auction.

Both family maps must exceed the declared EV floor.  Current-v5's frozen final
score remains the auction coordinate, so a mapper only has admission authority.
Outcome validity is consulted only after target-free dual admission has been
sealed and invalid outcomes never reserve simulated capacity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import replay_candidates
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import (
    CAUSAL_AUCTION_CURVE,
    _candidate_table,
    _metrics,
    _params,
)


POLICY = (
    "candidate_id", "policy_path_valid", "policy_net_bps", "policy_gross_bps",
    "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
    "policy_exit_reason",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def load_prediction(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    required = {
        "candidate_id", "__decision_ts__", "__symbol__", "final_score",
        "mc1_expected_bps", *POLICY,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} prediction lacks: {missing}")
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if frame["candidate_id"].duplicated().any():
        raise ValueError(f"{name} has duplicate candidate IDs")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--bcf", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threshold-bps", type=float, default=50.0)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")

    current = load_prediction(args.current, "current-v5")
    bcf = load_prediction(args.bcf, "BCF")
    overlap = current.merge(
        bcf.loc[:, ["candidate_id", "__decision_ts__", "mc1_expected_bps"]],
        on="candidate_id", how="inner", validate="one_to_one", suffixes=("", "__bcf"),
    )
    if len(overlap) != len(current):
        raise ValueError("every routed current-v5 row must have a matching BCF score")
    if not overlap["__decision_ts__"].eq(overlap["__decision_ts____bcf"]).all():
        raise ValueError("BCF/current decision identity mismatch")
    overlap = overlap.drop(columns="__decision_ts____bcf")
    overlap = overlap.rename(
        columns={"mc1_expected_bps": "current_mc1_expected_bps", "mc1_expected_bps__bcf": "bcf_mc1_expected_bps"}
    )
    overlap["dual_admitted"] = (
        overlap["current_mc1_expected_bps"].ge(args.threshold_bps)
        & overlap["bcf_mc1_expected_bps"].ge(args.threshold_bps)
    )
    # The minimum carries the exact dual rule into the existing no-lookahead
    # candidate-table adapter.  The score never owns auction priority.
    overlap["mc1_expected_bps"] = np.minimum(
        pd.to_numeric(overlap["current_mc1_expected_bps"], errors="coerce"),
        pd.to_numeric(overlap["bcf_mc1_expected_bps"], errors="coerce"),
    )
    if not overlap["dual_admitted"].eq(overlap["mc1_expected_bps"].ge(args.threshold_bps)).all():
        raise AssertionError("dual admission is not equivalent to the minimum-EV floor")

    policy = overlap.loc[:, list(POLICY)].copy()
    if policy["candidate_id"].duplicated().any():
        raise ValueError("dual policy identity is not unique")
    candidates = _candidate_table(overlap, policy, args.threshold_bps, invalid_outcome_mode="exclude")
    decisions, equity, _ = replay_candidates(
        candidates, _params(), mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE,
        market_mode="perps", initial_wallet=1000.0,
    )
    if len(decisions):
        lookup = candidates.loc[:, ["candidate_id", "policy_outcome_available", "mapped_expected_net_bps"]].reset_index(drop=True)
        lookup.index.name = "candidate_index"
        decisions = decisions.merge(lookup, on="candidate_index", how="left", validate="many_to_one")
    else:
        decisions["policy_outcome_available"] = pd.Series(dtype=bool)
    result = _metrics(decisions, equity, "dual_current_bcf", "all")
    result.update({
        "threshold_bps": float(args.threshold_bps),
        "dual_scored_rows": int(len(overlap)),
        "dual_admitted_rows_before_outcome_filter": int(overlap["dual_admitted"].sum()),
        "dual_valid_outcome_rows_after_admission": int(len(candidates)),
        "auction_priority": "current_v5_final_score",
    })
    accepted_mask = (
        decisions["accepted"].fillna(False).astype(bool)
        if "accepted" in decisions
        else pd.Series(False, index=decisions.index, dtype=bool)
    )
    monthly = decisions.loc[accepted_mask].copy()
    if len(monthly):
        monthly["month"] = pd.to_datetime(monthly["timestamp"], utc=True).dt.strftime("%Y-%m")
        monthly["week"] = pd.to_datetime(monthly["timestamp"], utc=True).dt.strftime("%G-W%V")
        monthly_metrics = monthly.groupby("month", sort=True).agg(
            entries=("candidate_id", "size"),
            net_ev_bps=("position_net_return", lambda s: float(pd.to_numeric(s, errors="coerce").mean() * 10_000.0)),
            net_sum_bps=("position_net_return", lambda s: float(pd.to_numeric(s, errors="coerce").sum() * 10_000.0)),
        ).reset_index()
        weekly_metrics = monthly.groupby("week", sort=True).agg(
            entries=("candidate_id", "size"),
            net_ev_bps=("position_net_return", lambda s: float(pd.to_numeric(s, errors="coerce").mean() * 10_000.0)),
            net_sum_bps=("position_net_return", lambda s: float(pd.to_numeric(s, errors="coerce").sum() * 10_000.0)),
        ).reset_index()
    else:
        monthly_metrics = pd.DataFrame(columns=["month", "entries", "net_ev_bps", "net_sum_bps"])
        weekly_metrics = pd.DataFrame(columns=["week", "entries", "net_ev_bps", "net_sum_bps"])

    args.out_dir.mkdir(parents=True)
    overlap.to_parquet(args.out_dir / "dual_mapping_target_free_then_outcome.parquet", index=False, compression="zstd")
    candidates.to_parquet(args.out_dir / "dual_admitted_candidates_after_outcome_join.parquet", index=False, compression="zstd")
    decisions.to_parquet(args.out_dir / "portfolio_decisions.parquet", index=False, compression="zstd")
    equity.to_parquet(args.out_dir / "portfolio_equity.parquet", index=False, compression="zstd")
    monthly_metrics.to_parquet(args.out_dir / "monthly_metrics.parquet", index=False)
    weekly_metrics.to_parquet(args.out_dir / "weekly_metrics.parquet", index=False)
    pd.DataFrame([result]).to_parquet(args.out_dir / "portfolio_metrics.parquet", index=False)
    manifest = {
        "schema": "strict_r3_phase_h1_dual_prequential_portfolio_v1",
        "target_free_until_policy_join": True,
        "admission": "current MC1 >= floor AND BCF MC1 >= floor",
        "threshold_bps": float(args.threshold_bps),
        "auction": "current-v5 final_score after dual EV admission",
        "outcome_policy": "invalid policy paths excluded after admission and before simulated capacity",
        "inputs": {
            "current": {"path": str(args.current), "sha256": sha256(args.current)},
            "bcf": {"path": str(args.bcf), "sha256": sha256(args.bcf)},
        },
        "metrics": result,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"event": "complete", **result}, sort_keys=True))


if __name__ == "__main__":
    main()
