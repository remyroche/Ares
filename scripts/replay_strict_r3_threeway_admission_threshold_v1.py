#!/usr/bin/env python3
"""Replay sealed MC1 receipts at a bounded dual-admission threshold.

This is terminal-policy reporting only.  It consumes immutable current/BCF MC1
prediction receipts, uses the existing dual-map admission and constrained
portfolio adapter, and never fits or changes a score, calibrator, feature, or
live artifact. It supports the bounded 30/35/40/45/50/60/70/80-bps gate grid on the
exact same frozen score panels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import run_strict_r3_enhanced_base_live_stack_challenger as core


def _monthly(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame(columns=["month", "accepted_rows", "net_ev_bps_per_trade", "net_sum_bps"])
    stamped = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    stamped["timestamp"] = pd.to_datetime(stamped["timestamp"], utc=True, errors="raise")
    stamped["month"] = stamped["timestamp"].dt.strftime("%Y-%m")
    stamped["net_bps"] = pd.to_numeric(stamped["position_net_return"], errors="coerce") * 10_000.0
    return stamped.groupby("month", sort=True).agg(
        accepted_rows=("net_bps", "size"),
        net_ev_bps_per_trade=("net_bps", "mean"),
        net_sum_bps=("net_bps", "sum"),
    ).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path, help="sealed full-stack replay root")
    source.add_argument("--live-current-mc1", type=Path, help="immutable current-family live MC1 receipt")
    parser.add_argument("--live-bcf-mc1", type=Path, help="immutable BCF-family live MC1 receipt; required with --live-current-mc1")
    parser.add_argument("--policy-root", type=Path, help="canonical policy labels; required with --live-current-mc1")
    parser.add_argument("--threshold-bps", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--period-start", default="2026-04-01")
    parser.add_argument("--period-end", default="2026-08-01")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.threshold_bps not in {30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0}:
        raise ValueError("only the bounded 30/35/40/45/50/60/70/80-bps gates are valid")
    args.out.mkdir(parents=True)
    start, end = core._utc(args.period_start), core._utc(args.period_end)
    if args.live_current_mc1 is not None:
        if args.live_bcf_mc1 is None or args.policy_root is None:
            raise ValueError("live-baseline mode requires --live-bcf-mc1 and --policy-root")
        paths = core.Paths(
            raw_ledger=Path("."), direct_root=Path("."), policy_root=args.policy_root,
            current_mc1=args.live_current_mc1, bcf_mc1=args.live_bcf_mc1, bundle_root=Path("."),
        )
        core.EVALUATION_PERIODS = {"held": (start, end)}
        combined = core._baseline(paths, core._load_policy(paths))
        source_manifest: dict[str, object] = {
            "schema": "immutable_live_current_bcf_mc1_baseline",
            "source": {"current_mc1": str(args.live_current_mc1), "bcf_mc1": str(args.live_bcf_mc1)},
        }
    else:
        source_manifest = json.loads((args.source / "run_manifest.json").read_text())
        current = pd.read_parquet(args.source / "enhanced_current_mc1_predictions.parquet")
        bcf = pd.read_parquet(args.source / "enhanced_bcf_mc1_predictions.parquet")
        for frame in (current, bcf):
            frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        current = current.loc[current["__decision_ts__"].ge(start) & current["__decision_ts__"].lt(end)].copy()
        bcf = bcf.loc[bcf["__decision_ts__"].ge(start) & bcf["__decision_ts__"].lt(end)].copy()
        if current["candidate_id"].duplicated().any() or bcf["candidate_id"].duplicated().any():
            raise AssertionError("sealed MC1 receipt has duplicate candidate IDs")
        combined = core._combined_challenger(current, bcf)
    original_threshold = core.MC1_THRESHOLD_BPS
    core.MC1_THRESHOLD_BPS = float(args.threshold_bps)
    try:
        metric = core._portfolio_metrics(combined, "sealed_score_replay", "held", args.out)
    finally:
        core.MC1_THRESHOLD_BPS = original_threshold
    decisions = pd.read_parquet(args.out / "sealed_score_replay_held_decisions.parquet")
    monthly = _monthly(decisions)
    monthly.to_parquet(args.out / "monthly_metrics.parquet", index=False, compression="zstd")
    pd.DataFrame([metric]).to_parquet(args.out / "portfolio_metrics.parquet", index=False, compression="zstd")
    (args.out / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_threeway_admission_threshold_replay_v1",
        "scope": "offline terminal replay from sealed MC1 receipts; no model, feature, calibration, live, or exchange mutation",
        "source": str(args.source) if args.source is not None else source_manifest.get("source"),
        "source_schema": source_manifest.get("schema"),
        "upstream": source_manifest.get("upstream_override"),
        "threshold_bps": float(args.threshold_bps),
        "period": [start.isoformat(), end.isoformat()],
        "policy": "canonical rich policy labels already joined after target-free score production; existing constrained global portfolio adapter",
        "causality": "no model is refit; held score identity and MC1 values are consumed verbatim from the sealed source receipts",
    }, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out": str(args.out), "threshold_bps": args.threshold_bps, **metric}), flush=True)


if __name__ == "__main__":
    main()
