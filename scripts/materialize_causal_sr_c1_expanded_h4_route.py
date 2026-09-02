#!/usr/bin/env python3
"""Build an immutable, target-free C1 H4 route across scored periods.

This utility deliberately recomputes dual-MC1 eligibility from already stored
target-free predictions.  It never reads a policy outcome, one-minute path, or
portfolio decision.  That makes it safe to use as the population input to
source coverage repair and exact-path materialisation at a research threshold
different from the incumbent's frozen +50-bps gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARM = "C1_refit_core_plus_causal_sr"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(root: Path, arm: str, threshold_bps: float) -> pd.DataFrame:
    path = root / f"{arm}_target_free_admission.parquet"
    frame = pd.read_parquet(path).copy()
    required = {
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "bcf_final_score", "bcf_mc1_expected_bps", "current_mc1_expected_bps",
        "auction_priority_bps",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise AssertionError(f"{path}: missing target-free score fields {missing}")
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["timestamp"] = pd.to_datetime(frame.pop("__decision_ts__"), utc=True, errors="raise")
    frame["entry_ts"] = frame["timestamp"] + pd.Timedelta(minutes=5)
    frame["symbol"] = frame.pop("__symbol__").astype(str)
    frame["side_name"] = frame["side_name"].astype(str).str.lower().str.strip()
    for column in ("bcf_final_score", "bcf_mc1_expected_bps", "current_mc1_expected_bps", "auction_priority_bps"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame.loc[
        frame["side_name"].eq("long")
        & frame["bcf_mc1_expected_bps"].ge(float(threshold_bps))
        & frame["current_mc1_expected_bps"].ge(float(threshold_bps))
    ].copy()
    frame["admission_threshold_bps"] = float(threshold_bps)
    frame["source_root"] = str(root.resolve())
    if frame["candidate_id"].duplicated().any():
        raise AssertionError(f"{path}: duplicate candidate identity")
    return frame.loc[:, [
        "candidate_id", "timestamp", "entry_ts", "symbol", "side_name",
        "bcf_final_score", "bcf_mc1_expected_bps", "current_mc1_expected_bps",
        "auction_priority_bps", "admission_threshold_bps", "source_root",
    ]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, action="append", required=True)
    parser.add_argument("--threshold-bps", type=float, default=40.0)
    parser.add_argument("--arm", default=ARM)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.threshold_bps <= 0:
        raise ValueError("threshold must be positive")
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    roots = tuple(path.resolve() for path in args.source_root)
    if len(set(roots)) != len(roots):
        raise ValueError("source roots must be unique")
    parts = [_load(root, args.arm, args.threshold_bps) for root in roots]
    route = pd.concat(parts, ignore_index=True).sort_values(["timestamp", "candidate_id"], kind="stable")
    if route["candidate_id"].duplicated().any():
        raise AssertionError("source roots overlap in candidate identity")
    out.mkdir(parents=True, exist_ok=False)
    route.to_parquet(out / "target_free_c1_dual_route.parquet", index=False, compression="zstd")
    month = route.assign(month=route["timestamp"].dt.strftime("%Y-%m")).groupby("month", as_index=False).agg(
        candidates=("candidate_id", "size"), symbols=("symbol", "nunique"),
        bcf_mc1_mean_bps=("bcf_mc1_expected_bps", "mean"),
        current_mc1_mean_bps=("current_mc1_expected_bps", "mean"),
    )
    month.to_parquet(out / "target_free_c1_dual_route_monthly.parquet", index=False, compression="zstd")
    (out / "run_manifest.json").write_text(json.dumps({
        "schema": "causal-sr-c1-expanded-h4-target-free-route-v1",
        "scope": "offline research only; long-only target-free C1 dual-MC1 route; no path, outcome, or portfolio field is read",
        "arm": str(args.arm),
        "admission": f"BCF MC1 >= {args.threshold_bps:g} AND current MC1 >= {args.threshold_bps:g}",
        "entry": "decision timestamp + five minutes",
        "source_panels": {
            str(root / f'{args.arm}_target_free_admission.parquet'): _sha256(root / f"{args.arm}_target_free_admission.parquet")
            for root in roots
        },
        "rows": int(len(route)),
        "start": str(route["timestamp"].min()),
        "end": str(route["timestamp"].max()),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
