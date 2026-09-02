#!/usr/bin/env python3
"""Prepare a target-free, common-family entry panel for the strict 2025 C1 replay.

The panel is deliberately the BCF/current intersection: only rows that can
subsequently satisfy the dual-family admission contract are materialised into
the causal S/R engine.  It contains no policy label or outcome field.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "data_perp/artifacts/strict_r3_score_family_matched_mc1_canonical_policy_20260817_v7_current/predictions_current_v5_mc1_d2.parquet"
BCF = ROOT / "data_perp/artifacts/strict_r3_score_family_matched_mc1_canonical_policy_20260817_v7_bcf/predictions_bcf_mc1_d2.parquet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _source(path: Path, name: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["candidate_id", "__decision_ts__", "__symbol__", "side_name"])
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame = frame.loc[frame["__decision_ts__"].ge(start) & frame["__decision_ts__"].lt(end)].copy()
    if frame["candidate_id"].duplicated().any():
        raise AssertionError(f"{name} has duplicate candidate identities")
    if not frame["side_name"].astype(str).str.lower().eq("long").all():
        raise AssertionError(f"{name} is not a long-only source")
    return frame.drop(columns="side_name")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, default=CURRENT)
    parser.add_argument("--bcf", type=Path, default=BCF)
    parser.add_argument("--start", default="2025-02-01T00:00:00Z")
    parser.add_argument("--end", default="2026-01-01T00:00:00Z")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    start, end = _utc(args.start), _utc(args.end)
    if not start < end:
        raise ValueError("--start must be before --end")
    current = _source(args.current.resolve(), "current", start, end)
    bcf = _source(args.bcf.resolve(), "bcf", start, end)
    common = bcf.merge(current, on=["candidate_id", "__decision_ts__", "__symbol__"], how="inner", validate="one_to_one")
    common = common.sort_values(["__symbol__", "__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if common.empty:
        raise RuntimeError("no common BCF/current target-free candidates")
    out.mkdir(parents=True)
    common.to_parquet(out / "target_free_entry_candidates.parquet", index=False, compression="zstd")
    pd.DataFrame(columns=["candidate_id", "__symbol__", "state_decision_ts", "state_bar_15m"]).to_parquet(
        out / "empty_continuation_states.parquet", index=False, compression="zstd"
    )
    manifest = {
        "schema": "causal-sr-c1-2025-target-free-inputs-v1",
        "scope": "offline research-only; target-free common BCF/current entry identities",
        "window": [start.isoformat(), end.isoformat()],
        "current": {"path": str(args.current.resolve()), "sha256": _sha256(args.current.resolve()), "rows": int(len(current))},
        "bcf": {"path": str(args.bcf.resolve()), "sha256": _sha256(args.bcf.resolve()), "rows": int(len(bcf))},
        "common_rows": int(len(common)),
        "symbols": int(common["__symbol__"].nunique()),
        "forbidden": ["policy labels", "future path", "outcome", "MC1 prediction"],
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
