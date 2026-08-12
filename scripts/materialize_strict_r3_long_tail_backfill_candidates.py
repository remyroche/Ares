#!/usr/bin/env python3
"""Freeze the outcome-blind global long tail needing exact path evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--evaluation-start", default="2025-01-01")
    parser.add_argument("--evaluation-end", default="2026-08-08")
    parser.add_argument("--tail-fraction", type=float, default=0.03)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    frame = pd.read_parquet(args.predictions)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    start = pd.to_datetime(args.evaluation_start, utc=True)
    end = pd.to_datetime(args.evaluation_end, utc=True)
    frame = frame.loc[
        frame["side_name"].astype(str).str.lower().eq("long")
        & frame["__decision_ts__"].ge(start)
        & frame["__decision_ts__"].lt(end)
        & np.isfinite(pd.to_numeric(frame["final_score"], errors="coerce"))
    ].copy()
    if frame.empty or frame["candidate_id"].duplicated().any():
        raise ValueError("long prediction population is empty or has duplicate identities")
    count = max(1, int(math.ceil(float(args.tail_fraction) * len(frame))))
    selected = frame.sort_values(
        ["final_score", "candidate_id"], ascending=[False, True], kind="stable",
    ).head(count).loc[:, [
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name", "final_score",
    ]].copy()
    args.out_dir.mkdir(parents=True)
    selected.to_parquet(args.out_dir / "candidates.parquet", index=False, compression="zstd")
    monthly = selected.assign(
        month=selected["__decision_ts__"].dt.strftime("%Y-%m"),
    ).groupby("month", as_index=False).agg(
        rows=("candidate_id", "size"), symbols=("__symbol__", "nunique"),
        min_score=("final_score", "min"), max_score=("final_score", "max"),
    )
    monthly.to_parquet(args.out_dir / "monthly_population.parquet", index=False)
    manifest = {
        "schema": "strict_r3_schema_v2_long_global_tail_backfill_population_v1",
        "selection_inputs": "candidate identity and frozen final_score only; no outcome/path fields",
        "ranking": "one pooled-global long ranking; not per timestamp or per month",
        "tail_fraction": float(args.tail_fraction),
        "score_population_rows": int(len(frame)),
        "selected_rows": int(len(selected)),
        "evaluation_start": start.isoformat(),
        "evaluation_end_exclusive": end.isoformat(),
        "prediction_sha256": _sha(args.predictions),
        "future_outcomes_consumed": False,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"event": "complete", "output": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
