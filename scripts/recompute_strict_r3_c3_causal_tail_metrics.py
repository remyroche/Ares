#!/usr/bin/env python3
"""Recompute C3 global-tail metrics without future-path-qualified selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_strict_r3_c3_window_cadence_ablation import (
    _global_tail_metrics,
    _stability,
)


REQUIRED = [
    "candidate_id", "__decision_ts__", "final_score", "policy_path_valid",
    "policy_gross_bps", "policy_net_bps",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    global_parts: list[pd.DataFrame] = []
    monthly_parts: list[pd.DataFrame] = []
    sources: list[str] = []
    for directory in args.input:
        path = directory / "predictions.parquet"
        frame = pd.read_parquet(path, columns=[*REQUIRED, "arm"])
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
        for arm, local in frame.groupby("arm", sort=True):
            global_metrics, monthly = _global_tail_metrics(local, str(arm))
            global_metrics["source_artifact"] = str(directory)
            monthly["source_artifact"] = str(directory)
            global_parts.append(global_metrics)
            monthly_parts.append(monthly)
        sources.append(str(directory))
    global_metrics = pd.concat(global_parts, ignore_index=True)
    monthly = pd.concat(monthly_parts, ignore_index=True)
    stability = _stability(monthly, global_metrics)
    global_metrics.to_parquet(args.out_dir / "metrics_global_causal_denominator.parquet", index=False)
    monthly.to_parquet(args.out_dir / "metrics_monthly_causal_denominator.parquet", index=False)
    stability.to_parquet(args.out_dir / "top2_stability_causal_denominator.parquet", index=False)
    manifest = {
        "schema": "strict_r3_c3_causal_tail_recompute_v1",
        "selection_order": "finite score population -> global top-k -> valid outcome coverage",
        "sources": sources,
        "status": "complete",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out": str(args.out_dir)}), flush=True)


if __name__ == "__main__":
    main()
