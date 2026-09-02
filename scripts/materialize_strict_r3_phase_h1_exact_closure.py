#!/usr/bin/env python3
"""Materialise one target-free full-history strict-R3 phase feature closure.

The incremental feature producer is intentionally fast for live use, but a
small group of frozen OI/liquidation/structural fields has a history longer
than its 72-hour append tail.  This research-only producer computes the
declared full causal feature contract *once per phase*, from the phase's
persisted source state, for every candidate required by May--July scoring and
its same-model reserves.  It consumes no labels, outcomes, or score columns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "scripts/materialize_strict_r3_forward_features_incremental_v13.py"
SCHEMA = "strict_r3_phase_h1_exact_feature_closure_v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--phase", type=int, required=True, choices=(0, 15, 30, 45))
    parser.add_argument("--start", required=True, help="inclusive decision timestamp")
    parser.add_argument("--end-exclusive", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    start, end = _utc(args.start), _utc(args.end_exclusive)
    if start.minute != args.phase or end.minute != args.phase or end <= start:
        raise ValueError("start/end must be a non-empty interval aligned to --phase")

    root = args.feature_root
    candidate_paths = (
        root / f"warmup_grid_phase{args.phase}" / "target_free_candidate_population.parquet",
        root / f"grid_phase{args.phase}" / "target_free_candidate_population.parquet",
    )
    pieces: list[pd.DataFrame] = []
    for path in candidate_paths:
        frame = pd.read_parquet(path)
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
        part = frame.loc[
            frame["__decision_ts__"].ge(start) & frame["__decision_ts__"].lt(end)
        ].copy()
        if not part.empty:
            pieces.append(part)
    if not pieces:
        raise ValueError("candidate grids have no rows in exact-closure window")
    candidates = pd.concat(pieces, ignore_index=True)
    identity = ["__decision_ts__", "__symbol__", "side_name"]
    duplicate = candidates["candidate_id"].duplicated(keep=False)
    if duplicate.any():
        conflict = (candidates.loc[duplicate].groupby("candidate_id", sort=False)[identity]
                    .nunique(dropna=False).gt(1).any(axis=None))
        if conflict:
            raise ValueError("candidate warm-up/replay boundary has conflicting identities")
        candidates = candidates.drop_duplicates("candidate_id", keep="last")
    candidates = candidates.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    counts = candidates.groupby("__ts__", sort=False)["__symbol__"].nunique()
    if not counts.eq(170).all():
        raise ValueError("exact closure requires every frozen-universe symbol at every phase timestamp")

    states = sorted((root / f"phase{args.phase}_streamed_v2/source_states").glob(
        "block_*/feature_panel_state.joblib"
    ))
    if len(states) != 1:
        raise ValueError("phase must retain exactly one final causal source state")
    args.out_dir.mkdir(parents=True)
    candidate_path = args.out_dir / "target_free_candidates.parquet"
    candidates.to_parquet(candidate_path, index=False, compression="zstd")
    cache_dir = args.out_dir / "feature_cache"
    feature_dir = args.out_dir / "features"
    command = [
        sys.executable, str(FEATURES),
        "--candidates", str(candidate_path),
        "--panel-state", str(states[0]),
        "--cache-dir", str(cache_dir),
        "--side", "long", "--out-dir", str(feature_dir),
        "--bootstrap-state", "--emit-all-candidate-timestamps",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
    (args.out_dir / "feature.stdout.log").write_text(completed.stdout)
    output = feature_dir / "canonical120_features.parquet"
    if not output.is_file():
        raise FileNotFoundError("full-history materialiser did not emit canonical120_features")
    result = pd.read_parquet(output, columns=["candidate_id", "__decision_ts__"])
    result["__decision_ts__"] = pd.to_datetime(result["__decision_ts__"], utc=True)
    if len(result) != len(candidates) or result["candidate_id"].duplicated().any():
        raise AssertionError("exact closure changed candidate identities")
    if not result["__decision_ts__"].between(start, end, inclusive="left").all():
        raise AssertionError("exact closure emitted outside its requested decision interval")
    manifest = {
        "schema": SCHEMA,
        "phase_minutes": args.phase,
        "start": start.isoformat(), "end_exclusive": end.isoformat(),
        "candidate_rows": int(len(candidates)),
        "candidate_sha256": _sha(candidate_path),
        "source_state": str(states[0]), "source_state_sha256": _sha(states[0]),
        "feature_output": str(output), "feature_output_sha256": _sha(output),
        "outcome_columns_consumed": [],
        "full_causal_history": True,
        "incremental_tail_imputation": False,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", **manifest}))


if __name__ == "__main__":
    main()
