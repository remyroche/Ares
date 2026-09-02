#!/usr/bin/env python3
"""Assemble one immutable target-free shifted-phase candidate history.

The phase-native feature and scoring chain deliberately receives separate
warm-up and replay candidate grids.  Policy labels are a post-score overlay
and need one identity-complete source.  This utility joins only those two
target-free grids, checks the phase/time convention, and writes no outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


UNIVERSE_ROWS = 170
FORBIDDEN_PREFIXES = ("policy_", "label_", "outcome_", "target_")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path, phase: int) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    required = {"candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing candidate identity fields: {missing}")
    forbidden = [name for name in frame.columns if name.startswith(FORBIDDEN_PREFIXES)]
    if forbidden:
        raise ValueError(f"{path} is not target-free: {forbidden}")
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if frame["candidate_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate candidate IDs")
    if not frame["__ts__"].dt.minute.eq(phase).all() or not frame["__decision_ts__"].dt.minute.eq(phase).all():
        raise ValueError(f"{path} does not use phase={phase} timestamps")
    if not (frame["__decision_ts__"] - frame["__ts__"] == pd.Timedelta(hours=1)).all():
        raise ValueError(f"{path} violates signal-close + one-hour decision convention")
    counts = frame.groupby("__ts__", sort=False)["__symbol__"].nunique()
    if not counts.eq(UNIVERSE_ROWS).all():
        raise ValueError(f"{path} is not a complete {UNIVERSE_ROWS}-symbol candidate universe")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--phase", required=True, type=int, choices=(15, 30, 45))
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    warmup = _read(args.warmup, args.phase)
    replay = _read(args.replay, args.phase)
    if not warmup["__ts__"].max() < replay["__ts__"].min():
        raise ValueError("warm-up and replay candidate ranges must be strictly disjoint")
    output = pd.concat([warmup, replay], ignore_index=True)
    if output["candidate_id"].duplicated().any():
        raise AssertionError("candidate IDs overlap after phase union")
    args.out_dir.mkdir(parents=True)
    path = args.out_dir / "target_free_candidate_population.parquet"
    output.to_parquet(path, index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_phase_h1_target_free_candidate_union_v1",
        "phase_minutes": args.phase,
        "warmup": {"path": str(args.warmup), "sha256": _sha(args.warmup), "rows": int(len(warmup))},
        "replay": {"path": str(args.replay), "sha256": _sha(args.replay), "rows": int(len(replay))},
        "output": {"path": str(path), "sha256": _sha(path), "rows": int(len(output))},
        "target_free": True,
        "outcome_columns_consumed": [],
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", **manifest["output"]}, sort_keys=True))


if __name__ == "__main__":
    main()
