#!/usr/bin/env python3
"""Assemble archived point-in-time BCF/current score receipts for an exact replay.

This utility is deliberately source-only: it neither scores, maps, fits, nor
looks at policy outcomes.  It creates append-only score panels from immutable
receipts so the downstream candidate materialiser can apply its strict
prequential BCF mapping.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp/artifacts"
BCF_BATCH = ART / "strict_r3_bcf_august_batch_scores_20260817_v1/predictions.parquet"
CURRENT_BATCH = ART / "strict_r3_bcf_current_dual_fullcycle_smoke_20260817T050000Z_v5/score/predictions.parquet"


def _stamp(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _held_at(path: Path, decision: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    timestamps = pd.to_datetime(frame["__decision_ts__"], utc=True)
    result = frame.loc[timestamps.eq(decision)].copy()
    if result.empty:
        raise ValueError(f"{path} has no held rows at {decision}")
    return result


def _priority(path: Path) -> tuple[int, str]:
    """Prefer primary live receipts, with audited recovery receipts only as gaps."""
    value = str(path)
    if "_live_" in value:
        return (0, value)
    if "_backfill_" in value:
        return (1, value)
    if "stateful_recovery" in value:
        return (2, value)
    if "newest_runtime" in value:
        return (3, value)
    return (4, value)


def _receipt_index(start: pd.Timestamp, end: pd.Timestamp) -> dict[pd.Timestamp, Path]:
    receipts: dict[pd.Timestamp, list[Path]] = {}
    containers = [*ART.glob("strict_r3_*20260817*"), *ART.glob("strict_r3_*20260818*")]
    for container in sorted(set(containers)):
      for score in container.rglob("cycle/score/predictions.parquet"):
        try:
            manifest = json.loads((score.parent / "run_manifest.json").read_text())
            bundle = str(manifest.get("bundle_sha256", ""))
            if bundle != "094b26e5fe9a18b0696d444f553b318be8b3cea6b1c0f5c43a01e84347a08fe7":
                continue
            frame = pd.read_parquet(score, columns=["__decision_ts__"])
            latest = pd.to_datetime(frame["__decision_ts__"], utc=True).max()
            if start <= latest < end:
                receipts.setdefault(latest, []).append(score)
        except Exception:
            continue
    # Keep a deterministic path where a live retry produced equivalent receipts.
    return {stamp: sorted(paths, key=_priority)[0] for stamp, paths in receipts.items()}


def _bcf_index(start: pd.Timestamp, end: pd.Timestamp) -> dict[pd.Timestamp, Path]:
    receipts: dict[pd.Timestamp, list[Path]] = {}
    containers = [*ART.glob("strict_r3_*20260817*"), *ART.glob("strict_r3_*20260818*")]
    for container in sorted(set(containers)):
      for score in container.rglob("cycle/bcf_score/predictions.parquet"):
        try:
            frame = pd.read_parquet(score, columns=["__decision_ts__"])
            latest = pd.to_datetime(frame["__decision_ts__"], utc=True).max()
            if start <= latest < end:
                receipts.setdefault(latest, []).append(score)
        except Exception:
            continue
    return {stamp: sorted(paths, key=_priority)[0] for stamp, paths in receipts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    start, end = _stamp(args.start), _stamp(args.end)
    out = args.out_dir.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    out.mkdir(parents=True)

    bcf = pd.read_parquet(BCF_BATCH)
    bcf["__decision_ts__"] = pd.to_datetime(bcf["__decision_ts__"], utc=True)
    bcf = bcf.loc[bcf["__decision_ts__"].ge(start) & bcf["__decision_ts__"].lt(end)].copy()
    current = pd.read_parquet(CURRENT_BATCH)
    current["__decision_ts__"] = pd.to_datetime(current["__decision_ts__"], utc=True)
    current = current.loc[current["__decision_ts__"].ge(start) & current["__decision_ts__"].lt(end)].copy()

    current_idx = _receipt_index(start, end)
    bcf_idx = _bcf_index(start, end)
    # Batch receipts take precedence through their actual final timestamp.
    covered_current = set(current["__decision_ts__"].unique())
    covered_bcf = set(bcf["__decision_ts__"].unique())
    current_added, bcf_added = [], []
    for stamp, path in current_idx.items():
        if stamp not in covered_current:
            current_added.append(_held_at(path, stamp))
    for stamp, path in bcf_idx.items():
        if stamp not in covered_bcf:
            bcf_added.append(_held_at(path, stamp))
    if current_added:
        current = pd.concat([current, *current_added], ignore_index=True)
    if bcf_added:
        bcf = pd.concat([bcf, *bcf_added], ignore_index=True)
    current = current.sort_values(["__decision_ts__", "candidate_id"], kind="stable").drop_duplicates("candidate_id", keep="last")
    bcf = bcf.sort_values(["__decision_ts__", "candidate_id"], kind="stable").drop_duplicates("candidate_id", keep="last")
    current.to_parquet(out / "current_scores.parquet", index=False, compression="zstd")
    bcf.to_parquet(out / "bcf_scores.parquet", index=False, compression="zstd")
    (out / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_archived_live_score_extension_v1",
        "target_free": True,
        "start": start.isoformat(), "end_exclusive": end.isoformat(),
        "current_rows": len(current), "bcf_rows": len(bcf),
        "current_latest": str(current["__decision_ts__"].max()),
        "bcf_latest": str(bcf["__decision_ts__"].max()),
        "current_added_hours": [str(x) for x in sorted(set(pd.to_datetime(pd.concat(current_added)["__decision_ts__"], utc=True))) ] if current_added else [],
        "bcf_added_hours": [str(x) for x in sorted(set(pd.to_datetime(pd.concat(bcf_added)["__decision_ts__"], utc=True))) ] if bcf_added else [],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
