#!/usr/bin/env python3
"""Validate and atomically publish an interrupted successor feature-panel build.

This is intentionally narrow: it can only finalise a complete temporary
directory produced by ``materialize_strict_r3_p8u_successor_feature_panel``.
It recomputes no features and will refuse a missing, extra, reordered, or
reference-mismatched feature part.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_strict_r3_p8u_successor_feature_panel import (  # noqa: E402
    IDENTITY,
    _reference_parity,
    _sha256,
    _utc,
    _write_once,
)


def _part_record(path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(path, columns=["__decision_ts__"])
    timestamps = _utc(frame["__decision_ts__"])
    if frame.empty:
        raise AssertionError(f"feature part is empty: {path.name}")
    return {
        "name": path.name,
        "rows": int(len(frame)),
        "first_timestamp": timestamps.min().isoformat(),
        "last_timestamp": timestamps.max().isoformat(),
        "sha256": _sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporary-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--feature-plan", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--reference-features", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    temporary, output = args.temporary_dir.resolve(), args.out_dir.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    parts_dir = temporary / "features"
    coverage_path = temporary / "feature_coverage.parquet"
    if not temporary.is_dir() or not parts_dir.is_dir() or not coverage_path.is_file():
        raise FileNotFoundError("temporary feature panel is incomplete")
    plan = json.loads(args.feature_plan.read_text(encoding="utf-8"))
    fields = tuple(map(str, plan.get("full_union") or ()))
    if not fields or len(fields) != len(set(fields)):
        raise AssertionError("feature plan must declare a unique nonempty union")
    candidates = pd.read_parquet(args.candidates, columns=list(IDENTITY)).copy()
    candidates["__decision_ts__"] = _utc(candidates["__decision_ts__"])
    expected_timestamps = pd.DatetimeIndex(sorted(candidates["__decision_ts__"].unique()))
    expected_parts = (len(expected_timestamps) + 168 - 1) // 168
    paths = sorted(parts_dir.glob("part_*.parquet"))
    if len(paths) != expected_parts:
        raise AssertionError(f"temporary panel has {len(paths)} parts; expected {expected_parts}")
    records: list[dict[str, Any]] = []
    rows = 0
    for expected_number, path in enumerate(paths):
        if path.name != f"part_{expected_number:04d}.parquet":
            raise AssertionError("feature part ordering is not contiguous")
        schema = set(pq.ParquetFile(path).schema.names)
        if set(IDENTITY).difference(schema) or set(fields).difference(schema):
            raise AssertionError(f"feature part lacks contract columns: {path.name}")
        record = _part_record(path)
        records.append(record)
        rows += int(record["rows"])
    if rows != len(candidates):
        raise AssertionError("temporary feature parts do not cover every target-free candidate exactly once")
    coverage = pd.read_parquet(coverage_path)
    if len(coverage) != len(fields) * len(paths):
        raise AssertionError("feature coverage receipt does not cover every field and part")
    reference = pd.read_parquet(args.reference_features, columns=["__decision_ts__"])
    reference_ts = _utc(reference["__decision_ts__"]).max()
    final_part = next(
        parts_dir / str(part["name"])
        for part in records
        if pd.Timestamp(part["first_timestamp"]) <= reference_ts <= pd.Timestamp(part["last_timestamp"])
    )
    parity = _reference_parity(
        reference=args.reference_features, final_part=final_part,
        fields=fields, timestamp=reference_ts,
    )
    _write_once(temporary / "run_manifest.json", {
        "schema": "strict_r3_p8u_successor_feature_panel_v1",
        "status": "complete_target_free",
        "scope": "offline canonical feature materialisation; no target/outcome/policy/portfolio/exchange/order input",
        "publication": "validated recovery of an otherwise complete interrupted temporary build; no feature recomputation",
        "candidate_rows": int(len(candidates)), "timestamps": int(len(expected_timestamps)), "symbols": int(candidates["__symbol__"].nunique()),
        "feature_fields": len(fields), "parts": records,
        "source_panel": str(args.source_panel.resolve()), "source_panel_sha256": _sha256(args.source_panel),
        "candidates": str(args.candidates.resolve()), "candidates_sha256": _sha256(args.candidates),
        "feature_plan": str(args.feature_plan.resolve()), "feature_plan_sha256": _sha256(args.feature_plan),
        "source_manifest": str(args.source_manifest.resolve()), "source_manifest_sha256": _sha256(args.source_manifest),
        "reference_features": str(args.reference_features.resolve()), "reference_features_sha256": _sha256(args.reference_features),
        "final_timestamp_reference_parity": parity,
        "finalization_runtime_seconds": time.monotonic() - started,
    })
    os.replace(temporary, output)
    print(output)


if __name__ == "__main__":
    main()
