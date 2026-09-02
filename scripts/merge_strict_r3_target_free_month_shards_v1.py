#!/usr/bin/env python3
"""Merge non-overlapping strict-R3 target-free monthly shards immutably.

The utility is deliberately narrow: it never derives features, opens labels,
or resolves outcomes.  It only combines pre-existing target-free shards after
checking identical ordered schemas and unique candidate identities.  This is
useful when an append-only causal feature extension covers the later part of a
calendar month and must coexist with an earlier immutable receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
PROHIBITED_TOKENS = (
    "label", "outcome", "future", "target_invalid", "policy_net_bps",
    "path_valid", "rich_policy_net_bps",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _exclusive_json(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--check-field", action="append", default=[])
    args = parser.parse_args()
    inputs = [path.resolve() for path in args.input]
    if len(inputs) < 2 or len(set(inputs)) != len(inputs):
        raise ValueError("provide at least two distinct --input shards")
    if args.out.exists() or args.manifest.exists():
        raise FileExistsError("output and manifest paths must be new immutable paths")
    schemas = [pq.ParquetFile(path).schema_arrow.names for path in inputs]
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise AssertionError("target-free shards do not have identical ordered schemas")
    names = list(schemas[0])
    missing_identity = sorted(set(IDENTITY).difference(names))
    if missing_identity:
        raise AssertionError(f"shard schema misses identity fields: {missing_identity}")
    prohibited = [
        name for name in names
        if any(token in str(name).lower() for token in PROHIBITED_TOKENS)
    ]
    if prohibited:
        raise AssertionError(f"shard schema is not target-free: {prohibited}")
    frames = [pd.read_parquet(path) for path in inputs]
    merged = pd.concat(frames, ignore_index=True)
    merged["__decision_ts__"] = pd.to_datetime(
        merged["__decision_ts__"], utc=True, errors="raise"
    )
    if merged.duplicated(list(IDENTITY)).any():
        raise AssertionError("append-only shards overlap candidate identities")
    if not merged.side_name.astype(str).eq("long").all():
        raise AssertionError("monthly target-free merge is long-only")
    merged = merged.sort_values(["__decision_ts__", "candidate_id"], kind="stable")
    checks = {}
    for field in args.check_field:
        if field not in merged:
            raise AssertionError(f"missing requested coverage field: {field}")
        checks[str(field)] = float(pd.to_numeric(merged[field], errors="coerce").notna().mean())
    args.out.parent.mkdir(parents=True, exist_ok=False)
    merged.to_parquet(args.out, index=False, compression="zstd")
    _exclusive_json(args.manifest, {
        "schema": "strict_r3_target_free_month_shard_merge_v1",
        "scope": "offline append-only target-free shard merge; no labels, outcomes, models, admission, portfolio, execution, or live state",
        "inputs": [str(path) for path in inputs],
        "input_sha256": {str(path): _sha(path) for path in inputs},
        "ordered_schema": names,
        "rows": int(len(merged)),
        "timestamps": int(merged["__decision_ts__"].nunique()),
        "symbols": int(merged["candidate_id"].astype(str).str.split("|", n=1, expand=True)[0].nunique()),
        "decision_start": str(merged["__decision_ts__"].min()),
        "decision_end": str(merged["__decision_ts__"].max()),
        "target_free": True,
        "outcome_columns_read": False,
        "identity_overlap_rows": 0,
        "field_finite_fraction": checks,
    })
    print(json.dumps({"rows": len(merged), "timestamps": int(merged["__decision_ts__"].nunique()), "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
