#!/usr/bin/env python3
"""Prepare immutable target-free requests for canonical P8U warm-state QA.

This helper projects only candidate identity/symbol fields from a historical
full-causal reference and writes the 175 requested feature columns into a
separate *audit-only* reference.  The candidate request itself contains no
label or future-path values.  It is useful for proving post-bootstrap state
reuse against canonical training semantics before any inference promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_warm_feature_state import sha256_file  # noqa: E402


IDENTITY = ("candidate_id", "__decision_ts__", "side_name", "__ts__", "__symbol__")


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-plan", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--timestamps", nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(out_dir)
    plan = json.loads(args.feature_plan.read_text())
    fields = tuple(map(str, plan.get("full_union") or ()))
    if not fields or len(fields) != len(set(fields)):
        raise ValueError("warm sequence must use a non-empty, duplicate-free sealed plan")
    identity = list(IDENTITY)
    source_hash = sha256_file(args.source_panel)
    plan_hash = sha256_file(args.feature_plan)
    out_dir.mkdir(parents=True)
    for raw in args.timestamps:
        timestamp = _utc(raw)
        values = pd.read_parquet(args.reference, columns=identity + list(fields))
        mask = pd.to_datetime(values["__ts__"], utc=True, errors="raise").eq(timestamp)
        reference = values.loc[mask].copy()
        if reference.empty:
            raise ValueError(f"canonical reference has no rows for {timestamp}")
        # The full primitive/cross-sectional graph still runs on 160 symbols.
        # The historical candidate table is a separately eligible subset (85
        # here), so parity compares only its exact target-free identities.
        if reference["__symbol__"].astype(str).nunique() != len(reference):
            raise ValueError(f"canonical reference has duplicate candidate symbols at {timestamp}")
        candidates = reference.loc[:, identity].copy()
        tag = timestamp.strftime("%Y%m%dT%H%M%SZ")
        candidate_path = out_dir / f"candidates_{tag}.parquet"
        reference_path = out_dir / f"reference_{tag}.parquet"
        candidates.to_parquet(candidate_path, index=False, compression="zstd")
        reference.to_parquet(reference_path, index=False, compression="zstd")
        request = {
            "schema": "strict_r3_p8u_canonical_warm_feature_request_v1",
            "signal_ts": timestamp.isoformat(),
            "candidates": str(candidate_path.relative_to(ROOT)),
            "candidates_sha256": sha256_file(candidate_path),
            "source_panel": str(args.source_panel.resolve().relative_to(ROOT)),
            "source_panel_sha256": source_hash,
            # This historical QA sequence deliberately reuses one immutable
            # primitive-source snapshot.  A live source producer advances this
            # field to the preceding snapshot hash on every append.
            "source_parent_sha256": source_hash,
            "reference_features": str(reference_path.relative_to(ROOT)),
            "reference_features_sha256": sha256_file(reference_path),
            "feature_plan_sha256": plan_hash,
            "outcome_columns_consumed": [],
        }
        _write_json(out_dir / f"request_{tag}.json", request)
    manifest = {
        "schema": "strict_r3_p8u_canonical_warm_sequence_v1",
        "feature_plan_sha256": plan_hash,
        "source_panel_sha256": source_hash,
        "timestamps": [_utc(value).isoformat() for value in args.timestamps],
        "outcome_columns_consumed": [],
    }
    _write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
