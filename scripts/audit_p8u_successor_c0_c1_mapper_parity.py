#!/usr/bin/env python3
"""Seal an exact target-free parity receipt for successor C0/C1 inference.

This audit has no feature, portfolio, exchange, account, order, or outcome
authority.  It re-scores an already-materialised target-free Router50 panel
through the hash-bound successor mapper and compares it to its sealed mapper
receipt.  Any identity or numeric difference fails the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extreme_price_movements.inference.p8u_successor_c0_c1_mapper import (
    IDENTITY,
    P8USuccessorC0C1Mapper,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_exact(reference: pd.DataFrame, actual: pd.DataFrame, *, name: str) -> dict[str, object]:
    keys = list(IDENTITY)
    merged = reference.merge(actual, on=keys, how="outer", validate="one_to_one", indicator=True, suffixes=("_reference", "_actual"))
    identity_mismatch = int((merged["_merge"] != "both").sum())
    if identity_mismatch:
        raise AssertionError(f"{name}: {identity_mismatch} identity mismatches")
    delta = np.abs(
        merged["mc1_expected_bps"].to_numpy(float)
        - merged["bcf_mc1_expected_bps"].to_numpy(float)
    )
    max_abs = float(delta.max()) if len(delta) else 0.0
    if max_abs != 0.0:
        raise AssertionError(f"{name}: maximum expected-EV delta is {max_abs}")
    return {"rows": int(len(merged)), "identity_mismatch": identity_mismatch, "max_abs_delta_bps": max_abs}


def _selector_exact(reference: pd.DataFrame, actual: pd.DataFrame) -> dict[str, object]:
    fields = [
        "candidate_id", "__decision_ts__", "agreement_tier", "admission_provenance",
        "score_coordinate_source", "dual_mc1_admitted", "portfolio_tier",
        "portfolio_order_priority_bps",
    ]
    merged = reference.loc[:, fields].merge(
        actual.loc[:, fields], on=["candidate_id", "__decision_ts__"], how="outer",
        validate="one_to_one", indicator=True, suffixes=("_reference", "_actual"),
    )
    identity_mismatch = int((merged["_merge"] != "both").sum())
    if identity_mismatch:
        raise AssertionError(f"selector: {identity_mismatch} identity mismatches")
    result: dict[str, object] = {"rows": int(len(merged)), "identity_mismatch": identity_mismatch}
    matched = merged["_merge"].eq("both")
    for field in fields[2:]:
        left, right = merged.loc[matched, f"{field}_reference"], merged.loc[matched, f"{field}_actual"]
        if pd.api.types.is_bool_dtype(left):
            mismatch = int((left.astype(bool) != right.astype(bool)).sum())
            if mismatch:
                raise AssertionError(f"selector {field}: {mismatch} mismatches")
            result[f"{field}_mismatches"] = mismatch
        elif pd.api.types.is_numeric_dtype(left):
            delta = float(np.abs(left.to_numpy(float) - right.to_numpy(float)).max()) if len(left) else 0.0
            if delta != 0.0:
                raise AssertionError(f"selector {field}: maximum delta is {delta}")
            result[f"{field}_max_abs_delta"] = delta
        else:
            mismatch = int((left.astype(str) != right.astype(str)).sum())
            if mismatch:
                raise AssertionError(f"selector {field}: {mismatch} mismatches")
            result[f"{field}_mismatches"] = mismatch
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapper-root", required=True, type=Path)
    parser.add_argument("--month", required=True)
    parser.add_argument("--coordinates", required=True, type=Path)
    parser.add_argument("--snapshots", required=True, type=Path)
    parser.add_argument("--reference-c0", required=True, type=Path)
    parser.add_argument("--reference-c1", required=True, type=Path)
    parser.add_argument("--reference-selector", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable output already exists: {args.output}")

    coordinates = pd.read_parquet(args.coordinates)
    coordinates["__decision_ts__"] = pd.to_datetime(coordinates["__decision_ts__"], utc=True, errors="raise")
    start = pd.Timestamp(f"{args.month}-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    coordinates = coordinates.loc[coordinates["__decision_ts__"].between(start, end, inclusive="left")].copy()
    scored = P8USuccessorC0C1Mapper.load(args.mapper_root, month=args.month).score(
        coordinates=coordinates, c1_snapshots=pd.read_parquet(args.snapshots)
    )
    c0 = _required_exact(pd.read_parquet(args.reference_c0), scored.c0, name="C0")
    c1 = _required_exact(pd.read_parquet(args.reference_c1), scored.c1, name="C1")
    selector = _selector_exact(pd.read_parquet(args.reference_selector), scored.selected)

    args.output.mkdir(parents=False)
    payload = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "target_free_successor_c0_c1_mapper_parity_only",
        "month": args.month,
        "exchange_or_order_calls": 0,
        "inputs": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in {
                "coordinates": args.coordinates, "snapshots": args.snapshots,
                "reference_c0": args.reference_c0, "reference_c1": args.reference_c1,
                "reference_selector": args.reference_selector,
            }.items()
        },
        "mapper_root": str(args.mapper_root),
        "mapper_run_manifest_sha256": _sha256(args.mapper_root / "run_manifest.json"),
        "checks": {"c0": c0, "c1": c1, "selector": selector},
    }
    (args.output / "parity_receipt.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["checks"], sort_keys=True))


if __name__ == "__main__":
    main()
