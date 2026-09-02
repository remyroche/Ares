#!/usr/bin/env python3
"""Seal end-to-end target-free parity for the P8U successor no-order stack."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_production_contract import IDENTITY_COLUMNS
from extreme_price_movements.inference.p8u_successor_no_order_stack import P8USuccessorNoOrderStack


IDENTITY = list(IDENTITY_COLUMNS)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _compare(
    reference: pd.DataFrame, actual: pd.DataFrame, *, fields: list[str], name: str,
) -> dict[str, object]:
    keys = [*IDENTITY, "__symbol__"]
    left = reference.loc[:, [*keys, *fields]].copy()
    right = actual.loc[:, [*keys, *fields]].copy()
    merged = left.merge(right, on=keys, how="outer", validate="one_to_one", indicator=True, suffixes=("_reference", "_actual"))
    missing = int((merged["_merge"] != "both").sum())
    if missing:
        raise AssertionError(f"{name}: {missing} identity mismatch(es)")
    result: dict[str, object] = {"rows": int(len(merged)), "identity_mismatches": 0}
    for field in fields:
        a, b = merged[f"{field}_reference"], merged[f"{field}_actual"]
        if pd.api.types.is_bool_dtype(a):
            mismatches = int((a.astype(bool) != b.astype(bool)).sum())
            if mismatches:
                raise AssertionError(f"{name}/{field}: {mismatches} boolean mismatches")
            result[f"{field}_mismatches"] = mismatches
        else:
            delta = float(np.abs(a.to_numpy(float) - b.to_numpy(float)).max()) if len(a) else 0.0
            if delta != 0.0:
                raise AssertionError(f"{name}/{field}: {delta} numeric delta")
            result[f"{field}_max_abs_delta"] = delta
    return result


def _selector(reference: pd.DataFrame, actual: pd.DataFrame) -> dict[str, object]:
    fields = [
        "agreement_tier", "admission_provenance", "score_coordinate_source",
        "dual_mc1_admitted", "portfolio_tier", "portfolio_order_priority_bps",
    ]
    keys = [*IDENTITY, "__symbol__"]
    merged = reference.loc[:, [*keys, *fields]].merge(
        actual.loc[:, [*keys, *fields]], on=keys, how="outer", validate="one_to_one",
        indicator=True, suffixes=("_reference", "_actual"),
    )
    missing = int((merged["_merge"] != "both").sum())
    if missing:
        raise AssertionError(f"selector: {missing} identity mismatch(es)")
    result: dict[str, object] = {"rows": int(len(merged)), "identity_mismatches": 0}
    for field in fields:
        a, b = merged[f"{field}_reference"], merged[f"{field}_actual"]
        if pd.api.types.is_bool_dtype(a):
            mismatches = int((a.astype(bool) != b.astype(bool)).sum())
            if mismatches:
                raise AssertionError(f"selector/{field}: {mismatches} mismatches")
            result[f"{field}_mismatches"] = mismatches
        elif pd.api.types.is_numeric_dtype(a):
            delta = float(np.abs(a.to_numpy(float) - b.to_numpy(float)).max()) if len(a) else 0.0
            if delta != 0.0:
                raise AssertionError(f"selector/{field}: {delta} numeric delta")
            result[f"{field}_max_abs_delta"] = delta
        else:
            mismatches = int((a.astype(str) != b.astype(str)).sum())
            if mismatches:
                raise AssertionError(f"selector/{field}: {mismatches} mismatches")
            result[f"{field}_mismatches"] = mismatches
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--upstream-manifest-sha256", required=True)
    parser.add_argument("--mapper-root", required=True, type=Path)
    parser.add_argument("--month", required=True)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--snapshots", required=True, type=Path)
    parser.add_argument("--reference-upstream", required=True, type=Path)
    parser.add_argument("--reference-c0", required=True, type=Path)
    parser.add_argument("--reference-c1", required=True, type=Path)
    parser.add_argument("--reference-selector", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable output already exists: {args.output}")
    start = pd.Timestamp(f"{args.month}-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    features = pd.read_parquet(args.features)
    features["__decision_ts__"] = pd.to_datetime(features["__decision_ts__"], utc=True, errors="raise")
    features = features.loc[features["__decision_ts__"].between(start, end, inclusive="left")].copy()
    stack = P8USuccessorNoOrderStack.load(
        upstream_root=args.upstream_root,
        expected_upstream_manifest_sha256=args.upstream_manifest_sha256,
        mapper_root=args.mapper_root,
        month=args.month,
    )
    actual = stack.score(full_population=features, c1_snapshots=pd.read_parquet(args.snapshots))
    upstream_ref = pd.read_parquet(args.reference_upstream)
    router_actual = features.loc[:, [*IDENTITY, "__symbol__"]].merge(
        actual.router, on=IDENTITY, validate="one_to_one"
    )
    # Historical forward receipts persist the raw and primary rank only;
    # ``router_score`` is their declared identical runtime alias.
    router = _compare(upstream_ref, router_actual, fields=["router_raw_score", "router_primary_rank"], name="router")
    base_ref = upstream_ref.loc[upstream_ref["router50_eligible"].fillna(False).astype(bool)].copy()
    base_actual = actual.coordinates.loc[:, [*IDENTITY, "__symbol__", "base_score", "base_rank_ts", "under_raw_score", "under_rank_ts"]]
    base_under = _compare(base_ref, base_actual, fields=["base_score", "base_rank_ts", "under_raw_score", "under_rank_ts"], name="base_under")
    def mapper_reference(path: Path) -> pd.DataFrame:
        reference = pd.read_parquet(path).copy()
        # The sealed builder persists one common expected-EV coordinate per
        # family; the no-order inference adapter deliberately exposes it in
        # the BCF/current/auction fields required by the selector.
        for field in ("bcf_mc1_expected_bps", "current_mc1_expected_bps", "auction_priority_bps"):
            reference[field] = reference["mc1_expected_bps"].to_numpy(float)
        return reference

    c0 = _compare(
        mapper_reference(args.reference_c0), actual.mapper.c0,
        fields=["bcf_mc1_expected_bps", "current_mc1_expected_bps", "auction_priority_bps"], name="c0",
    )
    c1 = _compare(
        mapper_reference(args.reference_c1), actual.mapper.c1,
        fields=["bcf_mc1_expected_bps", "current_mc1_expected_bps", "auction_priority_bps"], name="c1",
    )
    selector = _selector(pd.read_parquet(args.reference_selector), actual.mapper.selected)
    checks = {"router": router, "base_under": base_under, "c0": c0, "c1": c1, "selector": selector}
    args.output.mkdir(parents=False)
    inputs = {
        key: {"path": str(path), "sha256": _sha(path)} for key, path in {
            "features": args.features, "snapshots": args.snapshots,
            "reference_upstream": args.reference_upstream, "reference_c0": args.reference_c0,
            "reference_c1": args.reference_c1, "reference_selector": args.reference_selector,
        }.items()
    }
    receipt = {
        "status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "target_free_no_order_router50_base_under_c0_c1_parity",
        "exchange_or_order_calls": 0, "month": args.month, "inputs": inputs,
        "upstream_manifest_sha256": args.upstream_manifest_sha256,
        "mapper_run_manifest_sha256": _sha(args.mapper_root / "run_manifest.json"), "checks": checks,
    }
    (args.output / "parity_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(checks, sort_keys=True))


if __name__ == "__main__":
    main()
