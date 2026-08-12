#!/usr/bin/env python3
"""Attach immutable monthly-producer lineage to a strict-R3 OOF ledger.

Older schema-v5 ledgers persisted the conversion-bundle and frozen geometry
identities but not the upstream monthly bundle hash on every row.  The hash is
recoverable without rescoring: the producer selects exactly the bundle for the
candidate's UTC decision month.  This utility materialises that relationship
as a separate immutable sidecar so admission can require an exact fitted
conversion × upstream producer, rather than treating all CDF scores as one
economic population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_current import (  # noqa: E402
    _current_ev_score_family_id,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_for_month(root: Path, month: str) -> Path:
    path = root / f"month={month}" / "run_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing immutable upstream manifest for ledger month {month}: {path}",
        )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--upstream-bundle-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable lineage output already exists: {args.out_dir}")

    required = [
        "candidate_id", "__decision_ts__", "conversion_bundle_sha256",
        "geometry_bundle_sha256", "stack_is_prequential",
    ]
    ledger = pd.read_parquet(args.ledger, columns=required)
    missing = sorted(set(required).difference(ledger.columns))
    if missing:
        raise ValueError(f"strict-R3 ledger lacks producer-lineage inputs: {missing}")
    if ledger["candidate_id"].isna().any() or ledger["candidate_id"].duplicated().any():
        raise ValueError("strict-R3 ledger must have immutable unique candidate IDs")
    if not ledger["stack_is_prequential"].fillna(False).astype(bool).all():
        raise ValueError("producer lineage requires strict prequential ledger rows")
    for column in ("conversion_bundle_sha256", "geometry_bundle_sha256"):
        if ledger[column].isna().any():
            raise ValueError(f"producer lineage encountered null {column}")
    ledger["__decision_ts__"] = pd.to_datetime(
        ledger["__decision_ts__"], utc=True, errors="raise",
    )
    ledger["__upstream_month__"] = ledger["__decision_ts__"].dt.strftime("%Y-%m")

    upstream_by_month: dict[str, str] = {}
    manifest_hashes: dict[str, str] = {}
    for month in sorted(ledger["__upstream_month__"].unique().tolist()):
        manifest_path = _manifest_for_month(args.upstream_bundle_root, str(month))
        manifest = json.loads(manifest_path.read_text())
        bundle_hash = manifest.get("bundle_sha256")
        if not isinstance(bundle_hash, str) or not bundle_hash:
            raise ValueError(f"upstream manifest lacks bundle_sha256: {manifest_path}")
        cutoff = pd.to_datetime(manifest.get("cutoff"), utc=True, errors="raise")
        if cutoff.strftime("%Y-%m") != str(month):
            raise ValueError(
                f"upstream manifest cutoff does not match its directory month: {manifest_path}",
            )
        upstream_by_month[str(month)] = bundle_hash
        manifest_hashes[str(month)] = _sha(manifest_path)

    output = ledger.loc[:, [
        "candidate_id", "__decision_ts__", "conversion_bundle_sha256",
        "geometry_bundle_sha256",
    ]].copy()
    output["upstream_bundle_sha256"] = ledger["__upstream_month__"].map(upstream_by_month)
    output["ev_score_family_id"] = output["geometry_bundle_sha256"].astype(str).map(
        _current_ev_score_family_id,
    )
    if output["upstream_bundle_sha256"].isna().any():
        raise AssertionError("producer-lineage month map lost an upstream bundle")
    args.out_dir.mkdir(parents=True)
    output.to_parquet(
        args.out_dir / "producer_lineage.parquet", index=False, compression="zstd",
    )
    manifest = {
        "schema": "strict_r3_current_v5_producer_lineage_v1",
        "ledger": str(args.ledger),
        "ledger_sha256": _sha(args.ledger),
        "upstream_bundle_root": str(args.upstream_bundle_root),
        "rows": int(len(output)),
        "unique_conversion_vintages": int(output["conversion_bundle_sha256"].nunique()),
        "unique_upstream_vintages": int(output["upstream_bundle_sha256"].nunique()),
        "unique_geometry_vintages": int(output["geometry_bundle_sha256"].nunique()),
        "monthly_upstream_bundle_sha256": upstream_by_month,
        "monthly_upstream_manifest_sha256": manifest_hashes,
        "derivation": "decision_timestamp_utc_month_to_immutable_monthly_upstream_bundle",
        "outcome_columns_consumed": [],
        "scores_recomputed": False,
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
    )
    print(json.dumps({"event": "complete", "output": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
