#!/usr/bin/env python3
"""Build a no-order lock-step Geometry/K9 state for the paired August peer.

This is an explicit one-time historical bootstrap.  It scores only the
 preserved July 4--31 target-free reserve and a continuous held prefix under
 one exact upstream/conversion pair; it neither fits a model nor joins
 outcomes.  The historical August 1--7 check remains the default; a caller
 must explicitly name any later terminal decision timestamp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_current import (
    load_four_week_conversion_bundle,
    load_monthly_upstream_bundle,
)
from scripts.score_strict_r3_forward import _join, _score_current_lockstep


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversion", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--reference-candidates", type=Path, required=True)
    parser.add_argument("--reference-features", type=Path, required=True)
    parser.add_argument("--held-candidates", type=Path, required=True)
    parser.add_argument("--held-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunk-hours", type=int, default=72)
    parser.add_argument(
        "--expected-held-end",
        type=str,
        default=None,
        help=(
            "inclusive UTC decision timestamp for an explicitly extended "
            "continuous held prefix; default preserves the Aug 7 bootstrap"
        ),
    )
    args = parser.parse_args()
    out = args.out_dir
    if out.exists():
        raise FileExistsError(f"immutable bootstrap output exists: {out}")
    conversion = load_four_week_conversion_bundle(args.conversion)
    upstream = load_monthly_upstream_bundle(args.upstream)
    cutoff = pd.Timestamp(conversion.cutoff).tz_convert("UTC")
    if cutoff != pd.Timestamp(upstream.cutoff).tz_convert("UTC"):
        raise ValueError("paired bootstrap requires matching cutoffs")
    reference = _join(args.reference_candidates, args.reference_features)
    held = _join(args.held_candidates, args.held_features)
    held["__decision_ts__"] = pd.to_datetime(held["__decision_ts__"], utc=True)
    expected_start = cutoff + pd.Timedelta(hours=1)
    held = held.loc[held["__decision_ts__"].ge(expected_start)].copy()
    if held.empty or held["__decision_ts__"].min() != expected_start:
        raise ValueError("held prefix does not begin at activation + one hour")
    expected_end = (
        pd.Timestamp(args.expected_held_end).tz_localize("UTC")
        if pd.Timestamp(args.expected_held_end).tzinfo is None
        else pd.Timestamp(args.expected_held_end).tz_convert("UTC")
    ) if args.expected_held_end is not None else cutoff + pd.Timedelta(days=6, hours=23)
    if held["__decision_ts__"].max() != expected_end:
        raise ValueError("held prefix does not end at August 7 23:00 UTC")
    out.mkdir(parents=True)
    predictions, audit, hashes = _score_current_lockstep(
        conversion_bundle=conversion,
        upstream_bundle=upstream,
        reference=reference,
        held=held,
        chunk_hours=args.chunk_hours,
        geometry_k9_state_out=out / "geometry_k9_state",
    )
    predictions.to_parquet(out / "score_decomposition.parquet", index=False, compression="zstd")
    audit.to_parquet(out / "score_audit.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_paired_lockstep_bootstrap_v1",
        "status": "complete_no_order_target_free",
        "outcome_columns_consumed": [],
        "order_submission": False,
        "exchange_calls": 0,
        "conversion_bundle_sha256": conversion.manifest["bundle_sha256"],
        "upstream_bundle_sha256": upstream.manifest["bundle_sha256"],
        "reference_start": str(reference["__decision_ts__"].min()),
        "reference_end": str(reference["__decision_ts__"].max()),
        "held_start": str(held["__decision_ts__"].min()),
        "held_end": str(held["__decision_ts__"].max()),
        "reference_rows": int(len(reference)),
        "held_rows": int(len(held)),
        "source_hashes": {name: _sha(path) for name, path in {
            "reference_candidates": args.reference_candidates,
            "reference_features": args.reference_features,
            "held_candidates": args.held_candidates,
            "held_features": args.held_features,
        }.items()},
        "score_hashes": hashes,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
