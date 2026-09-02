#!/usr/bin/env python3
"""Recover a missing frozen target-free universe manifest from sealed grid receipts.

The candidate materializer consumes only the schema, spread limit, and ordered
``source_map`` keys of this historical manifest.  This tool requires preserved
immutable grid receipts that declare the lost manifest's original hash and
agree on that complete source map.  It writes a new content-addressed manifest;
it never expands, refits, or re-ranks the universe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LOST_MANIFEST_SHA256 = (
    "ceaa143bfaa0c22e7f45ead0d874b8394780b5608e57145a38f7307294c8edc5"
)
EXPECTED_SCHEMA = "strict_r3_canonical_forward_v2_target_free_hourly_grid"
DEFAULT_SOURCES = (
    ROOT / "data_perp/artifacts/strict_r3_successor_v152_live_20260823T180000Z_v5/candidate_grid/run_manifest.json",
    ROOT / "data_perp/artifacts/strict_r3_successor_v143_live_20260823T010000Z_v1/candidate_grid/run_manifest.json",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-grid-manifest", type=Path, action="append")
    args = parser.parse_args()

    output = args.out_dir.resolve()
    sources = tuple(path.resolve() for path in (args.source_grid_manifest or DEFAULT_SOURCES))
    if output.exists():
        raise FileExistsError(f"immutable output directory exists: {output}")
    if not sources:
        raise ValueError("at least one source grid manifest is required")

    first_map: dict[str, Any] | None = None
    source_receipts: list[dict[str, Any]] = []
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = _read(path)
        source_map = payload.get("source_map")
        if not isinstance(source_map, dict) or not source_map:
            raise ValueError(f"source receipt lacks source_map: {path}")
        if str(payload.get("universe_sha256")) != EXPECTED_LOST_MANIFEST_SHA256:
            raise ValueError(f"source receipt does not attest the lost manifest: {path}")
        if int(payload.get("universe_rows", -1)) != 170 or len(source_map) != 170:
            raise ValueError(f"source receipt does not contain the frozen 170 universe: {path}")
        if first_map is None:
            first_map = source_map
        elif first_map != source_map:
            raise ValueError("preserved source maps disagree; refusing to recover membership")
        source_receipts.append({
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha(path),
            "decision_or_end": str(payload.get("decision_ts") or payload.get("end_exclusive")),
        })
    assert first_map is not None

    recovery = {
        "schema": EXPECTED_SCHEMA,
        "side": "long",
        "spread_limit_bps": 100.0,
        "universe_rows": 170,
        "source_map": first_map,
        "recovery_mode": "preserved_grid_receipt_source_map_extraction_no_membership_change",
        "original_missing_manifest_sha256": EXPECTED_LOST_MANIFEST_SHA256,
        "source_map_sha256": _json_sha(first_map),
        "source_receipts": source_receipts,
        "candidate_contract": (
            "membership is the preserved historical source_map key order; current "
            "point-in-time data controls eligibility, and no historical outcome or "
            "future-path field is included"
        ),
    }
    output.mkdir(parents=True)
    manifest = output / "run_manifest.json"
    manifest.write_text(json.dumps(recovery, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "pass",
        "out_dir": str(output),
        "manifest_sha256": _sha(manifest),
        "original_missing_manifest_sha256": EXPECTED_LOST_MANIFEST_SHA256,
        "source_map_sha256": recovery["source_map_sha256"],
        "universe_rows": 170,
        "sources": source_receipts,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
