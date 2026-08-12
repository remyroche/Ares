#!/usr/bin/env python3
"""Fit and seal the canonical A4 model plus bounded-A5 calibration."""

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

from extreme_price_movements.strict_r3_a5_trust import (  # noqa: E402
    fit_a5_calibration,
    persist_a5_bundle,
    train_a4_bundle,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prequential-ledger", type=Path, action="append", required=True)
    parser.add_argument("--cell-day-provenance", type=Path, action="append", required=True)
    parser.add_argument("--expected-map-field", default="cell_day_trim_15pct__expected_net_bps")
    parser.add_argument(
        "--a4-oos-manifest", type=Path, required=True,
        help="A5 longer-validation manifest containing immutable prior A4 OOS paths.",
    )
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    score = pd.concat(
        [pd.read_parquet(path) for path in args.prequential_ledger], ignore_index=True,
    )
    mapped = pd.concat(
        [pd.read_parquet(path) for path in args.cell_day_provenance], ignore_index=True,
    )
    for frame, name in ((score, "prequential"), (mapped, "Cell-day")):
        if frame["candidate_id"].duplicated().any():
            raise ValueError(f"{name} partitions contain duplicate candidate IDs")
    required_map = {"candidate_id", "__decision_ts__", args.expected_map_field}
    missing = sorted(required_map.difference(mapped.columns))
    if missing:
        raise ValueError(f"Cell-day provenance lacks {missing}")
    mapped = mapped.loc[:, list(required_map)].rename(columns={
        "__decision_ts__": "__map_decision_ts__",
        args.expected_map_field: "raw_expected_bps",
    })
    score["candidate_id"] = score["candidate_id"].astype(str)
    mapped["candidate_id"] = mapped["candidate_id"].astype(str)
    joined = score.merge(mapped, on="candidate_id", how="left", validate="one_to_one")
    joined["__decision_ts__"] = pd.to_datetime(joined["__decision_ts__"], utc=True, errors="raise")
    joined["__map_decision_ts__"] = pd.to_datetime(
        joined["__map_decision_ts__"], utc=True, errors="coerce",
    )
    overlap = joined["__map_decision_ts__"].notna()
    if not joined.loc[overlap, "__decision_ts__"].eq(
        joined.loc[overlap, "__map_decision_ts__"]
    ).all():
        raise ValueError("Cell-day identity/timestamp mismatch")
    if not joined["stack_is_prequential"].fillna(False).astype(bool).all():
        raise ValueError("A4 source contains non-prequential upstream rows")
    geometry = joined["geometry_bundle_sha256"].dropna().astype(str).unique()
    if len(geometry) != 1:
        raise ValueError("A4 requires one frozen geometry/K9 semantic bundle")

    source_hashes = {
        "prequential_ledgers": [
            {"path": str(path), "sha256": _sha(path)} for path in args.prequential_ledger
        ],
        "cell_day_provenance": [
            {"path": str(path), "sha256": _sha(path)} for path in args.cell_day_provenance
        ],
        "expected_map_field": args.expected_map_field,
        "geometry_bundle_sha256": str(geometry[0]),
    }
    a4 = train_a4_bundle(joined, cutoff=args.cutoff, source_hashes=source_hashes)

    validation = json.loads(args.a4_oos_manifest.read_text())
    if validation.get("schema") != "a5_longer_prequential_calibration_v1":
        raise ValueError("A4 OOS manifest has the wrong schema")
    oos_parts: list[pd.DataFrame] = []
    oos_hashes: dict[str, str] = {}
    for item in validation.get("a4", []):
        path = ROOT / str(item["path"])
        observed = _sha(path)
        if observed != str(item["sha256"]):
            raise ValueError(f"A4 OOS artifact hash mismatch: {path}")
        oos_hashes[str(path.relative_to(ROOT))] = observed
        oos_parts.append(pd.read_parquet(path))
    if not oos_parts:
        raise ValueError("A4 OOS manifest contains no predictions")
    oos = pd.concat(oos_parts, ignore_index=True)
    if oos["candidate_id"].duplicated().any():
        raise ValueError("A4 OOS calibration ledger has duplicate candidate IDs")
    calibration = fit_a5_calibration(
        oos, cutoff=args.cutoff,
        source_hashes={
            "a4_oos_manifest": _sha(args.a4_oos_manifest),
            **oos_hashes,
        },
    )
    manifest = persist_a5_bundle(a4, calibration, args.out_dir)
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
