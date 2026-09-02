#!/usr/bin/env python3
"""Fit an isolated replacement for the missing August-2026 BCF monthly model.

This is deliberately a *challenger*, not an artifact repair.  The original
August model serialization is absent, so no output from this program may be
described as byte-identical to it.  The runner instead preserves every
recoverable semantic contract:

* strict prequential source ledger ending before 2026-08-01;
* original frozen BCF 120-field and 73-field contracts;
* the exact BCF Geometry/K9 object embedded in the intact July model;
* original fixed model parameters and 240k supervised cap; and
* a new immutable, content-addressed output directory.

No candidates are scored, no outcomes are joined after training, and no
exchange I/O is possible from this script.
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

from extreme_price_movements.strict_r3_canonical_v2 import (  # noqa: E402
    load_monthly_bundle,
    persist_monthly_bundle,
    train_monthly_bundle,
)


DEFAULT_LEDGER = ROOT / (
    "data_perp/artifacts/strict_r3_schema_v2_prequential_ledger_targetfree_long_"
    "2024_2026_raw15m_strictfull_20260812_v1/prequential_stack_ledger.parquet"
)
DEFAULT_CONTRACT = ROOT / "config/strict_r3_canonical_v2_feature_contract.json"
DEFAULT_GEOMETRY_SOURCE = ROOT / (
    "data_perp/artifacts/strict_r3_schema_v2_walkforward_targetfree_long_"
    "2025_aug7_2026_20260809_v1/bundles/month=2026-07"
)
DEFAULT_OUT = ROOT / (
    "data_perp/artifacts/strict_r3_bcf_august_monthly_challenger_20260823_v1"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--feature-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--geometry-source", type=Path, default=DEFAULT_GEOMETRY_SOURCE)
    parser.add_argument("--cutoff", default="2026-08-01T00:00:00Z")
    parser.add_argument("--train-cap", type=int, default=240_000)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    cutoff = _utc(args.cutoff)
    if args.out_dir.exists():
        raise FileExistsError(f"immutable challenger output already exists: {args.out_dir}")
    for path in (args.ledger, args.feature_contract, args.geometry_source / "monthly_bundle.joblib"):
        if not path.is_file():
            raise FileNotFoundError(path)

    contract = json.loads(args.feature_contract.read_text())
    base_fields = tuple(contract["base_fields_by_side"]["long"])
    context_fields = tuple(contract["severe_context_fields"])
    if len(base_fields) != 120 or len(context_fields) != 73:
        raise ValueError("frozen BCF feature contracts are not 120 base / 73 context fields")

    # This embedded object is the last intact BCF copy of the production
    # geometry.  It is copied by reference into the new model; it is never
    # fitted or otherwise mutated here.
    geometry_source = load_monthly_bundle(args.geometry_source)
    geometry = geometry_source.geometry
    original_geometry = "7a602dfb5f10bef3791fd869b17dcfaeb53f96264fa8983c01ef5fd79681191c"
    if str(geometry.bundle_sha256) != original_geometry:
        raise ValueError(
            "geometry source does not carry the expected frozen BCF identity: "
            f"{geometry.bundle_sha256}"
        )

    metadata = [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "stack_is_prequential", "r3_class", "r3_label_available_ts",
        "policy_label_available_ts", "policy_net_bps", "h12_label_available_ts",
        "h12_label_valid", "h12_tp6_sl4_net_bps", "prequential_base_rank42",
        "prequential_base_anchor_bps", "prequential_consensus_rank",
        "prequential_residual_rank", "prequential_upstream",
    ]
    columns = list(dict.fromkeys([*metadata, *base_fields, *context_fields]))
    ledger = pd.read_parquet(args.ledger, columns=columns)
    ledger["__decision_ts__"] = pd.to_datetime(ledger["__decision_ts__"], utc=True)
    if ledger["__decision_ts__"].ge(cutoff).any():
        raise ValueError("source ledger is not strictly before the August challenger cutoff")
    sides = set(ledger["side_name"].astype(str).str.lower())
    if sides != {"long"}:
        raise ValueError(f"source ledger must be long-only, found {sorted(sides)}")
    if not ledger["stack_is_prequential"].fillna(False).astype(bool).all():
        raise ValueError("source ledger contains non-prequential rows")

    source_hashes = {
        "prequential_ledger": _sha(args.ledger),
        "feature_contract": _sha(args.feature_contract),
        "geometry_source_monthly_bundle": _sha(args.geometry_source / "monthly_bundle.joblib"),
        "geometry_source_manifest": _sha(args.geometry_source / "run_manifest.json"),
    }
    bundle = train_monthly_bundle(
        cutoff=cutoff,
        training_ledger=ledger,
        frozen_geometry=geometry,
        base_fields=base_fields,
        context_fields=context_fields,
        train_cap=args.train_cap,
        source_hashes=source_hashes,
    )
    args.out_dir.mkdir(parents=True)
    monthly_dir = args.out_dir / "bundles" / "month=2026-08"
    manifest = persist_monthly_bundle(bundle, monthly_dir)
    challenger = {
        "schema": "strict_r3_bcf_missing_august_model_challenger_v1",
        "status": "complete",
        "purpose": "new BCF challenger replacing an unavailable August-2026 model serialization",
        "promotion_eligible": False,
        "cutoff": cutoff.isoformat(),
        "side_name": "long",
        "strict_prequential": True,
        "original_missing_bundle_sha256": "7c98bc4f048df717fee840f78df93a7928b0eab8651125957181ad28fc805ade",
        "challenger_bundle_sha256": manifest["bundle_sha256"],
        "geometry_contract": {
            "preserved_from": str(args.geometry_source),
            "geometry_bundle_sha256": str(geometry.bundle_sha256),
            "geometry_refit": False,
        },
        "source_hashes": source_hashes,
        "training": {
            "base_fields": len(base_fields),
            "context_fields": len(context_fields),
            "train_cap": int(args.train_cap),
            "model_parameters": {
                "base": manifest["base_params"],
                "residual": manifest["rank_params"],
                "severe": manifest["severe_params"],
            },
        },
        "safety": {
            "exchange_io": False,
            "order_submission": False,
            "outcomes_consumed_during_scoring": [],
            "requires": [
                "target-free scoring parity",
                "same-model 42-day reference audit",
                "BCF MC1 compatibility audit",
                "state-chain recovery",
                "explicit live successor seal",
            ],
        },
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(challenger, indent=2) + "\n")
    print(json.dumps({
        "event": "complete",
        "out_dir": str(args.out_dir),
        "challenger_bundle_sha256": manifest["bundle_sha256"],
        "geometry_bundle_sha256": geometry.bundle_sha256,
        "base_fit_rows": manifest["base_fit_rows"],
        "map_fit_rows": manifest["map_fit_rows"],
        "severe_fit_rows": manifest["severe_fit_rows"],
    }))


if __name__ == "__main__":
    main()
