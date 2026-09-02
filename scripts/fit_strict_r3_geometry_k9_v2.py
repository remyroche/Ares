#!/usr/bin/env python3
"""Fit the one immutable October-December 2024 geometry/K9 bundle."""

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
    GEOMETRY_TARGET_H12_TP6_VS_BASE,
    GEOMETRY_TARGET_POLICY_RESIDUAL,
    POLICY_RESIDUAL_GEOMETRY_SCHEMA,
    fit_frozen_geometry_k9,
    persist_geometry_bundle,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fields(path: Path, side: str) -> list[str]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [str(value) for value in payload]
    if "base_fields_by_side" in payload:
        return [str(value) for value in payload["base_fields_by_side"][side]]
    # P0/F90 freezes the field *selection* in its base-contract config rather
    # than duplicating 90 names in another mutable list.  Resolve that sealed
    # selection here, then persist the exact ordered fields in the Geometry/K9
    # manifest.  Long canonical contracts continue through the branch above.
    selection = payload.get("feature_contract", {}).get("selection_artifact")
    if selection:
        selection_path = Path(str(selection))
        if not selection_path.is_absolute():
            selection_path = ROOT / selection_path
        selected = json.loads(selection_path.read_text())
        size = int(selected.get("recommended_feature_size_development_only", 90))
        fields = [str(value) for value in selected.get("feature_sets", {}).get(str(size), [])]
        if not fields:
            raise ValueError("selected P0 feature contract is empty")
        return fields
    return [str(value) for value in payload.get("encoder_fields", payload.get("fields", []))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-ledger", type=Path, required=True)
    parser.add_argument("--encoder-contract", type=Path, required=True)
    parser.add_argument("--side", choices=("long", "short"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--target-mode", choices=(GEOMETRY_TARGET_H12_TP6_VS_BASE, GEOMETRY_TARGET_POLICY_RESIDUAL),
        default=GEOMETRY_TARGET_H12_TP6_VS_BASE,
    )
    parser.add_argument("--policy-residual-hurdle-bps", type=float, default=50.0)
    parser.add_argument(
        "--schema", type=str,
        help="Required for a noncanonical geometry target; prevents accidental canonical replacement.",
    )
    args = parser.parse_args()
    fields = _fields(args.encoder_contract, args.side)
    if not fields:
        raise ValueError("encoder contract is empty")
    warmup = pd.read_parquet(args.warmup_ledger)
    if "side_name" not in warmup:
        raise ValueError("geometry warm-up ledger lacks side_name")
    observed_side = warmup["side_name"].astype(str).str.strip().str.lower()
    if not observed_side.isin(("long", "short")).all():
        raise ValueError("geometry warm-up ledger contains noncanonical sides")
    source_rows_by_side = {
        str(key): int(value)
        for key, value in observed_side.value_counts(sort=True).to_dict().items()
    }
    warmup = warmup.loc[observed_side.eq(args.side)].copy()
    if warmup.empty:
        raise ValueError(f"geometry warm-up contains no {args.side} rows")
    if args.target_mode == GEOMETRY_TARGET_POLICY_RESIDUAL and args.schema != POLICY_RESIDUAL_GEOMETRY_SCHEMA:
        raise ValueError(
            f"policy-residual geometry requires --schema {POLICY_RESIDUAL_GEOMETRY_SCHEMA}"
        )
    geometry = fit_frozen_geometry_k9(
        warmup, encoder_fields=fields, side_name=args.side, target_mode=args.target_mode,
        policy_residual_hurdle_bps=float(args.policy_residual_hurdle_bps),
    )
    geometry.fit_audit["source_hashes"] = {
        "warmup_ledger": _sha(args.warmup_ledger),
        "encoder_contract": _sha(args.encoder_contract),
    }
    geometry.fit_audit["source_rows_by_side_before_filter"] = source_rows_by_side
    manifest = persist_geometry_bundle(geometry, args.out_dir, schema=args.schema or "strict_r3_geometry_k9_oct_dec_2024_v2")
    print(json.dumps({"event": "complete", "output": str(args.out_dir), **manifest}, default=str))


if __name__ == "__main__":
    main()
