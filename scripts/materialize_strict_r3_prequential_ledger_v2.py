#!/usr/bin/env python3
"""Materialise the reusable monthly strict-R3/map/residual OOF ledger."""

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
    SCHEMA,
    build_prequential_stack_ledger,
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
    return [str(value) for value in payload.get("base_fields", payload.get("fields", []))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--base-contract", type=Path, required=True)
    parser.add_argument("--first-held-month", required=True)
    parser.add_argument("--last-held-month")
    parser.add_argument("--reference-days", type=int, default=28)
    parser.add_argument("--side", choices=("long", "short"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    panel = pd.read_parquet(args.panel)
    panel = panel.loc[panel["side_name"].astype(str).str.lower().eq(args.side)].copy()
    ledger, audit = build_prequential_stack_ledger(
        panel,
        base_fields=_fields(args.base_contract, args.side),
        first_held_month=args.first_held_month,
        last_held_month=args.last_held_month,
        reference_days=args.reference_days,
    )
    args.out_dir.mkdir(parents=True)
    ledger.to_parquet(args.out_dir / "prequential_stack_ledger.parquet", index=False, compression="zstd")
    audit.to_parquet(args.out_dir / "prequential_fold_audit.parquet", index=False)
    manifest = {
        "schema": f"{SCHEMA}_prequential_ledger",
        "source_panel": str(args.panel), "rows": len(ledger),
        "first_held_month": args.first_held_month,
        "last_held_month": args.last_held_month,
        "side_name": args.side,
        "source_panel_sha256": _sha(args.panel),
        "base_contract_sha256": _sha(args.base_contract),
        "strict_prequential": True, "base_train_cap": 240000,
        "base_reference": f"same-model preceding {args.reference_days} days",
        "reference_window_days": args.reference_days,
        "map": "prior OOF 20-bin monotonic policy-net map",
        "residual": "ten policy-net LambdaRank heads, 4-hour UTC x side",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"event": "complete", "output": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
