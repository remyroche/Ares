#!/usr/bin/env python3
"""Seal a target-free P3/P4-routed enhanced-base panel for offline research.

The router has *only* timestamp-local routing authority.  This command
materialises the precise top-30% router population while preserving the
enhanced base score and its B0/efficiency/timing components unchanged for all
downstream consumers.  It deliberately performs no supervised fit and joins
no policy outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_strict_r3_router_downstream as downstream  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*.parquet")):
        digest.update(str(item).encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    target_free, fields, audit = downstream._materialize_target_free(
        args.router_root, args.source_root, args.out,
    )
    receipt = {
        "schema": "strict_r3_router_routed_enhanced_base_materialization_v1",
        "scope": "offline research only; target-free; no supervised fit, policy outcome, admission, portfolio, or live mutation",
        "route": "P3/P4 router-primary exact timestamp-local top 30 percent",
        "base_router_separation": "router writes only enhanced_base_routed; actual enhanced_base_bps and B0/efficiency/timing components are preserved from source",
        "target_free": str(target_free),
        "frozen_base_feature_count": len(fields),
        "months": audit["month"].tolist(),
        "rows": int(audit["rows"].sum()),
        "source_rows": int(audit["source_rows"].sum()),
        "source_hashes": {"router": _sha256(args.router_root), "enhanced_base": _sha256(args.source_root)},
    }
    (args.out / "materialization_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
