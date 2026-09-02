#!/usr/bin/env python3
"""Apply the canonical strict-R3 producer-vintage causal EV admission map."""

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

from extreme_price_movements.strict_r3_canonical_current import apply_current_admission_by_geometry


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--score-column", default="final_score")
    parser.add_argument("--geometry-mode", choices=("frozen", "episode-isolated"), default="frozen")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    ledger = pd.read_parquet(args.ledger)
    if args.score_column not in ledger:
        raise KeyError(f"score column absent: {args.score_column}")
    if args.score_column != "final_score":
        ledger = ledger.copy()
        ledger["canonical_final_score_before_admission"] = ledger["final_score"]
        ledger["final_score"] = ledger[args.score_column]
    mapped, audit = apply_current_admission_by_geometry(
        ledger, geometry_mode=args.geometry_mode,
    )
    args.out_dir.mkdir(parents=True)
    mapped.to_parquet(args.out_dir / "causal_admission_ledger.parquet", index=False, compression="zstd")
    audit.to_parquet(args.out_dir / "causal_admission_audit.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_current_vintage_admission_v1",
        "ledger": str(args.ledger),
        "ledger_sha256": _sha(args.ledger),
        "score_column": str(args.score_column),
        "geometry_mode": str(args.geometry_mode),
        "rows": int(len(mapped)),
        "mapping": (
            "Causal21dAdmissionSpec(hierarchical_tail_side_shrinkage_v2) partitioned by "
            "exact ev-score-family x conversion x upstream x frozen-geometry producer vintage; "
            "prior-resolved policy labels only; 50 bps floor; fail closed"
        ),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), "rows": len(mapped)}))


if __name__ == "__main__":
    main()
