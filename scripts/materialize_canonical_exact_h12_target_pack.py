#!/usr/bin/env python3
"""Create the versioned canonical exact-H12 target pack with support metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.canonical_target_pack import materialize_canonical_target_pack


ARTIFACTS = ROOT / "data_perp" / "artifacts"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ARTIFACTS / "root_cause_exact_h12_execution_target_pack_20260801_v2")
    parser.add_argument("--supportive-canonical", type=Path, default=ARTIFACTS / "target_alignment" / "alignment_audit_20260801_v2" / "supportive_labels_canonical.parquet")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "root_cause_exact_h12_execution_target_pack_20260801_v3")
    args = parser.parse_args()
    manifest = materialize_canonical_target_pack(args.source_dir, args.output_dir, args.supportive_canonical)
    print(f"status={manifest['status']} rows={manifest['rows']} metadata_columns={len(manifest['supportive_metadata_columns'])}")
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
