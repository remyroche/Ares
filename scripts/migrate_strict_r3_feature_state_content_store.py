#!/usr/bin/env python3
"""Serially migrate existing strict-R3 bundles into the content store.

Historical migration deliberately uses one Python process.  Several concurrent
Pandas/PyArrow imports can oversubscribe the host before any state file is
processed; the content store itself remains safe for concurrent *future* live
publication, but a one-time backfill needs predictable, low-impact behaviour.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compact_strict_r3_feature_state_content_store import compact_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-list", type=Path, required=True)
    parser.add_argument("--object-store", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--trust-sealed-inventory", action="store_true")
    args = parser.parse_args()
    bundles = [Path(line.strip()) for line in args.bundle_list.read_text().splitlines() if line.strip()]
    args.log.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    failed = 0
    with args.log.open("w") as handle:
        for index, bundle in enumerate(bundles, start=1):
            try:
                result = compact_bundle(
                    bundle=bundle,
                    object_store=args.object_store,
                    base_bundle=None,
                    cache_dir=None,
                    panel_update_dir=None,
                    retire_private_overlay=False,
                    trust_sealed_inventory=bool(args.trust_sealed_inventory),
                )
                row = {"status": "ok", "index": index, "bundle": str(bundle), "result": result}
                completed += 1
            except Exception as exc:  # Continue so one historical defect is visible.
                row = {
                    "status": "fail", "index": index, "bundle": str(bundle),
                    "error_type": type(exc).__name__, "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failed += 1
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            print(json.dumps({"index": index, "total": len(bundles), "status": row["status"], "bundle": str(bundle)}), flush=True)
    if failed:
        raise SystemExit(f"historical state migration failed for {failed} bundle(s)")
    print(json.dumps({"status": "pass", "bundles": completed}), flush=True)


if __name__ == "__main__":
    main()
