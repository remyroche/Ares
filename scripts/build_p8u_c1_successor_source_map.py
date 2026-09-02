#!/usr/bin/env python3
"""Seal the C1-state-eligible subset of a successor's frozen universe.

The C1 S/R state has its own immutable historical identity universe.  A
successor universe may gain or lose products, so this utility derives the
explicit intersection instead of treating a missing historical checkpoint as
a data-fetch failure.  Products outside the intersection remain target-free
candidate rows but are explicitly C1-unavailable downstream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-source-map", type=Path, required=True)
    parser.add_argument("--frozen-source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    legacy_path = args.legacy_source_map.resolve()
    frozen_path = args.frozen_source_manifest.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output must be new: {output}")

    legacy = _payload(legacy_path)
    frozen = _payload(frozen_path)
    legacy_map = legacy.get("source_map")
    frozen_map = frozen.get("source_map")
    frozen_symbols = frozen.get("symbols")
    if not isinstance(legacy_map, dict) or not legacy_map:
        raise ValueError("legacy source map lacks non-empty source_map")
    if not isinstance(frozen_map, dict) or not isinstance(frozen_symbols, list):
        raise ValueError("frozen successor manifest lacks source_map/symbols")
    if list(frozen_symbols) != list(frozen_map):
        raise ValueError("frozen successor symbols and source_map order differ")

    eligible = [str(symbol) for symbol in frozen_symbols if str(symbol) in legacy_map]
    unavailable = [str(symbol) for symbol in frozen_symbols if str(symbol) not in legacy_map]
    legacy_only = sorted(set(map(str, legacy_map)).difference(map(str, frozen_symbols)))
    if not eligible:
        raise ValueError("C1 source-map intersection is empty")

    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "p8u-c1-successor-source-map-v1",
        "status": "sealed_c1_state_intersection",
        "scope": "target-free C1 append-state eligibility only; no scoring, labels, admission, portfolio, or execution authority",
        "legacy_source_map": str(legacy_path),
        "legacy_source_map_sha256": _sha256(legacy_path),
        "frozen_successor_manifest": str(frozen_path),
        "frozen_successor_manifest_sha256": _sha256(frozen_path),
        "frozen_symbol_count": len(frozen_symbols),
        "c1_state_eligible_count": len(eligible),
        "c1_state_eligible_symbols": eligible,
        "c1_unavailable_count": len(unavailable),
        "c1_unavailable_symbols": unavailable,
        "legacy_only_count": len(legacy_only),
        "legacy_only_symbols": legacy_only,
        "source_map": {symbol: legacy_map[symbol] for symbol in eligible},
        "causality": {
            "candidate_identity": "the frozen successor universe remains complete",
            "c1_missingness": "a symbol outside the historical C1 state intersection is represented as unavailable, never substituted or filtered",
            "outcome_columns_consumed": [],
        },
    }
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()
