#!/usr/bin/env python3
"""Export the exact frozen P8U source universe from a sealed source state.

The public-source refresh utilities require a JSON manifest.  Older P8U
experiments retained the state object but not the separate manifest file.  This
tool deliberately reconstructs *only* the manifest representation from the
state's embedded ``source_map``; it neither discovers nor filters symbols.
The output is immutable and records the predecessor hash for auditability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    state_path = args.source_state.resolve()
    out_path = args.out.resolve()
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    if out_path.exists():
        raise FileExistsError(f"immutable manifest already exists: {out_path}")

    state = joblib.load(state_path)
    source_map = state.get("source_map") if isinstance(state, dict) else None
    if not isinstance(source_map, dict) or not source_map:
        raise ValueError("sealed source state has no source_map")
    normalized = {str(symbol): source_map[symbol] for symbol in sorted(source_map)}
    panel = state.get("panel") if isinstance(state, dict) else None
    close = panel.get("close") if isinstance(panel, dict) else None
    if close is None or set(normalized) != set(map(str, close.columns)):
        raise ValueError("source_map and panel symbol identity differ")

    payload: dict[str, object] = {
        "schema": "strict_r3_p8u_frozen_source_manifest_v1",
        "derivation": "sealed_source_state_source_map_only",
        "source_state": str(state_path),
        "source_state_sha256": _sha256(state_path),
        "symbols": list(normalized),
        "source_map": normalized,
        "symbol_count": len(normalized),
        "panel_start": str(close.index[0]),
        "panel_end": str(close.index[-1]),
        "outcome_columns_consumed": [],
        "future_path_or_outcome_filter_applied": False,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(out_path, payload)
    print(json.dumps({
        "status": "pass_exact_source_map_export",
        "out": str(out_path),
        "source_state_sha256": payload["source_state_sha256"],
        "symbols": len(normalized),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
