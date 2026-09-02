#!/usr/bin/env python3
"""Advance and score one exact, target-free P8U source timestamp offline.

This CLI is intentionally not an execution program.  It writes a sealed,
unactivated feature/score receipt only; neither exchange I/O nor submission is
available from this process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_canonical_single_timestamp_runtime import (  # noqa: E402
    P8UCanonicalSingleTimestampConfig,
    P8UCanonicalSingleTimestampRuntime,
    extract_one_timestamp_snapshot,
)
from extreme_price_movements.inference.p8u_warm_feature_state import utc_timestamp  # noqa: E402


def _resolve(raw: str) -> Path:
    path = Path(raw)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if ROOT not in resolved.parents and resolved != ROOT:
        raise ValueError(f"path escapes repository root: {raw}")
    return resolved


def _load_source(path: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    loaded = joblib.load(path)
    if not isinstance(loaded, Mapping):
        raise ValueError("P8U one-timestamp source state is malformed")
    panel = loaded.get("panel")
    symbols = tuple(sorted(map(str, loaded.get("symbols") or ())))
    if not isinstance(panel, Mapping) or len(symbols) != 160:
        raise ValueError("P8U one-timestamp source lacks a 160-symbol primitive panel")
    return dict(panel), symbols


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--source-state", required=True)
    parser.add_argument("--initial-state", required=True)
    parser.add_argument("--state-scope", required=True)
    parser.add_argument("--state-as-of", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    source_path = _resolve(args.source_state)
    panel, symbols = _load_source(source_path)
    runtime = P8UCanonicalSingleTimestampRuntime(
        P8UCanonicalSingleTimestampConfig(
            root=_resolve(args.out_root),
            bundle_path=_resolve(args.bundle),
            initial_state=_resolve(args.initial_state),
            state_scope=str(args.state_scope),
            state_as_of=utc_timestamp(args.state_as_of),
            symbols=symbols,
        )
    )
    timestamp = utc_timestamp(args.timestamp)
    snapshot = extract_one_timestamp_snapshot(panel, timestamp=timestamp, symbols=symbols)
    receipt = runtime.advance(timestamp=timestamp, snapshot=snapshot)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
