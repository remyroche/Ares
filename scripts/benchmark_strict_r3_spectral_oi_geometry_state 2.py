#!/usr/bin/env python3
"""Deterministic microbenchmark for the frozen spectral/OI state update."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.spectral_oi_geometry_state import (  # noqa: E402
    SpectralOiGeometryState,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=168, help="bootstrap rows")
    parser.add_argument("--steady-updates", type=int, default=24)
    parser.add_argument("--symbols", type=int, default=170)
    parser.add_argument("--spectral-fields", type=int, default=64)
    parser.add_argument("--oi-fields", type=int, default=13)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if min(args.rows, args.steady_updates, args.symbols, args.spectral_fields, args.oi_fields) <= 0:
        raise ValueError("all benchmark dimensions must be positive")
    rng = np.random.default_rng(args.seed)
    state = SpectralOiGeometryState(
        symbols=[f"S{value:03d}" for value in range(args.symbols)],
        spectral_source_columns=[f"source_{value}" for value in range(args.spectral_fields)],
        oi_parent_columns=[f"oi_{value}" for value in range(args.oi_fields)],
        spectral_definition_id="benchmark-frozen-spectral-v1",
        oi_geometry_definition_id="benchmark-frozen-oi-v1",
    )
    timestamps = pd.date_range("2026-01-01", periods=args.rows + args.steady_updates, freq="h", tz="UTC")
    source = rng.normal(size=(len(timestamps), args.spectral_fields)).astype(np.float32)
    oi = rng.normal(size=(len(timestamps), args.oi_fields, args.symbols)).astype(np.float32)
    bootstrap_started = time.perf_counter()
    # This is the actual cold bootstrap timing.  The subsequent loop is the
    # live-relevant steady-state timing after that state has been persisted.
    state.bootstrap(
        timestamps=timestamps[:args.rows],
        spectral_source=source[:args.rows],
        oi_parents=oi[:args.rows],
    )
    bootstrap_seconds = time.perf_counter() - bootstrap_started
    started = time.perf_counter()
    for row in range(args.rows, len(timestamps)):
        state.update(timestamp=timestamps[row], spectral_source=source[row], oi_parents=oi[row])
    elapsed = time.perf_counter() - started
    print(json.dumps({
        "rows": args.rows,
        "symbols": args.symbols,
        "spectral_fields": args.spectral_fields,
        "oi_fields": args.oi_fields,
        "bootstrap_rows": args.rows,
        "bootstrap_seconds_including_warmup": bootstrap_seconds,
        "steady_updates": args.steady_updates,
        "steady_seconds": elapsed,
        "steady_update_seconds": float(elapsed / max(args.steady_updates, 1)),
        "steady_updates_per_second": float(args.steady_updates / max(elapsed, 1e-12)),
        "state_contract_hash": state.contract_hash,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
