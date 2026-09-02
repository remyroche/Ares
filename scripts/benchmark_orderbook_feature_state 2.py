#!/usr/bin/env python3
"""Benchmark the bounded strict-R3 order-book feature state.

This is deliberately synthetic and side-effect free: it measures the state
operator itself, not source I/O or the broad feature graph.  A production
promotion must separately establish full feature-contract parity.
"""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.orderbook_feature_state import (
    OrderbookFeatureState,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", type=int, default=170)
    parser.add_argument("--hours", type=int, default=4_800)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if args.symbols <= 0 or args.hours <= 0:
        raise SystemExit("--symbols and --hours must be positive")
    rng = np.random.default_rng(args.seed)
    n, m = int(args.hours), int(args.symbols)
    names = [f"S{position:03d}/USD:USD" for position in range(m)]
    close = (100.0 + rng.normal(0.0, 0.25, (n, m)).cumsum(axis=0)).astype(np.float32)
    bid = (close * (1.0 - rng.uniform(3e-5, 2e-4, (n, m)))).astype(np.float32)
    ask = (close * (1.0 + rng.uniform(3e-5, 2e-4, (n, m)))).astype(np.float32)
    values = {
        "best_bid": bid,
        "best_ask": ask,
        "mid": ((bid + ask) * 0.5).astype(np.float32),
        "bid_qty_1": rng.lognormal(3.0, 0.5, (n, m)).astype(np.float32),
        "ask_qty_1": rng.lognormal(3.0, 0.5, (n, m)).astype(np.float32),
        "cum_bid_qty_l20": rng.lognormal(5.0, 0.5, (n, m)).astype(np.float32),
        "cum_ask_qty_l20": rng.lognormal(5.0, 0.5, (n, m)).astype(np.float32),
        "mean_trade_qty_1h": rng.lognormal(1.0, 0.5, (n, m)).astype(np.float32),
        "close": close,
        "volume": rng.lognormal(7.0, 0.8, (n, m)).astype(np.float32),
        "source_valid": rng.random((n, m)) > 0.03,
    }
    timestamps = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    state = OrderbookFeatureState(symbols=names)
    started = time.perf_counter()
    output = state.update_batch(values, timestamps=timestamps)
    elapsed = time.perf_counter() - started
    # A second one-row pass is the operational live cost after warm-up.
    future = {key: value[-1] for key, value in values.items()}
    next_timestamp = timestamps[-1] + pd.Timedelta(hours=1)
    started = time.perf_counter()
    state.update(future, timestamp=next_timestamp)
    live_seconds = time.perf_counter() - started
    finite = sum(np.isfinite(value).sum() for value in output.values())
    print(
        "orderbook_feature_state_benchmark"
        f" symbols={m} hours={n} batch_seconds={elapsed:.6f}"
        f" rows_per_second={n / max(elapsed, 1e-12):.1f}"
        f" live_one_row_seconds={live_seconds:.6f} finite_outputs={finite}"
    )


if __name__ == "__main__":
    main()
