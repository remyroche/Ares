#!/usr/bin/env python3
"""debug_step02_5.py
Comprehensive debugging and profiling tool for step02_5 S/R Optimization step.

This script provides a flexible CLI to:
1. Generate synthetic OHLCV datasets of arbitrary size
2. Execute `SROptimizationStep` with optional chunked processing
3. Collect detailed runtime metrics (execution time, memory usage, peak memory)
4. Capture and persist logs to a user-specified file
5. Optionally run built-in unit tests (`test_step02_5_*`) to quickly verify behaviour
6. Provide a JSON summary of results for downstream analysis

Usage examples
--------------
# Quick sanity run with default 10k rows
python debug_step02_5.py

# 100k rows, enable chunked processing, save logs
python debug_step02_5.py --rows 100000 --chunked --log-file step02_5_debug.log

# Run unit tests only
python debug_step02_5.py --run-tests
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import psutil
import tracemalloc

# ---------------------------------------------------------------------------
# Dynamic import path manipulation so that the script works standalone
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Lazy import heavy packages after CLI parsing to keep startup fast

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_memory_usage_mb() -> float:
    """Return current RSS memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def _create_synthetic_data(rows: int) -> pd.DataFrame:
    """Generate a realistic synthetic OHLCV DataFrame."""
    np.random.seed(42)
    timestamps = pd.date_range("2023-01-01", periods=rows, freq="1min")
    base_price = 30000
    price_changes = np.random.randn(rows) * 0.001  # ~0.1% volatility
    prices = base_price * (1 + np.cumsum(price_changes))
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices * (1 + np.random.randn(rows) * 0.0005),
            "high": prices * (1 + np.random.randn(rows) * 0.001).clip(lower=prices),
            "low": prices * (1 - np.random.randn(rows) * 0.001).clip(upper=prices),
            "close": prices,
            "volume": np.random.randint(1_000, 100_000, rows),
        }
    )
    # Ensure high >= max(open, close) and low <= min(open, close)
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    return df


def _configure_logger(log_file: str | None, verbose: bool) -> None:
    """Set up root logger with optional file handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        fh.setFormatter(fmt)
        logging.getLogger().addHandler(fh)
        logging.info("🗒️ Log file output enabled: %s", log_file)


async def _run_step(rows: int, chunked: bool) -> Dict[str, Any]:
    """Run SROptimizationStep and collect metrics."""
    from src.training.steps.data_collection.data_preparation.step02_5_sr_optimization import (
        SROptimizationStep,
    )

    # Synthetic data generation
    data = _create_synthetic_data(rows)

    # Dynamic config based on CLI args
    config: Dict[str, Any] = {
        "sr_optimization": {
            "min_touches": 2,
            "tolerance_pct": 0.5,
            "lookback_periods": 100,
            "use_chunked_processing": chunked,
        }
    }

    step = SROptimizationStep(config)
    await step.initialize()

    training_input = {"validated_data": data}
    pipeline_state = {"dataframe": data}

    # Timing & memory tracking
    initial_mem = _get_memory_usage_mb()
    tracemalloc.start()
    start_time = time.perf_counter()

    try:
        result: Dict[str, Any] = await step.execute(training_input, pipeline_state)
        success = result.get("success", False)
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("❌ Exception during step execution: %s", exc)
        tracemalloc.stop()
        raise

    exec_time = time.perf_counter() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    final_mem = _get_memory_usage_mb()

    metrics = {
        "rows": rows,
        "chunked": chunked,
        "execution_time_sec": round(exec_time, 3),
        "initial_memory_mb": round(initial_mem, 3),
        "final_memory_mb": round(final_mem, 3),
        "peak_memory_mb": round(peak_mem / 1024 / 1024, 3),
        "memory_delta_mb": round(final_mem - initial_mem, 3),
        "success": success,
        "sr_levels": len(result.get("sr_levels", {}).get("support_levels", []))
        + len(result.get("sr_levels", {}).get("resistance_levels", [])),
    }
    logging.info("🏁 Metrics: %s", json.dumps(metrics, indent=2))
    return metrics


def _run_tests() -> None:
    """Run pytest programmatically for step02_5 related tests."""
    import pytest

    logging.info("🔍 Running unit tests for step02_5…")
    tests_dir = PROJECT_ROOT
    # Select only tests containing 'step02_5' in filename
    exit_code = pytest.main([str(tests_dir), "-k", "step02_5"])
    if exit_code == 0:
        logging.info("✅ All step02_5 tests passed")
    else:
        logging.error("❌ Some tests failed (exit code=%s)", exit_code)
        sys.exit(exit_code)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901 (main entry complexity is fine)
    parser = argparse.ArgumentParser(description="Debug & profile step02_5")
    parser.add_argument("--rows", type=int, default=10_000, help="Number of rows to generate")
    parser.add_argument("--chunked", action="store_true", help="Enable chunked processing")
    parser.add_argument(
        "--log-file", dest="log_file", help="Optional path to persist logs"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--run-tests", action="store_true", help="Run unit tests instead of executing step"
    )
    parser.add_argument(
        "--json-out",
        dest="json_out",
        help="Write metrics summary to specified JSON file",
    )

    args = parser.parse_args()

    _configure_logger(args.log_file, args.verbose)

    if args.run_tests:
        _run_tests()
        return

    logging.info(
        "🚀 Executing step02_5 debug run | rows=%s | chunked=%s", args.rows, args.chunked
    )

    try:
        metrics = asyncio.run(_run_step(args.rows, args.chunked))
    except KeyboardInterrupt:
        logging.warning("⚠️ Interrupted by user")
        sys.exit(1)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        logging.info("📄 Metrics written to %s", args.json_out)


if __name__ == "__main__":
    main()