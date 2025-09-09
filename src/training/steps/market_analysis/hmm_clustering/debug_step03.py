#!/usr/bin/env python3
"""Step03 Debugging Utility.

This script provides a single entry-point to run the various Step03 helper
modules (memory manager, fast-fail validator, parallel I/O, intelligent cache,
etc.) in diagnostic mode so that developers can quickly identify problems with
Step03 without needing to manually exercise each helper.

Usage
-----
python -m src.training.steps.market_analysis.hmm_clustering.debug_step03 \
    --symbol ETHUSDT --exchange BINANCE --timeframe 1m --data-dir data_cache

Key Features
------------
1. Runs **memory diagnostics** and prints a concise report
2. Performs **system/resource validation**
3. Exercises **parallel I/O** on an optional test directory
4. Verifies **intelligent caching** round-trip
5. Optionally executes the **optimized Step03** initialisation path
6. Generates a single JSON debug report (`step03_debug_<timestamp>.json`)

All sub-tasks are executed asynchronously where possible so the tool remains
fast even on modest hardware.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Step03 helper modules
from .step03_enhanced_memory_manager import MemoryConfig, get_enhanced_memory_manager
from .step03_fast_fail_validation import ValidationConfig, get_fast_fail_validator
from .step03_parallel_io_operations import IOConfig, get_parallel_io_operations
from .step03_intelligent_caching import CacheConfig, get_intelligent_cache, memoize

# Optional – Optimised Step03 (may not be available in all environments)
try:
    from .step03_enhanced_optimized import OptimizedStep03, OptimizedStep03Config
    OPTIMIZED_AVAILABLE = True
except ImportError:
    OPTIMIZED_AVAILABLE = False

logger = logging.getLogger("step03.debug")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step03 debugging utility")
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--exchange", default="BINANCE")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--data-dir", default="data_cache",
                        help="Location of training/data_cache directory")
    parser.add_argument("--test-io-dir", default=None,
                        help="Directory containing parquet files for I/O tests (optional)")
    parser.add_argument("--skip-optimized", action="store_true",
                        help="Skip OptimizedStep03 initialisation test")
    return parser.parse_args()

async def run_memory_diagnostics() -> Dict[str, Any]:
    logger.info("🧠 Running memory diagnostics …")
    mem_cfg = MemoryConfig(enable_memory_monitoring=True, chunk_size_mb=50)
    mem_mgr = get_enhanced_memory_manager(mem_cfg)
    await mem_mgr.initialize()
    stats = mem_mgr.get_memory_stats().__dict__  # Convert dataclass to dict
    report = {k: float(v) if isinstance(v, (int, float)) else v for k, v in stats.items()}
    await mem_mgr.cleanup()
    logger.info("✅ Memory diagnostics complete")
    return report

async def run_validation(symbol: str, exchange: str, timeframe: str, data_dir: str) -> Dict[str, Any]:
    logger.info("🔍 Running fast-fail validation …")
    val_cfg = ValidationConfig(enable_extensive_logging=False)
    validator = get_fast_fail_validator(val_cfg)
    # Perform a subset of validations that do not require heavy data
    sys_res = await validator.validate_system_resources()
    cfg_ok = await validator.validate_configuration({
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
    })
    summary = validator.get_validation_summary()
    logger.info("✅ Validation complete – %s validations, %.1f%% success",
                summary["total_validations"], summary["success_rate"] * 100)
    return {
        "system_resources": sys_res,
        "config_validation": cfg_ok,
        "summary": summary,
    }

async def run_io_test(test_dir: Path | None) -> Dict[str, Any]:
    if test_dir is None:
        logger.info("⚠️  Skipping parallel I/O test (no --test-io-dir provided)")
        return {"skipped": True}

    logger.info("📁 Running parallel I/O test in %s …", test_dir)
    io_cfg = IOConfig(max_concurrent_files=4, max_workers=2, enable_compression=True)
    io_ops = get_parallel_io_operations(io_cfg)

    parquet_files = list(test_dir.glob("*.parquet"))[:3]
    if not parquet_files:
        logger.warning("No parquet files found – generating synthetic test file …")
        import pandas as pd, numpy as np
        df = pd.DataFrame({
            "id": range(1000),
            "value": np.random.rand(1000),
        })
        sample_file = test_dir / "sample.parquet"
        df.to_parquet(sample_file)
        parquet_files.append(sample_file)

    loaded = await io_ops.load_files_parallel(parquet_files)
    logger.info("Loaded %d parquet files", len(loaded))
    perf = io_ops.get_performance_report()
    await io_ops.cleanup()
    logger.info("✅ Parallel I/O test complete")
    return perf

async def run_cache_test() -> Dict[str, Any]:
    logger.info("💾 Running intelligent cache test …")
    cache_cfg = CacheConfig(max_memory_cache_size_mb=50, cache_ttl_seconds=30)
    cache = get_intelligent_cache(cache_cfg)

    # basic set/get
    cache.set("debug_key", {"ts": time.time()})
    _ = cache.get("debug_key")

    # memoization
    @memoize(ttl_seconds=10, tags=["debug"])
    def add(a: int, b: int) -> int:
        time.sleep(0.05)
        return a + b

    t0 = time.time(); _ = add(1, 2); first = time.time() - t0
    t0 = time.time(); _ = add(1, 2); second = time.time() - t0

    stats = cache.get_stats()
    stats["memoization_speedup"] = first / max(second, 1e-6)
    cache.clear()
    logger.info("✅ Cache test complete (speed-up ×%.1f)", stats["memoization_speedup"])
    return stats

async def run_optimized_initialisation(skip: bool, cfg_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if skip or not OPTIMIZED_AVAILABLE:
        logger.info("🚫 Skipping OptimizedStep03 initialisation test")
        return {"skipped": True, "available": OPTIMIZED_AVAILABLE}

    logger.info("🚀 Testing OptimizedStep03 initialisation …")
    cfg = OptimizedStep03Config(**cfg_kwargs)
    opt = OptimizedStep03(cfg)
    await opt.initialize()
    perf = await opt._generate_performance_report()
    await opt.cleanup()
    logger.info("✅ OptimizedStep03 initialised and cleaned up")
    return perf

async def main_async() -> None:
    args = parse_args()
    start = time.time()

    report: Dict[str, Any] = {
        "symbol": args.symbol,
        "exchange": args.exchange,
        "timeframe": args.timeframe,
        "timestamp": datetime.now().isoformat(),
    }

    # Perform diagnostics in parallel where feasible
    tasks = [
        run_memory_diagnostics(),
        run_validation(args.symbol, args.exchange, args.timeframe, args.data_dir),
        run_io_test(Path(args.test_io_dir) if args.test_io_dir else None),
        run_cache_test(),
        run_optimized_initialisation(
            args.skip_optimized,
            {
                "max_memory_usage_percent": 80.0,
                "chunk_size_mb": 50,
                "enable_memory_monitoring": True,
                "max_concurrent_files": 4,
                "max_workers": 2,
                "enable_compression": True,
                "max_memory_cache_size_mb": 100,
                "max_disk_cache_size_mb": 200,
                "cache_ttl_seconds": 60,
                "min_available_memory_gb": 1.0,
                "min_disk_space_gb": 1.0,
                "enable_extensive_logging": False,
                "enable_performance_monitoring": True,
                "enable_parallel_processing": True,
                "enable_chunked_processing": True,
            },
        ),
    ]

    (memory_report,
     validation_report,
     io_report,
     cache_report,
     optimized_report) = await asyncio.gather(*tasks)

    report.update({
        "memory_report": memory_report,
        "validation_report": validation_report,
        "io_report": io_report,
        "cache_report": cache_report,
        "optimized_initialisation": optimized_report,
        "duration_seconds": time.time() - start,
    })

    # Persist debug report
    out_file = Path("debug_reports") / f"step03_debug_{int(start)}.json"
    out_file.parent.mkdir(exist_ok=True)
    with out_file.open("w") as fp:
        json.dump(report, fp, indent=2)
    logger.info("📝 Debug report written to %s", out_file)

if __name__ == "__main__":
    asyncio.run(main_async())