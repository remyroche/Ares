# examples/wavelet_caching_workflow.py

from src.core.decorators import handles_errors

"""Complete workflow example for wavelet feature caching and backtesting."
Demonstrates the full pipeline from pre-computation to fast backtesting.
"""
import asyncio
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.training.steps.backtesting_with_cached_features import (
    BacktestingWithCachedFeatures,
)
from src.training.steps.precompute_wavelet_features import WaveletFeaturePrecomputer
from src.utils.data_optimizer import ohlcv_columns

from src.utils.logger import system_logger

@handles_errors(
    exceptions=(ValueError, RuntimeError, FileNotFoundError),
    default_return={},
    context="configuration loading",
)
async def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception:
        return {}

@handles_errors(
    exceptions=(ValueError, RuntimeError),
    default_return=pd.DataFrame(),
    context="sample data creation",
)
async def create_sample_data() -> pd.DataFrame:
    """Create sample price data for demonstration."""
    try:
        # Create sample OHLCV data
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="1min")
        n_points = len(dates)

        # Generate realistic price data
        np.random.seed(42)
        base_price = 1000
        returns = np.random.normal(0, 0.001, n_points)
        prices = base_price * np.exp(np.cumsum(returns))

        # Add some volatility clustering
        volatility = np.random.gamma(2, 0.001, n_points)
        prices = prices * (1 + np.random.normal(0, volatility))

        # Create OHLCV data
        data = pd.DataFrame(
            {
                "open": prices * (1 + np.random.normal(0, 0.0005, n_points)),
                "high": prices * (1 + np.abs(np.random.normal(0, 0.001, n_points))),
                "low": prices * (1 - np.abs(np.random.normal(0, 0.001, n_points))),
                "close": prices,
                "volume": np.random.uniform(1000, 10000, n_points),
            },
            index=dates,
        )

        # Ensure OHLC relationships
        data["high"] = data[["open", "high", "close"]].max(axis=1)
        data["low"] = data[["open", "low", "close"]].min(axis=1)

        return data

    except Exception:
        return pd.DataFrame()

@handles_errors(
    exceptions=(ValueError, RuntimeError, FileNotFoundError),
    default_return=False,
    context="feature precomputation",
)
async def step1_precompute_features(config: dict) -> bool | None:
    """Step 1: Pre-compute wavelet features for the entire dataset."""
    try:
        logger = system_logger.getChild("WaveletWorkflow")
        # Initialize pre-computer
        precomputer = WaveletFeaturePrecomputer(config)
        await precomputer.initialize()

        # Create sample data
        sample_data = await create_sample_data()

        if sample_data.empty:
            logger.error("Sample data generation failed")
            return False

        # Save sample data
        data_dir = Path("data/price_data")
        data_dir.mkdir(parents=True, exist_ok=True)

        sample_data.to_parquet("data/price_data/sample_data.parquet")

        # Pre-compute features
        start_time = time.time()

        success = await precomputer.precompute_dataset(
            data_path="data/price_data/sample_data.parquet",
            symbol="SAMPLE",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        processing_time = time.time() - start_time
        logger.info(f"Precomputation finished in {processing_time:.2f}s, success={success}")

        if success:
            # Print cache statistics
            stats = precomputer.get_precomputation_stats()
            logger.info(f"Precomputation stats: {stats}")
            return True
        return False

    except Exception:
        return False

async def step2_run_backtests(config: dict) -> bool | None:
    """Step 2: Run backtests using cached features."""
    try:
        logger = system_logger.getChild("WaveletWorkflow")
        # Initialize backtesting system
        backtester = BacktestingWithCachedFeatures(config)
        await backtester.initialize()

        # Load sample data (project OHLCV)
        try:
            _ = pd.read_parquet(
                "data/price_data/sample_data.parquet", columns=ohlcv_columns(),
            )
        except Exception:
            _ = pd.read_parquet("data/price_data/sample_data.parquet")

        # Create multiple backtest configurations
        backtest_configs = [
            {
                "data_path": "data/price_data/sample_data.parquet",
                "strategy_config": {
                    "strategy_type": "wavelet_energy",
                    "parameters": {"energy_threshold": 0.5},
                },
            },
            {
                "data_path": "data/price_data/sample_data.parquet",
                "strategy_config": {
                    "strategy_type": "wavelet_entropy",
                    "parameters": {"entropy_threshold": 0.3},
                },
            },
        ]

        start_time = time.time()

        # Run backtests
        results = await backtester.run_multiple_backtests(backtest_configs)

        _ = time.time() - start_time

        if results:
            # Print results
            for _i, result in enumerate(results):
                logger.info(f"Backtest result summary: {result.get('summary', {})}")

            # Print performance statistics
            perf_stats = backtester.get_performance_stats()
            logger.info(f"Performance stats: {perf_stats}")

            return True
        return False

    except Exception:
        return False

async def step3_performance_comparison(config: dict) -> bool | None:
    """Step 3: Compare performance with and without caching."""
    try:
        logger = system_logger.getChild("WaveletWorkflow")
        # Load sample data (project OHLCV)
        try:
            price_data = pd.read_parquet(
                "data/price_data/sample_data.parquet",
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
        except Exception:
            price_data = pd.read_parquet("data/price_data/sample_data.parquet")

        # Test 1: With caching (should be fast)
        backtester_cached = BacktestingWithCachedFeatures(config)
        await backtester_cached.initialize()

        start_time = time.time()
        await backtester_cached.run_backtest(price_data)
        cached_time = time.time() - start_time

        # Test 2: Without caching (should be slower)
        config_no_cache = config.copy()
        config_no_cache.setdefault("wavelet_cache", {})
        config_no_cache["wavelet_cache"]["cache_enabled"] = False

        backtester_no_cache = BacktestingWithCachedFeatures(config_no_cache)
        await backtester_no_cache.initialize()

        start_time = time.time()
        await backtester_no_cache.run_backtest(price_data)
        no_cache_time = time.time() - start_time

        # Print comparison
        logger.info(
            f"Backtest time with cache: {cached_time:.2f}s vs without cache: {no_cache_time:.2f}s",
        )

        if cached_time < no_cache_time:
            logger.info("Caching provided a speedup as expected.")
        else:
            logger.warning("Caching did not provide expected speedup in this run.")

        return True

    except Exception:
        return False

async def step4_cache_management(config: dict) -> bool | None:
    """Step 4: Demonstrate cache management features."""
    try:
        logger = system_logger.getChild("WaveletWorkflow")
        # Initialize cache management
import copy
        from src.training.steps.vectorized_advanced_feature_engineering import (
    except Exception as e:
        pass  # TODO: Handle exception properly

WaveletFeatureCache,
)

        cache = WaveletFeatureCache(config)

        # Get cache statistics
        stats = cache.get_cache_stats()
        logger.info(f"Cache stats: {stats}")

        # Demonstrate cache clearing (optional)
        # cache.clear_cache()  # Uncomment to clear cache

        return True

    except Exception:
        return False

async def main() -> None:
    """Main workflow function."""
    try:
        logger = system_logger.getChild("WaveletWorkflow")
        # Load configuration
        config_path = "config/wavelet_caching_config.yaml"
        if not Path(config_path).exists():
            config = {
                "wavelet_cache": {
                    "cache_enabled": True,
                    "cache_dir": "data/wavelet_cache",
                    "cache_format": "parquet",
                    "compression": "snappy",
                    "cache_expiry_days": 30,
                },
                "wavelet_precompute": {
                    "enable_batch_processing": True,
                    "batch_size": 10000,
                    "enable_progress_tracking": True,
                },
                "backtesting_with_cache": {
                    "enable_feature_caching": True,
                    "enable_performance_monitoring": True,
                },
                "vectorized_advanced_features": {
                    "enable_wavelet_transforms": True,
                },
            }
        else:
            config = await load_config(config_path)

        # Step 1: Pre-compute features
        step1_success = await step1_precompute_features(config)
        if not step1_success:
            logger.error("Step 1 failed. Aborting workflow.")
            return

        # Step 2: Run backtests
        step2_success = await step2_run_backtests(config)
        if not step2_success:
            logger.error("Step 2 failed. Aborting workflow.")
            return

        # Step 3: Performance comparison
        step3_success = await step3_performance_comparison(config)
        if not step3_success:
            logger.error("Step 3 failed. Aborting workflow.")
            return

        # Step 4: Cache management
        step4_success = await step4_cache_management(config)
        if not step4_success:
            logger.error("Step 4 failed.")
            return

        # Summary
        logger.info("Wavelet caching workflow completed successfully.")

    except Exception:
        system_logger.getChild("WaveletWorkflow").exception("Workflow failed")

if __name__ == "__main__":
    asyncio.run(main())
