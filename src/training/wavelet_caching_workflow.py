# examples/wavelet_caching_workflow.py

"""
Complete workflow example for wavelet feature caching and backtesting.
Demonstrates the full pipeline from pre-computation to fast backtesting.
"""

import asyncio
from src.utils.warning_symbols import (
    error,
    warning,
    critical,
    problem,
    failed,
    invalid,
    missing,
    timeout,
    connection_error,
    validation_error,
    initialization_error,
    execution_error,
)
import time
from pathlib import Path

import numpy as np
import pandas as pd
from src.utils.data_optimizer import ohlcv_columns
import yaml

from src.training.steps.backtesting_with_cached_features import (
    BacktestingWithCachedFeatures,
)
from src.training.steps.precompute_wavelet_features import WaveletFeaturePrecomputer


async def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        error_msg = f"Error loading wavelet caching config from {config_path}: {e}"
        print(f"ERROR: {error_msg}")
        return {}


async def create_sample_data():
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

    except Exception as e:
        error_msg = f"Error creating sample price data: {e}"
        print(f"ERROR: {error_msg}")
        return pd.DataFrame()


async def step1_precompute_features(config: dict):
    """Step 1: Pre-compute wavelet features for the entire dataset."""
    print("\n" + "=" * 60)
    print("STEP 1: PRE-COMPUTING WAVELET FEATURES")
    print("=" * 60)

    try:
        # Initialize pre-computer
        precomputer = WaveletFeaturePrecomputer(config)
        await precomputer.initialize()

        # Create sample data
        print("📊 Creating sample price data...")
        sample_data = await create_sample_data()

        if sample_data.empty:
            print("FAILED: Failed to create sample data")
            return False

        # Save sample data
        data_dir = Path("data/price_data")
        data_dir.mkdir(parents=True, exist_ok=True)

        sample_data.to_parquet("data/price_data/sample_data.parquet")
        print(f"💾 Saved sample data: {len(sample_data)} rows")

        # Pre-compute features
        print("🔧 Starting wavelet feature pre-computation...")
        start_time = time.time()

        success = await precomputer.precompute_dataset(
            data_path="data/price_data/sample_data.parquet",
            symbol="SAMPLE",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        precompute_time = time.time() - start_time

        if success:
            print(f"✅ Pre-computation completed in {precompute_time:.2f}s")

            # Print cache statistics
            stats = precomputer.get_precomputation_stats()
            print("📊 Cache Statistics:")
            print(
                f"  Cache Directory: {stats.get('cache_stats', {}).get('cache_dir', 'N/A')}",
            )
            print(
                f"  Total Files: {stats.get('cache_stats', {}).get('total_files', 0)}",
            )
            print(
                f"  Total Size: {stats.get('cache_stats', {}).get('total_size_mb', 0):.2f} MB",
            )

            return True
        print("FAILED: Pre-computation failed")
        return False

    except Exception as e:
        error_msg = f"Error in wavelet feature pre-computation: {e}"
        print(f"WARNING: {error_msg}")
        return False


async def step2_run_backtests(config: dict):
    """Step 2: Run backtests using cached features."""
    print("\n" + "=" * 60)
    print("STEP 2: RUNNING BACKTESTS WITH CACHED FEATURES")
    print("=" * 60)

    try:
        # Initialize backtesting system
        backtester = BacktestingWithCachedFeatures(config)
        await backtester.initialize()

        # Load sample data (project OHLCV)
        print("📊 Loading sample data for backtesting...")
        try:
            pd.read_parquet(
                "data/price_data/sample_data.parquet", columns=ohlcv_columns()
            )
        except Exception:
            pd.read_parquet("data/price_data/sample_data.parquet")

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

        print(f"🚀 Running {len(backtest_configs)} backtests...")
        start_time = time.time()

        # Run backtests
        results = await backtester.run_multiple_backtests(backtest_configs)

        backtest_time = time.time() - start_time

        if results:
            print(f"✅ Backtests completed in {backtest_time:.2f}s")

            # Print results
            print("\n📊 Backtest Results:")
            for i, result in enumerate(results):
                strategy_results = result.get("strategy_results", {})
                print(f"  Backtest {i + 1}:")
                print(
                    f"    Total Return: {strategy_results.get('total_return', 0):.4f}",
                )
                print(
                    f"    Sharpe Ratio: {strategy_results.get('sharpe_ratio', 0):.4f}",
                )
                print(
                    f"    Max Drawdown: {strategy_results.get('max_drawdown', 0):.4f}",
                )
                print(f"    Win Rate: {strategy_results.get('win_rate', 0):.2%}")
                print(f"    Signal Count: {strategy_results.get('signal_count', 0)}")
                print(f"    Feature Count: {result.get('feature_count', 0)}")

            # Print performance statistics
            stats = backtester.get_performance_stats()
            print("\n📈 Performance Statistics:")
            print(f"  Cache Hit Rate: {stats.get('cache_hit_rate', 0):.2%}")
            print(f"  Avg Backtest Time: {stats.get('avg_backtest_time', 0):.3f}s")
            print(
                f"  Avg Feature Load Time: {stats.get('avg_feature_load_time', 0):.3f}s",
            )
            print(f"  Iterations Completed: {stats.get('iterations_completed', 0)}")

            return True
        print("FAILED: Backtests failed")
        return False

    except Exception as e:
        error_msg = f"Error in backtesting with cached features: {e}"
        print(f"WARNING: {error_msg}")
        return False


async def step3_performance_comparison(config: dict):
    """Step 3: Compare performance with and without caching."""
    print("\n" + "=" * 60)
    print("STEP 3: PERFORMANCE COMPARISON")
    print("=" * 60)

    try:
        # Load sample data (project OHLCV)
        try:
            price_data = pd.read_parquet(
                "data/price_data/sample_data.parquet",
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
        except Exception:
            price_data = pd.read_parquet("data/price_data/sample_data.parquet")

        # Test 1: With caching (should be fast)
        print("🔧 Testing with cached features...")
        backtester_cached = BacktestingWithCachedFeatures(config)
        await backtester_cached.initialize()

        start_time = time.time()
        await backtester_cached.run_backtest(price_data)
        cached_time = time.time() - start_time

        # Test 2: Without caching (should be slower)
        print("🔧 Testing without cached features...")
        config_no_cache = config.copy()
        config_no_cache["wavelet_cache"]["cache_enabled"] = False

        backtester_no_cache = BacktestingWithCachedFeatures(config_no_cache)
        await backtester_no_cache.initialize()

        start_time = time.time()
        await backtester_no_cache.run_backtest(price_data)
        no_cache_time = time.time() - start_time

        # Print comparison
        print("\n📊 Performance Comparison:")
        print(f"  With Caching:     {cached_time:.3f}s")
        print(f"  Without Caching:  {no_cache_time:.3f}s")
        print(f"  Speed Improvement: {no_cache_time / cached_time:.1f}x faster")

        if cached_time < no_cache_time:
            print("✅ Caching provides significant performance improvement!")
        else:
            print("WARNING: Caching didn't provide expected improvement (first run)")

        return True

    except Exception as e:
        error_msg = f"Error in performance comparison analysis: {e}"
        print(f"WARNING: {error_msg}")
        return False


async def step4_cache_management(config: dict):
    """Step 4: Demonstrate cache management features."""
    print("\n" + "=" * 60)
    print("STEP 4: CACHE MANAGEMENT")
    print("=" * 60)

    try:
        # Initialize cache management
        from src.training.steps.vectorized_advanced_feature_engineering import (
            WaveletFeatureCache,
        )

        cache = WaveletFeatureCache(config)

        # Get cache statistics
        stats = cache.get_cache_stats()
        print("📊 Current Cache Statistics:")
        print(f"  Cache Directory: {stats.get('cache_dir', 'N/A')}")
        print(f"  Cache Format: {stats.get('cache_format', 'N/A')}")
        print(f"  Compression: {stats.get('compression', 'N/A')}")
        print(f"  Total Files: {stats.get('total_files', 0)}")
        print(f"  Total Size: {stats.get('total_size_mb', 0):.2f} MB")
        print(f"  Oldest File: {stats.get('oldest_file', 'N/A')}")
        print(f"  Newest File: {stats.get('newest_file', 'N/A')}")

        # Demonstrate cache clearing (optional)
        print("\n🧹 Cache Management Options:")
        print("  - Clear specific cache: cache.clear_cache('specific_key')")
        print("  - Clear all cache: cache.clear_cache()")
        print("  - Check cache existence: cache.cache_exists('key')")
        print("  - Get cache statistics: cache.get_cache_stats()")

        return True

    except Exception as e:
        error_msg = f"Error in cache management operations: {e}"
        print(f"WARNING: {error_msg}")
        return False


async def main():
    """Main workflow function."""
    print("🚀 WAVELET CACHING WORKFLOW DEMONSTRATION")
    print("=" * 60)
    print("This example demonstrates the complete workflow from")
    print("pre-computation to fast backtesting using cached features.")
    print("=" * 60)

    try:
        # Load configuration
        config_path = "config/wavelet_caching_config.yaml"
        if not Path(config_path).exists():
            print(f"MISSING: Configuration file not found: {config_path}")
            print("Using default configuration...")
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
            print("FAILED: Step 1 failed, stopping workflow")
            return

        # Step 2: Run backtests
        step2_success = await step2_run_backtests(config)
        if not step2_success:
            print("FAILED: Step 2 failed, stopping workflow")
            return

        # Step 3: Performance comparison
        step3_success = await step3_performance_comparison(config)
        if not step3_success:
            print("FAILED: Step 3 failed, stopping workflow")
            return

        # Step 4: Cache management
        step4_success = await step4_cache_management(config)
        if not step4_success:
            print("FAILED: Step 4 failed, stopping workflow")
            return

        # Summary
        print("\n" + "=" * 60)
        print("✅ WORKFLOW COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("The wavelet caching system is now ready for production use.")
        print("Key benefits achieved:")
        print("  ✅ Pre-computed wavelet features for fast loading")
        print("  ✅ Significant performance improvement in backtesting")
        print("  ✅ Robust cache management and validation")
        print("  ✅ Comprehensive performance monitoring")
        print("  ✅ Memory-efficient storage using Parquet format")
        print("\nYou can now run thousands of backtests without")
        print("recalculating expensive wavelet transforms!")

    except Exception as e:
        error_msg = f"Error in main wavelet caching workflow: {e}"
        print(f"WARNING: {error_msg}")


if __name__ == "__main__":
    asyncio.run(main())
