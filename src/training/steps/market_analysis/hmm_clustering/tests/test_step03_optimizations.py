#!/usr/bin/env python3
"""Test script for Step03 optimization integration.

This script tests the integration of all Step03 optimizations:
- Bayesian Optimization Acceleration
- Memory & I/O Optimization
- Ensemble Clustering Efficiency
- Vectorization & NumPy Optimization
- Pipeline Orchestration
"""

import asyncio
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.steps.market_analysis.hmm_clustering.step03_hmm_regime_discovery import run_step
from src.utils.logger import system_logger
import logging

logger = system_logger.getChild('Step03OptimizationTest')


async def test_step03_optimizations():
    """Test Step03 with optimizations enabled."""

    logger.info("🚀 Testing Step03 optimizations integration")
    logger.info("=" * 80)

    # Test parameters
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    timeframe = "1m"
    data_dir = "data_cache"

    # Test 1: Standard pipeline (baseline)
    logger.info("📊 Test 1: Standard pipeline (baseline)")
    try:
        result_standard = await run_step(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            use_optimized_pipeline=False,
            force_rerun=False
        )
        logger.info(f"✅ Standard pipeline result: {result_standard}")
    except Exception as e:
        logger.error(f"❌ Standard pipeline failed: {e}")

    # Test 2: Optimized pipeline
    logger.info("🚀 Test 2: Optimized pipeline")
    try:
        result_optimized = await run_step(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            use_optimized_pipeline=True,
            force_optimized_pipeline=True,
            force_rerun=False
        )
        logger.info(f"✅ Optimized pipeline result: {result_optimized}")
    except Exception as e:
        logger.error(f"❌ Optimized pipeline failed: {e}")

    # Test 3: Force optimized pipeline (even if components unavailable)
    logger.info("🎯 Test 3: Force optimized pipeline")
    try:
        result_force = await run_step(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            use_optimized_pipeline=True,
            force_optimized_pipeline=True,
            force_rerun=False
        )
        logger.info(f"✅ Force optimized pipeline result: {result_force}")
    except Exception as e:
        logger.error(f"❌ Force optimized pipeline failed: {e}")

    logger.info("✅ Step03 optimization tests completed")


def test_component_imports():
    """Test that all optimized components can be imported."""

    logger.info("🔍 Testing optimized component imports")
    logger.info("=" * 50)

    components = [
        ("Enhanced Bayesian Optimization", "step03_enhanced_bayesian_optimization"),
        ("Memory Manager", "step03_memory_manager"),
        ("Advanced Ensemble Clustering", "step03_advanced_ensemble_clustering"),
        ("Vectorized Operations", "step03_vectorized_operations"),
        ("Pipeline Orchestrator", "step03_pipeline_orchestrator")
    ]

    for component_name, module_name in components:
        try:
            module_path = f"src.training.steps.market_analysis.hmm_clustering.{module_name}"
            __import__(module_path)
            logger.info(f"✅ {component_name}: Import successful")
        except ImportError as e:
            logger.warning(f"⚠️ {component_name}: Import failed - {e}")
        except Exception as e:
            logger.error(f"❌ {component_name}: Import error - {e}")

    logger.info("✅ Component import tests completed")


def test_vectorized_operations():
    """Test vectorized operations independently."""

    logger.info("⚡ Testing vectorized operations")
    logger.info("=" * 40)

    try:
        from src.training.steps.market_analysis.hmm_clustering.step03_vectorized_operations import (
            get_vectorized_operations_manager,
            create_vectorized_config
        )

        # Create test data
        np.random.seed(42)
        test_data = pd.DataFrame({
            'close': np.random.randn(1000).cumsum() + 100,
            'high': np.random.randn(1000).cumsum() + 102,
            'low': np.random.randn(1000).cumsum() + 98,
            'volume': np.random.randint(1000, 10000, 1000)
        })

        # Test vectorized operations
        manager = get_vectorized_operations_manager()
        config = create_vectorized_config()

        processed_data = manager.process_dataset(test_data, config)

        logger.info(f"✅ Vectorized operations successful")
        logger.info(f"   Original shape: {test_data.shape}")
        logger.info(f"   Processed shape: {processed_data.shape}")
        logger.info(f"   New features: {len(processed_data.columns) - len(test_data.columns)}")

    except Exception as e:
        logger.error(f"❌ Vectorized operations test failed: {e}")

    logger.info("✅ Vectorized operations test completed")


def test_memory_manager():
    """Test memory manager independently."""

    logger.info("💾 Testing memory manager")
    logger.info("=" * 30)

    try:
        from src.training.steps.market_analysis.hmm_clustering.step03_memory_manager import (
            get_memory_manager
        )

        # Test memory manager
        config = {'memory_limit_gb': 2.0}
        memory_manager = get_memory_manager(config)

        logger.info("✅ Memory manager initialization successful")

        # Test DataFrame optimization
        test_df = pd.DataFrame({
            'int_col': np.random.randint(0, 100, 1000),
            'float_col': np.random.randn(1000),
            'str_col': ['category_' + str(i % 10) for i in range(1000)]
        })

        optimized_df = memory_manager.optimize_dataframe_memory(test_df)

        original_memory = test_df.memory_usage(deep=True).sum()
        optimized_memory = optimized_df.memory_usage(deep=True).sum()

        logger.info("✅ DataFrame memory optimization successful")
        logger.info(f"   Original memory: {original_memory / 1024:.1f} KB")
        logger.info(f"   Optimized memory: {optimized_memory / 1024:.1f} KB")
        logger.info(".1f")

    except Exception as e:
        logger.error(f"❌ Memory manager test failed: {e}")

    logger.info("✅ Memory manager test completed")


def main():
    """Main test function."""

    logger.info("🧪 Starting Step03 Optimization Integration Tests")
    logger.info("=" * 80)

    # Test component imports
    test_component_imports()
    logger.info("")

    # Test individual components
    test_vectorized_operations()
    logger.info("")

    test_memory_manager()
    logger.info("")

    # Test full pipeline (async)
    try:
        asyncio.run(test_step03_optimizations())
    except Exception as e:
        logger.error(f"❌ Full pipeline test failed: {e}")

    logger.info("=" * 80)
    logger.info("🏁 Step03 Optimization Integration Tests Completed")


if __name__ == "__main__":
    main()
