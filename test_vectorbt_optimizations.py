#!/usr/bin/env python3
"""
Test script for VectorBT optimizations

This script tests the implemented VectorBT optimizations to ensure they work correctly.
"""

import numpy as np
import pandas as pd
import time
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_memory_manager():
    """Test VectorBT memory manager."""
    logger.info("🧪 Testing VectorBT Memory Manager...")
    
    try:
        from src.utils.ml_common.vectorbt_memory_manager import get_memory_manager, optimize_memory_usage
        
        # Test memory manager
        manager = get_memory_manager()
        
        # Test memory allocation
        success = manager.allocate(0.1, "test_allocation", "testing")
        assert success, "Memory allocation should succeed"
        
        # Test memory optimization
        data = np.random.randn(1000, 100).astype(np.float64)
        optimized_data = optimize_memory_usage(data)
        
        # Check if optimization worked
        assert optimized_data.dtype == np.float32, "Data should be optimized to float32"
        
        # Test deallocation
        manager.deallocate("test_allocation")
        
        # Get stats
        stats = manager.get_memory_stats()
        assert 'current_usage_gb' in stats, "Stats should contain usage information"
        
        logger.info("✅ Memory Manager test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory Manager test failed: {e}")
        return False

def test_performance_monitor():
    """Test VectorBT performance monitor."""
    logger.info("🧪 Testing VectorBT Performance Monitor...")
    
    try:
        from src.utils.ml_common.vectorbt_performance_monitor import get_performance_monitor, monitor_operation
        
        # Test performance monitor
        monitor = get_performance_monitor()
        
        # Test operation monitoring
        with monitor_operation("test_operation", metadata={'test': True}):
            time.sleep(0.1)  # Simulate work
        
        # Get performance summary
        summary = monitor.get_performance_summary()
        assert 'total_operations' in summary, "Summary should contain operation count"
        
        logger.info("✅ Performance Monitor test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance Monitor test failed: {e}")
        return False

def test_backtesting_engine():
    """Test optimized VectorBT backtesting engine."""
    logger.info("🧪 Testing VectorBT Backtesting Engine...")
    
    try:
        from src.utils.ml_common.vectorbt_backtesting_engine import VectorBTBacktestingEngine, BacktestMode
        
        # Generate test data
        np.random.seed(42)
        n_periods = 100
        n_assets = 3
        
        prices = np.random.randn(n_periods, n_assets).cumsum(axis=0) + 100
        signals = np.random.choice([-1, 0, 1], size=(n_periods, n_assets), p=[0.1, 0.8, 0.1])
        timestamps = pd.date_range(start='2020-01-01', periods=n_periods, freq='1min')
        
        # Test backtesting engine
        engine = VectorBTBacktestingEngine()
        
        # Run backtest
        results = engine.run_backtest(signals, prices, timestamps, mode=BacktestMode.VECTORBT_CPU)
        
        # Check results
        assert hasattr(results, 'portfolio_values'), "Results should have portfolio values"
        assert hasattr(results, 'performance_metrics'), "Results should have performance metrics"
        assert len(results.portfolio_values) == n_periods, "Portfolio values should match data length"
        
        # Test performance stats
        stats = engine.get_performance_stats()
        assert 'memory_usage_gb' in stats, "Stats should contain memory usage"
        
        logger.info("✅ Backtesting Engine test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Backtesting Engine test failed: {e}")
        return False

def test_portfolio_optimizer():
    """Test optimized VectorBT portfolio optimizer."""
    logger.info("🧪 Testing VectorBT Portfolio Optimizer...")
    
    try:
        from src.utils.ml_common.vectorbt_portfolio_optimization import VectorBTPortfolioOptimizer, OptimizationMethod
        
        # Generate test data
        np.random.seed(42)
        n_periods = 100
        n_assets = 5
        
        returns = np.random.normal(0.001, 0.02, (n_periods, n_assets))
        asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
        
        # Test portfolio optimizer
        optimizer = VectorBTPortfolioOptimizer()
        
        # Run optimization
        results = optimizer.optimize_portfolio(returns, asset_names=asset_names)
        
        # Check results
        assert hasattr(results, 'weights'), "Results should have weights"
        assert hasattr(results, 'expected_return'), "Results should have expected return"
        assert len(results.weights) == n_assets, "Weights should match number of assets"
        assert abs(np.sum(results.weights) - 1.0) < 1e-6, "Weights should sum to 1"
        
        # Test performance stats
        stats = optimizer.get_optimization_stats()
        assert 'total_optimizations' in stats, "Stats should contain optimization count"
        
        logger.info("✅ Portfolio Optimizer test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Portfolio Optimizer test failed: {e}")
        return False

def test_feature_generator():
    """Test optimized VectorBT feature generator."""
    logger.info("🧪 Testing VectorBT Feature Generator...")
    
    try:
        from src.feature_generation.core.vectorbt_feature_generator import VectorBTFeatureGenerator
        from src.feature_generation.core.feature_generator import FeatureConfig, FeatureCategory
        
        # Generate test data
        np.random.seed(42)
        n_periods = 100
        
        data = pd.DataFrame({
            'open': np.random.randn(n_periods).cumsum() + 100,
            'high': np.random.randn(n_periods).cumsum() + 102,
            'low': np.random.randn(n_periods).cumsum() + 98,
            'close': np.random.randn(n_periods).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, n_periods)
        })
        
        # Test feature generator
        config = FeatureConfig(
            name="test_volatility",
            category=FeatureCategory.VOLATILITY,
            description="Test volatility feature",
            required_columns=["close"],
            default_lookback=20
        )
        
        generator = VectorBTFeatureGenerator(config)
        
        # Test batch feature generation
        feature_configs = [
            {
                'name': 'rsi_14',
                'type': 'indicator',
                'params': {'window': 14}
            },
            {
                'name': 'sma_20',
                'type': 'rolling',
                'column': 'close',
                'operation': 'mean',
                'window': 20
            },
            {
                'name': 'close_zscore',
                'type': 'scaling',
                'column': 'close',
                'method': 'zscore'
            }
        ]
        
        features = generator.generate_features_batch_optimized(data, feature_configs)
        
        # Check results
        assert len(features.columns) == len(feature_configs), "Should generate all requested features"
        assert len(features) == n_periods, "Features should match data length"
        
        # Test stats
        stats = generator.get_vectorbt_stats()
        assert 'vectorbt_operations' in stats, "Stats should contain operation count"
        
        logger.info("✅ Feature Generator test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature Generator test failed: {e}")
        return False

def main():
    """Run all optimization tests."""
    logger.info("🚀 Starting VectorBT Optimization Tests...")
    
    tests = [
        test_memory_manager,
        test_performance_monitor,
        test_backtesting_engine,
        test_portfolio_optimizer,
        test_feature_generator
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            logger.error(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All VectorBT optimizations are working correctly!")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Check the logs above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)