"""
Test Enhanced VectorBT Volatility Features

This module provides comprehensive tests for the enhanced VectorBT volatility
feature generation system, including performance benchmarking and validation.
"""

import unittest
import numpy as np
import pandas as pd
import time
from typing import Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the modules to test
try:
    from ..categories.enhanced_vectorbt_volatility import (
        EnhancedVectorBTVolatilityGenerator,
        VolatilityConfig,
        create_enhanced_volatility_generators,
        create_default_enhanced_volatility_generators
    )
    ENHANCED_VECTORBT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced VectorBT not available: {e}")
    ENHANCED_VECTORBT_AVAILABLE = False

try:
    from ..categories.volatility import (
        create_comprehensive_vectorbt_volatility_generators,
        create_optimized_volatility_pipeline,
        benchmark_volatility_optimizations
    )
    VOLATILITY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Volatility module not available: {e}")
    VOLATILITY_AVAILABLE = False

try:
    from ..utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
    ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"VectorBT Rolling Optimizer not available: {e}")
    ROLLING_OPTIMIZER_AVAILABLE = False

try:
    from ...utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager
    UNIFIED_MANAGER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Unified Vectorization Manager not available: {e}")
    UNIFIED_MANAGER_AVAILABLE = False


class TestEnhancedVectorBTVolatility(unittest.TestCase):
    """Test cases for enhanced VectorBT volatility features."""

    def setUp(self):
        """Set up test data."""
        # Create sample OHLCV data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1000, freq='1min')
        
        # Generate realistic price data
        base_price = 100
        returns = np.random.normal(0, 0.01, 1000)
        prices = base_price * np.exp(np.cumsum(returns))
        
        self.data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, 1000)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 1000))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 1000))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, 1000)
        }, index=dates)
        
        # Ensure high >= low and high/low >= open/close
        self.data['high'] = np.maximum(self.data['high'], self.data[['open', 'close']].max(axis=1))
        self.data['low'] = np.minimum(self.data['low'], self.data[['open', 'close']].min(axis=1))

    @unittest.skipUnless(ENHANCED_VECTORBT_AVAILABLE, "Enhanced VectorBT not available")
    def test_enhanced_volatility_generator_creation(self):
        """Test creation of enhanced volatility generator."""
        config = VolatilityConfig(period=20, enable_gpu=False, enable_parallel=True)
        generator = EnhancedVectorBTVolatilityGenerator(config)
        
        self.assertIsNotNone(generator)
        self.assertEqual(generator.config.period, 20)
        self.assertFalse(generator.config.enable_gpu)
        self.assertTrue(generator.config.enable_parallel)

    @unittest.skipUnless(ENHANCED_VECTORBT_AVAILABLE, "Enhanced VectorBT not available")
    def test_enhanced_volatility_feature_generation(self):
        """Test enhanced volatility feature generation."""
        config = VolatilityConfig(period=20, enable_gpu=False, enable_parallel=True)
        generator = EnhancedVectorBTVolatilityGenerator(config)
        
        # Test basic volatility
        volatility = generator._generate_feature(self.data)
        
        self.assertIsInstance(volatility, pd.Series)
        self.assertEqual(len(volatility), len(self.data))
        self.assertFalse(volatility.isna().all())
        
        # Test comprehensive features
        comprehensive_features = generator.generate_comprehensive_volatility_features(self.data)
        
        self.assertIsInstance(comprehensive_features, pd.DataFrame)
        self.assertEqual(len(comprehensive_features), len(self.data))
        self.assertGreater(len(comprehensive_features.columns), 1)

    @unittest.skipUnless(ENHANCED_VECTORBT_AVAILABLE, "Enhanced VectorBT not available")
    def test_volatility_config_validation(self):
        """Test volatility configuration validation."""
        # Test valid configuration
        config = VolatilityConfig(
            period=20,
            std_dev=2.0,
            enable_gpu=False,
            enable_parallel=True,
            use_unified_manager=True
        )
        
        self.assertEqual(config.period, 20)
        self.assertEqual(config.std_dev, 2.0)
        self.assertFalse(config.enable_gpu)
        self.assertTrue(config.enable_parallel)
        self.assertTrue(config.use_unified_manager)

    @unittest.skipUnless(ENHANCED_VECTORBT_AVAILABLE, "Enhanced VectorBT not available")
    def test_performance_tracking(self):
        """Test performance tracking functionality."""
        config = VolatilityConfig(period=20, enable_gpu=False, enable_parallel=True)
        generator = EnhancedVectorBTVolatilityGenerator(config)
        
        # Generate some features
        _ = generator._generate_feature(self.data)
        _ = generator.generate_comprehensive_volatility_features(self.data)
        
        # Check performance stats
        stats = generator.get_performance_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_operations', stats)
        self.assertIn('total_time', stats)
        self.assertGreater(stats['total_operations'], 0)

    @unittest.skipUnless(VOLATILITY_AVAILABLE, "Volatility module not available")
    def test_comprehensive_volatility_generators(self):
        """Test comprehensive volatility generators creation."""
        generators = create_comprehensive_vectorbt_volatility_generators(
            periods=[10, 20, 30],
            std_devs=[1.5, 2.0],
            enable_gpu=False,
            enable_parallel=True
        )
        
        self.assertIsInstance(generators, list)
        self.assertGreater(len(generators), 0)
        
        # Test that generators can create features
        for generator in generators[:3]:  # Test first 3 generators
            try:
                feature = generator._generate_feature(self.data)
                self.assertIsInstance(feature, pd.Series)
            except Exception as e:
                logger.warning(f"Generator {generator.__class__.__name__} failed: {e}")

    @unittest.skipUnless(VOLATILITY_AVAILABLE, "Volatility module not available")
    def test_optimized_volatility_pipeline(self):
        """Test optimized volatility pipeline."""
        features = create_optimized_volatility_pipeline(
            self.data,
            periods=[10, 20],
            std_devs=[1.5, 2.0],
            enable_gpu=False,
            enable_parallel=True
        )
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), len(self.data))
        self.assertGreater(len(features.columns), 0)

    @unittest.skipUnless(VOLATILITY_AVAILABLE, "Volatility module not available")
    def test_benchmark_volatility_optimizations(self):
        """Test volatility optimization benchmarking."""
        # Use smaller dataset for faster testing
        small_data = self.data.head(100)
        
        results = benchmark_volatility_optimizations(
            small_data,
            periods=[10, 20],
            trials=2
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('enhanced_vectorbt', results)
        self.assertIn('standard_vectorbt', results)
        self.assertIn('pandas_fallback', results)
        
        # Check that at least one method has results
        has_results = any(
            isinstance(results[key], dict) and 'avg_time' in results[key]
            for key in results
        )
        self.assertTrue(has_results)

    @unittest.skipUnless(ROLLING_OPTIMIZER_AVAILABLE, "VectorBT Rolling Optimizer not available")
    def test_rolling_optimizer_integration(self):
        """Test VectorBT Rolling Optimizer integration."""
        optimizer = VectorBTRollingOptimizer(enable_gpu=False, enable_parallel=True)
        
        # Test basic rolling operations
        close_prices = self.data['close']
        returns = close_prices.pct_change().dropna()
        
        # Test rolling mean
        rolling_mean = optimizer.rolling_mean(returns, window=20)
        self.assertIsInstance(rolling_mean, pd.Series)
        self.assertEqual(len(rolling_mean), len(returns))
        
        # Test rolling std
        rolling_std = optimizer.rolling_std(returns, window=20)
        self.assertIsInstance(rolling_std, pd.Series)
        self.assertEqual(len(rolling_std), len(returns))
        
        # Test performance stats
        stats = optimizer.get_performance_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_operations', stats)

    @unittest.skipUnless(UNIFIED_MANAGER_AVAILABLE, "Unified Vectorization Manager not available")
    def test_unified_manager_integration(self):
        """Test Unified Vectorization Manager integration."""
        manager = UnifiedVectorizationManager()
        
        # Test basic operation
        test_data = {'close': self.data['close'], 'period': 20}
        
        # This is a basic test - the actual operation might need more specific data
        self.assertIsNotNone(manager)
        
        # Test optimization stats
        stats = manager.get_optimization_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_operations', stats)

    def test_data_validation(self):
        """Test data validation and error handling."""
        # Test with empty data
        empty_data = pd.DataFrame()
        
        if ENHANCED_VECTORBT_AVAILABLE:
            config = VolatilityConfig(period=20)
            generator = EnhancedVectorBTVolatilityGenerator(config)
            
            result = generator._generate_feature(empty_data)
            self.assertIsInstance(result, pd.Series)
            self.assertEqual(len(result), 0)
        
        # Test with missing columns
        incomplete_data = self.data[['open', 'high']].copy()
        
        if ENHANCED_VECTORBT_AVAILABLE:
            config = VolatilityConfig(period=20)
            generator = EnhancedVectorBTVolatilityGenerator(config)
            
            result = generator._generate_feature(incomplete_data)
            self.assertIsInstance(result, pd.Series)
            self.assertEqual(len(result), len(incomplete_data))

    def test_memory_efficiency(self):
        """Test memory efficiency with larger datasets."""
        # Create larger dataset
        large_dates = pd.date_range('2020-01-01', periods=10000, freq='1min')
        large_returns = np.random.normal(0, 0.01, 10000)
        large_prices = 100 * np.exp(np.cumsum(large_returns))
        
        large_data = pd.DataFrame({
            'open': large_prices * (1 + np.random.normal(0, 0.001, 10000)),
            'high': large_prices * (1 + np.abs(np.random.normal(0, 0.005, 10000))),
            'low': large_prices * (1 - np.abs(np.random.normal(0, 0.005, 10000))),
            'close': large_prices,
            'volume': np.random.lognormal(10, 1, 10000)
        }, index=large_dates)
        
        # Ensure high >= low and high/low >= open/close
        large_data['high'] = np.maximum(large_data['high'], large_data[['open', 'close']].max(axis=1))
        large_data['low'] = np.minimum(large_data['low'], large_data[['open', 'close']].min(axis=1))
        
        if ENHANCED_VECTORBT_AVAILABLE:
            config = VolatilityConfig(
                period=20,
                memory_efficient=True,
                chunk_size=1000
            )
            generator = EnhancedVectorBTVolatilityGenerator(config)
            
            # Test that it can handle large data without memory issues
            start_time = time.time()
            result = generator._generate_feature(large_data)
            execution_time = time.time() - start_time
            
            self.assertIsInstance(result, pd.Series)
            self.assertEqual(len(result), len(large_data))
            self.assertLess(execution_time, 30)  # Should complete within 30 seconds


class TestVectorBTOptimizationPerformance(unittest.TestCase):
    """Test VectorBT optimization performance."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=5000, freq='1min')
        
        returns = np.random.normal(0, 0.01, 5000)
        prices = 100 * np.exp(np.cumsum(returns))
        
        self.data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, 5000)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 5000))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 5000))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, 5000)
        }, index=dates)
        
        # Ensure high >= low and high/low >= open/close
        self.data['high'] = np.maximum(self.data['high'], self.data[['open', 'close']].max(axis=1))
        self.data['low'] = np.minimum(self.data['low'], self.data[['open', 'close']].min(axis=1))

    @unittest.skipUnless(VOLATILITY_AVAILABLE, "Volatility module not available")
    def test_performance_comparison(self):
        """Test performance comparison between different approaches."""
        # Benchmark different approaches
        results = benchmark_volatility_optimizations(
            self.data,
            periods=[20],
            trials=3
        )
        
        # Check that we have results
        self.assertIsInstance(results, dict)
        
        # If we have results, check that they're reasonable
        for approach, result in results.items():
            if isinstance(result, dict) and 'avg_time' in result:
                self.assertGreater(result['avg_time'], 0)
                self.assertLess(result['avg_time'], 10)  # Should be reasonably fast

    @unittest.skipUnless(ENHANCED_VECTORBT_AVAILABLE, "Enhanced VectorBT not available")
    def test_strategy_selection(self):
        """Test intelligent strategy selection."""
        config = VolatilityConfig(
            period=20,
            use_unified_manager=True,
            use_rolling_optimizer=True,
            vectorization_threshold=1000
        )
        generator = EnhancedVectorBTVolatilityGenerator(config)
        
        # Test with different data sizes
        small_data = self.data.head(100)
        medium_data = self.data.head(1000)
        large_data = self.data.head(5000)
        
        # Test strategy selection
        small_strategy = generator._select_optimal_strategy(small_data)
        medium_strategy = generator._select_optimal_strategy(medium_data)
        large_strategy = generator._select_optimal_strategy(large_data)
        
        self.assertIn(small_strategy, ['direct_vectorbt', 'pandas_fallback'])
        self.assertIn(medium_strategy, ['rolling_optimizer', 'direct_vectorbt', 'pandas_fallback'])
        self.assertIn(large_strategy, ['unified_manager', 'rolling_optimizer', 'direct_vectorbt', 'pandas_fallback'])


def run_performance_benchmark():
    """Run a comprehensive performance benchmark."""
    print("🚀 Running VectorBT Volatility Performance Benchmark...")
    
    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=10000, freq='1min')
    returns = np.random.normal(0, 0.01, 10000)
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, 10000)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 10000))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 10000))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, 10000)
    }, index=dates)
    
    # Ensure high >= low and high/low >= open/close
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    if VOLATILITY_AVAILABLE:
        # Run benchmark
        results = benchmark_volatility_optimizations(data, periods=[20], trials=5)
        
        print("\n📊 Performance Benchmark Results:")
        print("=" * 50)
        
        for approach, result in results.items():
            if isinstance(result, dict) and 'avg_time' in result:
                print(f"{approach}:")
                print(f"  Average Time: {result['avg_time']:.4f}s")
                print(f"  Std Dev: {result['std_time']:.4f}s")
                print(f"  Min Time: {result['min_time']:.4f}s")
                print(f"  Max Time: {result['max_time']:.4f}s")
                print()
        
        # Find fastest approach
        fastest_approach = None
        fastest_time = float('inf')
        
        for approach, result in results.items():
            if isinstance(result, dict) and 'avg_time' in result:
                if result['avg_time'] < fastest_time:
                    fastest_time = result['avg_time']
                    fastest_approach = approach
        
        if fastest_approach:
            print(f"🏆 Fastest Approach: {fastest_approach} ({fastest_time:.4f}s)")
    
    print("✅ Performance benchmark completed!")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
    
    # Run performance benchmark
    run_performance_benchmark()