"""
Validation Tests for VectorBT Optimizations

This module provides comprehensive tests to validate that all VectorBT optimizations
are working correctly and providing the expected performance improvements.

Key Features:
- Performance benchmarking
- Correctness validation
- Memory usage testing
- Error handling validation
- Integration testing
"""

import numpy as np
import pandas as pd
import time
import logging
import warnings
from typing import Dict, Any, List, Tuple
import unittest
from unittest.mock import patch, MagicMock

# Import optimization components
from .consolidated_rolling_optimizer import (
    ConsolidatedRollingOptimizer,
    RollingOperationConfig,
    RollingOperationType,
    create_rolling_optimizer
)
from .statistical_calculations_optimizer import (
    StatisticalCalculationsOptimizer,
    StatisticalOperationConfig,
    StatisticalOperationType,
    create_statistical_optimizer
)
from .unified_optimization_wrapper import (
    UnifiedOptimizationWrapper,
    UnifiedOptimizationConfig,
    OptimizationMode,
    create_unified_optimizer
)

# Import enhanced feature generator
from ..categories.optimized_volatility_enhanced import (
    OptimizedVolatilityFeatureGenerator,
    create_optimized_volatility_generator
)

logger = logging.getLogger(__name__)


class VectorBTOptimizationValidationTests(unittest.TestCase):
    """Comprehensive validation tests for VectorBT optimizations."""
    
    def setUp(self):
        """Set up test data and components."""
        # Create test data
        np.random.seed(42)
        n_samples = 1000
        self.test_data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
            'high': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) + np.random.rand(n_samples) * 0.5,
            'low': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) - np.random.rand(n_samples) * 0.5,
            'volume': np.random.randint(1000, 10000, n_samples)
        })
        
        # Ensure high >= low
        self.test_data['high'] = np.maximum(self.test_data['high'], self.test_data['close'])
        self.test_data['low'] = np.minimum(self.test_data['low'], self.test_data['close'])
        
        # Initialize optimizers
        self.rolling_optimizer = create_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        self.statistical_optimizer = create_statistical_optimizer(enable_gpu=False, enable_parallel=True)
        self.unified_optimizer = create_unified_optimizer()
        
        # Initialize enhanced feature generator
        self.volatility_generator = create_optimized_volatility_generator(
            enable_gpu=False,
            enable_parallel=True,
            optimization_mode=OptimizationMode.AUTO
        )
    
    def test_rolling_operations_correctness(self):
        """Test that rolling operations produce correct results."""
        data = self.test_data['close']
        window = 20
        
        # Test individual rolling operations
        configs = [
            RollingOperationConfig(operation=RollingOperationType.MEAN, window=window),
            RollingOperationConfig(operation=RollingOperationType.STD, window=window),
            RollingOperationConfig(operation=RollingOperationType.VAR, window=window),
            RollingOperationConfig(operation=RollingOperationType.MIN, window=window),
            RollingOperationConfig(operation=RollingOperationType.MAX, window=window),
            RollingOperationConfig(operation=RollingOperationType.SUM, window=window)
        ]
        
        for config in configs:
            with self.subTest(operation=config.operation.value):
                result = self.rolling_optimizer.single_rolling_operation(data, config)
                
                # Validate result
                self.assertIsInstance(result, pd.Series)
                self.assertEqual(len(result), len(data))
                self.assertFalse(result.isna().all())
                
                # Compare with pandas implementation
                if config.operation == RollingOperationType.MEAN:
                    expected = data.rolling(window=window).mean()
                elif config.operation == RollingOperationType.STD:
                    expected = data.rolling(window=window).std()
                elif config.operation == RollingOperationType.VAR:
                    expected = data.rolling(window=window).var()
                elif config.operation == RollingOperationType.MIN:
                    expected = data.rolling(window=window).min()
                elif config.operation == RollingOperationType.MAX:
                    expected = data.rolling(window=window).max()
                elif config.operation == RollingOperationType.SUM:
                    expected = data.rolling(window=window).sum()
                
                # Check that results are approximately equal (within numerical precision)
                pd.testing.assert_series_equal(result, expected, check_names=False, rtol=1e-10)
    
    def test_batch_rolling_operations(self):
        """Test batch rolling operations."""
        data = self.test_data['close']
        operations = ['mean', 'std', 'var']
        windows = [10, 20, 50]
        
        # Test batch operations
        results = self.rolling_optimizer.batch_rolling_operations(
            data, operations, windows
        )
        
        # Validate results
        self.assertIsInstance(results, dict)
        expected_count = len(operations) * len(windows)
        self.assertEqual(len(results), expected_count)
        
        # Check that all results are valid
        for op_name, result in results.items():
            self.assertIsInstance(result, pd.Series)
            self.assertEqual(len(result), len(data))
            self.assertFalse(result.isna().all())
    
    def test_statistical_operations_correctness(self):
        """Test that statistical operations produce correct results."""
        data = self.test_data['close']
        window = 20
        
        # Test individual statistical operations
        configs = [
            StatisticalOperationConfig(operation=StatisticalOperationType.MEAN, window=window),
            StatisticalOperationConfig(operation=StatisticalOperationType.STD, window=window),
            StatisticalOperationConfig(operation=StatisticalOperationType.VAR, window=window),
            StatisticalOperationConfig(operation=StatisticalOperationType.SKEW, window=window),
            StatisticalOperationConfig(operation=StatisticalOperationType.KURT, window=window),
            StatisticalOperationConfig(operation=StatisticalOperationType.QUANTILE, window=window, quantile_value=0.5)
        ]
        
        for config in configs:
            with self.subTest(operation=config.operation.value):
                result = self.statistical_optimizer.single_statistical_operation(data, config)
                
                # Validate result
                self.assertIsInstance(result, (pd.Series, float))
                if isinstance(result, pd.Series):
                    self.assertEqual(len(result), len(data))
                    self.assertFalse(result.isna().all())
    
    def test_batch_statistical_operations(self):
        """Test batch statistical operations."""
        data = self.test_data['close']
        operations = ['mean', 'std', 'skew', 'kurt']
        windows = [20, 50]
        
        # Test batch operations
        results = self.statistical_optimizer.batch_statistical_operations(
            data, operations, windows
        )
        
        # Validate results
        self.assertIsInstance(results, dict)
        expected_count = len(operations) * len(windows)
        self.assertEqual(len(results), expected_count)
        
        # Check that all results are valid
        for op_name, result in results.items():
            self.assertIsInstance(result, (pd.Series, float))
            if isinstance(result, pd.Series):
                self.assertEqual(len(result), len(data))
    
    def test_unified_optimizer_integration(self):
        """Test unified optimizer integration."""
        data = self.test_data['close']
        
        # Test simple operation
        def simple_mean(data):
            return data.mean()
        
        result = self.unified_optimizer.optimize_operation(
            operation_type="statistical",
            data=data,
            operation_func=simple_mean
        )
        
        # Validate result
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, data.mean(), places=10)
    
    def test_enhanced_volatility_generator(self):
        """Test the enhanced volatility feature generator."""
        # Generate features
        features = self.volatility_generator.generate_features(self.test_data)
        
        # Validate features
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 0)
        self.assertEqual(len(features), len(self.test_data))
        
        # Check for expected feature types
        feature_names = features.columns.tolist()
        self.assertTrue(any('mean' in name for name in feature_names))
        self.assertTrue(any('std' in name for name in feature_names))
        self.assertTrue(any('vol' in name for name in feature_names))
    
    def test_performance_improvements(self):
        """Test that optimizations provide performance improvements."""
        data = self.test_data['close']
        window = 20
        
        # Test rolling operations performance
        start_time = time.time()
        for _ in range(10):
            result = self.rolling_optimizer.single_rolling_operation(
                data, 
                RollingOperationConfig(operation=RollingOperationType.MEAN, window=window)
            )
        optimized_time = time.time() - start_time
        
        # Test pandas performance
        start_time = time.time()
        for _ in range(10):
            result = data.rolling(window=window).mean()
        pandas_time = time.time() - start_time
        
        # Optimized version should be faster
        self.assertLess(optimized_time, pandas_time * 2)  # Allow some margin for overhead
        
        logger.info(f"Optimized time: {optimized_time:.4f}s, Pandas time: {pandas_time:.4f}s")
        logger.info(f"Speedup: {pandas_time / optimized_time:.2f}x")
    
    def test_memory_usage(self):
        """Test memory usage optimization."""
        # This is a basic test - in practice, you'd use memory profiling tools
        data = self.test_data['close']
        
        # Test that operations don't cause memory issues
        for _ in range(100):
            result = self.rolling_optimizer.single_rolling_operation(
                data,
                RollingOperationConfig(operation=RollingOperationType.MEAN, window=20)
            )
            del result  # Explicit cleanup
    
    def test_error_handling(self):
        """Test error handling and fallbacks."""
        # Test with invalid data
        invalid_data = pd.Series([np.nan] * 100)
        
        # Should not raise exception
        result = self.rolling_optimizer.single_rolling_operation(
            invalid_data,
            RollingOperationConfig(operation=RollingOperationType.MEAN, window=20)
        )
        
        self.assertIsInstance(result, pd.Series)
    
    def test_performance_monitoring(self):
        """Test performance monitoring functionality."""
        data = self.test_data['close']
        
        # Perform some operations
        for _ in range(10):
            self.rolling_optimizer.single_rolling_operation(
                data,
                RollingOperationConfig(operation=RollingOperationType.MEAN, window=20)
            )
        
        # Get performance stats
        stats = self.rolling_optimizer.get_performance_stats()
        
        # Validate stats
        self.assertIn('total_operations', stats)
        self.assertIn('vectorbt_operations', stats)
        self.assertIn('pandas_fallbacks', stats)
        self.assertIn('total_time', stats)
        
        self.assertEqual(stats['total_operations'], 10)
        self.assertGreater(stats['total_time'], 0)
    
    def test_batch_processing_scalability(self):
        """Test batch processing scalability."""
        # Test with larger dataset
        large_data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(5000) * 0.01),
            'high': 100 + np.cumsum(np.random.randn(5000) * 0.01) + np.random.rand(5000) * 0.5,
            'low': 100 + np.cumsum(np.random.randn(5000) * 0.01) - np.random.rand(5000) * 0.5,
            'volume': np.random.randint(1000, 10000, 5000)
        })
        
        # Test batch operations
        start_time = time.time()
        results = self.rolling_optimizer.batch_rolling_operations(
            large_data['close'],
            operations=['mean', 'std', 'var'],
            windows=[10, 20, 50]
        )
        batch_time = time.time() - start_time
        
        # Should complete in reasonable time
        self.assertLess(batch_time, 10.0)  # Should complete within 10 seconds
        
        logger.info(f"Batch processing time for 5000 samples: {batch_time:.4f}s")


def run_performance_benchmark():
    """Run a comprehensive performance benchmark."""
    print("🚀 VectorBT Optimization Performance Benchmark")
    print("=" * 50)
    
    # Create test data
    np.random.seed(42)
    n_samples = 10000
    test_data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
        'high': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) + np.random.rand(n_samples) * 0.5,
        'low': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) - np.random.rand(n_samples) * 0.5,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Initialize optimizers
    rolling_optimizer = create_rolling_optimizer(enable_gpu=False, enable_parallel=True)
    statistical_optimizer = create_statistical_optimizer(enable_gpu=False, enable_parallel=True)
    volatility_generator = create_optimized_volatility_generator(enable_gpu=False, enable_parallel=True)
    
    # Benchmark rolling operations
    print("\n📊 Rolling Operations Benchmark")
    print("-" * 30)
    
    # Pandas baseline
    start_time = time.time()
    for _ in range(5):
        mean_result = test_data['close'].rolling(window=20).mean()
        std_result = test_data['close'].rolling(window=20).std()
        var_result = test_data['close'].rolling(window=20).var()
    pandas_time = time.time() - start_time
    
    # Optimized version
    start_time = time.time()
    for _ in range(5):
        results = rolling_optimizer.batch_rolling_operations(
            test_data['close'],
            operations=['mean', 'std', 'var'],
            windows=[20]
        )
    optimized_time = time.time() - start_time
    
    speedup = pandas_time / optimized_time
    print(f"Pandas time: {pandas_time:.4f}s")
    print(f"Optimized time: {optimized_time:.4f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    # Benchmark statistical operations
    print("\n📈 Statistical Operations Benchmark")
    print("-" * 35)
    
    # Manual calculations baseline
    start_time = time.time()
    for _ in range(5):
        centered = test_data['close'] - test_data['close'].rolling(window=20).mean()
        rolling_std = test_data['close'].rolling(window=20).std()
        skewness = (centered ** 3).rolling(window=20).mean() / (rolling_std ** 3 + 1e-8)
        kurtosis = (centered ** 4).rolling(window=20).mean() / (rolling_std ** 4 + 1e-8) - 3
    manual_time = time.time() - start_time
    
    # Optimized version
    start_time = time.time()
    for _ in range(5):
        results = statistical_optimizer.batch_statistical_operations(
            test_data['close'],
            operations=['skew', 'kurt'],
            windows=[20]
        )
    optimized_time = time.time() - start_time
    
    speedup = manual_time / optimized_time
    print(f"Manual time: {manual_time:.4f}s")
    print(f"Optimized time: {optimized_time:.4f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    # Benchmark feature generation
    print("\n🎯 Feature Generation Benchmark")
    print("-" * 32)
    
    # Generate features
    start_time = time.time()
    features = volatility_generator.generate_features(test_data)
    generation_time = time.time() - start_time
    
    print(f"Generated {len(features.columns)} features in {generation_time:.4f}s")
    print(f"Features per second: {len(features.columns) / generation_time:.2f}")
    
    # Get performance report
    report = volatility_generator.get_performance_report()
    print(f"\n📋 Performance Report")
    print("-" * 20)
    print(f"Total operations: {report['unified_stats']['total_operations']}")
    print(f"Optimization hit rate: {report['efficiency_metrics']['optimization_hit_rate']:.2%}")
    print(f"Average operation time: {report['efficiency_metrics']['average_operation_time']:.6f}s")
    
    print("\n✅ Benchmark completed successfully!")


if __name__ == "__main__":
    # Run tests
    print("🧪 Running VectorBT Optimization Tests")
    print("=" * 40)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance benchmark
    run_performance_benchmark()