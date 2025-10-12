"""
Test suite for VectorBT-optimized candlestick pattern feature generation.

This module provides comprehensive tests for the candlestick pattern generators
to ensure VectorBT integration is working correctly and performance is optimal.
"""

import unittest
import numpy as np
import pandas as pd
import time
from typing import List, Dict, Any

# Import the modules to test
from ..categories.candlestick_pattern import (
    CandlestickPatternFeatureGenerator,
    VectorBTCandlestickPatternGenerator,
    VectorBTCandlestickPatternBatchProcessor,
    CandlestickPatternConfig,
    create_candlestick_pattern_generators,
    create_vectorbt_candlestick_generator,
    create_candlestick_batch_processor
)

from ..utils.unified_vectorization_manager import get_unified_vectorization_manager

class TestCandlestickPatternVectorBT(unittest.TestCase):
    """Test cases for VectorBT-optimized candlestick pattern generation."""
    
    def setUp(self):
        """Set up test data and configurations."""
        # Create test OHLCV data
        np.random.seed(42)
        n_periods = 1000
        
        # Generate realistic price data
        returns = np.random.normal(0, 0.02, n_periods)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = []
        for i, price in enumerate(prices):
            volatility = np.random.uniform(0.005, 0.02)
            
            open_price = price * (1 + np.random.normal(0, volatility/2))
            close_price = price * (1 + np.random.normal(0, volatility/2))
            high_price = max(open_price, close_price) * (1 + np.random.uniform(0, volatility))
            low_price = min(open_price, close_price) * (1 - np.random.uniform(0, volatility))
            volume = np.random.lognormal(10, 1)
            
            data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        self.test_data = pd.DataFrame(data, index=pd.date_range('2020-01-01', periods=n_periods, freq='1H'))
        
        # Test configurations
        self.basic_config = CandlestickPatternConfig()
        self.optimized_config = CandlestickPatternConfig(
            enable_vectorbt=True,
            enable_batch_processing=True,
            enable_memory_optimization=True,
            chunk_size=1000
        )
    
    def test_basic_pattern_detection(self):
        """Test basic pattern detection functionality."""
        generator = CandlestickPatternFeatureGenerator()
        
        # Test doji detection
        doji_result = generator._detect_doji_pattern(self.test_data)
        self.assertIsInstance(doji_result, pd.Series)
        self.assertEqual(len(doji_result), len(self.test_data))
        self.assertTrue(doji_result.dtype in [np.int32, np.int64, bool])
        
        # Test hammer detection
        hammer_result = generator._detect_hammer_pattern(self.test_data)
        self.assertIsInstance(hammer_result, pd.Series)
        self.assertEqual(len(hammer_result), len(self.test_data))
        
        # Test engulfing patterns
        bullish_engulfing = generator._detect_bullish_engulfing(self.test_data)
        bearish_engulfing = generator._detect_bearish_engulfing(self.test_data)
        
        self.assertIsInstance(bullish_engulfing, pd.Series)
        self.assertIsInstance(bearish_engulfing, pd.Series)
    
    def test_vectorbt_optimized_detection(self):
        """Test VectorBT-optimized pattern detection."""
        generator = VectorBTCandlestickPatternGenerator(pattern_config=self.optimized_config)
        
        # Test single pattern detection
        doji_result = generator._detect_doji_pattern(self.test_data)
        self.assertIsInstance(doji_result, pd.Series)
        self.assertEqual(len(doji_result), len(self.test_data))
        
        # Test batch pattern detection
        patterns = ['doji', 'hammer', 'engulfing_bullish']
        batch_result = generator._generate_patterns_batch(self.test_data, patterns)
        self.assertIsInstance(batch_result, pd.Series)
        
        # Test all patterns generation
        all_patterns = generator.generate_all_patterns(self.test_data)
        self.assertIsInstance(all_patterns, pd.DataFrame)
        self.assertGreater(len(all_patterns.columns), 0)
    
    def test_pattern_confidence_calculation(self):
        """Test pattern confidence calculation."""
        generator = VectorBTCandlestickPatternGenerator(pattern_config=self.optimized_config)
        
        # Generate patterns with confidence
        patterns_with_confidence = generator.generate_patterns_with_confidence(
            self.test_data, patterns=['doji', 'hammer']
        )
        
        self.assertIsInstance(patterns_with_confidence, pd.DataFrame)
        
        # Check that confidence scores are between 0 and 1
        for col in patterns_with_confidence.columns:
            if 'confidence' in col:
                confidence_values = patterns_with_confidence[col].dropna()
                if len(confidence_values) > 0:
                    self.assertTrue(all(0 <= val <= 1 for val in confidence_values))
    
    def test_batch_processing(self):
        """Test batch processing functionality."""
        # Create multiple configurations
        configs = [
            CandlestickPatternConfig(doji_threshold=0.05),
            CandlestickPatternConfig(doji_threshold=0.15)
        ]
        
        batch_processor = create_candlestick_batch_processor(configs)
        
        # Define pattern lists
        pattern_lists = [
            ['doji', 'hammer'],
            ['engulfing_bullish', 'engulfing_bearish']
        ]
        
        # Process batch
        results = batch_processor.process_batch(self.test_data, pattern_lists)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(configs))
        
        for result in results:
            self.assertIsInstance(result, pd.Series)
    
    def test_memory_optimization(self):
        """Test memory optimization features."""
        # Create larger dataset
        large_data = pd.concat([self.test_data] * 5, ignore_index=True)
        large_data.index = pd.date_range('2020-01-01', periods=len(large_data), freq='1H')
        
        generator = VectorBTCandlestickPatternGenerator(pattern_config=self.optimized_config)
        
        # Test memory-optimized processing
        start_time = time.time()
        results = generator.generate_all_patterns(large_data)
        execution_time = time.time() - start_time
        
        self.assertIsInstance(results, pd.DataFrame)
        self.assertLess(execution_time, 10.0)  # Should complete within 10 seconds
        
        # Test memory usage tracking
        if hasattr(generator, 'vectorization_manager') and generator.vectorization_manager:
            stats = generator.vectorization_manager.get_performance_stats()
            self.assertIn('memory_optimizations', stats)
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        generator = VectorBTCandlestickPatternGenerator(pattern_config=self.optimized_config)
        
        # Generate some patterns
        generator.generate_all_patterns(self.test_data)
        
        # Check performance stats
        stats = generator.get_pattern_stats()
        self.assertIn('patterns_detected', stats)
        self.assertIn('vectorbt_operations', stats)
        self.assertIn('total_execution_time', stats)
        
        # Check that stats are reasonable
        self.assertGreaterEqual(stats['patterns_detected'], 0)
        self.assertGreaterEqual(stats['total_execution_time'], 0)
    
    def test_configuration_validation(self):
        """Test configuration validation and application."""
        # Test default configuration
        default_config = CandlestickPatternConfig()
        self.assertEqual(default_config.doji_threshold, 0.1)
        self.assertEqual(default_config.hammer_threshold, 0.3)
        self.assertTrue(default_config.enable_vectorbt)
        
        # Test custom configuration
        custom_config = CandlestickPatternConfig(
            doji_threshold=0.05,
            hammer_threshold=0.2,
            enable_gpu_acceleration=True
        )
        self.assertEqual(custom_config.doji_threshold, 0.05)
        self.assertEqual(custom_config.hammer_threshold, 0.2)
        self.assertTrue(custom_config.enable_gpu_acceleration)
    
    def test_error_handling(self):
        """Test error handling and fallbacks."""
        generator = VectorBTCandlestickPatternGenerator(pattern_config=self.optimized_config)
        
        # Test with invalid data
        invalid_data = pd.DataFrame({'invalid': [1, 2, 3]})
        
        with self.assertRaises((KeyError, AttributeError)):
            generator._detect_doji_pattern(invalid_data)
        
        # Test with empty data
        empty_data = pd.DataFrame(columns=['open', 'high', 'low', 'close'])
        result = generator._detect_doji_pattern(empty_data)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 0)
    
    def test_vectorbt_availability(self):
        """Test VectorBT availability and fallbacks."""
        generator = VectorBTCandlestickPatternGenerator(pattern_config=self.optimized_config)
        
        # Test that generator initializes regardless of VectorBT availability
        self.assertIsNotNone(generator)
        
        # Test that patterns can be detected
        result = generator._detect_doji_pattern(self.test_data)
        self.assertIsInstance(result, pd.Series)
    
    def test_pattern_consistency(self):
        """Test that patterns are detected consistently."""
        generator = VectorBTCandlestickPatternGenerator(pattern_config=self.optimized_config)
        
        # Test multiple runs produce consistent results
        result1 = generator._detect_doji_pattern(self.test_data)
        result2 = generator._detect_doji_pattern(self.test_data)
        
        pd.testing.assert_series_equal(result1, result2)
    
    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        # Create large dataset
        large_data = pd.concat([self.test_data] * 10, ignore_index=True)
        large_data.index = pd.date_range('2020-01-01', periods=len(large_data), freq='1H')
        
        generator = VectorBTCandlestickPatternGenerator(pattern_config=self.optimized_config)
        
        # Time the operation
        start_time = time.time()
        results = generator.generate_all_patterns(large_data)
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time
        self.assertLess(execution_time, 30.0)
        self.assertIsInstance(results, pd.DataFrame)
        
        # Check that results are reasonable
        self.assertGreater(len(results.columns), 0)
        self.assertEqual(len(results), len(large_data))

class TestUnifiedVectorizationManager(unittest.TestCase):
    """Test cases for UnifiedVectorizationManager."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = pd.DataFrame({
            'close': np.random.randn(1000).cumsum() + 100,
            'volume': np.random.lognormal(10, 1, 1000)
        }, index=pd.date_range('2020-01-01', periods=1000, freq='1H'))
    
    def test_unified_vectorization_manager_initialization(self):
        """Test UnifiedVectorizationManager initialization."""
        manager = get_unified_vectorization_manager()
        self.assertIsNotNone(manager)
    
    def test_dataframe_optimization(self):
        """Test DataFrame optimization."""
        manager = get_unified_vectorization_manager()
        optimized_data = manager.optimize_dataframe(self.test_data)
        
        self.assertIsInstance(optimized_data, pd.DataFrame)
        self.assertEqual(len(optimized_data), len(self.test_data))
    
    def test_rolling_operations(self):
        """Test vectorized rolling operations."""
        manager = get_unified_vectorization_manager()
        
        # Test rolling mean
        rolling_mean = manager.vectorized_rolling_operation(
            self.test_data['close'], 'mean', window=20
        )
        self.assertIsInstance(rolling_mean, pd.Series)
        self.assertEqual(len(rolling_mean), len(self.test_data))
        
        # Test rolling std
        rolling_std = manager.vectorized_rolling_operation(
            self.test_data['close'], 'std', window=20
        )
        self.assertIsInstance(rolling_std, pd.Series)
        self.assertEqual(len(rolling_std), len(self.test_data))
    
    def test_scaling_operations(self):
        """Test vectorized scaling operations."""
        manager = get_unified_vectorization_manager()
        
        # Test z-score scaling
        zscore_result = manager.vectorized_scale(
            self.test_data['close'], method='zscore'
        )
        self.assertIsInstance(zscore_result, pd.Series)
        self.assertEqual(len(zscore_result), len(self.test_data))
        
        # Test minmax scaling
        minmax_result = manager.vectorized_scale(
            self.test_data['close'], method='minmax'
        )
        self.assertIsInstance(minmax_result, pd.Series)
        self.assertEqual(len(minmax_result), len(self.test_data))
    
    def test_batch_operations(self):
        """Test batch operations."""
        manager = get_unified_vectorization_manager()
        
        operations = [
            {
                'type': 'rolling',
                'name': 'close_mean_20',
                'params': {'operation': 'mean', 'window': 20, 'column': 'close'}
            },
            {
                'type': 'rolling',
                'name': 'close_std_20',
                'params': {'operation': 'std', 'window': 20, 'column': 'close'}
            }
        ]
        
        results = manager.batch_vectorized_operations(self.test_data, operations)
        self.assertIsInstance(results, pd.DataFrame)
        self.assertGreater(len(results.columns), 0)
    
    def test_performance_stats(self):
        """Test performance statistics collection."""
        manager = get_unified_vectorization_manager()
        
        # Perform some operations
        manager.vectorized_rolling_operation(self.test_data['close'], 'mean', 20)
        manager.vectorized_scale(self.test_data['close'], 'zscore')
        
        # Get stats
        stats = manager.get_performance_stats()
        self.assertIn('total_operations', stats)
        self.assertIn('vectorbt_operations', stats)
        self.assertIn('total_execution_time', stats)

def run_performance_benchmark():
    """Run performance benchmark tests."""
    print("🚀 Running Performance Benchmark Tests")
    print("=" * 50)
    
    # Create test data
    test_data = pd.DataFrame({
        'open': np.random.randn(5000).cumsum() + 100,
        'high': np.random.randn(5000).cumsum() + 105,
        'low': np.random.randn(5000).cumsum() + 95,
        'close': np.random.randn(5000).cumsum() + 100,
        'volume': np.random.lognormal(10, 1, 5000)
    }, index=pd.date_range('2020-01-01', periods=5000, freq='1H'))
    
    # Test 1: Basic generator
    print("Testing basic generator...")
    basic_generator = CandlestickPatternFeatureGenerator()
    
    start_time = time.time()
    basic_result = basic_generator._detect_doji_pattern(test_data)
    basic_time = time.time() - start_time
    
    print(f"Basic generator doji detection: {basic_time:.3f}s")
    
    # Test 2: VectorBT-optimized generator
    print("Testing VectorBT-optimized generator...")
    vectorbt_generator = VectorBTCandlestickPatternGenerator()
    
    start_time = time.time()
    vectorbt_result = vectorbt_generator._detect_doji_pattern(test_data)
    vectorbt_time = time.time() - start_time
    
    print(f"VectorBT generator doji detection: {vectorbt_time:.3f}s")
    
    # Test 3: All patterns generation
    print("Testing all patterns generation...")
    start_time = time.time()
    all_patterns = vectorbt_generator.generate_all_patterns(test_data)
    all_patterns_time = time.time() - start_time
    
    print(f"All patterns generation: {all_patterns_time:.3f}s")
    print(f"Patterns detected: {len(all_patterns.columns)}")
    
    # Performance summary
    if basic_time > 0:
        speedup = basic_time / vectorbt_time
        print(f"\nPerformance Summary:")
        print(f"VectorBT speedup: {speedup:.2f}x")
        print(f"All patterns generation: {all_patterns_time:.3f}s")

if __name__ == '__main__':
    # Run unit tests
    print("🧪 Running Unit Tests")
    print("=" * 30)
    
    unittest.main(verbosity=2, exit=False)
    
    # Run performance benchmark
    print("\n")
    run_performance_benchmark()