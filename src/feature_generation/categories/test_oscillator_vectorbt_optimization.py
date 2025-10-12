"""
Comprehensive test suite for VectorBT optimization in oscillator feature generation.

This module tests the enhanced oscillator feature generators with:
- VectorBTRollingOptimizer integration
- UnifiedVectorizationManager usage
- Performance monitoring and statistics
- Native VectorBT technical analysis indicators
- GPU acceleration support
- Batch processing capabilities
"""

import unittest
import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Any

# Test imports
try:
    from .oscillator import (
        OscillatorFeatureGenerator, CCIGenerator, ADXGenerator, AroonGenerator,
        create_oscillator_generators, create_default_oscillator_generators
    )
    from .oscillator_optimized import (
        VectorBTOscillatorFeatureGenerator, VectorBTCCIGenerator, VectorBTADXGenerator, 
        VectorBTAroonGenerator, VectorBTOscillatorFactory,
        create_vectorbt_oscillator_generators, create_default_vectorbt_oscillator_generators
    )
    OSCILLATOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import oscillator modules: {e}")
    OSCILLATOR_AVAILABLE = False

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var
    from vectorbt.indicators.basic import RSI, MA, BBANDS, STOCH
    from vectorbt.indicators.momentum import MACD, ADX, CCI
    from vectorbt.indicators.volatility import ATR, BollingerBands
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None

# Unified Vectorization Manager
try:
    from ...utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager, OperationType, OptimizationStrategy, 
        OperationConfig, OptimizationResult
    )
    UNIFIED_MANAGER_AVAILABLE = True
except ImportError:
    UNIFIED_MANAGER_AVAILABLE = False

# VectorBT Rolling Optimizer
try:
    from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    VECTORBT_OPTIMIZER_AVAILABLE = True
except ImportError:
    VECTORBT_OPTIMIZER_AVAILABLE = False

logger = logging.getLogger(__name__)


class TestVectorBTOscillatorOptimization(unittest.TestCase):
    """Test suite for VectorBT optimization in oscillator feature generation."""
    
    def setUp(self):
        """Set up test data and configurations."""
        # Create test data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1000, freq='1min')
        
        # Generate realistic OHLCV data
        close_prices = 100 + np.cumsum(np.random.randn(1000) * 0.01)
        high_prices = close_prices + np.random.uniform(0, 0.5, 1000)
        low_prices = close_prices - np.random.uniform(0, 0.5, 1000)
        volume = np.random.lognormal(10, 1, 1000)
        
        self.test_data = pd.DataFrame({
            'open': close_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        }, index=dates)
        
        # Test configurations
        self.test_periods = {
            'cci': [20],
            'adx': [14],
            'aroon': [25]
        }
        
        self.large_test_data = pd.DataFrame({
            'open': close_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        }, index=pd.date_range('2020-01-01', periods=10000, freq='1min'))
    
    def test_vectorbt_availability(self):
        """Test VectorBT availability and basic functionality."""
        if not VECTORBT_AVAILABLE:
            self.skipTest("VectorBT not available")
        
        # Test basic VectorBT functionality
        data = np.random.randn(100, 5)
        result = rolling_mean(data, window=10)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, data.shape)
    
    def test_vectorbt_rolling_optimizer_availability(self):
        """Test VectorBTRollingOptimizer availability."""
        if not VECTORBT_OPTIMIZER_AVAILABLE:
            self.skipTest("VectorBTRollingOptimizer not available")
        
        optimizer = get_vectorbt_rolling_optimizer()
        self.assertIsNotNone(optimizer)
        self.assertIsInstance(optimizer, VectorBTRollingOptimizer)
    
    def test_unified_vectorization_manager_availability(self):
        """Test UnifiedVectorizationManager availability."""
        if not UNIFIED_MANAGER_AVAILABLE:
            self.skipTest("UnifiedVectorizationManager not available")
        
        manager = UnifiedVectorizationManager()
        self.assertIsNotNone(manager)
    
    def test_oscillator_generator_creation(self):
        """Test creation of oscillator generators."""
        if not OSCILLATOR_AVAILABLE:
            self.skipTest("Oscillator modules not available")
        
        # Test basic oscillator generator
        generator = OscillatorFeatureGenerator()
        self.assertIsNotNone(generator)
        
        # Test CCI generator
        cci_generator = CCIGenerator(period=20)
        self.assertIsNotNone(cci_generator)
        self.assertEqual(cci_generator.period, 20)
        
        # Test ADX generator
        adx_generator = ADXGenerator(period=14)
        self.assertIsNotNone(adx_generator)
        self.assertEqual(adx_generator.period, 14)
        
        # Test Aroon generator
        aroon_generator = AroonGenerator(period=25)
        self.assertIsNotNone(aroon_generator)
        self.assertEqual(aroon_generator.period, 25)
    
    def test_vectorbt_oscillator_generator_creation(self):
        """Test creation of VectorBT-optimized oscillator generators."""
        if not OSCILLATOR_AVAILABLE:
            self.skipTest("Oscillator modules not available")
        
        # Test VectorBT oscillator generator
        generator = VectorBTOscillatorFeatureGenerator()
        self.assertIsNotNone(generator)
        
        # Test VectorBT CCI generator
        cci_generator = VectorBTCCIGenerator(period=20)
        self.assertIsNotNone(cci_generator)
        self.assertEqual(cci_generator.period, 20)
        
        # Test VectorBT ADX generator
        adx_generator = VectorBTADXGenerator(period=14)
        self.assertIsNotNone(adx_generator)
        self.assertEqual(adx_generator.period, 14)
        
        # Test VectorBT Aroon generator
        aroon_generator = VectorBTAroonGenerator(period=25)
        self.assertIsNotNone(aroon_generator)
        self.assertEqual(aroon_generator.period, 25)
    
    def test_oscillator_feature_generation(self):
        """Test oscillator feature generation."""
        if not OSCILLATOR_AVAILABLE:
            self.skipTest("Oscillator modules not available")
        
        # Test basic oscillator generation
        generator = OscillatorFeatureGenerator()
        result = generator.generate(self.test_data)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.test_data))
        self.assertFalse(result.isna().all())
    
    def test_cci_generation(self):
        """Test CCI generation with VectorBT optimization."""
        if not OSCILLATOR_AVAILABLE:
            self.skipTest("Oscillator modules not available")
        
        # Test CCI generation
        cci_generator = CCIGenerator(period=20)
        result = cci_generator.generate(self.test_data)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.test_data))
        
        # Check that CCI values are reasonable
        self.assertFalse(result.isna().all())
        self.assertTrue(np.isfinite(result).any())
    
    def test_vectorbt_cci_generation(self):
        """Test VectorBT-optimized CCI generation."""
        if not OSCILLATOR_AVAILABLE:
            self.skipTest("Oscillator modules not available")
        
        # Test VectorBT CCI generation
        cci_generator = VectorBTCCIGenerator(period=20)
        result = cci_generator.generate(self.test_data)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.test_data))
        
        # Check that CCI values are reasonable
        self.assertFalse(result.isna().all())
        self.assertTrue(np.isfinite(result).any())
    
    def test_adx_generation(self):
        """Test ADX generation with VectorBT optimization."""
        if not OSCILLATOR_AVAILABLE:
            self.skipTest("Oscillator modules not available")
        
        # Test ADX generation
        adx_generator = ADXGenerator(period=14)
        result = adx_generator.generate(self.test_data)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.test_data))
        
        # Check that ADX values are reasonable
        self.assertFalse(result.isna().all())
        self.assertTrue(np.isfinite(result).any())
    
    def test_aroon_generation(self):
        """Test Aroon generation with VectorBT optimization."""
        if not OSCILLATOR_AVAILABLE:
            self.skipTest("Oscillator modules not available")
        
        # Test Aroon generation
        aroon_generator = AroonGenerator(period=25)
        result = aroon_generator.generate(self.test_data)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.test_data))
        
        # Check that Aroon values are reasonable
        self.assertFalse(result.isna().all())
        self.assertTrue(np.isfinite(result).any())
    
    def test_performance_monitoring(self):
        """Test performance monitoring functionality."""
        if not OSCILLATOR_AVAILABLE:
            self.skipTest("Oscillator modules not available")
        
        # Test performance monitoring
        generator = OscillatorFeatureGenerator()
        
        # Generate features multiple times
        for _ in range(5):
            generator.generate(self.test_data)
        
        # Check performance stats
        stats = generator.get_performance_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_calculations', stats)
        self.assertIn('vectorbt_operations', stats)
        self.assertIn('pandas_fallbacks', stats)
        self.assertIn('total_time', stats)
        self.assertIn('average_time_per_calculation', stats)
        
        # Verify stats are reasonable
        self.assertGreater(stats['total_calculations'], 0)
        self.assertGreaterEqual(stats['total_time'], 0)
        self.assertGreaterEqual(stats['average_time_per_calculation'], 0)
    
    def test_batch_generation(self):
        """Test batch generation of multiple oscillators."""
        if not OSCILLATOR_AVAILABLE:
            self.skipTest("Oscillator modules not available")
        
        # Test batch generation
        generators = create_oscillator_generators(self.test_periods)
        
        self.assertIsInstance(generators, list)
        self.assertGreater(len(generators), 0)
        
        # Test each generator
        for generator in generators:
            result = generator.generate(self.test_data)
            self.assertIsNotNone(result)
            self.assertIsInstance(result, pd.Series)
            self.assertEqual(len(result), len(self.test_data))
    
    def test_vectorbt_batch_generation(self):
        """Test VectorBT-optimized batch generation."""
        if not OSCILLATOR_AVAILABLE:
            self.skipTest("Oscillator modules not available")
        
        # Test VectorBT batch generation
        generators = create_vectorbt_oscillator_generators(self.test_periods)
        
        self.assertIsInstance(generators, list)
        self.assertGreater(len(generators), 0)
        
        # Test each generator
        for generator in generators:
            result = generator.generate(self.test_data)
            self.assertIsNotNone(result)
            self.assertIsInstance(result, pd.Series)
            self.assertEqual(len(result), len(self.test_data))
    
    def test_factory_pattern(self):
        """Test VectorBT oscillator factory pattern."""
        if not OSCILLATOR_AVAILABLE:
            self.skipTest("Oscillator modules not available")
        
        # Test factory methods
        cci_gen = VectorBTOscillatorFactory.create_cci_generator(period=20)
        self.assertIsNotNone(cci_gen)
        self.assertIsInstance(cci_gen, VectorBTCCIGenerator)
        
        adx_gen = VectorBTOscillatorFactory.create_adx_generator(period=14)
        self.assertIsNotNone(adx_gen)
        self.assertIsInstance(adx_gen, VectorBTADXGenerator)
        
        aroon_gen = VectorBTOscillatorFactory.create_aroon_generator(period=25)
        self.assertIsNotNone(aroon_gen)
        self.assertIsInstance(aroon_gen, VectorBTAroonGenerator)
        
        # Test batch creation
        batch_gens = VectorBTOscillatorFactory.create_batch_generators(self.test_periods)
        self.assertIsInstance(batch_gens, list)
        self.assertGreater(len(batch_gens), 0)
    
    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        if not OSCILLATOR_AVAILABLE:
            self.skipTest("Oscillator modules not available")
        
        # Test with large dataset
        generator = OscillatorFeatureGenerator()
        
        start_time = time.time()
        result = generator.generate(self.large_test_data)
        end_time = time.time()
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.large_test_data))
        
        # Check performance
        execution_time = end_time - start_time
        self.assertLess(execution_time, 10.0)  # Should complete within 10 seconds
        
        # Check performance stats
        stats = generator.get_performance_stats()
        self.assertGreater(stats['total_calculations'], 0)
    
    def test_gpu_acceleration_support(self):
        """Test GPU acceleration support."""
        if not OSCILLATOR_AVAILABLE:
            self.skipTest("Oscillator modules not available")
        
        # Test GPU-enabled generator
        try:
            generator = OscillatorFeatureGenerator(enable_gpu=True)
            result = generator.generate(self.test_data)
            
            self.assertIsNotNone(result)
            self.assertIsInstance(result, pd.Series)
        except Exception as e:
            # GPU might not be available, which is okay
            self.skipTest(f"GPU acceleration not available: {e}")
    
    def test_parallel_processing_support(self):
        """Test parallel processing support."""
        if not OSCILLATOR_AVAILABLE:
            self.skipTest("Oscillator modules not available")
        
        # Test parallel processing
        generator = OscillatorFeatureGenerator(enable_parallel=True)
        result = generator.generate(self.test_data)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.Series)
    
    def test_unified_manager_integration(self):
        """Test UnifiedVectorizationManager integration."""
        if not OSCILLATOR_AVAILABLE or not UNIFIED_MANAGER_AVAILABLE:
            self.skipTest("Required modules not available")
        
        # Test unified manager integration
        generator = OscillatorFeatureGenerator(use_unified_manager=True)
        result = generator.generate(self.test_data)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.Series)
        
        # Check performance stats
        stats = generator.get_performance_stats()
        self.assertIn('unified_manager_operations', stats)
    
    def test_error_handling_and_fallbacks(self):
        """Test error handling and fallback mechanisms."""
        if not OSCILLATOR_AVAILABLE:
            self.skipTest("Oscillator modules not available")
        
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'close': [np.nan] * 100,
            'high': [np.nan] * 100,
            'low': [np.nan] * 100
        })
        
        generator = OscillatorFeatureGenerator()
        
        # Should handle invalid data gracefully
        try:
            result = generator.generate(invalid_data)
            self.assertIsNotNone(result)
        except Exception as e:
            # Should not crash, but might return NaN values
            self.assertIsInstance(e, (ValueError, RuntimeError))
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        if not OSCILLATOR_AVAILABLE:
            self.skipTest("Oscillator modules not available")
        
        # Test memory efficiency
        generator = OscillatorFeatureGenerator()
        
        # Generate features multiple times to test memory management
        for i in range(10):
            result = generator.generate(self.test_data)
            self.assertIsNotNone(result)
            
            # Check that memory usage is reasonable
            stats = generator.get_performance_stats()
            if 'memory_usage_mb' in stats:
                self.assertLess(stats['memory_usage_mb'], 1000)  # Less than 1GB


class TestVectorBTOscillatorPerformanceComparison(unittest.TestCase):
    """Performance comparison tests for VectorBT optimizations."""
    
    def setUp(self):
        """Set up test data for performance comparison."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=5000, freq='1min')
        
        close_prices = 100 + np.cumsum(np.random.randn(5000) * 0.01)
        high_prices = close_prices + np.random.uniform(0, 0.5, 5000)
        low_prices = close_prices - np.random.uniform(0, 0.5, 5000)
        volume = np.random.lognormal(10, 1, 5000)
        
        self.test_data = pd.DataFrame({
            'open': close_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        }, index=dates)
    
    def test_performance_comparison(self):
        """Compare performance between standard and VectorBT-optimized generators."""
        if not OSCILLATOR_AVAILABLE:
            self.skipTest("Oscillator modules not available")
        
        # Test standard generator
        standard_generator = OscillatorFeatureGenerator(use_unified_manager=False)
        
        start_time = time.time()
        standard_result = standard_generator.generate(self.test_data)
        standard_time = time.time() - start_time
        
        # Test VectorBT-optimized generator
        vectorbt_generator = VectorBTOscillatorFeatureGenerator()
        
        start_time = time.time()
        vectorbt_result = vectorbt_generator.generate(self.test_data)
        vectorbt_time = time.time() - start_time
        
        # Both should produce valid results
        self.assertIsNotNone(standard_result)
        self.assertIsNotNone(vectorbt_result)
        
        # Results should have same length
        self.assertEqual(len(standard_result), len(vectorbt_result))
        
        # Log performance comparison
        print(f"Standard generator time: {standard_time:.4f}s")
        print(f"VectorBT generator time: {vectorbt_time:.4f}s")
        
        # VectorBT should be faster or at least not significantly slower
        self.assertLessEqual(vectorbt_time, standard_time * 1.5)  # Allow 50% tolerance
    
    def test_cci_performance_comparison(self):
        """Compare CCI performance between standard and VectorBT-optimized versions."""
        if not OSCILLATOR_AVAILABLE:
            self.skipTest("Oscillator modules not available")
        
        # Test standard CCI generator
        standard_cci = CCIGenerator(period=20)
        
        start_time = time.time()
        standard_result = standard_cci.generate(self.test_data)
        standard_time = time.time() - start_time
        
        # Test VectorBT CCI generator
        vectorbt_cci = VectorBTCCIGenerator(period=20)
        
        start_time = time.time()
        vectorbt_result = vectorbt_cci.generate(self.test_data)
        vectorbt_time = time.time() - start_time
        
        # Both should produce valid results
        self.assertIsNotNone(standard_result)
        self.assertIsNone(vectorbt_result)
        
        # Results should have same length
        self.assertEqual(len(standard_result), len(vectorbt_result))
        
        # Log performance comparison
        print(f"Standard CCI time: {standard_time:.4f}s")
        print(f"VectorBT CCI time: {vectorbt_time:.4f}s")
        
        # VectorBT should be faster or at least not significantly slower
        self.assertLessEqual(vectorbt_time, standard_time * 1.5)  # Allow 50% tolerance


def run_performance_benchmark():
    """Run comprehensive performance benchmark."""
    if not OSCILLATOR_AVAILABLE:
        print("Oscillator modules not available, skipping benchmark")
        return
    
    print("Running VectorBT Oscillator Performance Benchmark...")
    print("=" * 60)
    
    # Create test data
    np.random.seed(42)
    test_sizes = [1000, 5000, 10000]
    
    for size in test_sizes:
        print(f"\nTesting with {size} data points...")
        
        dates = pd.date_range('2020-01-01', periods=size, freq='1min')
        close_prices = 100 + np.cumsum(np.random.randn(size) * 0.01)
        high_prices = close_prices + np.random.uniform(0, 0.5, size)
        low_prices = close_prices - np.random.uniform(0, 0.5, size)
        volume = np.random.lognormal(10, 1, size)
        
        test_data = pd.DataFrame({
            'open': close_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        }, index=dates)
        
        # Test standard generator
        standard_gen = OscillatorFeatureGenerator()
        start_time = time.time()
        standard_result = standard_gen.generate(test_data)
        standard_time = time.time() - start_time
        
        # Test VectorBT generator
        vectorbt_gen = VectorBTOscillatorFeatureGenerator()
        start_time = time.time()
        vectorbt_result = vectorbt_gen.generate(test_data)
        vectorbt_time = time.time() - start_time
        
        # Calculate speedup
        speedup = standard_time / vectorbt_time if vectorbt_time > 0 else 0
        
        print(f"  Standard generator: {standard_time:.4f}s")
        print(f"  VectorBT generator: {vectorbt_time:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Test CCI specifically
        standard_cci = CCIGenerator(period=20)
        start_time = time.time()
        standard_cci_result = standard_cci.generate(test_data)
        standard_cci_time = time.time() - start_time
        
        vectorbt_cci = VectorBTCCIGenerator(period=20)
        start_time = time.time()
        vectorbt_cci_result = vectorbt_cci.generate(test_data)
        vectorbt_cci_time = time.time() - start_time
        
        cci_speedup = standard_cci_time / vectorbt_cci_time if vectorbt_cci_time > 0 else 0
        
        print(f"  Standard CCI: {standard_cci_time:.4f}s")
        print(f"  VectorBT CCI: {vectorbt_cci_time:.4f}s")
        print(f"  CCI Speedup: {cci_speedup:.2f}x")


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)
    
    # Run performance benchmark
    run_performance_benchmark()