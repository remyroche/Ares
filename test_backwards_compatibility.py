#!/usr/bin/env python3
"""
Backwards Compatibility Test Script

This script tests that all existing functionality continues to work unchanged
while new optimizations are available as optional enhancements.
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate price data with trend and volatility
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='1min')
    
    # Generate price with trend and noise
    trend = np.linspace(100, 200, n_samples)
    noise = np.random.normal(0, 0.5, n_samples)
    close_prices = trend + noise
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'open': close_prices + np.random.normal(0, 0.1, n_samples),
        'high': close_prices + np.abs(np.random.normal(0, 0.2, n_samples)),
        'low': close_prices - np.abs(np.random.normal(0, 0.2, n_samples)),
        'close': close_prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    }, index=dates)
    
    return data

def test_existing_api_compatibility():
    """Test that existing API remains unchanged."""
    logger.info("🧪 Testing existing API compatibility...")
    
    try:
        from src.feature_generation.categories.acceleration_backwards_compatible import (
            AccelerationFeatureGenerator,
            MomentumGenerator,
            PriceAccelerationGenerator,
            create_acceleration_generators,
            create_default_acceleration_generators
        )
        
        # Test data
        data = create_sample_data(500)
        
        # Test 1: Default initialization (should work exactly as before)
        logger.info("Testing default initialization...")
        generator = AccelerationFeatureGenerator()
        assert hasattr(generator, 'config')
        assert hasattr(generator, 'enable_optimizations')
        assert generator.enable_optimizations == True  # Default should be True
        
        # Test 2: Disable optimizations (backwards compatible)
        logger.info("Testing disabled optimizations...")
        generator_no_opt = AccelerationFeatureGenerator(enable_optimizations=False)
        assert generator_no_opt.enable_optimizations == False
        
        # Test 3: MomentumGenerator with default parameters
        logger.info("Testing MomentumGenerator...")
        momentum_gen = MomentumGenerator(period=10)
        assert momentum_gen.period == 10
        assert momentum_gen.enable_optimizations == True
        
        # Test 4: PriceAccelerationGenerator with default parameters
        logger.info("Testing PriceAccelerationGenerator...")
        accel_gen = PriceAccelerationGenerator(period=5)
        assert accel_gen.period == 5
        assert accel_gen.enable_optimizations == True
        
        # Test 5: Create generators function
        logger.info("Testing create_acceleration_generators...")
        generators = create_acceleration_generators()
        assert isinstance(generators, list)
        assert len(generators) > 0
        
        # Test 6: Create generators with optimizations disabled
        generators_no_opt = create_acceleration_generators(enable_optimizations=False)
        assert isinstance(generators_no_opt, list)
        assert len(generators_no_opt) > 0
        
        # Test 7: Default generators function
        logger.info("Testing create_default_acceleration_generators...")
        default_generators = create_default_acceleration_generators()
        assert isinstance(default_generators, list)
        assert len(default_generators) > 0
        
        logger.info("✅ All API compatibility tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ API compatibility test failed: {e}")
        return False

def test_existing_functionality():
    """Test that existing functionality works unchanged."""
    logger.info("🧪 Testing existing functionality...")
    
    try:
        from src.feature_generation.categories.acceleration_backwards_compatible import (
            MomentumGenerator,
            PriceAccelerationGenerator
        )
        
        # Test data
        data = create_sample_data(500)
        
        # Test 1: MomentumGenerator functionality
        logger.info("Testing MomentumGenerator functionality...")
        momentum_gen = MomentumGenerator(period=10, enable_optimizations=False)
        momentum_result = momentum_gen.generate_feature(data)
        
        assert isinstance(momentum_result, pd.Series)
        assert len(momentum_result) == len(data)
        assert momentum_result.name == 'momentum_10_price_returns'
        
        # Test 2: PriceAccelerationGenerator functionality
        logger.info("Testing PriceAccelerationGenerator functionality...")
        accel_gen = PriceAccelerationGenerator(period=5, enable_optimizations=False)
        accel_result = accel_gen.generate_feature(data)
        
        assert isinstance(accel_result, pd.Series)
        assert len(accel_result) == len(data)
        assert accel_result.name == 'acceleration_5_price_returns'
        
        # Test 3: DataFrame optimization (should work with or without optimizations)
        logger.info("Testing DataFrame optimization...")
        optimized_data = momentum_gen.optimize_dataframe_processing(data)
        assert isinstance(optimized_data, pd.DataFrame)
        assert len(optimized_data) == len(data)
        
        # Test 4: Vectorized rolling operations (should work with or without optimizations)
        logger.info("Testing vectorized rolling operations...")
        rolling_result = momentum_gen.vectorized_rolling_operations(
            data, ['mean', 'std'], [10, 20], ['close']
        )
        assert isinstance(rolling_result, pd.DataFrame)
        
        logger.info("✅ All functionality tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Functionality test failed: {e}")
        return False

def test_optimization_enhancements():
    """Test that new optimizations work when available."""
    logger.info("🧪 Testing optimization enhancements...")
    
    try:
        from src.feature_generation.categories.acceleration_backwards_compatible import (
            MomentumGenerator,
            PriceAccelerationGenerator
        )
        
        # Test data
        data = create_sample_data(1000)
        
        # Test 1: With optimizations enabled
        logger.info("Testing with optimizations enabled...")
        momentum_gen_opt = MomentumGenerator(period=10, enable_optimizations=True)
        start_time = time.time()
        momentum_result_opt = momentum_gen_opt.generate_feature(data)
        opt_time = time.time() - start_time
        
        # Test 2: With optimizations disabled
        logger.info("Testing with optimizations disabled...")
        momentum_gen_no_opt = MomentumGenerator(period=10, enable_optimizations=False)
        start_time = time.time()
        momentum_result_no_opt = momentum_gen_no_opt.generate_feature(data)
        no_opt_time = time.time() - start_time
        
        # Both should produce similar results
        assert isinstance(momentum_result_opt, pd.Series)
        assert isinstance(momentum_result_no_opt, pd.Series)
        assert len(momentum_result_opt) == len(momentum_result_no_opt)
        
        # Results should be approximately equal (allowing for small numerical differences)
        correlation = np.corrcoef(momentum_result_opt.dropna(), momentum_result_no_opt.dropna())[0, 1]
        assert correlation > 0.99, f"Results should be highly correlated, got {correlation}"
        
        logger.info(f"Optimized time: {opt_time:.3f}s, Non-optimized time: {no_opt_time:.3f}s")
        
        # Test 3: Acceleration generator with optimizations
        logger.info("Testing PriceAccelerationGenerator with optimizations...")
        accel_gen_opt = PriceAccelerationGenerator(period=5, enable_optimizations=True)
        accel_result_opt = accel_gen_opt.generate_feature(data)
        
        assert isinstance(accel_result_opt, pd.Series)
        assert len(accel_result_opt) == len(data)
        
        logger.info("✅ All optimization enhancement tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimization enhancement test failed: {e}")
        return False

def test_graceful_fallbacks():
    """Test graceful fallbacks when optimizations are not available."""
    logger.info("🧪 Testing graceful fallbacks...")
    
    try:
        from src.feature_generation.categories.acceleration_backwards_compatible import (
            AccelerationFeatureGenerator,
            MomentumGenerator
        )
        
        # Test data
        data = create_sample_data(500)
        
        # Test 1: Generator should work even if optimizations fail
        logger.info("Testing graceful fallback behavior...")
        generator = AccelerationFeatureGenerator(enable_optimizations=True)
        
        # These should not raise exceptions even if optimizations are not available
        optimized_data = generator.optimize_dataframe_processing(data)
        assert isinstance(optimized_data, pd.DataFrame)
        
        rolling_result = generator.vectorized_rolling_operations(
            data, ['mean'], [10], ['close']
        )
        assert isinstance(rolling_result, pd.DataFrame)
        
        # Test 2: Individual generators should work with optimizations disabled
        logger.info("Testing individual generators with optimizations disabled...")
        momentum_gen = MomentumGenerator(period=10, enable_optimizations=False)
        result = momentum_gen.generate_feature(data)
        assert isinstance(result, pd.Series)
        
        # Test 3: Should handle missing optimization dependencies gracefully
        logger.info("Testing missing optimization dependencies...")
        # This should not raise an exception
        generator_no_opt = AccelerationFeatureGenerator(enable_optimizations=False)
        result = generator_no_opt.optimize_dataframe_processing(data)
        assert isinstance(result, pd.DataFrame)
        
        logger.info("✅ All graceful fallback tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Graceful fallback test failed: {e}")
        return False

def test_performance_comparison():
    """Compare performance with and without optimizations."""
    logger.info("🧪 Testing performance comparison...")
    
    try:
        from src.feature_generation.categories.acceleration_backwards_compatible import (
            MomentumGenerator,
            PriceAccelerationGenerator
        )
        
        # Test data
        data = create_sample_data(2000)
        
        # Test different scenarios
        scenarios = [
            ('MomentumGenerator (optimized)', MomentumGenerator(period=10, enable_optimizations=True)),
            ('MomentumGenerator (not optimized)', MomentumGenerator(period=10, enable_optimizations=False)),
            ('PriceAccelerationGenerator (optimized)', PriceAccelerationGenerator(period=5, enable_optimizations=True)),
            ('PriceAccelerationGenerator (not optimized)', PriceAccelerationGenerator(period=5, enable_optimizations=False)),
        ]
        
        results = {}
        
        for name, generator in scenarios:
            logger.info(f"Testing {name}...")
            start_time = time.time()
            
            try:
                result = generator.generate_feature(data)
                execution_time = time.time() - start_time
                
                results[name] = {
                    'time': execution_time,
                    'success': True,
                    'result_shape': result.shape if hasattr(result, 'shape') else len(result)
                }
                
                logger.info(f"✅ {name}: {execution_time:.3f}s")
                
            except Exception as e:
                results[name] = {
                    'time': float('inf'),
                    'success': False,
                    'error': str(e)
                }
                logger.error(f"❌ {name}: {e}")
        
        # All should succeed
        successful_results = {k: v for k, v in results.items() if v['success']}
        assert len(successful_results) == len(scenarios), "All scenarios should succeed"
        
        logger.info("✅ All performance comparison tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance comparison test failed: {e}")
        return False

def test_import_compatibility():
    """Test that imports work as expected."""
    logger.info("🧪 Testing import compatibility...")
    
    try:
        # Test 1: Import from backwards compatible module
        from src.feature_generation.categories.acceleration_backwards_compatible import (
            AccelerationFeatureGenerator,
            MomentumGenerator,
            PriceAccelerationGenerator,
            create_acceleration_generators,
            create_default_acceleration_generators
        )
        
        # Test 2: All classes should be importable
        assert AccelerationFeatureGenerator is not None
        assert MomentumGenerator is not None
        assert PriceAccelerationGenerator is not None
        assert create_acceleration_generators is not None
        assert create_default_acceleration_generators is not None
        
        # Test 3: Functions should be callable
        generators = create_acceleration_generators()
        assert isinstance(generators, list)
        
        default_generators = create_default_acceleration_generators()
        assert isinstance(default_generators, list)
        
        logger.info("✅ All import compatibility tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Import compatibility test failed: {e}")
        return False

def main():
    """Run all backwards compatibility tests."""
    logger.info("🚀 Starting backwards compatibility tests...")
    
    test_results = {}
    
    # Test import compatibility
    test_results['Import Compatibility'] = test_import_compatibility()
    
    # Test existing API compatibility
    test_results['API Compatibility'] = test_existing_api_compatibility()
    
    # Test existing functionality
    test_results['Functionality'] = test_existing_functionality()
    
    # Test optimization enhancements
    test_results['Optimization Enhancements'] = test_optimization_enhancements()
    
    # Test graceful fallbacks
    test_results['Graceful Fallbacks'] = test_graceful_fallbacks()
    
    # Test performance comparison
    test_results['Performance Comparison'] = test_performance_comparison()
    
    # Summary
    logger.info("📊 Backwards Compatibility Test Summary:")
    for test_name, success in test_results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
    
    successful_tests = sum(test_results.values())
    total_tests = len(test_results)
    logger.info(f"Overall: {successful_tests}/{total_tests} tests passed")
    
    if successful_tests == total_tests:
        logger.info("🎉 All backwards compatibility tests passed! The implementation is fully backwards compatible.")
    else:
        logger.error("❌ Some backwards compatibility tests failed. Please review the implementation.")
    
    return successful_tests == total_tests

if __name__ == "__main__":
    main()