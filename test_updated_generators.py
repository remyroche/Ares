#!/usr/bin/env python3
"""
Test script to validate that updated feature generators work correctly
with the new VectorBT optimization components.
"""

import sys
import os
import numpy as np
import pandas as pd
import time
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data(n_samples=1000):
    """Create test data for feature generation."""
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.normal(0.001, 0.02, n_samples)
    prices = 100 * (1 + returns).cumprod()
    
    # Generate OHLCV data
    high_low_range = np.random.uniform(0.5, 2.0, n_samples)
    highs = prices + high_low_range
    lows = prices - high_low_range
    
    data = pd.DataFrame({
        'close': prices,
        'high': highs,
        'low': lows,
        'open': np.roll(prices, 1),  # Previous close as open
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Ensure high >= low and high >= close
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['close'])
    
    return data

def test_volatility_generator():
    """Test the updated volatility generator."""
    logger.info("Testing VolatilityFeatureGenerator...")
    
    try:
        from src.feature_generation.categories.volatility import VolatilityFeatureGenerator
        
        # Create test data
        data = create_test_data(1000)
        
        # Test different periods
        periods = [10, 20, 50]
        
        for period in periods:
            logger.info(f"  Testing period {period}...")
            
            # Create generator
            generator = VolatilityFeatureGenerator(period=period, enable_gpu=False, enable_parallel=True)
            
            # Generate features
            start_time = time.time()
            features = generator.generate_features(data)
            generation_time = time.time() - start_time
            
            # Validate results
            assert isinstance(features, pd.DataFrame), f"Expected DataFrame, got {type(features)}"
            assert len(features) == len(data), f"Expected {len(data)} rows, got {len(features)}"
            assert not features.empty, "Features should not be empty"
            
            # Check for expected columns
            expected_cols = [f'volatility_{period}']
            for col in expected_cols:
                assert col in features.columns, f"Expected column {col} not found"
            
            # Check for reasonable values
            volatility_col = f'volatility_{period}'
            if volatility_col in features.columns:
                vol_values = features[volatility_col].dropna()
                assert len(vol_values) > 0, "Volatility values should not be all NaN"
                assert vol_values.min() >= 0, "Volatility should be non-negative"
                assert vol_values.max() < 1.0, "Volatility should be reasonable (< 1.0)"
            
            logger.info(f"    ✓ Period {period}: {len(features.columns)} features in {generation_time:.3f}s")
            
            # Test performance reporting
            if hasattr(generator, 'get_performance_report'):
                report = generator.get_performance_report()
                logger.info(f"    Performance report: {report}")
        
        logger.info("  ✓ VolatilityFeatureGenerator test passed")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ VolatilityFeatureGenerator test failed: {e}")
        return False

def test_momentum_generator():
    """Test the updated momentum generator."""
    logger.info("Testing MomentumFeatureGenerator...")
    
    try:
        from src.feature_generation.categories.momentum import MomentumFeatureGenerator
        
        # Create test data
        data = create_test_data(1000)
        
        # Create generator
        generator = MomentumFeatureGenerator(enable_gpu=False, enable_parallel=True)
        
        # Generate features
        start_time = time.time()
        features = generator.generate_features(data)
        generation_time = time.time() - start_time
        
        # Validate results
        assert isinstance(features, pd.DataFrame), f"Expected DataFrame, got {type(features)}"
        assert len(features) == len(data), f"Expected {len(data)} rows, got {len(features)}"
        assert not features.empty, "Features should not be empty"
        
        logger.info(f"  ✓ MomentumFeatureGenerator: {len(features.columns)} features in {generation_time:.3f}s")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ MomentumFeatureGenerator test failed: {e}")
        return False

def test_volume_generator():
    """Test the updated volume generator."""
    logger.info("Testing VolumeFeatureGenerator...")
    
    try:
        from src.feature_generation.categories.volume import VolumeFeatureGenerator
        
        # Create test data
        data = create_test_data(1000)
        
        # Create generator
        generator = VolumeFeatureGenerator(enable_gpu=False, enable_parallel=True)
        
        # Generate features
        start_time = time.time()
        features = generator.generate_features(data)
        generation_time = time.time() - start_time
        
        # Validate results
        assert isinstance(features, pd.DataFrame), f"Expected DataFrame, got {type(features)}"
        assert len(features) == len(data), f"Expected {len(data)} rows, got {len(features)}"
        assert not features.empty, "Features should not be empty"
        
        logger.info(f"  ✓ VolumeFeatureGenerator: {len(features.columns)} features in {generation_time:.3f}s")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ VolumeFeatureGenerator test failed: {e}")
        return False

def test_oscillator_generator():
    """Test the updated oscillator generator."""
    logger.info("Testing OscillatorFeatureGenerator...")
    
    try:
        from src.feature_generation.categories.oscillator import OscillatorFeatureGenerator
        
        # Create test data
        data = create_test_data(1000)
        
        # Create generator
        generator = OscillatorFeatureGenerator(enable_gpu=False, enable_parallel=True)
        
        # Generate features
        start_time = time.time()
        features = generator.generate_features(data)
        generation_time = time.time() - start_time
        
        # Validate results
        assert isinstance(features, pd.DataFrame), f"Expected DataFrame, got {type(features)}"
        assert len(features) == len(data), f"Expected {len(data)} rows, got {len(features)}"
        assert not features.empty, "Features should not be empty"
        
        logger.info(f"  ✓ OscillatorFeatureGenerator: {len(features.columns)} features in {generation_time:.3f}s")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ OscillatorFeatureGenerator test failed: {e}")
        return False

def test_trend_generator():
    """Test the updated trend generator."""
    logger.info("Testing TrendFeatureGenerator...")
    
    try:
        from src.feature_generation.categories.trend import TrendFeatureGenerator
        
        # Create test data
        data = create_test_data(1000)
        
        # Create generator
        generator = TrendFeatureGenerator(enable_gpu=False, enable_parallel=True)
        
        # Generate features
        start_time = time.time()
        features = generator.generate_features(data)
        generation_time = time.time() - start_time
        
        # Validate results
        assert isinstance(features, pd.DataFrame), f"Expected DataFrame, got {type(features)}"
        assert len(features) == len(data), f"Expected {len(data)} rows, got {len(features)}"
        assert not features.empty, "Features should not be empty"
        
        logger.info(f"  ✓ TrendFeatureGenerator: {len(features.columns)} features in {generation_time:.3f}s")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ TrendFeatureGenerator test failed: {e}")
        return False

def test_optimization_components():
    """Test the new optimization components directly."""
    logger.info("Testing optimization components...")
    
    try:
        from src.feature_generation.utils.consolidated_rolling_optimizer import get_global_rolling_optimizer
        from src.feature_generation.utils.statistical_calculations_optimizer import get_global_statistical_optimizer
        from src.feature_generation.utils.unified_optimization_wrapper import create_unified_optimizer
        
        # Test rolling optimizer
        rolling_optimizer = get_global_rolling_optimizer()
        assert rolling_optimizer is not None, "Rolling optimizer should be available"
        
        # Test statistical optimizer
        statistical_optimizer = get_global_statistical_optimizer()
        assert statistical_optimizer is not None, "Statistical optimizer should be available"
        
        # Test unified optimizer
        unified_optimizer = create_unified_optimizer()
        assert unified_optimizer is not None, "Unified optimizer should be available"
        
        logger.info("  ✓ All optimization components available")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Optimization components test failed: {e}")
        return False

def run_performance_comparison():
    """Run a performance comparison between old and new implementations."""
    logger.info("Running performance comparison...")
    
    try:
        from src.feature_generation.categories.volatility import VolatilityFeatureGenerator
        
        # Create test data
        data = create_test_data(5000)
        
        # Test new optimized version
        generator = VolatilityFeatureGenerator(period=20, enable_gpu=False, enable_parallel=True)
        
        # Warm up
        generator.generate_features(data.head(100))
        
        # Benchmark
        start_time = time.time()
        for _ in range(10):
            features = generator.generate_features(data)
        optimized_time = time.time() - start_time
        
        # Test pandas baseline
        start_time = time.time()
        for _ in range(10):
            returns = data['close'].pct_change().dropna()
            volatility = returns.rolling(window=20).std()
        pandas_time = time.time() - start_time
        
        speedup = pandas_time / optimized_time
        logger.info(f"  Optimized time: {optimized_time:.4f}s")
        logger.info(f"  Pandas time: {pandas_time:.4f}s")
        logger.info(f"  Speedup: {speedup:.2f}x")
        
        if speedup > 1.5:  # Expect at least 1.5x speedup
            logger.info("  ✓ Performance improvement achieved")
            return True
        else:
            logger.warning("  ⚠ Performance improvement less than expected")
            return False
        
    except Exception as e:
        logger.error(f"  ✗ Performance comparison failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting VectorBT Optimization Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Optimization Components", test_optimization_components),
        ("Volatility Generator", test_volatility_generator),
        ("Momentum Generator", test_momentum_generator),
        ("Volume Generator", test_volume_generator),
        ("Oscillator Generator", test_oscillator_generator),
        ("Trend Generator", test_trend_generator),
        ("Performance Comparison", run_performance_comparison),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Test Results Summary")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All tests passed! VectorBT optimizations are working correctly.")
        return True
    else:
        logger.error("⚠️ Some tests failed. Please check the logs above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)