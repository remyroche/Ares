#!/usr/bin/env python3
"""
Test script to validate VectorBT optimizations in volume features.

This script tests the enhanced volume feature generators to ensure
VectorBT optimizations are working correctly.
"""

import sys
import os
import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create test OHLCV data for volume feature testing."""
    np.random.seed(42)
    
    # Generate price data
    returns = np.random.normal(0.001, 0.02, n_samples)
    prices = 100 * (1 + returns).cumprod()
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    }, index=pd.date_range('2020-01-01', periods=n_samples, freq='1min'))
    
    return data

def test_volume_sma_generator():
    """Test VolumeSMAGenerator with VectorBT optimizations."""
    try:
        from src.feature_generation.categories.volume import VolumeSMAGenerator
        
        logger.info("Testing VolumeSMAGenerator...")
        
        # Create test data
        data = create_test_data(1000)
        
        # Create generator
        generator = VolumeSMAGenerator(period=20)
        
        # Test feature generation
        start_time = time.time()
        features = generator._generate_feature(data)
        end_time = time.time()
        
        # Validate results
        assert len(features) == len(data), f"Feature length mismatch: {len(features)} vs {len(data)}"
        assert not features.isna().all(), "All features are NaN"
        assert features.name == 'volume_sma_20', f"Unexpected feature name: {features.name}"
        
        logger.info(f"✅ VolumeSMAGenerator test passed - Time: {end_time - start_time:.4f}s")
        return True
        
    except Exception as e:
        logger.error(f"❌ VolumeSMAGenerator test failed: {e}")
        return False

def test_volume_feature_generator():
    """Test VolumeFeatureGenerator with unified optimizations."""
    try:
        from src.feature_generation.categories.volume import VolumeFeatureGenerator
        
        logger.info("Testing VolumeFeatureGenerator...")
        
        # Create test data
        data = create_test_data(1000)
        
        # Create generator
        generator = VolumeFeatureGenerator()
        
        # Test feature generation
        start_time = time.time()
        features = generator.generate_features(data)
        end_time = time.time()
        
        # Validate results
        assert isinstance(features, pd.DataFrame), "Features should be a DataFrame"
        assert len(features) == len(data), f"Feature length mismatch: {len(features)} vs {len(data)}"
        assert len(features.columns) > 0, "No features generated"
        
        # Check performance stats
        stats = generator.performance_stats
        logger.info(f"Performance stats: {stats}")
        
        logger.info(f"✅ VolumeFeatureGenerator test passed - Time: {end_time - start_time:.4f}s")
        return True
        
    except Exception as e:
        logger.error(f"❌ VolumeFeatureGenerator test failed: {e}")
        return False

def test_regime_volume_generator():
    """Test RegimeVolumeFeatureGenerator with VectorBT optimizations."""
    try:
        from src.feature_generation.categories.regime_volume import RegimeVolumeFeatureGenerator
        
        logger.info("Testing RegimeVolumeFeatureGenerator...")
        
        # Create test data
        data = create_test_data(1000)
        
        # Create generator
        generator = RegimeVolumeFeatureGenerator()
        
        # Test feature generation
        start_time = time.time()
        features = generator.generate_features(data)
        end_time = time.time()
        
        # Validate results
        assert isinstance(features, dict), "Features should be a dictionary"
        assert len(features) > 0, "No features generated"
        
        # Check some specific features
        for feature_name, feature_values in features.items():
            assert isinstance(feature_values, np.ndarray), f"Feature {feature_name} should be numpy array"
            assert len(feature_values) == len(data), f"Feature {feature_name} length mismatch"
        
        logger.info(f"✅ RegimeVolumeFeatureGenerator test passed - Time: {end_time - start_time:.4f}s")
        return True
        
    except Exception as e:
        logger.error(f"❌ RegimeVolumeFeatureGenerator test failed: {e}")
        return False

def test_vectorbt_rolling_optimizer():
    """Test VectorBTRollingOptimizer directly."""
    try:
        from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
        
        logger.info("Testing VectorBTRollingOptimizer...")
        
        # Create test data
        data = create_test_data(1000)
        volume = data['volume']
        
        # Create optimizer
        optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        
        # Test rolling operations
        start_time = time.time()
        
        # Test rolling mean
        mean_result = optimizer.rolling_mean(volume, window=20)
        assert len(mean_result) == len(volume), "Rolling mean length mismatch"
        
        # Test rolling std
        std_result = optimizer.rolling_std(volume, window=20)
        assert len(std_result) == len(volume), "Rolling std length mismatch"
        
        # Test rolling correlation
        corr_result = optimizer.rolling_corr(volume, data['close'], window=20)
        assert len(corr_result) == len(volume), "Rolling correlation length mismatch"
        
        end_time = time.time()
        
        # Check performance stats
        stats = optimizer.get_performance_stats()
        logger.info(f"Optimizer stats: {stats}")
        
        logger.info(f"✅ VectorBTRollingOptimizer test passed - Time: {end_time - start_time:.4f}s")
        return True
        
    except Exception as e:
        logger.error(f"❌ VectorBTRollingOptimizer test failed: {e}")
        return False

def test_unified_vectorization_manager():
    """Test UnifiedVectorizationManager integration."""
    try:
        from src.utils.ml_common.unified_vectorization_manager import get_unified_vectorization_manager
        
        logger.info("Testing UnifiedVectorizationManager...")
        
        # Create test data
        data = create_test_data(1000)
        
        # Create manager
        manager = get_unified_vectorization_manager()
        
        # Test basic functionality
        assert manager is not None, "UnifiedVectorizationManager should be available"
        
        # Check optimization stats
        stats = manager.optimization_stats
        logger.info(f"Unified manager stats: {stats}")
        
        logger.info("✅ UnifiedVectorizationManager test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ UnifiedVectorizationManager test failed: {e}")
        return False

def run_performance_benchmark():
    """Run performance benchmark comparing different optimization strategies."""
    try:
        from src.feature_generation.categories.volume import VolumeSMAGenerator
        from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
        
        logger.info("Running performance benchmark...")
        
        # Create test data
        data = create_test_data(5000)
        volume = data['volume']
        
        # Test pandas rolling
        start_time = time.time()
        pandas_result = volume.rolling(window=20).mean()
        pandas_time = time.time() - start_time
        
        # Test VectorBT rolling optimizer
        optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        start_time = time.time()
        vectorbt_result = optimizer.rolling_mean(volume, window=20)
        vectorbt_time = time.time() - start_time
        
        # Test volume generator
        generator = VolumeSMAGenerator(period=20)
        start_time = time.time()
        generator_result = generator._generate_feature(data)
        generator_time = time.time() - start_time
        
        # Calculate speedup
        pandas_speedup = pandas_time / pandas_time
        vectorbt_speedup = pandas_time / vectorbt_time
        generator_speedup = pandas_time / generator_time
        
        logger.info(f"Performance Results:")
        logger.info(f"  Pandas rolling: {pandas_time:.4f}s (1.00x)")
        logger.info(f"  VectorBT rolling: {vectorbt_time:.4f}s ({vectorbt_speedup:.2f}x)")
        logger.info(f"  Volume generator: {generator_time:.4f}s ({generator_speedup:.2f}x)")
        
        # Validate results are similar
        pandas_mean = pandas_result.mean()
        vectorbt_mean = vectorbt_result.mean()
        generator_mean = generator_result.mean()
        
        assert abs(pandas_mean - vectorbt_mean) < 1e-10, "VectorBT result differs from pandas"
        assert abs(pandas_mean - generator_mean) < 1e-10, "Generator result differs from pandas"
        
        logger.info("✅ Performance benchmark completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance benchmark failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting VectorBT optimization tests for volume features...")
    
    tests = [
        test_vectorbt_rolling_optimizer,
        test_unified_vectorization_manager,
        test_volume_sma_generator,
        test_volume_feature_generator,
        test_regime_volume_generator,
        run_performance_benchmark
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} failed with exception: {e}")
    
    logger.info(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! VectorBT optimizations are working correctly.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the logs for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)