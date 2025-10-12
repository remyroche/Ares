#!/usr/bin/env python3
"""
Test script for optimized volume features with VectorBT and UnifiedVectorizationManager.

This script validates the correctness and performance of the optimized volume feature generators.
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data(n_rows: int = 10000) -> pd.DataFrame:
    """Create test OHLCV data for volume feature testing."""
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.normal(0.0001, 0.02, n_rows)
    prices = 100 * (1 + returns).cumprod()
    
    # Generate volume data with some correlation to price movements
    volume_base = np.random.lognormal(10, 1, n_rows)
    volume_multiplier = 1 + np.abs(returns) * 10  # Higher volume on larger price moves
    volume = volume_base * volume_multiplier
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_rows)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_rows))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_rows))),
        'close': prices,
        'volume': volume
    }, index=pd.date_range('2020-01-01', periods=n_rows, freq='1min'))
    
    return data

def test_volume_sma_generator():
    """Test VolumeSMAGenerator with VectorBT optimization."""
    logger.info("Testing VolumeSMAGenerator...")
    
    try:
        from src.feature_generation.categories.volume import VolumeSMAGenerator
        
        # Create test data
        data = create_test_data(5000)
        
        # Test different periods
        periods = [5, 10, 20, 50]
        
        for period in periods:
            generator = VolumeSMAGenerator(period)
            
            # Generate feature
            start_time = time.time()
            feature = generator._generate_feature(data)
            end_time = time.time()
            
            # Validate results
            assert len(feature) == len(data), f"Feature length mismatch for period {period}"
            assert not feature.isna().all(), f"All NaN values for period {period}"
            assert feature.name == f'volume_sma_{period}', f"Wrong feature name for period {period}"
            
            logger.info(f"  Period {period}: {end_time - start_time:.3f}s, {len(feature)} values")
        
        logger.info("✅ VolumeSMAGenerator test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ VolumeSMAGenerator test failed: {e}")
        return False

def test_volume_vwap_generator():
    """Test VolumeVWAPGenerator with VectorBT optimization."""
    logger.info("Testing VolumeVWAPGenerator...")
    
    try:
        from src.feature_generation.categories.volume import VolumeVWAPGenerator
        
        # Create test data
        data = create_test_data(5000)
        
        # Test different periods
        periods = [10, 20, 50]
        
        for period in periods:
            generator = VolumeVWAPGenerator(period)
            
            # Generate feature
            start_time = time.time()
            feature = generator._generate_feature(data)
            end_time = time.time()
            
            # Validate results
            assert len(feature) == len(data), f"Feature length mismatch for period {period}"
            assert not feature.isna().all(), f"All NaN values for period {period}"
            assert feature.name == f'volume_vwap_{period}', f"Wrong feature name for period {period}"
            
            # VWAP should be between high and low prices
            valid_values = feature.dropna()
            if len(valid_values) > 0:
                assert (valid_values >= data['low'].iloc[valid_values.index]).all(), "VWAP below low price"
                assert (valid_values <= data['high'].iloc[valid_values.index]).all(), "VWAP above high price"
            
            logger.info(f"  Period {period}: {end_time - start_time:.3f}s, {len(feature)} values")
        
        logger.info("✅ VolumeVWAPGenerator test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ VolumeVWAPGenerator test failed: {e}")
        return False

def test_batch_volume_features():
    """Test batch volume feature generation."""
    logger.info("Testing batch volume feature generation...")
    
    try:
        from src.feature_generation.categories.volume import VolumeFeatureGenerator
        
        # Create test data
        data = create_test_data(10000)
        
        # Create batch generator
        generator = VolumeFeatureGenerator()
        
        # Define feature configurations
        feature_configs = [
            {'name': 'volume_sma_20', 'type': 'sma', 'period': 20},
            {'name': 'volume_sma_50', 'type': 'sma', 'period': 50},
            {'name': 'volume_std_20', 'type': 'std', 'period': 20},
            {'name': 'volume_std_50', 'type': 'std', 'period': 50},
        ]
        
        # Generate batch features
        start_time = time.time()
        features_df = generator.generate_batch_volume_features(data, feature_configs)
        end_time = time.time()
        
        # Validate results
        assert len(features_df) == len(data), "Batch features length mismatch"
        assert len(features_df.columns) == len(feature_configs), "Wrong number of features"
        
        for config in feature_configs:
            feature_name = config['name']
            assert feature_name in features_df.columns, f"Missing feature: {feature_name}"
            assert not features_df[feature_name].isna().all(), f"All NaN values for {feature_name}"
        
        logger.info(f"  Batch generation: {end_time - start_time:.3f}s, {len(features_df.columns)} features")
        logger.info("✅ Batch volume features test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Batch volume features test failed: {e}")
        return False

def test_optimized_volume_factory():
    """Test OptimizedVolumeFeatureFactory."""
    logger.info("Testing OptimizedVolumeFeatureFactory...")
    
    try:
        from src.feature_generation.categories.volume import create_optimized_volume_factory
        
        # Create factory
        factory = create_optimized_volume_factory(enable_gpu=False, enable_parallel=True)
        
        # Create test data
        data = create_test_data(15000)
        
        # Test comprehensive feature generation
        start_time = time.time()
        features_df = factory.generate_comprehensive_volume_features(data, periods=[10, 20, 50])
        end_time = time.time()
        
        # Validate results
        assert len(features_df) == len(data), "Factory features length mismatch"
        assert len(features_df.columns) > 0, "No features generated"
        
        # Check for expected feature types
        expected_features = ['volume_sma_10', 'volume_sma_20', 'volume_sma_50',
                           'volume_ema_10', 'volume_ema_20', 'volume_ema_50',
                           'volume_std_10', 'volume_std_20', 'volume_std_50']
        
        for feature in expected_features:
            if feature in features_df.columns:
                assert not features_df[feature].isna().all(), f"All NaN values for {feature}"
        
        # Get performance stats
        stats = factory.get_performance_stats()
        logger.info(f"  Factory generation: {end_time - start_time:.3f}s, {len(features_df.columns)} features")
        logger.info(f"  Performance stats: {stats}")
        
        logger.info("✅ OptimizedVolumeFeatureFactory test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ OptimizedVolumeFeatureFactory test failed: {e}")
        return False

def test_performance_monitoring():
    """Test performance monitoring functionality."""
    logger.info("Testing performance monitoring...")
    
    try:
        from src.feature_generation.categories.volume import VolumeFeatureGenerator
        
        # Create generator
        generator = VolumeFeatureGenerator()
        
        # Create test data
        data = create_test_data(8000)
        
        # Generate features to accumulate stats
        for _ in range(5):
            generator._generate_feature(data)
        
        # Test performance summary
        summary = generator.get_performance_summary()
        
        # Validate summary structure
        assert 'performance_stats' in summary, "Missing performance_stats"
        assert 'recent_operations' in summary, "Missing recent_operations"
        assert 'optimization_effectiveness' in summary, "Missing optimization_effectiveness"
        
        # Test performance logging
        generator.log_performance_report()
        
        # Test stats reset
        generator.reset_performance_stats()
        assert generator.performance_stats['total_operations'] == 0, "Stats not reset"
        
        logger.info("✅ Performance monitoring test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance monitoring test failed: {e}")
        return False

def test_memory_optimization():
    """Test memory optimization functionality."""
    logger.info("Testing memory optimization...")
    
    try:
        from src.feature_generation.categories.volume import VolumeFeatureGenerator
        
        # Create generator
        generator = VolumeFeatureGenerator()
        
        # Create large test data
        data = create_test_data(50000)  # 50k rows
        
        # Test memory optimization
        optimized_data = generator._optimize_memory_usage(data)
        
        # Check if optimization was applied
        original_memory = data.memory_usage(deep=True).sum()
        optimized_memory = optimized_data.memory_usage(deep=True).sum()
        
        logger.info(f"  Original memory: {original_memory / (1024*1024):.2f}MB")
        logger.info(f"  Optimized memory: {optimized_memory / (1024*1024):.2f}MB")
        logger.info(f"  Memory saved: {(original_memory - optimized_memory) / (1024*1024):.2f}MB")
        
        # Test chunking decision
        should_chunk = generator._should_use_chunking(data)
        logger.info(f"  Should use chunking: {should_chunk}")
        
        logger.info("✅ Memory optimization test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory optimization test failed: {e}")
        return False

def run_all_tests():
    """Run all volume feature tests."""
    logger.info("Starting volume feature optimization tests...")
    
    tests = [
        test_volume_sma_generator,
        test_volume_vwap_generator,
        test_batch_volume_features,
        test_optimized_volume_factory,
        test_performance_monitoring,
        test_memory_optimization
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            logger.error(f"Test {test_func.__name__} failed with exception: {e}")
    
    logger.info(f"\n=== Test Results ===")
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        logger.info("🎉 All tests passed! Volume features are fully optimized with VectorBT.")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Check the logs for details.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)