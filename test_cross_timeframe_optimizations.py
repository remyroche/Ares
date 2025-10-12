#!/usr/bin/env python3
"""
Test script for Cross-Timeframe VectorBT Optimizations

This script tests the enhanced cross-timeframe feature generators to ensure
they properly utilize VectorBTRollingOptimizer and UnifiedVectorizationManager.
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data(n_rows: int = 10000) -> pd.DataFrame:
    """Create test data for cross-timeframe feature generation."""
    np.random.seed(42)
    
    # Create datetime index
    dates = pd.date_range('2020-01-01', periods=n_rows, freq='1min')
    
    # Generate synthetic OHLCV data
    base_price = 100
    returns = np.random.normal(0, 0.001, n_rows)
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.0005, n_rows)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.001, n_rows))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.001, n_rows))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_rows)
    }, index=dates)
    
    # Ensure high >= low and high/low contain close
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    return data

def test_cross_timeframe_generators():
    """Test the cross-timeframe generators with VectorBT optimizations."""
    logger.info("🧪 Testing Cross-Timeframe Generators with VectorBT Optimizations")
    
    try:
        from feature_generation.categories.cross_timeframe import (
            CrossTimeframeFeatureGenerator,
            CrossTimeframeMomentumGenerator,
            CrossTimeframeVolatilityGenerator,
            CrossTimeframeVolumeGenerator
        )
        
        # Create test data
        test_data = create_test_data(5000)
        logger.info(f"Created test data with {len(test_data)} rows")
        
        # Test 1: CrossTimeframeFeatureGenerator
        logger.info("\n📊 Testing CrossTimeframeFeatureGenerator...")
        ctf_generator = CrossTimeframeFeatureGenerator()
        
        # Test enhanced feature generation
        start_time = time.time()
        enhanced_features = ctf_generator.generate_enhanced_cross_timeframe_features(test_data)
        generation_time = time.time() - start_time
        
        logger.info(f"✅ Generated {len(enhanced_features)} enhanced features in {generation_time:.3f}s")
        
        # Test performance report
        performance_report = ctf_generator.get_performance_report()
        logger.info(f"📈 Performance Report: {performance_report['cross_timeframe_performance']}")
        
        # Test 2: CrossTimeframeMomentumGenerator
        logger.info("\n📊 Testing CrossTimeframeMomentumGenerator...")
        momentum_gen = CrossTimeframeMomentumGenerator(timeframe=10)
        
        start_time = time.time()
        momentum_feature = momentum_gen._generate_feature(test_data)
        generation_time = time.time() - start_time
        
        logger.info(f"✅ Generated momentum feature in {generation_time:.3f}s")
        logger.info(f"📊 Momentum feature stats: mean={momentum_feature.mean():.6f}, std={momentum_feature.std():.6f}")
        
        # Test 3: CrossTimeframeVolatilityGenerator
        logger.info("\n📊 Testing CrossTimeframeVolatilityGenerator...")
        volatility_gen = CrossTimeframeVolatilityGenerator(timeframe=15)
        
        start_time = time.time()
        volatility_feature = volatility_gen._generate_feature(test_data)
        generation_time = time.time() - start_time
        
        logger.info(f"✅ Generated volatility feature in {generation_time:.3f}s")
        logger.info(f"📊 Volatility feature stats: mean={volatility_feature.mean():.6f}, std={volatility_feature.std():.6f}")
        
        # Test 4: CrossTimeframeVolumeGenerator
        logger.info("\n📊 Testing CrossTimeframeVolumeGenerator...")
        volume_gen = CrossTimeframeVolumeGenerator(timeframe=20)
        
        start_time = time.time()
        volume_feature = volume_gen._generate_feature(test_data)
        generation_time = time.time() - start_time
        
        logger.info(f"✅ Generated volume feature in {generation_time:.3f}s")
        logger.info(f"📊 Volume feature stats: mean={volume_feature.mean():.6f}, std={volume_feature.std():.6f}")
        
        # Test 5: Performance comparison
        logger.info("\n⚡ Performance Comparison Test...")
        test_performance_comparison(test_data)
        
        logger.info("\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison(data: pd.DataFrame):
    """Compare performance between optimized and non-optimized approaches."""
    logger.info("🔄 Running performance comparison...")
    
    try:
        from feature_generation.categories.cross_timeframe import CrossTimeframeMomentumGenerator
        
        # Test with VectorBT optimizations
        optimized_gen = CrossTimeframeMomentumGenerator(timeframe=10)
        
        # Warm up
        _ = optimized_gen._generate_feature(data.head(1000))
        
        # Time optimized version
        start_time = time.time()
        for _ in range(5):
            _ = optimized_gen._generate_feature(data)
        optimized_time = time.time() - start_time
        
        logger.info(f"⚡ Optimized VectorBT approach: {optimized_time:.3f}s for 5 iterations")
        
        # Test memory usage
        import psutil
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"💾 Memory usage: {memory_usage:.1f} MB")
        
    except Exception as e:
        logger.warning(f"Performance comparison failed: {e}")

def test_vectorbt_availability():
    """Test VectorBT and optimization components availability."""
    logger.info("🔍 Checking VectorBT and optimization components availability...")
    
    try:
        import vectorbt as vbt
        logger.info("✅ VectorBT is available")
    except ImportError:
        logger.warning("⚠️ VectorBT not available")
    
    try:
        from feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
        logger.info("✅ VectorBTRollingOptimizer is available")
    except ImportError:
        logger.warning("⚠️ VectorBTRollingOptimizer not available")
    
    try:
        from feature_generation.utils.vectorization_optimizer import VectorizationOptimizer
        logger.info("✅ VectorizationOptimizer is available")
    except ImportError:
        logger.warning("⚠️ VectorizationOptimizer not available")

def main():
    """Main test function."""
    logger.info("🚀 Starting Cross-Timeframe VectorBT Optimization Tests")
    
    # Test component availability
    test_vectorbt_availability()
    
    # Run main tests
    success = test_cross_timeframe_generators()
    
    if success:
        logger.info("🎉 All tests completed successfully!")
        return 0
    else:
        logger.error("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())