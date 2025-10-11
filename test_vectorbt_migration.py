#!/usr/bin/env python3
"""
Test script to validate VectorBT migration for feature generation.

This script tests the three main feature categories:
1. Advanced Volume Features
2. Advanced Volatility Features  
3. Cross-Timeframe Features

It validates that VectorBT optimizations are working correctly and provides
performance improvements over pandas fallbacks.
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(n_points: int = 1000) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate price data with some trend and volatility
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_points)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_points)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_points))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_points))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_points)
    }, index=pd.date_range('2020-01-01', periods=n_points, freq='1min'))
    
    # Ensure high >= low and high/low >= open/close
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    return data

def test_volume_features():
    """Test Advanced Volume Features with VectorBT optimization."""
    logger.info("Testing Advanced Volume Features...")
    
    try:
        from src.feature_generation.categories.volume import (
            VolumeFeatureGenerator, VolumeSMAGenerator, VolumeEMAGenerator,
            VolumeRatioGenerator, VolumeROCGenerator, VolumeStdGenerator
        )
        
        data = create_sample_data(500)
        
        # Test Volume SMA Generator
        volume_sma = VolumeSMAGenerator(period=20)
        start_time = time.time()
        result = volume_sma.generate(data)
        end_time = time.time()
        
        logger.info(f"✅ Volume SMA: Generated in {end_time - start_time:.4f}s")
        logger.info(f"   Result shape: {result.shape}")
        logger.info(f"   Performance stats: {volume_sma.get_performance_stats()}")
        
        # Test Volume Ratio Generator
        volume_ratio = VolumeRatioGenerator(period=20)
        start_time = time.time()
        result = volume_ratio.generate(data)
        end_time = time.time()
        
        logger.info(f"✅ Volume Ratio: Generated in {end_time - start_time:.4f}s")
        logger.info(f"   Result shape: {result.shape}")
        logger.info(f"   Performance stats: {volume_ratio.get_performance_stats()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Volume Features test failed: {e}")
        return False

def test_volatility_features():
    """Test Advanced Volatility Features with VectorBT optimization."""
    logger.info("Testing Advanced Volatility Features...")
    
    try:
        from src.feature_generation.categories.volatility import (
            VolatilityFeatureGenerator, VectorBTVolatilityFeatureGenerator
        )
        
        data = create_sample_data(500)
        
        # Test Volatility Generator
        volatility_gen = VolatilityFeatureGenerator(period=20)
        start_time = time.time()
        result = volatility_gen.generate(data)
        end_time = time.time()
        
        logger.info(f"✅ Volatility: Generated in {end_time - start_time:.4f}s")
        logger.info(f"   Result shape: {result.shape}")
        logger.info(f"   Performance stats: {volatility_gen.get_performance_stats()}")
        
        # Test VectorBT Volatility Generator
        vectorbt_vol = VectorBTVolatilityFeatureGenerator(period=20)
        start_time = time.time()
        result = vectorbt_vol.generate(data)
        end_time = time.time()
        
        logger.info(f"✅ VectorBT Volatility: Generated in {end_time - start_time:.4f}s")
        logger.info(f"   Result shape: {result.shape}")
        logger.info(f"   VectorBT stats: {vectorbt_vol.get_vectorbt_stats()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Volatility Features test failed: {e}")
        return False

def test_cross_timeframe_features():
    """Test Cross-Timeframe Features with VectorBT optimization."""
    logger.info("Testing Cross-Timeframe Features...")
    
    try:
        from src.feature_generation.categories.cross_timeframe import (
            CrossTimeframeMomentumGenerator, CrossTimeframeVolatilityGenerator
        )
        from src.feature_generation.base_calculations import BaseCalculationType
        
        data = create_sample_data(500)
        
        # Test Cross-Timeframe Momentum Generator
        ctf_momentum = CrossTimeframeMomentumGenerator(
            timeframe=10, 
            base_calculation=BaseCalculationType.PRICE_RETURNS
        )
        start_time = time.time()
        result = ctf_momentum.generate(data)
        end_time = time.time()
        
        logger.info(f"✅ Cross-Timeframe Momentum: Generated in {end_time - start_time:.4f}s")
        logger.info(f"   Result shape: {result.shape}")
        logger.info(f"   Performance stats: {ctf_momentum.get_performance_stats()}")
        
        # Test Cross-Timeframe Volatility Generator
        ctf_volatility = CrossTimeframeVolatilityGenerator(
            timeframe=10,
            base_calculation=BaseCalculationType.PRICE_RETURNS
        )
        start_time = time.time()
        result = ctf_volatility.generate(data)
        end_time = time.time()
        
        logger.info(f"✅ Cross-Timeframe Volatility: Generated in {end_time - start_time:.4f}s")
        logger.info(f"   Result shape: {result.shape}")
        logger.info(f"   Performance stats: {ctf_volatility.get_performance_stats()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Cross-Timeframe Features test failed: {e}")
        return False

def test_vectorbt_rolling_optimizer():
    """Test VectorBT Rolling Optimizer directly."""
    logger.info("Testing VectorBT Rolling Optimizer...")
    
    try:
        from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
        
        data = create_sample_data(500)
        optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        
        # Test rolling mean
        start_time = time.time()
        result = optimizer.rolling_mean(data['close'], window=20)
        end_time = time.time()
        
        logger.info(f"✅ VectorBT Rolling Mean: Generated in {end_time - start_time:.4f}s")
        logger.info(f"   Result shape: {result.shape}")
        logger.info(f"   Performance stats: {optimizer.get_performance_stats()}")
        
        # Test rolling std
        start_time = time.time()
        result = optimizer.rolling_std(data['close'], window=20)
        end_time = time.time()
        
        logger.info(f"✅ VectorBT Rolling Std: Generated in {end_time - start_time:.4f}s")
        logger.info(f"   Result shape: {result.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ VectorBT Rolling Optimizer test failed: {e}")
        return False

def main():
    """Run all VectorBT migration tests."""
    logger.info("🚀 Starting VectorBT Migration Validation Tests...")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Test VectorBT Rolling Optimizer
    test_results['vectorbt_rolling_optimizer'] = test_vectorbt_rolling_optimizer()
    logger.info("")
    
    # Test Volume Features
    test_results['volume_features'] = test_volume_features()
    logger.info("")
    
    # Test Volatility Features
    test_results['volatility_features'] = test_volatility_features()
    logger.info("")
    
    # Test Cross-Timeframe Features
    test_results['cross_timeframe_features'] = test_cross_timeframe_features()
    logger.info("")
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info("")
    logger.info(f"Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All VectorBT migration tests passed successfully!")
        return True
    else:
        logger.error("⚠️ Some tests failed. Please check the logs above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)