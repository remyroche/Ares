#!/usr/bin/env python3
"""
Test script for refactored feature generators.

This script tests that the refactored feature generators work correctly
with the centralized utilities.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_consolidated_generators():
    """Test the consolidated feature generators."""
    try:
        from consolidated_feature_generators import (
            ConsolidatedRSIGenerator,
            ConsolidatedMACDGenerator,
            ConsolidatedEMAGenerator,
            ConsolidatedSMAGenerator,
            register_consolidated_generators
        )
        
        logger.info("Testing consolidated generators...")
        
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=1000, freq='1min')
        np.random.seed(42)
        data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(1000) * 0.01),
            'high': 100 + np.cumsum(np.random.randn(1000) * 0.01) + np.random.randn(1000) * 0.5,
            'low': 100 + np.cumsum(np.random.randn(1000) * 0.01) - np.random.randn(1000) * 0.5,
            'volume': np.random.lognormal(10, 1, 1000)
        }, index=dates)
        
        # Test RSI generator
        rsi_gen = ConsolidatedRSIGenerator(period=14)
        rsi_result = rsi_gen._generate_feature(data)
        logger.info(f"RSI generator test: {len(rsi_result)} values generated")
        logger.info(f"RSI stats: mean={rsi_result.mean():.2f}, std={rsi_result.std():.2f}")
        
        # Test MACD generator
        macd_gen = ConsolidatedMACDGenerator(fast=12, slow=26, signal=9)
        macd_result = macd_gen._generate_feature(data)
        logger.info(f"MACD generator test: {len(macd_result)} values generated")
        logger.info(f"MACD stats: mean={macd_result.mean():.2f}, std={macd_result.std():.2f}")
        
        # Test EMA generator
        ema_gen = ConsolidatedEMAGenerator(period=20)
        ema_result = ema_gen._generate_feature(data)
        logger.info(f"EMA generator test: {len(ema_result)} values generated")
        logger.info(f"EMA stats: mean={ema_result.mean():.2f}, std={ema_result.std():.2f}")
        
        # Test SMA generator
        sma_gen = ConsolidatedSMAGenerator(period=20)
        sma_result = sma_gen._generate_feature(data)
        logger.info(f"SMA generator test: {len(sma_result)} values generated")
        logger.info(f"SMA stats: mean={sma_result.mean():.2f}, std={sma_result.std():.2f}")
        
        logger.info("✅ All consolidated generators working correctly!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Consolidated generators test failed: {e}")
        return False

def test_refactored_files():
    """Test that refactored files can be imported and used."""
    try:
        logger.info("Testing refactored files...")
        
        # Test momentum.py
        from src.feature_generation.categories.momentum import MomentumFeatureGenerator
        momentum_gen = MomentumFeatureGenerator()
        logger.info("✅ Momentum feature generator imported successfully")
        
        # Test trend.py
        from src.feature_generation.categories.trend import TrendFeatureGenerator
        trend_gen = TrendFeatureGenerator()
        logger.info("✅ Trend feature generator imported successfully")
        
        # Test oscillator.py
        from src.feature_generation.categories.oscillator import OscillatorFeatureGenerator
        oscillator_gen = OscillatorFeatureGenerator()
        logger.info("✅ Oscillator feature generator imported successfully")
        
        # Test volume.py
        from src.feature_generation.categories.volume import VolumeFeatureGenerator
        volume_gen = VolumeFeatureGenerator()
        logger.info("✅ Volume feature generator imported successfully")
        
        # Test volatility.py
        from src.feature_generation.categories.volatility import VolatilityFeatureGenerator
        volatility_gen = VolatilityFeatureGenerator()
        logger.info("✅ Volatility feature generator imported successfully")
        
        logger.info("✅ All refactored files imported successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Refactored files test failed: {e}")
        return False

def test_optimization_methods():
    """Test that optimization methods are available."""
    try:
        logger.info("Testing optimization methods...")
        
        from src.feature_generation.categories.momentum import MomentumFeatureGenerator
        
        # Create a generator instance
        gen = MomentumFeatureGenerator()
        
        # Test that optimization methods exist
        assert hasattr(gen, '_optimized_rolling_operation'), "Missing _optimized_rolling_operation method"
        assert hasattr(gen, '_fallback_rolling_operation'), "Missing _fallback_rolling_operation method"
        assert hasattr(gen, '_normalize_feature'), "Missing _normalize_feature method"
        assert hasattr(gen, '_fallback_normalize'), "Missing _fallback_normalize method"
        
        logger.info("✅ All optimization methods available!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimization methods test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting refactored generators validation...")
    
    tests = [
        test_consolidated_generators,
        test_refactored_files,
        test_optimization_methods
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        logger.info("-" * 50)
    
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Refactoring successful!")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)