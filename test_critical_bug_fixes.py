#!/usr/bin/env python3
"""
Test script to verify critical bug fixes in S/R backtesting validation system.
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from tactician.sr_breakout_predictor import SRBreakoutPredictor
from tactician.sr_backtesting_validator import SRBacktestingValidator
from tactician.sr_detection_optimization import SRDetectionOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_test_market_data(n_periods: int = 1000) -> pd.DataFrame:
    """Generate realistic test market data with clear S/R levels."""
    np.random.seed(42)
    
    # Generate base price movement
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_periods)  # 2% daily volatility
    prices = [base_price]
    
    # Create some clear support and resistance levels
    support_levels = [95.0, 98.0, 102.0]
    resistance_levels = [105.0, 108.0, 112.0]
    
    for i in range(1, n_periods):
        current_price = prices[-1]
        
        # Add some mean reversion to S/R levels
        for support in support_levels:
            if current_price < support:
                returns[i] += 0.01  # Bounce up from support
        
        for resistance in resistance_levels:
            if current_price > resistance:
                returns[i] -= 0.01  # Bounce down from resistance
        
        new_price = current_price * (1 + returns[i])
        prices.append(new_price)
    
    # Generate OHLC data
    data = []
    for i, close in enumerate(prices):
        high = close * (1 + abs(np.random.normal(0, 0.005)))
        low = close * (1 - abs(np.random.normal(0, 0.005)))
        open_price = close * (1 + np.random.normal(0, 0.002))
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.index = pd.date_range(start='2023-01-01', periods=len(df), freq='1min')
    
    return df

async def test_sr_level_detection():
    """Test that S/R levels are being detected properly."""
    logger.info("🧪 Testing S/R level detection...")
    
    # Generate test data
    market_data = generate_test_market_data(500)
    current_price = market_data['close'].iloc[-1]
    
    # Initialize SR predictor with default config
    config = {
        "sr_breakout_predictor": {
            "enable_enhanced_strength": True,
            "enable_dbscan_clustering": True,
            "max_sr_levels": 10,
            "sr_detection_method": "fractal",
            "touch_threshold": 0.001,
            "bounce_threshold": 0.005,
            "breakout_threshold": 0.01,
            "false_breakout_threshold": 0.003,
            "confirmation_periods": 3,
            "min_touches": 2,
            "volume_spike_threshold": 1.5,
            "institutional_volume_threshold": 0.7,
            "volume_clustering_threshold": 0.6,
            "level_age_threshold": 50,
            "age_decay_factor": 0.95,
            "multi_timeframe_alignment_threshold": 0.7,
            "confidence_threshold": 0.6,
            "min_bounce_rate": 0.6,
            "max_false_breakout_rate": 0.3,
            "min_volume_confirmation": 0.5,
            "strength_score_weights": {
                "touch_count": 0.3,
                "total_volume": 0.2,
                "level_age": 0.2,
                "bounce_rate": 0.2,
                "isolation_score": 0.1
            },
            "touch_count_lookback": 100,
            "isolation_distance_threshold": 0.02,
            "sr_proximity_threshold": 0.01,
            "bounce_rate_threshold": 0.005
        }
    }
    
    sr_predictor = SRBreakoutPredictor(config)
    await sr_predictor.initialize()
    
    # Test S/R context generation
    try:
        sr_context = await sr_predictor.get_sr_context(market_data, current_price)
        
        if sr_context:
            support_levels = sr_context.get('support_levels', [])
            resistance_levels = sr_context.get('resistance_levels', [])
            
            logger.info(f"✅ S/R context generated successfully")
            logger.info(f"   - Support levels detected: {len(support_levels)}")
            logger.info(f"   - Resistance levels detected: {len(resistance_levels)}")
            logger.info(f"   - Current price: {current_price:.2f}")
            
            if support_levels:
                logger.info(f"   - Nearest support: {sr_context.get('nearest_support', 'N/A')}")
            if resistance_levels:
                logger.info(f"   - Nearest resistance: {sr_context.get('nearest_resistance', 'N/A')}")
            
            return len(support_levels) > 0 or len(resistance_levels) > 0
        else:
            logger.error("❌ S/R context is empty")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in S/R level detection: {e}")
        return False

async def test_comprehensive_strength_calculation():
    """Test that comprehensive strength calculation works without errors."""
    logger.info("🧪 Testing comprehensive strength calculation...")
    
    # Generate test data
    market_data = generate_test_market_data(200)
    
    # Create some test S/R levels
    test_levels = [
        {'price': 100.0, 'strength': 0.7, 'type': 'support'},
        {'price': 105.0, 'strength': 0.8, 'type': 'resistance'},
        {'price': 95.0, 'strength': 0.6, 'type': 'support'}
    ]
    
    # Initialize SR predictor with default config
    config = {
        "sr_breakout_predictor": {
            "enable_enhanced_strength": True,
            "enable_dbscan_clustering": True,
            "max_sr_levels": 10,
            "sr_detection_method": "fractal",
            "strength_score_weights": {
                "touch_count": 0.3,
                "total_volume": 0.2,
                "level_age": 0.2,
                "bounce_rate": 0.2,
                "isolation_score": 0.1
            },
            "touch_count_lookback": 100,
            "age_decay_factor": 0.95
        }
    }
    
    sr_predictor = SRBreakoutPredictor(config)
    await sr_predictor.initialize()
    
    try:
        # Test comprehensive strength calculation
        strength_results = await sr_predictor.calculate_comprehensive_strength(market_data, test_levels)
        
        if strength_results:
            logger.info(f"✅ Comprehensive strength calculation successful")
            logger.info(f"   - Calculated strength for {len(strength_results)} levels")
            
            for level_id, strength_data in strength_results.items():
                comprehensive_strength = strength_data.get('comprehensive_strength', 0)
                logger.info(f"   - Level {level_id}: strength = {comprehensive_strength:.3f}")
            
            return True
        else:
            logger.error("❌ Comprehensive strength calculation returned empty result")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in comprehensive strength calculation: {e}")
        return False

async def test_backtesting_validation():
    """Test that backtesting validation works without errors."""
    logger.info("🧪 Testing backtesting validation...")
    
    # Generate test data
    market_data = generate_test_market_data(300)
    
    # Create test S/R levels
    test_levels = [
        {'price': 100.0, 'strength': 0.7, 'type': 'support'},
        {'price': 105.0, 'strength': 0.8, 'type': 'resistance'}
    ]
    
    # Initialize validator with config
    config = {
        "sr_backtesting_validator": {
            "touch_threshold": 0.001,
            "bounce_threshold": 0.005,
            "breakout_threshold": 0.01,
            "false_breakout_threshold": 0.003,
            "confirmation_periods": 3,
            "min_touches": 2,
            "volume_spike_threshold": 1.5,
            "institutional_volume_threshold": 0.7,
            "volume_clustering_threshold": 0.6,
            "level_age_threshold": 50,
            "age_decay_factor": 0.95,
            "multi_timeframe_alignment_threshold": 0.7,
            "confidence_threshold": 0.6,
            "min_bounce_rate": 0.6,
            "max_false_breakout_rate": 0.3,
            "min_volume_confirmation": 0.5
        }
    }
    
    validator = SRBacktestingValidator(config)
    
    try:
        # Test validation
        current_price = market_data['close'].iloc[-1]
        validation_result = await validator.validate_sr_levels(market_data, test_levels, current_price)
        
        if validation_result:
            logger.info(f"✅ Backtesting validation successful")
            logger.info(f"   - Overall validation score: {validation_result.sr_validation_score:.3f}")
            logger.info(f"   - Bounce rate: {validation_result.bounce_rate:.3f}")
            logger.info(f"   - False breakout rate: {validation_result.false_breakout_rate:.3f}")
            logger.info(f"   - Volume confirmation rate: {validation_result.volume_confirmation_rate:.3f}")
            
            return True
        else:
            logger.error("❌ Backtesting validation returned empty result")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in backtesting validation: {e}")
        return False

async def test_optimization_framework():
    """Test that optimization framework works without asyncio errors."""
    logger.info("🧪 Testing optimization framework...")
    
    # Generate test data
    market_data = generate_test_market_data(400)
    
    # Initialize optimizer with config
    config = {
        "sr_detection_optimization": {
            "n_trials": 10,
            "cv_folds": 3,
            "test_size": 0.2,
            "optimization_timeout": 300,
            "timeframe_config": {
                "1m": {
                    "touch_threshold": 0.0005,
                    "bounce_threshold": 0.002,
                    "breakout_threshold": 0.005,
                    "min_touches": 2,
                    "volume_spike_threshold": 1.2
                },
                "5m": {
                    "touch_threshold": 0.001,
                    "bounce_threshold": 0.003,
                    "breakout_threshold": 0.008,
                    "min_touches": 3,
                    "volume_spike_threshold": 1.3
                },
                "15m": {
                    "touch_threshold": 0.0015,
                    "bounce_threshold": 0.005,
                    "breakout_threshold": 0.01,
                    "min_touches": 4,
                    "volume_spike_threshold": 1.5
                },
                "30m": {
                    "touch_threshold": 0.002,
                    "bounce_threshold": 0.008,
                    "breakout_threshold": 0.015,
                    "min_touches": 5,
                    "volume_spike_threshold": 1.7
                }
            },
            "performance_thresholds": {
                "min_bounce_rate": 0.4,
                "max_false_breakout_rate": 0.4,
                "min_volume_confirmation": 0.3,
                "min_level_detection_accuracy": 0.3,
                "min_sr_validation_score": 0.3
            }
        }
    }
    
    optimizer = SRDetectionOptimizer(config)
    
    try:
        # Initialize the optimizer first
        if not await optimizer.initialize():
            logger.error("❌ Failed to initialize optimizer")
            return False
        
        # Test optimization for 15m timeframe
        result = await optimizer.optimize_sr_detection(market_data, target_timeframe="15m")
        
        if result:
            logger.info(f"✅ Optimization successful")
            logger.info(f"   - Optimization score: {result.optimization_score:.3f}")
            logger.info(f"   - Method: {result.optimization_method}")
            logger.info(f"   - Timeframe: {result.timeframe_optimized}")
            logger.info(f"   - Trials: {result.n_trials}")
            
            return True
        else:
            logger.error("❌ Optimization returned empty result")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in optimization: {e}")
        return False

async def main():
    """Run all critical bug fix tests."""
    logger.info("🚀 Starting critical bug fix verification tests...")
    
    test_results = []
    
    # Test 1: S/R Level Detection
    test_results.append(await test_sr_level_detection())
    
    # Test 2: Comprehensive Strength Calculation
    test_results.append(await test_comprehensive_strength_calculation())
    
    # Test 3: Backtesting Validation
    test_results.append(await test_backtesting_validation())
    
    # Test 4: Optimization Framework
    test_results.append(await test_optimization_framework())
    
    # Summary
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    logger.info("\n" + "="*60)
    logger.info("📊 CRITICAL BUG FIX VERIFICATION RESULTS")
    logger.info("="*60)
    logger.info(f"✅ Tests passed: {passed_tests}/{total_tests}")
    logger.info(f"❌ Tests failed: {total_tests - passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL CRITICAL BUGS HAVE BEEN FIXED!")
        logger.info("✅ S/R level detection is working")
        logger.info("✅ Comprehensive strength calculation is working")
        logger.info("✅ Backtesting validation is working")
        logger.info("✅ Optimization framework is working")
    else:
        logger.warning("⚠️  Some critical bugs may still exist")
        logger.warning("Please review the failed tests above")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)