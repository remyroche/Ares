#!/usr/bin/env python3
"""
Test script for S/R Backtesting Validation

This script demonstrates how the S/R optimization system now uses proper backtesting
to validate whether detected S/R levels are actually effective. It shows the real
success metrics and how we assess that an S/R level is indeed a valid S/R level.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.tactician.sr_detection_optimization import setup_sr_detection_optimizer
from src.tactician.sr_backtesting_validator import setup_sr_backtesting_validator
from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
from src.utils.logger import system_logger


def generate_realistic_market_data(n_periods: int = 1000) -> pd.DataFrame:
    """Generate realistic market data with actual S/R levels."""
    np.random.seed(42)
    
    # Generate base price movement
    base_price = 100.0
    prices = [base_price]
    
    # Create some realistic S/R levels
    sr_levels = [95.0, 98.0, 102.0, 105.0, 108.0]  # Known S/R levels
    
    for i in range(1, n_periods):
        current_price = prices[-1]
        
        # Check if price is near S/R levels
        near_sr = False
        for sr_level in sr_levels:
            if abs(current_price - sr_level) / sr_level < 0.02:  # Within 2% of S/R
                near_sr = True
                # Higher probability of bouncing off S/R levels
                if np.random.random() < 0.7:  # 70% chance of bounce
                    # Bounce direction based on whether it's support or resistance
                    if current_price < sr_level:  # Support
                        change = np.random.uniform(0.005, 0.02)  # Bounce up
                    else:  # Resistance
                        change = np.random.uniform(-0.02, -0.005)  # Bounce down
                else:
                    # Breakout
                    if current_price < sr_level:  # Break down through support
                        change = np.random.uniform(-0.02, -0.005)
                    else:  # Break up through resistance
                        change = np.random.uniform(0.005, 0.02)
                break
        
        if not near_sr:
            # Normal random walk
            change = np.random.normal(0, 0.01)  # 1% daily volatility
        
        new_price = current_price * (1 + change)
        prices.append(new_price)
    
    # Generate OHLCV data
    data = []
    for i, price in enumerate(prices):
        # Generate realistic OHLC from close price
        volatility = np.random.uniform(0.002, 0.01)
        high = price * (1 + np.random.uniform(0, volatility))
        low = price * (1 - np.random.uniform(0, volatility))
        open_price = np.random.uniform(low, high)
        
        # Generate volume (higher near S/R levels)
        base_volume = np.random.uniform(1000, 5000)
        volume_multiplier = 1.0
        
        # Check if near S/R level
        for sr_level in sr_levels:
            if abs(price - sr_level) / sr_level < 0.02:
                volume_multiplier = np.random.uniform(1.5, 3.0)  # Higher volume near S/R
                break
        
        volume = base_volume * volume_multiplier
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume,
            'timestamp': datetime.now() - timedelta(minutes=n_periods-i)
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


async def test_sr_backtesting_validation():
    """Test the S/R backtesting validation system."""
    logger = system_logger.getChild("TestSRBacktesting")
    
    try:
        logger.info("🚀 Starting S/R Backtesting Validation Test")
        
        # Configuration with backtesting enabled
        config = {
            "sr_breakout_predictor": {
                "enable_detailed_reporting": True,
                "use_optimized_params": True,
            },
            "sr_detection_optimization": {
                "n_trials": 10,  # Reduced for testing
                "cv_folds": 3,
                "test_size": 0.2,
                "optimization_timeout": 300,
            },
            "sr_backtesting": {
                "touch_threshold": 0.001,  # 0.1% touch threshold
                "bounce_threshold": 0.005,  # 0.5% bounce threshold
                "breakout_threshold": 0.01,  # 1% breakout threshold
                "false_breakout_threshold": 0.02,  # 2% false breakout
                "confirmation_periods": 3,
                "min_touches": 2,
                "volume_spike_threshold": 1.5,  # 1.5x average volume
                "institutional_volume_threshold": 2.0,  # 2x average volume
                "volume_confirmation_threshold": 1.2,  # 1.2x average volume
                "volume_lookback_periods": 20,  # 20 periods for volume baseline
                "volume_cluster_radius": 0.005,  # 0.5% price range for clustering
                "min_bounce_rate": 0.6,  # 60% minimum bounce rate
                "max_false_breakout_rate": 0.3,  # 30% max false breakouts
                "min_volume_confirmation": 0.5,  # 50% volume confirmation
            }
        }
        
        # Generate realistic market data with known S/R levels
        logger.info("📊 Generating realistic market data with known S/R levels...")
        market_data = generate_realistic_market_data(1000)
        
        logger.info(f"Generated {len(market_data)} data points")
        logger.info(f"Price range: {market_data['close'].min():.2f} - {market_data['close'].max():.2f}")
        
        # Test 1: Direct S/R Backtesting Validation
        logger.info("\n" + "="*60)
        logger.info("TEST 1: Direct S/R Backtesting Validation")
        logger.info("="*60)
        
        await test_direct_backtesting(config, market_data)
        
        # Test 2: S/R Detection with Backtesting Optimization
        logger.info("\n" + "="*60)
        logger.info("TEST 2: S/R Detection with Backtesting Optimization")
        logger.info("="*60)
        
        await test_optimization_with_backtesting(config, market_data)
        
        # Test 3: Success Metrics Analysis
        logger.info("\n" + "="*60)
        logger.info("TEST 3: Success Metrics Analysis")
        logger.info("="*60)
        
        await test_success_metrics_analysis(config, market_data)
        
        logger.info("\n" + "="*60)
        logger.info("🎉 All backtesting validation tests completed!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ Backtesting validation test failed: {e}")


async def test_direct_backtesting(config: dict, market_data: pd.DataFrame):
    """Test direct S/R backtesting validation."""
    logger = system_logger.getChild("DirectBacktesting")
    
    try:
        # Initialize S/R predictor
        sr_predictor = SRBreakoutPredictor(config)
        await sr_predictor.initialize()
        
        # Get S/R levels
        current_price = market_data['close'].iloc[-1]
        sr_context = await sr_predictor.get_sr_context(market_data, current_price)
        
        support_levels = sr_context.get("support_levels", [])
        resistance_levels = sr_context.get("resistance_levels", [])
        all_levels = support_levels + resistance_levels
        
        logger.info(f"Detected {len(support_levels)} support levels and {len(resistance_levels)} resistance levels")
        
        # Initialize backtesting validator
        validator = await setup_sr_backtesting_validator(config)
        
        # Validate S/R levels through backtesting
        backtest_result = await validator.validate_sr_levels(
            market_data=market_data,
            sr_levels=all_levels,
            current_price=current_price
        )
        
        if backtest_result:
            logger.info("📊 Backtesting Results:")
            logger.info(f"  Total Levels Tested: {backtest_result.total_levels_tested}")
            logger.info(f"  Successful Levels: {backtest_result.successful_levels}")
            logger.info(f"  Level Detection Accuracy: {backtest_result.level_detection_accuracy:.2%}")
            
            logger.info("\n🎯 S/R Performance Metrics:")
            logger.info(f"  Overall Bounce Rate: {backtest_result.overall_bounce_rate:.2%}")
            logger.info(f"  Overall Breakout Rate: {backtest_result.overall_breakout_rate:.2%}")
            logger.info(f"  False Breakout Rate: {backtest_result.overall_false_breakout_rate:.2%}")
            
            logger.info("\n📈 Support vs Resistance Performance:")
            logger.info(f"  Support Bounce Rate: {backtest_result.support_bounce_rate:.2%}")
            logger.info(f"  Resistance Bounce Rate: {backtest_result.resistance_bounce_rate:.2%}")
            logger.info(f"  Support Breakout Rate: {backtest_result.support_breakout_rate:.2%}")
            logger.info(f"  Resistance Breakout Rate: {backtest_result.resistance_breakout_rate:.2%}")
            
            logger.info("\n📊 Volume Analysis:")
            logger.info(f"  Average Volume Spike Ratio: {backtest_result.avg_volume_spike_ratio:.2f}x")
            logger.info(f"  Volume Confirmation Rate: {backtest_result.avg_volume_confirmation_rate:.2%}")
            logger.info(f"  Institutional Volume Ratio: {backtest_result.avg_institutional_volume_ratio:.2%}")
            logger.info(f"  Volume Cluster Score: {backtest_result.avg_volume_cluster_score:.2f}")
            
            logger.info("\n🎯 S/R Validation Score:")
            logger.info(f"  Overall S/R Validation Score: {backtest_result.sr_validation_score:.3f}")
            logger.info(f"  Level Detection Accuracy: {backtest_result.level_detection_accuracy:.2%}")
            logger.info(f"  Average Confidence Score: {backtest_result.avg_confidence_score:.3f}")
            
            logger.info("\n🎯 Individual Level Analysis:")
            for i, level_test in enumerate(backtest_result.level_tests[:5]):  # Show first 5
                logger.info(f"  Level {i+1} ({level_test.level_type} at {level_test.level_price:.2f}):")
                logger.info(f"    Touches: {level_test.touches}, Bounces: {level_test.bounces}")
                logger.info(f"    Bounce Rate: {level_test.bounce_rate:.2%}")
                logger.info(f"    Confidence Score: {level_test.confidence_score:.3f}")
                logger.info(f"    Volume Spike Ratio: {level_test.volume_spike_ratio:.2f}x")
                logger.info(f"    Volume Confirmation: {level_test.volume_confirmation_rate:.2%}")
                logger.info(f"    Institutional Volume: {level_test.institutional_volume_ratio:.2%}")
                logger.info(f"    Volume Cluster Score: {level_test.volume_cluster_score:.2f}")
            
            # Assess S/R level validity
            await assess_sr_level_validity(backtest_result)
        
    except Exception as e:
        logger.error(f"❌ Direct backtesting test failed: {e}")


async def test_optimization_with_backtesting(config: dict, market_data: pd.DataFrame):
    """Test S/R optimization using backtesting validation."""
    logger = system_logger.getChild("OptimizationBacktesting")
    
    try:
        # Initialize optimizer
        optimizer = await setup_sr_detection_optimizer(config)
        
        if not optimizer:
            logger.error("❌ Failed to initialize optimizer")
            return
        
        # Run optimization with backtesting
        logger.info("🎯 Running S/R optimization with backtesting validation...")
        result = await optimizer.optimize_sr_detection(
            market_data=market_data,
            multi_timeframe_data=None,
            target_data=None
        )
        
        if result:
            logger.info("✅ Optimization with backtesting completed!")
            logger.info(f"Best optimization score: {result.optimization_score:.4f}")
            logger.info(f"Optimization method: {result.optimization_method}")
            
            # Show optimized parameters
            logger.info("\n📈 Optimized Parameters:")
            logger.info(f"  Method Weights: {result.method_weights}")
            logger.info(f"  Strength Weights: {result.strength_weights}")
            logger.info(f"  DBSCAN Params: {result.dbscan_params}")
            
            # Show backtesting results from optimization
            if hasattr(optimizer, 'backtest_results') and optimizer.backtest_results:
                latest_backtest = optimizer.backtest_results[-1]['backtest_result']
                logger.info(f"\n🎯 Optimization Backtesting Results:")
                logger.info(f"  S/R Validation Score: {latest_backtest.sr_validation_score:.3f}")
                logger.info(f"  Bounce Rate: {latest_backtest.overall_bounce_rate:.2%}")
                logger.info(f"  Volume Confirmation: {latest_backtest.avg_volume_confirmation_rate:.2%}")
        
    except Exception as e:
        logger.error(f"❌ Optimization with backtesting test failed: {e}")


async def test_success_metrics_analysis(config: dict, market_data: pd.DataFrame):
    """Analyze success metrics for S/R level validation."""
    logger = system_logger.getChild("SuccessMetrics")
    
    try:
        logger.info("📊 Analyzing S/R Level Success Metrics...")
        
        # Initialize components
        sr_predictor = SRBreakoutPredictor(config)
        await sr_predictor.initialize()
        
        validator = await setup_sr_backtesting_validator(config)
        
        # Test different parameter sets
        parameter_sets = [
            {"name": "Default", "params": {}},
            {"name": "Conservative", "params": {
                "touch_threshold": 0.002,
                "bounce_threshold": 0.01,
                "breakout_threshold": 0.02,
            }},
            {"name": "Aggressive", "params": {
                "touch_threshold": 0.0005,
                "bounce_threshold": 0.003,
                "breakout_threshold": 0.008,
            }}
        ]
        
        results = []
        
        for param_set in parameter_sets:
            logger.info(f"\n🔧 Testing {param_set['name']} parameters...")
            
            # Update validator parameters
            for key, value in param_set['params'].items():
                setattr(validator, key, value)
            
            # Get S/R levels
            current_price = market_data['close'].iloc[-1]
            sr_context = await sr_predictor.get_sr_context(market_data, current_price)
            
            support_levels = sr_context.get("support_levels", [])
            resistance_levels = sr_context.get("resistance_levels", [])
            all_levels = support_levels + resistance_levels
            
            # Validate through backtesting
            backtest_result = await validator.validate_sr_levels(
                market_data=market_data,
                sr_levels=all_levels,
                current_price=current_price
            )
            
            if backtest_result:
                results.append({
                    "name": param_set['name'],
                    "total_levels": backtest_result.total_levels_tested,
                    "successful_levels": backtest_result.successful_levels,
                    "accuracy": backtest_result.level_detection_accuracy,
                    "bounce_rate": backtest_result.overall_bounce_rate,
                    "volume_confirmation": backtest_result.avg_volume_confirmation_rate,
                    "confidence_score": backtest_result.avg_confidence_score,
                    "validation_score": backtest_result.sr_validation_score
                })
        
        # Compare results
        logger.info("\n📊 Parameter Set Comparison:")
        logger.info("="*90)
        logger.info(f"{'Parameter Set':<15} {'Levels':<8} {'Success':<8} {'Accuracy':<10} {'Bounce':<8} {'Volume':<8} {'Confidence':<10} {'Score':<8}")
        logger.info("="*90)
        
        for result in results:
            logger.info(f"{result['name']:<15} {result['total_levels']:<8} {result['successful_levels']:<8} "
                       f"{result['accuracy']:<10.2%} {result['bounce_rate']:<8.2%} {result['volume_confirmation']:<8.2%} "
                       f"{result['confidence_score']:<10.3f} {result['validation_score']:<8.3f}")
        
        # Find best performing parameter set
        best_result = max(results, key=lambda x: x['validation_score'])
        logger.info(f"\n🏆 Best Performing Parameter Set: {best_result['name']}")
        logger.info(f"   S/R Validation Score: {best_result['validation_score']:.3f}")
        logger.info(f"   Level Detection Accuracy: {best_result['accuracy']:.2%}")
        logger.info(f"   Bounce Rate: {best_result['bounce_rate']:.2%}")
        logger.info(f"   Volume Confirmation: {best_result['volume_confirmation']:.2%}")
        
    except Exception as e:
        logger.error(f"❌ Success metrics analysis failed: {e}")


async def assess_sr_level_validity(backtest_result):
    """Assess whether detected S/R levels are truly valid."""
    logger = system_logger.getChild("SRValidityAssessment")
    
    try:
        logger.info("\n🔍 S/R Level Validity Assessment:")
        logger.info("="*50)
        
        # Define validity criteria
        validity_criteria = {
            "bounce_rate_threshold": 0.6,  # 60% bounce rate
            "confidence_threshold": 0.5,   # 50% confidence score
            "min_touches": 3,              # Minimum 3 touches
            "max_false_breakout_rate": 0.3,  # Maximum 30% false breakouts
            "volume_confirmation_threshold": 0.5,  # Minimum 50% volume confirmation
            "validation_score_threshold": 0.7,  # Minimum 70% validation score
        }
        
        # Assess overall validity
        overall_valid = True
        assessment = []
        
        # Check bounce rate
        if backtest_result.overall_bounce_rate >= validity_criteria["bounce_rate_threshold"]:
            assessment.append("✅ Bounce rate is acceptable")
        else:
            assessment.append(f"❌ Bounce rate too low: {backtest_result.overall_bounce_rate:.2%} < {validity_criteria['bounce_rate_threshold']:.2%}")
            overall_valid = False
        
        # Check confidence score
        if backtest_result.avg_confidence_score >= validity_criteria["confidence_threshold"]:
            assessment.append("✅ Average confidence score is acceptable")
        else:
            assessment.append(f"❌ Confidence score too low: {backtest_result.avg_confidence_score:.3f} < {validity_criteria['confidence_threshold']:.3f}")
            overall_valid = False
        
        # Check false breakout rate
        if backtest_result.overall_false_breakout_rate <= validity_criteria["max_false_breakout_rate"]:
            assessment.append("✅ False breakout rate is acceptable")
        else:
            assessment.append(f"❌ False breakout rate too high: {backtest_result.overall_false_breakout_rate:.2%} > {validity_criteria['max_false_breakout_rate']:.2%}")
            overall_valid = False
        
        # Check volume confirmation
        if backtest_result.avg_volume_confirmation_rate >= validity_criteria["volume_confirmation_threshold"]:
            assessment.append("✅ Volume confirmation rate is acceptable")
        else:
            assessment.append(f"❌ Volume confirmation too low: {backtest_result.avg_volume_confirmation_rate:.2%} < {validity_criteria['volume_confirmation_threshold']:.2%}")
            overall_valid = False
        
        # Check overall validation score
        if backtest_result.sr_validation_score >= validity_criteria["validation_score_threshold"]:
            assessment.append("✅ Overall S/R validation score is acceptable")
        else:
            assessment.append(f"❌ Validation score too low: {backtest_result.sr_validation_score:.3f} < {validity_criteria['validation_score_threshold']:.3f}")
            overall_valid = False
        
        # Check level detection accuracy
        if backtest_result.level_detection_accuracy >= 0.5:
            assessment.append("✅ Level detection accuracy is acceptable")
        else:
            assessment.append(f"❌ Level detection accuracy too low: {backtest_result.level_detection_accuracy:.2%} < 50%")
            overall_valid = False
        
        # Print assessment
        for item in assessment:
            logger.info(f"  {item}")
        
        # Overall verdict
        if overall_valid:
            logger.info(f"\n🎉 VERDICT: S/R levels are VALID and RELIABLE")
            logger.info(f"   S/R Validation Score: {backtest_result.sr_validation_score:.3f}")
        else:
            logger.info(f"\n⚠️  VERDICT: S/R levels need IMPROVEMENT")
            logger.info(f"   S/R Validation Score: {backtest_result.sr_validation_score:.3f}")
        
        # Recommendations
        logger.info(f"\n💡 Recommendations:")
        if backtest_result.overall_bounce_rate < 0.6:
            logger.info("  - Increase bounce rate by improving S/R detection algorithms")
        if backtest_result.overall_false_breakout_rate > 0.3:
            logger.info("  - Reduce false breakouts by improving confirmation logic")
        if backtest_result.avg_volume_confirmation_rate < 0.5:
            logger.info("  - Improve volume analysis to confirm S/R level significance")
        if backtest_result.level_detection_accuracy < 0.5:
            logger.info("  - Enhance level detection accuracy through parameter optimization")
        if backtest_result.sr_validation_score < 0.7:
            logger.info("  - Focus on improving overall S/R validation metrics")
        
    except Exception as e:
        logger.error(f"❌ S/R validity assessment failed: {e}")


async def main():
    """Main test function."""
    logger = system_logger.getChild("MainBacktestingTest")
    
    try:
        logger.info("🚀 Starting S/R Backtesting Validation Tests")
        logger.info("This demonstrates how S/R levels are validated through proper backtesting")
        logger.info("and how success metrics are calculated to assess S/R level effectiveness.")
        
        await test_sr_backtesting_validation()
        
    except Exception as e:
        logger.error(f"❌ Main test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())