#!/usr/bin/env python3
"""
Enhanced S/R Optimization Test Script

This script demonstrates the enhanced S/R optimization system with:
1. Timeframe-specific parameter optimization (1m, 5m, 15m, 30m)
2. Enhanced metrics validation using backtesting
3. Improved S/R level detection and validation
4. Comprehensive performance analysis
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


def generate_enhanced_market_data(n_periods: int = 2000) -> pd.DataFrame:
    """Generate enhanced market data with realistic S/R levels for different timeframes."""
    np.random.seed(42)
    
    # Generate base price movement with more realistic S/R levels
    base_price = 100.0
    prices = [base_price]
    
    # Create multiple S/R levels with different strengths
    sr_levels = {
        "strong_support": [95.0, 98.0],  # Strong support levels
        "weak_support": [96.5, 97.5],    # Weak support levels
        "strong_resistance": [102.0, 105.0],  # Strong resistance levels
        "weak_resistance": [101.0, 103.5],    # Weak resistance levels
    }
    
    # Volume patterns
    volume_patterns = []
    
    for i in range(1, n_periods):
        current_price = prices[-1]
        
        # Check if price is near S/R levels
        near_sr = False
        sr_strength = 0.0
        sr_type = None
        
        # Check strong support levels
        for level in sr_levels["strong_support"]:
            if abs(current_price - level) / level < 0.015:  # Within 1.5% of S/R
                near_sr = True
                sr_strength = 0.8  # Strong level
                sr_type = "support"
                break
        
        # Check weak support levels
        if not near_sr:
            for level in sr_levels["weak_support"]:
                if abs(current_price - level) / level < 0.01:  # Within 1% of S/R
                    near_sr = True
                    sr_strength = 0.4  # Weak level
                    sr_type = "support"
                    break
        
        # Check strong resistance levels
        if not near_sr:
            for level in sr_levels["strong_resistance"]:
                if abs(current_price - level) / level < 0.015:  # Within 1.5% of S/R
                    near_sr = True
                    sr_strength = 0.8  # Strong level
                    sr_type = "resistance"
                    break
        
        # Check weak resistance levels
        if not near_sr:
            for level in sr_levels["weak_resistance"]:
                if abs(current_price - level) / level < 0.01:  # Within 1% of S/R
                    near_sr = True
                    sr_strength = 0.4  # Weak level
                    sr_type = "resistance"
                    break
        
        if near_sr:
            # Higher probability of bouncing off S/R levels based on strength
            bounce_probability = 0.5 + (sr_strength * 0.4)  # 50-90% based on strength
            
            if np.random.random() < bounce_probability:
                # Bounce direction based on whether it's support or resistance
                if sr_type == "support":
                    change = np.random.uniform(0.005, 0.02)  # Bounce up
                else:  # Resistance
                    change = np.random.uniform(-0.02, -0.005)  # Bounce down
            else:
                # Breakout
                if sr_type == "support":  # Break down through support
                    change = np.random.uniform(-0.02, -0.005)
                else:  # Break up through resistance
                    change = np.random.uniform(0.005, 0.02)
        else:
            # Normal random walk with trend
            trend = np.sin(i / 100) * 0.001  # Cyclical trend
            change = np.random.normal(trend, 0.008)  # 0.8% daily volatility
        
        new_price = current_price * (1 + change)
        prices.append(new_price)
    
    # Generate OHLCV data with enhanced volume patterns
    data = []
    for i, price in enumerate(prices):
        # Generate realistic OHLC from close price
        volatility = np.random.uniform(0.002, 0.012)
        high = price * (1 + np.random.uniform(0, volatility))
        low = price * (1 - np.random.uniform(0, volatility))
        open_price = np.random.uniform(low, high)
        
        # Generate volume with enhanced patterns
        base_volume = np.random.uniform(1000, 5000)
        volume_multiplier = 1.0
        
        # Check if near S/R level for volume enhancement
        for level_type, levels in sr_levels.items():
            for level in levels:
                if abs(price - level) / level < 0.02:
                    if "strong" in level_type:
                        volume_multiplier = np.random.uniform(2.0, 4.0)  # Higher volume for strong levels
                    else:
                        volume_multiplier = np.random.uniform(1.3, 2.0)  # Moderate volume for weak levels
                    break
        
        # Add volume spikes during breakouts
        if i > 0 and abs(price - prices[i-1]) / prices[i-1] > 0.01:
            volume_multiplier *= np.random.uniform(1.2, 1.8)
        
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


async def test_enhanced_sr_optimization():
    """Test enhanced S/R optimization with timeframe-specific parameters."""
    logger = system_logger.getChild("EnhancedSROptimization")
    
    try:
        logger.info("🚀 Starting Enhanced S/R Optimization Test")
        logger.info("This demonstrates timeframe-specific optimization for 1-30m timeframes")
        
        # Enhanced configuration with timeframe-specific settings
        config = {
            "sr_breakout_predictor": {
                "enable_sr_breakout_tactics": True,
                "sr_proximity_threshold": 0.015,
                "breakout_confidence_threshold": 0.6,
                "sr_detection_method": "fractal",
                "min_sr_strength": 0.3,
                "max_sr_levels": 15,
                "sr_lookback_periods": 200,
                "volume_weight": 0.7,
                "price_weight": 0.3,
                "atr_multiplier": 1.5,
                "breakout_confirmation_periods": 3,
                "false_breakout_filter": True,
            },
            "sr_detection_optimization": {
                "n_trials": 20,  # Increased for better optimization
                "cv_folds": 5,
                "test_size": 0.2,
                "optimization_timeout": 600,  # 10 minutes
                "timeframe_config": {
                    "1m": {
                        "touch_threshold": 0.0005,  # 0.05% for 1m
                        "bounce_threshold": 0.002,  # 0.2% for 1m
                        "breakout_threshold": 0.005,  # 0.5% for 1m
                        "min_touches": 3,
                        "volume_spike_threshold": 1.3,
                    },
                    "5m": {
                        "touch_threshold": 0.001,  # 0.1% for 5m
                        "bounce_threshold": 0.003,  # 0.3% for 5m
                        "breakout_threshold": 0.008,  # 0.8% for 5m
                        "min_touches": 3,
                        "volume_spike_threshold": 1.4,
                    },
                    "15m": {
                        "touch_threshold": 0.0015,  # 0.15% for 15m
                        "bounce_threshold": 0.005,  # 0.5% for 15m
                        "breakout_threshold": 0.01,  # 1% for 15m
                        "min_touches": 2,
                        "volume_spike_threshold": 1.5,
                    },
                    "30m": {
                        "touch_threshold": 0.002,  # 0.2% for 30m
                        "bounce_threshold": 0.008,  # 0.8% for 30m
                        "breakout_threshold": 0.015,  # 1.5% for 30m
                        "min_touches": 2,
                        "volume_spike_threshold": 1.6,
                    }
                },
                "performance_thresholds": {
                    "min_sr_validation_score": 0.6,
                    "min_bounce_rate": 0.5,
                    "max_false_breakout_rate": 0.4,
                    "min_volume_confirmation": 0.4,
                    "min_level_detection_accuracy": 0.3,
                }
            },
            "sr_backtesting": {
                "touch_threshold": 0.001,
                "bounce_threshold": 0.005,
                "breakout_threshold": 0.01,
                "false_breakout_threshold": 0.02,
                "confirmation_periods": 3,
                "min_touches": 2,
                "volume_spike_threshold": 1.5,
                "institutional_volume_threshold": 2.0,
                "volume_confirmation_threshold": 1.2,
                "volume_lookback_periods": 20,
                "volume_cluster_radius": 0.005,
                "min_bounce_rate": 0.6,
                "max_false_breakout_rate": 0.3,
                "min_volume_confirmation": 0.5,
                "age_decay_factor": 0.95,
                "max_level_age_days": 365,
                "enable_multi_timeframe": True,
                "timeframe_weights": {
                    "1m": 0.05, "5m": 0.1, "15m": 0.15, "1h": 0.2, "4h": 0.25, "1d": 0.25
                },
            }
        }
        
        # Generate enhanced market data
        logger.info("📊 Generating enhanced market data with realistic S/R levels...")
        market_data = generate_enhanced_market_data(2000)
        
        logger.info(f"Generated {len(market_data)} data points")
        logger.info(f"Price range: {market_data['close'].min():.2f} - {market_data['close'].max():.2f}")
        logger.info(f"Volume range: {market_data['volume'].min():.0f} - {market_data['volume'].max():.0f}")
        
        # Test optimization for different timeframes
        timeframes = ["1m", "5m", "15m", "30m"]
        optimization_results = {}
        
        for timeframe in timeframes:
            logger.info(f"\n{'='*60}")
            logger.info(f"OPTIMIZING FOR {timeframe} TIMEFRAME")
            logger.info(f"{'='*60}")
            
            # Initialize optimizer
            optimizer = await setup_sr_detection_optimizer(config)
            
            if not optimizer:
                logger.error(f"❌ Failed to initialize optimizer for {timeframe}")
                continue
            
            # Run optimization for specific timeframe
            result = await optimizer.optimize_sr_detection(
                market_data=market_data,
                multi_timeframe_data=None,
                target_data=None,
                target_timeframe=timeframe
            )
            
            if result:
                optimization_results[timeframe] = result
                
                logger.info(f"✅ {timeframe} Optimization Results:")
                logger.info(f"  Optimization Score: {result.optimization_score:.4f}")
                logger.info(f"  Method: {result.optimization_method}")
                logger.info(f"  Trials: {result.n_trials}")
                logger.info(f"  Time: {result.optimization_time:.1f}s")
                
                # Show optimized parameters
                logger.info(f"  Method Weights: {result.method_weights}")
                logger.info(f"  Strength Weights: {result.strength_weights}")
                logger.info(f"  DBSCAN Params: {result.dbscan_params}")
                
                # Show backtesting results
                if hasattr(optimizer, 'backtest_results') and optimizer.backtest_results:
                    latest_backtest = optimizer.backtest_results[-1]['backtest_result']
                    logger.info(f"  S/R Validation Score: {latest_backtest.sr_validation_score:.3f}")
                    logger.info(f"  Bounce Rate: {latest_backtest.overall_bounce_rate:.2%}")
                    logger.info(f"  Volume Confirmation: {latest_backtest.avg_volume_confirmation_rate:.2%}")
                    logger.info(f"  Level Detection Accuracy: {latest_backtest.level_detection_accuracy:.2%}")
            else:
                logger.error(f"❌ Optimization failed for {timeframe}")
        
        # Compare results across timeframes
        await compare_timeframe_results(optimization_results, market_data, config)
        
        # Test parameter sensitivity
        await test_parameter_sensitivity(market_data, config)
        
        logger.info(f"\n{'='*60}")
        logger.info("🎉 Enhanced S/R Optimization Test Completed!")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"❌ Enhanced S/R optimization test failed: {e}")


async def compare_timeframe_results(optimization_results: dict, market_data: pd.DataFrame, config: dict):
    """Compare optimization results across different timeframes."""
    logger = system_logger.getChild("TimeframeComparison")
    
    try:
        logger.info(f"\n{'='*80}")
        logger.info("TIMEFRAME COMPARISON ANALYSIS")
        logger.info(f"{'='*80}")
        
        if not optimization_results:
            logger.warning("No optimization results to compare")
            return
        
        # Create comparison table
        comparison_data = []
        
        for timeframe, result in optimization_results.items():
            # Get backtesting results
            backtest_result = None
            if hasattr(result, 'backtest_results') and result.backtest_results:
                backtest_result = result.backtest_results[-1]['backtest_result']
            
            comparison_data.append({
                "timeframe": timeframe,
                "optimization_score": result.optimization_score,
                "sr_validation_score": backtest_result.sr_validation_score if backtest_result else 0.0,
                "bounce_rate": backtest_result.overall_bounce_rate if backtest_result else 0.0,
                "volume_confirmation": backtest_result.avg_volume_confirmation_rate if backtest_result else 0.0,
                "level_accuracy": backtest_result.level_detection_accuracy if backtest_result else 0.0,
                "false_breakout_rate": backtest_result.overall_false_breakout_rate if backtest_result else 0.0,
                "optimization_time": result.optimization_time,
                "n_trials": result.n_trials,
            })
        
        # Sort by optimization score
        comparison_data.sort(key=lambda x: x["optimization_score"], reverse=True)
        
        # Display comparison table
        logger.info(f"{'Timeframe':<8} {'Opt Score':<10} {'SR Score':<10} {'Bounce':<8} {'Volume':<8} {'Accuracy':<10} {'False':<8} {'Time':<8}")
        logger.info("-" * 80)
        
        for data in comparison_data:
            logger.info(f"{data['timeframe']:<8} {data['optimization_score']:<10.4f} {data['sr_validation_score']:<10.3f} "
                       f"{data['bounce_rate']:<8.2%} {data['volume_confirmation']:<8.2%} {data['level_accuracy']:<10.2%} "
                       f"{data['false_breakout_rate']:<8.2%} {data['optimization_time']:<8.1f}s")
        
        # Find best performing timeframe
        best_timeframe = comparison_data[0]
        logger.info(f"\n🏆 Best Performing Timeframe: {best_timeframe['timeframe']}")
        logger.info(f"   Optimization Score: {best_timeframe['optimization_score']:.4f}")
        logger.info(f"   S/R Validation Score: {best_timeframe['sr_validation_score']:.3f}")
        logger.info(f"   Bounce Rate: {best_timeframe['bounce_rate']:.2%}")
        
        # Recommendations
        logger.info(f"\n💡 Recommendations:")
        for data in comparison_data:
            if data['bounce_rate'] < 0.5:
                logger.info(f"  - {data['timeframe']}: Improve bounce rate detection")
            if data['volume_confirmation'] < 0.4:
                logger.info(f"  - {data['timeframe']}: Enhance volume analysis")
            if data['level_accuracy'] < 0.3:
                logger.info(f"  - {data['timeframe']}: Optimize level detection parameters")
        
    except Exception as e:
        logger.error(f"❌ Timeframe comparison failed: {e}")


async def test_parameter_sensitivity(market_data: pd.DataFrame, config: dict):
    """Test parameter sensitivity for different timeframes."""
    logger = system_logger.getChild("ParameterSensitivity")
    
    try:
        logger.info(f"\n{'='*80}")
        logger.info("PARAMETER SENSITIVITY ANALYSIS")
        logger.info(f"{'='*80}")
        
        # Test different parameter sets
        parameter_sets = {
            "Conservative": {
                "touch_threshold": 0.002,
                "bounce_threshold": 0.01,
                "breakout_threshold": 0.02,
                "min_touches": 4,
            },
            "Moderate": {
                "touch_threshold": 0.001,
                "bounce_threshold": 0.005,
                "breakout_threshold": 0.01,
                "min_touches": 3,
            },
            "Aggressive": {
                "touch_threshold": 0.0005,
                "bounce_threshold": 0.003,
                "breakout_threshold": 0.008,
                "min_touches": 2,
            }
        }
        
        # Test for 15m timeframe
        target_timeframe = "15m"
        results = []
        
        for param_name, params in parameter_sets.items():
            logger.info(f"\n🔧 Testing {param_name} parameters for {target_timeframe}...")
            
            # Update configuration
            test_config = config.copy()
            test_config["sr_backtesting"].update(params)
            
            # Initialize components
            sr_predictor = SRBreakoutPredictor(test_config)
            await sr_predictor.initialize()
            
            validator = await setup_sr_backtesting_validator(test_config)
            
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
                    "name": param_name,
                    "total_levels": backtest_result.total_levels_tested,
                    "successful_levels": backtest_result.successful_levels,
                    "accuracy": backtest_result.level_detection_accuracy,
                    "bounce_rate": backtest_result.overall_bounce_rate,
                    "volume_confirmation": backtest_result.avg_volume_confirmation_rate,
                    "confidence_score": backtest_result.avg_confidence_score,
                    "validation_score": backtest_result.sr_validation_score
                })
        
        # Compare parameter sensitivity
        if results:
            logger.info(f"\n📊 Parameter Sensitivity Results for {target_timeframe}:")
            logger.info("="*90)
            logger.info(f"{'Parameter Set':<15} {'Levels':<8} {'Success':<8} {'Accuracy':<10} {'Bounce':<8} {'Volume':<8} {'Confidence':<10} {'Score':<8}")
            logger.info("="*90)
            
            for result in results:
                logger.info(f"{result['name']:<15} {result['total_levels']:<8} {result['successful_levels']:<8} "
                           f"{result['accuracy']:<10.2%} {result['bounce_rate']:<8.2%} {result['volume_confirmation']:<8.2%} "
                           f"{result['confidence_score']:<10.3f} {result['validation_score']:<8.3f}")
            
            # Find best parameter set
            best_result = max(results, key=lambda x: x['validation_score'])
            logger.info(f"\n🏆 Best Parameter Set: {best_result['name']}")
            logger.info(f"   S/R Validation Score: {best_result['validation_score']:.3f}")
            logger.info(f"   Level Detection Accuracy: {best_result['accuracy']:.2%}")
            logger.info(f"   Bounce Rate: {best_result['bounce_rate']:.2%}")
        
    except Exception as e:
        logger.error(f"❌ Parameter sensitivity test failed: {e}")


async def main():
    """Main test function."""
    logger = system_logger.getChild("MainEnhancedTest")
    
    try:
        logger.info("🚀 Starting Enhanced S/R Optimization Tests")
        logger.info("This demonstrates timeframe-specific optimization and enhanced metrics validation")
        
        await test_enhanced_sr_optimization()
        
    except Exception as e:
        logger.error(f"❌ Main test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())