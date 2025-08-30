#!/usr/bin/env python3
"""
Test script for S/R Detection Optimization Integration

This script demonstrates how to use the comprehensive S/R detection optimization
system with real data testing and parameter optimization.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.tactician.sr_detection_optimization import SRDetectionOptimizer, setup_sr_detection_optimizer
from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
from src.utils.logger import system_logger


def generate_sample_market_data(n_periods: int = 1000) -> pd.DataFrame:
    """Generate sample market data for testing."""
    np.random.seed(42)
    
    # Generate realistic price data
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_periods)  # 2% daily volatility
    
    prices = [base_price]
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    # Generate OHLCV data
    data = []
    for i, price in enumerate(prices):
        # Generate realistic OHLC from close price
        volatility = np.random.uniform(0.005, 0.02)
        high = price * (1 + np.random.uniform(0, volatility))
        low = price * (1 - np.random.uniform(0, volatility))
        open_price = np.random.uniform(low, high)
        
        # Generate volume
        volume = np.random.uniform(1000, 10000)
        
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


def generate_multi_timeframe_data(main_data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Generate multi-timeframe data from main data."""
    timeframes = {
        '1m': main_data,
        '5m': main_data.resample('5T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna(),
        '15m': main_data.resample('15T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna(),
        '1h': main_data.resample('1H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna(),
    }
    return timeframes


async def test_sr_optimization():
    """Test the S/R detection optimization system."""
    logger = system_logger.getChild("TestSROptimization")
    
    try:
        logger.info("🚀 Starting S/R Detection Optimization Test")
        
        # Configuration
        config = {
            "sr_breakout_predictor": {
                "enable_detailed_reporting": True,
                "use_optimized_params": True,
                "optimization_results_file": "test_optimization_results.json",
            },
            "sr_detection_optimization": {
                "n_trials": 20,  # Reduced for testing
                "cv_folds": 3,
                "test_size": 0.2,
                "optimization_timeout": 300,  # 5 minutes for testing
                "performance_thresholds": {
                    "min_sharpe_ratio": 0.3,
                    "max_drawdown": -0.2,
                    "min_win_rate": 0.5,
                    "min_profit_factor": 1.2,
                    "min_signal_clarity": 0.05,
                }
            }
        }
        
        # Generate sample data
        logger.info("📊 Generating sample market data...")
        market_data = generate_sample_market_data(1000)
        multi_timeframe_data = generate_multi_timeframe_data(market_data)
        
        logger.info(f"Generated {len(market_data)} data points")
        logger.info(f"Multi-timeframe data: {list(multi_timeframe_data.keys())}")
        
        # Initialize optimizer
        logger.info("🔧 Initializing S/R Detection Optimizer...")
        optimizer = await setup_sr_detection_optimizer(config)
        
        if not optimizer:
            logger.error("❌ Failed to initialize optimizer")
            return False
        
        # Run optimization
        logger.info("🎯 Running comprehensive S/R detection optimization...")
        result = await optimizer.optimize_sr_detection(
            market_data=market_data,
            multi_timeframe_data=multi_timeframe_data,
            target_data=None  # No supervised learning for this test
        )
        
        if result:
            logger.info("✅ Optimization completed successfully!")
            logger.info(f"Best optimization score: {result.optimization_score:.4f}")
            logger.info(f"Optimization method: {result.optimization_method}")
            logger.info(f"Number of trials: {result.n_trials}")
            logger.info(f"Optimization time: {result.optimization_time:.2f} seconds")
            
            # Display optimized parameters
            logger.info("📈 Optimized Parameters:")
            logger.info(f"  Method Weights: {result.method_weights}")
            logger.info(f"  Strength Weights: {result.strength_weights}")
            logger.info(f"  DBSCAN Params: {result.dbscan_params}")
            logger.info(f"  Timeframe Weights: {result.timeframe_weights}")
            logger.info(f"  Advanced Params: {result.advanced_params}")
            
            # Save results
            logger.info("💾 Saving optimization results...")
            optimizer.save_optimization_results("test_optimization_results.json")
            
            # Test optimized S/R predictor
            logger.info("🧪 Testing optimized S/R predictor...")
            await test_optimized_sr_predictor(config, market_data, result)
            
            return True
        else:
            logger.error("❌ Optimization failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


async def test_optimized_sr_predictor(
    config: dict,
    market_data: pd.DataFrame,
    optimization_result
):
    """Test the S/R predictor with optimized parameters."""
    logger = system_logger.getChild("TestOptimizedSRPredictor")
    
    try:
        # Initialize S/R predictor
        sr_predictor = SRBreakoutPredictor(config)
        await sr_predictor.initialize()
        
        # Apply optimized parameters
        optimized_params = {
            "method_weights": optimization_result.method_weights,
            "strength_weights": optimization_result.strength_weights,
            "dbscan_params": optimization_result.dbscan_params,
            "timeframe_weights": optimization_result.timeframe_weights,
            "advanced_params": optimization_result.advanced_params,
        }
        
        await sr_predictor.set_optimized_parameters(optimized_params)
        
        # Test S/R context generation
        current_price = market_data['close'].iloc[-1]
        sr_context = await sr_predictor.get_sr_context(market_data, current_price)
        
        logger.info("📊 S/R Context Results:")
        logger.info(f"  Current Price: {current_price:.4f}")
        logger.info(f"  Support Levels: {len(sr_context.get('support_levels', []))}")
        logger.info(f"  Resistance Levels: {len(sr_context.get('resistance_levels', []))}")
        logger.info(f"  Support Strength: {sr_context.get('support_strength', 0):.3f}")
        logger.info(f"  Resistance Strength: {sr_context.get('resistance_strength', 0):.3f}")
        logger.info(f"  Clusters Detected: {sr_context.get('clustering_result', {}).get('n_clusters', 0)}")
        
        # Test S/R breakout prediction
        predictions = await sr_predictor.predict_sr_breakouts(market_data, current_price)
        
        logger.info("🎯 S/R Breakout Predictions:")
        logger.info(f"  Support Levels: {len(predictions.get('support_levels', []))}")
        logger.info(f"  Resistance Levels: {len(predictions.get('resistance_levels', []))}")
        logger.info(f"  Breakout Probabilities: {len(predictions.get('breakout_probabilities', {}))}")
        logger.info(f"  Confidence Scores: {len(predictions.get('confidence_scores', {}))}")
        
        # Compare with default parameters
        logger.info("🔄 Comparing with default parameters...")
        default_sr_predictor = SRBreakoutPredictor(config)
        await default_sr_predictor.initialize()
        
        default_sr_context = await default_sr_predictor.get_sr_context(market_data, current_price)
        
        logger.info("📊 Comparison Results:")
        logger.info(f"  Optimized Support Levels: {len(sr_context.get('support_levels', []))}")
        logger.info(f"  Default Support Levels: {len(default_sr_context.get('support_levels', []))}")
        logger.info(f"  Optimized Resistance Levels: {len(sr_context.get('resistance_levels', []))}")
        logger.info(f"  Default Resistance Levels: {len(default_sr_context.get('resistance_levels', []))}")
        logger.info(f"  Optimized Support Strength: {sr_context.get('support_strength', 0):.3f}")
        logger.info(f"  Default Support Strength: {default_sr_context.get('support_strength', 0):.3f}")
        
        # Calculate improvement
        optimized_levels = len(sr_context.get('support_levels', [])) + len(sr_context.get('resistance_levels', []))
        default_levels = len(default_sr_context.get('support_levels', [])) + len(default_sr_context.get('resistance_levels', []))
        
        if default_levels > 0:
            improvement = (optimized_levels - default_levels) / default_levels * 100
            logger.info(f"  Level Detection Improvement: {improvement:.1f}%")
        
        logger.info("✅ Optimized S/R predictor test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Optimized S/R predictor test failed: {e}")


async def test_parameter_loading():
    """Test loading optimized parameters from file."""
    logger = system_logger.getChild("TestParameterLoading")
    
    try:
        logger.info("📂 Testing parameter loading from file...")
        
        # Configuration with optimization file
        config = {
            "sr_breakout_predictor": {
                "use_optimized_params": True,
                "optimization_results_file": "test_optimization_results.json",
            }
        }
        
        # Initialize S/R predictor (should load optimized parameters)
        sr_predictor = SRBreakoutPredictor(config)
        await sr_predictor.initialize()
        
        # Get current parameters
        current_params = sr_predictor.get_current_parameters()
        
        logger.info("📊 Loaded Parameters:")
        logger.info(f"  Method Weights: {current_params['method_weights']}")
        logger.info(f"  Strength Weights: {current_params['strength_weights']}")
        logger.info(f"  DBSCAN Params: {current_params['dbscan_params']}")
        
        logger.info("✅ Parameter loading test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Parameter loading test failed: {e}")


async def main():
    """Main test function."""
    logger = system_logger.getChild("MainTest")
    
    try:
        logger.info("🚀 Starting S/R Detection Optimization Integration Tests")
        
        # Test 1: Basic optimization
        logger.info("\n" + "="*50)
        logger.info("TEST 1: Basic S/R Detection Optimization")
        logger.info("="*50)
        
        success1 = await test_sr_optimization()
        
        if success1:
            # Test 2: Parameter loading
            logger.info("\n" + "="*50)
            logger.info("TEST 2: Parameter Loading from File")
            logger.info("="*50)
            
            await test_parameter_loading()
        
        logger.info("\n" + "="*50)
        logger.info("🎉 All tests completed!")
        logger.info("="*50)
        
        if success1:
            logger.info("✅ S/R Detection Optimization Integration is working correctly!")
            logger.info("📁 Check 'test_optimization_results.json' for detailed results")
        else:
            logger.error("❌ Some tests failed. Check logs for details.")
        
    except Exception as e:
        logger.error(f"❌ Main test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())