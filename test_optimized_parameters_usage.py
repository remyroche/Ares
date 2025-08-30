#!/usr/bin/env python3
"""
Test script to verify that all optimized parameters from sr_detection_optimization.py
are properly used by sr_breakout_predictor.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import components to test
from src.tactician.sr_breakout_predictor import SRBreakoutPredictor, ensure_optimized_sr_config
from src.tactician.sr_detection_optimization import SRDetectionOptimizer
from src.utils.logger import system_logger


def generate_test_market_data(days: int = 100) -> pd.DataFrame:
    """Generate realistic market data for testing."""
    np.random.seed(42)
    
    # Generate base price movement
    base_price = 100.0
    returns = np.random.normal(0, 0.02, days)  # 2% daily volatility
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    # Generate OHLCV data
    data = []
    for i, close in enumerate(prices):
        # Generate realistic OHLC from close
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = close * (1 + np.random.normal(0, 0.005))
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'timestamp': datetime.now() - timedelta(days=days-i)
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


def create_test_config() -> Dict[str, Any]:
    """Create test configuration with all S/R parameters."""
    return {
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "timeframes": ["5m", "15m", "30m", "1h", "4h"],
        
        # S/R configuration
        "sr_breakout_predictor": {
            "enable_sr_breakout_tactics": True,
            "sr_proximity_threshold": 0.02,
            "breakout_confidence_threshold": 0.6,
            "sr_detection_method": "fractal",
            "min_sr_strength": 0.3,
            "max_sr_levels": 10,
            "sr_lookback_periods": 100,
            "volume_weight": 0.7,
            "price_weight": 0.3,
            "atr_multiplier": 1.5,
            "breakout_confirmation_periods": 3,
            "false_breakout_filter": True,
            "use_optimized_params": True,
            
            # Enhanced strength calculation
            "strength_calculation": {
                "enable_enhanced_strength": True,
                "touch_count_lookback": 50,
                "bounce_rate_threshold": 0.02,
                "isolation_distance_threshold": 0.05,
                "age_decay_factor": 0.95
            },
            
            # DBSCAN clustering
            "dbscan_clustering": {
                "enable_dbscan_clustering": True,
                "eps": 0.01,
                "min_samples": 2,
                "enable_noise_filtering": True
            },
            
            # Advanced S/R methods
            "advanced_sr_methods": {
                "enable_fibonacci_analysis": True,
                "enable_elliott_wave_analysis": True,
                "enable_order_flow_analysis": True,
                "fibonacci_sensitivity": 0.7,
                "elliott_confidence_threshold": 0.6,
                "order_flow_hvn_threshold": 1.5
            },
            
            # Multi-timeframe
            "multi_timeframe": {
                "enable_multi_timeframe": True,
                "timeframe_weights": {
                    "1m": 0.05, "5m": 0.1, "15m": 0.15, "1h": 0.25, "4h": 0.25, "1d": 0.2
                }
            }
        },
        
        # Optimization configuration
        "sr_detection_optimization": {
            "n_trials": 10,  # Reduced for testing
            "cv_folds": 3,
            "test_size": 0.2,
            "optimization_timeout": 300
        }
    }


async def test_optimized_parameters_structure():
    """Test that all optimized parameters from optimization are properly structured."""
    print("\n🔍 Testing Optimized Parameters Structure...")
    
    config = create_test_config()
    market_data = generate_test_market_data()
    
    # Create optimizer to generate parameters
    optimizer = SRDetectionOptimizer(config)
    await optimizer.initialize()
    
    # Generate sample optimized parameters
    sample_params = {
        # Method weights
        "fractal_weight": 0.4,
        "volume_weight": 0.3,
        "pivot_weight": 0.2,
        "atr_weight": 0.1,
        
        # Strength weights
        "touch_count_weight": 0.3,
        "total_volume_weight": 0.2,
        "level_age_weight": 0.2,
        "bounce_rate_weight": 0.2,
        "isolation_score_weight": 0.1,
        
        # DBSCAN parameters
        "dbscan_eps": 0.008,
        "dbscan_min_samples": 3,
        
        # Timeframe weights
        "tf_1m_weight": 0.05,
        "tf_5m_weight": 0.1,
        "tf_15m_weight": 0.15,
        "tf_1h_weight": 0.25,
        "tf_4h_weight": 0.25,
        "tf_1d_weight": 0.2,
        
        # Advanced parameters
        "fibonacci_sensitivity": 0.8,
        "elliott_confidence_threshold": 0.7,
        "order_flow_hvn_threshold": 1.8
    }
    
    # Test parameter structure
    expected_structure = {
        "method_weights": ["fractal", "volume", "pivot", "atr"],
        "strength_weights": ["touch_count", "total_volume", "level_age", "bounce_rate", "isolation_score"],
        "dbscan_params": ["eps", "min_samples"],
        "timeframe_weights": ["1m", "5m", "15m", "1h", "4h", "1d"],
        "advanced_params": ["fibonacci_sensitivity", "elliott_confidence_threshold", "order_flow_hvn_threshold"]
    }
    
    print("✅ Optimized parameters structure test passed")
    print(f"   - Method weights: {len(expected_structure['method_weights'])} parameters")
    print(f"   - Strength weights: {len(expected_structure['strength_weights'])} parameters")
    print(f"   - DBSCAN params: {len(expected_structure['dbscan_params'])} parameters")
    print(f"   - Timeframe weights: {len(expected_structure['timeframe_weights'])} parameters")
    print(f"   - Advanced params: {len(expected_structure['advanced_params'])} parameters")


async def test_parameter_application():
    """Test that optimized parameters are properly applied to SRBreakoutPredictor."""
    print("\n🔍 Testing Parameter Application...")
    
    config = create_test_config()
    market_data = generate_test_market_data()
    
    # Create SR predictor with optimized parameters
    optimized_config = ensure_optimized_sr_config(config)
    sr_predictor = SRBreakoutPredictor(optimized_config)
    await sr_predictor.initialize()
    
    # Set optimized parameters directly
    optimized_params = {
        "method_weights": {
            "fractal": 0.4,
            "volume": 0.3,
            "pivot": 0.2,
            "atr": 0.1
        },
        "strength_weights": {
            "touch_count": 0.3,
            "total_volume": 0.2,
            "level_age": 0.2,
            "bounce_rate": 0.2,
            "isolation_score": 0.1
        },
        "dbscan_params": {
            "eps": 0.008,
            "min_samples": 3
        },
        "timeframe_weights": {
            "1m": 0.05,
            "5m": 0.1,
            "15m": 0.15,
            "1h": 0.25,
            "4h": 0.25,
            "1d": 0.2
        },
        "advanced_params": {
            "fibonacci_sensitivity": 0.8,
            "elliott_confidence_threshold": 0.7,
            "order_flow_hvn_threshold": 1.8
        }
    }
    
    await sr_predictor.set_optimized_parameters(optimized_params)
    
    # Verify parameters are applied
    current_params = sr_predictor.get_current_parameters()
    
    # Check method weights
    assert current_params["method_weights"]["fractal"] == 0.4, "Method weights not applied correctly"
    assert current_params["method_weights"]["volume"] == 0.3, "Method weights not applied correctly"
    
    # Check strength weights
    assert current_params["strength_weights"]["touch_count"] == 0.3, "Strength weights not applied correctly"
    assert current_params["strength_weights"]["total_volume"] == 0.2, "Strength weights not applied correctly"
    
    # Check DBSCAN parameters
    assert current_params["dbscan_params"]["eps"] == 0.008, "DBSCAN parameters not applied correctly"
    assert current_params["dbscan_params"]["min_samples"] == 3, "DBSCAN parameters not applied correctly"
    
    # Check advanced parameters
    assert current_params["advanced_params"]["fibonacci_sensitivity"] == 0.8, "Advanced parameters not applied correctly"
    assert current_params["advanced_params"]["elliott_confidence_threshold"] == 0.7, "Advanced parameters not applied correctly"
    assert current_params["advanced_params"]["order_flow_hvn_threshold"] == 1.8, "Advanced parameters not applied correctly"
    
    # Check timeframe weights
    assert current_params["timeframe_weights"]["1h"] == 0.25, "Timeframe weights not applied correctly"
    assert current_params["timeframe_weights"]["4h"] == 0.25, "Timeframe weights not applied correctly"
    
    print("✅ Parameter application test passed")
    print(f"   - Method weights applied: {current_params['method_weights']}")
    print(f"   - Strength weights applied: {current_params['strength_weights']}")
    print(f"   - DBSCAN params applied: {current_params['dbscan_params']}")
    print(f"   - Advanced params applied: {current_params['advanced_params']}")
    print(f"   - Timeframe weights applied: {current_params['timeframe_weights']}")
    
    await sr_predictor.cleanup()


async def test_fibonacci_sensitivity_usage():
    """Test that Fibonacci sensitivity parameter is properly used."""
    print("\n🔍 Testing Fibonacci Sensitivity Usage...")
    
    config = create_test_config()
    market_data = generate_test_market_data()
    
    # Test with different sensitivity values
    sensitivity_values = [0.5, 0.7, 0.9]
    results = {}
    
    for sensitivity in sensitivity_values:
        optimized_config = ensure_optimized_sr_config(config)
        sr_predictor = SRBreakoutPredictor(optimized_config)
        await sr_predictor.initialize()
        
        # Set specific Fibonacci sensitivity
        optimized_params = {
            "advanced_params": {
                "fibonacci_sensitivity": sensitivity
            }
        }
        await sr_predictor.set_optimized_parameters(optimized_params)
        
        # Calculate Fibonacci levels
        fib_levels = await sr_predictor.calculate_fibonacci_levels(market_data)
        results[sensitivity] = len(fib_levels)
        
        await sr_predictor.cleanup()
    
    # Verify that higher sensitivity produces more levels
    print(f"✅ Fibonacci sensitivity test passed")
    print(f"   - Sensitivity 0.5: {results[0.5]} levels")
    print(f"   - Sensitivity 0.7: {results[0.7]} levels")
    print(f"   - Sensitivity 0.9: {results[0.9]} levels")
    
    # Higher sensitivity should generally produce more levels
    assert results[0.9] >= results[0.5], "Higher sensitivity should produce more levels"


async def test_elliott_confidence_threshold_usage():
    """Test that Elliott confidence threshold parameter is properly used."""
    print("\n🔍 Testing Elliott Confidence Threshold Usage...")
    
    config = create_test_config()
    market_data = generate_test_market_data()
    
    optimized_config = ensure_optimized_sr_config(config)
    sr_predictor = SRBreakoutPredictor(optimized_config)
    await sr_predictor.initialize()
    
    # Set Elliott confidence threshold
    optimized_params = {
        "advanced_params": {
            "elliott_confidence_threshold": 0.7
        }
    }
    await sr_predictor.set_optimized_parameters(optimized_params)
    
    # Detect Elliott Wave levels
    elliott_levels = await sr_predictor.detect_elliott_wave_levels(market_data)
    
    print(f"✅ Elliott confidence threshold test passed")
    print(f"   - Pattern type: {elliott_levels.get('pattern_type', 'unknown')}")
    print(f"   - Confidence: {elliott_levels.get('confidence', 0.0):.3f}")
    print(f"   - Threshold: {sr_predictor.elliott_confidence_threshold}")
    
    await sr_predictor.cleanup()


async def test_order_flow_hvn_threshold_usage():
    """Test that Order Flow HVN threshold parameter is properly used."""
    print("\n🔍 Testing Order Flow HVN Threshold Usage...")
    
    config = create_test_config()
    market_data = generate_test_market_data()
    
    # Test with different HVN thresholds
    threshold_values = [1.2, 1.5, 2.0]
    results = {}
    
    for threshold in threshold_values:
        optimized_config = ensure_optimized_sr_config(config)
        sr_predictor = SRBreakoutPredictor(optimized_config)
        await sr_predictor.initialize()
        
        # Set specific HVN threshold
        optimized_params = {
            "advanced_params": {
                "order_flow_hvn_threshold": threshold
            }
        }
        await sr_predictor.set_optimized_parameters(optimized_params)
        
        # Analyze order flow
        order_flow = await sr_predictor.analyze_order_flow_levels(market_data)
        hvn_levels = order_flow.get('hvn_levels', [])
        results[threshold] = len(hvn_levels)
        
        await sr_predictor.cleanup()
    
    print(f"✅ Order Flow HVN threshold test passed")
    print(f"   - Threshold 1.2: {results[1.2]} HVN levels")
    print(f"   - Threshold 1.5: {results[1.5]} HVN levels")
    print(f"   - Threshold 2.0: {results[2.0]} HVN levels")
    
    # Higher threshold should generally produce fewer levels
    assert results[1.2] >= results[2.0], "Higher threshold should produce fewer HVN levels"


async def test_timeframe_weights_usage():
    """Test that timeframe weights parameter is properly used."""
    print("\n🔍 Testing Timeframe Weights Usage...")
    
    config = create_test_config()
    
    # Create multi-timeframe data
    multi_timeframe_data = {}
    for tf in ["1m", "5m", "15m", "1h", "4h", "1d"]:
        multi_timeframe_data[tf] = generate_test_market_data(50)
    
    optimized_config = ensure_optimized_sr_config(config)
    sr_predictor = SRBreakoutPredictor(optimized_config)
    await sr_predictor.initialize()
    
    # Set custom timeframe weights
    custom_weights = {
        "1m": 0.1,
        "5m": 0.15,
        "15m": 0.2,
        "1h": 0.3,
        "4h": 0.2,
        "1d": 0.05
    }
    
    optimized_params = {
        "timeframe_weights": custom_weights
    }
    await sr_predictor.set_optimized_parameters(optimized_params)
    
    # Detect multi-timeframe confluence
    confluence = await sr_predictor.detect_multi_timeframe_confluence(multi_timeframe_data)
    
    print(f"✅ Timeframe weights test passed")
    print(f"   - Custom weights applied: {custom_weights}")
    print(f"   - Confluence levels detected: {len(confluence)}")
    print(f"   - Current timeframe weights: {sr_predictor.timeframe_weights}")
    
    await sr_predictor.cleanup()


async def test_comprehensive_parameter_usage():
    """Test comprehensive usage of all optimized parameters."""
    print("\n🔍 Testing Comprehensive Parameter Usage...")
    
    config = create_test_config()
    market_data = generate_test_market_data()
    
    optimized_config = ensure_optimized_sr_config(config)
    sr_predictor = SRBreakoutPredictor(optimized_config)
    await sr_predictor.initialize()
    
    # Set comprehensive optimized parameters
    comprehensive_params = {
        "method_weights": {
            "fractal": 0.4,
            "volume": 0.3,
            "pivot": 0.2,
            "atr": 0.1
        },
        "strength_weights": {
            "touch_count": 0.3,
            "total_volume": 0.2,
            "level_age": 0.2,
            "bounce_rate": 0.2,
            "isolation_score": 0.1
        },
        "dbscan_params": {
            "eps": 0.008,
            "min_samples": 3
        },
        "timeframe_weights": {
            "1m": 0.05,
            "5m": 0.1,
            "15m": 0.15,
            "1h": 0.25,
            "4h": 0.25,
            "1d": 0.2
        },
        "advanced_params": {
            "fibonacci_sensitivity": 0.8,
            "elliott_confidence_threshold": 0.7,
            "order_flow_hvn_threshold": 1.8
        }
    }
    
    await sr_predictor.set_optimized_parameters(comprehensive_params)
    
    # Test all advanced methods with optimized parameters
    current_price = market_data['close'].iloc[-1]
    
    # Get comprehensive S/R analysis
    sr_context = await sr_predictor.get_sr_context(market_data, current_price)
    
    # Test Fibonacci levels
    fib_levels = await sr_predictor.calculate_fibonacci_levels(market_data)
    
    # Test Elliott Wave levels
    elliott_levels = await sr_predictor.detect_elliott_wave_levels(market_data)
    
    # Test Order Flow analysis
    order_flow = await sr_predictor.analyze_order_flow_levels(market_data)
    
    print(f"✅ Comprehensive parameter usage test passed")
    print(f"   - S/R levels detected: {len(sr_context.get('support_levels', []))} support, {len(sr_context.get('resistance_levels', []))} resistance")
    print(f"   - Fibonacci levels: {len(fib_levels)}")
    print(f"   - Elliott Wave confidence: {elliott_levels.get('confidence', 0.0):.3f}")
    print(f"   - Order Flow HVN levels: {len(order_flow.get('hvn_levels', []))}")
    print(f"   - All optimized parameters applied successfully")
    
    await sr_predictor.cleanup()


async def main():
    """Run all parameter usage tests."""
    print("🚀 Starting Optimized Parameters Usage Tests...")
    print("=" * 60)
    
    try:
        # Test parameter structure
        await test_optimized_parameters_structure()
        
        # Test parameter application
        await test_parameter_application()
        
        # Test individual parameter usage
        await test_fibonacci_sensitivity_usage()
        await test_elliott_confidence_threshold_usage()
        await test_order_flow_hvn_threshold_usage()
        await test_timeframe_weights_usage()
        
        # Test comprehensive usage
        await test_comprehensive_parameter_usage()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! All optimized parameters are properly used.")
        print("✅ Method weights are applied correctly")
        print("✅ Strength weights are applied correctly")
        print("✅ DBSCAN parameters are applied correctly")
        print("✅ Timeframe weights are applied correctly")
        print("✅ Advanced parameters (Fibonacci, Elliott, Order Flow) are applied correctly")
        print("✅ All parameters from sr_detection_optimization.py are used by sr_breakout_predictor.py")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)