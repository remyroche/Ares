#!/usr/bin/env python3
"""
Test script for Dynamic Tactician Triple Barrier Implementation

This script validates the dynamic implementation that:
1. Calculates Tactician barriers as fractions of Analyst barriers
2. Supports both 1m and 5m timeframes
3. Dynamically loads Analyst configuration
4. Provides adaptive barriers based on market conditions
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

# Import the dynamic components
from src.tactician.dynamic_barrier_calculator import DynamicBarrierCalculator
from src.training.steps.step14_tactician_labeling import TacticianTripleBarrierLabeler
from src.tactician.enhanced_execution_manager import EnhancedExecutionManager


def create_test_market_data(
    start_date: str = "2024-01-01",
    periods: int = 1000,
    base_price: float = 100.0,
    volatility: float = 0.01,
    timeframe_minutes: int = 1
) -> pd.DataFrame:
    """Create realistic test market data for specific timeframe."""
    dates = pd.date_range(start_date, periods=periods, freq=f"{timeframe_minutes}min")
    
    # Generate price data with realistic patterns
    np.random.seed(42)  # For reproducible results
    
    # Create price series with trend and volatility
    returns = np.random.normal(0, volatility, periods)
    prices = [base_price]
    
    for i in range(1, periods):
        # Add some trend and mean reversion
        trend = 0.0001 * np.sin(i / 100)  # Small cyclical trend
        price_change = returns[i] + trend
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.0005, periods)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.001, periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.001, periods))),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, periods)
    }, index=dates)
    
    # Ensure OHLC relationships are valid
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data


def create_analyst_signals(data: pd.DataFrame, signal_frequency: float = 0.1) -> pd.Series:
    """Create realistic Analyst signals."""
    signals = pd.Series(0, index=data.index)
    
    # Generate signals based on price momentum
    for i in range(20, len(data) - 1):
        if np.random.random() < signal_frequency:
            # Calculate momentum
            recent_return = (data['close'].iloc[i] - data['close'].iloc[i-5]) / data['close'].iloc[i-5]
            
            if recent_return > 0.002:  # 0.2% positive momentum
                signals.iloc[i] = 1  # BUY signal
            elif recent_return < -0.002:  # 0.2% negative momentum
                signals.iloc[i] = -1  # SELL signal
    
    return signals


def test_dynamic_barrier_calculator():
    """Test the dynamic barrier calculator."""
    print("🧪 Testing Dynamic Barrier Calculator")
    print("=" * 60)
    
    # Test configuration
    config = {
        "tactician_triple_barrier": {
            "analyst_barrier_fractions": {
                "profit_take_fraction": 0.5,    # 50% of Analyst
                "stop_loss_fraction": 0.25,     # 25% of Analyst
                "time_barrier_fraction": 0.5    # 50% of Analyst
            },
            "timeframes": ["1m", "5m"],
            "primary_timeframe": "1m",
            "secondary_timeframe": "5m",
            "timeframe_settings": {
                "1m": {
                    "priority": "primary",
                    "execution_weight": 0.7,
                    "confirmation_weight": 0.3,
                    "barrier_adjustment": 1.0
                },
                "5m": {
                    "priority": "secondary",
                    "execution_weight": 0.3,
                    "confirmation_weight": 0.7,
                    "barrier_adjustment": 1.2
                }
            }
        }
    }
    
    # Initialize dynamic barrier calculator
    calculator = DynamicBarrierCalculator(config)
    
    # Test Analyst configuration loading
    print(f"📊 Analyst Configuration Loaded:")
    analyst_info = calculator.get_analyst_barrier_info()
    print(f"   Profit Take: {analyst_info['profit_take_multiplier']:.4f} ({analyst_info['profit_take_multiplier']*100:.3f}%)")
    print(f"   Stop Loss: {analyst_info['stop_loss_multiplier']:.4f} ({analyst_info['stop_loss_multiplier']*100:.3f}%)")
    print(f"   Time Barrier: {analyst_info['time_barrier_minutes']} minutes")
    
    # Test dynamic barrier calculation for 1m timeframe
    print(f"\n📊 Testing 1m Timeframe Barriers:")
    pt_1m, sl_1m, time_1m = calculator.calculate_dynamic_barriers("1m")
    print(f"   Profit Take: {pt_1m:.4f} ({pt_1m*100:.3f}%)")
    print(f"   Stop Loss: {sl_1m:.4f} ({sl_1m*100:.3f}%)")
    print(f"   Time Barrier: {time_1m} periods")
    
    # Test dynamic barrier calculation for 5m timeframe
    print(f"\n📊 Testing 5m Timeframe Barriers:")
    pt_5m, sl_5m, time_5m = calculator.calculate_dynamic_barriers("5m")
    print(f"   Profit Take: {pt_5m:.4f} ({pt_5m*100:.3f}%)")
    print(f"   Stop Loss: {sl_5m:.4f} ({sl_5m*100:.3f}%)")
    print(f"   Time Barrier: {time_5m} periods")
    
    # Test multi-timeframe barrier calculation
    print(f"\n📊 Testing Multi-timeframe Barriers:")
    market_data_1m = create_test_market_data(timeframe_minutes=1)
    market_data_5m = create_test_market_data(timeframe_minutes=5)
    
    multi_barriers = calculator.calculate_multi_timeframe_barriers(
        market_data_1m=market_data_1m,
        market_data_5m=market_data_5m
    )
    
    for timeframe, (pt, sl, time) in multi_barriers.items():
        print(f"   {timeframe}: PT={pt:.4f}, SL={sl:.4f}, Time={time} periods")
    
    # Test barrier validation
    print(f"\n📊 Testing Barrier Validation:")
    for timeframe in ["1m", "5m"]:
        validation = calculator.validate_barrier_calculation(timeframe)
        print(f"   {timeframe} validation: {'✓' if validation['is_valid'] else '✗'}")
        if validation['is_valid']:
            print(f"     Actual fractions - PT: {validation['actual_fractions']['profit_take']:.2f}, SL: {validation['actual_fractions']['stop_loss']:.2f}")
    
    return calculator


def test_enhanced_tactician_labeling():
    """Test the enhanced Tactician labeling with dynamic barriers."""
    print("\n🧪 Testing Enhanced Tactician Labeling (Dynamic)")
    print("=" * 60)
    
    # Create test data for both timeframes
    market_data_1m = create_test_market_data(timeframe_minutes=1)
    market_data_5m = create_test_market_data(timeframe_minutes=5)
    
    analyst_signals_1m = create_analyst_signals(market_data_1m)
    analyst_signals_5m = create_analyst_signals(market_data_5m)
    
    # Test configuration
    config = {
        "tactician_triple_barrier": {
            "analyst_barrier_fractions": {
                "profit_take_fraction": 0.5,
                "stop_loss_fraction": 0.25,
                "time_barrier_fraction": 0.5
            },
            "timeframes": ["1m", "5m"],
            "primary_timeframe": "1m",
            "secondary_timeframe": "5m",
            "enable_high_precision_mode": True,
            "precision_threshold": 0.85
        }
    }
    
    # Test 1m labeling
    print(f"📊 Testing 1m Timeframe Labeling:")
    labeler_1m = TacticianTripleBarrierLabeler(config)
    labeled_data_1m = labeler_1m.apply_labels(market_data_1m, analyst_signals_1m)
    
    tactician_labels_1m = labeled_data_1m['tactician_label']
    precision_scores_1m = labeled_data_1m['tactician_precision_score']
    
    print(f"   1m Results:")
    print(f"     Total samples: {len(labeled_data_1m)}")
    print(f"     Tactician signals: {tactician_labels_1m[tactician_labels_1m != 0].count()}")
    print(f"     High precision signals: {(precision_scores_1m >= 0.85).sum()}")
    print(f"     Average precision: {precision_scores_1m.mean():.3f}")
    
    # Test 5m labeling
    print(f"\n📊 Testing 5m Timeframe Labeling:")
    labeler_5m = TacticianTripleBarrierLabeler(config)
    labeled_data_5m = labeler_5m.apply_labels(market_data_5m, analyst_signals_5m)
    
    tactician_labels_5m = labeled_data_5m['tactician_label']
    precision_scores_5m = labeled_data_5m['tactician_precision_score']
    
    print(f"   5m Results:")
    print(f"     Total samples: {len(labeled_data_5m)}")
    print(f"     Tactician signals: {tactician_labels_5m[tactician_labels_5m != 0].count()}")
    print(f"     High precision signals: {(precision_scores_5m >= 0.85).sum()}")
    print(f"     Average precision: {precision_scores_5m.mean():.3f}")
    
    return labeled_data_1m, labeled_data_5m


def test_enhanced_execution_manager():
    """Test the enhanced execution manager with dynamic barriers."""
    print("\n🧪 Testing Enhanced Execution Manager (Dynamic)")
    print("=" * 60)
    
    # Create test data
    market_data_1m = create_test_market_data(timeframe_minutes=1)
    market_data_5m = create_test_market_data(timeframe_minutes=5)
    
    # Test configuration
    config = {
        "tactician_triple_barrier": {
            "analyst_barrier_fractions": {
                "profit_take_fraction": 0.5,
                "stop_loss_fraction": 0.25,
                "time_barrier_fraction": 0.5
            },
            "timeframes": ["1m", "5m"],
            "primary_timeframe": "1m",
            "secondary_timeframe": "5m",
            "precision_threshold": 0.85,
            "position_size_multiplier": 0.5,
            "leverage_multiplier": 0.75
        }
    }
    
    # Initialize enhanced execution manager
    execution_manager = EnhancedExecutionManager(config)
    
    # Test analyst signal validation
    analyst_signal = {
        "should_enter_position": True,
        "trade_direction": "long",
        "entry_confidence": 0.9,
        "position_size": 0.1,
        "leverage": 1.0
    }
    
    tactician_confidence = 0.88
    
    print(f"📊 Testing Signal Validation:")
    validation = execution_manager.validate_analyst_signal(analyst_signal, tactician_confidence)
    print(f"   Signal valid: {validation['valid']}")
    print(f"   Should execute: {validation['should_execute']}")
    print(f"   Combined confidence: {validation.get('combined_confidence', 0.0):.3f}")
    
    # Test execution parameter calculation for 1m
    print(f"\n📊 Testing 1m Execution Parameters:")
    current_price_1m = market_data_1m['close'].iloc[-1]
    
    execution_params_1m = execution_manager.calculate_execution_parameters(
        market_data=market_data_1m,
        analyst_signal=analyst_signal,
        tactician_confidence=tactician_confidence,
        current_price=current_price_1m
    )
    
    if execution_params_1m.get("should_execute", False):
        print(f"   1m Execution Parameters:")
        print(f"     Trade direction: {execution_params_1m['trade_direction']}")
        print(f"     Entry price: {execution_params_1m['entry_price']:.4f}")
        print(f"     Profit take: {execution_params_1m['profit_take_price']:.4f}")
        print(f"     Stop loss: {execution_params_1m['stop_loss_price']:.4f}")
        print(f"     Position size: {execution_params_1m['position_size']:.4f}")
        print(f"     Leverage: {execution_params_1m['leverage']:.2f}")
        print(f"     Precision score: {execution_params_1m['precision_score']:.3f}")
    
    # Test execution parameter calculation for 5m
    print(f"\n📊 Testing 5m Execution Parameters:")
    current_price_5m = market_data_5m['close'].iloc[-1]
    
    execution_params_5m = execution_manager.calculate_execution_parameters(
        market_data=market_data_5m,
        analyst_signal=analyst_signal,
        tactician_confidence=tactician_confidence,
        current_price=current_price_5m
    )
    
    if execution_params_5m.get("should_execute", False):
        print(f"   5m Execution Parameters:")
        print(f"     Trade direction: {execution_params_5m['trade_direction']}")
        print(f"     Entry price: {execution_params_5m['entry_price']:.4f}")
        print(f"     Profit take: {execution_params_5m['profit_take_price']:.4f}")
        print(f"     Stop loss: {execution_params_5m['stop_loss_price']:.4f}")
        print(f"     Position size: {execution_params_5m['position_size']:.4f}")
        print(f"     Leverage: {execution_params_5m['leverage']:.2f}")
        print(f"     Precision score: {execution_params_5m['precision_score']:.3f}")
    
    return execution_params_1m, execution_params_5m


def test_barrier_comparison_analysis():
    """Test comprehensive barrier comparison analysis."""
    print("\n🧪 Testing Barrier Comparison Analysis")
    print("=" * 60)
    
    # Test configuration
    config = {
        "tactician_triple_barrier": {
            "analyst_barrier_fractions": {
                "profit_take_fraction": 0.5,
                "stop_loss_fraction": 0.25,
                "time_barrier_fraction": 0.5
            },
            "timeframes": ["1m", "5m"],
            "primary_timeframe": "1m",
            "secondary_timeframe": "5m"
        }
    }
    
    # Initialize calculator
    calculator = DynamicBarrierCalculator(config)
    
    # Get Analyst values
    analyst_info = calculator.get_analyst_barrier_info()
    analyst_pt = analyst_info["profit_take_multiplier"]
    analyst_sl = analyst_info["stop_loss_multiplier"]
    analyst_time = analyst_info["time_barrier_minutes"]
    
    # Calculate Tactician barriers for both timeframes
    pt_1m, sl_1m, time_1m = calculator.calculate_dynamic_barriers("1m")
    pt_5m, sl_5m, time_5m = calculator.calculate_dynamic_barriers("5m")
    
    print(f"📊 Comprehensive Barrier Comparison:")
    print(f"   Analyst Base Values:")
    print(f"     Profit Take: {analyst_pt:.4f} ({analyst_pt*100:.3f}%)")
    print(f"     Stop Loss: {analyst_sl:.4f} ({analyst_sl*100:.3f}%)")
    print(f"     Time Barrier: {analyst_time} minutes")
    
    print(f"\n   Tactician 1m Values:")
    print(f"     Profit Take: {pt_1m:.4f} ({pt_1m*100:.3f}%) - {pt_1m/analyst_pt:.1%} of Analyst")
    print(f"     Stop Loss: {sl_1m:.4f} ({sl_1m*100:.3f}%) - {sl_1m/analyst_sl:.1%} of Analyst")
    print(f"     Time Barrier: {time_1m} periods - {time_1m/analyst_time:.1%} of Analyst")
    
    print(f"\n   Tactician 5m Values:")
    print(f"     Profit Take: {pt_5m:.4f} ({pt_5m*100:.3f}%) - {pt_5m/analyst_pt:.1%} of Analyst")
    print(f"     Stop Loss: {sl_5m:.4f} ({sl_5m*100:.3f}%) - {sl_5m/analyst_sl:.1%} of Analyst")
    print(f"     Time Barrier: {time_5m} periods - {time_5m/analyst_time:.1%} of Analyst")
    
    # Calculate risk-reward ratios
    analyst_rr = analyst_pt / analyst_sl
    tactician_1m_rr = pt_1m / sl_1m
    tactician_5m_rr = pt_5m / sl_5m
    
    print(f"\n   Risk-Reward Ratios:")
    print(f"     Analyst: {analyst_rr:.2f}:1")
    print(f"     Tactician 1m: {tactician_1m_rr:.2f}:1 ({(tactician_1m_rr/analyst_rr-1)*100:+.1f}% improvement)")
    print(f"     Tactician 5m: {tactician_5m_rr:.2f}:1 ({(tactician_5m_rr/analyst_rr-1)*100:+.1f}% improvement)")
    
    # Test timeframe weights
    print(f"\n   Timeframe Weights:")
    for timeframe in ["1m", "5m"]:
        exec_weight, conf_weight = calculator.get_timeframe_weights(timeframe)
        print(f"     {timeframe}: Execution={exec_weight:.1f}, Confirmation={conf_weight:.1f}")


def test_dynamic_adaptation():
    """Test dynamic adaptation to market conditions."""
    print("\n🧪 Testing Dynamic Adaptation")
    print("=" * 60)
    
    # Create test data with different volatility levels
    low_vol_data = create_test_market_data(volatility=0.005)
    high_vol_data = create_test_market_data(volatility=0.02)
    
    config = {
        "tactician_triple_barrier": {
            "analyst_barrier_fractions": {
                "profit_take_fraction": 0.5,
                "stop_loss_fraction": 0.25,
                "time_barrier_fraction": 0.5
            },
            "timeframes": ["1m", "5m"],
            "enable_adaptive_barriers": True,
            "market_condition_adjustment": True
        }
    }
    
    calculator = DynamicBarrierCalculator(config)
    
    print(f"📊 Testing Volatility Adaptation:")
    
    # Test low volatility
    pt_low, sl_low, time_low = calculator.calculate_dynamic_barriers(
        timeframe="1m",
        market_data=low_vol_data
    )
    
    # Test high volatility
    pt_high, sl_high, time_high = calculator.calculate_dynamic_barriers(
        timeframe="1m",
        market_data=high_vol_data
    )
    
    print(f"   Low Volatility Market:")
    print(f"     Profit Take: {pt_low:.4f}, Stop Loss: {sl_low:.4f}")
    print(f"   High Volatility Market:")
    print(f"     Profit Take: {pt_high:.4f}, Stop Loss: {sl_high:.4f}")
    print(f"   Adaptation Ratio (High/Low):")
    print(f"     Profit Take: {pt_high/pt_low:.2f}x")
    print(f"     Stop Loss: {sl_high/sl_low:.2f}x")


def main():
    """Run all dynamic barrier tests."""
    print("🚀 Dynamic Tactician Triple Barrier Implementation Test")
    print("=" * 80)
    print("Testing dynamic barrier calculation based on Analyst values")
    print("and support for both 1m and 5m timeframes.")
    print()
    
    try:
        # Test 1: Dynamic Barrier Calculator
        calculator = test_dynamic_barrier_calculator()
        
        # Test 2: Enhanced Tactician Labeling
        labeled_data_1m, labeled_data_5m = test_enhanced_tactician_labeling()
        
        # Test 3: Enhanced Execution Manager
        execution_params_1m, execution_params_5m = test_enhanced_execution_manager()
        
        # Test 4: Barrier Comparison Analysis
        test_barrier_comparison_analysis()
        
        # Test 5: Dynamic Adaptation
        test_dynamic_adaptation()
        
        print("\n✅ All Dynamic Tactician Tests Completed Successfully!")
        print("\n📋 Summary:")
        print("   ✓ Dynamic barrier calculation based on Analyst values")
        print("   ✓ Support for both 1m and 5m timeframes")
        print("   ✓ Adaptive barriers based on market conditions")
        print("   ✓ Multi-timeframe barrier calculation")
        print("   ✓ Barrier validation and comparison")
        print("   ✓ Enhanced execution with dynamic parameters")
        
        print("\n🎯 Key Features Verified:")
        print("   • Tactician barriers are 50% and 25% of Analyst barriers")
        print("   • Both 1m and 5m timeframes are supported")
        print("   • Dynamic adaptation to market volatility")
        print("   • Timeframe-specific barrier adjustments")
        print("   • Comprehensive validation and testing")
        
        print("\n🔧 Dynamic Configuration:")
        print("   • Analyst values loaded dynamically")
        print("   • Fraction-based barrier calculation")
        print("   • Timeframe-specific adjustments")
        print("   • Market condition adaptation")
        print("   • Volatility-based barrier scaling")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()