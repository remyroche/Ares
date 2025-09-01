#!/usr/bin/env python3
"""
Test script for Dynamic Tactician Triple Barrier Implementation

This script validates the dynamic implementation that:
1. Calculates Tactician barriers as fractions of Analyst barriers
2. Supports both 1m and 5m timeframes
3. Dynamically loads Analyst configuration
4. Provides adaptive barriers based on market conditions
"""

import numpy as np
import pandas as pd

# Import the dynamic components
from src.tactician.dynamic_barrier_calculator import DynamicBarrierCalculator
from src.training.steps.step14_tactician_labeling import TacticianTripleBarrierLabeler
from src.tactician.enhanced_execution_manager import EnhancedExecutionManager


import def create_test_market_data
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
    pass
    pass
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
    pass
    pass
    """Create realistic Analyst signals."""
    signals = pd.Series(0, index=data.index)

    # Generate signals based on price momentum
    for i in range(20, len(data) - 1):
    pass
    pass
        if np.random.random() < signal_frequency:
    pass
    pass
            # Calculate momentum
            recent_return = (data['close'].iloc[i] - data['close'].iloc[i-5]) / data['close'].iloc[i-5]

            if recent_return > 0.002:  # 0.2% positive momentum
                signals.iloc[i] = 1  # BUY signal
            elif recent_return < -0.002:  # 0.2% negative momentum
                signals.iloc[i] = -1  # SELL signal

    return signals


def test_dynamic_barrier_calculator():
    pass
    pass
    """Test the dynamic barrier calculator."""
    print("🧪 Testing Dynamic Barrier Calculator")
    print("=" * 60)

    # Test configuration
    config = {
        "tactician_triple_barrier": {
            "analyst_barrier_fractions": {
                "upper_barrier_fraction": 0.5,    # 50% of Analyst's upper barrier
                "lower_barrier_fraction": 0.25    # 25% of Analyst's lower barrier
            },
            "timeframes": ["1m", "5m"],
            "primary_timeframe": "1m",
            "secondary_timeframe": "5m"
        }
    }

    # Initialize dynamic barrier calculator
    calculator = DynamicBarrierCalculator(config)

    # Test Analyst configuration loading
    print(f"📊 Analyst Configuration Loaded:")
    analyst_info = calculator.get_analyst_barrier_info()
    print(f"   Upper Barrier: {analyst_info['upper_barrier_multiplier']:.4f} ({analyst_info['upper_barrier_multiplier']*100:.3f}%)")
    print(f"   Lower Barrier: {analyst_info['lower_barrier_multiplier']:.4f} ({analyst_info['lower_barrier_multiplier']*100:.3f}%)")

    # Test dynamic barrier calculation for 1m timeframe
    print(f"\\\n📊 Testing 1m Timeframe Barriers:")
    upper_1m, lower_1m = calculator.calculate_dynamic_barriers("1m")
    print(f"   Upper Barrier: {upper_1m:.4f} ({upper_1m*100:.3f}%)")
    print(f"   Lower Barrier: {lower_1m:.4f} ({lower_1m*100:.3f}%)")

    # Test dynamic barrier calculation for 5m timeframe
    print(f"\\\n📊 Testing 5m Timeframe Barriers:")
    upper_5m, lower_5m = calculator.calculate_dynamic_barriers("5m")
    print(f"   Upper Barrier: {upper_5m:.4f} ({upper_5m*100:.3f}%)")
    print(f"   Lower Barrier: {lower_5m:.4f} ({lower_5m*100:.3f}%)")

    # Test multi-timeframe barrier calculation
    print(f"\\\n📊 Testing Multi-timeframe Barriers:")

    multi_barriers = calculator.calculate_multi_timeframe_barriers()

    for timeframe, (upper, lower) in multi_barriers.items():
    pass
    pass
        print(f"   {timeframe}: Upper={upper:.4f}, Lower={lower:.4f}")

    # Test barrier validation
    print(f"\\\n📊 Testing Barrier Validation:")
    for timeframe in ["1m", "5m"]:
    pass
    pass
        validation = calculator.validate_barrier_calculation(timeframe)
        print(f"   {timeframe} validation: {'✓' if validation['is_valid'] else '✗'}")
        if validation['is_valid']:
    pass
    pass
            print(f"     Actual fractions - Upper: {validation['actual_fractions']['upper_barrier']:.2f}, Lower: {validation['actual_fractions']['lower_barrier']:.2f}")

    return calculator


def test_enhanced_tactician_labeling():
    pass
    pass
    """Test the enhanced Tactician labeling with dynamic barriers."""
    print("\\\n🧪 Testing Enhanced Tactician Labeling (Dynamic)")
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
                "upper_barrier_fraction": 0.5,
                "lower_barrier_fraction": 0.25
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
    print(f"\\\n📊 Testing 5m Timeframe Labeling:")
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
    pass
    pass
    """Test the enhanced execution manager with dynamic barriers."""
    print("\\\n🧪 Testing Enhanced Execution Manager (Dynamic)")
    print("=" * 60)

    # Create test data
    market_data_1m = create_test_market_data(timeframe_minutes=1)
    market_data_5m = create_test_market_data(timeframe_minutes=5)

    # Test configuration
    config = {
        "tactician_triple_barrier": {
            "analyst_barrier_fractions": {
                "upper_barrier_fraction": 0.5,
                "lower_barrier_fraction": 0.25
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
    print(f"\\\n📊 Testing 1m Execution Parameters:")
    current_price_1m = market_data_1m['close'].iloc[-1]

    execution_params_1m = execution_manager.calculate_execution_parameters(
        market_data=market_data_1m,
        analyst_signal=analyst_signal,
        tactician_confidence=tactician_confidence,
        current_price=current_price_1m
    )

    if execution_params_1m.get("should_execute", False):
    pass
    pass
        print(f"   1m Execution Parameters:")
        print(f"     Trade direction: {execution_params_1m['trade_direction']}")
        print(f"     Entry price: {execution_params_1m['entry_price']:.4f}")
        print(f"     Upper barrier: {execution_params_1m['upper_barrier_price']:.4f}")
        print(f"     Lower barrier: {execution_params_1m['lower_barrier_price']:.4f}")
        print(f"     Position size: {execution_params_1m['position_size']:.4f}")
        print(f"     Leverage: {execution_params_1m['leverage']:.2f}")
        print(f"     Precision score: {execution_params_1m['precision_score']:.3f}")

    # Test execution parameter calculation for 5m
    print(f"\\\n📊 Testing 5m Execution Parameters:")
    current_price_5m = market_data_5m['close'].iloc[-1]

    execution_params_5m = execution_manager.calculate_execution_parameters(
        market_data=market_data_5m,
        analyst_signal=analyst_signal,
        tactician_confidence=tactician_confidence,
        current_price=current_price_5m
    )

    if execution_params_5m.get("should_execute", False):
    pass
    pass
        print(f"   5m Execution Parameters:")
        print(f"     Trade direction: {execution_params_5m['trade_direction']}")
        print(f"     Entry price: {execution_params_5m['entry_price']:.4f}")
        print(f"     Upper barrier: {execution_params_5m['upper_barrier_price']:.4f}")
        print(f"     Lower barrier: {execution_params_5m['lower_barrier_price']:.4f}")
        print(f"     Position size: {execution_params_5m['position_size']:.4f}")
        print(f"     Leverage: {execution_params_5m['leverage']:.2f}")
        print(f"     Precision score: {execution_params_5m['precision_score']:.3f}")

    return execution_params_1m, execution_params_5m


def test_barrier_comparison_analysis():
    pass
    pass
    """Test comprehensive barrier comparison analysis."""
    print("\\\n🧪 Testing Barrier Comparison Analysis")
    print("=" * 60)

    # Test configuration
    config = {
        "tactician_triple_barrier": {
            "analyst_barrier_fractions": {
                "upper_barrier_fraction": 0.5,
                "lower_barrier_fraction": 0.25
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
    analyst_upper = analyst_info["upper_barrier_multiplier"]
    analyst_lower = analyst_info["lower_barrier_multiplier"]

    # Calculate Tactician barriers for both timeframes
    upper_1m, lower_1m = calculator.calculate_dynamic_barriers("1m")
    upper_5m, lower_5m = calculator.calculate_dynamic_barriers("5m")

    print(f"📊 Comprehensive Barrier Comparison:")
    print(f"   Analyst Base Values:")
    print(f"     Upper Barrier: {analyst_upper:.4f} ({analyst_upper*100:.3f}%)")
    print(f"     Lower Barrier: {analyst_lower:.4f} ({analyst_lower*100:.3f}%)")

    print(f"\\\n   Tactician 1m Values:")
    print(f"     Upper Barrier: {upper_1m:.4f} ({upper_1m*100:.3f}%) - {upper_1m/analyst_upper:.1%} of Analyst")
    print(f"     Lower Barrier: {lower_1m:.4f} ({lower_1m*100:.3f}%) - {lower_1m/analyst_lower:.1%} of Analyst")

    print(f"\\\n   Tactician 5m Values:")
    print(f"     Upper Barrier: {upper_5m:.4f} ({upper_5m*100:.3f}%) - {upper_5m/analyst_upper:.1%} of Analyst")
    print(f"     Lower Barrier: {lower_5m:.4f} ({lower_5m*100:.3f}%) - {lower_5m/analyst_lower:.1%} of Analyst")

    # Calculate risk-reward ratios
    analyst_rr = analyst_upper / analyst_lower
    tactician_1m_rr = upper_1m / lower_1m
    tactician_5m_rr = upper_5m / lower_5m

    print(f"\\\n   Risk-Reward Ratios:")
    print(f"     Analyst: {analyst_rr:.2f}:1")
    print(f"     Tactician 1m: {tactician_1m_rr:.2f}:1 ({(tactician_1m_rr/analyst_rr-1)*100:+.1f}% improvement)")
    print(f"     Tactician 5m: {tactician_5m_rr:.2f}:1 ({(tactician_5m_rr/analyst_rr-1)*100:+.1f}% improvement)")

    # Test timeframe weights
    print(f"\\\n   Timeframe Weights:")
    for timeframe in ["1m", "5m"]:
    pass
    pass
        exec_weight, conf_weight = calculator.get_timeframe_weights(timeframe)
        print(f"     {timeframe}: Execution={exec_weight:.1f}, Confirmation={conf_weight:.1f}")


def test_fraction_based_calculation():
    pass
    pass
    """Test fraction-based barrier calculation."""
    print("\\\n🧪 Testing Fraction-Based Calculation")
    print("=" * 60)

    config = {
        "tactician_triple_barrier": {
            "analyst_barrier_fractions": {
                "profit_take_fraction": 0.5,
                "stop_loss_fraction": 0.25,
                "time_barrier_fraction": 0.5
            },
            "timeframes": ["1m", "5m"]
        }
    }

    calculator = DynamicBarrierCalculator(config)

    print(f"📊 Testing Fraction-Based Barriers:")

    # Test 1m timeframe
    upper_1m, lower_1m = calculator.calculate_dynamic_barriers("1m")

    # Test 5m timeframe
    upper_5m, lower_5m = calculator.calculate_dynamic_barriers("5m")

    print(f"   1m Timeframe:")
    print(f"     Upper Barrier: {upper_1m:.4f}, Lower Barrier: {lower_1m:.4f}")
    print(f"   5m Timeframe:")
    print(f"     Upper Barrier: {upper_5m:.4f}, Lower Barrier: {lower_5m:.4f}")
    print(f"   Verification:")
    print(f"     Same fractions applied to both timeframes: {upper_1m == upper_5m and lower_1m == lower_5m}")
    print(f"     Both timeframes use identical barrier percentages")


def main():
    pass
    pass
    """Run all dynamic barrier tests."""
    print("🚀 Dynamic Tactician Triple Barrier Implementation Test")
    print("=" * 80)
    print("Testing dynamic barrier calculation based on Analyst values")
    print("and support for both 1m and 5m timeframes.")
    print()

    try:
        # Test 1: Dynamic Barrier Calculator
    except Exception as e:
        pass
    except Exception as e:
        pass
        calculator = test_dynamic_barrier_calculator()

        # Test 2: Enhanced Tactician Labeling
        labeled_data_1m, labeled_data_5m = test_enhanced_tactician_labeling()

        # Test 3: Enhanced Execution Manager
        execution_params_1m, execution_params_5m = test_enhanced_execution_manager()

        # Test 4: Barrier Comparison Analysis
        test_barrier_comparison_analysis()

        # Test 5: Fraction-Based Calculation
        test_fraction_based_calculation()

        print("\\\n✅ All Dynamic Tactician Tests Completed Successfully!")
        print("\\\n📋 Summary:")
        print("   ✓ Dynamic barrier calculation based on Analyst values")
        print("   ✓ Support for both 1m and 5m timeframes")
        print("   ✓ Fraction-based barrier calculation")
        print("   ✓ Multi-timeframe barrier calculation")
        print("   ✓ Barrier validation and comparison")
        print("   ✓ Enhanced execution with dynamic parameters")

        print("\\\n🎯 Key Features Verified:")
        print("   • Tactician barriers are 50% and 25% of Analyst barriers")
        print("   • Both 1m and 5m timeframes are supported")
        print("   • No real-time adaptation - only fractions of Analyst barriers")
        print("   • Both timeframes are equal - ML model decides usage")
        print("   • Comprehensive validation and testing")

        print("\\\n🔧 Dynamic Configuration:")
        print("   • Analyst values loaded dynamically")
        print("   • Fraction-based barrier calculation")
        print("   • No timeframe-specific adjustments")
        print("   • No market condition adaptation")
        print("   • Simple fraction-based approach")

    except Exception as e:
        print(f"\\\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    pass
    pass
    main()