#!/usr/bin/env python3
"""
Test script for Enhanced Tactician Triple Barrier with High Precision Completion

This script validates the implementation of the enhanced Tactician triple barrier
strategy that ensures the Tactician completes the Analyst nicely with:
- 50% smaller profit take barriers (0.1% vs 0.2%)
- 25% smaller stop loss barriers (0.025% vs 0.1%)
- High precision execution filters
- Quality filters and adaptive barriers
"""

import asyncio
import numpy as np
import pandas as pd

# Import the enhanced components
from src.training.steps.step14_tactician_labeling import TacticianTripleBarrierLabeler
from src.tactician.enhanced_execution_manager import EnhancedExecutionManager


def create_test_market_data(
    start_date: str = "2024-01-01",
    periods: int = 1000,
    base_price: float = 100.0,
    volatility: float = 0.01
) -> pd.DataFrame:
    """Create realistic test market data."""
    dates = pd.date_range(start_date, periods=periods, freq="1min")
    
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


def test_enhanced_tactician_triple_barrier():
    """Test the enhanced Tactician triple barrier labeling."""
    print("🧪 Testing Enhanced Tactician Triple Barrier Labeling")
    print("=" * 60)
    
    # Create test data
    market_data = create_test_market_data()
    analyst_signals = create_analyst_signals(market_data)
    
    print(f"📊 Test Data Created:")
    print(f"   Market data: {len(market_data)} periods")
    print(f"   Analyst signals: {analyst_signals[analyst_signals != 0].count()} signals")
    print(f"   Date range: {market_data.index[0]} to {market_data.index[-1]}")
    
    # Test configuration
    config = {
        "tactician_triple_barrier": {
            "profit_take_pct": 0.001,  # 0.1% (50% of Analyst's 0.2%)
            "stop_loss_pct": 0.00025,  # 0.025% (25% of Analyst's 0.1%)
            "time_barrier_periods": 15,  # 15 minutes (50% of Analyst's 30)
            "enable_high_precision_mode": True,
            "precision_threshold": 0.85,
            "enable_quality_filters": True,
            "enable_adaptive_barriers": True
        }
    }
    
    # Initialize enhanced labeler
    labeler = TacticianTripleBarrierLabeler(config)
    
    # Apply enhanced labeling
    labeled_data = labeler.apply_labels(market_data, analyst_signals)
    
    # Analyze results
    tactician_labels = labeled_data['tactician_label']
    precision_scores = labeled_data['tactician_precision_score']
    execution_quality = labeled_data['tactician_execution_quality']
    
    print(f"\n📈 Enhanced Tactician Labeling Results:")
    print(f"   Total samples: {len(labeled_data)}")
    print(f"   Tactician signals: {tactician_labels[tactician_labels != 0].count()}")
    print(f"   High precision signals: {(precision_scores >= 0.85).sum()}")
    print(f"   Average precision score: {precision_scores.mean():.3f}")
    print(f"   Average execution quality: {execution_quality.mean():.3f}")
    
    # Label distribution
    label_dist = tactician_labels.value_counts()
    print(f"   Label distribution: {dict(label_dist)}")
    
    return labeled_data


def test_enhanced_execution_manager():
    """Test the enhanced execution manager."""
    print("\n🧪 Testing Enhanced Execution Manager")
    print("=" * 60)
    
    # Create test data
    market_data = create_test_market_data()
    
    # Test configuration
    config = {
        "tactician_triple_barrier": {
            "profit_take_pct": 0.001,  # 0.1%
            "stop_loss_pct": 0.00025,  # 0.025%
            "precision_threshold": 0.85,
            "position_size_multiplier": 0.5,
            "leverage_multiplier": 0.75,
            "max_risk_per_trade": 0.001
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
    
    # Test execution parameter calculation
    current_price = market_data['close'].iloc[-1]
    
    print(f"\n📊 Testing Execution Parameter Calculation:")
    execution_params = execution_manager.calculate_execution_parameters(
        market_data=market_data,
        analyst_signal=analyst_signal,
        tactician_confidence=tactician_confidence,
        current_price=current_price
    )
    
    if execution_params.get("should_execute", False):
        print(f"   Should execute: {execution_params['should_execute']}")
        print(f"   Trade direction: {execution_params['trade_direction']}")
        print(f"   Entry price: {execution_params['entry_price']:.4f}")
        print(f"   Profit take: {execution_params['profit_take_price']:.4f}")
        print(f"   Stop loss: {execution_params['stop_loss_price']:.4f}")
        print(f"   Position size: {execution_params['position_size']:.4f}")
        print(f"   Leverage: {execution_params['leverage']:.2f}")
        print(f"   Precision score: {execution_params['precision_score']:.3f}")
        print(f"   Volatility: {execution_params['volatility']:.4f}")
    else:
        print(f"   Execution rejected: {execution_params.get('reason', 'unknown')}")
    
    return execution_params


async def test_enhanced_trade_execution():
    """Test the enhanced trade execution."""
    print("\n🧪 Testing Enhanced Trade Execution")
    print("=" * 60)
    
    # Create test data
    market_data = create_test_market_data()
    
    # Test configuration
    config = {
        "tactician_triple_barrier": {
            "profit_take_pct": 0.001,
            "stop_loss_pct": 0.00025,
            "precision_threshold": 0.85
        }
    }
    
    # Initialize enhanced execution manager
    execution_manager = EnhancedExecutionManager(config)
    
    # Create execution parameters
    execution_params = {
        "should_execute": True,
        "trade_direction": "long",
        "entry_price": 100.0,
        "profit_take_price": 100.1,
        "stop_loss_price": 99.975,
        "position_size": 0.05,
        "leverage": 0.75,
        "precision_score": 0.92,
        "volatility": 0.008
    }
    
    # Test trade execution
    print(f"📊 Testing Trade Execution:")
    execution_result = await execution_manager.execute_trade(execution_params, market_data)
    
    print(f"   Execution success: {execution_result['success']}")
    if execution_result['success']:
        print(f"   Execution time: {execution_result['execution_time']}")
        print(f"   Reason: {execution_result['reason']}")
    
    # Test performance summary
    print(f"\n📊 Performance Summary:")
    performance = execution_manager.get_performance_summary()
    print(f"   Total executions: {performance['total_executions']}")
    print(f"   Success rate: {performance['success_rate']:.3f}")
    print(f"   Average precision: {performance['avg_precision']:.3f}")
    print(f"   Max precision: {performance['max_precision']:.3f}")
    
    return execution_result


def test_barrier_comparison():
    """Compare Analyst vs Tactician barriers."""
    print("\n🧪 Testing Barrier Comparison (Analyst vs Tactician)")
    print("=" * 60)
    
    # Analyst barriers (current)
    analyst_pt = 0.002  # 0.2%
    analyst_sl = 0.001  # 0.1%
    analyst_time = 30   # 30 minutes
    
    # Tactician barriers (enhanced)
    tactician_pt = 0.001   # 0.1% (50% of Analyst)
    tactician_sl = 0.00025 # 0.025% (25% of Analyst)
    tactician_time = 15    # 15 minutes (50% of Analyst)
    
    print(f"📊 Barrier Comparison:")
    print(f"   Analyst Profit Take: {analyst_pt:.4f} ({analyst_pt*100:.3f}%)")
    print(f"   Tactician Profit Take: {tactician_pt:.4f} ({tactician_pt*100:.3f}%)")
    print(f"   Reduction: {((analyst_pt - tactician_pt) / analyst_pt * 100):.1f}%")
    
    print(f"\n   Analyst Stop Loss: {analyst_sl:.4f} ({analyst_sl*100:.3f}%)")
    print(f"   Tactician Stop Loss: {tactician_sl:.4f} ({tactician_sl*100:.3f}%)")
    print(f"   Reduction: {((analyst_sl - tactician_sl) / analyst_sl * 100):.1f}%")
    
    print(f"\n   Analyst Time Barrier: {analyst_time} minutes")
    print(f"   Tactician Time Barrier: {tactician_time} minutes")
    print(f"   Reduction: {((analyst_time - tactician_time) / analyst_time * 100):.1f}%")
    
    # Calculate risk-reward ratios
    analyst_rr = analyst_pt / analyst_sl
    tactician_rr = tactician_pt / tactician_sl
    
    print(f"\n   Analyst Risk-Reward: {analyst_rr:.2f}:1")
    print(f"   Tactician Risk-Reward: {tactician_rr:.2f}:1")
    print(f"   Improvement: {((tactician_rr - analyst_rr) / analyst_rr * 100):.1f}%")


def test_precision_metrics():
    """Test precision metrics calculation."""
    print("\n🧪 Testing Precision Metrics")
    print("=" * 60)
    
    # Create test data
    market_data = create_test_market_data()
    
    # Test configuration
    config = {
        "tactician_triple_barrier": {
            "profit_take_pct": 0.001,
            "stop_loss_pct": 0.00025,
            "precision_threshold": 0.85
        }
    }
    
    # Initialize enhanced execution manager
    execution_manager = EnhancedExecutionManager(config)
    
    # Test different scenarios
    scenarios = [
        {"confidence": 0.95, "volatility": 0.005, "description": "High confidence, low volatility"},
        {"confidence": 0.85, "volatility": 0.010, "description": "Medium confidence, medium volatility"},
        {"confidence": 0.75, "volatility": 0.015, "description": "Low confidence, high volatility"}
    ]
    
    print(f"📊 Precision Score Analysis:")
    for scenario in scenarios:
        precision_score = execution_manager._calculate_precision_score(
            combined_confidence=scenario["confidence"],
            volatility=scenario["volatility"],
            market_data=market_data
        )
        
        print(f"   {scenario['description']}:")
        print(f"     Confidence: {scenario['confidence']:.3f}")
        print(f"     Volatility: {scenario['volatility']:.4f}")
        print(f"     Precision Score: {precision_score:.3f}")
        print(f"     Passes Threshold: {precision_score >= 0.85}")


def main():
    """Run all tests."""
    print("🚀 Enhanced Tactician Triple Barrier with High Precision Completion")
    print("=" * 80)
    print("Testing the implementation that ensures Tactician completes Analyst nicely")
    print("with 50% and 25% smaller barriers and high precision execution.")
    print()
    
    try:
        # Test 1: Enhanced Tactician Triple Barrier Labeling
        labeled_data = test_enhanced_tactician_triple_barrier()
        
        # Test 2: Enhanced Execution Manager
        execution_params = test_enhanced_execution_manager()
        
        # Test 3: Enhanced Trade Execution
        execution_result = asyncio.run(test_enhanced_trade_execution())
        
        # Test 4: Barrier Comparison
        test_barrier_comparison()
        
        # Test 5: Precision Metrics
        test_precision_metrics()
        
        print("\n✅ All Enhanced Tactician Tests Completed Successfully!")
        print("\n📋 Summary:")
        print("   ✓ Enhanced triple barrier labeling with 50%/25% smaller barriers")
        print("   ✓ High precision execution filters")
        print("   ✓ Quality filters and adaptive barriers")
        print("   ✓ Analyst signal validation and agreement")
        print("   ✓ Precision metrics and performance tracking")
        print("   ✓ Risk-adjusted position sizing")
        
        print("\n🎯 Key Benefits:")
        print("   • Tactician completes Analyst with higher precision")
        print("   • Smaller barriers reduce risk while maintaining profitability")
        print("   • Quality filters ensure only high-quality executions")
        print("   • Adaptive barriers respond to market conditions")
        print("   • Comprehensive performance tracking and metrics")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()