#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced triple barrier method with profit tracking.

This script shows how the triple barrier method now includes potential profit/loss
information when going beyond the set thresholds.
"""

import numpy as np
import pandas as pd
from src.training.steps.step4_analyst_labeling_feature_engineering_components.optimized_triple_barrier_labeling import (
    OptimizedTripleBarrierLabeling
)

def create_sample_data(n_points: int = 1000) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)  # For reproducible results
    
    # Create a trending price series with some volatility
    base_price = 100.0
    trend = np.linspace(0, 0.1, n_points)  # 10% upward trend
    noise = np.random.normal(0, 0.005, n_points)  # 0.5% volatility
    price_changes = trend + noise
    
    close_prices = [base_price]
    for change in price_changes[1:]:
        close_prices.append(close_prices[-1] * (1 + change))
    
    close_prices = np.array(close_prices)
    
    # Generate OHLC data
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.002, n_points)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.002, n_points)))
    open_prices = close_prices * (1 + np.random.normal(0, 0.001, n_points))
    volumes = np.random.uniform(1000, 10000, n_points)
    
    # Ensure OHLC consistency
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
    
    dates = pd.date_range("2024-01-01", periods=n_points, freq="1min")
    
    return pd.DataFrame({
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes
    }, index=dates)

def test_profit_tracking():
    """Test the profit tracking functionality."""
    print("🧪 Testing Triple Barrier Method with Profit Tracking")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data(1000)
    print(f"📊 Created sample data with {len(data)} points")
    print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"   Average volatility: {((data['high'] - data['low']) / data['close']).mean():.4f}")
    print()
    
    # Test 1: Without profit tracking (original behavior)
    print("🔍 Test 1: Without Profit Tracking")
    print("-" * 40)
    labeler_no_profit = OptimizedTripleBarrierLabeling(
        profit_take_multiplier=0.002,  # 0.2%
        stop_loss_multiplier=0.001,    # 0.1%
        time_barrier_minutes=30,
        include_profit_tracking=False
    )
    
    result_no_profit = labeler_no_profit.apply_triple_barrier_labeling_vectorized(data.copy())
    print(f"   Labels generated: {len(result_no_profit)}")
    print(f"   Buy signals: {(result_no_profit['label'] == 1).sum()}")
    print(f"   Sell signals: {(result_no_profit['label'] == -1).sum()}")
    print(f"   Columns: {list(result_no_profit.columns)}")
    print()
    
    # Test 2: With profit tracking (new behavior)
    print("💰 Test 2: With Profit Tracking")
    print("-" * 40)
    labeler_with_profit = OptimizedTripleBarrierLabeling(
        profit_take_multiplier=0.002,  # 0.2%
        stop_loss_multiplier=0.001,    # 0.1%
        time_barrier_minutes=30,
        include_profit_tracking=True
    )
    
    result_with_profit = labeler_with_profit.apply_triple_barrier_labeling_vectorized(data.copy())
    print(f"   Labels generated: {len(result_with_profit)}")
    print(f"   Buy signals: {(result_with_profit['label'] == 1).sum()}")
    print(f"   Sell signals: {(result_with_profit['label'] == -1).sum()}")
    print(f"   Columns: {list(result_with_profit.columns)}")
    
    if 'potential_profit_pct' in result_with_profit.columns:
        profits = result_with_profit['potential_profit_pct']
        buy_profits = result_with_profit[result_with_profit['label'] == 1]['potential_profit_pct']
        sell_profits = result_with_profit[result_with_profit['label'] == -1]['potential_profit_pct']
        
        print(f"   Profit tracking statistics:")
        print(f"     Overall - Avg: {profits.mean():.4f}, Std: {profits.std():.4f}")
        print(f"     Buy signals - Avg: {buy_profits.mean():.4f}, Max: {buy_profits.max():.4f}")
        print(f"     Sell signals - Avg: {sell_profits.mean():.4f}, Min: {sell_profits.min():.4f}")
        
        # Show some examples
        print(f"   Sample profit values:")
        sample_profits = profits.head(10)
        for i, profit in enumerate(sample_profits):
            label = result_with_profit.iloc[i]['label']
            direction = "BUY" if label == 1 else "SELL"
            print(f"     {i+1:2d}. {direction}: {profit:+.4f} ({profit*100:+.2f}%)")
    
    print()
    
    # Test 3: Compare different threshold settings
    print("⚙️ Test 3: Different Threshold Settings")
    print("-" * 40)
    
    thresholds = [
        (0.001, 0.0005),  # Tight thresholds
        (0.002, 0.001),   # Default thresholds
        (0.005, 0.0025),  # Wide thresholds
    ]
    
    for pt_mult, sl_mult in thresholds:
        labeler = OptimizedTripleBarrierLabeling(
            profit_take_multiplier=pt_mult,
            stop_loss_multiplier=sl_mult,
            time_barrier_minutes=30,
            include_profit_tracking=True
        )
        
        result = labeler.apply_triple_barrier_labeling_vectorized(data.copy())
        profits = result['potential_profit_pct']
        
        print(f"   PT: {pt_mult*100:.1f}%, SL: {sl_mult*100:.1f}%")
        print(f"     Signals: {(result['label'] == 1).sum()} buy, {(result['label'] == -1).sum()} sell")
        print(f"     Avg profit: {profits.mean():.4f}, Max profit: {profits.max():.4f}")
        print()
    
    print("✅ Profit tracking test completed!")

def test_edge_cases():
    """Test edge cases for profit tracking."""
    print("\n🔬 Testing Edge Cases")
    print("=" * 60)
    
    # Test with very small dataset
    print("📊 Test: Small Dataset")
    small_data = create_sample_data(50)
    labeler = OptimizedTripleBarrierLabeling(include_profit_tracking=True)
    result = labeler.apply_triple_barrier_labeling_vectorized(small_data)
    print(f"   Small dataset ({len(small_data)} points) -> {len(result)} labels")
    print()
    
    # Test with extreme price movements
    print("📈 Test: Extreme Price Movements")
    extreme_data = create_sample_data(100)
    # Add some extreme movements
    extreme_data.loc[25:30, 'high'] *= 1.05  # 5% spike
    extreme_data.loc[60:65, 'low'] *= 0.95   # 5% drop
    
    labeler = OptimizedTripleBarrierLabeling(
        profit_take_multiplier=0.01,  # 1%
        stop_loss_multiplier=0.005,   # 0.5%
        include_profit_tracking=True
    )
    result = labeler.apply_triple_barrier_labeling_vectorized(extreme_data)
    
    if 'potential_profit_pct' in result.columns:
        profits = result['potential_profit_pct']
        print(f"   Extreme movements -> Max profit: {profits.max():.4f}, Min profit: {profits.min():.4f}")
    
    print("✅ Edge case tests completed!")

if __name__ == "__main__":
    try:
        test_profit_tracking()
        test_edge_cases()
        print("\n🎉 All tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()