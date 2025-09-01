#!/usr/bin/env python3
"""Minimal test script for enhanced triple barrier method with profit tracking."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_test_data(n_samples: int = 1000) -> pd.DataFrame:
    pass
    pass
    """Create test market data."""
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="1min")

    # Create realistic price movements
    np.random.seed(42)  # For reproducible results

    # Start with a base price
    base_price = 100.0
    prices = [base_price]

    # Generate price movements with some trend and volatility
    for i in range(1, n_samples):
    pass
    pass
        # Add some trend and random walk
        change = np.random.normal(0, 0.001) + 0.0001  # Small upward trend
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)

    # Create OHLC data
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.002))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.002))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_samples),
    }, index=dates)

    # Ensure high >= close >= low
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['close'])

    return data

def simple_triple_barrier_with_profit_tracking(data: pd.DataFrame,
                                             profit_take_multiplier: float = 0.002,
                                             stop_loss_multiplier: float = 0.001,
                                             max_lookahead: int = 100) -> pd.DataFrame:
    """Simple triple barrier implementation with profit tracking."""

    close_prices = data['close'].values
    high_prices = data['high'].values
    low_prices = data['low'].values

    labels = np.zeros(len(close_prices), dtype=np.int8)
    profit_pcts = np.zeros(len(close_prices), dtype=np.float64)

    for i in range(len(close_prices) - 1):
    pass
    pass
        entry_price = close_prices[i]
        profit_barrier = entry_price * (1 + profit_take_multiplier)
        stop_barrier = entry_price * (1 - stop_loss_multiplier)

        # Look ahead for barrier hits
        for j in range(i + 1, min(i + max_lookahead, len(close_prices))):
    pass
    pass
            if high_prices[j] >= profit_barrier:
    pass
    pass
                labels[i] = 1  # Buy signal
                profit_pcts[i] = profit_take_multiplier  # Profit take hit
                break
            elif low_prices[j] <= stop_barrier:
                labels[i] = -1  # Sell signal
                profit_pcts[i] = -stop_loss_multiplier  # Stop loss hit
                break
            # If no barrier hit, label remains 0 (hold) and profit_pct remains 0.0

    # Create result DataFrame
    result_data = data.copy()
    result_data['label'] = labels
    result_data['potential_profit_pct'] = profit_pcts

    # Filter out HOLD samples for binary classification
    result_data = result_data[result_data['label'] != 0].copy()

    return result_data

def create_enhanced_labels(data: pd.DataFrame) -> pd.DataFrame:
    pass
    pass
    """Create enhanced labels that include profit information."""

    enhanced_data = data.copy()

    # Create profit-binned labels
    profit_bins = [-np.inf, -0.005, -0.002, 0, 0.002, 0.005, np.inf]
    profit_labels = ['Large Loss', 'Medium Loss', 'Small Loss', 'No Profit', 'Small Profit', 'Large Profit']
    enhanced_data['profit_category'] = pd.cut(
        enhanced_data['potential_profit_pct'],
        bins=profit_bins,
        labels=profit_labels,
        include_lowest=True
    )

    # Create combined direction-profit labels
    enhanced_data['direction_profit_label'] = enhanced_data.apply(
        lambda row: f"{'BUY' if row['label'] == 1 else 'SELL'}_{row['profit_category']}",
        axis=1
    )

    # Create profit-weighted labels
    enhanced_data['profit_weighted_label'] = enhanced_data['label'] * enhanced_data['potential_profit_pct']

    # Create confidence scores
    max_profit = enhanced_data['potential_profit_pct'].abs().max()
    if max_profit > 0:
    pass
    pass
        enhanced_data['signal_confidence'] = enhanced_data['potential_profit_pct'].abs() / max_profit
    else:
        enhanced_data['signal_confidence'] = 0.0

    return enhanced_data

def test_enhanced_triple_barrier():
    pass
    pass
    """Test the enhanced triple barrier method with profit tracking."""
    print("🧪 Testing Enhanced Triple Barrier Method with Profit Tracking")
    print("=" * 60)

    # Create test data
    print("📊 Creating test market data...")
    test_data = create_test_data(1000)
    print(f"   Created {len(test_data)} data points")
    print(f"   Price range: ${test_data['close'].min():.2f} - ${test_data['close'].max():.2f}")

    # Test triple barrier labeling with profit tracking
    print("\\\n🔧 Testing triple barrier labeling with profit tracking...")
    labeled_data = simple_triple_barrier_with_profit_tracking(
        test_data,
        profit_take_multiplier=0.002,  # 0.2%
        stop_loss_multiplier=0.001,    # 0.1%
        max_lookahead=100
    )

    print(f"✅ Generated {len(labeled_data)} labeled samples")
    print(f"   - LONG positions: {(labeled_data['label'] == 1).sum()}")
    print(f"   - SHORT positions: {(labeled_data['label'] == -1).sum()}")

    # Analyze profit tracking
    print("\\\n💰 Profit Tracking Analysis:")
    long_profits = labeled_data[labeled_data['label'] == 1]['potential_profit_pct']
    short_profits = labeled_data[labeled_data['label'] == -1]['potential_profit_pct']

    print(f"   LONG positions:")
    print(f"     - Count: {len(long_profits)}")
    print(f"     - Avg profit: {long_profits.mean():.4f}")
    print(f"     - Std profit: {long_profits.std():.4f}")
    print(f"     - Min profit: {long_profits.min():.4f}")
    print(f"     - Max profit: {long_profits.max():.4f}")

    print(f"   SHORT positions:")
    print(f"     - Count: {len(short_profits)}")
    print(f"     - Avg profit: {short_profits.mean():.4f}")
    print(f"     - Std profit: {short_profits.std():.4f}")
    print(f"     - Min profit: {short_profits.min():.4f}")
    print(f"     - Max profit: {short_profits.max():.4f}")

    # Test enhanced labeling
    print("\\\n🎯 Testing Enhanced Labeling...")
    enhanced_data = create_enhanced_labels(labeled_data)

    print("✅ Enhanced labeling completed")
    print(f"   - Profit categories: {enhanced_data['profit_category'].value_counts().to_dict()}")
    print(f"   - Direction-profit combinations: {enhanced_data['direction_profit_label'].value_counts().to_dict()}")
    print(f"   - Average signal confidence: {enhanced_data['signal_confidence'].mean():.4f}")

    # Show sample results
    print("\\\n📋 Sample Results:")
    sample_cols = ['label', 'potential_profit_pct', 'profit_category', 'direction_profit_label', 'signal_confidence']
    print(enhanced_data[sample_cols].head(10).to_string())

    # Verify profit calculations
    print("\\\n🔍 Verifying Profit Calculations:")
    print("   - All LONG positions should have positive profit percentages")
    print(f"   - LONG positions with positive profits: {(long_profits > 0).sum()}/{len(long_profits)}")
    print("   - All SHORT positions should have negative profit percentages")
    print(f"   - SHORT positions with negative profits: {(short_profits < 0).sum()}/{len(short_profits)}")

    # Test performance
    print("\\\n⚡ Performance Test:")
    import time

    start_time = time.time()
    for _ in range(10):
    pass
    pass
        _ = simple_triple_barrier_with_profit_tracking(test_data)
    total_time = time.time() - start_time

    print(f"   - Average time per run: {total_time/10:.4f} seconds")
    print(f"   - Processing speed: {len(test_data)/(total_time/10):.0f} samples/second")

    print("\\\n✅ Enhanced Triple Barrier Method Test Completed Successfully!")

    # Return the enhanced data for further analysis
    return enhanced_data

if __name__ == "__main__":
    pass
    pass
    enhanced_data = test_enhanced_triple_barrier()

    # Additional analysis
    print("\\\n📈 Additional Analysis:")
    print(f"   - Total profitable positions: {(enhanced_data['potential_profit_pct'] > 0).sum()}")
    print(f"   - Total loss positions: {(enhanced_data['potential_profit_pct'] < 0).sum()}")
    print(f"   - Overall profit distribution:")
    print(f"     * Large Profit: {(enhanced_data['profit_category'] == 'Large Profit').sum()}")
    print(f"     * Small Profit: {(enhanced_data['profit_category'] == 'Small Profit').sum()}")
    print(f"     * Small Loss: {(enhanced_data['profit_category'] == 'Small Loss').sum()}")
    print(f"     * Medium Loss: {(enhanced_data['profit_category'] == 'Medium Loss').sum()}")
    print(f"     * Large Loss: {(enhanced_data['profit_category'] == 'Large Loss').sum()}")