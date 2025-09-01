#!/usr/bin/env python3
"""Test script for enhanced triple barrier method with profit tracking."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.steps.step4_analyst_labeling_feature_engineering_components.optimized_triple_barrier_labeling import (
import OptimizedTripleBarrierLabeling
    OptimizedTripleBarrierLabeling
)

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

    # Test optimized triple barrier labeling
    print("\\\n🔧 Testing optimized triple barrier labeling...")
    optimizer = OptimizedTripleBarrierLabeling(
        profit_take_multiplier=0.002,  # 0.2%
        stop_loss_multiplier=0.001,    # 0.1%
        time_barrier_minutes=30,
        max_lookahead=100,
        binary_classification=True
    )

    # Apply triple barrier labeling
    labeled_data = optimizer.apply_triple_barrier_labeling_vectorized(test_data)

    print(f"✅ Generated {len(labeled_data)} labeled samples")
    print(f"   - BUY signals: {(labeled_data['label'] == 1).sum()}")
    print(f"   - SELL signals: {(labeled_data['label'] == -1).sum()}")

    # Analyze profit tracking
    print("\\\n💰 Profit Tracking Analysis:")
    buy_profits = labeled_data[labeled_data['label'] == 1]['potential_profit_pct']
    sell_profits = labeled_data[labeled_data['label'] == -1]['potential_profit_pct']

    print(f"   BUY signals:")
    print(f"     - Count: {len(buy_profits)}")
    print(f"     - Avg profit: {buy_profits.mean():.4f}")
    print(f"     - Std profit: {buy_profits.std():.4f}")
    print(f"     - Min profit: {buy_profits.min():.4f}")
    print(f"     - Max profit: {buy_profits.max():.4f}")

    print(f"   SELL signals:")
    print(f"     - Count: {len(sell_profits)}")
    print(f"     - Avg profit: {sell_profits.mean():.4f}")
    print(f"     - Std profit: {sell_profits.std():.4f}")
    print(f"     - Min profit: {sell_profits.min():.4f}")
    print(f"     - Max profit: {sell_profits.max():.4f}")

    # Test enhanced labeling
    print("\\\n🎯 Testing Enhanced Labeling...")

    # Create enhanced labels
    enhanced_data = labeled_data.copy()

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
    print("   - All BUY signals should have positive profit percentages")
    print(f"   - BUY signals with positive profits: {(buy_profits > 0).sum()}/{len(buy_profits)}")
    print("   - All SELL signals should have negative profit percentages")
    print(f"   - SELL signals with negative profits: {(sell_profits < 0).sum()}/{len(sell_profits)}")

    # Test Numba vs Python performance
    print("\\\n⚡ Performance Test:")
    import time

    # Test Python implementation
    start_time = time.time()
    python_result = optimizer.apply_triple_barrier_labeling_vectorized(test_data)
    python_time = time.time() - start_time

    print(f"   - Python implementation: {python_time:.4f} seconds")
    print(f"   - Processing speed: {len(test_data)/python_time:.0f} samples/second")

    print("\\\n✅ Enhanced Triple Barrier Method Test Completed Successfully!")

if __name__ == "__main__":
    pass
    pass
    test_enhanced_triple_barrier()