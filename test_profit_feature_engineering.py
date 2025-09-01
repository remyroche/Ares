#!/usr/bin/env python3
"""Test script for profit-based feature engineering system."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.steps.step4_analyst_labeling_feature_engineering_components.profit_based_feature_engineering import (
import ProfitBasedFeatureEngineering,
    ProfitBasedFeatureEngineering,
    benchmark_profit_feature_engineering
)

def create_test_data(n_samples: int = 1000) -> pd.DataFrame:
    pass
    pass
    """Create test market data with profit percentages."""
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

    # Add profit percentages (simulating triple barrier results)
    # Mix of positive and negative profits for realistic testing
    profit_pcts = np.random.uniform(-0.01, 0.01, n_samples)
    # Add some structure to make it more realistic
    profit_pcts = profit_pcts + np.sin(np.arange(n_samples) * 0.1) * 0.002
    data['potential_profit_pct'] = profit_pcts

    # Add labels (1 for LONG, -1 for SHORT, 0 for HOLD)
    labels = np.sign(profit_pcts)
    data['label'] = labels

    return data

def test_profit_feature_engineering():
    pass
    pass
    """Test the profit-based feature engineering system."""
    print("🧪 Testing Profit-Based Feature Engineering System")
    print("=" * 60)

    # Create test data
    print("📊 Creating test market data...")
    test_data = create_test_data(1000)
    print(f"   Created {len(test_data)} data points")
    print(f"   Price range: ${test_data['close'].min():.2f} - ${test_data['close'].max():.2f}")
    print(f"   Profit range: {test_data['potential_profit_pct'].min():.4f} - {test_data['potential_profit_pct'].max():.4f}")

    # Test feature engineering
    print("\\\n🔧 Testing profit-based feature engineering...")
    feature_eng = ProfitBasedFeatureEngineering(
        profit_column="potential_profit_pct",
        volume_column="volume",
        price_column="close",
        use_numba=True,
        memory_efficient=True
    )

    # Apply all features
    result_data = feature_eng.apply_all_features(test_data)

    print(f"✅ Feature engineering completed")
    print(f"   - Input shape: {test_data.shape}")
    print(f"   - Output shape: {result_data.shape}")
    print(f"   - Features added: {len(result_data.columns) - len(test_data.columns)}")

    # Get feature summary
    print("\\\n📋 Feature Summary:")
    summary = feature_eng.get_feature_summary(result_data)
    print(f"   - Total profit features: {summary['total_features']}")

    for category, features in summary['feature_categories'].items():
    pass
    pass
        print(f"   - {category}: {len(features)} features")

    # Test individual feature categories
    print("\\\n🎯 Testing Individual Feature Categories:")

    # Test basic profit features
    basic_data = feature_eng._apply_basic_profit_features(test_data.copy())
    basic_features = [col for col in basic_data.columns if "potential_profit_pct" in col and col != "potential_profit_pct"]
    print(f"   - Basic profit features: {len(basic_features)}")
    print(f"     Features: {basic_features}")

    # Test categorical features
    cat_data = feature_eng._apply_categorical_features(test_data.copy())
    cat_features = [col for col in cat_data.columns if "potential_profit_pct" in col and col != "potential_profit_pct"]
    print(f"   - Categorical features: {len(cat_features)}")
    print(f"     Features: {cat_features}")

    # Test risk-reward features
    risk_data = feature_eng._apply_risk_reward_features(test_data.copy())
    risk_features = [col for col in risk_data.columns if "potential_profit_pct" in col and col != "potential_profit_pct"]
    print(f"   - Risk-reward features: {len(risk_features)}")
    print(f"     Features: {risk_features}")

    # Test momentum features
    momentum_data = feature_eng._apply_momentum_features(test_data.copy())
    momentum_features = [col for col in momentum_data.columns if "potential_profit_pct" in col and col != "potential_profit_pct"]
    print(f"   - Momentum features: {len(momentum_features)}")
    print(f"     Features: {momentum_features}")

    # Test volatility features
    vol_data = feature_eng._apply_volatility_features(test_data.copy())
    vol_features = [col for col in vol_data.columns if "potential_profit_pct" in col and col != "potential_profit_pct"]
    print(f"   - Volatility features: {len(vol_features)}")
    print(f"     Features: {vol_features}")

    # Test volume features
    volume_data = feature_eng._apply_volume_features(test_data.copy())
    volume_features = [col for col in volume_data.columns if "potential_profit_pct" in col and col != "potential_profit_pct"]
    print(f"   - Volume features: {len(volume_features)}")
    print(f"     Features: {volume_features}")

    # Test rolling features
    rolling_data = feature_eng._apply_rolling_features(test_data.copy())
    rolling_features = [col for col in rolling_data.columns if "potential_profit_pct" in col and col != "potential_profit_pct"]
    print(f"   - Rolling features: {len(rolling_features)}")
    print(f"     Features: {rolling_features}")

    # Test feature selection
    print("\\\n🔍 Testing Feature Selection:")
    selected_features = feature_eng.select_features(
        result_data,
        method="correlation",
        threshold=0.01,
        max_features=20
    )
    print(f"   - Selected {len(selected_features)} features using correlation method")
    print(f"   - Top features: {selected_features[:10]}")

    # Test with variance method
    selected_variance = feature_eng.select_features(
        result_data,
        method="variance",
        threshold=0.0001,
        max_features=15
    )
    print(f"   - Selected {len(selected_variance)} features using variance method")

    # Analyze feature quality
    print("\\\n📊 Feature Quality Analysis:")

    # Check for missing values
    missing_data = result_data.isnull().sum()
    high_missing = missing_data[missing_data > 0]
    if len(high_missing) > 0:
    pass
    pass
        print(f"   - Features with missing values: {len(high_missing)}")
        print(f"     High missing: {high_missing.head().to_dict()}")
    else:
        print("   - No missing values found")

    # Check feature correlations (only numerical features)
    profit_features = [col for col in result_data.columns if "potential_profit_pct" in col and col != "potential_profit_pct"]
    numerical_features = []
    for feature in profit_features:
    pass
    pass
        if result_data[feature].dtype in ['int64', 'float64']:
    pass
    pass
            numerical_features.append(feature)

    if numerical_features:
    pass
    pass
        correlations = result_data[numerical_features].corrwith(result_data['potential_profit_pct']).abs()
        high_corr = correlations[correlations > 0.1]
        print(f"   - Features with high correlation (>0.1): {len(high_corr)}")
        if len(high_corr) > 0:
    pass
    pass
            print(f"     Top correlations: {high_corr.head().to_dict()}")
    else:
        print("   - No numerical features found for correlation analysis")

    # Test performance
    print("\\\n⚡ Performance Test:")
    benchmark_results = benchmark_profit_feature_engineering(test_data)
    print(f"   - Numba time: {benchmark_results['numba_time']:.4f} seconds")
    print(f"   - Python time: {benchmark_results['python_time']:.4f} seconds")
    print(f"   - Speedup: {benchmark_results['speedup']:.2f}x")
    print(f"   - Features generated: {benchmark_results['numba_features']}")

    # Show sample results
    print("\\\n📋 Sample Results:")
    sample_cols = ['potential_profit_pct', 'label'] + numerical_features[:5]
    print(result_data[sample_cols].head(10).to_string())

    print("\\\n✅ Profit-Based Feature Engineering Test Completed Successfully!")

    return result_data

def test_long_short_compatibility():
    pass
    pass
    """Test that the feature engineering works correctly with long/short labels."""
    print("\\\n🔄 Testing Long/Short Compatibility")
    print("=" * 40)

    # Create test data with explicit long/short labels
    test_data = create_test_data(500)

    # Ensure we have both long and short positions
    long_mask = test_data['label'] == 1
    short_mask = test_data['label'] == -1

    print(f"   - LONG positions: {long_mask.sum()}")
    print(f"   - SHORT positions: {short_mask.sum()}")
    print(f"   - HOLD positions: {(test_data['label'] == 0).sum()}")

    # Test feature engineering
    feature_eng = ProfitBasedFeatureEngineering()
    result_data = feature_eng.apply_all_features(test_data)

    # Analyze features by position type
    long_data = result_data[long_mask]
    short_data = result_data[short_mask]

    print(f"\\\n📊 Feature Analysis by Position Type:")

    # Basic profit features
    long_profits = long_data['potential_profit_pct']
    short_profits = short_data['potential_profit_pct']

    print(f"   LONG positions:")
    print(f"     - Count: {len(long_profits)}")
    print(f"     - Avg profit: {long_profits.mean():.4f}")
    print(f"     - Profit range: {long_profits.min():.4f} to {long_profits.max():.4f}")

    print(f"   SHORT positions:")
    print(f"     - Count: {len(short_profits)}")
    print(f"     - Avg profit: {short_profits.mean():.4f}")
    print(f"     - Profit range: {short_profits.min():.4f} to {short_profits.max():.4f}")

    # Test categorical features
    if 'potential_profit_pct_bins' in result_data.columns:
    pass
    pass
        long_bins = long_data['potential_profit_pct_bins'].value_counts()
        short_bins = short_data['potential_profit_pct_bins'].value_counts()

        print(f"\\\n   Profit distribution by position type:")
        print(f"     LONG bins: {long_bins.to_dict()}")
        print(f"     SHORT bins: {short_bins.to_dict()}")

    # Test momentum features
    momentum_features = [col for col in result_data.columns if 'momentum' in col]
    if momentum_features:
    pass
    pass
        print(f"\\\n   Momentum features by position type:")
        for feature in momentum_features[:3]:  # Show first 3
            long_momentum = long_data[feature].mean()
            short_momentum = short_data[feature].mean()
            print(f"     {feature}: LONG={long_momentum:.4f}, SHORT={short_momentum:.4f}")

    print("\\\n✅ Long/Short Compatibility Test Completed!")

if __name__ == "__main__":
    pass
    pass
    # Run main test
    result_data = test_profit_feature_engineering()

    # Run long/short compatibility test
    test_long_short_compatibility()

    print("\\\n🎉 All tests completed successfully!")
    print(f"Final dataset shape: {result_data.shape}")
    print(f"Total features: {len(result_data.columns)}")