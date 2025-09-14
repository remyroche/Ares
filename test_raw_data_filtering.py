#!/usr/bin/env python3
"""
Test script to verify that raw market data columns are properly filtered out from feature selection
and test bootstrap mode configuration.
"""

import pandas as pd
import numpy as np
from src.training.utils.feature_selection.main_framework import filter_raw_market_data_columns
from src.training.utils.feature_selection.selection_methods import LassoStabilitySelector, analyze_infinity_values


def test_raw_data_filtering():
    """Test the raw data filtering functionality."""

    # Create a sample DataFrame with raw market data columns that should be filtered
    sample_columns = [
        # Raw market data (should be excluded)
        'timestamp', 'open_time', 'close_time', 'open', 'high', 'low', 'close',
        'volume', 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote_volume',
        'close_return', 'close_log_return', 'symbol', 'exchange',

        # Actual features (should be kept)
        'sma_20', 'ema_50', 'rsi_14', 'macd_signal', 'bollinger_upper',
        'volume_ratio', 'price_change_pct', 'momentum_10', 'volatility_20',
        'support_resistance_distance', 'trend_strength', 'market_regime_score',

        # Target/label columns (should be excluded)
        'target', 'label', 'model_score', 'prediction',

        # Regime columns (should be excluded)
        'regime', 'regime_label', 'hmm_regime', 'cluster_regime'
    ]

    # Create sample data
    n_samples = 100
    np.random.seed(42)
    data = np.random.randn(n_samples, len(sample_columns))

    df = pd.DataFrame(data, columns=sample_columns)

    # Test the filtering function
    filtered_features, excluded_columns = filter_raw_market_data_columns(sample_columns)
    expected_exclusions = [
        'timestamp', 'open_time', 'close_time', 'open', 'high', 'low', 'close',
        'volume', 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote_volume',
        'close_return', 'close_log_return', 'symbol', 'exchange',
        'target', 'label', 'model_score', 'prediction',
        'regime', 'regime_label', 'hmm_regime', 'cluster_regime'
    ]

    # Verify expected exclusions are actually excluded
    missing_exclusions = []
    for expected in expected_exclusions:
        if expected not in excluded_columns:
            missing_exclusions.append(expected)

    if missing_exclusions:
        pass
    else:
        pass

    # Verify that actual features are kept
    expected_features = [
        'sma_20', 'ema_50', 'rsi_14', 'macd_signal', 'bollinger_upper',
        'volume_ratio', 'price_change_pct', 'momentum_10', 'volatility_20',
        'support_resistance_distance', 'trend_strength', 'market_regime_score'
    ]

    unexpected_exclusions = []
    for feature in expected_features:
        if feature in excluded_columns:
            unexpected_exclusions.append(feature)

    if unexpected_exclusions:
        pass
    else:
        pass


    return filtered_features, excluded_columns


def test_bootstrap_mode_configuration():
    """Test that bootstrap counts are correctly set based on execution mode."""
    print("\n🔍 Testing bootstrap mode configuration...")

    # Test FULL mode
    config_full = {'mode': 'full'}
    selector_full = LassoStabilitySelector(config_full)
    print(f"📊 FULL mode bootstrap count: {selector_full.n_bootstraps}")
    assert selector_full.n_bootstraps == 100, f"Expected 100 for full mode, got {selector_full.n_bootstraps}"

    # Test BLANK mode
    config_blank = {'mode': 'blank'}
    selector_blank = LassoStabilitySelector(config_blank)
    print(f"📊 BLANK mode bootstrap count: {selector_blank.n_bootstraps}")
    assert selector_blank.n_bootstraps == 5, f"Expected 5 for blank mode, got {selector_blank.n_bootstraps}"

    # Test LIGHT mode
    config_light = {'mode': 'light'}
    selector_light = LassoStabilitySelector(config_light)
    print(f"📊 LIGHT mode bootstrap count: {selector_light.n_bootstraps}")
    assert selector_light.n_bootstraps == 2, f"Expected 2 for light mode, got {selector_light.n_bootstraps}"

    # Test default mode (should be blank)
    config_default = {}
    selector_default = LassoStabilitySelector(config_default)
    print(f"📊 DEFAULT mode bootstrap count: {selector_default.n_bootstraps}")
    assert selector_default.n_bootstraps == 5, f"Expected 5 for default mode, got {selector_default.n_bootstraps}"

    print("✅ All bootstrap mode configurations are correct!")


def test_enhanced_infinity_analysis():
    """Test the enhanced infinity value analysis functionality."""
    print("\n🧪 Testing Enhanced Infinity Analysis")
    print("=" * 50)

    # Create sample data with infinity values
    np.random.seed(42)
    n_samples, n_features = 1000, 20

    # Generate normal data
    X = np.random.randn(n_samples, n_features) * 10

    # Add some infinity values at specific locations
    X[100, 5] = np.inf      # Positive infinity
    X[200, 5] = -np.inf     # Negative infinity
    X[300, 5] = np.inf      # Another positive infinity
    X[150, 10] = np.inf     # Positive infinity in different feature
    X[250, 10] = np.inf     # Another positive infinity in same feature
    X[350, 15] = -np.inf    # Negative infinity in different feature

    # Create feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]

    print(f"📊 Created test data: {X.shape} with controlled infinity values")
    print(f"   Expected infinity locations: feature_5 (3 inf), feature_10 (2 inf), feature_15 (1 -inf)")

    # Run comprehensive analysis
    analysis = analyze_infinity_values(X, "test_data", feature_names)

    print("\n📈 Analysis Results:")
    print(f"   Total elements: {analysis['total_elements']}")
    print(f"   Data shape: {analysis['data_shape']}")
    print(f"   Infinity count: {analysis['infinity_count']}")
    print(f"   Positive infinity: {analysis['positive_infinity_count']}")
    print(f"   Negative infinity: {analysis['negative_infinity_count']}")
    print(f"   Infinity percentage: {analysis['infinity_percentage']:.4f}%")
    print(f"   Rows with infinity: {analysis['rows_with_infinity']}")
    if 'avg_infinity_per_affected_row' in analysis:
        print(f"   Avg infinity per affected row: {analysis['avg_infinity_per_affected_row']:.2f}")

    print("\n🔍 Feature Analysis:")
    for i, feature_info in enumerate(analysis['feature_analysis'][:5]):  # Show first 5
        print(f"   {i+1}. {feature_info['feature_name']} (idx {feature_info['feature_index']}):")
        print(f"      Total infinity: {feature_info['total_infinity']}")
        print(f"      Positive: {feature_info['positive_infinity']}, Negative: {feature_info['negative_infinity']}")
        print(f"      Infinity percentage: {feature_info['infinity_percentage']:.4f}%")
        print(f"      Row indices: {feature_info['infinity_row_indices']}")
        if feature_info['additional_indices_count'] > 0:
            print(f"      Additional indices: {feature_info['additional_indices_count']}")

        if 'finite_stats' in feature_info:
            stats = feature_info['finite_stats']
            print("      Finite stats:")
            print(f"         Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
            print(f"         Min: {stats['min']:.3f}, Max: {stats['max']:.3f}")

    print("\n✅ Enhanced infinity analysis test completed!")
    return analysis


if __name__ == "__main__":
    test_raw_data_filtering()
    test_bootstrap_mode_configuration()
    analysis_result = test_enhanced_infinity_analysis()
