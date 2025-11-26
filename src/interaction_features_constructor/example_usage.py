"""
Example Usage of Feature Interaction Constructor

This script demonstrates how to use the feature interaction constructor
to decompose features, save metadata, and calculate features for live trading.
"""

import pandas as pd
import numpy as np
from src.interaction_features_constructor import (
    FeatureDecomposer,
    FeatureCalculator,
    FeatureMetadataStore
)


def example_1_decompose_features():
    """Example 1: Decompose feature names to understand their structure."""
    print("=" * 80)
    print("EXAMPLE 1: Feature Decomposition")
    print("=" * 80)

    # Example features from the actual selected features
    example_features = [
        'candlestick_doji_pattern_base_27x_ratio',
        'fibonacci_0.236_5_price_returns_vwap',
        'fibonacci_0.236_5_price_returns_vwap_27x_ratio_x_wavelet_energy_base_6x_ratio',
        'returns_volatility_20_price_returns_base_log_ratio_fibonacci_0.618_20_price_returns_vwap_x_27x'
    ]

    decomposer = FeatureDecomposer()

    for feature in example_features:
        print(f"\nFeature: {feature}")
        print("-" * 80)

        components = decomposer.decompose(feature)

        print(f"  Base features needed: {components.base_features}")
        print(f"  Variant type: {components.variant_type}")
        print(f"  Timeframe multiplier: {components.timeframe_multiplier}")
        print(f"  Operators: {components.operators}")
        print(f"  Dependencies: {components.dependencies}")
        print(f"\n  Calculation steps:")
        for i, step in enumerate(components.calculation_steps, 1):
            print(f"    {i}. {step}")


def example_2_create_and_save_metadata():
    """Example 2: Create and save feature metadata."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Create and Save Feature Metadata")
    print("=" * 80)

    # Example selected features (from actual training results)
    selected_features_60 = [
        'candlestick_dark_cloud_cover_pattern_vwap_3x_ratio_log_fibonacci_0.786_10_price_returns_vwap',
        'candlestick_doji_pattern_base_27x_ratio',
        'candlestick_doji_pattern_vwap_3x_ratio',
        'candlestick_engulfing_pattern_base_9x_ratio',
        'directional_signal_vwap',
        'fibonacci_0.236_5_price_returns_vwap',
        'fibonacci_0.618_20_price_returns_vwap_27x_ratio',
        'vectorbt_enhanced_obv_10_base_3x_ratio',
        'volume_price_trend_vwap',
        'wavelet_energy_base_6x_ratio'
    ]

    # Create metadata store
    store = FeatureMetadataStore()
    store.create_from_selection(
        selected_features=selected_features_60,
        symbol='ETHUSDT',
        exchange='binance',
        timeframe='15m',
        direction='long',
        model='analyst'
    )

    print(f"\nMetadata Store: {store}")
    print(f"\nContext: {store.get_context()}")
    print(f"\nStatistics: {store.get_statistics()}")
    print(f"\nBase features required: {store.get_base_features_required()}")

    # Save metadata
    output_file = '/tmp/feature_metadata_example.json'
    store.save(output_file)
    print(f"\n✅ Metadata saved to: {output_file}")

    # Load it back
    loaded_store = FeatureMetadataStore.load(output_file)
    print(f"\n✅ Metadata loaded successfully: {loaded_store}")


def example_3_calculate_features():
    """Example 3: Calculate features using the FeatureCalculator."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Calculate Features for Live Trading")
    print("=" * 80)

    # Create sample OHLCV data
    np.random.seed(42)
    n_candles = 1000

    ohlcv_data_pd = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_candles, freq='15T'),
        'open': 3000 + np.cumsum(np.random.randn(n_candles) * 10),
        'high': 3000 + np.cumsum(np.random.randn(n_candles) * 10) + 5,
        'low': 3000 + np.cumsum(np.random.randn(n_candles) * 10) - 5,
        'close': 3000 + np.cumsum(np.random.randn(n_candles) * 10),
        'volume': np.random.randint(100, 1000, n_candles)
    })

    # Create sample base features (these would normally come from feature_bank)
    base_features_pd = pd.DataFrame({
        'candlestick_dark_cloud_cover_pattern': np.random.randint(0, 2, n_candles),
        'candlestick_doji_pattern': np.random.randint(0, 2, n_candles),
        'candlestick_engulfing_pattern': np.random.randint(0, 2, n_candles),
        'directional_signal': np.random.randn(n_candles),
        'fibonacci_0.236_5_price_returns': np.random.randn(n_candles) * 0.01,
        'fibonacci_0.786_10_price_returns': np.random.randn(n_candles) * 0.01,
        'fibonacci_0.618_20_price_returns': np.random.randn(n_candles) * 0.01,
        'vectorbt_enhanced_obv_10': np.cumsum(np.random.randn(n_candles) * 100),
        'volume_price_trend': np.random.randn(n_candles) * 10,
        'wavelet_energy': np.random.randn(n_candles) * 5
    }, index=ohlcv_data_pd.index)

    # Prefer Polars DataFrames when available to exercise Polars → FeatureCalculator path
    try:
        import polars as pl  # type: ignore[import]

        ohlcv_data = pl.DataFrame(ohlcv_data_pd)
        base_features = pl.DataFrame(base_features_pd)
    except Exception:
        # Fallback to pandas if Polars is not available
        ohlcv_data = ohlcv_data_pd
        base_features = base_features_pd

    # Selected features to calculate
    selected_features = [
        'candlestick_doji_pattern_base_27x_ratio',
        'candlestick_doji_pattern_vwap_3x_ratio',
        'fibonacci_0.236_5_price_returns_vwap',
        'directional_signal_vwap',
        'vectorbt_enhanced_obv_10_base_3x_ratio'
    ]

    # Create feature calculator
    calculator = FeatureCalculator(selected_features)

    print(f"\nSelected features: {len(selected_features)}")
    print(f"Base features required: {calculator.get_required_base_features()}")

    # Calculate features (FeatureCalculator will internally handle pandas/Polars inputs)
    calculated_features = calculator.calculate(
        ohlcv_data=ohlcv_data,
        base_features=base_features,
        return_type='dataframe'
    )

    print(f"\n✅ Calculated features shape: {calculated_features.shape}")
    print(f"\nCalculated feature names:")
    for col in calculated_features.columns:
        print(f"  - {col}")

    print(f"\nSample calculated features (first 5 rows):")
    print(calculated_features.head())

    # Save metadata for later use
    calculator.save_metadata(
        filepath='/tmp/feature_calculator_metadata.json',
        symbol='ETHUSDT',
        exchange='binance',
        timeframe='15m',
        direction='long',
        model='analyst'
    )
    print(f"\n✅ Calculator metadata saved to: /tmp/feature_calculator_metadata.json")


def example_4_load_and_calculate():
    """Example 4: Load saved metadata and calculate features."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Load Metadata and Calculate Features")
    print("=" * 80)

    # Load calculator from saved metadata
    try:
        calculator = FeatureCalculator.from_metadata_file(
            '/tmp/feature_calculator_metadata.json'
        )
        print(f"\n✅ Loaded calculator: {calculator}")
        print(f"  Selected features: {len(calculator.selected_features)}")
        print(f"  Base features required: {len(calculator.get_required_base_features())}")
    except FileNotFoundError:
        print("\n⚠️ Metadata file not found. Run example_3 first.")


def run_all_examples():
    """Run all examples."""
    example_1_decompose_features()
    example_2_create_and_save_metadata()
    example_3_calculate_features()
    example_4_load_and_calculate()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    run_all_examples()
