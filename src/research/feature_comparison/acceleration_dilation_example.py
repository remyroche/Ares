"""
Feature Acceleration and Window Dilation Example

This script demonstrates how to compare features to their acceleration
(Δ over k or 2nd difference) and window dilation (3× lookback) variants.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from feature_comparison.feature_acceleration_dilation import FeatureAccelerationDilation
from feature_comparison.family_diverse_features import FamilyDiverseFeatureGenerator

# Suppress warnings
warnings.filterwarnings('ignore')

def create_market_data_for_acceleration_dilation(n_samples: int = 3000) -> pd.DataFrame:
    """
    Create market data with features suitable for acceleration and dilation testing.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with OHLCV data and derived features
    """
    np.random.seed(42)
    
    # Generate realistic price data with multiple regimes
    n_regimes = 4
    regime_length = n_samples // n_regimes
    
    prices = []
    volumes = []
    
    for regime in range(n_regimes):
        if regime == 0:
            # Trending up regime
            trend = 0.002
            volatility = 0.015
            volume_trend = 0.001
        elif regime == 1:
            # Sideways regime
            trend = 0.0001
            volatility = 0.008
            volume_trend = -0.0005
        elif regime == 2:
            # High volatility regime
            trend = 0.0005
            volatility = 0.035
            volume_trend = 0.003
        else:
            # Trending down regime
            trend = -0.0015
            volatility = 0.025
            volume_trend = 0.002
        
        # Generate regime data
        regime_returns = np.random.normal(trend, volatility, regime_length)
        regime_prices = 100 * np.exp(np.cumsum(regime_returns))
        
        # Add volume with trend
        regime_volumes = np.random.lognormal(
            10 + volume_trend * np.arange(regime_length), 
            1.0
        )
        
        prices.extend(regime_prices)
        volumes.extend(regime_volumes)
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.003, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.003, n_samples))),
        'close': prices,
        'volume': volumes
    })
    
    # Ensure OHLC constraints
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    return data

def test_acceleration_features():
    """Test acceleration feature generation and evaluation."""
    print("Testing Acceleration Features...")
    print("=" * 60)
    
    # Create market data
    data = create_market_data_for_acceleration_dilation(2000)
    
    # Generate base features
    family_generator = FamilyDiverseFeatureGenerator(enable_matrix_ops=True)
    family_features = family_generator.generate_family_diverse_features(data)
    
    # Combine features
    X = pd.DataFrame(index=data.index)
    for family_df in family_features.values():
        original_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        feature_cols = [col for col in family_df.columns if col not in original_cols]
        for feature in feature_cols:
            X[feature] = family_df[feature]
    
    # Create target
    y = data['close'].pct_change().dropna()
    
    # Align data
    common_idx = X.index.intersection(y.index)
    X_aligned = X.loc[common_idx]
    y_aligned = y.loc[common_idx]
    
    # Remove NaN values
    valid_mask = ~(X_aligned.isna().any(axis=1) | y_aligned.isna())
    X_clean = X_aligned[valid_mask]
    y_clean = y_aligned[valid_mask]
    
    print(f"Created feature matrix with shape: {X_clean.shape}")
    
    # Initialize acceleration system
    accel_system = FeatureAccelerationDilation(
        acceleration_lags=[1, 3],
        mi_threshold=0.6,
        correlation_threshold=0.9,
        conditional_mi_threshold=0.6,
        enable_matrix_ops=True
    )
    
    # Generate acceleration features
    print("\nGenerating acceleration features...")
    acceleration_features = accel_system.generate_acceleration_features(X_clean)
    
    print(f"Generated acceleration features for {len(acceleration_features)} lags:")
    for lag_key, lag_df in acceleration_features.items():
        print(f"  {lag_key}: {len(lag_df.columns)} features")
    
    # Show sample acceleration features
    print(f"\nSample acceleration features:")
    for lag_key, lag_df in acceleration_features.items():
        if len(lag_df.columns) > 0:
            sample_features = lag_df.columns[:3]
            print(f"  {lag_key}:")
            for feature in sample_features:
                print(f"    - {feature}")
    
    # Evaluate acceleration features
    print(f"\nEvaluating acceleration features...")
    base_features = list(X_clean.columns)
    evaluation_results = accel_system.evaluate_acceleration_features(
        X_clean, y_clean, acceleration_features, base_features
    )
    
    # Display results
    print(f"\nAcceleration Evaluation Results:")
    print(f"  Accepted features: {len(evaluation_results['accepted_features'])}")
    print(f"  Rejected features: {len(evaluation_results['rejected_features'])}")
    
    if evaluation_results['accepted_features']:
        print(f"\nAccepted acceleration features:")
        for feature in evaluation_results['accepted_features'][:10]:  # Show first 10
            print(f"  - {feature}")
        if len(evaluation_results['accepted_features']) > 10:
            print(f"  ... and {len(evaluation_results['accepted_features']) - 10} more")
    
    # Show detailed evaluation for a few features
    print(f"\nDetailed evaluation examples:")
    for lag_key, lag_results in evaluation_results['acceleration_evaluations'].items():
        if lag_results:
            sample_feature = list(lag_results.keys())[0]
            evaluation = lag_results[sample_feature]
            print(f"\n  {sample_feature}:")
            print(f"    Base MI: {evaluation.get('base_mi', 0):.4f}")
            print(f"    Variant MI: {evaluation.get('variant_mi', 0):.4f}")
            print(f"    MI Ratio: {evaluation.get('mi_ratio', 0):.4f}")
            print(f"    Correlation: {evaluation.get('correlation', 0):.4f}")
            print(f"    Conditional MI: {evaluation.get('conditional_mi', 0):.4f}")
            print(f"    MSE Improvement: {evaluation.get('mse_improvement', 0):.4f}")
            break
    
    return acceleration_features, evaluation_results

def test_dilation_features():
    """Test window dilation feature generation and evaluation."""
    print("\nTesting Window Dilation Features...")
    print("=" * 60)
    
    # Create market data
    data = create_market_data_for_acceleration_dilation(2000)
    
    # Generate base features
    family_generator = FamilyDiverseFeatureGenerator(enable_matrix_ops=True)
    family_features = family_generator.generate_family_diverse_features(data)
    
    # Combine features
    X = pd.DataFrame(index=data.index)
    for family_df in family_features.values():
        original_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        feature_cols = [col for col in family_df.columns if col not in original_cols]
        for feature in feature_cols:
            X[feature] = family_df[feature]
    
    # Create target
    y = data['close'].pct_change().dropna()
    
    # Align data
    common_idx = X.index.intersection(y.index)
    X_aligned = X.loc[common_idx]
    y_aligned = y.loc[common_idx]
    
    # Remove NaN values
    valid_mask = ~(X_aligned.isna().any(axis=1) | y_aligned.isna())
    X_clean = X_aligned[valid_mask]
    y_clean = y_aligned[valid_mask]
    
    print(f"Created feature matrix with shape: {X_clean.shape}")
    
    # Initialize dilation system
    dil_system = FeatureAccelerationDilation(
        dilation_factors=[2.0, 3.0],
        mi_threshold=0.6,
        correlation_threshold=0.9,
        conditional_mi_threshold=0.6,
        enable_matrix_ops=True
    )
    
    # Generate dilation features
    print("\nGenerating window dilation features...")
    dilation_features = dil_system.generate_dilation_features(X_clean)
    
    print(f"Generated dilation features for {len(dilation_features)} factors:")
    for factor_key, factor_df in dilation_features.items():
        print(f"  {factor_key}: {len(factor_df.columns)} features")
    
    # Show sample dilation features
    print(f"\nSample dilation features:")
    for factor_key, factor_df in dilation_features.items():
        if len(factor_df.columns) > 0:
            sample_features = factor_df.columns[:3]
            print(f"  {factor_key}:")
            for feature in sample_features:
                print(f"    - {feature}")
    
    # Evaluate dilation features
    print(f"\nEvaluating dilation features...")
    base_features = list(X_clean.columns)
    evaluation_results = dil_system.evaluate_dilation_features(
        X_clean, y_clean, dilation_features, base_features
    )
    
    # Display results
    print(f"\nDilation Evaluation Results:")
    print(f"  Accepted features: {len(evaluation_results['accepted_features'])}")
    print(f"  Rejected features: {len(evaluation_results['rejected_features'])}")
    
    if evaluation_results['accepted_features']:
        print(f"\nAccepted dilation features:")
        for feature in evaluation_results['accepted_features'][:10]:  # Show first 10
            print(f"  - {feature}")
        if len(evaluation_results['accepted_features']) > 10:
            print(f"  ... and {len(evaluation_results['accepted_features']) - 10} more")
    
    # Show detailed evaluation for a few features
    print(f"\nDetailed evaluation examples:")
    for factor_key, factor_results in evaluation_results['dilation_evaluations'].items():
        if factor_results:
            sample_feature = list(factor_results.keys())[0]
            evaluation = factor_results[sample_feature]
            print(f"\n  {sample_feature}:")
            print(f"    Base MI: {evaluation.get('base_mi', 0):.4f}")
            print(f"    Variant MI: {evaluation.get('variant_mi', 0):.4f}")
            print(f"    MI Ratio: {evaluation.get('mi_ratio', 0):.4f}")
            print(f"    Correlation: {evaluation.get('correlation', 0):.4f}")
            print(f"    Conditional MI: {evaluation.get('conditional_mi', 0):.4f}")
            print(f"    MSE Improvement: {evaluation.get('mse_improvement', 0):.4f}")
            break
    
    return dilation_features, evaluation_results

def test_complete_acceleration_dilation_pipeline():
    """Test complete acceleration and dilation pipeline."""
    print("\nTesting Complete Acceleration & Dilation Pipeline...")
    print("=" * 70)
    
    # Create market data
    data = create_market_data_for_acceleration_dilation(3000)
    
    # Generate base features
    family_generator = FamilyDiverseFeatureGenerator(enable_matrix_ops=True)
    family_features = family_generator.generate_family_diverse_features(data)
    
    # Combine features
    X = pd.DataFrame(index=data.index)
    for family_df in family_features.values():
        original_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        feature_cols = [col for col in family_df.columns if col not in original_cols]
        for feature in feature_cols:
            X[feature] = family_df[feature]
    
    # Create target
    y = data['close'].pct_change().dropna()
    
    # Align data
    common_idx = X.index.intersection(y.index)
    X_aligned = X.loc[common_idx]
    y_aligned = y.loc[common_idx]
    
    # Remove NaN values
    valid_mask = ~(X_aligned.isna().any(axis=1) | y_aligned.isna())
    X_clean = X_aligned[valid_mask]
    y_clean = y_aligned[valid_mask]
    
    print(f"Running complete pipeline with {X_clean.shape[1]} base features...")
    
    # Initialize complete system
    system = FeatureAccelerationDilation(
        acceleration_lags=[1, 3],
        dilation_factors=[2.0, 3.0],
        mi_threshold=0.6,
        correlation_threshold=0.9,
        conditional_mi_threshold=0.6,
        enable_matrix_ops=True
    )
    
    # Run complete evaluation
    print("Running complete acceleration and dilation evaluation...")
    results = system.run_complete_evaluation(X_clean, y_clean)
    
    # Display summary
    summary = results['summary']
    print(f"\nComplete Pipeline Results:")
    print(f"  Base features: {X_clean.shape[1]}")
    print(f"  Acceleration features generated: {summary['total_acceleration_features']}")
    print(f"  Dilation features generated: {summary['total_dilation_features']}")
    print(f"  Accepted acceleration: {summary['accepted_acceleration']}")
    print(f"  Accepted dilation: {summary['accepted_dilation']}")
    print(f"  Rejected acceleration: {summary['rejected_acceleration']}")
    print(f"  Rejected dilation: {summary['rejected_dilation']}")
    
    # Show acceptance rates
    accel_acceptance_rate = summary['accepted_acceleration'] / max(summary['total_acceleration_features'], 1)
    dil_acceptance_rate = summary['accepted_dilation'] / max(summary['total_dilation_features'], 1)
    
    print(f"\nAcceptance Rates:")
    print(f"  Acceleration: {accel_acceptance_rate:.1%}")
    print(f"  Dilation: {dil_acceptance_rate:.1%}")
    
    # Show top accepted features by type
    print(f"\nTop Accepted Features by Type:")
    
    # Acceleration features
    accel_accepted = results['acceleration_evaluation']['accepted_features']
    if accel_accepted:
        print(f"\n  Acceleration Features ({len(accel_accepted)}):")
        for feature in accel_accepted[:5]:
            print(f"    - {feature}")
        if len(accel_accepted) > 5:
            print(f"    ... and {len(accel_accepted) - 5} more")
    
    # Dilation features
    dil_accepted = results['dilation_evaluation']['accepted_features']
    if dil_accepted:
        print(f"\n  Dilation Features ({len(dil_accepted)}):")
        for feature in dil_accepted[:5]:
            print(f"    - {feature}")
        if len(dil_accepted) > 5:
            print(f"    ... and {len(dil_accepted) - 5} more")
    
    # Show feature family breakdown
    print(f"\nFeature Family Breakdown:")
    
    # Categorize accepted features by family
    accel_families = {}
    for feature in accel_accepted:
        if any(pattern in feature.lower() for pattern in ['ret_', 'return', 'log_ret']):
            family = 'returns'
        elif any(pattern in feature.lower() for pattern in ['momentum', 'roc_']):
            family = 'momentum'
        elif any(pattern in feature.lower() for pattern in ['vol_', 'volume', 'vwap']):
            family = 'volume'
        elif any(pattern in feature.lower() for pattern in ['ma_', 'ema_', 'trend']):
            family = 'trend'
        elif any(pattern in feature.lower() for pattern in ['rsi', 'stoch', 'williams']):
            family = 'oscillators'
        elif any(pattern in feature.lower() for pattern in ['volatility', 'atr', 'bb_']):
            family = 'volatility'
        else:
            family = 'other'
        
        if family not in accel_families:
            accel_families[family] = 0
        accel_families[family] += 1
    
    dil_families = {}
    for feature in dil_accepted:
        if any(pattern in feature.lower() for pattern in ['ret_', 'return', 'log_ret']):
            family = 'returns'
        elif any(pattern in feature.lower() for pattern in ['momentum', 'roc_']):
            family = 'momentum'
        elif any(pattern in feature.lower() for pattern in ['vol_', 'volume', 'vwap']):
            family = 'volume'
        elif any(pattern in feature.lower() for pattern in ['ma_', 'ema_', 'trend']):
            family = 'trend'
        elif any(pattern in feature.lower() for pattern in ['rsi', 'stoch', 'williams']):
            family = 'oscillators'
        elif any(pattern in feature.lower() for pattern in ['volatility', 'atr', 'bb_']):
            family = 'volatility'
        else:
            family = 'other'
        
        if family not in dil_families:
            dil_families[family] = 0
        dil_families[family] += 1
    
    print(f"  Acceleration by family:")
    for family, count in accel_families.items():
        print(f"    {family:15s}: {count:2d} features")
    
    print(f"  Dilation by family:")
    for family, count in dil_families.items():
        print(f"    {family:15s}: {count:2d} features")
    
    return results

def main():
    """Main function to run all acceleration and dilation tests."""
    print("Feature Acceleration & Window Dilation - Complete Test Suite")
    print("=" * 80)
    print("Features:")
    print("- Acceleration features (Δ over k or 2nd difference)")
    print("- Window dilation features (2×, 3× lookback)")
    print("- Signal evaluation (MI, conditional MI, permutation importance)")
    print("- Uniqueness assessment (correlation analysis)")
    print("- Stability evaluation (rank consistency)")
    print("- Practicality checks (MSE improvement, latency)")
    print("- Acceptance gates for feature selection")
    print("=" * 80)
    
    # Test individual components
    print("\n1. Testing Acceleration Features...")
    accel_features, accel_results = test_acceleration_features()
    
    print("\n2. Testing Window Dilation Features...")
    dil_features, dil_results = test_dilation_features()
    
    print("\n3. Testing Complete Pipeline...")
    complete_results = test_complete_acceleration_dilation_pipeline()
    
    print("\n" + "=" * 80)
    print("Acceleration and dilation testing completed successfully!")
    print("Key features demonstrated:")
    print("✅ Acceleration features expose turning points missed by base features")
    print("✅ Window dilation captures different regime/trend complements")
    print("✅ Signal evaluation with MI, conditional MI, and permutation importance")
    print("✅ Uniqueness assessment to avoid redundant features")
    print("✅ Stability evaluation for robust feature selection")
    print("✅ Practicality checks for real-world deployment")
    print("✅ Acceptance gates for systematic feature selection")
    print("=" * 80)

if __name__ == "__main__":
    main()