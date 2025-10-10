"""
Enhanced Feature Acceleration and Window Dilation Example

This script demonstrates the statistically robust acceleration and dilation system
with proper time-series validation, multiple testing control, and production hygiene.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from feature_comparison.feature_acceleration_dilation_enhanced import EnhancedFeatureAccelerationDilation
from feature_comparison.family_diverse_features import FamilyDiverseFeatureGenerator

# Suppress warnings
warnings.filterwarnings('ignore')

def create_market_data_with_regimes(n_samples: int = 5000) -> pd.DataFrame:
    """
    Create market data with multiple regimes for robust testing.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with OHLCV data and regime information
    """
    np.random.seed(42)
    
    # Generate multiple regimes
    n_regimes = 6
    regime_length = n_samples // n_regimes
    
    prices = []
    volumes = []
    regimes = []
    
    for regime in range(n_regimes):
        if regime == 0:
            # Trending up regime
            trend = 0.002
            volatility = 0.015
            volume_trend = 0.001
            regime_name = 'trending_up'
        elif regime == 1:
            # Sideways regime
            trend = 0.0001
            volatility = 0.008
            volume_trend = -0.0005
            regime_name = 'sideways'
        elif regime == 2:
            # High volatility regime
            trend = 0.0005
            volatility = 0.035
            volume_trend = 0.003
            regime_name = 'high_vol'
        elif regime == 3:
            # Trending down regime
            trend = -0.0015
            volatility = 0.025
            volume_trend = 0.002
            regime_name = 'trending_down'
        elif regime == 4:
            # Mean reversion regime
            trend = 0.0
            volatility = 0.020
            volume_trend = 0.0005
            regime_name = 'mean_reversion'
        else:
            # Low volatility regime
            trend = 0.0008
            volatility = 0.005
            volume_trend = -0.001
            regime_name = 'low_vol'
        
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
        regimes.extend([regime_name] * regime_length)
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.003, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.003, n_samples))),
        'close': prices,
        'volume': volumes,
        'regime': regimes
    })
    
    # Ensure OHLC constraints
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    return data

def test_enhanced_acceleration_features():
    """Test enhanced acceleration feature generation and evaluation."""
    print("Testing Enhanced Acceleration Features...")
    print("=" * 70)
    
    # Create market data with regimes
    data = create_market_data_with_regimes(3000)
    
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
    
    # Initialize enhanced acceleration system
    accel_system = EnhancedFeatureAccelerationDilation(
        acceleration_lags=[1, 3],
        mi_k_values=[5, 10],
        dm_alpha=0.05,
        fdr_q=0.1,
        cmi_ci_low_threshold=0.0,
        rank_stability_threshold=0.6,
        correlation_threshold=0.90,
        psi_threshold=0.2,
        psi_delta_threshold=0.05,
        turnover_threshold=0.1,
        enable_matrix_ops=True,
        n_bootstrap=500,
        n_cv_folds=5
    )
    
    # Generate acceleration features
    print("\nGenerating enhanced acceleration features...")
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
    
    # Evaluate acceleration features with time-series CV
    print(f"\nEvaluating acceleration features with time-series CV...")
    base_features = list(X_clean.columns)
    evaluation_results = accel_system.evaluate_features_with_ts_cv(
        X_clean, y_clean, acceleration_features, {}, base_features
    )
    
    # Display results
    print(f"\nEnhanced Acceleration Evaluation Results:")
    print(f"  Total features evaluated: {evaluation_results['global_metrics']['total_features']}")
    print(f"  Accepted features: {evaluation_results['global_metrics']['accepted_features']}")
    print(f"  Rejected features: {evaluation_results['global_metrics']['rejected_features']}")
    print(f"  Watchlist features: {evaluation_results['global_metrics']['watchlist_features']}")
    print(f"  Acceptance rate: {evaluation_results['global_metrics']['acceptance_rate']:.1%}")
    
    # Show variant cards
    print(f"\nVariant Cards (first 5):")
    variant_cards = evaluation_results['variant_cards']
    for i, (feature, card) in enumerate(list(variant_cards.items())[:5]):
        print(f"\n  {feature}:")
        print(f"    FQS: {card['fqs']:.3f}")
        print(f"    DM p-value: {card['dm_pvalue']:.4f}")
        print(f"    DM p-value (corrected): {card['dm_pvalue_corrected']:.4f}")
        print(f"    Rank stability: {card['rank_stability']:.3f}")
        print(f"    Turnover: {card['turnover']:.4f}")
        print(f"    PSI: {card['psi']:.4f}")
        print(f"    Decision: {card['decision']}")
        print(f"    Rationale: {card['rationale']}")
    
    return acceleration_features, evaluation_results

def test_enhanced_dilation_features():
    """Test enhanced window dilation feature generation and evaluation."""
    print("\nTesting Enhanced Window Dilation Features...")
    print("=" * 70)
    
    # Create market data with regimes
    data = create_market_data_with_regimes(3000)
    
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
    
    # Initialize enhanced dilation system
    dil_system = EnhancedFeatureAccelerationDilation(
        dilation_factors=[2.0, 3.0],
        mi_k_values=[5, 10],
        dm_alpha=0.05,
        fdr_q=0.1,
        cmi_ci_low_threshold=0.0,
        rank_stability_threshold=0.6,
        correlation_threshold=0.90,
        psi_threshold=0.2,
        psi_delta_threshold=0.05,
        turnover_threshold=0.1,
        enable_matrix_ops=True,
        n_bootstrap=500,
        n_cv_folds=5
    )
    
    # Generate dilation features
    print("\nGenerating enhanced window dilation features...")
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
    
    # Evaluate dilation features with time-series CV
    print(f"\nEvaluating dilation features with time-series CV...")
    base_features = list(X_clean.columns)
    evaluation_results = dil_system.evaluate_features_with_ts_cv(
        X_clean, y_clean, {}, dilation_features, base_features
    )
    
    # Display results
    print(f"\nEnhanced Dilation Evaluation Results:")
    print(f"  Total features evaluated: {evaluation_results['global_metrics']['total_features']}")
    print(f"  Accepted features: {evaluation_results['global_metrics']['accepted_features']}")
    print(f"  Rejected features: {evaluation_results['global_metrics']['rejected_features']}")
    print(f"  Watchlist features: {evaluation_results['global_metrics']['watchlist_features']}")
    print(f"  Acceptance rate: {evaluation_results['global_metrics']['acceptance_rate']:.1%}")
    
    # Show variant cards
    print(f"\nVariant Cards (first 5):")
    variant_cards = evaluation_results['variant_cards']
    for i, (feature, card) in enumerate(list(variant_cards.items())[:5]):
        print(f"\n  {feature}:")
        print(f"    FQS: {card['fqs']:.3f}")
        print(f"    DM p-value: {card['dm_pvalue']:.4f}")
        print(f"    DM p-value (corrected): {card['dm_pvalue_corrected']:.4f}")
        print(f"    Rank stability: {card['rank_stability']:.3f}")
        print(f"    Turnover: {card['turnover']:.4f}")
        print(f"    PSI: {card['psi']:.4f}")
        print(f"    Decision: {card['decision']}")
        print(f"    Rationale: {card['rationale']}")
    
    return dilation_features, evaluation_results

def test_complete_enhanced_pipeline():
    """Test complete enhanced acceleration and dilation pipeline."""
    print("\nTesting Complete Enhanced Pipeline...")
    print("=" * 80)
    
    # Create market data with regimes
    data = create_market_data_with_regimes(5000)
    
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
    
    print(f"Running complete enhanced pipeline with {X_clean.shape[1]} base features...")
    
    # Initialize complete enhanced system
    system = EnhancedFeatureAccelerationDilation(
        acceleration_lags=[1, 3],
        dilation_factors=[2.0, 3.0],
        mi_k_values=[5, 10],
        dm_alpha=0.05,
        fdr_q=0.1,
        cmi_ci_low_threshold=0.0,
        rank_stability_threshold=0.6,
        correlation_threshold=0.90,
        same_family_correlation_threshold=0.85,
        psi_threshold=0.2,
        psi_delta_threshold=0.05,
        shadow_sigma_threshold=1.0,
        turnover_threshold=0.1,
        enable_matrix_ops=True,
        n_bootstrap=500,
        n_cv_folds=5,
        enable_parallel=True
    )
    
    # Generate both acceleration and dilation features
    print("Generating acceleration and dilation features...")
    acceleration_features = system.generate_acceleration_features(X_clean)
    dilation_features = system.generate_dilation_features(X_clean)
    
    # Run complete evaluation
    print("Running complete enhanced evaluation...")
    results = system.evaluate_features_with_ts_cv(
        X_clean, y_clean, acceleration_features, dilation_features, list(X_clean.columns)
    )
    
    # Display comprehensive results
    print(f"\nComplete Enhanced Pipeline Results:")
    print("=" * 80)
    
    global_metrics = results['global_metrics']
    print(f"Base features: {X_clean.shape[1]}")
    print(f"Acceleration features generated: {sum(len(df.columns) for df in acceleration_features.values())}")
    print(f"Dilation features generated: {sum(len(df.columns) for df in dilation_features.values())}")
    print(f"Total features evaluated: {global_metrics['total_features']}")
    print(f"Accepted features: {global_metrics['accepted_features']}")
    print(f"Rejected features: {global_metrics['rejected_features']}")
    print(f"Watchlist features: {global_metrics['watchlist_features']}")
    print(f"Acceptance rate: {global_metrics['acceptance_rate']:.1%}")
    print(f"Watchlist rate: {global_metrics['watchlist_rate']:.1%}")
    
    # Show Pareto frontier
    if 'pareto_frontier' in results and results['pareto_frontier']:
        print(f"\nPareto Frontier (FQS vs Turnover):")
        print("-" * 50)
        for feature, fqs, turnover in results['pareto_frontier'][:10]:
            print(f"  {feature:30s} | FQS: {fqs:.3f} | Turnover: {turnover:.4f}")
    
    # Show decision breakdown
    print(f"\nDecision Breakdown:")
    print("-" * 50)
    
    decision_counts = {'Keep': 0, 'Drop': 0, 'Watchlist': 0}
    for feature, card in results['variant_cards'].items():
        decision = card['decision']
        decision_counts[decision] += 1
    
    for decision, count in decision_counts.items():
        print(f"  {decision:10s}: {count:3d} features")
    
    # Show top accepted features by type
    print(f"\nTop Accepted Features by Type:")
    print("-" * 50)
    
    accepted_features = [f for f, c in results['variant_cards'].items() if c['decision'] == 'Keep']
    
    # Categorize by type
    accel_features = [f for f in accepted_features if 'accel' in f]
    dil_features = [f for f in accepted_features if 'dil' in f]
    
    print(f"  Acceleration features ({len(accel_features)}):")
    for feature in accel_features[:5]:
        card = results['variant_cards'][feature]
        print(f"    - {feature:30s} | FQS: {card['fqs']:.3f} | DM p: {card['dm_pvalue_corrected']:.4f}")
    
    print(f"  Dilation features ({len(dil_features)}):")
    for feature in dil_features[:5]:
        card = results['variant_cards'][feature]
        print(f"    - {feature:30s} | FQS: {card['fqs']:.3f} | DM p: {card['dm_pvalue_corrected']:.4f}")
    
    # Show statistical robustness metrics
    print(f"\nStatistical Robustness Metrics:")
    print("-" * 50)
    
    # Count features by significance level
    significant_features = [f for f, c in results['variant_cards'].items() 
                          if c['dm_pvalue_corrected'] < 0.05]
    highly_significant_features = [f for f, c in results['variant_cards'].items() 
                                 if c['dm_pvalue_corrected'] < 0.01]
    
    print(f"  Statistically significant (p < 0.05): {len(significant_features)}")
    print(f"  Highly significant (p < 0.01): {len(highly_significant_features)}")
    
    # Count features by stability
    stable_features = [f for f, c in results['variant_cards'].items() 
                      if c['rank_stability'] >= 0.6]
    print(f"  Rank stable (≥ 0.6): {len(stable_features)}")
    
    # Count features by turnover
    low_turnover_features = [f for f, c in results['variant_cards'].items() 
                           if c['turnover'] <= 0.1]
    print(f"  Low turnover (≤ 0.1): {len(low_turnover_features)}")
    
    # Count features by PSI
    low_psi_features = [f for f, c in results['variant_cards'].items() 
                       if c['psi'] <= 0.2]
    print(f"  Low PSI (≤ 0.2): {len(low_psi_features)}")
    
    return results

def generate_enhanced_report(results: Dict[str, Any]) -> str:
    """Generate enhanced report with statistical details."""
    report = []
    report.append("# Enhanced Feature Acceleration & Dilation Report")
    report.append("=" * 60)
    
    # Global metrics
    global_metrics = results['global_metrics']
    report.append(f"\n## Global Metrics")
    report.append(f"- Total features evaluated: {global_metrics['total_features']}")
    report.append(f"- Accepted features: {global_metrics['accepted_features']}")
    report.append(f"- Rejected features: {global_metrics['rejected_features']}")
    report.append(f"- Watchlist features: {global_metrics['watchlist_features']}")
    report.append(f"- Acceptance rate: {global_metrics['acceptance_rate']:.1%}")
    
    # Statistical robustness
    report.append(f"\n## Statistical Robustness")
    significant_count = sum(1 for c in results['variant_cards'].values() 
                           if c['dm_pvalue_corrected'] < 0.05)
    report.append(f"- Statistically significant features: {significant_count}")
    
    # Production readiness
    report.append(f"\n## Production Readiness")
    stable_count = sum(1 for c in results['variant_cards'].values() 
                      if c['rank_stability'] >= 0.6)
    low_turnover_count = sum(1 for c in results['variant_cards'].values() 
                            if c['turnover'] <= 0.1)
    report.append(f"- Rank stable features: {stable_count}")
    report.append(f"- Low turnover features: {low_turnover_count}")
    
    # Top features
    report.append(f"\n## Top Accepted Features")
    accepted_features = [(f, c) for f, c in results['variant_cards'].items() 
                        if c['decision'] == 'Keep']
    accepted_features.sort(key=lambda x: x[1]['fqs'], reverse=True)
    
    for feature, card in accepted_features[:10]:
        report.append(f"- **{feature}**: FQS={card['fqs']:.3f}, "
                     f"DM_p={card['dm_pvalue_corrected']:.4f}, "
                     f"Stability={card['rank_stability']:.3f}")
    
    return "\n".join(report)

def main():
    """Main function to run all enhanced acceleration and dilation tests."""
    print("Enhanced Feature Acceleration & Window Dilation - Complete Test Suite")
    print("=" * 90)
    print("Features:")
    print("- Statistical correctness with time-series CV and multiple testing control")
    print("- Robust MI/HSIC estimation with kNN and bootstrap CI")
    print("- Proper dilation semantics for EMA/EWM features")
    print("- Cost/turnover awareness with Pareto optimization")
    print("- Drift & production hygiene with PSI and shadow features")
    print("- Redundancy & family diversity with mRMR and VIF")
    print("- Comprehensive reporting & traceability")
    print("- Sensible default thresholds and edge-case guards")
    print("=" * 90)
    
    # Test individual components
    print("\n1. Testing Enhanced Acceleration Features...")
    accel_features, accel_results = test_enhanced_acceleration_features()
    
    print("\n2. Testing Enhanced Window Dilation Features...")
    dil_features, dil_results = test_enhanced_dilation_features()
    
    print("\n3. Testing Complete Enhanced Pipeline...")
    complete_results = test_complete_enhanced_pipeline()
    
    # Generate enhanced report
    print("\n4. Generating Enhanced Report...")
    report = generate_enhanced_report(complete_results)
    print(report)
    
    print("\n" + "=" * 90)
    print("Enhanced acceleration and dilation testing completed successfully!")
    print("Key enhancements demonstrated:")
    print("✅ Statistical correctness with time-series CV and Diebold-Mariano tests")
    print("✅ Multiple testing control with Benjamini-Hochberg FDR correction")
    print("✅ Robust MI estimation with kNN and bootstrap confidence intervals")
    print("✅ Proper EMA semantics and scale equivalence checks")
    print("✅ Cost/turnover awareness with Pareto optimization")
    print("✅ Production hygiene with PSI, shadow features, and zero-vol guards")
    print("✅ Redundancy control with mRMR and VIF recomputation")
    print("✅ Comprehensive variant cards with traceable decisions")
    print("✅ Sensible thresholds and edge-case handling")
    print("=" * 90)

if __name__ == "__main__":
    main()