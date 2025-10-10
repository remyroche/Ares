"""
Comprehensive Feature Comparison Example

This script demonstrates the complete enhanced feature comparison framework with:
- Time-series safe validation (purged CV, walk-forward, out-of-sample)
- Stability metrics (bootstrap CIs, rank consistency, Jaccard overlap, temporal drift)
- Comprehensive diagnostics (target leakage, scaling sensitivity, collinearity, shadow features)
- Standardized method settings (LGBM/SHAP, LASSO, MI, Permutation Importance)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from feature_comparison.enhanced_relevance_analyzer import EnhancedRelevanceAnalyzer
from feature_comparison.standardized_features import StandardizedFeatureGenerator
from feature_comparison.time_series_validation import TimeSeriesValidator
from feature_comparison.stability_metrics import FeatureStabilityAnalyzer
from feature_comparison.diagnostics import FeatureDiagnostics
from feature_comparison.method_settings import MethodSettings

# Suppress warnings
warnings.filterwarnings('ignore')

def create_realistic_market_data_with_regimes(n_samples: int = 3000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create realistic market data with multiple regimes and assets.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (data, groups) DataFrames
    """
    np.random.seed(42)
    
    # Generate multiple assets
    assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    n_assets = len(assets)
    samples_per_asset = n_samples // n_assets
    
    all_data = []
    all_groups = []
    
    for i, asset in enumerate(assets):
        # Generate asset-specific data
        asset_samples = samples_per_asset + (n_samples % n_assets if i == 0 else 0)
        
        # Different regimes for each asset
        n_regimes = 3
        regime_length = asset_samples // n_regimes
        
        prices = []
        volumes = []
        regimes = []
        
        for regime in range(n_regimes):
            if regime == 0:
                # Bull market regime
                trend = 0.002 + i * 0.0005
                volatility = 0.02 + i * 0.005
                volume_trend = 0.001
                regime_name = 'bull'
            elif regime == 1:
                # Sideways market regime
                trend = 0.0001
                volatility = 0.015 + i * 0.003
                volume_trend = -0.0005
                regime_name = 'sideways'
            else:
                # High volatility regime
                trend = 0.0005
                volatility = 0.04 + i * 0.01
                volume_trend = 0.002
                regime_name = 'high_vol'
            
            # Generate regime data
            regime_returns = np.random.normal(trend, volatility, regime_length)
            regime_prices = 100 * (1 + i * 0.5) * np.exp(np.cumsum(regime_returns))
            
            # Add volume with trend
            regime_volumes = np.random.lognormal(
                10 + volume_trend * np.arange(regime_length), 
                1.0 + 0.2 * i
            )
            
            prices.extend(regime_prices)
            volumes.extend(regime_volumes)
            regimes.extend([regime_name] * regime_length)
        
        # Generate OHLCV data
        asset_data = pd.DataFrame({
            'timestamp': pd.date_range(f'2023-01-01', periods=asset_samples, freq='1H'),
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.003, asset_samples))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.003, asset_samples))),
            'close': prices,
            'volume': volumes
        })
        
        # Ensure OHLC constraints
        asset_data['high'] = np.maximum(asset_data['high'], np.maximum(asset_data['open'], asset_data['close']))
        asset_data['low'] = np.minimum(asset_data['low'], np.minimum(asset_data['open'], asset_data['close']))
        
        # Add asset and regime information
        asset_data['asset'] = asset
        asset_data['regime'] = regimes
        
        all_data.append(asset_data)
        
        # Create groups DataFrame
        asset_groups = pd.DataFrame({
            'timestamp': asset_data['timestamp'],
            'asset': asset_data['asset'],
            'regime': asset_data['regime']
        })
        all_groups.append(asset_groups)
    
    # Combine all assets
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_groups = pd.concat(all_groups, ignore_index=True)
    
    # Sort by timestamp
    combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
    combined_groups = combined_groups.sort_values('timestamp').reset_index(drop=True)
    
    return combined_data, combined_groups

def test_time_series_validation():
    """Test time-series validation methods."""
    print("Testing Time-Series Validation Methods...")
    print("=" * 60)
    
    # Create sample data
    data, groups = create_realistic_market_data_with_regimes(1000)
    
    # Generate features
    feature_generator = StandardizedFeatureGenerator(data)
    versions = feature_generator.generate_standardized_features()
    
    # Use initial version for testing
    X = versions['initial']
    feature_cols = [col for col in X.columns 
                   if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
    X_features = X[feature_cols].dropna()
    
    # Create target
    y = X['close'].pct_change().dropna()
    
    # Align data
    common_idx = X_features.index.intersection(y.index)
    X_aligned = X_features.loc[common_idx]
    y_aligned = y.loc[common_idx]
    groups_aligned = groups.loc[common_idx]
    
    # Test time-series validator
    validator = TimeSeriesValidator(n_splits=3, embargo_periods=2)
    
    # Test different validation methods
    from feature_comparison.method_settings import MethodSettings
    method_settings = MethodSettings()
    lgbm_model = method_settings.create_lgbm_model('regression')
    
    print("Running purged CV validation...")
    purged_results = validator.validate_model(
        lgbm_model, X_aligned, y_aligned, groups_aligned, 'purged_cv'
    )
    print(f"Purged CV R²: {purged_results['mean_scores']['r2']:.4f} ± {purged_results['std_scores']['r2']:.4f}")
    
    print("Running walk-forward validation...")
    walk_forward_results = validator.validate_model(
        lgbm_model, X_aligned, y_aligned, groups_aligned, 'walk_forward'
    )
    print(f"Walk-forward R²: {walk_forward_results['mean_scores']['r2']:.4f} ± {walk_forward_results['std_scores']['r2']:.4f}")
    
    print("Running out-of-asset validation...")
    out_of_asset_results = validator.validate_model(
        lgbm_model, X_aligned, y_aligned, groups_aligned, 'out_of_asset'
    )
    print(f"Out-of-asset R²: {out_of_asset_results['mean_scores']['r2']:.4f} ± {out_of_asset_results['std_scores']['r2']:.4f}")
    
    print("Running out-of-regime validation...")
    out_of_regime_results = validator.validate_model(
        lgbm_model, X_aligned, y_aligned, groups_aligned, 'out_of_regime'
    )
    print(f"Out-of-regime R²: {out_of_regime_results['mean_scores']['r2']:.4f} ± {out_of_regime_results['std_scores']['r2']:.4f}")
    
    return {
        'purged_cv': purged_results,
        'walk_forward': walk_forward_results,
        'out_of_asset': out_of_asset_results,
        'out_of_regime': out_of_regime_results
    }

def test_stability_metrics():
    """Test stability metrics."""
    print("\nTesting Stability Metrics...")
    print("=" * 60)
    
    # Create sample data
    data, groups = create_realistic_market_data_with_regimes(1000)
    
    # Generate features
    feature_generator = StandardizedFeatureGenerator(data)
    versions = feature_generator.generate_standardized_features()
    
    # Use initial version for testing
    X = versions['initial']
    feature_cols = [col for col in X.columns 
                   if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
    X_features = X[feature_cols].dropna()
    
    # Create target
    y = X['close'].pct_change().dropna()
    
    # Align data
    common_idx = X_features.index.intersection(y.index)
    X_aligned = X_features.loc[common_idx]
    y_aligned = y.loc[common_idx]
    
    # Create mock analysis results
    from feature_comparison.method_settings import MethodSettings
    method_settings = MethodSettings()
    
    # LGBM analysis
    lgbm_model = method_settings.create_lgbm_model('regression')
    lgbm_model.fit(X_aligned, y_aligned)
    lgbm_importance = pd.Series(lgbm_model.feature_importances_, index=X_aligned.columns)
    
    # LASSO analysis
    lasso_model = method_settings.create_lasso_model()
    lasso_model.fit(X_aligned, y_aligned)
    lasso_importance = pd.Series(np.abs(lasso_model.coef_), index=X_aligned.columns)
    
    # MI analysis
    mi_importance = method_settings.calculate_mutual_info_importance(X_aligned, y_aligned)
    
    # Permutation importance
    perm_importance = method_settings.calculate_permutation_importance(lgbm_model, X_aligned, y_aligned)
    
    # Create analysis results
    analysis_results = {
        'lgbm': {'feature_importance': lgbm_importance},
        'lasso': {'feature_importance': lasso_importance},
        'mutual_info': {'feature_importance': mi_importance},
        'permutation': {'feature_importance': perm_importance}
    }
    
    # Test stability analyzer
    stability_analyzer = FeatureStabilityAnalyzer()
    
    print("Calculating rank consistency...")
    rank_consistency = stability_analyzer.calculate_rank_consistency(analysis_results)
    if 'overall_consistency' in rank_consistency:
        mean_corr = rank_consistency['overall_consistency']['mean_spearman_corr']
        print(f"Mean rank correlation: {mean_corr:.4f}")
    
    print("Calculating Jaccard overlap...")
    jaccard_overlap = stability_analyzer.calculate_jaccard_overlap(analysis_results)
    for k, metrics in jaccard_overlap.items():
        print(f"Jaccard@k={k}: {metrics['mean_jaccard']:.4f} ± {metrics['std_jaccard']:.4f}")
    
    return {
        'rank_consistency': rank_consistency,
        'jaccard_overlap': jaccard_overlap
    }

def test_diagnostics():
    """Test comprehensive diagnostics."""
    print("\nTesting Comprehensive Diagnostics...")
    print("=" * 60)
    
    # Create sample data
    data, groups = create_realistic_market_data_with_regimes(1000)
    
    # Generate features
    feature_generator = StandardizedFeatureGenerator(data)
    versions = feature_generator.generate_standardized_features()
    
    # Use initial version for testing
    X = versions['initial']
    feature_cols = [col for col in X.columns 
                   if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
    X_features = X[feature_cols].dropna()
    
    # Create target
    y = X['close'].pct_change().dropna()
    
    # Align data
    common_idx = X_features.index.intersection(y.index)
    X_aligned = X_features.loc[common_idx]
    y_aligned = y.loc[common_idx]
    groups_aligned = groups.loc[common_idx]
    
    # Test diagnostics
    diagnostics = FeatureDiagnostics()
    
    print("Running comprehensive diagnostics...")
    diagnostics_results = diagnostics.run_comprehensive_diagnostics(
        X_aligned, y_aligned, 
        timestamp_col='timestamp',
        vwap_cols=['vwap_w20'] if 'vwap_w20' in X_aligned.columns else None
    )
    
    # Print diagnostics summary
    if 'summary' in diagnostics_results:
        summary = diagnostics_results['summary']
        print(f"Tests passed: {summary['passed_tests']}")
        print(f"Tests failed: {summary['failed_tests']}")
        print(f"Warnings: {summary['warnings']}")
        
        if summary['critical_issues']:
            print("Critical issues:")
            for issue in summary['critical_issues']:
                print(f"  - {issue}")
    
    return diagnostics_results

def test_method_settings():
    """Test standardized method settings."""
    print("\nTesting Standardized Method Settings...")
    print("=" * 60)
    
    # Create sample data
    data, groups = create_realistic_market_data_with_regimes(1000)
    
    # Generate features
    feature_generator = StandardizedFeatureGenerator(data)
    versions = feature_generator.generate_standardized_features()
    
    # Use initial version for testing
    X = versions['initial']
    feature_cols = [col for col in X.columns 
                   if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
    X_features = X[feature_cols].dropna()
    
    # Create target
    y = X['close'].pct_change().dropna()
    
    # Align data
    common_idx = X_features.index.intersection(y.index)
    X_aligned = X_features.loc[common_idx]
    y_aligned = y.loc[common_idx]
    
    # Test method settings
    method_settings = MethodSettings()
    
    print("Testing LGBM settings...")
    lgbm_model = method_settings.create_lgbm_model('regression')
    lgbm_model.fit(X_aligned, y_aligned)
    lgbm_r2 = lgbm_model.score(X_aligned, y_aligned)
    print(f"LGBM R²: {lgbm_r2:.4f}")
    
    print("Testing LASSO settings...")
    lasso_model = method_settings.create_lasso_model()
    lasso_model.fit(X_aligned, y_aligned)
    lasso_r2 = lasso_model.score(X_aligned, y_aligned)
    print(f"LASSO R²: {lasso_r2:.4f}")
    
    print("Testing Ridge settings...")
    ridge_model = method_settings.create_ridge_model()
    ridge_model.fit(X_aligned, y_aligned)
    ridge_r2 = ridge_model.score(X_aligned, y_aligned)
    print(f"Ridge R²: {ridge_r2:.4f}")
    
    print("Testing SHAP importance...")
    shap_importance = method_settings.calculate_shap_importance(lgbm_model, X_aligned)
    print(f"SHAP importance calculated for {len(shap_importance)} features")
    
    print("Testing Mutual Information...")
    mi_importance = method_settings.calculate_mutual_info_importance(X_aligned, y_aligned)
    print(f"MI importance calculated for {len(mi_importance)} features")
    
    print("Testing Permutation Importance...")
    perm_importance = method_settings.calculate_permutation_importance(lgbm_model, X_aligned, y_aligned)
    print(f"Permutation importance calculated for {len(perm_importance)} features")
    
    # Test regularization path
    print("Testing regularization path...")
    reg_path = method_settings.get_regularization_path(lasso_model)
    if 'alphas' in reg_path:
        print(f"LASSO regularization path: {len(reg_path['alphas'])} alpha values")
        print(f"Optimal alpha: {reg_path['alpha_optimal']:.6f}")
    
    return {
        'lgbm_r2': lgbm_r2,
        'lasso_r2': lasso_r2,
        'ridge_r2': ridge_r2,
        'shap_importance': shap_importance,
        'mi_importance': mi_importance,
        'perm_importance': perm_importance,
        'regularization_path': reg_path
    }

def run_comprehensive_analysis():
    """Run comprehensive enhanced analysis."""
    print("\nRunning Comprehensive Enhanced Analysis...")
    print("=" * 70)
    
    # Create realistic data
    data, groups = create_realistic_market_data_with_regimes(2000)
    print(f"Created data with shape: {data.shape}")
    
    # Generate standardized features
    feature_generator = StandardizedFeatureGenerator(data, enable_matrix_ops=True)
    versions = feature_generator.generate_standardized_features()
    
    # Use initial version for analysis
    X = versions['initial']
    feature_cols = [col for col in X.columns 
                   if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
    X_features = X[feature_cols].dropna()
    
    # Create target
    y = X['close'].pct_change().dropna()
    
    # Align data
    common_idx = X_features.index.intersection(y.index)
    X_aligned = X_features.loc[common_idx]
    y_aligned = y.loc[common_idx]
    groups_aligned = groups.loc[common_idx]
    
    print(f"Aligned data: {X_aligned.shape[0]} samples, {X_aligned.shape[1]} features")
    
    # Initialize enhanced analyzer
    analyzer = EnhancedRelevanceAnalyzer(
        scaling_method='robust',
        random_state=42,
        enable_diagnostics=True,
        enable_stability=True
    )
    
    # Run comprehensive analysis
    results = analyzer.comprehensive_analysis(
        X_aligned, y_aligned,
        task_type='regression',
        groups=groups_aligned,
        vwap_cols=['vwap_w20'] if 'vwap_w20' in X_aligned.columns else None
    )
    
    # Print comprehensive summary
    analyzer.print_comprehensive_summary()
    
    # Get additional reports
    feature_ranking = analyzer.get_feature_ranking_summary()
    stability_report = analyzer.get_stability_report()
    diagnostics_report = analyzer.get_diagnostics_report()
    
    print(f"\nTop 10 Features by Average Rank:")
    print(feature_ranking.head(10)[['avg_rank', 'rank_std', 'rank_cv']])
    
    if stability_report and 'overall_stability_score' in stability_report:
        print(f"\nOverall Stability Score: {stability_report['overall_stability_score']:.4f}")
    
    return results

def main():
    """Main function to run all comprehensive tests."""
    print("Comprehensive Feature Comparison Framework - Complete Test Suite")
    print("=" * 80)
    print("Features:")
    print("- Time-series safe validation (purged CV, walk-forward, out-of-sample)")
    print("- Stability metrics (bootstrap CIs, rank consistency, Jaccard overlap)")
    print("- Comprehensive diagnostics (target leakage, scaling sensitivity, collinearity)")
    print("- Standardized method settings (LGBM/SHAP, LASSO, MI, Permutation Importance)")
    print("- Enhanced relevance analysis with all components")
    print("=" * 80)
    
    # Test individual components
    print("\n1. Testing Time-Series Validation...")
    validation_results = test_time_series_validation()
    
    print("\n2. Testing Stability Metrics...")
    stability_results = test_stability_metrics()
    
    print("\n3. Testing Diagnostics...")
    diagnostics_results = test_diagnostics()
    
    print("\n4. Testing Method Settings...")
    method_results = test_method_settings()
    
    print("\n5. Running Comprehensive Analysis...")
    comprehensive_results = run_comprehensive_analysis()
    
    print("\n" + "=" * 80)
    print("Comprehensive testing completed successfully!")
    print("Key improvements:")
    print("✅ Time-series safe validation with purged CV and embargo")
    print("✅ Walk-forward validation mirroring deployment latency")
    print("✅ Out-of-sample testing (out-of-asset, out-of-regime)")
    print("✅ Bootstrap stability with confidence intervals")
    print("✅ Rank consistency and Jaccard overlap metrics")
    print("✅ Comprehensive diagnostics for common pitfalls")
    print("✅ Standardized method settings for reproducibility")
    print("✅ Enhanced relevance analysis with all components")
    print("=" * 80)

if __name__ == "__main__":
    main()