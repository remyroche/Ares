"""
Analyst-Labeler Integration Example

This script demonstrates how to integrate feature relevance analysis with
analyst-labeler approaches for predicting price action and market movements.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from feature_comparison.analyst_labeler_integration import AnalystLabelerIntegration
from feature_comparison.standardized_features import StandardizedFeatureGenerator
from feature_comparison.enhanced_relevance_analyzer import EnhancedRelevanceAnalyzer

# Suppress warnings
warnings.filterwarnings('ignore')

def create_market_data_with_regimes(n_samples: int = 3000) -> pd.DataFrame:
    """
    Create realistic market data with multiple regimes for analyst-labeler testing.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with OHLCV data and regime information
    """
    np.random.seed(42)
    
    # Generate price data with multiple regimes
    n_regimes = 4
    regime_length = n_samples // n_regimes
    
    prices = []
    volumes = []
    regimes = []
    
    for regime in range(n_regimes):
        if regime == 0:
            # Bull market regime - trending up
            trend = 0.002
            volatility = 0.015
            volume_trend = 0.001
            regime_name = 'bull_market'
        elif regime == 1:
            # Sideways market regime - low volatility
            trend = 0.0001
            volatility = 0.008
            volume_trend = -0.0005
            regime_name = 'sideways_market'
        elif regime == 2:
            # High volatility regime - choppy
            trend = 0.0005
            volatility = 0.035
            volume_trend = 0.003
            regime_name = 'high_volatility'
        else:
            # Bear market regime - trending down
            trend = -0.0015
            volatility = 0.025
            volume_trend = 0.002
            regime_name = 'bear_market'
        
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

def test_analyst_labeler_integration():
    """Test analyst-labeler integration with different target types."""
    print("Testing Analyst-Labeler Integration...")
    print("=" * 60)
    
    # Create market data
    data = create_market_data_with_regimes(2000)
    print(f"Created market data with shape: {data.shape}")
    print(f"Regimes: {data['regime'].value_counts().to_dict()}")
    
    # Generate standardized features
    feature_generator = StandardizedFeatureGenerator(data, enable_matrix_ops=True)
    versions = feature_generator.generate_standardized_features()
    
    # Use initial version for analysis
    X = versions['initial']
    feature_cols = [col for col in X.columns 
                   if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
    X_features = X[feature_cols].dropna()
    
    print(f"Generated {len(feature_cols)} standardized features")
    
    # Initialize analyst-labeler integration
    analyst_integration = AnalystLabelerIntegration(
        price_threshold=0.002,  # 0.2% threshold for significant moves
        volatility_threshold=0.02,  # 2% volatility threshold
        lookforward_periods=1  # 1-period lookahead
    )
    
    # Create analyst-style targets
    print("\nCreating analyst-style targets...")
    targets = analyst_integration.create_analyst_style_targets(data)
    
    print(f"Created {len(targets)} target types:")
    for target_name, target_series in targets.items():
        print(f"  {target_name}: {target_series.value_counts().to_dict()}")
    
    # Evaluate feature relevance for different targets
    print("\nEvaluating feature relevance for different targets...")
    results = analyst_integration.evaluate_feature_relevance_for_targets(
        X_features, targets, methods=['lgbm', 'lasso', 'mi', 'permutation']
    )
    
    # Create analyst-style report
    print("\nGenerating analyst-style report...")
    report = analyst_integration.create_analyst_style_report(results)
    
    # Print summary
    analyst_integration.print_analyst_style_summary(report)
    
    return results, report

def test_price_action_prediction():
    """Test specific price action prediction scenarios."""
    print("\nTesting Price Action Prediction Scenarios...")
    print("=" * 60)
    
    # Create market data
    data = create_market_data_with_regimes(1500)
    
    # Generate features
    feature_generator = StandardizedFeatureGenerator(data)
    versions = feature_generator.generate_standardized_features()
    
    X = versions['initial']
    feature_cols = [col for col in X.columns 
                   if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
    X_features = X[feature_cols].dropna()
    
    # Test different price action scenarios
    analyst_integration = AnalystLabelerIntegration(
        price_threshold=0.001,  # 0.1% threshold
        lookforward_periods=1
    )
    
    # Scenario 1: Directional prediction
    print("\nScenario 1: Directional Price Movement Prediction")
    print("-" * 50)
    
    directional_labels = analyst_integration.create_price_action_labels(
        data['close'], method='directional'
    )
    
    # Align data
    common_idx = X_features.index.intersection(directional_labels.index)
    X_aligned = X_features.loc[common_idx]
    y_aligned = directional_labels.loc[common_idx]
    
    # Remove NaN values
    valid_mask = ~(X_aligned.isna().any(axis=1) | y_aligned.isna())
    X_clean = X_aligned[valid_mask]
    y_clean = y_aligned[valid_mask]
    
    print(f"Directional labels distribution: {y_clean.value_counts().to_dict()}")
    
    # Evaluate features for directional prediction
    directional_results = analyst_integration.evaluate_feature_relevance_for_targets(
        X_clean, {'directional': y_clean}, methods=['lgbm', 'lasso', 'mi']
    )
    
    if 'directional' in directional_results and 'lgbm' in directional_results['directional']:
        lgbm_results = directional_results['directional']['lgbm']
        if 'performance' in lgbm_results:
            print(f"LGBM Performance: {lgbm_results['performance']}")
        
        if 'feature_importance' in lgbm_results:
            top_features = lgbm_results['feature_importance'].nlargest(5)
            print("Top 5 features for directional prediction:")
            for feature, importance in top_features.items():
                print(f"  {feature}: {importance:.4f}")
    
    # Scenario 2: Magnitude prediction
    print("\nScenario 2: Price Movement Magnitude Prediction")
    print("-" * 50)
    
    magnitude_labels = analyst_integration.create_price_action_labels(
        data['close'], method='magnitude'
    )
    
    # Align data
    common_idx = X_features.index.intersection(magnitude_labels.index)
    X_aligned = X_features.loc[common_idx]
    y_aligned = magnitude_labels.loc[common_idx]
    
    # Remove NaN values
    valid_mask = ~(X_aligned.isna().any(axis=1) | y_aligned.isna())
    X_clean = X_aligned[valid_mask]
    y_clean = y_aligned[valid_mask]
    
    print(f"Magnitude labels distribution: {y_clean.value_counts().to_dict()}")
    
    # Evaluate features for magnitude prediction
    magnitude_results = analyst_integration.evaluate_feature_relevance_for_targets(
        X_clean, {'magnitude': y_clean}, methods=['lgbm', 'lasso', 'mi']
    )
    
    if 'magnitude' in magnitude_results and 'lgbm' in magnitude_results['magnitude']:
        lgbm_results = magnitude_results['magnitude']['lgbm']
        if 'performance' in lgbm_results:
            print(f"LGBM Performance: {lgbm_results['performance']}")
        
        if 'feature_importance' in lgbm_results:
            top_features = lgbm_results['feature_importance'].nlargest(5)
            print("Top 5 features for magnitude prediction:")
            for feature, importance in top_features.items():
                print(f"  {feature}: {importance:.4f}")
    
    # Scenario 3: Regime prediction
    print("\nScenario 3: Market Regime Prediction")
    print("-" * 50)
    
    regime_labels = analyst_integration.create_price_action_labels(
        data['close'], method='regime'
    )
    
    # Align data
    common_idx = X_features.index.intersection(regime_labels.index)
    X_aligned = X_features.loc[common_idx]
    y_aligned = regime_labels.loc[common_idx]
    
    # Remove NaN values
    valid_mask = ~(X_aligned.isna().any(axis=1) | y_aligned.isna())
    X_clean = X_aligned[valid_mask]
    y_clean = y_aligned[valid_mask]
    
    print(f"Regime labels distribution: {y_clean.value_counts().to_dict()}")
    
    # Evaluate features for regime prediction
    regime_results = analyst_integration.evaluate_feature_relevance_for_targets(
        X_clean, {'regime': y_clean}, methods=['lgbm', 'lasso', 'mi']
    )
    
    if 'regime' in regime_results and 'lgbm' in regime_results['regime']:
        lgbm_results = regime_results['regime']['lgbm']
        if 'performance' in lgbm_results:
            print(f"LGBM Performance: {lgbm_results['performance']}")
        
        if 'feature_importance' in lgbm_results:
            top_features = lgbm_results['feature_importance'].nlargest(5)
            print("Top 5 features for regime prediction:")
            for feature, importance in top_features.items():
                print(f"  {feature}: {importance:.4f}")
    
    return {
        'directional': directional_results,
        'magnitude': magnitude_results,
        'regime': regime_results
    }

def test_enhanced_analysis_with_analyst_targets():
    """Test enhanced analysis with analyst-style targets."""
    print("\nTesting Enhanced Analysis with Analyst-Style Targets...")
    print("=" * 70)
    
    # Create market data
    data = create_market_data_with_regimes(2000)
    
    # Generate features
    feature_generator = StandardizedFeatureGenerator(data)
    versions = feature_generator.generate_standardized_features()
    
    X = versions['initial']
    feature_cols = [col for col in X.columns 
                   if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
    X_features = X[feature_cols].dropna()
    
    # Create analyst-style targets
    analyst_integration = AnalystLabelerIntegration()
    targets = analyst_integration.create_analyst_style_targets(data)
    
    # Use price direction as primary target
    if 'price_direction' in targets:
        y = targets['price_direction']
        
        # Align data
        common_idx = X_features.index.intersection(y.index)
        X_aligned = X_features.loc[common_idx]
        y_aligned = y.loc[common_idx]
        
        # Remove NaN values
        valid_mask = ~(X_aligned.isna().any(axis=1) | y_aligned.isna())
        X_clean = X_aligned[valid_mask]
        y_clean = y_aligned[valid_mask]
        
        print(f"Using price direction as target: {y_clean.value_counts().to_dict()}")
        
        # Run enhanced analysis
        analyzer = EnhancedRelevanceAnalyzer(
            scaling_method='robust',
            enable_diagnostics=True,
            enable_stability=True
        )
        
        # Create groups for time-series validation
        groups = pd.DataFrame({
            'timestamp': data.loc[common_idx, 'timestamp'],
            'regime': data.loc[common_idx, 'regime']
        })
        
        results = analyzer.comprehensive_analysis(
            X_clean, y_clean,
            task_type='classification',  # Price direction is classification
            groups=groups
        )
        
        # Print enhanced summary
        analyzer.print_comprehensive_summary()
        
        return results
    
    return None

def main():
    """Main function to run all analyst-labeler integration tests."""
    print("Analyst-Labeler Integration for Feature Relevance Analysis")
    print("=" * 80)
    print("Features:")
    print("- Price action prediction (directional, magnitude, regime)")
    print("- Analyst-style target creation and labeling")
    print("- Feature relevance evaluation for different targets")
    print("- Integration with enhanced analysis framework")
    print("=" * 80)
    
    # Test 1: Basic analyst-labeler integration
    print("\n1. Testing Analyst-Labeler Integration...")
    results1, report1 = test_analyst_labeler_integration()
    
    # Test 2: Price action prediction scenarios
    print("\n2. Testing Price Action Prediction Scenarios...")
    results2 = test_price_action_prediction()
    
    # Test 3: Enhanced analysis with analyst targets
    print("\n3. Testing Enhanced Analysis with Analyst-Style Targets...")
    results3 = test_enhanced_analysis_with_analyst_targets()
    
    print("\n" + "=" * 80)
    print("Analyst-labeler integration testing completed successfully!")
    print("Key features demonstrated:")
    print("✅ Price action prediction (directional, magnitude, regime)")
    print("✅ Analyst-style target creation and labeling")
    print("✅ Feature relevance evaluation for different targets")
    print("✅ Integration with enhanced analysis framework")
    print("✅ Time-series safe validation for price prediction")
    print("✅ Comprehensive diagnostics for price prediction models")
    print("=" * 80)

if __name__ == "__main__":
    main()