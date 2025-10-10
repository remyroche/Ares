"""
Family-Diverse Feature Generation Example

This script demonstrates how the pre-screening pipeline generates and selects
features from different families to ensure diversity.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from feature_comparison.family_diverse_features import FamilyDiverseFeatureGenerator
from feature_comparison.pre_screening_pipeline import PreScreeningPipeline
from feature_comparison.feature_scorecard import FeatureScorecard

# Suppress warnings
warnings.filterwarnings('ignore')

def create_market_data_for_families(n_samples: int = 3000) -> pd.DataFrame:
    """
    Create market data for family-diverse feature generation.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with OHLCV data
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

def test_family_diverse_feature_generation():
    """Test family-diverse feature generation."""
    print("Testing Family-Diverse Feature Generation...")
    print("=" * 60)
    
    # Create market data
    data = create_market_data_for_families(2000)
    print(f"Created market data with shape: {data.shape}")
    
    # Initialize family-diverse feature generator
    family_generator = FamilyDiverseFeatureGenerator(enable_matrix_ops=True)
    
    # Generate features from different families
    print("\nGenerating features from different families...")
    family_features = family_generator.generate_family_diverse_features(data)
    
    # Get family summary
    family_summary = family_generator.get_family_summary(family_features)
    
    print(f"\nFamily Feature Summary:")
    print("-" * 60)
    total_features = 0
    for family_name, summary in family_summary.items():
        print(f"{family_name:15s} | Features: {summary['n_features']:3d} | Memory: {summary['memory_usage_mb']:6.1f} MB")
        total_features += summary['n_features']
    
    print(f"{'Total':15s} | Features: {total_features:3d}")
    
    # Show sample features from each family
    print(f"\nSample Features by Family:")
    print("-" * 60)
    for family_name, family_df in family_features.items():
        original_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        feature_cols = [col for col in family_df.columns if col not in original_cols]
        
        if feature_cols:
            print(f"\n{family_name.upper()}:")
            sample_features = feature_cols[:5]  # Show first 5 features
            for feature in sample_features:
                print(f"  - {feature}")
            if len(feature_cols) > 5:
                print(f"  ... and {len(feature_cols) - 5} more")
    
    return family_features, family_summary

def test_pre_screening_with_families():
    """Test pre-screening pipeline with family-diverse features."""
    print("\nTesting Pre-screening with Family-Diverse Features...")
    print("=" * 60)
    
    # Create market data
    data = create_market_data_for_families(2500)
    
    # Create target
    y = data['close'].pct_change().dropna()
    
    # Initialize pre-screening pipeline with family generation
    pipeline = PreScreeningPipeline(
        top_k_per_bucket=15,
        mi_percentile=0.6,
        correlation_threshold=0.95,
        vif_threshold=10.0,
        n_total_features=80,
        enable_matrix_ops=True
    )
    
    # Run pre-screening with family-diverse features
    print("Running pre-screening with family-diverse features...")
    results = pipeline.run_pre_screening(data, y, generate_family_features=True)
    
    # Analyze results by family
    print(f"\nPre-screening Results by Family:")
    print("-" * 60)
    
    final_features = results['final_features']
    family_breakdown = {}
    
    for feature in final_features:
        # Determine family based on feature name
        if any(pattern in feature.lower() for pattern in ['ret_', 'return', 'log_ret', 'abs_ret', 'sq_ret']):
            family = 'returns'
        elif any(pattern in feature.lower() for pattern in ['momentum', 'roc_', 'price_position']):
            family = 'momentum'
        elif any(pattern in feature.lower() for pattern in ['vol_', 'volume', 'vwap', 'obv', 'vpt']):
            family = 'volume'
        elif any(pattern in feature.lower() for pattern in ['ma_', 'ema_', 'trend', 'macd', 'psar']):
            family = 'trend'
        elif any(pattern in feature.lower() for pattern in ['rsi', 'stoch', 'williams', 'cci', 'mfi']):
            family = 'oscillators'
        elif any(pattern in feature.lower() for pattern in ['volatility', 'atr', 'bb_', 'parkinson', 'gk_vol']):
            family = 'volatility'
        elif any(pattern in feature.lower() for pattern in ['vwap_', 'vwap_dev', 'vwap_band']):
            family = 'vwap'
        elif any(pattern in feature.lower() for pattern in ['tenkan', 'kijun', 'senkou', 'fib_', 'pivot']):
            family = 'technical'
        elif any(pattern in feature.lower() for pattern in ['skewness', 'kurtosis', 'autocorr', 'hurst', 'entropy']):
            family = 'statistical'
        else:
            family = 'other'
        
        if family not in family_breakdown:
            family_breakdown[family] = []
        family_breakdown[family].append(feature)
    
    # Display family breakdown
    for family, features in family_breakdown.items():
        print(f"{family:15s} | Selected: {len(features):2d} features")
        if len(features) <= 5:
            for feature in features:
                print(f"  - {feature}")
        else:
            for feature in features[:3]:
                print(f"  - {feature}")
            print(f"  ... and {len(features) - 3} more")
        print()
    
    # Show diversity metrics
    print(f"Diversity Metrics:")
    print(f"  Total families represented: {len(family_breakdown)}")
    print(f"  Features per family (mean): {np.mean([len(features) for features in family_breakdown.values()]):.1f}")
    print(f"  Features per family (std): {np.std([len(features) for features in family_breakdown.values()]):.1f}")
    
    return results, family_breakdown

def test_feature_scorecard_by_families():
    """Test feature scorecard analysis by families."""
    print("\nTesting Feature Scorecard by Families...")
    print("=" * 60)
    
    # Create market data
    data = create_market_data_for_families(2000)
    
    # Generate family features
    family_generator = FamilyDiverseFeatureGenerator(enable_matrix_ops=True)
    family_features = family_generator.generate_family_diverse_features(data)
    
    # Combine all family features
    all_features = {}
    for family_name, family_df in family_features.items():
        original_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        feature_cols = [col for col in family_df.columns if col not in original_cols]
        all_features[family_name] = feature_cols
    
    # Create combined feature matrix
    combined_features = []
    for family_df in family_features.values():
        original_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        feature_cols = [col for col in family_df.columns if col not in original_cols]
        combined_features.extend(feature_cols)
    
    # Create feature matrix
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
    
    print(f"Computing scorecard for {X_clean.shape[1]} features from {len(all_features)} families...")
    
    # Initialize feature scorecard
    scorecard = FeatureScorecard()
    
    # Compute scorecard
    scorecard_results = scorecard.compute_feature_scorecard(X_clean, y_clean, all_features)
    
    # Generate report
    report_df = scorecard.generate_scorecard_report(scorecard_results)
    
    # Analyze by family
    print(f"\nFeature Quality by Family:")
    print("-" * 60)
    
    family_quality = {}
    for family_name in all_features.keys():
        family_features_list = all_features[family_name]
        family_report = report_df[report_df['feature'].isin(family_features_list)]
        
        if len(family_report) > 0:
            family_quality[family_name] = {
                'n_features': len(family_report),
                'mean_fqs': family_report['fqs'].mean(),
                'std_fqs': family_report['fqs'].std(),
                'high_quality': len(family_report[family_report['fqs'] >= 0.7]),
                'medium_quality': len(family_report[(family_report['fqs'] >= 0.4) & (family_report['fqs'] < 0.7)]),
                'low_quality': len(family_report[family_report['fqs'] < 0.4])
            }
    
    # Display family quality metrics
    for family_name, quality in family_quality.items():
        print(f"\n{family_name.upper()}:")
        print(f"  Features: {quality['n_features']:3d} | Mean FQS: {quality['mean_fqs']:.3f} | Std FQS: {quality['std_fqs']:.3f}")
        print(f"  High quality: {quality['high_quality']:2d} | Medium quality: {quality['medium_quality']:2d} | Low quality: {quality['low_quality']:2d}")
        
        # Show top features from this family
        family_report = report_df[report_df['feature'].isin(all_features[family_name])]
        top_features = family_report.nlargest(3, 'fqs')
        if len(top_features) > 0:
            print(f"  Top features:")
            for idx, row in top_features.iterrows():
                print(f"    - {row['feature']:25s} | FQS: {row['fqs']:.3f}")
    
    return scorecard_results, report_df, family_quality

def test_complete_family_diverse_pipeline():
    """Test complete pipeline with family-diverse features."""
    print("\nTesting Complete Family-Diverse Pipeline...")
    print("=" * 70)
    
    # Create market data
    data = create_market_data_for_families(3000)
    
    # Create target
    y = data['close'].pct_change().dropna()
    
    print(f"Running complete pipeline with family-diverse features...")
    print(f"Data shape: {data.shape}")
    
    # Step 1: Generate family-diverse features
    print("\nStep 1: Generating family-diverse features...")
    family_generator = FamilyDiverseFeatureGenerator(enable_matrix_ops=True)
    family_features = family_generator.generate_family_diverse_features(data)
    
    family_summary = family_generator.get_family_summary(family_features)
    total_features = sum(summary['n_features'] for summary in family_summary.values())
    print(f"  Generated {total_features} features across {len(family_summary)} families")
    
    # Step 2: Pre-screening with family diversity
    print("\nStep 2: Pre-screening with family diversity...")
    pipeline = PreScreeningPipeline(
        top_k_per_bucket=20,
        mi_percentile=0.6,
        correlation_threshold=0.95,
        vif_threshold=10.0,
        n_total_features=100,
        enable_matrix_ops=True
    )
    
    prescreening_results = pipeline.run_pre_screening(data, y, generate_family_features=True)
    selected_features = prescreening_results['final_features']
    print(f"  Selected {len(selected_features)} features from pre-screening")
    
    # Step 3: Feature scorecard analysis
    print("\nStep 3: Feature scorecard analysis...")
    scorecard = FeatureScorecard()
    
    # Create feature matrix for selected features
    X_selected = pd.DataFrame(index=data.index)
    for family_df in family_features.values():
        for feature in selected_features:
            if feature in family_df.columns:
                X_selected[feature] = family_df[feature]
    
    # Align data
    common_idx = X_selected.index.intersection(y.index)
    X_aligned = X_selected.loc[common_idx]
    y_aligned = y.loc[common_idx]
    
    valid_mask = ~(X_aligned.isna().any(axis=1) | y_aligned.isna())
    X_clean = X_aligned[valid_mask]
    y_clean = y_aligned[valid_mask]
    
    scorecard_results = scorecard.compute_feature_scorecard(X_clean, y_clean)
    report_df = scorecard.generate_scorecard_report(scorecard_results)
    
    print(f"  Analyzed {len(report_df)} features with scorecard")
    
    # Step 4: Final analysis by family
    print("\nStep 4: Final analysis by family...")
    
    # Categorize final features by family
    final_family_breakdown = {}
    for feature in selected_features:
        if any(pattern in feature.lower() for pattern in ['ret_', 'return', 'log_ret', 'abs_ret', 'sq_ret']):
            family = 'returns'
        elif any(pattern in feature.lower() for pattern in ['momentum', 'roc_', 'price_position']):
            family = 'momentum'
        elif any(pattern in feature.lower() for pattern in ['vol_', 'volume', 'vwap', 'obv', 'vpt']):
            family = 'volume'
        elif any(pattern in feature.lower() for pattern in ['ma_', 'ema_', 'trend', 'macd', 'psar']):
            family = 'trend'
        elif any(pattern in feature.lower() for pattern in ['rsi', 'stoch', 'williams', 'cci', 'mfi']):
            family = 'oscillators'
        elif any(pattern in feature.lower() for pattern in ['volatility', 'atr', 'bb_', 'parkinson', 'gk_vol']):
            family = 'volatility'
        elif any(pattern in feature.lower() for pattern in ['vwap_', 'vwap_dev', 'vwap_band']):
            family = 'vwap'
        elif any(pattern in feature.lower() for pattern in ['tenkan', 'kijun', 'senkou', 'fib_', 'pivot']):
            family = 'technical'
        elif any(pattern in feature.lower() for pattern in ['skewness', 'kurtosis', 'autocorr', 'hurst', 'entropy']):
            family = 'statistical'
        else:
            family = 'other'
        
        if family not in final_family_breakdown:
            final_family_breakdown[family] = []
        final_family_breakdown[family].append(feature)
    
    # Display final results
    print(f"\nFinal Family-Diverse Selection Results:")
    print("-" * 70)
    print(f"Original features: {total_features}")
    print(f"Selected features: {len(selected_features)}")
    print(f"Reduction: {((total_features - len(selected_features)) / total_features * 100):.1f}%")
    print(f"Families represented: {len(final_family_breakdown)}")
    
    print(f"\nFeatures by Family:")
    for family, features in final_family_breakdown.items():
        print(f"  {family:15s} | {len(features):2d} features")
        
        # Show top features from this family
        family_report = report_df[report_df['feature'].isin(features)]
        if len(family_report) > 0:
            top_features = family_report.nlargest(3, 'fqs')
            for idx, row in top_features.iterrows():
                print(f"    - {row['feature']:25s} | FQS: {row['fqs']:.3f}")
    
    return {
        'family_features': family_features,
        'prescreening_results': prescreening_results,
        'scorecard_results': scorecard_results,
        'final_family_breakdown': final_family_breakdown
    }

def main():
    """Main function to run all family-diverse tests."""
    print("Family-Diverse Feature Generation - Complete Test Suite")
    print("=" * 80)
    print("Features:")
    print("- Returns family (basic returns, log returns, absolute, squared)")
    print("- Momentum family (price momentum, ROC, acceleration, position)")
    print("- Volume family (volume returns, ratios, VWAP, OBV, VPT)")
    print("- Trend family (MA, EMA, MACD, trend strength, PSAR)")
    print("- Oscillators family (RSI, Stochastic, Williams %R, CCI, MFI)")
    print("- Volatility family (rolling vol, Parkinson, Garman-Klass, ATR, BB)")
    print("- VWAP family (VWAP calculations, deviations, bands)")
    print("- Technical family (Ichimoku, Fibonacci, support/resistance)")
    print("- Statistical family (skewness, kurtosis, autocorr, Hurst, entropy)")
    print("- Cross-asset family (relative strength, momentum)")
    print("=" * 80)
    
    # Test individual components
    print("\n1. Testing Family-Diverse Feature Generation...")
    family_features, family_summary = test_family_diverse_feature_generation()
    
    print("\n2. Testing Pre-screening with Families...")
    prescreening_results, family_breakdown = test_pre_screening_with_families()
    
    print("\n3. Testing Feature Scorecard by Families...")
    scorecard_results, report_df, family_quality = test_feature_scorecard_by_families()
    
    print("\n4. Testing Complete Family-Diverse Pipeline...")
    complete_results = test_complete_family_diverse_pipeline()
    
    print("\n" + "=" * 80)
    print("Family-diverse feature generation testing completed successfully!")
    print("Key features demonstrated:")
    print("✅ 10 different feature families with diverse characteristics")
    print("✅ Family-aware pre-screening to maintain diversity")
    print("✅ Top K per bucket selection to avoid correlation herding")
    print("✅ Feature quality assessment by family")
    print("✅ Complete pipeline integration with family diversity")
    print("✅ Matrix operations optimization for large feature sets")
    print("=" * 80)

if __name__ == "__main__":
    main()