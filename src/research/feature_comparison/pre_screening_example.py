"""
Pre-screening Pipeline Example

This script demonstrates the complete pre-screening pipeline with compute-aware
optimization and detailed feature scorecards.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from feature_comparison.pre_screening_pipeline import PreScreeningPipeline
from feature_comparison.feature_scorecard import FeatureScorecard
from feature_comparison.compute_aware_optimizer import ComputeAwareOptimizer
from feature_comparison.standardized_features import StandardizedFeatureGenerator

# Suppress warnings
warnings.filterwarnings('ignore')

def create_market_data_for_prescreening(n_samples: int = 5000) -> pd.DataFrame:
    """
    Create market data for pre-screening pipeline testing.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)
    
    # Generate realistic price data with multiple regimes
    n_regimes = 5
    regime_length = n_samples // n_regimes
    
    prices = []
    volumes = []
    
    for regime in range(n_regimes):
        if regime == 0:
            # Bull market regime
            trend = 0.002
            volatility = 0.015
            volume_trend = 0.001
        elif regime == 1:
            # Sideways market regime
            trend = 0.0001
            volatility = 0.008
            volume_trend = -0.0005
        elif regime == 2:
            # High volatility regime
            trend = 0.0005
            volatility = 0.035
            volume_trend = 0.003
        elif regime == 3:
            # Bear market regime
            trend = -0.0015
            volatility = 0.025
            volume_trend = 0.002
        else:
            # Recovery regime
            trend = 0.001
            volatility = 0.012
            volume_trend = 0.0005
        
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

def test_pre_screening_pipeline():
    """Test the pre-screening pipeline."""
    print("Testing Pre-screening Pipeline...")
    print("=" * 60)
    
    # Create market data
    data = create_market_data_for_prescreening(3000)
    print(f"Created market data with shape: {data.shape}")
    
    # Generate standardized features
    feature_generator = StandardizedFeatureGenerator(data, enable_matrix_ops=True)
    versions = feature_generator.generate_standardized_features()
    
    # Use initial version for pre-screening
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
    
    print(f"Aligned data: {X_aligned.shape[0]} samples, {X_aligned.shape[1]} features")
    
    # Initialize pre-screening pipeline
    pipeline = PreScreeningPipeline(
        top_k_per_bucket=15,
        mi_percentile=0.6,
        correlation_threshold=0.95,
        vif_threshold=10.0,
        n_total_features=50,
        enable_matrix_ops=True
    )
    
    # Run pre-screening
    print("\nRunning pre-screening pipeline...")
    results = pipeline.run_pre_screening(X_aligned, y_aligned)
    
    # Print results
    print(f"\nPre-screening Results:")
    print(f"  Phase A - Selected features: {len(results['phase_a']['selected_features'])}")
    print(f"  Phase B - Pruned features: {len(results['phase_b']['pruned_features'])}")
    print(f"  Phase C - Final features: {len(results['phase_c']['selected_features'])}")
    
    # Show family breakdown
    print(f"\nFamily Breakdown:")
    for family, family_result in results['phase_a']['family_results'].items():
        print(f"  {family}: {family_result['total_features']} -> {len(family_result['selected_features'])}")
    
    return results

def test_feature_scorecard():
    """Test the feature scorecard."""
    print("\nTesting Feature Scorecard...")
    print("=" * 60)
    
    # Create market data
    data = create_market_data_for_prescreening(2000)
    
    # Generate features
    feature_generator = StandardizedFeatureGenerator(data)
    versions = feature_generator.generate_standardized_features()
    
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
    
    print(f"Computing scorecard for {X_aligned.shape[1]} features...")
    
    # Initialize feature scorecard
    scorecard = FeatureScorecard(
        stability_threshold=0.4,
        regime_delta_threshold=10,
        correlation_threshold=0.9,
        psi_threshold=0.2,
        n_bootstrap=10
    )
    
    # Compute scorecard
    scorecard_results = scorecard.compute_feature_scorecard(X_aligned, y_aligned)
    
    # Generate report
    report_df = scorecard.generate_scorecard_report(scorecard_results)
    
    print(f"\nFeature Scorecard Results:")
    print(f"  Total features analyzed: {len(report_df)}")
    print(f"  High quality features (FQS >= 0.7): {len(report_df[report_df['fqs'] >= 0.7])}")
    print(f"  Medium quality features (0.4 <= FQS < 0.7): {len(report_df[(report_df['fqs'] >= 0.4) & (report_df['fqs'] < 0.7)])}")
    print(f"  Low quality features (FQS < 0.4): {len(report_df[report_df['fqs'] < 0.4])}")
    
    # Show top 10 features
    print(f"\nTop 10 Features by FQS:")
    top_10 = report_df.head(10)
    for idx, row in top_10.iterrows():
        print(f"  {row['feature']:25s} | FQS: {row['fqs']:.3f} | Family: {row['family']:10s} | Flags: {row['flags']}")
    
    return scorecard_results, report_df

def test_compute_aware_optimizer():
    """Test the compute-aware optimizer."""
    print("\nTesting Compute-Aware Optimizer...")
    print("=" * 60)
    
    # Create market data
    data = create_market_data_for_prescreening(1500)
    
    # Generate features
    feature_generator = StandardizedFeatureGenerator(data)
    versions = feature_generator.generate_standardized_features()
    
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
    
    print(f"Optimizing computation for {X_aligned.shape[1]} features...")
    
    # Initialize compute-aware optimizer
    optimizer = ComputeAwareOptimizer(
        max_memory_gb=4.0,
        max_compute_time_ms=500.0,
        enable_caching=True,
        enable_parallel=True
    )
    
    # Run optimization
    optimization_results = optimizer.optimize_feature_computation(X_aligned, y_aligned)
    
    # Get compute summary
    compute_summary = optimizer.get_compute_summary()
    
    print(f"\nCompute-Aware Optimization Results:")
    print(f"  Total compute time: {compute_summary['total_compute_time_ms']:.2f} ms")
    print(f"  Total memory usage: {compute_summary['total_memory_usage_gb']:.2f} GB")
    print(f"  Final features selected: {len(optimization_results['optimization_results']['final_features'])}")
    
    # Show family profiles
    print(f"\nFamily Compute Profiles:")
    for family, profile in compute_summary['family_profiles'].items():
        print(f"  {family:15s} | Time: {profile['compute_time_ms']:6.1f} ms | Memory: {profile['memory_usage_gb']:5.2f} GB | Features: {profile['n_features']:3d} -> {profile['n_selected']:2d}")
    
    return optimization_results

def test_complete_pipeline():
    """Test the complete pre-screening pipeline."""
    print("\nTesting Complete Pre-screening Pipeline...")
    print("=" * 70)
    
    # Create market data
    data = create_market_data_for_prescreening(4000)
    
    # Generate features
    feature_generator = StandardizedFeatureGenerator(data, enable_matrix_ops=True)
    versions = feature_generator.generate_standardized_features()
    
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
    
    print(f"Running complete pipeline on {X_aligned.shape[1]} features...")
    
    # Step 1: Pre-screening pipeline
    print("\nStep 1: Pre-screening Pipeline...")
    pipeline = PreScreeningPipeline(
        top_k_per_bucket=20,
        mi_percentile=0.6,
        correlation_threshold=0.95,
        vif_threshold=10.0,
        n_total_features=100,
        enable_matrix_ops=True
    )
    
    prescreening_results = pipeline.run_pre_screening(X_aligned, y_aligned)
    selected_features = prescreening_results['final_features']
    
    print(f"  Selected {len(selected_features)} features from pre-screening")
    
    # Step 2: Feature scorecard
    print("\nStep 2: Feature Scorecard...")
    scorecard = FeatureScorecard()
    
    # Use selected features for scorecard
    X_selected = X_aligned[selected_features]
    scorecard_results = scorecard.compute_feature_scorecard(X_selected, y_aligned)
    
    # Generate report
    report_df = scorecard.generate_scorecard_report(scorecard_results)
    
    print(f"  Analyzed {len(report_df)} features with scorecard")
    print(f"  High quality features: {len(report_df[report_df['fqs'] >= 0.7])}")
    
    # Step 3: Compute-aware optimization
    print("\nStep 3: Compute-Aware Optimization...")
    optimizer = ComputeAwareOptimizer(
        max_memory_gb=6.0,
        max_compute_time_ms=1000.0,
        enable_caching=True,
        enable_parallel=True
    )
    
    optimization_results = optimizer.optimize_feature_computation(X_selected, y_aligned)
    final_features = optimization_results['optimization_results']['final_features']
    
    print(f"  Final optimized features: {len(final_features)}")
    
    # Step 4: Generate final report
    print("\nStep 4: Final Report...")
    final_report = report_df[report_df['feature'].isin(final_features)].copy()
    final_report = final_report.sort_values('fqs', ascending=False)
    
    print(f"\nFinal Feature Selection Results:")
    print(f"  Original features: {X_aligned.shape[1]}")
    print(f"  Pre-screened features: {len(selected_features)}")
    print(f"  Final optimized features: {len(final_features)}")
    print(f"  Reduction: {((X_aligned.shape[1] - len(final_features)) / X_aligned.shape[1] * 100):.1f}%")
    
    print(f"\nTop 15 Final Features:")
    for idx, row in final_report.head(15).iterrows():
        print(f"  {row['feature']:25s} | FQS: {row['fqs']:.3f} | Family: {row['family']:10s} | Flags: {row['flags']}")
    
    return {
        'prescreening_results': prescreening_results,
        'scorecard_results': scorecard_results,
        'optimization_results': optimization_results,
        'final_report': final_report
    }

def main():
    """Main function to run all pre-screening tests."""
    print("Pre-screening Pipeline - Complete Test Suite")
    print("=" * 80)
    print("Features:")
    print("- Phase A: Fast univariate signal screen")
    print("- Phase B: Redundancy pruning")
    print("- Phase C: Light model sanity check")
    print("- Feature scorecard with detailed metrics")
    print("- Compute-aware optimization")
    print("- Complete pipeline integration")
    print("=" * 80)
    
    # Test individual components
    print("\n1. Testing Pre-screening Pipeline...")
    prescreening_results = test_pre_screening_pipeline()
    
    print("\n2. Testing Feature Scorecard...")
    scorecard_results, scorecard_report = test_feature_scorecard()
    
    print("\n3. Testing Compute-Aware Optimizer...")
    optimization_results = test_compute_aware_optimizer()
    
    print("\n4. Testing Complete Pipeline...")
    complete_results = test_complete_pipeline()
    
    print("\n" + "=" * 80)
    print("Pre-screening pipeline testing completed successfully!")
    print("Key features demonstrated:")
    print("✅ Fast univariate signal screen with MI and distance correlation")
    print("✅ Redundancy pruning with correlation and VIF thresholds")
    print("✅ Light model sanity check with shallow models")
    print("✅ Feature scorecard with predictive, stability, uniqueness, and risk metrics")
    print("✅ Compute-aware optimization with memory and time profiling")
    print("✅ Complete pipeline integration with feature quality scoring")
    print("✅ Matrix operations and parallel processing optimization")
    print("=" * 80)

if __name__ == "__main__":
    main()