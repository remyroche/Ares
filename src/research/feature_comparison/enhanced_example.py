"""
Enhanced Feature Comparison Example

This script demonstrates the enhanced feature comparison framework with:
- Returns-based calculations instead of raw prices
- Rolling averages and EWMA features
- Lagged/lead features for predictive vs reactive analysis
- Matrix operations and hardware optimizations
- Reduced bootstrap samples (10 instead of 50+)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from feature_comparison.run_comparison import FeatureComparisonRunner

def create_realistic_crypto_data_with_regimes(n_samples: int = 3000) -> pd.DataFrame:
    """
    Create realistic cryptocurrency data with multiple market regimes.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with realistic OHLCV data
    """
    np.random.seed(42)
    
    # Generate realistic price data with multiple regimes
    n_regimes = 4
    regime_length = n_samples // n_regimes
    
    prices = []
    volumes = []
    
    for regime in range(n_regimes):
        if regime == 0:
            # Bull market regime - high returns, moderate volatility
            trend = 0.003
            volatility = 0.025
            volume_trend = 0.002
        elif regime == 1:
            # Sideways market regime - low returns, low volatility
            trend = 0.0002
            volatility = 0.015
            volume_trend = -0.001
        elif regime == 2:
            # High volatility regime - moderate returns, high volatility
            trend = 0.001
            volatility = 0.04
            volume_trend = 0.003
        else:
            # Bear market regime - negative returns, high volatility
            trend = -0.002
            volatility = 0.035
            volume_trend = 0.004
        
        # Generate regime data
        regime_returns = np.random.normal(trend, volatility, regime_length)
        regime_prices = 50000 * np.exp(np.cumsum(regime_returns))
        
        # Add volume with trend and regime-specific patterns
        regime_volumes = np.random.lognormal(
            12 + volume_trend * np.arange(regime_length), 
            1.2 + 0.3 * regime  # Higher volume variance in later regimes
        )
        
        prices.extend(regime_prices)
        volumes.extend(regime_volumes)
    
    # Ensure we have exactly n_samples
    prices = prices[:n_samples]
    volumes = volumes[:n_samples]
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
        'close': prices
    })
    
    # Generate realistic OHLC from close prices
    data['open'] = data['close'].shift(1) * (1 + np.random.normal(0, 0.001, n_samples))
    data['high'] = np.maximum(data['open'], data['close']) * (1 + np.abs(np.random.normal(0, 0.005, n_samples)))
    data['low'] = np.minimum(data['open'], data['close']) * (1 - np.abs(np.random.normal(0, 0.005, n_samples)))
    data['volume'] = volumes
    
    # Ensure OHLC constraints
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Add some realistic gaps and jumps
    jump_indices = np.random.choice(n_samples, size=n_samples//100, replace=False)
    for idx in jump_indices:
        jump_factor = np.random.choice([0.95, 1.05])  # 5% jump up or down
        data.loc[idx:, ['open', 'high', 'low', 'close']] *= jump_factor
    
    return data

def test_returns_based_features():
    """Test returns-based feature generation."""
    print("Testing Returns-Based Feature Generation...")
    print("=" * 60)
    
    # Create sample data
    data = create_realistic_crypto_data_with_regimes(1000)
    print(f"Created sample data with shape: {data.shape}")
    
    # Test optimized feature versions
    from feature_comparison.optimized_feature_versions import OptimizedFeatureVersions
    
    feature_versions = OptimizedFeatureVersions(
        data, 'returns', 
        enable_matrix_ops=True, 
        enable_hardware_opt=True
    )
    
    # Generate all versions
    versions = feature_versions.generate_all_versions()
    
    print(f"\nGenerated {len(versions)} feature versions:")
    for version_name, version_df in versions.items():
        feature_cols = feature_versions.get_feature_matrix(version_name).columns
        print(f"  {version_name}: {len(feature_cols)} features")
        
        # Show some key features
        returns_features = [col for col in feature_cols if 'returns' in col]
        print(f"    Returns-based features: {len(returns_features)}")
        if returns_features:
            print(f"    Examples: {returns_features[:5]}")
        
        # Show lagged/lead features
        lagged_features = [col for col in feature_cols if 'lag' in col or 'lead' in col]
        print(f"    Lagged/Lead features: {len(lagged_features)}")
        if lagged_features:
            print(f"    Examples: {lagged_features[:5]}")
    
    return versions

def test_matrix_operations_integration():
    """Test matrix operations integration."""
    print("\nTesting Matrix Operations Integration...")
    print("=" * 60)
    
    # Create sample data
    data = create_realistic_crypto_data_with_regimes(500)
    
    # Test with and without matrix operations
    from feature_comparison.optimized_feature_versions import OptimizedFeatureVersions
    
    # With matrix operations
    print("Testing with matrix operations...")
    feature_versions_matrix = OptimizedFeatureVersions(
        data, 'returns', 
        enable_matrix_ops=True, 
        enable_hardware_opt=True
    )
    
    import time
    start_time = time.time()
    versions_matrix = feature_versions_matrix.generate_all_versions()
    matrix_time = time.time() - start_time
    
    print(f"Matrix operations time: {matrix_time:.3f} seconds")
    print(f"Matrix operations enabled: {feature_versions_matrix.enable_matrix_ops}")
    print(f"Hardware optimizations enabled: {feature_versions_matrix.enable_hardware_opt}")
    
    # Without matrix operations
    print("\nTesting without matrix operations...")
    feature_versions_standard = OptimizedFeatureVersions(
        data, 'returns', 
        enable_matrix_ops=False, 
        enable_hardware_opt=False
    )
    
    start_time = time.time()
    versions_standard = feature_versions_standard.generate_all_versions()
    standard_time = time.time() - start_time
    
    print(f"Standard operations time: {standard_time:.3f} seconds")
    print(f"Speedup: {standard_time / matrix_time:.2f}x")
    
    return versions_matrix, versions_standard

def run_enhanced_comparison():
    """Run enhanced feature comparison analysis."""
    print("\nRunning Enhanced Feature Comparison Analysis...")
    print("=" * 70)
    
    # Create realistic data
    data = create_realistic_crypto_data_with_regimes(2000)
    print(f"Created realistic data with shape: {data.shape}")
    
    # Initialize runner with optimized features
    runner = FeatureComparisonRunner(
        data=data, 
        task_type='regression',
        scaling_method='robust',
        use_optimized=True  # Use optimized feature versions
    )
    
    # Run analysis
    results = runner.run_complete_analysis()
    
    # Print enhanced summary
    runner.print_summary(results)
    
    return results

def analyze_predictive_vs_reactive():
    """Analyze predictive vs reactive feature importance."""
    print("\nAnalyzing Predictive vs Reactive Feature Importance...")
    print("=" * 70)
    
    # Create sample data
    data = create_realistic_crypto_data_with_regimes(1500)
    
    # Initialize runner
    runner = FeatureComparisonRunner(
        data=data,
        task_type='regression',
        use_optimized=True
    )
    
    # Run analysis
    results = runner.run_complete_analysis()
    
    # Analyze lagged vs lead features
    analysis_results = results['analysis_results']
    
    print("\nFeature Importance Analysis:")
    print("-" * 40)
    
    for version_name, analysis in analysis_results.items():
        print(f"\n{version_name}:")
        
        if 'combined_ranking' in analysis and not analysis['combined_ranking'].empty:
            ranking = analysis['combined_ranking']
            
            # Categorize features
            lagged_features = ranking[ranking['feature'].str.contains('lag', na=False)]
            lead_features = ranking[ranking['feature'].str.contains('lead', na=False)]
            returns_features = ranking[ranking['feature'].str.contains('returns', na=False)]
            vwap_features = ranking[ranking['feature'].str.contains('vwap', na=False)]
            
            print(f"  Total features: {len(ranking)}")
            print(f"  Lagged features: {len(lagged_features)}")
            print(f"  Lead features: {len(lead_features)}")
            print(f"  Returns features: {len(returns_features)}")
            print(f"  VWAP features: {len(vwap_features)}")
            
            # Top features by category
            if not lagged_features.empty:
                print(f"  Top lagged feature: {lagged_features.iloc[0]['feature']} (rank: {lagged_features.iloc[0]['avg_rank']:.2f})")
            
            if not lead_features.empty:
                print(f"  Top lead feature: {lead_features.iloc[0]['feature']} (rank: {lead_features.iloc[0]['avg_rank']:.2f})")
            
            if not returns_features.empty:
                print(f"  Top returns feature: {returns_features.iloc[0]['feature']} (rank: {returns_features.iloc[0]['avg_rank']:.2f})")
    
    return results

def main():
    """Main function to run all enhanced analysis examples."""
    print("Enhanced Feature Comparison Framework - Advanced Analysis")
    print("=" * 80)
    print("Features:")
    print("- Returns-based calculations (no raw prices)")
    print("- Rolling averages and EWMA features")
    print("- Lagged/lead features for predictive analysis")
    print("- Matrix operations and hardware optimizations")
    print("- Reduced bootstrap samples (10 instead of 50+)")
    print("- Robust evaluation with temporal stability")
    print("=" * 80)
    
    # Test returns-based features
    print("\n1. Testing Returns-Based Feature Generation...")
    versions = test_returns_based_features()
    
    # Test matrix operations integration
    print("\n2. Testing Matrix Operations Integration...")
    versions_matrix, versions_standard = test_matrix_operations_integration()
    
    # Run enhanced comparison
    print("\n3. Running Enhanced Feature Comparison...")
    enhanced_results = run_enhanced_comparison()
    
    # Analyze predictive vs reactive
    print("\n4. Analyzing Predictive vs Reactive Features...")
    predictive_results = analyze_predictive_vs_reactive()
    
    print("\n" + "=" * 80)
    print("Enhanced analysis completed successfully!")
    print("Key improvements:")
    print("✅ All calculations use returns instead of raw prices")
    print("✅ Rolling averages and EWMA features implemented")
    print("✅ Lagged/lead features for predictive analysis")
    print("✅ Matrix operations and hardware optimizations")
    print("✅ Bootstrap samples reduced to 10")
    print("✅ Robust evaluation with temporal stability")
    print("=" * 80)

if __name__ == "__main__":
    main()