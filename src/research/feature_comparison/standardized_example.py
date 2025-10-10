"""
Standardized Feature Comparison Example

This script demonstrates the enhanced feature comparison framework with:
- Standardized feature definitions and naming conventions
- Feature consolidation and redundancy removal
- Multicollinearity screening
- Feature validation
- Returns-based calculations with explicit windows
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from feature_comparison.enhanced_comparison_runner import EnhancedFeatureComparisonRunner
from feature_comparison.standardized_features import StandardizedFeatureGenerator
from feature_comparison.feature_consolidation import FeatureConsolidator, FeatureValidator

def create_realistic_market_data(n_samples: int = 3000) -> pd.DataFrame:
    """
    Create realistic market data with multiple regimes and patterns.
    
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
            # Bull market regime
            trend = 0.003
            volatility = 0.025
            volume_trend = 0.002
        elif regime == 1:
            # Sideways market regime
            trend = 0.0002
            volatility = 0.015
            volume_trend = -0.001
        elif regime == 2:
            # High volatility regime
            trend = 0.001
            volatility = 0.04
            volume_trend = 0.003
        else:
            # Bear market regime
            trend = -0.002
            volatility = 0.035
            volume_trend = 0.004
        
        # Generate regime data
        regime_returns = np.random.normal(trend, volatility, regime_length)
        regime_prices = 50000 * np.exp(np.cumsum(regime_returns))
        
        # Add volume with trend
        regime_volumes = np.random.lognormal(
            12 + volume_trend * np.arange(regime_length), 
            1.2 + 0.3 * regime
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
    
    return data

def test_standardized_feature_generation():
    """Test standardized feature generation."""
    print("Testing Standardized Feature Generation...")
    print("=" * 60)
    
    # Create sample data
    data = create_realistic_market_data(1000)
    print(f"Created sample data with shape: {data.shape}")
    
    # Test standardized feature generator
    feature_generator = StandardizedFeatureGenerator(data, enable_matrix_ops=True)
    versions = feature_generator.generate_standardized_features()
    
    print(f"\nGenerated {len(versions)} standardized feature versions:")
    
    for version_name, version_df in versions.items():
        feature_cols = [col for col in version_df.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
        print(f"\n{version_name}:")
        print(f"  Total features: {len(feature_cols)}")
        
        # Categorize features by type
        feature_categories = {
            'returns': [col for col in feature_cols if col.startswith('ret_')],
            'vwap': [col for col in feature_cols if 'vwap' in col],
            'volatility': [col for col in feature_cols if 'vol_' in col],
            'volume': [col for col in feature_cols if 'vol_' in col and 'ret' not in col],
            'regime': [col for col in feature_cols if 'regime' in col],
            'beta': [col for col in feature_cols if 'beta' in col],
            'drawdown': [col for col in feature_cols if 'dd_' in col],
            'entropy': [col for col in feature_cols if 'entropy' in col],
            'interaction': [col for col in feature_cols if 'interact' in col]
        }
        
        for category, features in feature_categories.items():
            if features:
                print(f"  {category}: {len(features)} features")
                print(f"    Examples: {features[:3]}")
    
    # Show feature definitions
    definitions = feature_generator.get_feature_definitions()
    print(f"\nFeature Definitions (showing first 10):")
    for i, (feature, definition) in enumerate(list(definitions.items())[:10]):
        print(f"  {feature}: {definition}")
    
    return versions

def test_feature_consolidation():
    """Test feature consolidation and redundancy removal."""
    print("\nTesting Feature Consolidation...")
    print("=" * 60)
    
    # Create sample data
    data = create_realistic_market_data(1000)
    
    # Generate features
    feature_generator = StandardizedFeatureGenerator(data)
    versions = feature_generator.generate_standardized_features()
    
    # Test consolidation
    consolidator = FeatureConsolidator()
    
    print("Before consolidation:")
    for version_name, version_df in versions.items():
        feature_cols = [col for col in version_df.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
        print(f"  {version_name}: {len(feature_cols)} features")
    
    # Consolidate features
    consolidated_versions = {}
    for version_name, version_df in versions.items():
        consolidated_df = consolidator.consolidate_features(version_df, version_name)
        consolidated_df = consolidator.remove_multicollinearity(consolidated_df, version_name)
        consolidated_df = consolidator.winsorize_features(consolidated_df)
        consolidated_versions[version_name] = consolidated_df
    
    print("\nAfter consolidation:")
    for version_name, version_df in consolidated_versions.items():
        feature_cols = [col for col in version_df.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
        print(f"  {version_name}: {len(feature_cols)} features")
    
    # Show consolidation summary
    summary = consolidator.get_consolidation_summary()
    print(f"\nConsolidation Summary:")
    print(f"  Total features removed: {summary['total_removed']}")
    for version, removed in summary['removed_features'].items():
        if removed:
            print(f"  {version}: {len(removed)} features removed")
    
    return consolidated_versions

def test_feature_validation():
    """Test feature validation."""
    print("\nTesting Feature Validation...")
    print("=" * 60)
    
    # Create sample data
    data = create_realistic_market_data(1000)
    
    # Generate features
    feature_generator = StandardizedFeatureGenerator(data)
    versions = feature_generator.generate_standardized_features()
    
    # Test validation
    validator = FeatureValidator()
    
    for version_name, version_df in versions.items():
        validation = validator.validate_features(version_df, version_name)
        
        print(f"\n{version_name} validation:")
        print(f"  Features: {validation['data_quality']['n_features']}")
        print(f"  Samples: {validation['data_quality']['n_samples']}")
        print(f"  Missing data: {validation['data_quality']['missing_data_pct']:.2f}%")
        print(f"  Warnings: {len(validation['warnings'])}")
        print(f"  Errors: {len(validation['errors'])}")
        
        if validation['warnings']:
            print(f"    Warnings: {validation['warnings'][:3]}")
        if validation['errors']:
            print(f"    Errors: {validation['errors'][:3]}")
    
    # Show validation summary
    summary = validator.get_validation_summary()
    print(f"\nOverall Validation Summary:")
    print(f"  Total versions: {summary['total_versions']}")
    print(f"  Total warnings: {summary['overall_quality']['total_warnings']}")
    print(f"  Total errors: {summary['overall_quality']['total_errors']}")
    
    return summary

def run_enhanced_comparison():
    """Run enhanced feature comparison analysis."""
    print("\nRunning Enhanced Feature Comparison Analysis...")
    print("=" * 70)
    
    # Create realistic data
    data = create_realistic_market_data(2000)
    print(f"Created realistic data with shape: {data.shape}")
    
    # Initialize enhanced runner
    runner = EnhancedFeatureComparisonRunner(
        data=data,
        task_type='regression',
        scaling_method='robust',
        enable_consolidation=True,
        enable_validation=True
    )
    
    # Run analysis
    results = runner.run_enhanced_analysis()
    
    # Print enhanced summary
    runner.print_enhanced_summary(results)
    
    return results

def demonstrate_feature_naming_conventions():
    """Demonstrate the standardized naming conventions."""
    print("\nDemonstrating Standardized Naming Conventions...")
    print("=" * 70)
    
    # Create sample data
    data = create_realistic_market_data(500)
    
    # Generate features
    feature_generator = StandardizedFeatureGenerator(data)
    versions = feature_generator.generate_standardized_features()
    
    # Show examples of each naming convention
    conventions = {
        'Returns': ['ret_t1', 'ret_abs_t1', 'ret_sq_t1'],
        'Rolling Windows': ['ret_ma_w5', 'ret_ma_w10', 'ret_ma_w20', 'ret_ma_w50'],
        'EWMA': ['ret_ewm_w5', 'ret_ewm_w10', 'ret_ewm_w20'],
        'Lagged Features': ['ret_lag1', 'ret_lag2', 'ret_lag3', 'ret_lag5'],
        'Momentum': ['ret_mom_k1', 'ret_mom_k2', 'ret_mom_k3', 'ret_mom_k5'],
        'VWAP Features': ['vwap_w20', 'vwap_ret_w20', 'vwap_basis_w20', 'rel_vwap_dev_w20'],
        'Volatility Features': ['vol_w10', 'vol_w20', 'vol_w50'],
        'Volatility Normalized': ['ret_ma_w5_normvol20', 'ret_std_w10_normvol20'],
        'Regime Features': ['regime_highvol', 'ret_highvol_interact'],
        'Beta Features': ['beta_market_w20', 'ret_normbeta_w20'],
        'Volume Features': ['vol_ret_t1', 'vol_ma_w20', 'vol_adv_w20'],
        'Drawdown Features': ['dd_current', 'dd_max_w20'],
        'Entropy Features': ['ret_perm_entropy_w20'],
        'Interaction Features': ['ret_vol_interact', 'vwap_vol_interact']
    }
    
    # Find examples in the generated features
    all_features = set()
    for version_df in versions.values():
        all_features.update(version_df.columns)
    
    print("Naming Convention Examples:")
    print("-" * 40)
    
    for category, examples in conventions.items():
        found_examples = [ex for ex in examples if ex in all_features]
        if found_examples:
            print(f"\n{category}:")
            for example in found_examples:
                print(f"  ✓ {example}")
        else:
            print(f"\n{category}: No examples found")
    
    return all_features

def main():
    """Main function to run all standardized feature examples."""
    print("Standardized Feature Comparison Framework - Complete Example")
    print("=" * 80)
    print("Features:")
    print("- Standardized naming conventions")
    print("- Returns-based calculations with explicit windows")
    print("- Feature consolidation and redundancy removal")
    print("- Multicollinearity screening")
    print("- Feature validation and quality checks")
    print("- Robust evaluation with 10 bootstrap samples")
    print("=" * 80)
    
    # Test standardized feature generation
    print("\n1. Testing Standardized Feature Generation...")
    versions = test_standardized_feature_generation()
    
    # Test feature consolidation
    print("\n2. Testing Feature Consolidation...")
    consolidated_versions = test_feature_consolidation()
    
    # Test feature validation
    print("\n3. Testing Feature Validation...")
    validation_summary = test_feature_validation()
    
    # Demonstrate naming conventions
    print("\n4. Demonstrating Naming Conventions...")
    all_features = demonstrate_feature_naming_conventions()
    
    # Run enhanced comparison
    print("\n5. Running Enhanced Feature Comparison...")
    enhanced_results = run_enhanced_comparison()
    
    print("\n" + "=" * 80)
    print("Standardized feature analysis completed successfully!")
    print("Key improvements:")
    print("✅ Standardized naming conventions (ret_t(h), vwap_wW, vol_t(W))")
    print("✅ Explicit window specifications (_wW, _ewmA)")
    print("✅ Returns-based calculations (no raw prices)")
    print("✅ Feature consolidation and redundancy removal")
    print("✅ Multicollinearity screening (|ρ|>0.95, VIF>10)")
    print("✅ Feature validation and quality checks")
    print("✅ Winsorization for outlier handling")
    print("✅ Bootstrap samples reduced to 10")
    print("=" * 80)

if __name__ == "__main__":
    main()