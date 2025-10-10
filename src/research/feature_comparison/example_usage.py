"""
Example Usage of Feature Comparison Framework

This script demonstrates how to use the feature comparison framework
to compare different feature engineering approaches.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from feature_comparison.run_comparison import FeatureComparisonRunner

def create_sample_crypto_data(n_samples: int = 2000) -> pd.DataFrame:
    """
    Create sample cryptocurrency data for testing.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)
    
    # Generate realistic crypto price data
    returns = np.random.normal(0.0005, 0.03, n_samples)  # 0.05% mean return, 3% volatility
    prices = 50000 * np.exp(np.cumsum(returns))  # Start at $50k
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
        'open': prices * (1 + np.random.normal(0, 0.0005, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(12, 1.5, n_samples)  # Realistic volume distribution
    })
    
    # Ensure OHLC constraints
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    return data

def run_basic_comparison():
    """Run a basic feature comparison analysis."""
    print("Running Basic Feature Comparison Analysis...")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_crypto_data(1000)
    print(f"Created sample data with shape: {data.shape}")
    
    # Initialize runner
    runner = FeatureComparisonRunner(data=data, task_type='regression')
    
    # Run analysis
    results = runner.run_complete_analysis()
    
    # Print summary
    runner.print_summary(results)
    
    return results

def run_custom_analysis():
    """Run analysis with custom parameters."""
    print("Running Custom Feature Comparison Analysis...")
    print("=" * 60)
    
    # Create larger dataset
    data = create_sample_crypto_data(2000)
    print(f"Created sample data with shape: {data.shape}")
    
    # Initialize runner with custom target
    runner = FeatureComparisonRunner(data=data, target_col='returns', task_type='regression')
    
    # Run analysis
    results = runner.run_complete_analysis()
    
    # Print summary
    runner.print_summary(results)
    
    return results

def analyze_specific_version():
    """Analyze a specific feature version in detail."""
    print("Analyzing Specific Feature Version...")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_crypto_data(1000)
    
    # Initialize feature versions
    from feature_comparison.feature_versions import FeatureVersions
    feature_versions = FeatureVersions(data, 'returns')
    
    # Create target
    target = feature_versions.create_target(method='future_returns', periods=1)
    
    # Generate versions
    versions = feature_versions.generate_all_versions()
    
    # Analyze VWAP-based features specifically
    print("Analyzing VWAP-based features...")
    X_vwap = feature_versions.get_feature_matrix('vwap_based')
    print(f"VWAP features shape: {X_vwap.shape}")
    print(f"VWAP feature names: {list(X_vwap.columns)}")
    
    # Show feature info
    version_info = feature_versions.get_version_info()
    print(f"\nVersion info: {version_info['vwap_based']}")
    
    return versions

if __name__ == "__main__":
    print("Feature Comparison Framework - Example Usage")
    print("=" * 60)
    
    # Run basic comparison
    print("\n1. Running Basic Comparison...")
    basic_results = run_basic_comparison()
    
    # Run custom analysis
    print("\n2. Running Custom Analysis...")
    custom_results = run_custom_analysis()
    
    # Analyze specific version
    print("\n3. Analyzing Specific Version...")
    specific_results = analyze_specific_version()
    
    print("\nAll examples completed successfully!")