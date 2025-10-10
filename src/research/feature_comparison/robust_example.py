"""
Robust Feature Comparison Example

This script demonstrates the enhanced feature comparison framework with:
- Robust scaling methods
- Spearman rank correlation between methods
- Bootstrap resampling for feature importance variance
- Temporal stability analysis
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from feature_comparison.run_comparison import FeatureComparisonRunner
from feature_comparison.robust_scaling import RobustFeatureScaler, MultiMethodScaler

def create_realistic_crypto_data(n_samples: int = 3000) -> pd.DataFrame:
    """
    Create realistic cryptocurrency data with trends, volatility clusters, and noise.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with realistic OHLCV data
    """
    np.random.seed(42)
    
    # Generate realistic price data with multiple regimes
    n_regimes = 3
    regime_length = n_samples // n_regimes
    
    prices = []
    volumes = []
    
    for regime in range(n_regimes):
        if regime == 0:
            # Bull market regime
            trend = 0.002
            volatility = 0.02
            volume_trend = 0.001
        elif regime == 1:
            # Sideways market regime
            trend = 0.0001
            volatility = 0.015
            volume_trend = -0.0005
        else:
            # Bear market regime
            trend = -0.001
            volatility = 0.025
            volume_trend = 0.002
        
        # Generate regime data
        regime_returns = np.random.normal(trend, volatility, regime_length)
        regime_prices = 50000 * np.exp(np.cumsum(regime_returns))
        
        # Add volume with trend
        regime_volumes = np.random.lognormal(12 + volume_trend * np.arange(regime_length), 1.2)
        
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
    jump_indices = np.random.choice(n_samples, size=n_samples//50, replace=False)
    for idx in jump_indices:
        jump_factor = np.random.choice([0.95, 1.05])  # 5% jump up or down
        data.loc[idx:, ['open', 'high', 'low', 'close']] *= jump_factor
    
    return data

def test_robust_scaling():
    """Test different robust scaling methods."""
    print("Testing Robust Scaling Methods...")
    print("=" * 50)
    
    # Create sample data with outliers
    data = create_realistic_crypto_data(1000)
    
    # Add some outliers
    outlier_indices = np.random.choice(len(data), size=50, replace=False)
    data.loc[outlier_indices, 'close'] *= np.random.choice([0.5, 2.0], size=50)
    
    # Test different scaling methods
    methods = ['standard', 'robust', 'minmax', 'quantile', 'power']
    
    print(f"Original data shape: {data.shape}")
    print(f"Original data stats:")
    print(f"  Mean: {data['close'].mean():.2f}")
    print(f"  Std: {data['close'].std():.2f}")
    print(f"  Min: {data['close'].min():.2f}")
    print(f"  Max: {data['close'].max():.2f}")
    print()
    
    for method in methods:
        try:
            scaler = RobustFeatureScaler(method=method)
            scaled_data = scaler.fit_transform(data[['close']])
            
            print(f"{method.upper()} Scaling:")
            print(f"  Mean: {scaled_data['close'].mean():.4f}")
            print(f"  Std: {scaled_data['close'].std():.4f}")
            print(f"  Min: {scaled_data['close'].min():.4f}")
            print(f"  Max: {scaled_data['close'].max():.4f}")
            
            # Validate scaling
            validation = scaler.validate_scaling(data[['close']], scaled_data)
            print(f"  Has NaN: {validation['scaling_quality']['has_nan']}")
            print(f"  Has Inf: {validation['scaling_quality']['has_infinite']}")
            print()
            
        except Exception as e:
            print(f"{method.upper()} Scaling failed: {e}")
            print()

def run_robust_comparison():
    """Run robust feature comparison analysis."""
    print("Running Robust Feature Comparison Analysis...")
    print("=" * 60)
    
    # Create realistic data
    data = create_realistic_crypto_data(2000)
    print(f"Created realistic data with shape: {data.shape}")
    
    # Initialize runner with robust scaling
    runner = FeatureComparisonRunner(
        data=data, 
        task_type='regression',
        scaling_method='robust'
    )
    
    # Run analysis
    results = runner.run_complete_analysis()
    
    # Print enhanced summary
    runner.print_summary(results)
    
    return results

def analyze_scaling_impact():
    """Analyze the impact of different scaling methods on feature importance."""
    print("Analyzing Scaling Method Impact...")
    print("=" * 50)
    
    # Create sample data
    data = create_realistic_crypto_data(1500)
    
    # Test different scaling methods
    scaling_methods = ['standard', 'robust', 'minmax', 'quantile']
    scaling_results = {}
    
    for method in scaling_methods:
        print(f"\nTesting {method.upper()} scaling...")
        
        try:
            # Initialize runner with specific scaling method
            runner = FeatureComparisonRunner(
                data=data,
                task_type='regression',
                scaling_method=method
            )
            
            # Run analysis
            results = runner.run_complete_analysis()
            scaling_results[method] = results
            
            # Extract key metrics
            analysis_results = results['analysis_results']
            if 'initial' in analysis_results:
                initial_analysis = analysis_results['initial']
                
                # Get performance metrics
                if 'lgbm_shap' in initial_analysis and 'performance' in initial_analysis['lgbm_shap']:
                    r2 = initial_analysis['lgbm_shap']['performance'].get('r2', 0)
                    print(f"  LGBM R² Score: {r2:.4f}")
                
                # Get rank correlation
                if 'rank_correlations' in initial_analysis:
                    mean_corr = initial_analysis['rank_correlations'].get('mean_correlation', 0)
                    print(f"  Mean Rank Correlation: {mean_corr:.4f}")
                
                # Get bootstrap stability
                if 'bootstrap_analysis' in initial_analysis and 'method_results' in initial_analysis['bootstrap_analysis']:
                    bootstrap = initial_analysis['bootstrap_analysis']['method_results']
                    if 'lgbm' in bootstrap:
                        mean_cv = bootstrap['lgbm'].get('cv_importance', pd.Series()).mean()
                        print(f"  LGBM Mean CV: {mean_cv:.4f}")
                
        except Exception as e:
            print(f"  Error with {method}: {e}")
    
    return scaling_results

def main():
    """Main function to run all robust analysis examples."""
    print("Robust Feature Comparison Framework - Enhanced Analysis")
    print("=" * 70)
    print("Features:")
    print("- Robust scaling methods (Standard, Robust, MinMax, Quantile, Power)")
    print("- Spearman rank correlation between methods")
    print("- Bootstrap resampling for feature importance variance")
    print("- Temporal stability analysis")
    print("- Comprehensive evaluation metrics")
    print("=" * 70)
    
    # Test robust scaling
    print("\n1. Testing Robust Scaling Methods...")
    test_robust_scaling()
    
    # Run robust comparison
    print("\n2. Running Robust Feature Comparison...")
    robust_results = run_robust_comparison()
    
    # Analyze scaling impact
    print("\n3. Analyzing Scaling Method Impact...")
    scaling_results = analyze_scaling_impact()
    
    print("\n" + "=" * 70)
    print("Robust analysis completed successfully!")
    print("Check the reports/ directory for detailed results and visualizations.")
    print("=" * 70)

if __name__ == "__main__":
    main()