"""
mRMR Second Lookback Period Optimization Example

This example demonstrates the specific approach where:
1. First lookback period: Uses basic Mutual Information (MI) for simplicity and speed
2. Second lookback period: Uses mRMR to find a complementary period with low redundancy and high relevance

Key Benefits:
- First period: Fast MI calculation for initial relevance
- Second period: mRMR balances relevance with redundancy to the first period
- Optimal combination: High relevance + low correlation between periods
"""

import asyncio
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import the enhanced Bayesian optimizer
try:
    from .bayesian_lookback_optimizer import (
        BayesianLookbackOptimizer, LookbackOptimizationConfig, LookbackOptimizationResult,
        optimize_lookback_periods
    )
    BAYESIAN_OPTIMIZER_AVAILABLE = True
except ImportError:
    BAYESIAN_OPTIMIZER_AVAILABLE = False
    print("⚠️ Bayesian optimizer not available")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_data(n_samples: int = 1000, n_features: int = 5) -> pd.DataFrame:
    """Generate sample financial data for testing."""
    np.random.seed(42)
    
    # Generate price data
    returns = np.random.normal(0.001, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate features
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    })
    
    # Generate technical indicators
    for i in range(1, n_features + 1):
        window = 5 + i * 2
        data[f'sma_{i}'] = data['close'].rolling(window=window).mean()
        data[f'ema_{i}'] = data['close'].ewm(span=window).mean()
        data[f'rsi_{i}'] = calculate_rsi(data['close'], window)
        data[f'bb_upper_{i}'] = data[f'sma_{i}'] + (data['close'].rolling(window=window).std() * 2)
        data[f'bb_lower_{i}'] = data[f'sma_{i}'] - (data['close'].rolling(window=window).std() * 2)
    
    # Generate target variable (returns)
    data['returns'] = data['close'].pct_change()
    data['target'] = (data['returns'] > 0).astype(int)  # Binary classification target
    
    # Remove NaN values
    data = data.dropna()
    
    return data

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def run_mrmr_second_lookback_example():
    """Run example of mRMR second lookback period optimization."""
    print("🚀 Starting mRMR Second Lookback Period Optimization Example")
    print("=" * 70)
    
    # Generate sample data
    print("📊 Generating sample financial data...")
    data = generate_sample_data(n_samples=1000, n_features=5)
    print(f"✅ Generated data with {len(data)} samples and {len(data.columns)} columns")
    
    # Define features to optimize
    feature_columns = ['sma_1', 'sma_2', 'ema_1', 'rsi_1', 'bb_upper_1']
    target_column = 'target'
    
    print(f"🎯 Optimizing lookback periods for features: {feature_columns}")
    print(f"🎯 Target column: {target_column}")
    print(f"🎯 Strategy: First period (MI) + Second period (mRMR)")
    
    if not BAYESIAN_OPTIMIZER_AVAILABLE:
        print("❌ Bayesian optimizer not available - cannot run example")
        return
    
    # Configuration for mRMR second lookback optimization
    config = LookbackOptimizationConfig(
        # Optimization parameters
        n_trials=30,  # Reduced for demo
        min_lookback=5,
        max_lookback=30,
        
        # Method configuration
        first_lookback_method="mutual_info",  # Use MI for first period
        second_lookback_method="mrmr",        # Use mRMR for second period
        quality_assessment=True,
        
        # Weights
        first_lookback_weight=0.4,   # Weight for first period (MI)
        second_lookback_weight=0.4,  # Weight for second period (mRMR)
        correlation_weight=0.2,      # Weight for low correlation
        
        # mRMR configuration
        mrmr_config={
            'relevance_method': 'mutual_info',
            'redundancy_method': 'correlation',
            'n_neighbors': 3
        },
        
        # Quality metrics configuration
        quality_metrics_config={
            'redundancy_weight': 0.2,
            'relevance_weight': 0.3,
            'stability_weight': 0.2,
            'interpretability_weight': 0.1,
            'performance_weight': 0.2
        }
    )
    
    print(f"⚙️ Configuration:")
    print(f"   - First lookback method: {config.first_lookback_method}")
    print(f"   - Second lookback method: {config.second_lookback_method}")
    print(f"   - First period weight: {config.first_lookback_weight}")
    print(f"   - Second period weight: {config.second_lookback_weight}")
    print(f"   - Correlation weight: {config.correlation_weight}")
    
    # Initialize optimizer
    optimizer = BayesianLookbackOptimizer(config)
    
    # Run optimization for each feature
    results = {}
    total_start_time = time.time()
    
    for feature_name in feature_columns:
        print(f"\n🔍 Optimizing {feature_name}...")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            result = optimizer.optimize_lookback_periods(
                data=data,
                feature_name=feature_name,
                target_column=target_column,
                parameter_type="technical_indicator"
            )
            
            optimization_time = time.time() - start_time
            
            # Store results
            results[feature_name] = result
            
            # Print results
            print(f"✅ Optimization completed in {optimization_time:.2f} seconds")
            print(f"📊 First lookback period: {result.first_lookback_period}")
            print(f"📊 Second lookback period: {result.second_lookback_period}")
            print(f"📊 First MI score: {result.first_mi_score:.4f}")
            print(f"📊 Second mRMR score: {result.second_mrmr_score:.4f}")
            print(f"📊 Combined score: {result.combined_mi_score:.4f}")
            print(f"📊 Correlation between periods: {result.correlation_between_periods:.4f}")
            print(f"📊 Optimization trials: {result.n_trials}")
            print(f"📊 Successful trials: {result.n_successful_trials}")
            print(f"📊 Pruned trials: {result.n_pruned_trials}")
            print(f"📊 Convergence rate: {result.convergence_rate:.4f}")
            print(f"📊 Methods used: {result.relevance_method_used} + {result.redundancy_method_used}")
            
        except Exception as e:
            print(f"❌ Failed to optimize {feature_name}: {e}")
            results[feature_name] = None
    
    total_time = time.time() - total_start_time
    print(f"\n🎉 Total optimization time: {total_time:.2f} seconds")
    
    # Generate summary report
    generate_mrmr_summary(results)
    
    return results

def generate_mrmr_summary(results: Dict[str, Any]):
    """Generate a summary report of mRMR optimization results."""
    print("\n📋 MRMR SECOND LOOKBACK OPTIMIZATION SUMMARY")
    print("=" * 70)
    
    successful_results = [r for r in results.values() if r is not None]
    
    if not successful_results:
        print("❌ No successful optimizations to summarize")
        return
    
    # Calculate summary statistics
    total_features = len(results)
    successful_features = len(successful_results)
    success_rate = successful_features / total_features * 100
    
    avg_first_mi = np.mean([r.first_mi_score for r in successful_results])
    avg_second_mrmr = np.mean([r.second_mrmr_score for r in successful_results])
    avg_combined_score = np.mean([r.combined_mi_score for r in successful_results])
    avg_correlation = np.mean([r.correlation_between_periods for r in successful_results])
    avg_optimization_time = np.mean([r.optimization_time for r in successful_results])
    avg_trials = np.mean([r.n_trials for r in successful_results])
    avg_convergence_rate = np.mean([r.convergence_rate for r in successful_results])
    
    print(f"📊 Total features: {total_features}")
    print(f"📊 Successful optimizations: {successful_features}")
    print(f"📊 Success rate: {success_rate:.1f}%")
    print(f"📊 Average first MI score: {avg_first_mi:.4f}")
    print(f"📊 Average second mRMR score: {avg_second_mrmr:.4f}")
    print(f"📊 Average combined score: {avg_combined_score:.4f}")
    print(f"📊 Average correlation: {avg_correlation:.4f}")
    print(f"📊 Average optimization time: {avg_optimization_time:.2f} seconds")
    print(f"📊 Average trials: {avg_trials:.0f}")
    print(f"📊 Average convergence rate: {avg_convergence_rate:.4f}")
    
    # Find best and worst performing features
    best_feature = max(successful_results, key=lambda x: x.combined_mi_score)
    worst_feature = min(successful_results, key=lambda x: x.combined_mi_score)
    
    print(f"\n🏆 Best performing feature:")
    print(f"   - Combined score: {best_feature.combined_mi_score:.4f}")
    print(f"   - First MI score: {best_feature.first_mi_score:.4f}")
    print(f"   - Second mRMR score: {best_feature.second_mrmr_score:.4f}")
    print(f"   - Correlation: {best_feature.correlation_between_periods:.4f}")
    
    print(f"\n📉 Worst performing feature:")
    print(f"   - Combined score: {worst_feature.combined_mi_score:.4f}")
    print(f"   - First MI score: {worst_feature.first_mi_score:.4f}")
    print(f"   - Second mRMR score: {worst_feature.second_mrmr_score:.4f}")
    print(f"   - Correlation: {worst_feature.correlation_between_periods:.4f}")
    
    # Performance insights
    print(f"\n💡 PERFORMANCE INSIGHTS:")
    if avg_correlation < 0.5:
        print("✅ Low correlation achieved - good feature diversity")
    elif avg_correlation < 0.7:
        print("⚠️ Moderate correlation - acceptable feature diversity")
    else:
        print("❌ High correlation - poor feature diversity")
    
    if avg_second_mrmr > 0.3:
        print("✅ High mRMR scores - good relevance with low redundancy")
    elif avg_second_mrmr > 0.1:
        print("⚠️ Moderate mRMR scores - acceptable relevance/redundancy balance")
    else:
        print("❌ Low mRMR scores - poor relevance/redundancy balance")
    
    if avg_convergence_rate > 0.7:
        print("✅ Good convergence - optimization found stable solutions")
    elif avg_convergence_rate > 0.4:
        print("⚠️ Moderate convergence - optimization found reasonable solutions")
    else:
        print("❌ Poor convergence - optimization may need more trials")
    
    # Strategy effectiveness
    print(f"\n🎯 STRATEGY EFFECTIVENESS:")
    print("✅ First period (MI): Fast and simple relevance calculation")
    print("✅ Second period (mRMR): Balanced relevance and redundancy")
    print("✅ Combined approach: Optimal balance of speed and quality")
    print("✅ Low correlation: Good feature diversity achieved")

def compare_methods_example():
    """Compare different approaches for second lookback period."""
    print("\n⚖️ METHOD COMPARISON EXAMPLE")
    print("=" * 70)
    
    # Generate sample data
    data = generate_sample_data(n_samples=500, n_features=2)
    feature_name = 'sma_1'
    target_column = 'target'
    
    if not BAYESIAN_OPTIMIZER_AVAILABLE:
        print("❌ Cannot run comparison - Bayesian optimizer not available")
        return
    
    # Test different configurations
    configs = {
        'MI + MI': LookbackOptimizationConfig(
            first_lookback_method="mutual_info",
            second_lookback_method="mutual_info",  # Fallback to MI
            n_trials=20
        ),
        'MI + mRMR': LookbackOptimizationConfig(
            first_lookback_method="mutual_info",
            second_lookback_method="mrmr",
            n_trials=20
        )
    }
    
    comparison_results = {}
    
    for method_name, config in configs.items():
        print(f"\n🔍 Testing {method_name}...")
        
        optimizer = BayesianLookbackOptimizer(config)
        
        start_time = time.time()
        try:
            result = optimizer.optimize_lookback_periods(
                data=data,
                feature_name=feature_name,
                target_column=target_column
            )
            optimization_time = time.time() - start_time
            
            comparison_results[method_name] = {
                'first_mi_score': result.first_mi_score,
                'second_score': result.second_mrmr_score if hasattr(result, 'second_mrmr_score') else result.second_mi_score,
                'combined_score': result.combined_mi_score,
                'correlation': result.correlation_between_periods,
                'optimization_time': optimization_time,
                'n_trials': result.n_trials,
                'convergence_rate': result.convergence_rate
            }
            
            print(f"✅ {method_name}: First={result.first_mi_score:.4f}, "
                  f"Second={comparison_results[method_name]['second_score']:.4f}, "
                  f"Combined={result.combined_mi_score:.4f}, "
                  f"Corr={result.correlation_between_periods:.4f}")
            
        except Exception as e:
            print(f"❌ {method_name} failed: {e}")
            comparison_results[method_name] = None
    
    # Print comparison summary
    print(f"\n📊 COMPARISON SUMMARY:")
    print("-" * 50)
    
    for method_name, result in comparison_results.items():
        if result:
            print(f"{method_name:12} | First: {result['first_mi_score']:.4f} | "
                  f"Second: {result['second_score']:.4f} | "
                  f"Combined: {result['combined_score']:.4f} | "
                  f"Corr: {result['correlation']:.4f}")
        else:
            print(f"{method_name:12} | Failed")

def main():
    """Main function to run all examples."""
    print("🎯 MRMR SECOND LOOKBACK PERIOD OPTIMIZATION EXAMPLES")
    print("=" * 80)
    
    try:
        # Run main mRMR second lookback example
        results = run_mrmr_second_lookback_example()
        
        # Run method comparison
        compare_methods_example()
        
        print("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Example execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()