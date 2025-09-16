"""
Bayesian Lookback Period Optimization Example

This example demonstrates how to use the Bayesian optimization system to find
optimal lookback periods for feature parameters based on:
1. Mutual Information (MI) maximization for the first lookback period
2. Low correlation & high mutual importance for the second lookback period

Key Features Demonstrated:
- TPE (Tree-structured Parzen Estimator) sampling
- Intelligent pruning strategies
- Multi-objective optimization
- Real-time monitoring and analytics
- Performance comparison with traditional methods
"""

import asyncio
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import the Bayesian optimizer
try:
    from .bayesian_lookback_optimizer import (
        BayesianLookbackOptimizer, LookbackOptimizationConfig, LookbackOptimizationResult,
        optimize_lookback_periods
    )
    BAYESIAN_OPTIMIZER_AVAILABLE = True
except ImportError:
    BAYESIAN_OPTIMIZER_AVAILABLE = False
    print("⚠️ Bayesian optimizer not available")

# Import the main optimization component
try:
    from .feature_lookback_optimization import FeatureLookbackOptimizationComponent
    MAIN_COMPONENT_AVAILABLE = True
except ImportError:
    MAIN_COMPONENT_AVAILABLE = False
    print("⚠️ Main optimization component not available")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_data(n_samples: int = 1000, n_features: int = 10) -> pd.DataFrame:
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

def run_bayesian_optimization_example():
    """Run a comprehensive example of Bayesian lookback optimization."""
    print("🚀 Starting Bayesian Lookback Period Optimization Example")
    print("=" * 60)
    
    # Generate sample data
    print("📊 Generating sample financial data...")
    data = generate_sample_data(n_samples=1000, n_features=5)
    print(f"✅ Generated data with {len(data)} samples and {len(data.columns)} columns")
    
    # Define features to optimize
    feature_columns = ['sma_1', 'sma_2', 'ema_1', 'rsi_1', 'bb_upper_1']
    target_column = 'target'
    
    print(f"🎯 Optimizing lookback periods for features: {feature_columns}")
    print(f"🎯 Target column: {target_column}")
    
    if not BAYESIAN_OPTIMIZER_AVAILABLE:
        print("❌ Bayesian optimizer not available - cannot run example")
        return
    
    # Configuration for optimization
    config = LookbackOptimizationConfig(
        n_trials=30,  # Reduced for demo
        min_lookback=5,
        max_lookback=30,
        max_correlation_threshold=0.7,
        min_mutual_info_threshold=0.1,
        enable_pruning=True,
        enable_parallel=True,
        save_intermediate_results=True
    )
    
    print(f"⚙️ Configuration: {config.n_trials} trials, lookback range {config.min_lookback}-{config.max_lookback}")
    
    # Initialize optimizer
    optimizer = BayesianLookbackOptimizer(config)
    
    # Run optimization for each feature
    results = {}
    total_start_time = time.time()
    
    for feature_name in feature_columns:
        print(f"\n🔍 Optimizing {feature_name}...")
        print("-" * 40)
        
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
            print(f"📊 Second MI score: {result.second_mi_score:.4f}")
            print(f"📊 Combined MI score: {result.combined_mi_score:.4f}")
            print(f"📊 Correlation between periods: {result.correlation_between_periods:.4f}")
            print(f"📊 Optimization trials: {result.n_trials}")
            print(f"📊 Successful trials: {result.n_successful_trials}")
            print(f"📊 Pruned trials: {result.n_pruned_trials}")
            print(f"📊 Convergence rate: {result.convergence_rate:.4f}")
            
        except Exception as e:
            print(f"❌ Failed to optimize {feature_name}: {e}")
            results[feature_name] = None
    
    total_time = time.time() - total_start_time
    print(f"\n🎉 Total optimization time: {total_time:.2f} seconds")
    
    # Generate summary report
    generate_optimization_summary(results)
    
    return results

def run_component_integration_example():
    """Run example using the main optimization component."""
    print("\n🔧 Running Component Integration Example")
    print("=" * 60)
    
    if not MAIN_COMPONENT_AVAILABLE:
        print("❌ Main optimization component not available")
        return
    
    # Generate sample data
    data = generate_sample_data(n_samples=500, n_features=3)
    feature_columns = ['sma_1', 'ema_1', 'rsi_1']
    
    # Initialize component
    from .feature_lookback_optimization import FeatureLookbackOptimizationConfig
    
    config = FeatureLookbackOptimizationConfig(
        symbol="BTCUSDT",
        exchange="binance",
        timeframe="1h",
        lookback_range=(5, 30),
        optimization_method="bayesian"
    )
    
    component = FeatureLookbackOptimizationComponent(config)
    
    # Run Bayesian optimization
    print("🔍 Running Bayesian optimization through main component...")
    results = component.optimize_lookback_periods_bayesian(
        data=data,
        feature_columns=feature_columns,
        target_column='target'
    )
    
    # Print results
    if 'error' not in results:
        print("✅ Component integration successful!")
        print(f"📊 Optimized {results['_summary']['total_features_optimized']} features")
        print(f"📊 Success rate: {results['_summary']['successful_optimizations']}/{results['_summary']['total_features_optimized']}")
        print(f"📊 Average MI score: {results['_summary']['average_mi_score']:.4f}")
        print(f"📊 Average correlation: {results['_summary']['average_correlation']:.4f}")
    else:
        print(f"❌ Component integration failed: {results['error']}")
    
    return results

def generate_optimization_summary(results: Dict[str, Any]):
    """Generate a summary report of optimization results."""
    print("\n📋 OPTIMIZATION SUMMARY REPORT")
    print("=" * 60)
    
    successful_results = [r for r in results.values() if r is not None]
    
    if not successful_results:
        print("❌ No successful optimizations to summarize")
        return
    
    # Calculate summary statistics
    total_features = len(results)
    successful_features = len(successful_results)
    success_rate = successful_features / total_features * 100
    
    avg_first_mi = np.mean([r.first_mi_score for r in successful_results])
    avg_second_mi = np.mean([r.second_mi_score for r in successful_results])
    avg_combined_mi = np.mean([r.combined_mi_score for r in successful_results])
    avg_correlation = np.mean([r.correlation_between_periods for r in successful_results])
    avg_optimization_time = np.mean([r.optimization_time for r in successful_results])
    avg_trials = np.mean([r.n_trials for r in successful_results])
    avg_convergence_rate = np.mean([r.convergence_rate for r in successful_results])
    
    print(f"📊 Total features: {total_features}")
    print(f"📊 Successful optimizations: {successful_features}")
    print(f"📊 Success rate: {success_rate:.1f}%")
    print(f"📊 Average first MI score: {avg_first_mi:.4f}")
    print(f"📊 Average second MI score: {avg_second_mi:.4f}")
    print(f"📊 Average combined MI score: {avg_combined_mi:.4f}")
    print(f"📊 Average correlation: {avg_correlation:.4f}")
    print(f"📊 Average optimization time: {avg_optimization_time:.2f} seconds")
    print(f"📊 Average trials: {avg_trials:.0f}")
    print(f"📊 Average convergence rate: {avg_convergence_rate:.4f}")
    
    # Find best and worst performing features
    best_feature = max(successful_results, key=lambda x: x.combined_mi_score)
    worst_feature = min(successful_results, key=lambda x: x.combined_mi_score)
    
    print(f"\n🏆 Best performing feature:")
    print(f"   - Combined MI score: {best_feature.combined_mi_score:.4f}")
    print(f"   - Correlation: {best_feature.correlation_between_periods:.4f}")
    
    print(f"\n📉 Worst performing feature:")
    print(f"   - Combined MI score: {worst_feature.combined_mi_score:.4f}")
    print(f"   - Correlation: {worst_feature.correlation_between_periods:.4f}")
    
    # Performance insights
    print(f"\n💡 PERFORMANCE INSIGHTS:")
    if avg_correlation < 0.5:
        print("✅ Low correlation achieved - good feature diversity")
    elif avg_correlation < 0.7:
        print("⚠️ Moderate correlation - acceptable feature diversity")
    else:
        print("❌ High correlation - poor feature diversity")
    
    if avg_combined_mi > 0.3:
        print("✅ High mutual information - strong predictive power")
    elif avg_combined_mi > 0.1:
        print("⚠️ Moderate mutual information - acceptable predictive power")
    else:
        print("❌ Low mutual information - weak predictive power")
    
    if avg_convergence_rate > 0.7:
        print("✅ Good convergence - optimization found stable solutions")
    elif avg_convergence_rate > 0.4:
        print("⚠️ Moderate convergence - optimization found reasonable solutions")
    else:
        print("❌ Poor convergence - optimization may need more trials")

def compare_optimization_methods():
    """Compare Bayesian optimization with traditional methods."""
    print("\n⚖️ OPTIMIZATION METHOD COMPARISON")
    print("=" * 60)
    
    # Generate sample data
    data = generate_sample_data(n_samples=300, n_features=2)
    feature_name = 'sma_1'
    target_column = 'target'
    
    if not BAYESIAN_OPTIMIZER_AVAILABLE:
        print("❌ Cannot run comparison - Bayesian optimizer not available")
        return
    
    # Test different configurations
    configs = {
        'Bayesian TPE': LookbackOptimizationConfig(
            optimization_method="bayesian",
            sampler_type="tpe",
            pruner_type="median",
            n_trials=20,
            enable_pruning=True
        ),
        'Bayesian Random': LookbackOptimizationConfig(
            optimization_method="bayesian",
            sampler_type="random",
            pruner_type="none",
            n_trials=20,
            enable_pruning=False
        ),
        'Grid Search': LookbackOptimizationConfig(
            optimization_method="grid",
            n_trials=20,
            enable_pruning=False
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
                'combined_mi_score': result.combined_mi_score,
                'correlation': result.correlation_between_periods,
                'optimization_time': optimization_time,
                'n_trials': result.n_trials,
                'convergence_rate': result.convergence_rate
            }
            
            print(f"✅ {method_name}: MI={result.combined_mi_score:.4f}, "
                  f"Corr={result.correlation_between_periods:.4f}, "
                  f"Time={optimization_time:.2f}s")
            
        except Exception as e:
            print(f"❌ {method_name} failed: {e}")
            comparison_results[method_name] = None
    
    # Print comparison summary
    print(f"\n📊 COMPARISON SUMMARY:")
    print("-" * 40)
    
    for method_name, result in comparison_results.items():
        if result:
            print(f"{method_name:15} | MI: {result['combined_mi_score']:.4f} | "
                  f"Corr: {result['correlation']:.4f} | Time: {result['optimization_time']:.2f}s")
        else:
            print(f"{method_name:15} | Failed")

def main():
    """Main function to run all examples."""
    print("🎯 BAYESIAN LOOKBACK PERIOD OPTIMIZATION EXAMPLES")
    print("=" * 80)
    
    try:
        # Run main Bayesian optimization example
        results = run_bayesian_optimization_example()
        
        # Run component integration example
        component_results = run_component_integration_example()
        
        # Run method comparison
        compare_optimization_methods()
        
        print("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Example execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()