#!/usr/bin/env python3
"""
Example script demonstrating HMM regime parameter optimization.

This script shows how to use the optimization system to find the best parameters
for capturing distinct market conditions in HMM regime discovery.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

# Add the current directory to path to import the optimizer
sys.path.insert(0, str(Path(__file__).parent))

from optimize_hmm_regime_parameters import HMMRegimeOptimizer, identify_market_condition_columns


def create_sample_market_data(...) -> ...:
    pass"""..."""
    passprint(f"🔧 Creating sample market data with {n_samples} samples...")

    # Generate sample data
    np.random.seed(42)

    # Create time series data
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='1min')

    # Generate market conditions
    data = {
        'timestamp': timestamps,
        'close': np.random.normal(100, 10, n_samples),
        'volume': np.random.normal(1000000, 200000, n_samples),
    }

    # Create distinct market regimes with different characteristics
    regime_length = n_samples // 8  # 8 different regimes

    for i in range(n_samples):
    passpassregime = i // regime_length

        if regime == 0:  # Low volatility, low momentum
            data['volatility_20'] = data.get('volatility_20', []) + [np.random.normal(0.01, 0.002)]
            data['price_momentum_10'] = data.get('price_momentum_10', []) + [np.random.normal(0.001, 0.001)]
            data['volume_ratio_10'] = data.get('volume_ratio_10', []) + [np.random.normal(0.8, 0.1)]
            data['rsi_14'] = data.get('rsi_14', []) + [np.random.normal(50, 10)]
            data['macd'] = data.get('macd', []) + [np.random.normal(0, 0.1)]

        elif regime == 1:  # High volatility, high momentum
            data['volatility_20'] = data.get('volatility_20', []) + [np.random.normal(0.05, 0.01)]
            data['price_momentum_10'] = data.get('price_momentum_10', []) + [np.random.normal(0.005, 0.005)]
            data['volume_ratio_10'] = data.get('volume_ratio_10', []) + [np.random.normal(1.5, 0.3)]
            data['rsi_14'] = data.get('rsi_14', []) + [np.random.normal(70, 15)]
            data['macd'] = data.get('macd', []) + [np.random.normal(0.2, 0.3)]

        elif regime == 2:  # Trending up
            data['volatility_20'] = data.get('volatility_20', []) + [np.random.normal(0.03, 0.005)]
            data['price_momentum_10'] = data.get('price_momentum_10', []) + [np.random.normal(0.01, 0.002)]
            data['volume_ratio_10'] = data.get('volume_ratio_10', []) + [np.random.normal(1.2, 0.2)]
            data['rsi_14'] = data.get('rsi_14', []) + [np.random.normal(60, 8)]
            data['macd'] = data.get('macd', []) + [np.random.normal(0.1, 0.2)]

        elif regime == 3:  # Trending down
            data['volatility_20'] = data.get('volatility_20', []) + [np.random.normal(0.03, 0.005)]
            data['price_momentum_10'] = data.get('price_momentum_10', []) + [np.random.normal(-0.01, 0.002)]
            data['volume_ratio_10'] = data.get('volume_ratio_10', []) + [np.random.normal(1.1, 0.2)]
            data['rsi_14'] = data.get('rsi_14', []) + [np.random.normal(40, 8)]
            data['macd'] = data.get('macd', []) + [np.random.normal(-0.1, 0.2)]

        elif regime == 4:  # Mean reversion
            data['volatility_20'] = data.get('volatility_20', []) + [np.random.normal(0.02, 0.003)]
            data['price_momentum_10'] = data.get('price_momentum_10', []) + [np.random.normal(-0.002, 0.001)]
            data['volume_ratio_10'] = data.get('volume_ratio_10', []) + [np.random.normal(0.9, 0.15)]
            data['rsi_14'] = data.get('rsi_14', []) + [np.random.normal(45, 5)]
            data['macd'] = data.get('macd', []) + [np.random.normal(-0.05, 0.1)]

        elif regime == 5:  # High volume, low volatility
            data['volatility_20'] = data.get('volatility_20', []) + [np.random.normal(0.015, 0.003)]
            data['price_momentum_10'] = data.get('price_momentum_10', []) + [np.random.normal(0.0005, 0.002)]
            data['volume_ratio_10'] = data.get('volume_ratio_10', []) + [np.random.normal(1.8, 0.4)]
            data['rsi_14'] = data.get('rsi_14', []) + [np.random.normal(55, 12)]
            data['macd'] = data.get('macd', []) + [np.random.normal(0.05, 0.15)]

        elif regime == 6:  # Low volume, high volatility
            data['volatility_20'] = data.get('volatility_20', []) + [np.random.normal(0.04, 0.008)]
            data['price_momentum_10'] = data.get('price_momentum_10', []) + [np.random.normal(0.003, 0.004)]
            data['volume_ratio_10'] = data.get('volume_ratio_10', []) + [np.random.normal(0.6, 0.1)]
            data['rsi_14'] = data.get('rsi_14', []) + [np.random.normal(35, 15)]
            data['macd'] = data.get('macd', []) + [np.random.normal(-0.15, 0.25)]

        else:  # Neutral regime
            data['volatility_20'] = data.get('volatility_20', []) + [np.random.normal(0.025, 0.004)]
            data['price_momentum_10'] = data.get('price_momentum_10', []) + [np.random.normal(0.0005, 0.002)]
            data['volume_ratio_10'] = data.get('volume_ratio_10', []) + [np.random.normal(1.0, 0.2)]
            data['rsi_14'] = data.get('rsi_14', []) + [np.random.normal(50, 10)]
            data['macd'] = data.get('macd', []) + [np.random.normal(0, 0.1)]

    # Add some additional features
    data['returns'] = np.random.normal(0, 0.02, n_samples)
    data['price_change'] = np.random.normal(0, 0.015, n_samples)
    data['trend_strength'] = np.random.normal(0, 0.5, n_samples)
    data['market_regime'] = np.random.normal(0, 1, n_samples)
    data['condition_state'] = np.random.normal(0, 0.8, n_samples)

    # Add some technical indicators
    data['bollinger_position'] = np.random.uniform(0, 1, n_samples)
    data['atr_14'] = np.random.normal(0.02, 0.005, n_samples)
    data['adx_14'] = np.random.normal(25, 10, n_samples)
    data['stoch_k'] = np.random.uniform(0, 100, n_samples)
    data['cci_14'] = np.random.normal(0, 100, n_samples)

    df = pd.DataFrame(data)

    print(f"✅ Created sample market data with {len(df)} samples")
    print(f"📊 Market condition columns: {[col for col in df.columns if any(keyword in col.lower() for keyword in ['volatility', 'momentum', 'volume', 'rsi', 'macd', 'bollinger', 'atr', 'adx', 'stoch', 'cci'])]}")

    return df


def run_basic_optimization_example(...):
    passpasspass"""Run a basic optimization example."""
    print("="*60)
    print("BASIC HMM OPTIMIZATION EXAMPLE")
    print("="*60)

    # Create sample data
    data = create_sample_market_data(n_samples=5000)

    # Identify feature and market condition columns
    feature_columns = [col for col in data.columns
                      if col not in ['timestamp', 'composite_cluster_id']]
    market_condition_columns = identify_market_condition_columns(data)

    print(f"\n🔧 Features: {len(feature_columns)}")
    print(f"📈 Market conditions: {len(market_condition_columns)}")
    print(f"📈 Market condition columns: {market_condition_columns}")

    # Initialize optimizer
    optimizer = HMMRegimeOptimizer()

    # Run optimization with fewer trials for demonstration
    print(f"\n🚀 Running optimization with 20 trials...")
    results = optimizer.optimize(
        data=data,
        feature_columns=feature_columns,
        market_condition_columns=market_condition_columns,
        n_trials=20,
        study_name="basic_optimization_example"
    )

    # Print results
    print(f"\n📊 Optimization Results:")
    print(f"Best Score: {results['best_score']:.4f}")
    print(f"Best Parameters:")
    for param, value in results['best_params'].items():
    passprint(f"  {param}: {value}")

    return results


def run_advanced_optimization_example(...):
    pass"""Run an advanced optimization example with custom configuration."""
    print("\n" + "="*60)
    print("ADVANCED HMM OPTIMIZATION EXAMPLE")
    print("="*60)

    # Create sample data
    data = create_sample_market_data(n_samples=8000)

    # Load custom configuration
    config = {
        "optimization_settings": {
            "n_trials": 50,
            "timeout": 1800,
            "study_name": "advanced_optimization_example"
        },
        "evaluation_weights": {
            "market_differentiation": 0.5,  # Emphasize market differentiation more
            "cluster_quality": 0.2,
            "market_consistency": 0.2,
            "cluster_balance": 0.05,
            "market_separation": 0.05
        }
    }

    # Identify feature and market condition columns
    feature_columns = [col for col in data.columns
                      if col not in ['timestamp', 'composite_cluster_id']]
    market_condition_columns = identify_market_condition_columns(data)

    # Initialize optimizer with custom config
    optimizer = HMMRegimeOptimizer(config)

    # Run optimization
    print(f"\n🚀 Running advanced optimization with 50 trials...")
    results = optimizer.optimize(
        data=data,
        feature_columns=feature_columns,
        market_condition_columns=market_condition_columns,
        n_trials=50,
        study_name="advanced_optimization_example"
    )

    # Generate detailed report
    print(f"\n📄 Generating detailed report...")
    report = optimizer.generate_optimization_report()
    print(report)

    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    optimizer.create_optimization_visualizations()

    return results


def run_comparison_optimization_example(...):
    passpasspasspass"""Run optimization with different evaluation strategies."""
    print("\n" + "="*60)
    print("OPTIMIZATION STRATEGY COMPARISON")
    print("="*60)

    # Create sample data
    data = create_sample_market_data(n_samples=6000)

    # Identify feature and market condition columns
    feature_columns = [col for col in data.columns
                      if col not in ['timestamp', 'composite_cluster_id']]
    market_condition_columns = identify_market_condition_columns(data)

    # Different evaluation strategies
    strategies = {
        "market_focused": {
            "market_differentiation": 0.6,
            "cluster_quality": 0.1,
            "market_consistency": 0.2,
            "cluster_balance": 0.05,
            "market_separation": 0.05
        },
        "balanced": {
            "market_differentiation": 0.3,
            "cluster_quality": 0.3,
            "market_consistency": 0.2,
            "cluster_balance": 0.1,
            "market_separation": 0.1
        },
        "quality_focused": {
            "market_differentiation": 0.2,
            "cluster_quality": 0.5,
            "market_consistency": 0.1,
            "cluster_balance": 0.1,
            "market_separation": 0.1
        }
    }

    results_comparison = {}

    for strategy_name, weights in strategies.items():
    passprint(f"\n🔍 Testing {strategy_name} strategy...")

        # Create config with custom weights
        config = {
            "evaluation_weights": weights,
            "optimization_settings": {
                "n_trials": 30,
                "study_name": f"strategy_{strategy_name}"
            }
        }

        # Initialize optimizer
        optimizer = HMMRegimeOptimizer(config)

        # Run optimization
        results = optimizer.optimize(
            data=data,
            feature_columns=feature_columns,
            market_condition_columns=market_condition_columns,
            n_trials=30,
            study_name=f"strategy_{strategy_name}"
        )

        results_comparison[strategy_name] = {
            'best_score': results['best_score'],
            'best_params': results['best_params'],
            'weights': weights
        }

    # Print comparison results
    print(f"\n📊 Strategy Comparison Results:")
    print(f"{'Strategy':<15} {'Score':<10} {'Key Parameters'}")
    print("-" * 50)

    for strategy_name, result in results_comparison.items():
    passparams = result['best_params']
        key_params = f"n_components={params.get('n_components', 'N/A')}, " \
                    f"clustering_method={params.get('clustering_method', 'N/A')}"
        print(f"{strategy_name:<15} {result['best_score']:<10.4f} {key_params}")

    # Find best strategy
    best_strategy = max(results_comparison.items(), key=lambda x: x[1]['best_score'])
    print(f"\n🏆 Best strategy: {best_strategy[0]} (score: {best_strategy[1]['best_score']:.4f})")

    return results_comparison


def demonstrate_parameter_application(...):
    pass"""Demonstrate how to apply optimized parameters."""
    print("\n" + "="*60)
    print("PARAMETER APPLICATION DEMONSTRATION")
    print("="*60)

    # Create sample data
    data = create_sample_market_data(n_samples=4000)

    # Identify feature and market condition columns
    feature_columns = [col for col in data.columns
                      if col not in ['timestamp', 'composite_cluster_id']]
    market_condition_columns = identify_market_condition_columns(data)

    # Run a quick optimization to get best parameters
    optimizer = HMMRegimeOptimizer()
    results = optimizer.optimize(
        data=data,
        feature_columns=feature_columns,
        market_condition_columns=market_condition_columns,
        n_trials=15,
        study_name="parameter_application_demo"
    )

    best_params = results['best_params']

    print(f"🔧 Best parameters found:")
    for param, value in best_params.items():
    passprint(f"  {param}: {value}")

    # Demonstrate how to apply these parameters
    print(f"\n📋 How to apply these parameters to your Step 3 HMM regime discovery:")
    print(f"1. Update your configuration file with the best parameters")
    print(f"2. Modify your step3_hmm_regime_discovery.py to use these parameters")
    print(f"3. Run the validation script to confirm improved cluster quality")

    # Create a sample configuration snippet
    config_snippet = {
        "hmm_parameters": {
            "n_components": best_params.get('n_components', 5),
            "covariance_type": best_params.get('covariance_type', 'full'),
            "n_iter": best_params.get('n_iter', 100),
            "tol": best_params.get('tol', 1e-4),
            "reg_covar": best_params.get('reg_covar', 1e-6)
        },
        "clustering_parameters": {
            "method": best_params.get('clustering_method', 'kmeans'),
            "n_clusters": best_params.get('n_clusters', 5)
        },
        "feature_parameters": {
            "use_pca": best_params.get('use_pca', False),
            "scaling_method": best_params.get('scaling_method', 'standard')
        }
    }

    print(f"\n📄 Sample configuration snippet:")
    print(json.dumps(config_snippet, indent=2))

    return best_params


def main(...):
    pass"""Main function to run all optimization examples."""
    print("🚀 HMM Regime Parameter Optimization Examples")
    print("This script demonstrates how to optimize HMM parameters for capturing distinct market conditions.")

    try:
    passpasspass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
        # Run basic example
        basic_results = run_basic_optimization_example()

        # Run advanced example
        advanced_results = run_advanced_optimization_example()

        # Run comparison example
        comparison_results = run_comparison_optimization_example()

        # Demonstrate parameter application
        best_params = demonstrate_parameter_application()

        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*60)
        print("💡 Key takeaways:")
        print("1. Use the optimization script to find best parameters for your data")
        print("2. Focus on market condition differentiation rather than transition prediction")
        print("3. Experiment with different evaluation weight strategies")
        print("4. Apply the best parameters to your Step 3 HMM regime discovery")
        print("5. Validate the optimized clusters using the cluster validation tools")
        print("6. Integrate the optimization into your pipeline for continuous improvement")

    except Exception as e:
    passpasspasspasspasspasspasspasspassprint(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    passmain()