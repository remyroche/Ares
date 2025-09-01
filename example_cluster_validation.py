#!/usr/bin/env python3
"""
Example script demonstrating HMM cluster validation.

This script shows how to use the cluster validation tools to test
the relevance of HMM clusters before ML training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the current directory to path to import the validator
sys.path.insert(0, str(Path(__file__).parent))

from test_hmm_cluster_relevance import HMMClusterValidator


def create_sample_cluster_data(n_samples: int = 10000, n_clusters: int = 5) -> pd.DataFrame:
    """Create sample cluster data for demonstration."""
    print(f"🔧 Creating sample data with {n_samples} samples and {n_clusters} clusters...")

    # Generate sample data
    np.random.seed(42)

    # Create time series data
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='1min')

    # Generate cluster assignments (with some temporal consistency)
    cluster_ids = []
    current_cluster = 0
    cluster_duration = n_samples // (n_clusters * 3)  # Each cluster appears multiple times

    for i in range(n_samples):
        if i % cluster_duration == 0:
            current_cluster = (current_cluster + 1) % n_clusters
        cluster_ids.append(current_cluster)

    # Add some noise to cluster transitions
    for i in range(1, len(cluster_ids)):
        if np.random.random() < 0.01:  # 1% chance of random transition
            cluster_ids[i] = np.random.randint(0, n_clusters)

    # Generate features based on clusters
    data = {
        'timestamp': timestamps,
        'composite_cluster_id': cluster_ids,
        'close': np.random.normal(100, 10, n_samples),
        'returns': np.random.normal(0, 0.02, n_samples),
    }

    # Add cluster-specific characteristics
    for i in range(n_samples):
        cluster = cluster_ids[i]

        # Volatility varies by cluster
        if cluster == 0:  # Low volatility cluster
            data['volatility_20'] = data.get('volatility_20', []) + [np.random.normal(0.01, 0.002)]
            data['price_momentum_10'] = data.get('price_momentum_10', []) + [np.random.normal(0.001, 0.001)]
            data['volume_ratio_10'] = data.get('volume_ratio_10', []) + [np.random.normal(0.8, 0.1)]
        elif cluster == 1:  # High volatility cluster
            data['volatility_20'] = data.get('volatility_20', []) + [np.random.normal(0.05, 0.01)]
            data['price_momentum_10'] = data.get('price_momentum_10', []) + [np.random.normal(0.005, 0.005)]
            data['volume_ratio_10'] = data.get('volume_ratio_10', []) + [np.random.normal(1.5, 0.3)]
        elif cluster == 2:  # Trending cluster
            data['volatility_20'] = data.get('volatility_20', []) + [np.random.normal(0.03, 0.005)]
            data['price_momentum_10'] = data.get('price_momentum_10', []) + [np.random.normal(0.01, 0.002)]
            data['volume_ratio_10'] = data.get('volume_ratio_10', []) + [np.random.normal(1.2, 0.2)]
        elif cluster == 3:  # Mean reversion cluster
            data['volatility_20'] = data.get('volatility_20', []) + [np.random.normal(0.02, 0.003)]
            data['price_momentum_10'] = data.get('price_momentum_10', []) + [np.random.normal(-0.002, 0.001)]
            data['volume_ratio_10'] = data.get('volume_ratio_10', []) + [np.random.normal(0.9, 0.15)]
        else:  # Neutral cluster
            data['volatility_20'] = data.get('volatility_20', []) + [np.random.normal(0.025, 0.004)]
            data['price_momentum_10'] = data.get('price_momentum_10', []) + [np.random.normal(0.0005, 0.002)]
            data['volume_ratio_10'] = data.get('volume_ratio_10', []) + [np.random.normal(1.0, 0.2)]

    # Add some additional features
    data['feature_1'] = np.random.normal(0, 1, n_samples)
    data['feature_2'] = np.random.normal(0, 1, n_samples)
    data['feature_3'] = np.random.normal(0, 1, n_samples)

    df = pd.DataFrame(data)

    print(f"✅ Created sample data with {len(df)} samples")
    print(f"📊 Cluster distribution: {df['composite_cluster_id'].value_counts().to_dict()}")

    return df


def run_basic_validation_example():
    """Run a basic validation example."""
    print("="*60)
    print("BASIC CLUSTER VALIDATION EXAMPLE")
    print("="*60)

    # Create sample data
    cluster_data = create_sample_cluster_data(n_samples=5000, n_clusters=5)

    # Initialize validator
    validator = HMMClusterValidator()

    # Run comprehensive validation
    print("\n🔍 Running comprehensive validation...")
    validation_results = validator.comprehensive_validation(cluster_data)

    # Print results
    print(f"\n📊 Validation Results:")
    print(f"Overall Score: {validation_results['overall_score']:.3f}")

    if 'quality_metrics' in validation_results and 'error' not in validation_results['quality_metrics']:
        qm = validation_results['quality_metrics']
        print(f"Silhouette Score: {qm.get('silhouette_score', 0):.4f}")
        print(f"Cluster Balance: {qm.get('cluster_balance', 0):.4f}")

    if 'predictive_power' in validation_results and 'error' not in validation_results['predictive_power']:
        pp = validation_results['predictive_power']
        print(f"Predictive Power: {pp.get('avg_predictability', 0):.4f}")

    if 'stability' in validation_results and 'error' not in validation_results['stability']:
        st = validation_results['stability']
        print(f"Stability: {st.get('avg_stability', 0):.4f}")

    if 'market_differentiation' in validation_results and 'error' not in validation_results['market_differentiation']:
        md = validation_results['market_differentiation']
        print(f"Differentiation: {md.get('avg_differentiation', 0):.4f}")

    # Print recommendations
    if validation_results.get("recommendations"):
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(validation_results["recommendations"], 1):
            print(f"  {i}. {rec}")

    return validation_results


def run_advanced_validation_example():
    """Run an advanced validation example with custom thresholds."""
    print("\n" + "="*60)
    print("ADVANCED CLUSTER VALIDATION EXAMPLE")
    print("="*60)

    # Create sample data
    cluster_data = create_sample_cluster_data(n_samples=10000, n_clusters=8)

    # Define custom quality thresholds
    custom_thresholds = {
        "min_silhouette": 0.4,        # Higher threshold for better quality
        "min_predictability": 0.5,    # Higher threshold for better predictability
        "min_stability": 0.6,         # Higher threshold for better stability
        "min_differentiation": 0.15,  # Higher threshold for better differentiation
        "min_return_predictability": 0.002
    }

    # Initialize validator
    validator = HMMClusterValidator()

    # Run validation with custom thresholds
    print("\n🔍 Running validation with custom thresholds...")
    validation_results = validator.comprehensive_validation(cluster_data, custom_thresholds)

    # Generate detailed report
    print("\n📄 Generating detailed report...")
    report = validator.generate_report(validation_results)
    print(report)

    # Create visualizations
    print("\n📊 Creating visualizations...")
    validator.create_visualizations(cluster_data, validation_results)

    return validation_results


def run_comparison_example():
    """Run a comparison between different cluster configurations."""
    print("\n" + "="*60)
    print("CLUSTER CONFIGURATION COMPARISON")
    print("="*60)

    # Test different numbers of clusters
    cluster_configs = [3, 5, 8, 12]
    results_comparison = {}

    for n_clusters in cluster_configs:
        print(f"\n🔍 Testing {n_clusters} clusters...")

        # Create data with different cluster count
        cluster_data = create_sample_cluster_data(n_samples=8000, n_clusters=n_clusters)

        # Run validation
        validator = HMMClusterValidator()
        validation_results = validator.comprehensive_validation(cluster_data)

        results_comparison[n_clusters] = {
            'overall_score': validation_results['overall_score'],
            'silhouette': validation_results['quality_metrics'].get('silhouette_score', 0) if 'error' not in validation_results['quality_metrics'] else 0,
            'predictability': validation_results['predictive_power'].get('avg_predictability', 0) if 'error' not in validation_results['predictive_power'] else 0,
            'stability': validation_results['stability'].get('avg_stability', 0) if 'error' not in validation_results['stability'] else 0,
            'differentiation': validation_results['market_differentiation'].get('avg_differentiation', 0) if 'error' not in validation_results['market_differentiation'] else 0
        }

    # Print comparison results
    print("\n📊 Configuration Comparison Results:")
    print(f"{'Clusters':<10} {'Overall':<10} {'Silhouette':<12} {'Predictability':<15} {'Stability':<12} {'Differentiation':<15}")
    print("-" * 80)

    for n_clusters, results in results_comparison.items():
        print(f"{n_clusters:<10} {results['overall_score']:<10.3f} {results['silhouette']:<12.4f} "
              f"{results['predictability']:<15.4f} {results['stability']:<12.4f} {results['differentiation']:<15.4f}")

    # Find best configuration
    best_config = max(results_comparison.items(), key=lambda x: x[1]['overall_score'])
    print(f"\n🏆 Best configuration: {best_config[0]} clusters (score: {best_config[1]['overall_score']:.3f})")

    return results_comparison


def main():
    """Main function to run all examples."""
    print("🚀 HMM Cluster Validation Examples")
    print("This script demonstrates how to test HMM cluster relevance before ML training.")

    try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
        # Run basic example
        basic_results = run_basic_validation_example()

        # Run advanced example
        advanced_results = run_advanced_validation_example()

        # Run comparison example
        comparison_results = run_comparison_example()

        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*60)
        print("💡 Key takeaways:")
        print("1. Use the validation script to test cluster quality before ML training")
        print("2. Adjust thresholds based on your specific requirements")
        print("3. Compare different cluster configurations to find optimal settings")
        print("4. Use the generated reports and visualizations for analysis")
        print("5. Follow recommendations to improve cluster quality")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()