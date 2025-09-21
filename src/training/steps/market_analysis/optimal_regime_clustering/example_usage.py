"""
Example Usage of Optimal Regime Clustering

This script demonstrates how to use the optimal regime clustering system to create
20 optimal clusters from HMM regime discovery output for ML model training.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from optimal_regime_clustering import (
    run_optimal_clustering,
    run_high_quality_clustering,
    run_fast_clustering,
    OptimalClusteringConfig,
    create_optimal_clusterer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def example_basic_clustering():
    """Example 1: Basic optimal clustering usage."""
    print("\n" + "="*60)
    print("Example 1: Basic Optimal Clustering")
    print("="*60)

    try:
        # Use default HMM cluster data path
        data_path = "historical_data/binance/ethusdt/hmm_clusters/hmm_composite_clusters_binance_ETHUSDT_1h.parquet"

        # Check if data exists, if not use sample data
        if not Path(data_path).exists():
            print(f"Data file not found: {data_path}")
            print("Creating sample data for demonstration...")

            # Create sample regime data
            n_samples = 10000
            np.random.seed(42)

            sample_data = pd.DataFrame({
                'volume': np.random.exponential(100, n_samples),
                'volatility': np.random.beta(2, 5, n_samples) * 0.1,
                'momentum': np.random.normal(0, 0.02, n_samples),
                'trend': np.random.normal(0, 0.05, n_samples),
                'timestamp': pd.date_range(start='2020-01-01', periods=n_samples, freq='H')
            })

            data_path = "sample_regime_data.parquet"
            sample_data.to_parquet(data_path)
            print(f"Created sample data: {data_path}")

        # Run basic clustering
        results = run_optimal_clustering(
            data_path=data_path,
            output_dir="optimal_clusters_example_1/",
            symbol="ETHUSDT",
            exchange="binance",
            timeframe="1h"
        )

        if results['success']:
            print("✅ Basic clustering completed successfully!")
            print(f"   Execution time: {results['execution_time']".2f"} seconds")
            print(f"   Number of clusters: {results['clustering_result'].statistics.n_clusters}")
            print(f"   Coverage: {results['clustering_result'].statistics.coverage_percentage".3f"}")
            print(f"   Noise: {results['clustering_result'].statistics.noise_percentage".3f"}")
            print(f"   Files saved: {len(results['saved_files'])}")
        else:
            print(f"❌ Basic clustering failed: {results['error']}")

    except Exception as e:
        print(f"❌ Error in basic clustering example: {e}")
        print("This is expected if HMM data is not available")

def example_high_quality_clustering():
    """Example 2: High-quality clustering with enhanced parameters."""
    print("\n" + "="*60)
    print("Example 2: High-Quality Clustering")
    print("="*60)

    try:
        # Create sample data for demonstration
        n_samples = 5000
        np.random.seed(42)

        sample_data = pd.DataFrame({
            'volume': np.random.exponential(100, n_samples),
            'volatility': np.random.beta(2, 5, n_samples) * 0.1,
            'momentum': np.random.normal(0, 0.02, n_samples),
            'trend': np.random.normal(0, 0.05, n_samples),
            'timestamp': pd.date_range(start='2020-01-01', periods=n_samples, freq='H')
        })

        # Run high-quality clustering
        results = run_high_quality_clustering(
            data_path=sample_data,  # Pass DataFrame directly
            output_dir="optimal_clusters_example_2/",
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="1h"
        )

        if results['success']:
            print("✅ High-quality clustering completed successfully!")
            print(f"   Execution time: {results['execution_time']".2f"} seconds")
            print(f"   Number of clusters: {results['clustering_result'].statistics.n_clusters}")
            print(f"   Coverage: {results['clustering_result'].statistics.coverage_percentage".3f"}")
            print(f"   Noise: {results['clustering_result'].statistics.noise_percentage".3f"}")
            print(f"   Silhouette score: {results['clustering_result'].quality_metrics.get('silhouette', 0)".3f"}")
        else:
            print(f"❌ High-quality clustering failed: {results['error']}")

    except Exception as e:
        print(f"❌ Error in high-quality clustering example: {e}")

def example_fast_clustering():
    """Example 3: Fast clustering for quick results."""
    print("\n" + "="*60)
    print("Example 3: Fast Clustering")
    print("="*60)

    try:
        # Create sample data
        n_samples = 2000
        np.random.seed(42)

        sample_data = pd.DataFrame({
            'volume': np.random.exponential(100, n_samples),
            'volatility': np.random.beta(2, 5, n_samples) * 0.1,
            'momentum': np.random.normal(0, 0.02, n_samples),
            'trend': np.random.normal(0, 0.05, n_samples),
            'timestamp': pd.date_range(start='2020-01-01', periods=n_samples, freq='H')
        })

        # Run fast clustering
        results = run_fast_clustering(
            data_path=sample_data,
            output_dir="optimal_clusters_example_3/",
            symbol="ADAUSDT",
            exchange="binance",
            timeframe="1h"
        )

        if results['success']:
            print("✅ Fast clustering completed successfully!")
            print(f"   Execution time: {results['execution_time']".2f"} seconds")
            print(f"   Number of clusters: {results['clustering_result'].statistics.n_clusters}")
            print(f"   Coverage: {results['clustering_result'].statistics.coverage_percentage".3f"}")
            print(f"   Noise: {results['clustering_result'].statistics.noise_percentage".3f"}")
        else:
            print(f"❌ Fast clustering failed: {results['error']}")

    except Exception as e:
        print(f"❌ Error in fast clustering example: {e}")

def example_custom_configuration():
    """Example 4: Custom configuration for specific requirements."""
    print("\n" + "="*60)
    print("Example 4: Custom Configuration")
    print("="*60)

    try:
        # Create custom configuration
        custom_config = OptimalClusteringConfig()
        custom_config.target_n_clusters = 15  # Fewer clusters
        custom_config.max_noise_pct = 0.03    # Lower noise tolerance
        custom_config.min_cluster_size_pct = 0.04  # Larger minimum clusters
        custom_config.max_cluster_size_pct = 0.12  # Larger maximum clusters

        # Create sample data
        n_samples = 3000
        np.random.seed(42)

        sample_data = pd.DataFrame({
            'volume': np.random.exponential(100, n_samples),
            'volatility': np.random.beta(2, 5, n_samples) * 0.1,
            'momentum': np.random.normal(0, 0.02, n_samples),
            'trend': np.random.normal(0, 0.05, n_samples),
            'timestamp': pd.date_range(start='2020-01-01', periods=n_samples, freq='H')
        })

        # Run clustering with custom config
        from optimal_regime_clustering import OptimalRegimeClusteringOrchestrator
        orchestrator = OptimalRegimeClusteringOrchestrator(custom_config)

        results = orchestrator.run_clustering_pipeline(
            data_path=sample_data,
            output_dir="optimal_clusters_example_4/",
            symbol="DOTUSDT",
            exchange="binance",
            timeframe="1h"
        )

        if results['success']:
            print("✅ Custom configuration clustering completed successfully!")
            print(f"   Target clusters: {custom_config.target_n_clusters}")
            print(f"   Actual clusters: {results['clustering_result'].statistics.n_clusters}")
            print(f"   Execution time: {results['execution_time']".2f"} seconds")
            print(f"   Coverage: {results['clustering_result'].statistics.coverage_percentage".3f"}")
            print(f"   Noise: {results['clustering_result'].statistics.noise_percentage".3f"}")
        else:
            print(f"❌ Custom configuration clustering failed: {results['error']}")

    except Exception as e:
        print(f"❌ Error in custom configuration example: {e}")

def example_ml_integration():
    """Example 5: ML integration workflow."""
    print("\n" + "="*60)
    print("Example 5: ML Integration Workflow")
    print("="*60)

    try:
        # Create sample data
        n_samples = 5000
        np.random.seed(42)

        sample_data = pd.DataFrame({
            'volume': np.random.exponential(100, n_samples),
            'volatility': np.random.beta(2, 5, n_samples) * 0.1,
            'momentum': np.random.normal(0, 0.02, n_samples),
            'trend': np.random.normal(0, 0.05, n_samples),
            'timestamp': pd.date_range(start='2020-01-01', periods=n_samples, freq='H')
        })

        # Run clustering
        results = run_optimal_clustering(
            data_path=sample_data,
            output_dir="ml_integration_example/",
            symbol="ETHUSDT",
            exchange="binance",
            timeframe="1h"
        )

        if results['success']:
            print("✅ ML integration clustering completed successfully!")

            # Show how to access ML-ready datasets
            ml_datasets = results['ml_datasets']
            print(f"   ML datasets created: {len(ml_datasets)}")

            for dataset_name, dataset_path in ml_datasets.items():
                if dataset_name.startswith('cluster_'):
                    cluster_id = dataset_name.split('_')[1]
                    print(f"   Cluster {cluster_id}: {dataset_path}")

                    # Load example cluster data
                    cluster_df = pd.read_parquet(dataset_path)
                    print(f"     - Size: {len(cluster_df)} samples")
                    print(f"     - Features: {len(cluster_df.columns)} columns")

                    # Show basic statistics
                    print(f"     - Volume mean: {cluster_df['volume'].mean()".2f"}")
                    print(f"     - Volatility mean: {cluster_df['volatility'].mean()".4f"}")
                    print(f"     - Momentum mean: {cluster_df['momentum'].mean()".4f"}")

                    break  # Show only first cluster

            print("\n   💡 These datasets are ready for ML training!")
            print("   💡 Each cluster represents a distinct market regime")
            print("   💡 Use cluster characteristics for regime-aware model training")

        else:
            print(f"❌ ML integration clustering failed: {results['error']}")

    except Exception as e:
        print(f"❌ Error in ML integration example: {e}")

def example_hmm_integration():
    """Example 6: Integration with HMM regime discovery output."""
    print("\n" + "="*60)
    print("Example 6: HMM Integration")
    print("="*60)

    try:
        # This example shows how to integrate with actual HMM output
        print("🔍 Looking for HMM regime data...")

        # Try to find actual HMM data
        possible_paths = [
            "historical_data/binance/ethusdt/hmm_clusters/hmm_composite_clusters_binance_ETHUSDT_1h.parquet",
            "artifacts/hmm_regime_unified_artifacts.json",
            "/workspace/historical_data/binance/ethusdt/hmm_clusters/hmm_composite_clusters_binance_ETHUSDT_1h.parquet"
        ]

        hmm_data = None
        data_path = None

        for path in possible_paths:
            if Path(path).exists():
                if path.endswith('.parquet'):
                    try:
                        hmm_data = pd.read_parquet(path)
                        data_path = path
                        print(f"✅ Found HMM data: {path}")
                        break
                    except Exception as e:
                        print(f"   Error loading {path}: {e}")
                elif path.endswith('.json'):
                    try:
                        with open(path, 'r') as f:
                            json_data = json.load(f)
                        print(f"   Found HMM artifacts: {path}")
                        # Extract cluster data from JSON if available
                        continue
                    except Exception as e:
                        print(f"   Error loading {path}: {e}")

        if hmm_data is None:
            print("❌ No HMM data found. Creating sample data for demonstration...")
            n_samples = 8000
            np.random.seed(42)

            # Create realistic HMM-like regime data
            hmm_data = pd.DataFrame({
                'volume': np.random.exponential(100, n_samples),
                'volatility': np.random.beta(2, 5, n_samples) * 0.1,
                'momentum': np.random.normal(0, 0.02, n_samples),
                'trend': np.random.normal(0, 0.05, n_samples),
                'timestamp': pd.date_range(start='2020-01-01', periods=n_samples, freq='H')
            })

            data_path = "hmm_sample_data.parquet"
            hmm_data.to_parquet(data_path)
            print(f"   Created sample HMM data: {data_path}")

        # Run clustering on HMM data
        results = run_optimal_clustering(
            data_path=hmm_data,  # Pass DataFrame directly
            output_dir="hmm_integration_clusters/",
            symbol="ETHUSDT",
            exchange="binance",
            timeframe="1h"
        )

        if results['success']:
            print("✅ HMM integration clustering completed successfully!")
            print(f"   Original HMM data: {len(hmm_data)} samples")
            print(f"   Optimal clusters: {results['clustering_result'].statistics.n_clusters}")
            print(f"   Coverage: {results['clustering_result'].statistics.coverage_percentage".3f"}")
            print(f"   Noise: {results['clustering_result'].statistics.noise_percentage".3f"}")

            # Show cluster distribution
            print("\n   📊 Cluster Distribution:")
            for i, (size, pct) in enumerate(zip(
                results['clustering_result'].statistics.cluster_sizes,
                results['clustering_result'].statistics.cluster_percentages
            )):
                print(f"     Cluster {i}: {size} samples ({pct".3f"})")

            print("\n   💡 These clusters are ready for regime-aware ML training!")
            print("   💡 Each cluster represents a consolidated market regime")
            print("   💡 Use cluster assignments for supervised learning")

        else:
            print(f"❌ HMM integration clustering failed: {results['error']}")

    except Exception as e:
        print(f"❌ Error in HMM integration example: {e}")

def main():
    """Run all examples."""
    print("🚀 Optimal Regime Clustering Examples")
    print("This demonstrates how to create optimal clusters from HMM regime discovery")
    print("for ML model training with 90-95% coverage and <5% noise.\n")

    # Run examples
    example_basic_clustering()
    example_high_quality_clustering()
    example_fast_clustering()
    example_custom_configuration()
    example_ml_integration()
    example_hmm_integration()

    print("\n" + "="*60)
    print("📋 Summary")
    print("="*60)
    print("✅ All examples completed!")
    print("📁 Check the output directories for generated files:")
    print("   - optimal_clusters_example_1/")
    print("   - optimal_clusters_example_2/")
    print("   - optimal_clusters_example_3/")
    print("   - optimal_clusters_example_4/")
    print("   - ml_integration_example/")
    print("   - hmm_integration_clusters/")
    print("\n💡 Key Features:")
    print("   • 20 optimal clusters from HMM regime discovery")
    print("   • 90-95% data coverage with 3-8% cluster sizes")
    print("   • <5% noise with hybrid clustering algorithms")
    print("   • ML-ready datasets for each cluster")
    print("   • Comprehensive validation and reporting")
    print("\n🔧 Integration Options:")
    print("   • run_optimal_clustering(): Balanced performance")
    print("   • run_high_quality_clustering(): Enhanced quality")
    print("   • run_fast_clustering(): Quick processing")
    print("   • Custom configuration for specific needs")

if __name__ == "__main__":
    main()