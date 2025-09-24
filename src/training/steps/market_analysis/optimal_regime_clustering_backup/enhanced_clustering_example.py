"""
Enhanced Clustering Usage Example

This example demonstrates how to use the enhanced clustering system with 4D frontier optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

# Import the enhanced clustering system from the main optimized_clustering module
from .optimized_clustering import (
    MatrixOptimizedClusterer,
    cluster_regimes_enhanced,
    cluster_regimes_optimized,
    create_matrix_optimized_clusterer
)
from .config import OptimalClusteringConfig, ENHANCED_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_regime_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample regime data for testing.

    Args:
        n_samples: Number of samples to generate

    Returns:
        DataFrame with sample regime data
    """
    np.random.seed(42)

    # Generate synthetic 4D regime data
    data = {
        'volume': np.random.exponential(1.0, n_samples),
        'volatility': np.random.beta(2, 5, n_samples),
        'momentum': np.random.normal(0, 1, n_samples),
        'trend': np.random.normal(0, 0.5, n_samples),
        'volume_sma_20': np.random.exponential(1.2, n_samples),
        'volatility_sma_10': np.random.beta(1.5, 4, n_samples),
        'momentum_ema_5': np.random.normal(0.1, 0.8, n_samples),
        'trend_sma_15': np.random.normal(0.05, 0.3, n_samples),
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'regime_id': range(n_samples)
    }

    return pd.DataFrame(data)

def demonstrate_enhanced_clustering():
    """Demonstrate the enhanced clustering capabilities."""

    logger.info("🚀 Starting enhanced clustering demonstration")

    # Step 1: Create sample data
    logger.info("📊 Creating sample regime data...")
    regime_data = create_sample_regime_data(1000)

    # Step 2: Create enhanced configuration
    logger.info("⚙️ Creating enhanced clustering configuration...")
    config = ENHANCED_CONFIG

    # Step 3: Run enhanced clustering
    logger.info("🔬 Running enhanced clustering...")
    result = cluster_regimes_enhanced(regime_data, config)

    # Step 4: Analyze results
    if result.success:
        logger.info("✅ Enhanced clustering completed successfully!")

        # Print key metrics
        print("\n" + "="*60)
        print("ENHANCED CLUSTERING RESULTS")
        print("="*60)

        print(f"📈 Number of clusters: {result.statistics.n_clusters}")
        print(f"📊 Coverage: {result.statistics.coverage_percentage:.3f}")
        print(f"🔊 Noise percentage: {result.statistics.noise_percentage:.3f}")

        print(f"\n📊 Quality Metrics:")
        print(f"  • Silhouette Score: {result.quality_metrics.get('silhouette', 0.0):.3f}")
        print(f"  • Davies-Bouldin Score: {result.quality_metrics.get('davies_bouldin', float('inf')):.3f}")
        print(f"  • Calinski-Harabasz Score: {result.quality_metrics.get('calinski_harabasz', 0.0):.3f}")

        print(f"\n📊 Enhanced CV Metrics:")
        print(f"  • Mean within-cluster CV: {result.quality_metrics.get('mean_within_cluster_cv', 0.0):.3f}")
        print(f"  • Enhanced quality score: {result.quality_metrics.get('enhanced_quality_score', 0.0):.3f}")

        print(f"\n📊 Performance Improvements:")
        perf_metrics = result.performance_metrics
        print(f"  • Silhouette improvement: {perf_metrics.get('improvement_silhouette', 0.0):.3f}")
        print(f"  • Davies-Bouldin improvement: {perf_metrics.get('improvement_davies_bouldin', 0.0):.3f}")

        print(f"\n🔄 Optimization Details:")
        print(f"  • Frontier optimization applied: {result.metadata.get('frontier_optimization_applied', False)}")
        print(f"  • Optimization iterations: {result.metadata.get('optimization_iterations', 0)}")
        print(f"  • Transfer operations: {result.metadata.get('transfer_operations', 0)}")

        # Analyze frontiers
        print(f"\n🗺️ 4D Frontiers Established:")
        frontiers = result.metadata.get('frontiers', {})
        total_frontiers = sum(len(frontier_list) for frontier_list in frontiers.values())
        print(f"  • Total frontiers: {total_frontiers}")

        for frontier_type, frontier_list in frontiers.items():
            print(f"  • {frontier_type}: {len(frontier_list)} frontiers")

        # Analyze cluster sizes
        print(f"\n📏 Cluster Size Distribution:")
        sizes = result.statistics.cluster_sizes
        percentages = result.statistics.cluster_percentages

        print(f"  • Mean cluster size: {result.statistics.mean_cluster_size:.2f} ({result.statistics.mean_cluster_size/len(result.labels)*100:.2f}%)")
        print(f"  • Std cluster size: {result.statistics.std_cluster_size:.2f}")
        print(f"  • Min cluster size: {result.statistics.min_cluster_size:.2f} ({percentages.min()*100:.2f}%)")
        print(f"  • Max cluster size: {result.statistics.max_cluster_size:.2f} ({percentages.max()*100:.2f}%)")

        # Count clusters in target range (3-8%)
        target_range_count = np.sum((percentages >= 0.03) & (percentages <= 0.08))
        print(f"  • Clusters in 3-8% range: {target_range_count}/{len(sizes)} ({target_range_count/len(sizes)*100:.1f}%)")

        # Analyze transfer history
        print(f"\n🔄 Regime Transfer History:")
        transfer_history = result.metadata.get('transfer_history', [])
        print(f"  • Total transfers: {len(transfer_history)}")

        if transfer_history:
            benefits = [transfer['benefit'] for transfer in transfer_history]
            print(f"  • Mean transfer benefit: {np.mean(benefits):.3f}")
            print(f"  • Max transfer benefit: {np.max(benefits):.3f}")

        print(f"\n" + "="*60)

    else:
        logger.error(f"❌ Enhanced clustering failed: {result.error_message}")

def demonstrate_custom_configuration():
    """Demonstrate custom configuration options."""

    logger.info("⚙️ Demonstrating custom configuration options...")

    # Create custom configuration for high-quality clustering
    config = OptimalClusteringConfig.create_high_quality()

    # Adjust for 5% average cluster size
    config.min_cluster_size_pct = 0.03  # 3%
    config.max_cluster_size_pct = 0.08  # 8%
    config.target_coverage_pct = 0.92   # 92% coverage

    # Enable all enhanced features
    config.weighted_4d_mapping = True
    config.equidistant_centroids = True
    config.size_constrained_merging = True
    config.cv_based_similarity = True
    config.cv_optimized_splitting = True
    config.enhanced_redistribution = True
    config.iterative_refinement = True

    # Set higher quality thresholds
    config.min_silhouette_score = 0.4
    config.min_calinski_harabasz_score = 250.0
    config.min_davies_bouldin_score = 1.2

    print("✅ Custom configuration created:")
    print(f"  • Target clusters: {config.target_n_clusters}")
    print(f"  • Size range: {config.min_cluster_size_pct*100:.1f}% - {config.max_cluster_size_pct*100:.1f}%")
    print(f"  • Quality thresholds - Silhouette: {config.min_silhouette_score}, CH: {config.min_calinski_harabasz_score}, DB: {config.min_davies_bouldin_score}")
    print(f"  • Enhanced features enabled: {config.weighted_4d_mapping and config.cv_based_similarity}")

def analyze_frontier_characteristics(result):
    """Analyze the characteristics of established frontiers.

    Args:
        result: Clustering result with frontiers in metadata
    """
    logger.info("🗺️ Analyzing frontier characteristics...")

    frontiers = result.metadata.get('frontiers', {})
    if not frontiers:
        logger.warning("No frontiers found in results")
        return

    print(f"\n" + "="*60)
    print("4D FRONTIER ANALYSIS")
    print("="*60)

    # Analyze frontiers by type
    for frontier_type, frontier_list in frontiers.items():
        if not frontier_list:
            continue

        print(f"\n📊 Frontier Type: {frontier_type.upper()}")
        print(f"  • Number of frontiers: {len(frontier_list)}")

        # Calculate statistics for this frontier type
        similarities = [f['similarity_score'] for f in frontier_list]
        cv_ratios = [f['cv_ratio'] for f in frontier_list]
        size_ratios = [f['size_ratio'] for f in frontier_list]

        print(f"  • Mean similarity: {np.mean(similarities):.3f} ± {np.std(similarities):.3f}")
        print(f"  • Mean CV ratio: {np.mean(cv_ratios):.3f} ± {np.std(cv_ratios):.3f}")
        print(f"  • Mean size ratio: {np.mean(size_ratios):.3f} ± {np.std(size_ratios):.3f}")

        # Count high-quality frontiers
        high_quality_count = sum(1 for f in frontier_list
                               if f['similarity_score'] > 0.7 and f['cv_ratio'] < 1.5 and f['size_ratio'] < 2.0)
        print(f"  • High-quality frontiers: {high_quality_count}/{len(frontier_list)} ({high_quality_count/len(frontier_list)*100:.1f}%)")

    # Overall frontier statistics
    all_frontiers = [f for frontier_list in frontiers.values() for f in frontier_list]
    if all_frontiers:
        all_similarities = [f['similarity_score'] for f in all_frontiers]
        all_cv_ratios = [f['cv_ratio'] for f in all_frontiers]
        all_size_ratios = [f['size_ratio'] for f in all_frontiers]

        print("\n📊 Overall Frontier Statistics:")
        print(f"  • Total frontiers: {len(all_frontiers)}")
        print(f"  • Mean similarity: {np.mean(all_similarities):.3f}")
        print(f"  • Mean CV ratio: {np.mean(all_cv_ratios):.3f}")
        print(f"  • Mean size ratio: {np.mean(all_size_ratios):.3f}")

    print(f"\n" + "="*60)

def compare_with_standard_clustering():
    """Compare enhanced clustering with standard clustering."""

    logger.info("🔄 Comparing enhanced vs standard clustering...")

    # Create sample data
    regime_data = create_sample_regime_data(800)

    # Standard clustering
    logger.info("📊 Running standard clustering...")
    standard_config = OptimalClusteringConfig()
    standard_config.min_silhouette_score = 0.3  # Lower threshold
    standard_config.min_davies_bouldin_score = 1.5  # Higher threshold (worse)

    # Use standard clustering
    standard_result = cluster_regimes_optimized(regime_data, standard_config)

    # Enhanced clustering
    logger.info("🚀 Running enhanced clustering...")
    enhanced_config = ENHANCED_CONFIG
    enhanced_result = cluster_regimes_enhanced(regime_data, enhanced_config)

    print(f"\n" + "="*60)
    print("CLUSTERING COMPARISON")
    print("="*60)

    print("📊 Standard Clustering Results:")
    print(f"  • Silhouette: {standard_result.quality_metrics.get('silhouette', 0.0):.3f}")
    print(f"  • Davies-Bouldin: {standard_result.quality_metrics.get('davies_bouldin', float('inf')):.3f}")
    print(f"  • Within-cluster CV: {standard_result.quality_metrics.get('mean_within_cluster_cv', 0.0):.3f}")
    print(f"  • Enhanced quality: {standard_result.quality_metrics.get('enhanced_quality_score', 0.0):.3f}")

    print("\n🚀 Enhanced Clustering Results:")
    print(f"  • Silhouette: {enhanced_result.quality_metrics.get('silhouette', 0.0):.3f}")
    print(f"  • Davies-Bouldin: {enhanced_result.quality_metrics.get('davies_bouldin', float('inf')):.3f}")
    print(f"  • Within-cluster CV: {enhanced_result.quality_metrics.get('mean_within_cluster_cv', 0.0):.3f}")
    print(f"  • Enhanced quality: {enhanced_result.quality_metrics.get('enhanced_quality_score', 0.0):.3f}")

    # Calculate improvements
    if enhanced_result.success and standard_result.success:
        silhouette_improvement = (enhanced_result.quality_metrics.get('silhouette', 0.0) -
                                standard_result.quality_metrics.get('silhouette', 0.0))
        db_improvement = (standard_result.quality_metrics.get('davies_bouldin', float('inf')) -
                         enhanced_result.quality_metrics.get('davies_bouldin', float('inf')))
        cv_improvement = (standard_result.quality_metrics.get('mean_within_cluster_cv', 0.0) -
                         enhanced_result.quality_metrics.get('mean_within_cluster_cv', 0.0))
        quality_improvement = (enhanced_result.quality_metrics.get('enhanced_quality_score', 0.0) -
                             standard_result.quality_metrics.get('enhanced_quality_score', 0.0))

        print("\n📈 Improvements:")
        print(f"  • Silhouette improvement: {silhouette_improvement:.3f} ({silhouette_improvement/standard_result.quality_metrics.get('silhouette', 0.0)*100:.1f}%)")
        print(f"  • Davies-Bouldin improvement: {db_improvement:.3f} ({db_improvement/max(1, standard_result.quality_metrics.get('davies_bouldin', 1.0))*100:.1f}%)")
        print(f"  • Within-cluster CV improvement: {cv_improvement:.3f} ({cv_improvement/max(0.01, standard_result.quality_metrics.get('mean_within_cluster_cv', 0.01))*100:.1f}%)")
        print(f"  • Enhanced quality improvement: {quality_improvement:.3f} ({quality_improvement/standard_result.quality_metrics.get('enhanced_quality_score', 0.01)*100:.1f}%)")

    print(f"\n" + "="*60)

if __name__ == "__main__":
    """Run all demonstrations."""

    print("🚀 ENHANCED CLUSTERING DEMONSTRATION")
    print("="*60)

    # Run main demonstration
    demonstrate_enhanced_clustering()

    # Show custom configuration
    demonstrate_custom_configuration()

    # Compare with standard clustering
    compare_with_standard_clustering()

    print("\n✅ All demonstrations completed!")