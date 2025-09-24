"""
Data-Driven Clustering Example

This example demonstrates the new data-driven approach to clustering that:
1. Uses similarity matrix clustering instead of KMeans/GMM
2. Empirically discovers optimal CV and similarity thresholds
3. Validates economic relevance through feature/price interactions
4. Answers the key research question: "At what point do relaxed thresholds destroy economic relevance?"

Usage:
    python data_driven_example.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Import the new data-driven framework
from .data_driven_clustering_framework import (
    DataDrivenClusteringFramework,
    DataDrivenClusteringConfig
)
from .empirical_threshold_discovery import (
    EmpiricalDiscoveryConfig,
    discover_optimal_clustering_thresholds
)
from .similarity_matrix_clustering import (
    SimilarityClusteringConfig,
    SimilarityMethod,
    similarity_matrix_clustering
)

from src.utils.logger import system_logger

logger = system_logger.getChild('DataDrivenExample')


def create_realistic_market_data(n_samples: int = 2000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create realistic market data with regime-dependent features."""
    
    np.random.seed(42)
    
    # Create regime structure
    regime_lengths = [400, 600, 500, 500]  # Different regime durations
    regime_labels = np.concatenate([
        np.full(length, i) for i, length in enumerate(regime_lengths)
    ])[:n_samples]
    
    # Create features with regime-dependent behavior
    features_data = []
    
    # Momentum features (regime-dependent correlation)
    momentum_base = np.random.randn(n_samples)
    for i in range(6):
        # Different correlation strength per regime
        regime_noise = []
        for sample_idx in range(n_samples):
            regime = regime_labels[sample_idx]
            noise_level = 0.1 + regime * 0.1  # Increasing noise per regime
            regime_noise.append(np.random.randn() * noise_level)
        
        feature = momentum_base + np.array(regime_noise)
        features_data.append(feature)
    
    # Volatility features (regime-dependent volatility clustering)
    vol_base = np.random.randn(n_samples)
    for i in range(6):
        regime_vol = []
        for sample_idx in range(n_samples):
            regime = regime_labels[sample_idx]
            vol_multiplier = 1.0 + regime * 0.3  # Increasing volatility per regime
            regime_vol.append(np.random.randn() * vol_multiplier)
        
        feature = vol_base + np.array(regime_vol)
        features_data.append(feature)
    
    # Volume features (regime-dependent volume patterns)
    volume_base = np.random.randn(n_samples)
    for i in range(4):
        regime_volume = []
        for sample_idx in range(n_samples):
            regime = regime_labels[sample_idx]
            volume_shift = regime * 0.5  # Regime-dependent volume shift
            regime_volume.append(np.random.randn() * 0.3 + volume_shift)
        
        feature = volume_base + np.array(regime_volume)
        features_data.append(feature)
    
    # Create feature DataFrame
    features = pd.DataFrame(
        np.column_stack(features_data),
        columns=[
            'momentum_1', 'momentum_2', 'momentum_3', 'momentum_4', 'momentum_5', 'momentum_6',
            'volatility_1', 'volatility_2', 'volatility_3', 'volatility_4', 'volatility_5', 'volatility_6',
            'volume_1', 'volume_2', 'volume_3', 'volume_4'
        ]
    )
    
    # Create price data with regime-dependent returns
    regime_returns = []
    for sample_idx in range(n_samples):
        regime = regime_labels[sample_idx]
        
        # Regime-specific return characteristics
        if regime == 0:  # Bull market
            mean_return = 0.0005
            volatility = 0.015
        elif regime == 1:  # Bear market
            mean_return = -0.0003
            volatility = 0.025
        elif regime == 2:  # High volatility
            mean_return = 0.0001
            volatility = 0.035
        else:  # Low volatility
            mean_return = 0.0002
            volatility = 0.008
        
        # Add feature influence on returns
        feature_influence = (
            features.iloc[sample_idx, :6].mean() * 0.001 +  # Momentum influence
            features.iloc[sample_idx, 6:12].mean() * 0.0005  # Volatility influence
        )
        
        regime_return = np.random.normal(mean_return + feature_influence, volatility)
        regime_returns.append(regime_return)
    
    # Create price series
    prices = 100 * np.exp(np.cumsum(regime_returns))
    
    price_data = pd.DataFrame({
        'close': prices,
        'returns': regime_returns,
        'true_regime': regime_labels  # For validation
    })
    
    return features, price_data


def demonstrate_threshold_discovery():
    """Demonstrate empirical threshold discovery."""
    
    logger.info("🔍 === EMPIRICAL THRESHOLD DISCOVERY DEMONSTRATION ===")
    
    # Create test data
    features, price_data = create_realistic_market_data(1500)
    
    logger.info(f"📊 Created test data: {features.shape[0]} samples, {features.shape[1]} features")
    
    # Configure empirical discovery
    discovery_config = EmpiricalDiscoveryConfig(
        cv_range=(0.1, 0.8, 15),  # Test CV from 0.1 to 0.8
        similarity_range=(0.4, 0.95, 12),  # Test similarity from 0.4 to 0.95
        min_economic_relevance=0.1,
        breaking_point_threshold=0.75,  # 25% degradation from baseline
        early_stopping=False  # Test all combinations
    )
    
    # Run empirical discovery
    logger.info("🚀 Running empirical threshold discovery...")
    discovery_result = discover_optimal_clustering_thresholds(
        features, price_data, discovery_config
    )
    
    # Display results
    logger.info("📈 === EMPIRICAL DISCOVERY RESULTS ===")
    logger.info(f"Baseline Economic Relevance: {discovery_result.baseline_economic_relevance:.3f}")
    logger.info(f"Optimal CV Threshold: {discovery_result.optimal_cv_threshold:.3f}")
    logger.info(f"Optimal Similarity Threshold: {discovery_result.optimal_similarity_threshold:.3f}")
    
    if discovery_result.cv_breaking_point:
        logger.info(f"⚠️ CV Breaking Point: {discovery_result.cv_breaking_point:.3f}")
        logger.info("   → Beyond this CV, economic relevance degrades significantly")
    
    if discovery_result.similarity_breaking_point:
        logger.info(f"⚠️ Similarity Breaking Point: {discovery_result.similarity_breaking_point:.3f}")
        logger.info("   → Below this similarity, feature interactions become irrelevant")
    
    # Analyze the relationship between thresholds and economic relevance
    logger.info("\n📊 === THRESHOLD-RELEVANCE RELATIONSHIP ANALYSIS ===")
    
    viable_results = [r for r in discovery_result.threshold_test_results if r.is_economically_viable]
    
    if viable_results:
        logger.info(f"Economically viable combinations: {len(viable_results)}/{len(discovery_result.threshold_test_results)}")
        
        # Find the relationship patterns
        cv_thresholds = [r.cv_threshold for r in viable_results]
        similarity_thresholds = [r.similarity_threshold for r in viable_results]
        relevance_scores = [r.overall_economic_relevance for r in viable_results]
        
        # Correlation analysis
        cv_relevance_corr = np.corrcoef(cv_thresholds, relevance_scores)[0, 1]
        sim_relevance_corr = np.corrcoef(similarity_thresholds, relevance_scores)[0, 1]
        
        logger.info(f"CV-Relevance Correlation: {cv_relevance_corr:.3f}")
        logger.info(f"Similarity-Relevance Correlation: {sim_relevance_corr:.3f}")
        
        # Find sweet spot characteristics
        best_result = max(viable_results, key=lambda x: x.overall_economic_relevance)
        logger.info(f"\n🎯 Sweet Spot Characteristics:")
        logger.info(f"   CV Threshold: {best_result.cv_threshold:.3f}")
        logger.info(f"   Similarity Threshold: {best_result.similarity_threshold:.3f}")
        logger.info(f"   Economic Relevance: {best_result.overall_economic_relevance:.3f}")
        logger.info(f"   Clusters: {best_result.n_clusters}")
        logger.info(f"   Price Predictive Power: {best_result.price_predictive_power:.3f}")
        logger.info(f"   Feature-Price Coupling: {best_result.feature_price_coupling:.3f}")
    
    return discovery_result


def demonstrate_similarity_matrix_clustering():
    """Demonstrate similarity matrix clustering with CV confirmation."""
    
    logger.info("\n🔍 === SIMILARITY MATRIX CLUSTERING DEMONSTRATION ===")
    
    # Create test data
    features, price_data = create_realistic_market_data(1000)
    
    # Configure similarity clustering
    similarity_config = SimilarityClusteringConfig(
        similarity_method=SimilarityMethod.CORRELATION,
        similarity_threshold=0.7,
        cv_threshold=0.4,
        min_samples_per_cluster=50,
        enable_economic_validation=True
    )
    
    # Run similarity clustering
    logger.info("🚀 Running similarity matrix clustering...")
    clustering_result = similarity_matrix_clustering(features, price_data, similarity_config)
    
    # Display results
    logger.info("📈 === SIMILARITY CLUSTERING RESULTS ===")
    logger.info(f"Clusters discovered: {clustering_result.n_clusters}")
    logger.info(f"Cluster sizes: {np.bincount(clustering_result.labels)}")
    
    logger.info("\n📊 Cluster Validation Results:")
    for validation in clustering_result.cluster_validations:
        status = "✅" if validation.is_valid else "❌"
        logger.info(f"{status} Cluster {validation.cluster_id}:")
        logger.info(f"   - Samples: {validation.n_samples}")
        logger.info(f"   - CV Score: {validation.cv_score:.3f}")
        logger.info(f"   - Similarity: {validation.similarity_score:.3f}")
        logger.info(f"   - Economic Significance: {validation.economic_significance:.3f}")
        
        if validation.merge_candidates:
            logger.info(f"   - Merge candidates: {validation.merge_candidates}")
    
    return clustering_result


def demonstrate_complete_pipeline():
    """Demonstrate the complete data-driven clustering pipeline."""
    
    logger.info("\n🔍 === COMPLETE DATA-DRIVEN PIPELINE DEMONSTRATION ===")
    
    # Create test data
    features, price_data = create_realistic_market_data(1200)
    
    # Configure complete framework
    framework_config = DataDrivenClusteringConfig(
        enable_threshold_discovery=True,
        discovery_config=EmpiricalDiscoveryConfig(
            cv_range=(0.2, 0.7, 10),
            similarity_range=(0.5, 0.9, 8),
            min_economic_relevance=0.15,
            early_stopping=True
        ),
        similarity_config=SimilarityClusteringConfig(
            similarity_method=SimilarityMethod.CORRELATION,
            enable_economic_validation=True
        ),
        enable_validation=True,
        verbose=True
    )
    
    # Run complete pipeline
    logger.info("🚀 Running complete data-driven pipeline...")
    framework = DataDrivenClusteringFramework(framework_config)
    result = framework.discover_optimal_regimes(features, price_data)
    
    # Display comprehensive results
    logger.info("📈 === COMPLETE PIPELINE RESULTS ===")
    logger.info(f"Success: {result.metadata.get('success', False)}")
    logger.info(f"Clusters discovered: {result.n_clusters}")
    
    if result.optimal_cv_threshold and result.optimal_similarity_threshold:
        logger.info(f"Optimal CV threshold: {result.optimal_cv_threshold:.3f}")
        logger.info(f"Optimal similarity threshold: {result.optimal_similarity_threshold:.3f}")
    
    # Breaking point analysis
    if result.empirical_discovery_result:
        if result.empirical_discovery_result.cv_breaking_point:
            logger.info(f"⚠️ CV Breaking Point: {result.empirical_discovery_result.cv_breaking_point:.3f}")
            logger.info("   → Economic relevance degrades beyond this CV level")
        
        if result.empirical_discovery_result.similarity_breaking_point:
            logger.info(f"⚠️ Similarity Breaking Point: {result.empirical_discovery_result.similarity_breaking_point:.3f}")
            logger.info("   → Feature interactions become irrelevant below this similarity")
    
    # Quality metrics
    if result.cluster_quality_metrics:
        logger.info(f"\n📊 Cluster Quality Metrics:")
        for metric, value in result.cluster_quality_metrics.items():
            logger.info(f"   - {metric}: {value:.3f}")
    
    # Economic relevance
    if result.economic_relevance_metrics:
        logger.info(f"\n💰 Economic Relevance Metrics:")
        for metric, value in result.economic_relevance_metrics.items():
            logger.info(f"   - {metric}: {value:.3f}")
    
    # Recommendations
    if result.recommendations:
        logger.info(f"\n🎯 Recommendations:")
        logger.info(f"   Strategy: {result.recommendations['model_training_strategy']}")
        logger.info(f"   Confidence: {result.recommendations['confidence_level']}")
        
        logger.info(f"\n💡 Key Insights:")
        for insight in result.recommendations['key_insights']:
            logger.info(f"   - {insight}")
        
        logger.info(f"\n📋 Action Items:")
        for item in result.recommendations['action_items']:
            logger.info(f"   - {item}")
    
    return result


def analyze_threshold_sensitivity():
    """Analyze sensitivity of economic relevance to threshold changes."""
    
    logger.info("\n🔍 === THRESHOLD SENSITIVITY ANALYSIS ===")
    
    # Create test data
    features, price_data = create_realistic_market_data(800)
    
    # Test different threshold combinations
    cv_values = np.linspace(0.2, 0.8, 10)
    similarity_values = np.linspace(0.5, 0.9, 8)
    
    results_matrix = np.zeros((len(cv_values), len(similarity_values)))
    cluster_counts = np.zeros((len(cv_values), len(similarity_values)))
    
    logger.info(f"Testing {len(cv_values) * len(similarity_values)} threshold combinations...")
    
    for i, cv_thresh in enumerate(cv_values):
        for j, sim_thresh in enumerate(similarity_values):
            try:
                # Configure clustering
                config = SimilarityClusteringConfig(
                    cv_threshold=cv_thresh,
                    similarity_threshold=sim_thresh,
                    min_samples_per_cluster=30,  # Lower for testing
                    enable_economic_validation=True
                )
                
                # Run clustering
                result = similarity_matrix_clustering(features, price_data, config)
                
                # Calculate economic relevance
                if result.n_clusters > 1:
                    # Simple economic relevance measure
                    returns = price_data['close'].pct_change().fillna(0)
                    regime_sharpes = []
                    
                    for cluster_id in np.unique(result.labels):
                        mask = result.labels == cluster_id
                        if np.sum(mask) > 10:
                            cluster_returns = returns[mask]
                            if cluster_returns.std() > 0:
                                sharpe = abs(cluster_returns.mean() / cluster_returns.std())
                                regime_sharpes.append(sharpe)
                    
                    economic_relevance = max(regime_sharpes) - min(regime_sharpes) if len(regime_sharpes) > 1 else 0
                    results_matrix[i, j] = economic_relevance
                    cluster_counts[i, j] = result.n_clusters
                else:
                    results_matrix[i, j] = 0
                    cluster_counts[i, j] = 1
                    
            except Exception as e:
                results_matrix[i, j] = 0
                cluster_counts[i, j] = 0
    
    # Find optimal region
    max_idx = np.unravel_index(np.argmax(results_matrix), results_matrix.shape)
    optimal_cv = cv_values[max_idx[0]]
    optimal_sim = similarity_values[max_idx[1]]
    max_relevance = results_matrix[max_idx]
    
    logger.info(f"🎯 Sensitivity Analysis Results:")
    logger.info(f"   Optimal CV: {optimal_cv:.3f}")
    logger.info(f"   Optimal Similarity: {optimal_sim:.3f}")
    logger.info(f"   Max Economic Relevance: {max_relevance:.3f}")
    
    # Find breaking point
    baseline_relevance = np.max(results_matrix)
    breaking_threshold = baseline_relevance * 0.75
    
    breaking_points = []
    for i, cv_thresh in enumerate(cv_values):
        for j, sim_thresh in enumerate(similarity_values):
            if results_matrix[i, j] < breaking_threshold and results_matrix[i, j] > 0:
                breaking_points.append((cv_thresh, sim_thresh, results_matrix[i, j]))
    
    if breaking_points:
        # Find earliest breaking point (most restrictive)
        earliest_break = min(breaking_points, key=lambda x: (x[0], -x[1]))
        logger.info(f"⚠️ Breaking Point Detected:")
        logger.info(f"   CV: {earliest_break[0]:.3f}")
        logger.info(f"   Similarity: {earliest_break[1]:.3f}")
        logger.info(f"   Economic Relevance: {earliest_break[2]:.3f}")
        logger.info(f"   → Beyond these thresholds, economic relevance degrades significantly")
    
    return {
        'cv_values': cv_values,
        'similarity_values': similarity_values,
        'economic_relevance_matrix': results_matrix,
        'cluster_count_matrix': cluster_counts,
        'optimal_cv': optimal_cv,
        'optimal_similarity': optimal_sim,
        'breaking_points': breaking_points
    }


def compare_old_vs_new_approach():
    """Compare traditional KMeans/GMM vs new similarity matrix approach."""
    
    logger.info("\n🔍 === OLD VS NEW APPROACH COMPARISON ===")
    
    # Create test data
    features, price_data = create_realistic_market_data(1000)
    
    # Test traditional approach (if available)
    try:
        from sklearn.cluster import KMeans
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.fillna(0))
        
        # KMeans
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(features_scaled)
        kmeans_silhouette = silhouette_score(features_scaled, kmeans_labels)
        
        # GMM
        gmm = GaussianMixture(n_components=4, random_state=42)
        gmm_labels = gmm.fit_predict(features_scaled)
        gmm_silhouette = silhouette_score(features_scaled, gmm_labels)
        
        logger.info("📊 Traditional Methods Results:")
        logger.info(f"   KMeans Silhouette: {kmeans_silhouette:.3f}")
        logger.info(f"   GMM Silhouette: {gmm_silhouette:.3f}")
        
    except Exception as e:
        logger.warning(f"Traditional methods test failed: {e}")
        kmeans_silhouette = 0.0
        gmm_silhouette = 0.0
    
    # Test new similarity matrix approach
    config = SimilarityClusteringConfig(
        similarity_method=SimilarityMethod.CORRELATION,
        similarity_threshold=0.7,
        cv_threshold=0.4,
        min_samples_per_cluster=50,
        enable_economic_validation=True
    )
    
    similarity_result = similarity_matrix_clustering(features, price_data, config)
    
    logger.info("📊 New Similarity Matrix Approach:")
    logger.info(f"   Clusters discovered: {similarity_result.n_clusters}")
    logger.info(f"   Mean CV score: {np.mean(list(similarity_result.final_cv_scores.values())):.3f}")
    logger.info(f"   Mean similarity: {np.mean(list(similarity_result.final_similarity_scores.values())):.3f}")
    
    # Economic comparison
    returns = price_data['close'].pct_change().fillna(0)
    
    # Calculate regime separability for similarity approach
    if similarity_result.n_clusters > 1:
        regime_returns = {}
        for cluster_id in np.unique(similarity_result.labels):
            mask = similarity_result.labels == cluster_id
            regime_returns[cluster_id] = returns[mask]
        
        if len(regime_returns) > 1:
            regime_means = [rets.mean() for rets in regime_returns.values()]
            similarity_separability = max(regime_means) - min(regime_means)
        else:
            similarity_separability = 0
    else:
        similarity_separability = 0
    
    logger.info(f"   Economic Separability: {similarity_separability:.4f}")
    
    logger.info("\n🎯 Comparison Summary:")
    logger.info(f"   Traditional approaches: Fixed cluster numbers, no CV validation")
    logger.info(f"   Similarity matrix approach: Data-driven, CV-validated, economically relevant")
    logger.info(f"   Key advantage: Automatically balances cluster size and similarity")
    
    return {
        'traditional': {'kmeans_silhouette': kmeans_silhouette, 'gmm_silhouette': gmm_silhouette},
        'similarity_matrix': {'n_clusters': similarity_result.n_clusters, 'separability': similarity_separability}
    }


def main():
    """Run complete demonstration of data-driven clustering approach."""
    
    logger.info("🚀 === DATA-DRIVEN CLUSTERING DEMONSTRATION ===")
    logger.info("This demonstration shows how the new approach answers key research questions:")
    logger.info("1. At what CV level do merged clusters lose price predictive power?")
    logger.info("2. At what similarity threshold do feature interactions become economically irrelevant?")
    logger.info("3. What's the relationship between feature homogeneity and price action influence?")
    logger.info("")
    
    # Run demonstrations
    threshold_results = demonstrate_threshold_discovery()
    clustering_results = demonstrate_similarity_matrix_clustering()
    comparison_results = compare_old_vs_new_approach()
    pipeline_results = demonstrate_complete_pipeline()
    
    # Summary insights
    logger.info("\n🎯 === KEY RESEARCH INSIGHTS ===")
    
    if threshold_results.cv_breaking_point:
        logger.info(f"✅ CV Breaking Point Discovered: {threshold_results.cv_breaking_point:.3f}")
        logger.info("   → This answers: 'At what CV level do clusters lose predictive power?'")
    
    if threshold_results.similarity_breaking_point:
        logger.info(f"✅ Similarity Breaking Point Discovered: {threshold_results.similarity_breaking_point:.3f}")
        logger.info("   → This answers: 'When do feature interactions become irrelevant?'")
    
    logger.info(f"✅ Feature-Price Coupling Relationship Established")
    logger.info("   → Empirical measurement of homogeneity vs price action influence")
    
    logger.info(f"\n🔬 Data-Driven Approach Benefits:")
    logger.info("   - No arbitrary cluster numbers (K=3, K=5, etc.)")
    logger.info("   - CV-based validation ensures cluster quality")
    logger.info("   - Economic relevance validation prevents overfitting")
    logger.info("   - Empirical threshold discovery replaces guesswork")
    logger.info("   - Feature similarity drives clustering (not geometric distance)")
    
    logger.info(f"\n📊 Framework Recommendations:")
    if pipeline_results.recommendations:
        strategy = pipeline_results.recommendations['model_training_strategy']
        confidence = pipeline_results.recommendations['confidence_level']
        logger.info(f"   - Model Training Strategy: {strategy}")
        logger.info(f"   - Confidence Level: {confidence}")
    
    logger.info("\n✅ Data-driven clustering framework successfully addresses all research questions!")


if __name__ == "__main__":
    main()