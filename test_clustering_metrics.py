#!/usr/bin/env python3
"""
Test script for the new clustering quality metrics.

This script demonstrates how to use the Silhouette Score, Calinski-Harabasz Score,
Davies-Bouldin Index, and Gap Statistic for evaluating regime clustering quality.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

# Import the new clustering quality metrics
from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.clustering_quality_metrics import (
    ClusteringQualityMetrics, ClusteringQualityConfig, ClusteringQualityResult,
    create_clustering_quality_evaluator, quick_clustering_evaluation
)


def test_clustering_metrics():
    """Test the clustering quality metrics with synthetic data."""

    print("🧪 Testing Clustering Quality Metrics")
    print("=" * 50)

    # Generate synthetic clustering data
    print("📊 Generating synthetic clustering data...")
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=10,
                          random_state=42, cluster_std=1.0)

    # Convert to DataFrame for testing
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)

    print(f"✅ Generated {X.shape[0]} samples with {X.shape[1]} features")
    print(f"✅ True number of clusters: {len(np.unique(y_true))}")

    # Test 1: Quick evaluation
    print("\n🔍 Test 1: Quick clustering evaluation")
    print("-" * 30)

    quick_result = quick_clustering_evaluation(df, y_true)
    print(f"Silhouette Score: {quick_result.silhouette_score:.4f}")
    print(f"Calinski-Harabasz Score: {quick_result.calinski_harabasz_score:.4f}")
    print(f"Davies-Bouldin Index: {quick_result.davies_bouldin_index:.4f}")
    print(f"Overall Quality Score: {quick_result.overall_quality_score:.4f}")
    print(f"Quality Interpretation: {quick_result.quality_interpretation}")

    # Test 2: Custom configuration
    print("\n🔧 Test 2: Custom configuration")
    print("-" * 30)

    custom_config = ClusteringQualityConfig(
        compute_gap_statistic=True,
        gap_n_clusters_range=(2, 8),
        gap_n_bootstraps=5,  # Reduced for faster testing
        enable_feature_scaling=True,
        feature_scaling_method="standard"
    )

    evaluator = create_clustering_quality_evaluator(custom_config)
    custom_result = evaluator.evaluate_clustering_quality(df, y_true)

    print(f"Silhouette Score: {custom_result.silhouette_score:.4f}")
    print(f"Calinski-Harabasz Score: {custom_result.calinski_harabasz_score:.4f}")
    print(f"Davies-Bouldin Index: {custom_result.davies_bouldin_index:.4f}")
    print(f"Gap Statistic Optimal Clusters: {custom_result.gap_optimal_clusters}")
    print(f"Overall Quality Score: {custom_result.overall_quality_score:.4f}")
    print(f"Quality Interpretation: {custom_result.quality_interpretation}")

    # Test 3: Different cluster labels (simulating regime detection)
    print("\n🎭 Test 3: Different cluster assignments")
    print("-" * 30)

    # Simulate different clustering results
    y_regime_1 = y_true.copy()
    y_regime_2 = np.random.randint(0, 3, len(y_true))  # 3 clusters
    y_regime_3 = np.random.randint(0, 6, len(y_true))  # 6 clusters

    for i, (name, labels) in enumerate([
        ("Ground Truth (4 clusters)", y_true),
        ("Regime Detection (3 clusters)", y_regime_2),
        ("Regime Detection (6 clusters)", y_regime_3)
    ], 1):
        result = quick_clustering_evaluation(df, labels)
        n_clusters = len(np.unique(labels))
        print(f"\n{name} ({n_clusters} clusters):")
        print(f"  Silhouette Score: {result.silhouette_score:.4f}")
        print(f"  Calinski-Harabasz Score: {result.calinski_harabasz_score:.4f}")
        print(f"  Davies-Bouldin Index: {result.davies_bouldin_index:.4f}")
        print(f"  Overall Quality: {result.overall_quality_score:.4f} ({result.quality_interpretation})")

    # Test 4: Market data simulation
    print("\n📈 Test 4: Market data simulation")
    print("-" * 30)

    # Simulate market features
    np.random.seed(42)
    n_samples = 500
    n_features = 8

    # Generate market-like features
    market_features = np.random.randn(n_samples, n_features)

    # Add some structure to simulate market regimes
    regime_mask = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.35, 0.25])

    # Add regime-specific characteristics
    for regime in np.unique(regime_mask):
        mask = regime_mask == regime
        market_features[mask] += np.random.randn(n_features) * 0.5

    market_df = pd.DataFrame(market_features,
                           columns=[f'market_feature_{i}' for i in range(n_features)])

    # Simulate regime predictions
    simulated_regime_labels = regime_mask.copy()

    market_result = quick_clustering_evaluation(market_df, simulated_regime_labels)

    print("Market Regime Analysis:")
    print(f"  Samples: {market_result.n_samples}")
    print(f"  Features: {market_result.n_features}")
    print(f"  Regimes: {market_result.n_clusters}")
    print(f"  Silhouette Score: {market_result.silhouette_score:.4f}")
    print(f"  Calinski-Harabasz Score: {market_result.calinski_harabasz_score:.4f}")
    print(f"  Davies-Bouldin Index: {market_result.davies_bouldin_index:.4f}")
    print(f"  Overall Quality: {market_result.overall_quality_score:.4f} ({market_result.quality_interpretation})")

    # Test 5: Economic significance evaluation
    print("\n💰 Test 5: Economic significance evaluation")
    print("-" * 40)

    # Generate market data with realistic price movements
    np.random.seed(42)
    n_samples = 500
    initial_price = 100.0

    # Generate realistic price series with different regimes
    prices = [initial_price]
    regime_returns = {
        0: 0.001,  # Bullish regime
        1: -0.0005,  # Bearish regime
        2: 0.0002   # Sideways regime
    }

    for i in range(1, n_samples):
        current_regime = simulated_regime_labels[i]
        # Add some noise and regime-specific drift
        noise = np.random.normal(0, 0.01)
        drift = regime_returns.get(current_regime, 0.0)
        new_price = prices[-1] * (1 + drift + noise)
        prices.append(new_price)

    market_prices = np.array(prices)

    # Evaluate with economic significance
    economic_result = quick_clustering_evaluation(
        market_df, simulated_regime_labels,
        timestamps=np.arange(len(market_df)),
        market_data=np.column_stack([market_prices, market_prices, market_prices, market_prices, np.ones(len(market_prices))])  # OHLCV format
    )

    print("Economic Analysis:")
    print(f"  Economic Quality Score: {economic_result.economic_quality_score:.4f} ({economic_result.economic_interpretation})")
    print(f"  Economically Consistent Regimes: {economic_result.economically_significant_regimes}/{economic_result.n_clusters}")
    print(f"  Highly Consistent Regimes: {economic_result.economically_viable_regimes}/{economic_result.n_clusters}")

    # Show detailed economic profiles for each regime
    if economic_result.regime_economic_profiles:
        print("\nRegime Economic Consistency Profiles:")
        for regime_id, profile in economic_result.regime_economic_profiles.items():
            if profile:  # Only show if profile is not empty
                print(f"\n{regime_id}:")
                print(f"  Return Consistency: {profile.get('avg_return', 0):.4f} ± {profile.get('return_std', 0):.4f}")
                print(f"  Volatility Stability: {profile.get('volatility', 0):.4f}")
                print(f"  Sharpe Ratio: {profile.get('sharpe_ratio', 0):.4f}")
                print(f"  Max Drawdown: {profile.get('max_drawdown', 0):.4f}")
                print(f"  Regime Size: {profile.get('regime_size', 0)} samples")

    print("\n✅ All clustering quality metrics tests completed!")
    print("\n📋 Summary of Available Metrics:")
    print("  • Silhouette Score: Measures cluster separation and cohesion")
    print("  • Calinski-Harabasz Score: Ratio of between-cluster to within-cluster variance")
    print("  • Davies-Bouldin Index: Average similarity ratio of each cluster")
    print("  • Gap Statistic: Optimal number of clusters using reference distributions")
    print("  • Additional metrics: Inertia, separation index, cohesion index")
    print("\n💰 Economic Distinctness Metrics:")
    print("  • Economic Distinctness: How different regimes are from each other economically")
    print("  • Within-Regime Consistency: How stable economic characteristics are within each regime")
    print("  • Return Consistency: Stability of returns within regimes")
    print("  • Volatility Stability: Consistency of volatility within regimes")
    print("  • Economic Quality Score: Overall economic distinctness and consistency")
    print("  • Consistent/Highly Consistent Regimes: Count of economically stable regimes")


if __name__ == "__main__":
    test_clustering_metrics()
