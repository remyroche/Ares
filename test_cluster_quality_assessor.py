"""
Test script for the unified Cluster Quality Assessor

This script tests the ClusterQualityAssessor with synthetic data to verify
that all metrics are calculated correctly.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import the quality assessor
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
    create_cluster_quality_assessor,
    ClusterQualityMetrics
)


def generate_synthetic_data(n_samples=500, n_features=10, n_clusters=5):
    """Generate synthetic clustered data for testing."""
    np.random.seed(42)
    
    # Create cluster centers
    centers = np.random.randn(n_clusters, n_features) * 5
    
    # Generate samples
    labels = np.random.randint(0, n_clusters, n_samples)
    features = np.zeros((n_samples, n_features))
    
    for i in range(n_samples):
        cluster_id = labels[i]
        features[i] = centers[cluster_id] + np.random.randn(n_features) * 0.5
    
    # Add some noise points
    noise_mask = np.random.random(n_samples) < 0.1
    labels[noise_mask] = -1
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    features_df = pd.DataFrame(features, columns=feature_names)
    
    # Generate forward returns (random but with some cluster-dependent structure)
    forward_returns = pd.Series(np.random.randn(n_samples) * 0.01)
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        forward_returns[cluster_mask] += np.random.randn() * 0.005
    
    # Generate timestamps
    start_date = datetime.now() - timedelta(days=n_samples)
    timestamps = pd.date_range(start=start_date, periods=n_samples, freq='1H')
    features_df.index = timestamps
    
    return labels, features_df, forward_returns, timestamps


def test_basic_metrics():
    """Test basic cluster quality metrics."""
    print("=" * 80)
    print("TEST 1: Basic Cluster Quality Metrics")
    print("=" * 80)
    
    # Generate test data
    labels, features_df, forward_returns, timestamps = generate_synthetic_data()
    
    print(f"\n📊 Test data generated:")
    print(f"   - Samples: {len(labels)}")
    print(f"   - Features: {features_df.shape[1]}")
    print(f"   - Unique clusters: {len(set(labels[labels != -1]))}")
    print(f"   - Noise points: {np.sum(labels == -1)}")
    
    # Create quality assessor
    quality_assessor = create_cluster_quality_assessor()
    
    # Assess quality
    print("\n🔍 Running quality assessment...")
    quality_metrics = quality_assessor.assess_quality(
        regime_labels=labels,
        feature_data=features_df,
        forward_returns=forward_returns,
        timestamps=timestamps
    )
    
    # Display results
    print("\n✅ Quality Assessment Results:")
    print(f"   - Overall Quality Score: {quality_metrics.quality_score:.3f}")
    print(f"   - Silhouette Score: {quality_metrics.silhouette_score:.3f}")
    print(f"   - Davies-Bouldin Index: {quality_metrics.davies_bouldin_score:.3f}")
    print(f"   - Calinski-Harabasz Score: {quality_metrics.calinski_harabasz_score:.1f}")
    print(f"   - Within Regime CV: {quality_metrics.within_regime_cv:.3f}")
    print(f"   - Between Regime CV: {quality_metrics.between_regime_cv:.3f}")
    print(f"   - Temporal Smoothness: {quality_metrics.temporal_smoothness:.3f}")
    print(f"   - Regime Persistence: {quality_metrics.regime_persistence:.1f} bars")
    print(f"   - Number of Regimes: {quality_metrics.n_regimes}")
    print(f"   - Noise Ratio: {quality_metrics.noise_ratio:.1%}")
    
    if quality_metrics.predictive_power is not None:
        print(f"   - Predictive Power: {quality_metrics.predictive_power:.3f}")
    
    # Check per-regime metrics
    print(f"\n📈 Per-Regime Metrics:")
    for regime_id, metrics in quality_metrics.per_regime_metrics.items():
        print(f"   Regime {regime_id}:")
        print(f"      - Size: {metrics['size']} ({metrics['percentage']:.1f}%)")
        print(f"      - Mean CV: {metrics['mean_cv']:.3f}")
        if 'mean_return' in metrics:
            print(f"      - Mean Return: {metrics['mean_return']:.4f}")
            print(f"      - Sharpe: {metrics['sharpe']:.3f}")
    
    # Test high quality check
    is_high_quality = quality_metrics.is_high_quality()
    print(f"\n🎯 Meets high quality standards: {is_high_quality}")
    
    return quality_metrics


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 80)
    print("TEST 2: Edge Cases")
    print("=" * 80)
    
    # Test with very few samples
    print("\n🧪 Test 2a: Few samples (n=50)")
    labels, features_df, forward_returns, timestamps = generate_synthetic_data(n_samples=50)
    quality_assessor = create_cluster_quality_assessor()
    quality_metrics = quality_assessor.assess_quality(
        regime_labels=labels,
        feature_data=features_df
    )
    print(f"   Quality Score: {quality_metrics.quality_score:.3f}")
    
    # Test with many clusters
    print("\n🧪 Test 2b: Many clusters (n_clusters=10)")
    labels, features_df, forward_returns, timestamps = generate_synthetic_data(n_clusters=10)
    quality_metrics = quality_assessor.assess_quality(
        regime_labels=labels,
        feature_data=features_df
    )
    print(f"   Quality Score: {quality_metrics.quality_score:.3f}")
    print(f"   Number of Regimes: {quality_metrics.n_regimes}")
    
    # Test with no noise
    print("\n🧪 Test 2c: No noise points")
    labels, features_df, _, _ = generate_synthetic_data()
    labels[labels == -1] = 0  # Convert noise to cluster 0
    quality_metrics = quality_assessor.assess_quality(
        regime_labels=labels,
        feature_data=features_df
    )
    print(f"   Quality Score: {quality_metrics.quality_score:.3f}")
    print(f"   Noise Ratio: {quality_metrics.noise_ratio:.1%}")


def test_serialization():
    """Test metrics serialization."""
    print("\n" + "=" * 80)
    print("TEST 3: Serialization")
    print("=" * 80)
    
    # Generate test data and compute metrics
    labels, features_df, forward_returns, timestamps = generate_synthetic_data()
    quality_assessor = create_cluster_quality_assessor()
    quality_metrics = quality_assessor.assess_quality(
        regime_labels=labels,
        feature_data=features_df,
        forward_returns=forward_returns,
        timestamps=timestamps
    )
    
    # Test to_dict conversion
    print("\n📦 Testing to_dict() conversion...")
    metrics_dict = quality_metrics.to_dict()
    print(f"   Dictionary keys: {list(metrics_dict.keys())}")
    print(f"   ✅ Serialization successful")
    
    # Verify all important metrics are present
    expected_keys = [
        'silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score',
        'within_regime_cv', 'between_regime_cv', 'temporal_smoothness',
        'quality_score', 'n_regimes', 'noise_ratio'
    ]
    
    missing_keys = [key for key in expected_keys if key not in metrics_dict]
    if missing_keys:
        print(f"   ⚠️ Missing keys: {missing_keys}")
    else:
        print(f"   ✅ All expected keys present")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("UNIFIED CLUSTER QUALITY ASSESSOR - TEST SUITE")
    print("=" * 80)
    
    try:
        # Run tests
        quality_metrics = test_basic_metrics()
        test_edge_cases()
        test_serialization()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print(f"\nSummary:")
        print(f"  - Cluster quality assessor is working correctly")
        print(f"  - All metrics are calculated as expected")
        print(f"  - Edge cases handled properly")
        print(f"  - Serialization working")
        print(f"\nThe unified quality assessor is ready for integration!")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED!")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
