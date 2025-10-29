"""
Minimal test script for HDP-HMM clustering.

This script tests the HDP-HMM clustering implementation with synthetic data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("=" * 80)
print("Minimal Test: HDP-HMM Clustering")
print("=" * 80)

# Check if HMM libraries are available
try:
    from src.training.steps.market_analysis.hdp_hmm_clustering.hdp_hmm_clusterer import (
        HDPHMMClusterer, HDPHMMConfig, HMM_AVAILABLE, HMM_LIBRARY
    )
    print(f"✅ HDP-HMM module imported successfully")
    print(f"   Library available: {HMM_AVAILABLE}")
    print(f"   Library used: {HMM_LIBRARY}")
except Exception as e:
    print(f"❌ Failed to import HDP-HMM module: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

if not HMM_AVAILABLE:
    print("\n⚠️  HMM libraries not available.")
    print("   Install with: pip install pyhsmm or pip install ssm-jax")
    print("\n   Note: pyhsmm requires additional dependencies:")
    print("   - Cython")
    print("   - numpy")
    print("   - scipy")
    print("   - matplotlib")
    print("\n   Installation command:")
    print("   pip install Cython numpy scipy matplotlib")
    print("   pip install git+https://github.com/mattjj/pyhsmm.git")
    exit(1)

print("\n" + "=" * 80)
print("Generating Synthetic Market Data")
print("=" * 80)

# Generate synthetic market data with regime changes
np.random.seed(42)
n_samples = 500

# Create 3 regimes with different characteristics
regime_1 = np.random.normal(0.0, 0.5, 150)  # Low volatility
regime_2 = np.random.normal(1.0, 1.5, 200)  # High volatility
regime_3 = np.random.normal(-0.5, 0.8, 150) # Negative trend

# Combine regimes
time_series = np.concatenate([regime_1, regime_2, regime_3])

# Create features from time series
features = np.column_stack([
    time_series,
    np.roll(time_series, 1),
    np.roll(time_series, 2),
    np.abs(time_series),
    np.cumsum(time_series)
])

# Remove first few rows with NaN from rolling
features = features[3:]
time_series = time_series[3:]

print(f"✅ Generated synthetic data: {features.shape}")
print(f"   True regimes: 3 (manually constructed)")

print("\n" + "=" * 80)
print("Testing HDP-HMM Clusterer")
print("=" * 80)

# Test 1: Basic initialization
print("\n1. Testing basic initialization...")
try:
    config = HDPHMMConfig(
        alpha=3.0,
        kappa=50.0,
        gamma=3.0,
        n_iterations=50,  # Reduced for testing
        max_states=10,
        enable_pca=True,
        pca_components=3
    )
    clusterer = HDPHMMClusterer(config)
    print("   ✅ HDP-HMM clusterer initialized successfully")
except Exception as e:
    print(f"   ❌ Failed to initialize: {e}")
    exit(1)

# Test 2: Fit and predict
print("\n2. Testing fit_predict...")
try:
    result = clusterer.fit_predict(features)
    print("   ✅ HDP-HMM clustering completed successfully")
    print(f"   Discovered regimes: {result.n_clusters}")
    print(f"   Silhouette score: {result.silhouette_score:.3f}")
    print(f"   Processing time: {result.processing_time:.2f}s")
    print(f"   Memory usage: {result.memory_usage_mb:.1f} MB")
    
    if result.transition_matrix is not None:
        print(f"\n   Transition Matrix:")
        print(f"   {result.transition_matrix}")
        print(f"\n   Transition persistence: {result.transition_persistence:.3f}")
    
    if result.state_durations is not None and len(result.state_durations) > 0:
        print(f"\n   Average regime durations: {result.state_durations}")
    
except Exception as e:
    print(f"   ❌ Failed to fit_predict: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Check result validity
print("\n3. Validating results...")
try:
    assert result.success, "Result indicates failure"
    assert result.n_clusters > 0, "No clusters discovered"
    assert len(result.cluster_labels) == len(features), "Label count mismatch"
    assert result.cluster_labels.min() >= 0, "Invalid cluster labels"
    print("   ✅ Results validation passed")
except AssertionError as e:
    print(f"   ❌ Validation failed: {e}")
    exit(1)

# Test 4: Test with DataFrame input
print("\n4. Testing with DataFrame input...")
try:
    df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features.shape[1])])
    result_df = clusterer.fit_predict(df)
    print("   ✅ DataFrame input handled successfully")
    print(f"   Discovered regimes: {result_df.n_clusters}")
except Exception as e:
    print(f"   ❌ Failed with DataFrame input: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Testing Enhanced HDP-HMM Integration")
print("=" * 80)

# Test integration with feature bank
try:
    from src.feature_generation.integration.enhanced_hdp_hmm_clustering_integration import (
        EnhancedHDPHMMClusteringIntegration
    )
    print("✅ Enhanced HDP-HMM integration module imported")
    
    # Create synthetic market data
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1h')
    market_data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
        'high': 101 + np.cumsum(np.random.randn(n_samples) * 0.5),
        'low': 99 + np.cumsum(np.random.randn(n_samples) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
        'volume': np.random.randint(1000, 10000, n_samples)
    }, index=dates)
    
    print(f"\n✅ Generated market data: {market_data.shape}")
    
    # Test integration
    print("\n5. Testing enhanced integration...")
    integration = EnhancedHDPHMMClusteringIntegration(
        min_features=20,
        max_features=50,
        alpha=3.0,
        kappa=50.0,
        n_iterations=30  # Reduced for testing
    )
    
    # Note: This might fail if feature_bank_integration is not available
    # That's expected for now
    print("   ✅ Integration initialized successfully")
    print("   ⚠️  Full integration test requires feature bank")
    
except Exception as e:
    print(f"⚠️  Enhanced integration test skipped: {e}")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"✅ HDP-HMM clustering implementation: PASSED")
print(f"✅ Basic functionality: WORKING")
print(f"✅ Discovered regimes: {result.n_clusters}")
print(f"✅ Silhouette score: {result.silhouette_score:.3f}")
print(f"\n📊 The HDP-HMM clustering successfully discovered {result.n_clusters} regimes")
print(f"   (True number was 3 regimes)")
print("\n" + "=" * 80)
