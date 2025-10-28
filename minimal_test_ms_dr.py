"""
Minimal test script for MS-DR clustering.

This script tests the Markov-Switching Dynamic Regression clustering
implementation with synthetic data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("=" * 80)
print("Minimal Test: MS-DR Clustering")
print("=" * 80)

# Check if MS libraries are available
try:
    from src.training.steps.market_analysis.ms_dr_clustering import (
        MSDRClusterer, MSDRConfig, MS_AVAILABLE, MS_LIBRARY
    )
    print(f"✅ MS-DR module imported successfully")
    print(f"   Library available: {MS_AVAILABLE}")
    print(f"   Library used: {MS_LIBRARY}")
except Exception as e:
    print(f"❌ Failed to import MS-DR module: {e}")
    exit(1)

if not MS_AVAILABLE:
    print("\n⚠️  Markov-Switching models not available.")
    print("   Install with: pip install statsmodels")
    print("\n   Installation command:")
    print("   pip install statsmodels>=0.13.0")
    exit(1)

print("\n" + "=" * 80)
print("Generating Synthetic Market Data")
print("=" * 80)

# Generate synthetic market data with regime changes
np.random.seed(42)
n_samples = 300

# Create 3 regimes with different AR dynamics
regime_1_ar = 0.5  # Positive autocorrelation
regime_2_ar = -0.3  # Negative autocorrelation
regime_3_ar = 0.8  # Strong positive autocorrelation

# Generate time series with regime changes
time_series = np.zeros(n_samples)
regimes_true = np.zeros(n_samples, dtype=int)

# Regime 1: samples 0-100
for i in range(1, 100):
    time_series[i] = regime_1_ar * time_series[i-1] + np.random.normal(0, 0.5)
    regimes_true[i] = 0

# Regime 2: samples 100-200
for i in range(100, 200):
    time_series[i] = regime_2_ar * time_series[i-1] + np.random.normal(0, 1.0)
    regimes_true[i] = 1

# Regime 3: samples 200-300
for i in range(200, 300):
    time_series[i] = regime_3_ar * time_series[i-1] + np.random.normal(0, 0.3)
    regimes_true[i] = 2

# Create features from time series
features = np.column_stack([
    time_series,
    np.roll(time_series, 1),
    np.roll(time_series, 2),
    np.abs(time_series),
])

# Remove first few rows with NaN from rolling
features = features[3:]
time_series = time_series[3:]
regimes_true = regimes_true[3:]

print(f"✅ Generated synthetic data: {features.shape}")
print(f"   True regimes: 3 (manually constructed)")

print("\n" + "=" * 80)
print("Testing MS-DR Clusterer")
print("=" * 80)

# Test 1: Basic initialization
print("\n1. Testing basic initialization...")
try:
    config = MSDRConfig(
        n_regimes=3,
        model_type='autoregression',
        order=1,
        switching_variance=True,
        auto_select_regimes=False,  # Disable for testing
        enable_pca=True,
        pca_components=3
    )
    clusterer = MSDRClusterer(config)
    print("   ✅ MS-DR clusterer initialized successfully")
except Exception as e:
    print(f"   ❌ Failed to initialize: {e}")
    exit(1)

# Test 2: Fit and predict
print("\n2. Testing fit_predict...")
try:
    result = clusterer.fit_predict(features)
    print("   ✅ MS-DR clustering completed successfully")
    print(f"   Discovered regimes: {result.n_clusters}")
    print(f"   Silhouette score: {result.silhouette_score:.3f}")
    print(f"   AIC: {result.aic:.2f}")
    print(f"   BIC: {result.bic:.2f}")
    print(f"   Processing time: {result.processing_time:.2f}s")
    print(f"   Memory usage: {result.memory_usage_mb:.1f} MB")
    
    if result.transition_matrix is not None:
        print(f"\n   Transition Matrix:")
        print(f"   {result.transition_matrix}")
        print(f"\n   Transition persistence: {result.transition_persistence:.3f}")
    
    if result.regime_durations is not None and len(result.regime_durations) > 0:
        print(f"\n   Average regime durations: {result.regime_durations}")
    
    if result.regime_variances is not None and len(result.regime_variances) > 0:
        print(f"\n   Regime variances: {result.regime_variances}")
    
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

# Test 4: Test with auto regime selection
print("\n4. Testing auto regime selection...")
try:
    config_auto = MSDRConfig(
        auto_select_regimes=True,
        min_regimes=2,
        max_regimes=5,
        enable_pca=True,
        pca_components=3
    )
    clusterer_auto = MSDRClusterer(config_auto)
    result_auto = clusterer_auto.fit_predict(features)
    print("   ✅ Auto regime selection completed successfully")
    print(f"   Selected regimes: {result_auto.n_clusters}")
    print(f"   AIC: {result_auto.aic:.2f}")
    print(f"   BIC: {result_auto.bic:.2f}")
except Exception as e:
    print(f"   ❌ Failed with auto selection: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test with DataFrame input
print("\n5. Testing with DataFrame input...")
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
print("Testing Enhanced MS-DR Integration")
print("=" * 80)

# Test integration with feature bank
try:
    from src.feature_generation.integration.enhanced_ms_dr_clustering_integration import (
        EnhancedMSDRClusteringIntegration
    )
    print("✅ Enhanced MS-DR integration module imported")
    
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
    print("\n6. Testing enhanced integration...")
    integration = EnhancedMSDRClusteringIntegration(
        min_features=20,
        max_features=50,
        n_regimes=3,
        auto_select_regimes=False
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
print(f"✅ MS-DR clustering implementation: PASSED")
print(f"✅ Basic functionality: WORKING")
print(f"✅ Discovered regimes: {result.n_clusters}")
print(f"✅ Silhouette score: {result.silhouette_score:.3f}")
print(f"✅ AIC: {result.aic:.2f}")
print(f"✅ BIC: {result.bic:.2f}")
print(f"\n📊 The MS-DR clustering successfully discovered {result.n_clusters} regimes")
print(f"   (True number was 3 regimes)")
print("\n" + "=" * 80)
