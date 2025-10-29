#!/usr/bin/env python3
"""
Simple HDP-HMM Clustering Test with Auto-Tuning
Uses real market data and auto-tunes parameters using clustering optimization goals.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("=" * 80)
print("HDP-HMM Clustering Test with Auto-Tuning")
print("=" * 80)

# Check if pyhsmm is available and print NumPy version
try:
    import sys
    import types
    
    # Apply compatibility shims
    try:
        import scipy.special as _sp_special
        _misc = types.ModuleType('scipy.misc')
        _misc.logsumexp = _sp_special.logsumexp
        sys.modules.setdefault('scipy.misc', _misc)
    except Exception:
        pass

    try:
        _umath_tests = types.ModuleType('numpy.core.umath_tests')
        _umath_tests.inner1d = np.inner
        sys.modules.setdefault('numpy.core.umath_tests', _umath_tests)
    except Exception:
        pass

    import pyhsmm
    print(f"✅ pyhsmm imported successfully")
    print(f"   NumPy version: {np.__version__}")
except ImportError as e:
    print(f"❌ pyhsmm not imported: {e}")
    exit(1)

# Import HDP-HMM clusterer
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
    print("⚠️ HMM libraries not available. Skipping clustering test.")
    exit(0)

# Load real market data from artifact manager
print("\n📊 Loading real market data from artifact manager...")

from src.training.steps.pre_training.utils.artifact_manager import get_pretraining_artifact_manager
from src.utils.data.klines_parquet import KlinesParquetManager

# Initialize artifact manager
artifact_manager = get_pretraining_artifact_manager()

# Set context for ETHUSDT data
artifact_manager.set_context(
    symbol="ETHUSDT",
    exchange="binance",
    information="hdp_hmm_test",
    timeframe="1h"
)

# Try to load existing feature data
print("   🔍 Looking for existing feature data...")
feature_data = artifact_manager.load("feature_generation_feature_generation_step", "feature_dataframe")

if feature_data is not None:
    print(f"   ✅ Found existing feature data: {feature_data.shape}")
    df = feature_data
else:
    print("   ⚠️ No existing feature data found, loading raw market data...")
    
    # Load raw market data
    klines_manager = KlinesParquetManager()
    df = klines_manager.read_data(
        symbol="ETHUSDT",
        interval="1h",
        start_date="2024-01-01",
        end_date="2024-12-31"
    )
    
    print(f"   📊 Loaded raw market data: {df.shape}")
    print(f"   📊 Columns: {list(df.columns)}")
    
    # Generate regime features
    print("   🔄 Generating regime features...")
    from src.feature_generation.categories.regime_feature_categorization import RegimeFeatureCategorizer
    from src.feature_generation.categories.regime_feature_integration import RegimeFeatureIntegration
    
    regime_integrator = RegimeFeatureIntegration()
    
    # Generate regime features for the entire dataset
    print("   🔄 Generating regime features...")
    regime_features_list = []
    
    # Process data in chunks to avoid memory issues
    chunk_size = 500  # Smaller chunks for better memory management
    max_chunks = 20   # Process up to 20 chunks (10,000 samples)
    
    print(f"   📊 Processing up to {max_chunks} chunks of {chunk_size} samples each...")
    
    for i in range(0, min(len(df), max_chunks * chunk_size), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        if len(chunk) >= 20:  # Need minimum lookback
            try:
                regime_features = regime_integrator._generate_regime_features(chunk)
                
                # Convert dict to DataFrame row
                if isinstance(regime_features, dict):
                    regime_df = pd.DataFrame([regime_features])
                    regime_features_list.append(regime_df)
                else:
                    # If it's already a DataFrame, append it
                    regime_features_list.append(regime_features)
                
                print(f"   ✅ Processed chunk {len(regime_features_list)}: {len(chunk)} samples")
            except Exception as e:
                print(f"   ⚠️ Error generating features for chunk {len(regime_features_list)+1}: {e}")
                continue
    
    if not regime_features_list:
        raise ValueError("Failed to generate regime features")
    
    # Concatenate all feature DataFrames
    df = pd.concat(regime_features_list, ignore_index=True)
    # Fill NaN values with 0
    df = df.fillna(0)
        
        # Convert string columns to numeric where possible
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    # If conversion fails, keep as string but encode for clustering
                    df[col] = pd.Categorical(df[col]).codes
        
        # Fill any remaining NaN values
        df = df.fillna(0)
        
        print(f"   ✅ Generated regime features: {df.shape}")
        print(f"   📊 Feature columns: {list(df.columns)[:10]}...")  # Show first 10 columns
    else:
        raise ValueError("No regime features generated")

print(f"   📊 Final data shape: {df.shape}")
print(f"   📊 Data range: {df.min().min():.4f} to {df.max().max():.4f}")
print(f"   📊 Columns: {list(df.columns)[:10]}...")  # Show first 10 columns

# Configure HDP-HMM clustering with auto-tuning
print("\n🔧 Configuring HDP-HMM clustering with auto-tuning...")

# Import optimization goals
from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
    DEFAULT_CLUSTERING_GOALS, DEFAULT_OPTIMIZATION_TARGETS,
    calculate_composite_score, MetricCalculator
)

# Determine PCA components based on actual data
n_features = len(df.columns)
if n_features > 10:
    pca_components = min(10, n_features // 2)  # Use half the features or max 10
else:
    pca_components = max(3, n_features - 1)  # Keep at least 3 components

# Create multiple configurations for auto-tuning
configs = [
    HDPHMMConfig(
        alpha=2.0, kappa=30.0, gamma=2.0,
        n_iterations=50, max_states=8,
        enable_pca=True, pca_components=pca_components,
        min_samples_required=50
    ),
    HDPHMMConfig(
        alpha=3.0, kappa=50.0, gamma=3.0,
        n_iterations=75, max_states=12,
        enable_pca=True, pca_components=pca_components,
        min_samples_required=50
    ),
    HDPHMMConfig(
        alpha=4.0, kappa=70.0, gamma=4.0,
        n_iterations=100, max_states=15,
        enable_pca=True, pca_components=pca_components,
        min_samples_required=50
    )
]

print(f"   Testing {len(configs)} different configurations...")

# Auto-tune HDP-HMM parameters
print(f"\n🔍 Auto-tuning HDP-HMM parameters...")
best_config = None
best_score = float('-inf')
best_result = None
results = []

for i, config in enumerate(configs):
    print(f"\n   Testing configuration {i+1}/{len(configs)}:")
    print(f"     Alpha: {config.alpha}, Kappa: {config.kappa}, Gamma: {config.gamma}")
    print(f"     Max states: {config.max_states}, Iterations: {config.n_iterations}")
    
    try:
        clusterer = HDPHMMClusterer(config)
        result = clusterer.fit_predict(df)
        
        # Calculate composite score using optimization goals
        metric_calc = MetricCalculator()
        
        # Calculate economic utility (simplified - using cluster variance as proxy)
        if result.cluster_means is not None and len(result.cluster_means) > 1:
            cluster_vars = []
            for cluster_id in np.unique(result.cluster_labels):
                if cluster_id >= 0:  # Skip noise points
                    mask = result.cluster_labels == cluster_id
                    if np.sum(mask) > 1:
                        cluster_data = df[mask]
                        cluster_vars.append(np.var(cluster_data.values))
            
            if cluster_vars:
                # Use variance as proxy for economic utility (higher variance = more distinct regimes)
                economic_utility = np.mean(cluster_vars)
            else:
                economic_utility = 0.0
        else:
            economic_utility = 0.0
        
        # Calculate composite score
        composite_score = calculate_composite_score(
            rolling_ll=result.log_likelihood,
            one_step_ll=result.log_likelihood,  # Simplified
            economic_utility=economic_utility,
            goals=DEFAULT_CLUSTERING_GOALS
        )
        
        results.append({
            'config': config,
            'result': result,
            'composite_score': composite_score,
            'n_clusters': result.n_clusters,
            'log_likelihood': result.log_likelihood,
            'economic_utility': economic_utility
        })
        
        print(f"     ✅ Score: {composite_score:.4f}, Clusters: {result.n_clusters}")
        
        if composite_score > best_score:
            best_score = composite_score
            best_config = config
            best_result = result
            
    except Exception as e:
        print(f"     ❌ Failed: {e}")
        continue

if best_result is None:
    print("❌ All configurations failed!")
    exit(1)

print(f"\n🏆 Best Configuration Found:")
print(f"   Score: {best_score:.4f}")
print(f"   Alpha: {best_config.alpha}")
print(f"   Kappa: {best_config.kappa}")
print(f"   Gamma: {best_config.gamma}")
print(f"   Max states: {best_config.max_states}")
print(f"   Iterations: {best_config.n_iterations}")

# Display best results
print(f"\n📈 Best Clustering Results:")
print(f"   Number of clusters discovered: {best_result.n_clusters}")
print(f"   Cluster assignments shape: {best_result.cluster_labels.shape}")
print(f"   Unique clusters: {np.unique(best_result.cluster_labels)}")
print(f"   Cluster distribution:")

unique_clusters, counts = np.unique(best_result.cluster_labels, return_counts=True)
for cluster, count in zip(unique_clusters, counts):
    percentage = (count / len(best_result.cluster_labels)) * 100
    print(f"     Cluster {cluster}: {count} samples ({percentage:.1f}%)")

print(f"\n📊 Model Metrics:")
print(f"   Log likelihood: {best_result.log_likelihood:.4f}")
print(f"   AIC: {best_result.aic:.4f}")
print(f"   BIC: {best_result.bic:.4f}")
print(f"   Convergence: {best_result.converged}")
print(f"   Iterations: {best_result.n_iterations}")

print(f"\n🎯 Transition Matrix:")
if best_result.transition_matrix is not None:
    print(f"   Shape: {best_result.transition_matrix.shape}")
    print(f"   Matrix:\n{best_result.transition_matrix}")
else:
    print("   Transition matrix not available")

print(f"\n📉 Cluster Means:")
if best_result.cluster_means is not None:
    print(f"   Shape: {best_result.cluster_means.shape}")
    for i, mean in enumerate(best_result.cluster_means):
        print(f"     Cluster {i}: {mean}")
else:
    print("   Cluster means not available")

# Summary of all results
print(f"\n📊 All Results Summary:")
print(f"   {'Config':<3} {'Score':<8} {'Clusters':<8} {'Log-LL':<10} {'Economic':<10}")
print(f"   {'-'*3} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
for i, res in enumerate(results):
    print(f"   {i+1:<3} {res['composite_score']:<8.4f} {res['n_clusters']:<8} {res['log_likelihood']:<10.4f} {res['economic_utility']:<10.4f}")

print("\n✅ HDP-HMM clustering test with auto-tuning completed successfully!")
