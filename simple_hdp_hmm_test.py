#!/usr/bin/env python3
"""
Simple HDP-HMM clustering test that bypasses complex imports.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("=" * 80)
print("Simple HDP-HMM Clustering Test")
print("=" * 80)

# Check NumPy version
print(f"   NumPy version: {np.__version__}")

# Test HDP-HMM clusterer import
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

# Load real market data and generate regime features
print("\n📊 Loading real market data and generating regime features...")

from src.training.steps.pre_training.utils.artifact_manager import get_pretraining_artifact_manager
from src.utils.data.klines_parquet import KlinesParquetManager
from src.feature_generation.categories.regime_feature_categorization import get_regime_clustering_features
from src.feature_generation.categories.regime_feature_integration import RegimeFeatureIntegration, RegimeFeatureConfig

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
    print("   ⚠️ No existing feature data found, loading raw OHLCV data...")
    
    # Initialize KlinesParquetManager
    klines_manager = KlinesParquetManager()
    
    # Load recent ETHUSDT data
    df = klines_manager.read_data(
        symbol="ETHUSDT",
        interval="1h",
        start_date="2024-01-01",
        end_date="2024-12-31"
    )
    
    if df is not None and not df.empty:
        print(f"   ✅ Loaded raw OHLCV data: {df.shape}")
        print(f"   📅 Date range: {df.index.min()} to {df.index.max()}")
        
        # Generate regime features using the proper feature generation system
        print("   🔧 Generating regime features for clustering...")
        
        # Get the list of regime clustering features
        regime_feature_names = get_regime_clustering_features()
        print(f"   📋 Using {len(regime_feature_names)} regime clustering features")
        
        # Initialize regime feature generator
        regime_config = RegimeFeatureConfig(
            enable_regime_detection=True,
            enable_adaptive_features=True,
            enable_regime_transitions=True,
            lookback_period=20
        )
        regime_generator = RegimeFeatureIntegration(regime_config)
        
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
                    regime_features = regime_generator._generate_regime_features(chunk)
                    regime_features_list.append(regime_features)
                    print(f"   ✅ Processed chunk {len(regime_features_list)}: {len(chunk)} samples")
                except Exception as e:
                    print(f"   ⚠️ Error generating features for chunk {len(regime_features_list)+1}: {e}")
                    continue
        
        if not regime_features_list:
            raise ValueError("Failed to generate regime features")
        
        # Convert to DataFrame
        print("   🔄 Converting regime features to DataFrame...")
        df_features = []
        for features in regime_features_list:
            # Convert dict to Series and append
            feature_series = pd.Series(features)
            df_features.append(feature_series)
        
        if df_features:
            df = pd.DataFrame(df_features)
            # Fill NaN values with 0
            df = df.fillna(0)
            
            # Convert string columns to numeric where possible
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        # If conversion fails, keep as string but encode for clustering
                        df[col] = df[col].astype('category').cat.codes
            
            # Fill any remaining NaN values after conversion
            df = df.fillna(0)
            
            print(f"   ✅ Generated regime features: {df.shape}")
            print(f"   📊 Feature columns: {list(df.columns)[:10]}...")  # Show first 10 columns
        else:
            raise ValueError("No regime features generated")
    else:
        raise ValueError("No market data available")

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

print(f"   Alpha: {config.alpha}")
print(f"   Kappa: {config.kappa}")
print(f"   Gamma: {config.gamma}")
print(f"   Max iterations: {config.n_iterations}")
print(f"   Max states: {config.max_states}")
print(f"   PCA enabled: {config.enable_pca}")
print(f"   PCA components: {config.pca_components} (from {n_features} features)")
print(f"   Min samples required: {config.min_samples_required}")

# Initialize clusterer
print("\n🚀 Initializing HDP-HMM clusterer...")
clusterer = HDPHMMClusterer(config)
print("   Clusterer initialized successfully")

# Run clustering
print("\n🔄 Running HDP-HMM clustering...")
try:
    result = clusterer.fit_predict(df)
    print("✅ Clustering completed successfully!")
    
    # Display results
    print(f"\n📈 Clustering Results:")
    print(f"   Number of clusters discovered: {result.n_clusters}")
    print(f"   Cluster assignments shape: {result.cluster_labels.shape}")
    print(f"   Unique clusters: {np.unique(result.cluster_labels)}")
    print(f"   Cluster distribution:")
    
    unique_clusters, counts = np.unique(result.cluster_labels, return_counts=True)
    for cluster, count in zip(unique_clusters, counts):
        percentage = (count / len(result.cluster_labels)) * 100
        print(f"     Cluster {cluster}: {count} samples ({percentage:.1f}%)")
    
    print(f"\n📊 Model Metrics:")
    print(f"   Log likelihood: {result.log_likelihood:.4f}")
    print(f"   Silhouette score: {result.silhouette_score:.4f}")
    print(f"   Calinski-Harabasz score: {result.calinski_harabasz_score:.4f}")
    print(f"   Davies-Bouldin score: {result.davies_bouldin_score:.4f}")
    print(f"   Noise ratio: {result.noise_ratio:.4f}")
    
    print(f"\n🎯 Transition Matrix:")
    if result.transition_matrix is not None:
        print(f"   Shape: {result.transition_matrix.shape}")
        print(f"   Matrix:\n{result.transition_matrix}")
    else:
        print("   Transition matrix not available")
    
    print(f"\n📉 Cluster Probabilities:")
    if result.cluster_probabilities is not None:
        print(f"   Shape: {result.cluster_probabilities.shape}")
        print(f"   Max probability per sample: {result.cluster_probabilities.max():.4f}")
        print(f"   Min probability per sample: {result.cluster_probabilities.min():.4f}")
    else:
        print("   Cluster probabilities not available")
        
    print("\n✅ HDP-HMM clustering test completed successfully!")
    
except Exception as e:
    print(f"❌ Clustering failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("Test completed!")
print("=" * 80)
