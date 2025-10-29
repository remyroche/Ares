#!/usr/bin/env python3
"""
Ultra-Optimized HDP-HMM Clustering Test with Maximum Regime Discovery
- Ultra-low stickiness (κ=2.0)
- Smaller chunks (50 samples)
- Sliding window approach
- Point-wise feature generation
- Maximum regime discovery potential
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("=" * 80)
print("Ultra-Optimized HDP-HMM Clustering Test with Maximum Regime Discovery")
print("=" * 80)

# Check NumPy version
print(f"   NumPy version: {np.__version__}")

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

# Load real market data using artifact manager
print("\n📊 Loading real market data using artifact manager...")

from src.training.steps.pre_training.utils.artifact_manager import get_pretraining_artifact_manager
from src.utils.data.klines_parquet import KlinesParquetManager
from src.feature_generation.categories.regime_feature_integration import RegimeFeatureIntegration

# Initialize artifact manager
artifact_manager = get_pretraining_artifact_manager()

# Set context for ETHUSDT data
artifact_manager.set_context(
    symbol="ETHUSDT",
    exchange="binance",
    information="hdp_hmm_ultra_optimized",
    timeframe="1h"
)

# Try to load existing feature data first
print("   🔍 Looking for existing feature data...")
feature_data = artifact_manager.load("feature_generation_feature_generation_step", "feature_dataframe")

if feature_data is not None and len(feature_data) > 200:
    print(f"   ✅ Found existing feature data: {feature_data.shape}")
    df = feature_data
else:
    print("   ⚠️ No sufficient existing feature data found, loading raw market data...")
    
    # Load raw market data using KlinesParquetManager - Extended period
    klines_manager = KlinesParquetManager()
    
    # Load extended period of ETHUSDT data for maximum regime discovery
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6 months of data
    
    df = klines_manager.read_data(
        symbol="ETHUSDT",
        interval="1h",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
    
    if df is None or df.empty:
        print("   ⚠️ No data found for 180 days, trying with longer period...")
        # Try with a longer period
        start_date = end_date - timedelta(days=365)  # 1 year of data
        df = klines_manager.read_data(
            symbol="ETHUSDT",
            interval="1h",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        if df is None or df.empty:
            raise ValueError("Failed to load any market data.")
    
    print(f"   ✅ Loaded raw market data: {df.shape}")
    print(f"   📊 Columns: {list(df.columns)}")
    print(f"   📊 Date range: {df.index.min()} to {df.index.max()}")
    
    # Generate regime features with ultra-optimized approach
    print("   🔄 Generating regime features with ultra-optimized approach...")
    regime_integrator = RegimeFeatureIntegration()
    
    # Ultra-optimized processing: smaller chunks with sliding window
    chunk_size = 50       # Ultra-small chunks for maximum samples
    overlap = 25          # 50% overlap for sliding window effect
    regime_features_list = []
    
    print(f"   📊 Processing {len(df)} samples in chunks of {chunk_size} with {overlap} overlap")
    print(f"   📊 Expected samples: ~{len(df) // (chunk_size - overlap)} chunks")
    
    try:
        chunk_count = 0
        for i in range(0, len(df) - chunk_size + 1, chunk_size - overlap):
            chunk = df.iloc[i:i+chunk_size]
            if len(chunk) >= 20:  # Need minimum lookback for features
                try:
                    # Generate features for this chunk
                    regime_features = regime_integrator._generate_regime_features(chunk)
                    
                    # Convert dict to DataFrame row
                    if isinstance(regime_features, dict):
                        regime_df = pd.DataFrame([regime_features])
                        regime_features_list.append(regime_df)
                    else:
                        # If it's already a DataFrame, append it
                        regime_features_list.append(regime_features)
                    
                    chunk_count += 1
                    if chunk_count % 20 == 0:
                        print(f"   ✅ Processed chunk {chunk_count}: {len(chunk)} samples")
                except Exception as e:
                    print(f"   ⚠️ Error generating features for chunk {chunk_count+1}: {e}")
                    continue
        
        if not regime_features_list:
            raise ValueError("Failed to generate regime features for any chunks")
        
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
                    print(f"   ⚠️ Could not convert column '{col}' to numeric, encoding as category.")
                    df[col] = pd.Categorical(df[col]).codes
        
        # Fill any remaining NaN values
        df = df.fillna(0)
        
        print(f"   ✅ Generated regime features: {df.shape}")
        print(f"   📊 Feature columns: {list(df.columns)[:10]}...")  # Show first 10 columns
        print(f"   📊 Total chunks processed: {chunk_count}")
        
        # Save the generated features to artifact manager
        print("   💾 Saving generated features to artifact manager...")
        artifact_manager.save(
            "feature_generation_feature_generation_step",
            {"feature_dataframe": df},
            {"generation_method": "RegimeFeatureIntegration", "source": "raw_klines", "chunk_size": chunk_size, "overlap": overlap, "ultra_optimized": True}
        )
        print("   ✅ Features saved to artifact manager")
        
    except Exception as e:
        print(f"   ❌ Error generating regime features: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

# Data preprocessing
print(f"\n📊 Final data shape: {df.shape}")

# Calculate data range for numeric columns only
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    numeric_data = df[numeric_cols]
    print(f"📊 Data range (numeric): {numeric_data.min().min():.4f} to {numeric_data.max().max():.4f}")
else:
    print("📊 No numeric columns found for range calculation")

print(f"📊 Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
print(f"📊 Data types: {df.dtypes.value_counts().to_dict()}")

# Check if we have enough data
if len(df) < 100:
    print(f"⚠️ Warning: Only {len(df)} samples available. Consider using more data for better regime discovery.")
else:
    print(f"✅ Sufficient data available: {len(df)} samples")

# Configure HDP-HMM clustering with ultra-optimized parameters
print("\n🔧 Configuring HDP-HMM clustering with ultra-optimized parameters...")

# Import optimization goals
from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
    DEFAULT_CLUSTERING_GOALS, DEFAULT_OPTIMIZATION_TARGETS,
    calculate_composite_score, MetricCalculator
)

# Determine PCA components based on actual data
n_features = len(df.columns)
if n_features > 50:
    pca_components = min(25, n_features // 2)  # Use 1/2 of features or max 25
elif n_features > 20:
    pca_components = min(20, n_features // 2)  # Use 1/2 of features or max 20
else:
    pca_components = max(8, n_features - 1)  # Keep at least 8 components

print(f"   📊 Features: {n_features}, PCA components: {pca_components}")

# Create ultra-optimized configurations with very low stickiness
configs = [
    HDPHMMConfig(
        alpha=1.5, kappa=2.0, gamma=1.5,  # Ultra-low stickiness for maximum regime changes
        n_iterations=200, max_states=6,
        enable_pca=True, pca_components=pca_components,
        min_samples_required=30
    ),
    HDPHMMConfig(
        alpha=2.0, kappa=3.0, gamma=2.0,  # Very low stickiness
        n_iterations=250, max_states=8,
        enable_pca=True, pca_components=pca_components,
        min_samples_required=30
    ),
    HDPHMMConfig(
        alpha=2.5, kappa=5.0, gamma=2.5,  # Low stickiness
        n_iterations=300, max_states=10,
        enable_pca=True, pca_components=pca_components,
        min_samples_required=30
    ),
    HDPHMMConfig(
        alpha=3.0, kappa=8.0, gamma=3.0,  # Moderate stickiness
        n_iterations=350, max_states=12,
        enable_pca=True, pca_components=pca_components,
        min_samples_required=30
    ),
    HDPHMMConfig(
        alpha=4.0, kappa=10.0, gamma=4.0,  # Slightly higher stickiness
        n_iterations=400, max_states=15,
        enable_pca=True, pca_components=pca_components,
        min_samples_required=30
    )
]

print(f"   Testing {len(configs)} ultra-optimized configurations...")

best_score = -np.inf
best_config = None
best_result = None

metric_calculator = MetricCalculator()

for i, config in enumerate(configs):
    print(f"\n--- Running with Configuration {i+1}/{len(configs)} ---")
    print(f"   Alpha: {config.alpha}")
    print(f"   Kappa: {config.kappa}")
    print(f"   Gamma: {config.gamma}")
    print(f"   Max iterations: {config.n_iterations}")
    print(f"   Max states: {config.max_states}")
    print(f"   PCA enabled: {config.enable_pca}")
    print(f"   PCA components: {config.pca_components} (from {n_features} features)")
    print(f"   Min samples required: {config.min_samples_required}")

    clusterer = HDPHMMClusterer(config)
    print(f"ℹ️ Initializing HDPHMMClusterer with config: {config}")

    try:
        print("🚀 Starting HDP-HMM clustering...")
        result = clusterer.fit_predict(df)
        print("✅ HDP-HMM clustering completed successfully.")
        
        # Calculate metrics for auto-tuning
        # Handle cluster_probabilities shape - convert 1D to 2D if needed
        regime_probs = result.cluster_probabilities
        if regime_probs is not None:
            if regime_probs.ndim == 1:
                # Convert to 2D: (n_samples, n_clusters)
                n_samples = len(regime_probs)
                n_clusters = result.n_clusters
                regime_probs_2d = np.zeros((n_samples, n_clusters))
                for i, label in enumerate(result.cluster_labels):
                    if label < n_clusters:
                        regime_probs_2d[i, int(label)] = 1.0
                regime_probs = regime_probs_2d
        else:
            # Create default probabilities
            n_samples = len(result.cluster_labels)
            n_clusters = result.n_clusters
            regime_probs = np.zeros((n_samples, n_clusters))
            for i, label in enumerate(result.cluster_labels):
                if label < n_clusters:
                    regime_probs[i, int(label)] = 1.0
        
        try:
            rolling_ll, _ = metric_calculator.calculate_rolling_log_likelihood(
                data=df.values,
                regime_probs=regime_probs,
                regime_params=result.cluster_parameters
            )
        except Exception as e:
            print(f"⚠️ Failed to calculate rolling log likelihood: {e}")
            rolling_ll = 0.0
        
        try:
            one_step_ll, _ = metric_calculator.calculate_one_step_log_likelihood(
                data=df.values,
                regime_labels=result.cluster_labels,
                regime_params=result.cluster_parameters
            )
        except Exception as e:
            print(f"⚠️ Failed to calculate one-step log likelihood: {e}")
            one_step_ll = 0.0
        
        # For economic utility, we need returns. Assuming 'close_log_return' is available.
        # If not, a placeholder or a more robust way to get returns is needed.
        if 'close_log_return' in df.columns:
            returns = df['close_log_return'].values
        else:
            print("⚠️ 'close_log_return' not found, using dummy returns for economic utility.")
            returns = np.random.rand(len(df)) * 0.01 # Dummy returns
        
        try:
            economic_metrics = metric_calculator.calculate_economic_utility(
                returns=returns,
                regime_labels=result.cluster_labels
            )
            sharpe = economic_metrics.get('sharpe', 0.0)
        except Exception as e:
            print(f"⚠️ Failed to calculate economic utility: {e}")
            sharpe = 0.0

        try:
            composite_score = calculate_composite_score(
                rolling_ll=rolling_ll,
                one_step_ll=one_step_ll,
                economic_utility=sharpe,
                goals=DEFAULT_CLUSTERING_GOALS # Use default goals for now
            )
        except Exception as e:
            print(f"⚠️ Failed to calculate composite score: {e}")
            composite_score = -np.inf

        print(f"   Composite Score: {composite_score:.4f}")

        if composite_score > best_score:
            best_score = composite_score
            best_config = config
            best_result = result

    except Exception as e:
        print(f"❌ HDP-HMM clustering failed for this config: {e}")
        import traceback
        traceback.print_exc()
        continue

if best_result is None:
    print("\n❌ No successful clustering results found across all configurations.")
    exit(1)

print("\n" + "=" * 80)
print("✅ Best HDP-HMM Clustering Result (Ultra-Optimized)")
print("=" * 80)
print(f"   Best Composite Score: {best_score:.4f}")
print(f"   Best Configuration: {best_config}")

# Display results for the best configuration
print(f"\n📈 Clustering Results:")
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
print(f"   Silhouette score: {best_result.silhouette_score:.4f}")
print(f"   Calinski-Harabasz score: {best_result.calinski_harabasz_score:.4f}")
print(f"   Davies-Bouldin score: {best_result.davies_bouldin_score:.4f}")
print(f"   Noise ratio: {best_result.noise_ratio:.4f}")
        
print(f"\n🎯 Transition Matrix:")
if best_result.transition_matrix is not None:
    print(f"   Shape: {best_result.transition_matrix.shape}")
    print(f"   Matrix:\n{best_result.transition_matrix}")
else:
    print("   Transition matrix not available")
        
print(f"\n📉 Cluster Probabilities:")
if best_result.cluster_probabilities is not None:
    print(f"   Shape: {best_result.cluster_probabilities.shape}")
    print(f"   Max probability per sample: {best_result.cluster_probabilities.max():.4f}")
    print(f"   Min probability per sample: {best_result.cluster_probabilities.min():.4f}")
else:
    print("   Cluster probabilities not available")

# Save results to artifact manager
print(f"\n💾 Saving clustering results to artifact manager...")
try:
    artifact_manager.save(
        "hdp_hmm_clustering_step",
        {
            "clustering_result": best_result,
            "best_config": best_config,
            "composite_score": best_score,
            "feature_dataframe": df
        },
        {
            "clustering_method": "HDP-HMM",
            "library": HMM_LIBRARY,
            "n_configurations_tested": len(configs),
            "best_configuration": str(best_config),
            "optimization": "ultra_optimized_regime_discovery",
            "chunk_size": 50,
            "overlap": 25,
            "ultra_low_stickiness": True
        }
    )
    print("✅ Clustering results saved to artifact manager")
except Exception as e:
    print(f"⚠️ Failed to save results to artifact manager: {e}")
            
print("\n✅ Ultra-Optimized HDP-HMM clustering test completed successfully!")
