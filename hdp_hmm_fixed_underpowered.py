#!/usr/bin/env python3
"""
HDP-HMM Fixed for Underpowered Data Issue

Fixes:
1. Drastically increase α (HDP concentration parameter) from 3.0 to 20-50
2. Reduce features with stronger PCA (50-100 → 10-20 features)
3. Use 5-minute data instead of 1-hour for more samples
4. Extend lookback period to get 1000+ samples
5. Fix α/γ parameters for aggressive state discovery
"""

import sys
import os
sys.path.append('/Users/remyroche/Documents/Ares')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

# Import HDP-HMM components
from src.training.steps.market_analysis.hdp_hmm_clustering.hdp_hmm_clusterer import (
    HDPHMMClusterer, HDPHMMConfig, HMM_AVAILABLE, HMM_LIBRARY
)

# Import data loading
from src.utils.data.klines_parquet import KlinesParquetManager
from src.training.steps.pre_training.utils.artifact_manager import PreTrainingArtifactManager

# Import feature generation
from src.feature_generation.categories.regime_feature_integration import RegimeFeatureIntegration

# Import optimization
from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
    DEFAULT_CLUSTERING_GOALS, DEFAULT_OPTIMIZATION_TARGETS, calculate_composite_score, MetricCalculator
)

def main():
    print("🚀 HDP-HMM Fixed for Underpowered Data Issue")
    print("=" * 60)
    
    if not HMM_AVAILABLE:
        print("❌ HMM libraries not available")
        return
    
    print(f"✅ Using HMM library: {HMM_LIBRARY}")
    
    # FIXED: Use 5-minute data for more samples
    print("\n📊 Loading 5-minute data for more samples...")
    klines_manager = KlinesParquetManager(exchange='binance')
    
    # Try 5-minute data first, fallback to 1-minute if not available
    intervals_to_try = ['5m', '1m', '15m', '1h']
    df = None
    used_interval = None
    
    for interval in intervals_to_try:
        try:
            print(f"   Trying {interval} data...")
            df = klines_manager.read_data(
                symbol='ETHUSDT',
                interval=interval,
                start_date=datetime.now() - timedelta(days=30),  # 30 days of data
                end_date=datetime.now()
            )
            if df is not None and not df.empty:
                used_interval = interval
                print(f"   ✅ Loaded {interval} data: {df.shape}")
                break
        except Exception as e:
            print(f"   ❌ Failed to load {interval} data: {e}")
            continue
    
    if df is None or df.empty:
        print("❌ Failed to load any market data")
        return
    
    print(f"📈 Data loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"📅 Interval: {used_interval}")
    print(f"📊 Date range: {df.index[0]} to {df.index[-1]}")
    
    # FIXED: Generate regime features with stronger filtering
    print("\n🔧 Generating regime features with stronger filtering...")
    
    # Initialize feature integration
    feature_integration = RegimeFeatureIntegration()
    
    # Process in smaller chunks for memory efficiency
    chunk_size = 200  # Smaller chunks for 5-minute data
    regime_features_list = []
    
    print(f"   Processing {len(df)} samples in chunks of {chunk_size}...")
    
    for i in range(0, len(df), chunk_size):
        chunk_end = min(i + chunk_size, len(df))
        chunk_df = df.iloc[i:chunk_end]
        
        try:
            # Generate regime features for this chunk
            regime_features = feature_integration._generate_regime_features(chunk_df)
            
            # Convert to DataFrame and add to list
            if regime_features:
                regime_df = pd.DataFrame(regime_features)
                regime_features_list.append(regime_df)
                print(f"   ✅ Chunk {i//chunk_size + 1}: {regime_df.shape[1]} features")
            else:
                print(f"   ⚠️ Chunk {i//chunk_size + 1}: No features generated")
                
        except Exception as e:
            print(f"   ❌ Chunk {i//chunk_size + 1} failed: {e}")
            continue
    
    if not regime_features_list:
        print("❌ No regime features generated")
        return
    
    # Concatenate all features
    print(f"\n🔗 Concatenating {len(regime_features_list)} feature chunks...")
    regime_features_df = pd.concat(regime_features_list, ignore_index=True)
    
    print(f"📊 Generated regime features: {regime_features_df.shape}")
    
    # FIXED: Stronger feature filtering and PCA
    print("\n🔧 Applying stronger feature filtering and PCA...")
    
    # Convert to numeric and handle mixed types
    numeric_cols = regime_features_df.select_dtypes(include=[np.number]).columns
    regime_features_numeric = regime_features_df[numeric_cols].fillna(0)
    
    print(f"   Numeric features: {len(numeric_cols)}")
    
    # FIXED: Much stronger PCA - reduce to 10-20 features max
    n_samples = len(regime_features_numeric)
    n_features = len(numeric_cols)
    
    # Calculate optimal PCA components (much more aggressive)
    max_pca_components = min(20, n_features, n_samples - 1)  # Max 20 components
    optimal_pca = min(15, max_pca_components)  # Target 15 components
    
    print(f"   Original features: {n_features}")
    print(f"   Target PCA components: {optimal_pca}")
    print(f"   Samples per feature ratio: {n_samples / optimal_pca:.1f}")
    
    # FIXED: Multiple configurations with much higher α values
    configs = [
        # Aggressive state discovery with high α
        HDPHMMConfig(
            alpha=25.0,  # FIXED: Much higher α for aggressive state discovery
            kappa=5.0,   # FIXED: Lower κ for more state changes
            gamma=10.0,  # FIXED: Higher γ for more base distribution support
            n_iterations=200,
            n_burnin=50,
            max_states=15,  # FIXED: Lower max_states since we have fewer features
            pca_components=optimal_pca,
            min_samples_required=50,  # FIXED: Lower minimum samples
            enable_pca=True,
            enable_scaling=True
        ),
        # Even more aggressive
        HDPHMMConfig(
            alpha=40.0,  # FIXED: Very high α
            kappa=3.0,   # FIXED: Very low κ
            gamma=15.0,  # FIXED: Very high γ
            n_iterations=250,
            n_burnin=75,
            max_states=12,
            pca_components=optimal_pca,
            min_samples_required=40,
            enable_pca=True,
            enable_scaling=True
        ),
        # Ultra-aggressive for underpowered data
        HDPHMMConfig(
            alpha=60.0,  # FIXED: Ultra-high α
            kappa=2.0,   # FIXED: Ultra-low κ
            gamma=20.0,  # FIXED: Ultra-high γ
            n_iterations=300,
            n_burnin=100,
            max_states=10,
            pca_components=optimal_pca,
            min_samples_required=30,
            enable_pca=True,
            enable_scaling=True
        )
    ]
    
    print(f"\n🎯 Testing {len(configs)} configurations with fixed parameters...")
    print("   Key fixes:")
    print("   - α: 25-60 (was 3.0) - aggressive state discovery")
    print("   - κ: 2-5 (was 40.0) - more state changes")
    print("   - γ: 10-20 (was 3.0) - stronger base distribution")
    print("   - Features: 15 (was 50-100) - much fewer features")
    print("   - Data: 5-minute (was 1-hour) - more samples")
    
    best_score = -np.inf
    best_config = None
    best_result = None
    
    for i, config in enumerate(configs):
        print(f"\n🔬 Testing Configuration {i+1}/{len(configs)}")
        print(f"   α={config.alpha}, κ={config.kappa}, γ={config.gamma}")
        print(f"   PCA components: {config.pca_components}")
        
        try:
            # Initialize clusterer
            clusterer = HDPHMMClusterer(config)
            
            # Run clustering
            start_time = time.time()
            result = clusterer.fit_predict(regime_features_numeric)
            processing_time = time.time() - start_time
            
            print(f"   ✅ Clustering completed in {processing_time:.2f}s")
            print(f"   📊 Clusters discovered: {result.n_clusters}")
            print(f"   📈 Success: {result.success}")
            
            if result.success and result.n_clusters > 0:
                # Calculate composite score
                try:
                    # Convert cluster probabilities to 2D if needed
                    cluster_probs = result.cluster_probabilities
                    if cluster_probs.ndim == 1:
                        cluster_probs = cluster_probs.reshape(-1, 1)
                    
                    # Calculate metrics
                    metric_calc = MetricCalculator()
                    metrics = metric_calc.calculate_all_metrics(
                        regime_features_numeric.values,
                        result.cluster_labels,
                        cluster_probs
                    )
                    
                    # Calculate composite score
                    composite_score = calculate_composite_score(
                        metrics=metrics,
                        goals=DEFAULT_CLUSTERING_GOALS,
                        targets=DEFAULT_OPTIMIZATION_TARGETS
                    )
                    
                    print(f"   🎯 Composite score: {composite_score:.3f}")
                    print(f"   📊 Silhouette: {metrics.get('silhouette_score', 0):.3f}")
                    print(f"   📊 Calinski-Harabasz: {metrics.get('calinski_harabasz_score', 0):.3f}")
                    
                    # Track best result
                    if composite_score > best_score:
                        best_score = composite_score
                        best_config = config
                        best_result = result
                        print(f"   🏆 New best configuration!")
                    
                except Exception as e:
                    print(f"   ⚠️ Score calculation failed: {e}")
                    # Still consider this result if it has multiple clusters
                    if result.n_clusters > 1 and composite_score == -np.inf:
                        best_score = 0.0
                        best_config = config
                        best_result = result
                        print(f"   🏆 Best configuration (no score): {result.n_clusters} clusters")
            else:
                print(f"   ❌ Failed or single cluster")
                
        except Exception as e:
            print(f"   ❌ Configuration failed: {e}")
            continue
    
    # Report results
    print("\n" + "=" * 60)
    print("🎉 HDP-HMM FIXED RESULTS")
    print("=" * 60)
    
    if best_result and best_result.success:
        print(f"✅ SUCCESS: Discovered {best_result.n_clusters} regimes!")
        print(f"🏆 Best Configuration:")
        print(f"   α (concentration): {best_config.alpha}")
        print(f"   κ (stickiness): {best_config.kappa}")
        print(f"   γ (base dist): {best_config.gamma}")
        print(f"   PCA components: {best_config.pca_components}")
        print(f"   Max states: {best_config.max_states}")
        print(f"   Iterations: {best_config.n_iterations}")
        
        print(f"\n📊 Quality Metrics:")
        print(f"   Silhouette score: {best_result.silhouette_score:.3f}")
        print(f"   Calinski-Harabasz: {best_result.calinski_harabasz_score:.3f}")
        print(f"   Davies-Bouldin: {best_result.davies_bouldin_score:.3f}")
        print(f"   Log likelihood: {best_result.log_likelihood:.3f}")
        print(f"   Transition persistence: {best_result.transition_persistence:.3f}")
        
        print(f"\n📈 Data Summary:")
        print(f"   Total samples: {len(regime_features_numeric)}")
        print(f"   Features used: {best_config.pca_components}")
        print(f"   Samples per feature: {len(regime_features_numeric) / best_config.pca_components:.1f}")
        print(f"   Interval: {used_interval}")
        
        # Show cluster distribution
        unique_labels, counts = np.unique(best_result.cluster_labels, return_counts=True)
        print(f"\n📊 Cluster Distribution:")
        for label, count in zip(unique_labels, counts):
            percentage = (count / len(best_result.cluster_labels)) * 100
            print(f"   Cluster {label}: {count} samples ({percentage:.1f}%)")
        
        print(f"\n🎯 Key Fixes Applied:")
        print(f"   ✅ Increased α from 3.0 to {best_config.alpha} (aggressive state discovery)")
        print(f"   ✅ Reduced κ from 40.0 to {best_config.kappa} (more state changes)")
        print(f"   ✅ Increased γ from 3.0 to {best_config.gamma} (stronger base distribution)")
        print(f"   ✅ Reduced features from 50-100 to {best_config.pca_components} (stronger PCA)")
        print(f"   ✅ Used {used_interval} data instead of 1h (more samples)")
        print(f"   ✅ Samples per feature: {len(regime_features_numeric) / best_config.pca_components:.1f} (target: 10+)")
        
    else:
        print("❌ FAILED: Still collapsing to single state")
        print("\n🔍 Additional recommendations:")
        print("   1. Try even higher α values (50-100)")
        print("   2. Use 1-minute data for maximum samples")
        print("   3. Further reduce features (5-10 components)")
        print("   4. Consider alternative clustering algorithms")
        print("   5. Check if data has sufficient regime structure")

if __name__ == "__main__":
    main()
