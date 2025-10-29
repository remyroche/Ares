#!/usr/bin/env python3
"""
Quick MS-DR Clustering Test to Get Complete Metrics
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, Any

# Add src to path
sys.path.insert(0, 'src')

# Import MS-DR clustering components
from src.training.steps.market_analysis.ms_dr_clustering.ms_dr_clusterer import MSDRClusterer, MSDRConfig, MSDRResult
from src.utils.data.klines_parquet import KlinesParquetManager

print("=" * 80)
print("MS-DR CLUSTERING - COMPREHENSIVE METRICS REPORT")
print("=" * 80)

# Load real market data
print("\n📊 Loading real market data...")
klines_manager = KlinesParquetManager()

try:
    # Try to load ETHUSDT data
    symbol = "ETHUSDT"
    timeframe = "1h"
    
    df = klines_manager.get_klines(
        symbol=symbol,
        interval=timeframe,
        limit=1000
    )
    
    if df is None or len(df) == 0:
        raise ValueError("No data loaded")
    
    print(f"✅ Loaded {len(df)} samples")
    
    # Generate regime features (simplified version)
    print("\n🔧 Generating regime features...")
    from src.feature_generation.core.feature_bank import FeatureBank
    
    feature_bank = FeatureBank()
    
    # Generate regime features
    regime_features = feature_bank.generate_regime_features(
        df,
        include_categories=['returns', 'momentum', 'volatility', 'volume', 'trend']
    )
    
    print(f"✅ Generated {regime_features.shape[1]} features from {regime_features.shape[0]} samples")
    
    # Prepare data
    numeric_cols = regime_features.select_dtypes(include=[np.number]).columns
    feature_data = regime_features[numeric_cols].fillna(0).values
    
    print(f"📊 Final feature matrix: {feature_data.shape}")
    
    # Run MS-DR clustering with quick settings
    print("\n🚀 Running MS-DR clustering...")
    config = MSDRConfig(
        n_regimes=3,  # Fixed number for faster execution
        auto_select_regimes=False,  # Disable auto-selection for speed
        model_type='msar',
        switching_variance=True,
        pca_aggregation='first',
        pca_variance_threshold=0.95,
        random_state=42,
        use_memory_optimization=True,
        use_hardware_acceleration=True,
        show_progress=True
    )
    
    clusterer = MSDRClusterer(config)
    result = clusterer.fit_predict(feature_data)
    
    # Generate comprehensive metrics report
    print("\n" + "=" * 80)
    print("📈 COMPREHENSIVE METRICS REPORT")
    print("=" * 80)
    
    # Basic clustering metrics
    print("\n🎯 CLUSTERING RESULTS:")
    print(f"  • Number of Regimes Discovered: {result.n_clusters}")
    print(f"  • Success: {'✅ YES' if result.success else '❌ NO'}")
    if result.error_message:
        print(f"  • Error: {result.error_message}")
    
    # Regime distribution
    print("\n📊 REGIME DISTRIBUTION:")
    unique, counts = np.unique(result.cluster_labels, return_counts=True)
    for regime_id, count in zip(unique, counts):
        percentage = (count / len(result.cluster_labels)) * 100
        print(f"  • Regime {regime_id}: {count} samples ({percentage:.2f}%)")
    
    # Quality metrics
    print("\n🎨 QUALITY METRICS:")
    print(f"  • Silhouette Score: {result.silhouette_score:.4f}" if result.silhouette_score else "  • Silhouette Score: N/A")
    print(f"  • Calinski-Harabasz Score: {result.calinski_harabasz_score:.2f}" if result.calinski_harabasz_score else "  • Calinski-Harabasz Score: N/A")
    print(f"  • Davies-Bouldin Index: {result.davies_bouldin_score:.4f}" if result.davies_bouldin_score else "  • Davies-Bouldin Index: N/A")
    print(f"  • Noise Ratio: {result.noise_ratio:.4f}")
    
    # Model selection metrics
    print("\n📐 MODEL FIT METRICS:")
    print(f"  • Log Likelihood: {result.log_likelihood:.2f}" if result.log_likelihood else "  • Log Likelihood: N/A")
    print(f"  • AIC (Akaike Information Criterion): {result.aic:.2f}" if result.aic else "  • AIC: N/A")
    print(f"  • BIC (Bayesian Information Criterion): {result.bic:.2f}" if result.bic else "  • BIC: N/A")
    print(f"  • HQIC (Hannan-Quinn Information Criterion): {result.hqic:.2f}" if result.hqic else "  • HQIC: N/A")
    
    # Transition matrix
    print("\n🔄 TRANSITION MATRIX:")
    if result.transition_matrix is not None:
        print("  Transition Probabilities (row = from, column = to):")
        print(f"\n{result.transition_matrix}")
        
        # Calculate transition statistics
        print("\n  Transition Statistics:")
        transition_persistence = np.mean(np.diag(result.transition_matrix))
        print(f"    • Average Self-Transition Probability: {transition_persistence:.4f}")
        print(f"    • Transition Persistence: {result.transition_persistence:.4f}")
        
        # Most likely transitions
        n_regimes = result.transition_matrix.shape[0]
        for i in range(n_regimes):
            self_prob = result.transition_matrix[i, i]
            max_transition = np.max([result.transition_matrix[i, j] for j in range(n_regimes) if j != i])
            print(f"    • Regime {i}: Self-transition={self_prob:.4f}, Max-other={max_transition:.4f}")
    else:
        print("  • Transition Matrix: N/A")
    
    # Regime parameters
    print("\n⚙️ REGIME PARAMETERS:")
    if result.regime_params:
        for regime_id, params in result.regime_params.items():
            print(f"  • Regime {regime_id}:")
            if isinstance(params, dict):
                for key, value in params.items():
                    if isinstance(value, (int, float)):
                        print(f"      - {key}: {value:.4f}")
                    else:
                        print(f"      - {key}: {value}")
            else:
                print(f"      {params}")
    else:
        print("  • Regime Parameters: N/A")
    
    # Regime variances
    print("\n📉 REGIME VARIANCES:")
    if result.regime_variances is not None:
        for i, variance in enumerate(result.regime_variances):
            print(f"  • Regime {i}: Variance = {variance:.4f}")
    else:
        print("  • Regime Variances: N/A")
    
    # Regime durations
    print("\n⏱️ REGIME DURATIONS:")
    if result.regime_durations is not None:
        for i, duration in enumerate(result.regime_durations):
            avg_duration = np.mean(duration) if isinstance(duration, np.ndarray) else duration
            print(f"  • Regime {i}: Average Duration = {avg_duration:.2f} periods")
    else:
        print("  • Regime Durations: N/A")
    
    # Regime probabilities
    print("\n📊 REGIME PROBABILITIES (Smoothed):")
    if result.cluster_probabilities is not None and result.cluster_probabilities.ndim == 2:
        prob_mean = np.mean(result.cluster_probabilities, axis=0)
        prob_std = np.std(result.cluster_probabilities, axis=0)
        for i in range(result.cluster_probabilities.shape[1]):
            print(f"  • Regime {i}: Mean={prob_mean[i]:.4f}, Std={prob_std[i]:.4f}")
    
    # Processing metrics
    print("\n⚡ PROCESSING METRICS:")
    print(f"  • Processing Time: {result.processing_time:.2f} seconds")
    print(f"  • Memory Usage: {result.memory_usage_mb:.2f} MB")
    print(f"  • Features Used: {len(result.feature_names)}")
    print(f"  • Sample Size: {len(result.cluster_labels)}")
    
    # Data characteristics
    print("\n📋 DATA CHARACTERISTICS:")
    print(f"  • Input Shape: {feature_data.shape}")
    print(f"  • Feature Names: {result.feature_names[:10]}..." if len(result.feature_names) > 10 else f"  • Feature Names: {result.feature_names}")
    
    # Additional metadata
    if result.metadata:
        print("\n📝 ADDITIONAL METADATA:")
        for key, value in result.metadata.items():
            if isinstance(value, (int, float)):
                print(f"  • {key}: {value:.4f}")
            else:
                print(f"  • {key}: {value}")
    
    # Regime characteristics summary
    print("\n📈 REGIME CHARACTERISTICS SUMMARY:")
    if result.regime_variances is not None and result.regime_params:
        for i in range(result.n_clusters):
            regime_mask = result.cluster_labels == i
            regime_size = np.sum(regime_mask)
            regime_variance = result.regime_variances[i] if i < len(result.regime_variances) else None
            
            print(f"\n  Regime {i}:")
            print(f"    • Size: {regime_size} samples ({regime_size/len(result.cluster_labels)*100:.2f}%)")
            if regime_variance:
                print(f"    • Variance: {regime_variance:.4f}")
            if result.regime_params and str(i) in result.regime_params:
                params = result.regime_params[str(i)]
                if isinstance(params, dict):
                    if 'mean' in params:
                        print(f"    • Mean: {params['mean']:.4f}")
                    if 'std' in params:
                        print(f"    • Std: {params['std']:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ METRICS REPORT COMPLETE")
    print("=" * 80)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

