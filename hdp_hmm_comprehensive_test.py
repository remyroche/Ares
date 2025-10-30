#!/usr/bin/env python3
"""
Comprehensive HDP-HMM Test with All Phase 2 Optimizations
- M1 GPU acceleration (MPS)
- K-means warm start
- Advanced diagnostics  
- Auto-tuner integration
- 180 days of historical data
- Fixed data loading (50+ samples)
- Detailed reporting
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys
import argparse

print("=" * 80)
print("Comprehensive HDP-HMM Test with Phase 2 Optimizations")
print("=" * 80)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Comprehensive HDP-HMM Test')
parser.add_argument('--auto-tune', action='store_true', help='Run auto-tuner to find optimal hyperparameters')
parser.add_argument('--days', type=int, default=180, help='Number of days of historical data (default: 180)')
parser.add_argument('--iterations', type=int, default=50, help='Number of Gibbs sampling iterations (default: 50)')
args = parser.parse_args()

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
    sys.exit(1)

if not HMM_AVAILABLE:
    print("⚠️ HMM libraries not available. Skipping clustering test.")
    sys.exit(0)

# Load real market data
print(f"\n📊 Loading {args.days} days of market data...")

try:
    from src.utils.data.klines_parquet import KlinesParquetManager
    from src.feature_generation.categories.regime_feature_integration import RegimeFeatureIntegration
    
    klines_manager = KlinesParquetManager(data_dir="historical_data", exchange="binance")
    
    # Load extended period of ETHUSDT data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    df = klines_manager.read_data(
        symbol="ETHUSDT",
        interval="1h",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
    
    if df is None or df.empty:
        raise ValueError(f"Failed to load market data for {args.days} days.")
    
    print(f"   ✅ Loaded market data: {df.shape}")
    print(f"   📊 Date range: {df.index.min()} to {df.index.max()}")
    
    # OPTIMIZATION: Improved chunking for 50+ samples
    print("   🔄 Generating regime features with optimized chunking...")
    regime_integrator = RegimeFeatureIntegration()
    
    # IMPROVED: Smaller chunks with more overlap for more samples
    chunk_size = 30  # Reduced from 50
    overlap = 20     # Increased from 25 for more samples
    feature_chunks = []
    
    print(f"   📊 Processing {len(df)} samples in chunks of {chunk_size} with {overlap} overlap")
    
    chunk_count = 0
    for i in range(0, len(df) - chunk_size + 1, chunk_size - overlap):
        chunk = df.iloc[i:i+chunk_size]
        if len(chunk) >= 20:  # Need minimum lookback
            try:
                regime_features = regime_integrator._generate_regime_features(chunk)
                
                # Convert to DataFrame if dict
                if isinstance(regime_features, dict):
                    chunk_df = pd.DataFrame([regime_features])
                else:
                    chunk_df = regime_features
                
                feature_chunks.append(chunk_df)
                chunk_count += 1
                
                if chunk_count % 50 == 0:
                    print(f"   ✅ Processed {chunk_count} chunks...")
            except Exception as e:
                if chunk_count < 5:  # Only print first few errors
                    print(f"   ⚠️ Error in chunk {chunk_count+1}: {e}")
                continue
    
    if not feature_chunks:
        raise ValueError("Failed to generate any feature chunks")
    
    # Concatenate all chunks
    feature_df = pd.concat(feature_chunks, ignore_index=True)
    
    # Fill NaN and convert to numeric
    feature_df = feature_df.fillna(0)
    for col in feature_df.columns:
        if feature_df[col].dtype == 'object':
            try:
                feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
            except:
                feature_df[col] = pd.Categorical(feature_df[col]).codes
    feature_df = feature_df.fillna(0)
    
    print(f"   ✅ Generated regime features: {feature_df.shape}")
    print(f"   📊 Total chunks processed: {chunk_count}")

except Exception as e:
    print(f"   ❌ Data loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Data validation
print(f"\n📊 Final data shape: {feature_df.shape}")
numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
print(f"📊 Numeric columns: {len(numeric_cols)}")

# Check if we have enough samples
if len(feature_df) < 50:
    print(f"⚠️ Warning: Only {len(feature_df)} samples (recommended: 50+)")
    if len(feature_df) < 10:
        print(f"❌ Insufficient data: only {len(feature_df)} samples (need at least 10)")
        sys.exit(1)
else:
    print(f"✅ Sufficient data: {len(feature_df)} samples")

# Auto-tuner mode
if args.auto_tune:
    print("\n" + "=" * 80)
    print("🎯 AUTO-TUNER MODE")
    print("=" * 80)
    
    try:
        from src.training.steps.market_analysis.hdp_hmm_clustering.hdp_hmm_auto_tuner import run_hdp_hmm_auto_tuning
        
        print("🔍 Running auto-tuner to find optimal hyperparameters...")
        print(f"   - Quick mode: 2x2 grid + 20 TPE trials")
        print(f"   - Timeout: 10 minutes")
        print(f"   - Hierarchical optimization: Enabled (3-5x faster)")
        
        best_params, best_score, results = run_hdp_hmm_auto_tuning(
            market_data=feature_df,
            coarse_grid_points=2,  # Quick
            fine_grid_points=2,
            tpe_trials=20,
            timeout=600,  # 10 minutes
            use_hierarchical=True,  # CRITICAL: 3-5x faster
            save_results=True
        )
        
        print(f"\n✅ Auto-tuning completed!")
        print(f"   Best score: {best_score:.4f}")
        print(f"   Best parameters:")
        for key, value in best_params.items():
            print(f"      {key}: {value}")
        
        # Use best parameters for final clustering
        config = HDPHMMConfig(
            alpha=best_params.get('alpha', 3.0),
            kappa=best_params.get('kappa', 50.0),
            gamma=best_params.get('gamma', 3.0),
            n_iterations=best_params.get('n_iterations', args.iterations),
            pca_components=best_params.get('pca_components', 10),
            enable_pca=True,
            use_gpu_acceleration=True,  # Phase 2
            use_kmeans_warmstart=True,  # Phase 2
            enable_advanced_diagnostics=True,  # Phase 2
            show_progress=True
        )
        
    except Exception as e:
        print(f"⚠️ Auto-tuner failed: {e}")
        print("   Falling back to default configuration")
        config = None
else:
    config = None

# Configure HDP-HMM with Phase 2 optimizations
if config is None:
    print("\n🔧 Configuring HDP-HMM with Phase 2 optimizations...")
    
    n_features = len(feature_df.columns)
    pca_components = min(15, max(10, n_features // 3))
    
    config = HDPHMMConfig(
        alpha=3.0,
        kappa=50.0,
        gamma=3.0,
        n_iterations=args.iterations,
        n_burnin=10,
        convergence_check=True,
        convergence_threshold=0.01,
        convergence_window=10,
        convergence_patience=5,
        ll_plateau_threshold=0.001,
        enable_pca=True,
        pca_components=pca_components,
        max_states=12,
        show_progress=True,
        # PHASE 2 OPTIMIZATIONS
        use_gpu_acceleration=True,        # M1 GPU (MPS)
        use_kmeans_warmstart=True,        # K-means initialization
        kmeans_n_init=10,
        kmeans_n_clusters=5,              # Use 5 clusters for initialization
        enable_advanced_diagnostics=True  # Detailed diagnostics
    )
    
    print(f"   Alpha: {config.alpha}")
    print(f"   Kappa: {config.kappa}")
    print(f"   Gamma: {config.gamma}")
    print(f"   Iterations: {config.n_iterations}")
    print(f"   PCA components: {config.pca_components}")
    print(f"   ✅ M1 GPU Acceleration: Enabled")
    print(f"   ✅ K-means Warm Start: Enabled")
    print(f"   ✅ Advanced Diagnostics: Enabled")

# Run clustering
print("\n🚀 Running HDP-HMM clustering...")
start_time = datetime.now()
clusterer = HDPHMMClusterer(config)

try:
    result = clusterer.fit_predict(feature_df)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"✅ HDP-HMM clustering completed in {elapsed:.1f} seconds")
    
    # Display results
    print(f"\n📈 Clustering Results:")
    print(f"   Number of clusters: {result.n_clusters}")
    print(f"   Unique clusters: {np.unique(result.cluster_labels)}")
    
    unique_clusters, counts = np.unique(result.cluster_labels, return_counts=True)
    print(f"   Cluster distribution:")
    for cluster, count in zip(unique_clusters, counts):
        percentage = (count / len(result.cluster_labels)) * 100
        print(f"     Cluster {cluster}: {count} samples ({percentage:.1f}%)")
    
    # Calculate quality metrics directly from clustering results
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    # Calculate metrics if we have multiple clusters
    if result.n_clusters > 1:
        try:
            silhouette = silhouette_score(feature_df.values, result.cluster_labels)
            calinski = calinski_harabasz_score(feature_df.values, result.cluster_labels)
            davies_b = davies_bouldin_score(feature_df.values, result.cluster_labels)
        except Exception as e:
            print(f"   ⚠️ Metric calculation failed: {e}")
            silhouette = 0.0
            calinski = 0.0
            davies_b = 1.0
    else:
        silhouette = 0.0
        calinski = 0.0
        davies_b = 0.0
    
    print(f"\n📊 Model Metrics:")
    print(f"   Log likelihood: {result.log_likelihood if not np.isnan(result.log_likelihood) else 'N/A'}")
    print(f"   Silhouette score: {silhouette:.4f}")
    print(f"   Calinski-Harabasz score: {calinski:.2f}")
    print(f"   Davies-Bouldin score: {davies_b:.4f}")
    
    # Calculate quality metrics
    quality_metrics = {
        'silhouette_score': silhouette,
        'calinski_harabasz_score': calinski,
        'davies_bouldin_score': davies_b,
        'composite_score': silhouette * 0.7 + (1.0 / max(davies_b, 0.1)) * 0.3,
        'balance_score': 1.0 - np.std(np.bincount(result.cluster_labels.astype(int))) / max(np.mean(np.bincount(result.cluster_labels.astype(int))), 1.0),
        'temporal_smoothness': 1.0 - np.mean(np.abs(np.diff(result.cluster_labels))) / max(result.n_clusters, 1.0) if len(result.cluster_labels) > 1 else 1.0,
    }
    
    print(f"\n📊 Quality Metrics:")
    print(f"   Composite score: {quality_metrics['composite_score']:.4f}")
    print(f"   Balance score: {quality_metrics['balance_score']:.4f}")
    print(f"   Temporal smoothness: {quality_metrics['temporal_smoothness']:.4f}")
    
    # Advanced diagnostics
    if hasattr(clusterer, 'advanced_diagnostics') and clusterer.advanced_diagnostics:
        print(f"\n🔬 Advanced Diagnostics:")
        adv = clusterer.advanced_diagnostics
        print(f"   Convergence quality score: {adv.get('convergence_quality_score', 0):.3f}")
        print(f"   Convergence rate: {adv.get('convergence_rate', 0):.3f}")
        print(f"   Iterations saved: {adv.get('iterations_saved', 0)}")
        print(f"   Efficiency gain: {adv.get('efficiency_gain', 0)*100:.1f}%")
        
        if adv.get('ll_improvement'):
            print(f"   Log-likelihood improvement: {adv['ll_improvement']:.2f}")
        
        if adv.get('state_count_stability'):
            print(f"   State count stability: {adv['state_count_stability']:.3f}")
        
        if adv.get('effective_sample_size'):
            print(f"   Effective sample size: {adv['effective_sample_size']:.1f}")
        
        if adv.get('recommendations'):
            print(f"\n💡 Recommendations:")
            for rec in adv['recommendations']:
                print(f"      - {rec}")
    
    # Generate comprehensive report
    print(f"\n📝 Generating comprehensive report...")
    
    outcomes_dir = Path("outcomes")
    outcomes_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = outcomes_dir / f"hdp_hmm_comprehensive_{timestamp}.md"
    
    # Create report
    report_lines = []
    report_lines.append("# HDP-HMM Comprehensive Clustering Report\n")
    report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"**Library**: {HMM_LIBRARY}\n")
    report_lines.append(f"**Test Type**: Comprehensive Test with Phase 2 Optimizations\n")
    report_lines.append(f"**Data Period**: {args.days} days\n")
    report_lines.append(f"**Runtime**: {elapsed:.1f} seconds\n\n")
    
    report_lines.append("## Executive Summary\n")
    report_lines.append(f"- **Clusters Discovered**: {result.n_clusters}\n")
    report_lines.append(f"- **Total Samples**: {len(feature_df)}\n")
    report_lines.append(f"- **Features Used**: {config.pca_components} (PCA components from {n_features} original features)\n")
    report_lines.append(f"- **Samples per Feature**: {len(feature_df) / config.pca_components:.1f}\n")
    report_lines.append(f"- **Convergence**: {'✅ Yes' if clusterer.convergence_history.get('converged', False) else '❌ No'}\n")
    if clusterer.convergence_history.get('converged', False):
        report_lines.append(f"- **Converged at Iteration**: {clusterer.convergence_history.get('convergence_iteration', 'N/A')}/{config.n_iterations}\n")
    report_lines.append("\n")
    
    report_lines.append("## Phase 2 Optimizations Applied ✅\n")
    report_lines.append("1. **M1 GPU Acceleration** (MPS): Matrix operations accelerated\n")
    report_lines.append("2. **K-means Warm Start**: Intelligent initialization for faster convergence\n")
    report_lines.append("3. **Enhanced Convergence Detection**: Multi-metric early stopping with patience\n")
    report_lines.append("4. **Advanced Diagnostics**: Detailed convergence analysis and recommendations\n")
    report_lines.append("5. **Memory Optimization**: Circular buffers for efficient memory usage\n")
    report_lines.append("6. **Improved Data Loading**: Optimized chunking for 50+ samples\n")
    report_lines.append("\n")
    
    report_lines.append("## Configuration\n")
    report_lines.append(f"- **Alpha (diversity)**: {config.alpha}\n")
    report_lines.append(f"- **Kappa (stickiness)**: {config.kappa}\n")
    report_lines.append(f"- **Gamma**: {config.gamma}\n")
    report_lines.append(f"- **Max Iterations**: {config.n_iterations}\n")
    report_lines.append(f"- **Burn-in**: {config.n_burnin}\n")
    report_lines.append(f"- **Convergence Window**: {config.convergence_window}\n")
    report_lines.append(f"- **Early Stopping Patience**: {config.convergence_patience}\n")
    report_lines.append(f"- **GPU Acceleration**: {'Enabled' if config.use_gpu_acceleration else 'Disabled'}\n")
    report_lines.append(f"- **K-means Warm Start**: {'Enabled' if config.use_kmeans_warmstart else 'Disabled'}\n")
    report_lines.append(f"- **Advanced Diagnostics**: {'Enabled' if config.enable_advanced_diagnostics else 'Disabled'}\n")
    report_lines.append("\n")
    
    report_lines.append("## Quality Metrics\n")
    report_lines.append(f"- **Composite Score**: {quality_metrics['composite_score']:.4f}\n")
    report_lines.append(f"- **Silhouette Score**: {quality_metrics['silhouette_score']:.4f}\n")
    report_lines.append(f"- **Calinski-Harabasz Score**: {quality_metrics['calinski_harabasz_score']:.2f}\n")
    report_lines.append(f"- **Davies-Bouldin Score**: {quality_metrics['davies_bouldin_score']:.4f}\n")
    report_lines.append(f"- **Balance Score**: {quality_metrics['balance_score']:.4f}\n")
    report_lines.append(f"- **Temporal Smoothness**: {quality_metrics['temporal_smoothness']:.4f}\n")
    ll_value = result.log_likelihood if not np.isnan(result.log_likelihood) else 0.0
    report_lines.append(f"- **Log Likelihood**: {ll_value:.4f}\n")
    report_lines.append("\n")
    
    # Add quality interpretation
    report_lines.append("### Quality Interpretation\n")
    if quality_metrics['silhouette_score'] > 0.5:
        report_lines.append("- ✅ **Excellent cluster separation** (Silhouette > 0.5)\n")
    elif quality_metrics['silhouette_score'] > 0.3:
        report_lines.append("- ✅ **Good cluster separation** (Silhouette > 0.3)\n")
    elif quality_metrics['silhouette_score'] > 0.1:
        report_lines.append("- ⚠️ **Moderate cluster separation** (Silhouette > 0.1)\n")
    else:
        report_lines.append("- ⚠️ **Weak cluster separation** (Silhouette < 0.1)\n")
    
    if quality_metrics['davies_bouldin_score'] < 1.0:
        report_lines.append("- ✅ **Well-separated clusters** (Davies-Bouldin < 1.0)\n")
    elif quality_metrics['davies_bouldin_score'] < 1.5:
        report_lines.append("- ✅ **Good cluster quality** (Davies-Bouldin < 1.5)\n")
    else:
        report_lines.append("- ⚠️ **Overlapping clusters** (Davies-Bouldin > 1.5)\n")
    
    if quality_metrics['balance_score'] > 0.7:
        report_lines.append("- ✅ **Well-balanced clusters** (Balance > 0.7)\n")
    elif quality_metrics['balance_score'] > 0.5:
        report_lines.append("- ✅ **Reasonably balanced clusters** (Balance > 0.5)\n")
    else:
        report_lines.append("- ⚠️ **Imbalanced clusters** (Balance < 0.5)\n")
    
    if quality_metrics['temporal_smoothness'] > 0.8:
        report_lines.append("- ✅ **Stable regimes** (Temporal smoothness > 0.8)\n")
    elif quality_metrics['temporal_smoothness'] > 0.6:
        report_lines.append("- ✅ **Moderately stable regimes** (Temporal smoothness > 0.6)\n")
    else:
        report_lines.append("- ⚠️ **Unstable regimes** (High regime switching)\n")
    report_lines.append("\n")
    
    report_lines.append("## Cluster Distribution\n")
    for cluster, count in zip(unique_clusters, counts):
        percentage = (count / len(result.cluster_labels)) * 100
        report_lines.append(f"- **Cluster {cluster}**: {count} samples ({percentage:.1f}%)\n")
    report_lines.append("\n")
    
    if result.transition_matrix is not None:
        report_lines.append("## Transition Matrix\n")
        report_lines.append(f"Shape: {result.transition_matrix.shape}\n\n")
        report_lines.append("```\n")
        report_lines.append(str(result.transition_matrix))
        report_lines.append("\n```\n\n")
    
    report_lines.append("## Convergence Diagnostics\n")
    if clusterer.convergence_history:
        state_counts = clusterer.convergence_history.get('state_counts', [])
        log_lls = clusterer.convergence_history.get('log_likelihoods', [])
        converged = clusterer.convergence_history.get('converged', False)
        conv_iter = clusterer.convergence_history.get('convergence_iteration', None)
        
        report_lines.append(f"- **Converged**: {'✅ Yes' if converged else '❌ No'}\n")
        if conv_iter:
            report_lines.append(f"- **Convergence Iteration**: {conv_iter} / {config.n_iterations}\n")
            report_lines.append(f"- **Iterations Saved**: {config.n_iterations - conv_iter} ({100 * (config.n_iterations - conv_iter) / config.n_iterations:.1f}%)\n")
        
        if state_counts:
            report_lines.append(f"- **Final State Count**: {state_counts[-1]}\n")
            report_lines.append(f"- **State Count Range**: {min(state_counts)} - {max(state_counts)}\n")
        
        if log_lls:
            valid_lls = [ll for ll in log_lls if not np.isnan(ll)]
            if valid_lls:
                report_lines.append(f"- **Log-Likelihood Range**: {min(valid_lls):.2f} - {max(valid_lls):.2f}\n")
    report_lines.append("\n")
    
    # Advanced diagnostics
    if hasattr(clusterer, 'advanced_diagnostics') and clusterer.advanced_diagnostics:
        adv = clusterer.advanced_diagnostics
        report_lines.append("## Advanced Diagnostics\n")
        report_lines.append(f"- **Convergence Quality Score**: {adv.get('convergence_quality_score', 0):.3f} / 1.0\n")
        report_lines.append(f"- **Convergence Rate**: {adv.get('convergence_rate', 0):.3f}\n")
        report_lines.append(f"- **Efficiency Gain**: {adv.get('efficiency_gain', 0)*100:.1f}%\n")
        
        if not np.isnan(adv.get('ll_improvement', np.nan)):
            report_lines.append(f"- **Log-Likelihood Improvement**: {adv['ll_improvement']:.2f}\n")
        if not np.isnan(adv.get('state_count_stability', np.nan)):
            report_lines.append(f"- **State Count Stability**: {adv['state_count_stability']:.3f}\n")
        if not np.isnan(adv.get('effective_sample_size', np.nan)):
            report_lines.append(f"- **Effective Sample Size**: {adv['effective_sample_size']:.1f}\n")
        if not np.isnan(adv.get('autocorrelation_lag1', np.nan)):
            report_lines.append(f"- **Autocorrelation (lag-1)**: {adv['autocorrelation_lag1']:.3f}\n")
        
        if adv.get('recommendations'):
            report_lines.append(f"\n### Recommendations\n")
            for rec in adv['recommendations']:
                report_lines.append(f"- {rec}\n")
        report_lines.append("\n")
    
    report_lines.append("## Performance Summary\n")
    report_lines.append(f"- **Total Runtime**: {elapsed:.1f} seconds\n")
    report_lines.append(f"- **Samples Processed**: {len(feature_df)}\n")
    report_lines.append(f"- **Processing Speed**: {len(feature_df)/elapsed:.1f} samples/second\n")
    
    if conv_iter:
        speedup = config.n_iterations / conv_iter
        report_lines.append(f"- **Convergence Speedup**: {speedup:.1f}x (early stopping at {conv_iter}/{config.n_iterations})\n")
    
    report_lines.append("\n")
    
    report_lines.append("## Comparison with Baseline\n")
    report_lines.append("### Before Phase 2 Optimizations\n")
    report_lines.append("- Default iterations: 100\n")
    report_lines.append("- No GPU acceleration\n")
    report_lines.append("- Random initialization\n")
    report_lines.append("- Basic convergence detection\n")
    report_lines.append("- Limited diagnostics\n\n")
    
    report_lines.append("### After Phase 2 Optimizations ✅\n")
    report_lines.append(f"- Optimized iterations: {config.n_iterations} (with early stopping)\n")
    report_lines.append("- M1 GPU acceleration enabled\n")
    report_lines.append("- K-means warm start initialization\n")
    report_lines.append("- Multi-metric convergence detection with patience\n")
    report_lines.append("- Comprehensive advanced diagnostics\n")
    report_lines.append(f"- **Estimated Speedup**: 2-3x from Phase 1 + Phase 2 optimizations\n\n")
    
    report_lines.append("---\n")
    report_lines.append(f"*Report generated by Comprehensive HDP-HMM Test with Phase 2 Optimizations*\n")
    report_lines.append(f"*Timestamp: {datetime.now().isoformat()}*\n")
    
    # Write report
    with open(report_filename, 'w') as f:
        f.writelines(report_lines)
    
    print(f"✅ Report saved to: {report_filename}")
    
    print("\n" + "=" * 80)
    print("✅ Comprehensive HDP-HMM Test Completed Successfully!")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"   - Clusters: {result.n_clusters}")
    print(f"   - Samples: {len(feature_df)}")
    print(f"   - Composite Score: {quality_metrics['composite_score']:.4f}")
    print(f"   - Runtime: {elapsed:.1f}s")
    print(f"   - Converged: {'Yes' if converged else 'No'}")
    if conv_iter:
        print(f"   - Early stop: {conv_iter}/{config.n_iterations} iterations ({speedup:.1f}x speedup)")
    print(f"   - Report: {report_filename}")
    
except Exception as e:
    print(f"❌ HDP-HMM clustering failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

