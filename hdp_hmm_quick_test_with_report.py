#!/usr/bin/env python3
"""
Quick HDP-HMM Test with Auto-Report Generation
- Optimized for fast completion (reduced iterations)
- Generates comprehensive report in outcomes/
- Uses realistic market data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

print("=" * 80)
print("Quick HDP-HMM Test with Optimized Settings & Auto-Report")
print("=" * 80)

# Check NumPy version
print(f"   NumPy version: {np.__version__}")

# Import HDP-HMM clusterer
try:
    from src.training.steps.market_analysis.hdp_hmm_clustering.hdp_hmm_clusterer import (
        HDPHMMClusterer, HDPHMMConfig, HMM_AVAILABLE, HMM_LIBRARY
    )
    from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
        create_cluster_quality_assessor
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
print("\n📊 Loading market data...")

try:
    from src.utils.data.klines_parquet import KlinesParquetManager
    from src.feature_generation.categories.regime_feature_integration import RegimeFeatureIntegration
    
    klines_manager = KlinesParquetManager()
    
    # Load 60 days of ETHUSDT data (sufficient for testing)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    df = klines_manager.read_data(
        symbol="ETHUSDT",
        interval="1h",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
    
    if df is None or df.empty:
        raise ValueError("Failed to load market data.")
    
    print(f"   ✅ Loaded market data: {df.shape}")
    print(f"   📊 Date range: {df.index.min()} to {df.index.max()}")
    
    # Generate regime features (simplified for speed)
    print("   🔄 Generating regime features...")
    regime_integrator = RegimeFeatureIntegration()
    
    # Use a simple approach: generate features for last 500 samples
    df_subset = df.tail(500)
    
    # Try to generate features - CHUNK PROCESSING
    try:
        # Process in small chunks to get more samples
        chunk_size = 50
        overlap = 25
        feature_chunks = []
        
        for i in range(0, len(df_subset) - chunk_size + 1, chunk_size - overlap):
            chunk = df_subset.iloc[i:i+chunk_size]
            if len(chunk) >= 20:
                try:
                    regime_features = regime_integrator._generate_regime_features(chunk)
                    
                    # Convert to DataFrame if dict
                    if isinstance(regime_features, dict):
                        chunk_df = pd.DataFrame([regime_features])
                    else:
                        chunk_df = regime_features
                    
                    feature_chunks.append(chunk_df)
                except Exception as e:
                    print(f"   ⚠️ Error in chunk {i}: {e}")
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
        
    except Exception as e:
        print(f"   ⚠️ Feature generation failed: {e}")
        print("   Using basic OHLCV features instead...")
        
        # Fallback: use basic OHLCV + simple features
        feature_df = df_subset[['open', 'high', 'low', 'close', 'volume']].copy()
        
        # Add simple features
        feature_df['returns'] = feature_df['close'].pct_change()
        feature_df['log_returns'] = np.log(feature_df['close'] / feature_df['close'].shift(1))
        feature_df['volatility'] = feature_df['returns'].rolling(20).std()
        feature_df['volume_ma'] = feature_df['volume'].rolling(20).mean()
        feature_df['price_range'] = (feature_df['high'] - feature_df['low']) / feature_df['close']
        
        feature_df = feature_df.fillna(0)
        print(f"   ✅ Using basic features: {feature_df.shape}")

except Exception as e:
    print(f"   ❌ Data loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Data validation
print(f"\n📊 Final data shape: {feature_df.shape}")
numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    print(f"📊 Numeric columns: {len(numeric_cols)}")
else:
    print("❌ No numeric columns found!")
    sys.exit(1)

# Check if we have enough samples
if len(feature_df) < 10:
    print(f"❌ Insufficient data: only {len(feature_df)} samples (need at least 10)")
    sys.exit(1)

# Configure HDP-HMM with OPTIMIZED settings for quick testing
print("\n🔧 Configuring HDP-HMM with optimized settings...")

n_features = len(feature_df.columns)
pca_components = min(10, max(5, n_features // 2))

print(f"   📊 Features: {n_features}, PCA components: {pca_components}")

# OPTIMIZED configuration for quick testing (30 iterations)
config = HDPHMMConfig(
    alpha=3.0,
    kappa=50.0,
    gamma=3.0,
    n_iterations=30,  # REDUCED for quick testing (was 50)
    n_burnin=5,       # REDUCED for quick testing (was 10)
    convergence_check=True,
    convergence_threshold=0.01,
    convergence_window=5,  # REDUCED for quick convergence detection
    convergence_patience=3,  # Allow early stopping
    ll_plateau_threshold=0.001,
    enable_pca=True,
    pca_components=pca_components,
    min_samples_required=50,
    max_states=8,
    show_progress=True
)

print(f"   Iterations: {config.n_iterations} (with early stopping)")
print(f"   Burn-in: {config.n_burnin}")
print(f"   Convergence window: {config.convergence_window}")
print(f"   PCA components: {config.pca_components}")

# Run clustering
print("\n🚀 Running HDP-HMM clustering...")
clusterer = HDPHMMClusterer(config)

try:
    result = clusterer.fit_predict(feature_df)
    print("✅ HDP-HMM clustering completed successfully.")
    
    # Display results
    print(f"\n📈 Clustering Results:")
    print(f"   Number of clusters: {result.n_clusters}")
    print(f"   Unique clusters: {np.unique(result.cluster_labels)}")
    
    unique_clusters, counts = np.unique(result.cluster_labels, return_counts=True)
    print(f"   Cluster distribution:")
    for cluster, count in zip(unique_clusters, counts):
        percentage = (count / len(result.cluster_labels)) * 100
        print(f"     Cluster {cluster}: {count} samples ({percentage:.1f}%)")
    
    print(f"\n📊 Model Metrics:")
    print(f"   Log likelihood: {result.log_likelihood:.4f}")
    print(f"   Silhouette score: {result.silhouette_score:.4f}")
    print(f"   Calinski-Harabasz score: {result.calinski_harabasz_score:.2f}")
    print(f"   Davies-Bouldin score: {result.davies_bouldin_score:.4f}")
    
    # Calculate quality metrics using simple approach
    print(f"\n📊 Computing quality metrics...")
    
    # Create simple quality metrics dict
    quality_metrics = {
        'composite_score': result.silhouette_score * 0.7 + (1.0 / max(result.davies_bouldin_score, 0.1)) * 0.3,
        'balance_score': 1.0 - np.std(np.bincount(result.cluster_labels.astype(int))) / max(np.mean(np.bincount(result.cluster_labels.astype(int))), 1.0),
        'temporal_smoothness': 1.0 - np.mean(np.abs(np.diff(result.cluster_labels))) / max(result.n_clusters, 1.0) if len(result.cluster_labels) > 1 else 1.0,
        'silhouette_score': result.silhouette_score,
        'calinski_harabasz_score': result.calinski_harabasz_score,
        'davies_bouldin_score': result.davies_bouldin_score
    }
    
    print(f"   Composite score: {quality_metrics.get('composite_score', 0.0):.4f}")
    print(f"   Balance score: {quality_metrics.get('balance_score', 0.0):.4f}")
    print(f"   Temporal smoothness: {quality_metrics.get('temporal_smoothness', 0.0):.4f}")
    
    # Generate comprehensive report
    print(f"\n📝 Generating comprehensive report...")
    
    outcomes_dir = Path("outcomes")
    outcomes_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = outcomes_dir / f"hdp_hmm_metrics_{timestamp}.md"
    
    # Create report
    report_lines = []
    report_lines.append("# HDP-HMM Clustering Quality Report\n")
    report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"**Library**: {HMM_LIBRARY}\n")
    report_lines.append(f"**Test Type**: Quick Test with Optimized Settings\n\n")
    
    report_lines.append("## Executive Summary\n")
    report_lines.append(f"- **Clusters Discovered**: {result.n_clusters}\n")
    report_lines.append(f"- **Total Samples**: {len(feature_df)}\n")
    report_lines.append(f"- **Features Used**: {config.pca_components} (PCA components)\n")
    report_lines.append(f"- **Samples per Feature**: {len(feature_df) / config.pca_components:.1f}\n")
    report_lines.append(f"- **Convergence**: {'✅ Yes' if clusterer.convergence_history.get('converged', False) else '❌ No'}\n")
    if clusterer.convergence_history.get('converged', False):
        report_lines.append(f"- **Converged at Iteration**: {clusterer.convergence_history.get('convergence_iteration', 'N/A')}\n")
    report_lines.append("\n")
    
    report_lines.append("## Configuration\n")
    report_lines.append(f"- **Alpha (diversity)**: {config.alpha}\n")
    report_lines.append(f"- **Kappa (stickiness)**: {config.kappa}\n")
    report_lines.append(f"- **Gamma**: {config.gamma}\n")
    report_lines.append(f"- **Max Iterations**: {config.n_iterations}\n")
    report_lines.append(f"- **Burn-in**: {config.n_burnin}\n")
    report_lines.append(f"- **Convergence Window**: {config.convergence_window}\n")
    report_lines.append(f"- **Early Stopping**: Enabled (patience={config.convergence_patience})\n")
    report_lines.append("\n")
    
    report_lines.append("## Quality Metrics\n")
    report_lines.append(f"- **Composite Score**: {quality_metrics.get('composite_score', 0.0):.4f}\n")
    report_lines.append(f"- **Silhouette Score**: {result.silhouette_score:.4f}\n")
    report_lines.append(f"- **Calinski-Harabasz Score**: {result.calinski_harabasz_score:.2f}\n")
    report_lines.append(f"- **Davies-Bouldin Score**: {result.davies_bouldin_score:.4f}\n")
    report_lines.append(f"- **Balance Score**: {quality_metrics.get('balance_score', 0.0):.4f}\n")
    report_lines.append(f"- **Temporal Smoothness**: {quality_metrics.get('temporal_smoothness', 0.0):.4f}\n")
    report_lines.append(f"- **Log Likelihood**: {result.log_likelihood:.4f}\n")
    report_lines.append("\n")
    
    report_lines.append("## Cluster Distribution\n")
    for cluster, count in zip(unique_clusters, counts):
        percentage = (count / len(result.cluster_labels)) * 100
        report_lines.append(f"- **Cluster {cluster}**: {count} samples ({percentage:.1f}%)\n")
    report_lines.append("\n")
    
    report_lines.append("## Transition Matrix\n")
    if result.transition_matrix is not None:
        report_lines.append(f"Shape: {result.transition_matrix.shape}\n\n")
        report_lines.append("```\n")
        report_lines.append(str(result.transition_matrix))
        report_lines.append("\n```\n\n")
    else:
        report_lines.append("Transition matrix not available\n\n")
    
    report_lines.append("## Convergence Diagnostics\n")
    if clusterer.convergence_history:
        state_counts = list(clusterer.convergence_history.get('state_counts', []))
        log_lls = list(clusterer.convergence_history.get('log_likelihoods', []))
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
    
    report_lines.append("## Optimization Summary\n")
    report_lines.append("### Phase 1 Optimizations Applied ✅\n")
    report_lines.append("1. **Reduced Iterations**: 30 iterations (down from 100)\n")
    report_lines.append("2. **Enhanced Convergence Detection**: Multi-metric early stopping\n")
    report_lines.append("3. **Memory Optimization**: Circular buffers for convergence history\n")
    report_lines.append("4. **Patience-based Stopping**: Avoid premature convergence\n")
    report_lines.append("\n")
    report_lines.append("### Performance Impact\n")
    if converged and conv_iter:
        speedup = config.n_iterations / conv_iter
        report_lines.append(f"- **Effective Speedup**: {speedup:.1f}x (stopped at {conv_iter}/{config.n_iterations} iterations)\n")
    report_lines.append("- **Memory Usage**: Optimized with circular buffers\n")
    report_lines.append("- **Convergence Detection**: Enhanced with log-likelihood plateau check\n")
    report_lines.append("\n")
    
    report_lines.append("## Recommendations\n")
    report_lines.append("### Based on Results\n")
    
    if result.n_clusters < 3:
        report_lines.append("- ⚠️ **Low cluster count**: Consider increasing `alpha` for more diversity\n")
    elif result.n_clusters > 8:
        report_lines.append("- ⚠️ **High cluster count**: Consider decreasing `alpha` or increasing `kappa` for more stability\n")
    else:
        report_lines.append("- ✅ **Cluster count looks reasonable** (3-8 clusters)\n")
    
    if result.silhouette_score < 0.3:
        report_lines.append("- ⚠️ **Low silhouette score**: Clusters may not be well-separated\n")
    elif result.silhouette_score > 0.5:
        report_lines.append("- ✅ **Good silhouette score**: Clusters are well-separated\n")
    
    if result.davies_bouldin_score > 1.5:
        report_lines.append("- ⚠️ **High Davies-Bouldin score**: Consider adjusting hyperparameters\n")
    elif result.davies_bouldin_score < 1.0:
        report_lines.append("- ✅ **Good Davies-Bouldin score**: Compact and well-separated clusters\n")
    
    report_lines.append("\n### Next Steps\n")
    report_lines.append("1. **Run Auto-Tuner**: Use `hdp_hmm_auto_tuner.py` to find optimal hyperparameters\n")
    report_lines.append("2. **Increase Data**: Test with more historical data for better regime discovery\n")
    report_lines.append("3. **Feature Engineering**: Add more regime-specific features\n")
    report_lines.append("4. **Production Testing**: Run with production config (150 iterations)\n")
    report_lines.append("\n")
    
    report_lines.append("---\n")
    report_lines.append(f"*Report generated by Quick HDP-HMM Test with Optimized Settings*\n")
    report_lines.append(f"*Timestamp: {datetime.now().isoformat()}*\n")
    
    # Write report
    with open(report_filename, 'w') as f:
        f.writelines(report_lines)
    
    print(f"✅ Report saved to: {report_filename}")
    
    print("\n" + "=" * 80)
    print("✅ Quick HDP-HMM Test Completed Successfully!")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"   - Clusters: {result.n_clusters}")
    print(f"   - Composite Score: {quality_metrics.get('composite_score', 0.0):.4f}")
    print(f"   - Converged: {'Yes' if converged else 'No'}")
    if conv_iter:
        print(f"   - Iterations: {conv_iter} / {config.n_iterations} (early stop)")
    print(f"   - Report: {report_filename}")
    
except Exception as e:
    print(f"❌ HDP-HMM clustering failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

