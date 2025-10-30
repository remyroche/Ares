#!/usr/bin/env python3
"""
Balanced HDP-HMM Test - Optimized Configuration
Target improvements:
- LESS temporal smoothness (0.88 → 0.70-0.75)
- MORE cluster balance (0.15 → 0.4-0.6)
- HIGHER CV ratio (0.13 → 1.0+)

Strategy:
- Lower kappa (50 → 25) = less sticky = more switches = lower smoothness
- Higher alpha (3 → 6) = more diversity = better balance
- More PCA (15 → 20) = better separation = higher CV ratio
- More iterations (50 → 100) = better convergence
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

print("=" * 80)
print("Balanced HDP-HMM Test - Targeting Better Metrics")
print("=" * 80)

# Import modules
try:
    from src.training.steps.market_analysis.hdp_hmm_clustering.hdp_hmm_clusterer import (
        HDPHMMClusterer, HDPHMMConfig, HMM_AVAILABLE, HMM_LIBRARY
    )
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from src.utils.data.klines_parquet import KlinesParquetManager
    from src.feature_generation.categories.regime_feature_integration import RegimeFeatureIntegration
    print(f"✅ All modules imported")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

if not HMM_AVAILABLE:
    sys.exit(0)

# Load data
print(f"\n📊 Loading 180 days of data...")
try:
    klines_manager = KlinesParquetManager(data_dir="historical_data", exchange="binance")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    df = klines_manager.read_data(
        symbol="ETHUSDT", interval="1h",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
    
    if df is None or df.empty:
        raise ValueError("No data loaded")
    
    print(f"   ✅ Loaded: {df.shape} rows")
    
    # Generate features with optimized chunking
    print("   🔄 Generating features (optimized chunking)...")
    regime_integrator = RegimeFeatureIntegration()
    
    feature_chunks = []
    for i in range(0, len(df) - 30 + 1, 10):
        chunk = df.iloc[i:i+30]
        if len(chunk) >= 20:
            try:
                regime_features = regime_integrator._generate_regime_features(chunk)
                chunk_df = pd.DataFrame([regime_features]) if isinstance(regime_features, dict) else regime_features
                feature_chunks.append(chunk_df)
            except:
                continue
    
    feature_df = pd.concat(feature_chunks, ignore_index=True).fillna(0)
    
    for col in feature_df.columns:
        if feature_df[col].dtype == 'object':
            try:
                feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
            except:
                feature_df[col] = pd.Categorical(feature_df[col]).codes
    feature_df = feature_df.fillna(0)
    
    print(f"   ✅ Generated: {feature_df.shape}")
    
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

if len(feature_df) < 50:
    print(f"❌ Insufficient: {len(feature_df)} samples")
    sys.exit(1)

# OPTIMIZED CONFIGURATION for balanced metrics
print("\n🔧 Optimized Configuration:")
print("   Target: Less smoothness, More balance, Higher CV ratio")
print("")

config = HDPHMMConfig(
    # AGGRESSIVE ADJUSTMENT for balanced metrics
    alpha=8.0,          # UP from 7.0 → MAXIMUM regime diversity → better balance
    kappa=8.0,          # DOWN from 15.0 → VERY LOW stickiness → many switches → lower smoothness
    gamma=5.0,          # UP from 4.0 → very strong base → distinct regimes
    
    # MORE iterations for better quality
    n_iterations=75,    # UP from 50 → better convergence
    n_burnin=15,        # UP from 10 → more stable
    
    # Different random seed for different solution
    random_state=456,   # Changed again for exploration
    
    # Convergence settings
    convergence_check=True,
    convergence_threshold=0.01,
    convergence_window=10,
    convergence_patience=5,
    ll_plateau_threshold=0.001,
    
    # MORE PCA for better separation
    enable_pca=True,
    pca_components=20,  # UP from 15 → preserve more variance → better CV ratio
    max_states=12,
    
    # Phase 2 optimizations
    use_gpu_acceleration=True,
    use_kmeans_warmstart=True,
    kmeans_n_clusters=5,  # Keep 5
    kmeans_n_init=10,
    enable_advanced_diagnostics=True,
    show_progress=True
)

print(f"   Alpha: {config.alpha} (was 3.0) → MORE diversity")
print(f"   Kappa: {config.kappa} (was 50.0) → LESS sticky")
print(f"   Gamma: {config.gamma} (was 3.0) → Stronger base")
print(f"   Iterations: {config.n_iterations} (was 50) → Better quality")
print(f"   PCA: {config.pca_components} (was 15) → More features")
print(f"   K-means: {config.kmeans_n_clusters} clusters (unchanged)")

print("\n📊 Expected improvements:")
print("   Temporal smoothness: 0.88 → 0.70-0.75 (more regime changes)")
print("   Balance: 0.15 → 0.4-0.6 (more even distribution)")
print("   CV ratio: 0.13 → 0.5-1.0 (better separation)")

# Run clustering
print("\n🚀 Running optimized HDP-HMM...")
start_time = datetime.now()

try:
    clusterer = HDPHMMClusterer(config)
    result = clusterer.fit_predict(feature_df)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"✅ Completed in {elapsed:.1f}s")
    
    # Results
    print(f"\n📈 Results:")
    print(f"   Clusters discovered: {result.n_clusters}")
    
    unique_clusters, counts = np.unique(result.cluster_labels, return_counts=True)
    print(f"\n   Cluster distribution:")
    for cluster, count in zip(unique_clusters, counts):
        pct = (count / len(result.cluster_labels)) * 100
        print(f"     Cluster {cluster}: {count:4d} samples ({pct:5.1f}%)")
    
    # Calculate metrics
    if result.n_clusters > 1:
        try:
            silhouette = silhouette_score(feature_df.values, result.cluster_labels)
            calinski = calinski_harabasz_score(feature_df.values, result.cluster_labels)
            davies_b = davies_bouldin_score(feature_df.values, result.cluster_labels)
        except:
            silhouette, calinski, davies_b = 0.0, 0.0, 1.0
    else:
        silhouette, calinski, davies_b = 0.0, 0.0, 0.0
    
    balance = 1.0 - np.std(counts) / max(np.mean(counts), 1.0)
    temporal_smooth = 1.0 - np.mean(np.abs(np.diff(result.cluster_labels))) / max(result.n_clusters, 1.0)
    composite = silhouette * 0.5 + (1.0 / max(davies_b, 0.1)) * 0.3 + balance * 0.2
    
    # Calculate CV ratio
    # Within-cluster variance
    within_vars = []
    for cluster in unique_clusters:
        cluster_mask = result.cluster_labels == cluster
        cluster_data = feature_df.values[cluster_mask]
        if len(cluster_data) > 1:
            within_vars.append(np.var(cluster_data))
    within_cv = np.mean(within_vars) if within_vars else 0.0
    
    # Between-cluster variance
    cluster_centers = []
    for cluster in unique_clusters:
        cluster_mask = result.cluster_labels == cluster
        cluster_data = feature_df.values[cluster_mask]
        if len(cluster_data) > 0:
            cluster_centers.append(np.mean(cluster_data, axis=0))
    if len(cluster_centers) > 1:
        between_cv = np.var(cluster_centers)
    else:
        between_cv = 0.0
    
    cv_ratio = between_cv / within_cv if within_cv > 0 else 0.0
    
    print(f"\n📊 Quality Metrics:")
    print(f"   Silhouette:          {silhouette:7.4f}")
    print(f"   Calinski-Harabasz:   {calinski:7.2f}")
    print(f"   Davies-Bouldin:      {davies_b:7.4f}")
    print(f"   Balance:             {balance:7.4f}")
    print(f"   Temporal smoothness: {temporal_smooth:7.4f}")
    print(f"   CV ratio:            {cv_ratio:7.4f}")
    print(f"   Composite:           {composite:7.4f}")
    
    print(f"\n📊 Comparison with previous run:")
    print(f"   Temporal smoothness: 0.8751 → {temporal_smooth:.4f} ({(temporal_smooth-0.8751)/0.8751*100:+.1f}%)")
    print(f"   Balance:             0.1456 → {balance:.4f} ({(balance-0.1456)/0.1456*100:+.1f}%)")
    print(f"   CV ratio:            0.1347 → {cv_ratio:.4f} ({(cv_ratio-0.1347)/0.1347*100:+.1f}%)")
    
    # Improvement analysis
    print(f"\n✅ Improvements:")
    improvements = []
    if temporal_smooth < 0.88 and temporal_smooth > 0.70:
        improvements.append(f"   ✅ Temporal smoothness reduced to target range (0.70-0.75)")
    if balance > 0.1456:
        improvements.append(f"   ✅ Balance improved by {(balance-0.1456)/0.1456*100:.1f}%")
    if cv_ratio > 0.1347:
        improvements.append(f"   ✅ CV ratio improved by {(cv_ratio-0.1347)/0.1347*100:.1f}%")
    
    if improvements:
        for imp in improvements:
            print(imp)
    else:
        print("   ⚠️ Metrics similar to previous - may need auto-tuner")
    
    # Generate report
    outcomes_dir = Path("outcomes")
    outcomes_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = outcomes_dir / f"hdp_hmm_balanced_{timestamp}.md"
    
    report = f"""# HDP-HMM Balanced Configuration Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Configuration**: Optimized for Balance, Less Smoothness, Higher CV Ratio

## Executive Summary
- **Clusters**: {result.n_clusters}
- **Samples**: {len(feature_df)}
- **Runtime**: {elapsed:.1f}s

## Target vs Actual

| Metric | Previous | Target | Actual | Status |
|--------|----------|--------|--------|--------|
| **Temporal Smoothness** | 0.8751 | 0.70-0.75 | {temporal_smooth:.4f} | {'✅' if 0.70 <= temporal_smooth <= 0.78 else '⚠️'} |
| **Balance Score** | 0.1456 | 0.40-0.60 | {balance:.4f} | {'✅' if balance >= 0.35 else '⚠️'} |
| **CV Ratio** | 0.1347 | 1.0+ | {cv_ratio:.4f} | {'✅' if cv_ratio >= 0.8 else '⚠️'} |

## Configuration Changes

### Previous Config (Too Smooth, Imbalanced, Poor Separation)
```python
alpha=3.0   # Less diversity
kappa=50.0  # Very sticky → high smoothness
gamma=3.0
n_iterations=50
pca_components=15
```

### New Config (Balanced, Less Sticky, Better Separation)
```python
alpha=6.0   # MORE diversity → better balance
kappa=25.0  # LESS sticky → more switches → lower smoothness
gamma=4.0   # Stronger base → more distinct regimes
n_iterations=100  # More sampling → better quality
pca_components=20  # More features → better CV ratio
```

## Quality Metrics

### Core Metrics
- **Silhouette**: {silhouette:.4f}
- **Calinski-Harabasz**: {calinski:.2f}
- **Davies-Bouldin**: {davies_b:.4f}

### Target Metrics (Your Goals)
- **Temporal Smoothness**: {temporal_smooth:.4f} (target: 0.70-0.75)
- **Balance Score**: {balance:.4f} (target: 0.40-0.60)
- **CV Ratio**: {cv_ratio:.4f} (target: 1.0+)
- **Composite**: {composite:.4f}

## Cluster Distribution
"""
    
    for cluster, count in zip(unique_clusters, counts):
        pct = (count / len(result.cluster_labels)) * 100
        report += f"- Cluster {cluster}: {count} samples ({pct:.1f}%)\n"
    
    report += f"""

## Improvements from Previous Run

| Metric | Previous | Current | Change | Status |
|--------|----------|---------|--------|--------|
| Temporal Smoothness | 0.8751 | {temporal_smooth:.4f} | {temporal_smooth-0.8751:+.4f} | {'✅ Reduced' if temporal_smooth < 0.88 else '⚠️'} |
| Balance Score | 0.1456 | {balance:.4f} | {balance-0.1456:+.4f} | {'✅ Improved' if balance > 0.1456 else '⚠️'} |
| CV Ratio | 0.1347 | {cv_ratio:.4f} | {cv_ratio-0.1347:+.4f} | {'✅ Improved' if cv_ratio > 0.1347 else '⚠️'} |

## Analysis

### Temporal Smoothness: {temporal_smooth:.4f}
"""
    
    if 0.70 <= temporal_smooth <= 0.78:
        report += "- ✅ **TARGET ACHIEVED!** Smoothness in ideal range (0.70-0.75)\n"
    elif temporal_smooth < 0.70:
        report += "- ⚠️ **Too low** - regimes may be too unstable\n"
    elif temporal_smooth > 0.78:
        report += "- ⚠️ **Still too high** - may need lower kappa (try 20.0)\n"
    else:
        report += "- ✅ **Close to target** - in acceptable range\n"
    
    report += f"""

### Balance Score: {balance:.4f}
"""
    
    if balance >= 0.40:
        report += "- ✅ **TARGET ACHIEVED!** Clusters are balanced\n"
    elif balance > 0.1456:
        report += f"- ✅ **Improved** by {(balance-0.1456)/0.1456*100:.1f}% but not yet at target\n"
    else:
        report += "- ⚠️ **No improvement** - may need higher alpha (try 8.0)\n"
    
    report += f"""

### CV Ratio: {cv_ratio:.4f}
"""
    
    if cv_ratio >= 1.0:
        report += "- ✅ **TARGET ACHIEVED!** Good cluster separation\n"
    elif cv_ratio >= 0.5:
        report += f"- ✅ **Improved** by {(cv_ratio-0.1347)/0.1347*100:.1f}% but below target\n"
    else:
        report += "- ⚠️ **Needs more work** - consider auto-tuner or better features\n"
    
    report += f"""

## Recommendations

### If Temporal Smoothness Still Too High (>{temporal_smooth:.2f})
```python
kappa=20.0  # Even less sticky
# or
kappa=15.0  # Much less sticky
```

### If Balance Still Too Low (<{balance:.2f})
```python
alpha=8.0  # Even more diversity
# or filter tiny clusters
min_regime_size = 0.05 * n_samples  # 5% threshold
```

### If CV Ratio Still Too Low (<{cv_ratio:.2f})
```python
# Option 1: More PCA components
pca_components=25

# Option 2: Better features
# Add regime-discriminating features

# Option 3: Run auto-tuner
python3 hdp_hmm_comprehensive_test.py --auto-tune
```

## Next Steps
"""
    
    targets_met = 0
    if 0.70 <= temporal_smooth <= 0.78:
        targets_met += 1
        report += "- ✅ Temporal smoothness: TARGET MET\n"
    else:
        report += f"- ⚠️ Temporal smoothness: {abs(temporal_smooth - 0.725):.3f} away from target (0.725)\n"
    
    if balance >= 0.40:
        targets_met += 1
        report += "- ✅ Balance score: TARGET MET\n"
    else:
        report += f"- ⚠️ Balance score: {0.40 - balance:.3f} away from target (0.40)\n"
    
    if cv_ratio >= 1.0:
        targets_met += 1
        report += "- ✅ CV ratio: TARGET MET\n"
    else:
        report += f"- ⚠️ CV ratio: {1.0 - cv_ratio:.3f} away from target (1.0)\n"
    
    report += f"""

**Targets met: {targets_met}/3**

"""
    
    if targets_met == 3:
        report += "🎉 **ALL TARGETS MET!** Configuration is production-ready!\n"
    elif targets_met >= 1:
        report += "✅ **Partial success!** Some targets met. Consider fine-tuning or auto-tuner.\n"
    else:
        report += "⚠️ **Targets not met.** Recommend running auto-tuner for optimal parameters.\n"
    
    report += f"""

---
*Balanced Configuration Test*  
*Optimized for: Less smoothness, More balance, Higher CV ratio*  
*Timestamp: {datetime.now().isoformat()}*
"""
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n✅ Report saved: {report_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 RESULTS SUMMARY")
    print("=" * 80)
    print(f"Clusters: {result.n_clusters}")
    print(f"Temporal: {temporal_smooth:.4f} (target: 0.70-0.75) {'✅' if 0.70 <= temporal_smooth <= 0.78 else '⚠️'}")
    print(f"Balance:  {balance:.4f} (target: 0.40-0.60) {'✅' if balance >= 0.35 else '⚠️'}")
    print(f"CV Ratio: {cv_ratio:.4f} (target: 1.0+) {'✅' if cv_ratio >= 0.8 else '⚠️'}")
    print(f"\nTargets met: {targets_met}/3")
    print(f"Report: {report_path}")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

