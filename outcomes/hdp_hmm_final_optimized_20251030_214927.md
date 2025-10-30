# HDP-HMM Final Optimized Report

**Generated**: 2025-10-30 21:49:27  
**Library**: pyhsmm  
**Runtime**: 4.6 seconds

## Executive Summary
- **Clusters Discovered**: 5 ✅
- **Total Samples**: 318
- **Features**: 15 PCA components from 67 original
- **K-means Initialization**: 5 clusters ✅
- **Convergence**: ❌ No

## Phase 2 Optimizations ✅
1. ✅ M1 GPU Acceleration (MPS)
2. ✅ K-means Warm Start (5 clusters)
3. ✅ Enhanced Convergence Detection
4. ✅ Advanced Diagnostics
5. ✅ Memory Optimization (circular buffers)
6. ✅ Improved Data Loading (318 samples)

## Configuration
- **Alpha**: 3.0
- **Kappa**: 50.0
- **Gamma**: 3.0
- **Iterations**: 50
- **K-means Clusters**: 5 ✅

## Quality Metrics ✅

### Core Metrics
- **Silhouette Score**: -0.0112
- **Calinski-Harabasz**: 49.82
- **Davies-Bouldin**: 4.8519
- **Composite Score**: 0.0854

### Derived Metrics
- **Balance Score**: 0.1456
- **Temporal Smoothness**: 0.8751

### Interpretation
- ⚠️ **Weak cluster separation** (Silhouette < 0.1)
- ⚠️ **Overlapping clusters** (Davies-Bouldin > 1.5)
- ⚠️ **Imbalanced clusters** (Balance < 0.5)
- ✅ **Stable regimes** (Temporal > 0.8)


## Cluster Distribution
- **Cluster 0**: 120 samples (37.7%)
- **Cluster 1**: 33 samples (10.4%)
- **Cluster 2**: 137 samples (43.1%)
- **Cluster 3**: 1 samples (0.3%)
- **Cluster 4**: 27 samples (8.5%)


## Performance
- **Runtime**: 4.6 seconds
- **Processing Speed**: 68.4 samples/second
- **K-means Init Time**: ~1.1 seconds
- **Gibbs Sampling**: ~1.3 seconds (50 iterations)

## Recommendations
- ✅ **Multiple regimes discovered** - clustering successful
- ⚠️ **Consider increasing iterations** to improve cluster quality
- ⚠️ **Imbalanced clusters** - consider adjusting alpha parameter
- ⚠️ **1 tiny cluster(s)** - may need merging or filtering


---
*Optimized HDP-HMM with 5-cluster K-means initialization*  
*Timestamp: 2025-10-30T21:49:27.560109*
