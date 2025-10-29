# Quick Reference: Enhanced Cluster Quality Assessment

## ✅ What Was Done

### 1. Enhanced `cluster_quality_assessor.py`
**New Metrics Added:**
- `balance_score`: Global cluster balance (0-1, higher = better)
- `cluster_size_distribution`: List of cluster sizes as percentages
- `min_cluster_size_pct`, `max_cluster_size_pct`: Size bounds
- `cluster_size_std`: Standard deviation of cluster sizes
- `within_regime_cv_std`, `between_regime_cv_std`: CV standard deviations
- `per_regime_cv`: Dictionary of per-regime CV values
- `log_likelihood`: Field for probabilistic models (MS-DR, HDP-HMM)

**Enhanced Composite Score:**
- Now includes balance score with 15% weight
- Better normalized for comparison across methods

### 2. Integrated into MS-DR & HDP-HMM
Both clusterers now use `ComprehensiveQualityAssessor` from `quality_assessment.py`:
- Full suite of quality metrics (DBCV, predictive power, temporal, etc.)
- Composite quality scores for model comparison
- Standardized metric format

### 3. Refactored HDBSCAN Modules
**`hdbscan_regime_optimizer.py`:**
- Removed `_calculate_quality_metrics()` (~70 lines)
- Now uses comprehensive quality assessor
- Passes clusterer for DBCV calculation

**`optimized_hdbscan_regime_discovery.py`:**
- Removed `_assess_clustering_quality()` (~45 lines)
- Removed `_calculate_quality_improvement()` (~30 lines)
- Now uses composite scores for quality comparison
- Better fallback detection

---

## 📊 Key Metrics Available

### Balance Metrics (NEW)
```python
metrics.balance_score  # 0-1, higher = better balanced
metrics.min_cluster_size_pct  # Smallest cluster %
metrics.max_cluster_size_pct  # Largest cluster %
metrics.cluster_size_std  # Size variability
metrics.cluster_size_distribution  # List of all cluster %
```

### Enhanced CV Metrics
```python
metrics.within_regime_cv  # Mean within-cluster CV
metrics.within_regime_cv_std  # Std dev of within CVs (NEW)
metrics.between_regime_cv  # Mean between-cluster CV
metrics.between_regime_cv_std  # Std dev of between CVs (NEW)
metrics.per_regime_cv  # Dict: {regime_id: cv_value} (NEW)
```

### Model-Specific
```python
metrics.log_likelihood  # For MS-DR, HDP-HMM (NEW)
```

### Composite Score (Enhanced)
```python
metrics.quality_score  # 0-1, now includes balance (15% weight)
```

---

## 💡 Usage Examples

### Using cluster_quality_assessor.py
```python
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
    create_cluster_quality_assessor
)

assessor = create_cluster_quality_assessor()
metrics = assessor.assess_quality(
    regime_labels=labels,
    feature_data=features_df,
    forward_returns=returns,
    timestamps=timestamps
)

# Access new metrics
print(f"Balance Score: {metrics.balance_score:.3f}")
print(f"Within CV: {metrics.within_regime_cv:.3f} ± {metrics.within_regime_cv_std:.3f}")
print(f"Per-Regime CV: {metrics.per_regime_cv}")
```

### Using quality_assessment.py (HDBSCAN-specific)
```python
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
    create_quality_assessor
)

assessor = create_quality_assessor()
metrics = assessor.assess_clustering_quality(
    cluster_labels=labels,
    features=features,
    clusterer=hdbscan_model,  # For DBCV
    timestamps=data.index,
    returns=returns
)

# Comprehensive metrics including DBCV, predictive power
print(f"DBCV: {metrics.dbcv_score:.3f}")
print(f"Composite: {metrics.composite_quality_score:.3f}")
```

---

## 📈 Composite Score Weights (Updated)

| Metric | Weight | Notes |
|--------|--------|-------|
| Silhouette | 20% | Reduced from 25% |
| Davies-Bouldin | 15% | Reduced from 20% |
| Calinski-Harabasz | 15% | Reduced from 20% |
| CV Ratio | 15% | Same |
| **Balance Score** | **15%** | **NEW!** |
| Temporal Smoothness | 10% | Same |
| Noise Ratio | 10% | Same |

---

## 🎯 Files Modified

1. `/workspace/src/training/steps/market_analysis/clusters/cluster_quality_assessor.py` - Enhanced
2. `/workspace/src/training/steps/market_analysis/ms_dr_clustering/ms_dr_clusterer.py` - Integrated
3. `/workspace/src/training/steps/market_analysis/hdp_hmm_clustering/hdp_hmm_clusterer.py` - Integrated
4. `/workspace/src/training/steps/market_analysis/hdbscan_clustering/optimization/hdbscan_regime_optimizer.py` - Refactored
5. `/workspace/src/training/steps/market_analysis/hdbscan_clustering/optimization/optimized_hdbscan_regime_discovery.py` - Refactored

**Total:** 6 files, ~175 lines reduced, multiple new capabilities added

---

## ✨ Benefits

1. **Better Balance Assessment**: New balance metrics identify imbalanced clusters
2. **Enhanced CV Analysis**: Std deviation shows CV variability across regimes
3. **Standardized Quality**: All clustering methods use same assessment framework
4. **Code Reduction**: ~175 lines of duplicate code removed
5. **Easier Comparison**: Composite scores enable direct model comparison

---

## 📚 Full Documentation

See `CLUSTER_QUALITY_CONSOLIDATION_COMPLETE.md` for comprehensive details.
