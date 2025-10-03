# CV Ratio Improvement Strategies - Comprehensive Review

## Executive Summary

This document provides a comprehensive review of the clustering optimization system and actionable strategies to further improve the **CV Ratio (Variance Ratio)** beyond the current implementation.

**Current Status**: After implementing Weighted Category PCA and enhanced optimization parameters  
**Target**: Achieve CV Ratio of 1.5-2.0 consistently  
**Critical Issue Identified**: Balance metric was over-constraining and forcing equal cluster sizes

---

## 🔴 CRITICAL ISSUE IDENTIFIED: Balance Metric Over-Constraint

### Problem Analysis

**Symptom**: Most regimes had the same distribution percentage  
**Root Cause**: Balance score calculation was forcing clusters to be perfectly equal in size

```python
# OLD PROBLEMATIC CODE:
def get_balance_score(self) -> float:
    target_size = self.n_samples / self.n_clusters
    for size in self.cluster_sizes:
        penalty = (size / self.n_samples - 1.0 / self.n_clusters) ** 2  # ❌ FORCES EQUAL SIZES
    return 1.0 - np.mean(size_penalties)
```

**Impact on Metrics**:
- ❌ **CV Ratio Degraded**: Forced equal sizes → poor regime separation
- ❌ **Silhouette Degraded**: Points assigned to wrong clusters for balance
- ❌ **Temporal Stability Degraded**: Inappropriate assignments for size balance
- ✅ **Balance Score High**: But at the cost of everything else!

### Solution Implemented

**NEW SOFT BALANCE CONSTRAINT** (✅ Fixed):
```python
def get_balance_score(self) -> float:
    """
    SOFT CONSTRAINT: Only penalize EXTREME imbalances (> 3x deviation).
    Allows natural cluster size variation for better regime separation.
    """
    # Only penalize extreme cases: size > 3x mean or size < 0.33x mean
    # This allows 0.5x to 2x variation naturally
    extreme_penalties = []
    for size in sizes:
        ratio = size / mean_size
        if ratio > 3.0:  # Too large
            extreme_penalties.append((ratio - 3.0) ** 2)
        elif ratio < 0.33:  # Too small
            extreme_penalties.append((0.33 - ratio) ** 2)
    
    # Very gentle penalty (0.1x weight)
    return max(0.0, 1.0 - 0.1 * np.mean(extreme_penalties))
```

**Key Changes**:
1. ✅ Allows 2-3x size variation naturally (0.33x to 3x mean)
2. ✅ Only penalizes extreme imbalances
3. ✅ Gentle penalty (0.1x multiplier) vs old harsh penalty
4. ✅ Balance weight reduced: 10% → 5% in objective function

**Expected Impact**:
- 📈 **CV Ratio**: +30-50% improvement (allows natural regime size differences)
- 📈 **Silhouette**: +20-30% improvement (better cluster assignments)
- 📈 **Temporal Stability**: +15-25% improvement (more stable regimes)
- ⚖️ **Balance**: Still prevents extreme cases (1 giant cluster, etc.)

---

## 🎯 Strategy 1: Enhanced Feature Engineering (HIGHEST IMPACT)

### 1.1 Regime-Discriminative Features

**Add features specifically designed to separate regimes:**

```python
def add_regime_discriminative_features(df):
    """Add features designed to maximize between-regime variance."""
    
    # 1. Volatility Regime Features (HIGH IMPACT)
    df['vol_regime_zscore'] = (df['realized_vol_20d'] - df['realized_vol_60d'].rolling(60).mean()) / df['realized_vol_60d'].rolling(60).std()
    df['vol_regime_percentile'] = df['realized_vol_20d'].rolling(252).rank(pct=True)
    df['vol_regime_transition'] = (df['realized_vol_5d'] / df['realized_vol_20d']).apply(np.log)
    
    # 2. Return Distribution Features (HIGH IMPACT)
    df['return_skew_20d'] = df['returns'].rolling(20).skew()
    df['return_kurt_20d'] = df['returns'].rolling(20).kurt()
    df['return_regime_zscore'] = (df['returns_20d'] - df['returns_60d'].rolling(60).mean()) / df['returns_60d'].rolling(60).std()
    
    # 3. Trend Strength Features (MEDIUM IMPACT)
    df['trend_strength'] = np.abs(df['ma_20'] - df['ma_50']) / df['atr_14']
    df['trend_consistency'] = (df['returns'] > 0).rolling(20).mean() - 0.5  # Directional bias
    df['trend_acceleration'] = df['ma_20'].diff(5) / df['ma_20'].diff(20)
    
    # 4. Correlation Regime Features (MEDIUM IMPACT)
    df['beta_to_market_20d'] = rolling_beta(df['returns'], market_returns, window=20)
    df['correlation_regime'] = df['beta_to_market_20d'] - df['beta_to_market_60d']
    
    # 5. Liquidity Regime Features (LOW-MEDIUM IMPACT)
    df['liquidity_regime'] = df['volume_20d'] / df['volume_60d']
    df['bid_ask_regime'] = df['bid_ask_spread_20d'] / df['bid_ask_spread_60d']
    
    return df
```

**Expected Gain**: +20-40% CV Ratio improvement

### 1.2 Multi-Timeframe Features

```python
def add_multi_timeframe_features(df):
    """Add features across multiple timeframes to capture regime persistence."""
    
    timeframes = [5, 10, 20, 40, 60]
    
    for tf in timeframes:
        # Volatility across timeframes
        df[f'vol_{tf}d'] = df['returns'].rolling(tf).std() * np.sqrt(252)
        
        # Return momentum across timeframes
        df[f'momentum_{tf}d'] = df['close'].pct_change(tf)
        
        # Regime consistency
        df[f'regime_consistency_{tf}d'] = (df['vol_regime'] == df['vol_regime'].shift(1)).rolling(tf).mean()
    
    # Timeframe alignment features (regime convergence)
    df['vol_timeframe_alignment'] = np.std([df[f'vol_{tf}d'] for tf in timeframes], axis=0)
    df['momentum_timeframe_alignment'] = np.std([df[f'momentum_{tf}d'] for tf in timeframes], axis=0)
    
    return df
```

**Expected Gain**: +15-25% CV Ratio improvement

---

## 🎯 Strategy 2: Advanced PCA Enhancements (MEDIUM-HIGH IMPACT)

### 2.1 Kernel PCA for Non-Linear Relationships

```python
from sklearn.decomposition import KernelPCA

def apply_kernel_pca_by_category(features, categories):
    """Use Kernel PCA to capture non-linear regime patterns."""
    
    for cat_name, cat_config in categories.items():
        cat_features = features[:, cat_config['indices']]
        
        # Use RBF kernel for non-linear relationships
        kpca = KernelPCA(
            n_components=cat_config['n_components'],
            kernel='rbf',
            gamma=1.0 / cat_features.shape[1],  # Auto-scale
            fit_inverse_transform=True
        )
        
        cat_transformed = kpca.fit_transform(cat_features)
        # ... rest of pipeline
```

**Expected Gain**: +10-20% CV Ratio improvement

### 2.2 Supervised PCA (Using Forward Returns as Target)

```python
def supervised_feature_selection(features, forward_returns, n_components=20):
    """Select features that best predict forward returns → regime changes."""
    
    from sklearn.cross_decomposition import PLSRegression
    
    # Use PLS to find features that predict returns
    pls = PLSRegression(n_components=n_components)
    pls.fit(features, forward_returns)
    
    # Transform features to PLS components
    features_pls = pls.transform(features)
    
    # Weight by prediction R²
    r2_scores = []
    for i in range(n_components):
        y_pred = pls.predict(features_pls[:, :i+1])
        r2 = r2_score(forward_returns, y_pred)
        r2_scores.append(r2)
    
    # Weight components by predictive power
    weights = np.array(r2_scores) / np.sum(r2_scores)
    features_weighted = features_pls * np.sqrt(weights)
    
    return features_weighted
```

**Expected Gain**: +15-30% CV Ratio improvement

---

## 🎯 Strategy 3: Clustering Algorithm Enhancements (MEDIUM IMPACT)

### 3.1 Custom Distance Metric for Regime Similarity

```python
def regime_distance_metric(point_a, point_b, feature_indices):
    """
    Custom distance metric emphasizing regime-critical features.
    Weights volatility and return features higher than others.
    """
    
    # Extract feature categories
    vol_features = point_a[feature_indices['volatility']]
    vol_features_b = point_b[feature_indices['volatility']]
    
    returns_features = point_a[feature_indices['returns']]
    returns_features_b = point_b[feature_indices['returns']]
    
    other_features = point_a[feature_indices['other']]
    other_features_b = point_b[feature_indices['other']]
    
    # Weighted distance (emphasize regime-critical features)
    vol_dist = np.linalg.norm(vol_features - vol_features_b)
    ret_dist = np.linalg.norm(returns_features - returns_features_b)
    oth_dist = np.linalg.norm(other_features - other_features_b)
    
    # Combine with weights matching PCA category weights
    weighted_dist = (
        0.40 * ret_dist +
        0.30 * vol_dist +
        0.30 * oth_dist
    )
    
    return weighted_dist
```

**Integration**:
```python
from sklearn.metrics import pairwise_distances

# Use custom metric in clustering
distances = pairwise_distances(features, metric=regime_distance_metric)
# Use distance matrix for clustering initialization
```

**Expected Gain**: +10-15% CV Ratio improvement

### 3.2 Hierarchical Initialization for Better Starting Point

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster

def hierarchical_initialization(features, n_clusters, linkage_method='ward'):
    """Use hierarchical clustering for better initialization."""
    
    # Build linkage tree
    Z = linkage(features, method=linkage_method, metric='euclidean')
    
    # Cut tree to get n_clusters
    initial_labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
    
    # Compute initial centroids
    centroids = np.array([
        features[initial_labels == k].mean(axis=0)
        for k in range(n_clusters)
    ])
    
    return initial_labels, centroids
```

**Expected Gain**: +5-10% CV Ratio improvement

---

## 🎯 Strategy 4: Objective Function Refinement (MEDIUM IMPACT)

### 4.1 Add Calinski-Harabasz Score to Objective

```python
from sklearn.metrics import calinski_harabasz_score

def enhanced_objective_with_ch(self, stats, constraints):
    """Enhanced objective including Calinski-Harabasz (variance ratio proxy)."""
    
    # Current CV ratio (between/within variance)
    cv_ratio = stats.get_cv_ratio()
    
    # Calinski-Harabasz score (another variance ratio metric)
    try:
        ch_score = calinski_harabasz_score(
            stats.features, 
            stats.assignments
        )
        # Normalize CH score to [0, 1] range
        ch_normalized = ch_score / (ch_score + 100)  # Sigmoid-like normalization
    except:
        ch_normalized = 0.5
    
    # Combine both variance ratio metrics
    combined_cv = 0.7 * cv_ratio + 0.3 * ch_normalized
    
    # Rest of objective
    objective = (
        self.w_cv * combined_cv +
        self.w_temp * temporal_score +
        self.w_sil * silhouette_score +
        self.w_bal * balance_score
    )
    
    return objective
```

**Expected Gain**: +5-10% CV Ratio improvement

### 4.2 Adaptive Weights Based on Iteration

```python
def adaptive_weights(iteration, max_iterations):
    """
    Gradually increase CV weight and decrease balance weight.
    Early iterations: explore with balanced objectives
    Late iterations: aggressively optimize CV
    """
    
    progress = iteration / max_iterations
    
    # Increase CV weight over time (0.45 → 0.60)
    w_cv = 0.45 + 0.15 * progress
    
    # Decrease balance weight over time (0.05 → 0.02)
    w_bal = 0.05 * (1 - 0.6 * progress)
    
    # Keep temporal and silhouette stable
    w_temp = 0.35
    w_sil = 0.15
    
    # Renormalize (optional)
    total = w_cv + w_temp + w_sil + w_bal
    
    return {
        'w_cv': w_cv / total,
        'w_temp': w_temp / total,
        'w_sil': w_sil / total,
        'w_bal': w_bal / total
    }
```

**Expected Gain**: +8-12% CV Ratio improvement

---

## 🎯 Strategy 5: Post-Processing Refinements (LOW-MEDIUM IMPACT)

### 5.1 Regime Merging Based on Low Between-Variance

```python
def merge_similar_regimes(features, assignments, cv_threshold=0.2):
    """
    Merge regimes that have low between-cluster variance.
    This reduces K but increases overall CV ratio.
    """
    
    unique_regimes = np.unique(assignments)
    n_regimes = len(unique_regimes)
    
    # Calculate pairwise between-cluster variance
    merge_candidates = []
    
    for i in range(n_regimes):
        for j in range(i+1, n_regimes):
            # Features in each regime
            feat_i = features[assignments == unique_regimes[i]]
            feat_j = features[assignments == unique_regimes[j]]
            
            # Combined features
            feat_combined = np.vstack([feat_i, feat_j])
            
            # Within-cluster variance of combined
            within_var_combined = np.var(feat_combined, axis=0).mean()
            
            # If very similar, consider merging
            if within_var_combined < cv_threshold:
                merge_candidates.append((i, j, within_var_combined))
    
    # Merge regimes with lowest within-variance when combined
    merge_candidates.sort(key=lambda x: x[2])
    
    # Perform merges (greedy)
    for i, j, _ in merge_candidates[:max_merges]:
        assignments[assignments == unique_regimes[j]] = unique_regimes[i]
    
    return assignments
```

**Expected Gain**: +5-8% CV Ratio improvement

### 5.2 Regime Splitting Based on High Within-Variance

```python
def split_high_variance_regimes(features, assignments, var_threshold_percentile=90):
    """
    Split regimes with high within-cluster variance.
    This increases K but increases overall CV ratio.
    """
    
    unique_regimes = np.unique(assignments)
    
    # Calculate within-cluster variance for each regime
    within_vars = []
    for regime_id in unique_regimes:
        feat_regime = features[assignments == regime_id]
        within_var = np.var(feat_regime, axis=0).mean()
        within_vars.append((regime_id, within_var, len(feat_regime)))
    
    # Sort by within-variance
    within_vars.sort(key=lambda x: x[1], reverse=True)
    
    # Find threshold (90th percentile)
    var_values = [v for _, v, _ in within_vars]
    threshold = np.percentile(var_values, var_threshold_percentile)
    
    # Split high-variance regimes
    new_regime_id = max(unique_regimes) + 1
    
    for regime_id, within_var, size in within_vars:
        if within_var > threshold and size >= 40:  # Ensure enough points
            # Split using k-means k=2
            feat_regime = features[assignments == regime_id]
            
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=2, random_state=42, n_init=10)
            sub_labels = km.fit_predict(feat_regime)
            
            # Assign to new regime
            regime_indices = np.where(assignments == regime_id)[0]
            assignments[regime_indices[sub_labels == 1]] = new_regime_id
            new_regime_id += 1
    
    return assignments
```

**Expected Gain**: +3-7% CV Ratio improvement

---

## 🎯 Strategy 6: Ensemble Clustering (MEDIUM-HIGH IMPACT)

### 6.1 Multi-Run Consensus Clustering

```python
def ensemble_clustering(features, n_clusters, n_runs=10):
    """
    Run clustering multiple times with different initializations.
    Select configuration with highest CV ratio.
    """
    
    best_cv_ratio = -np.inf
    best_assignments = None
    
    for run in range(n_runs):
        # Random seed for this run
        seed = 42 + run
        
        # Run clustering
        assignments = run_iterative_optimization(
            features, 
            n_clusters=n_clusters,
            random_seed=seed
        )
        
        # Calculate CV ratio
        cv_ratio = calculate_variance_ratio(features, assignments)
        
        if cv_ratio > best_cv_ratio:
            best_cv_ratio = cv_ratio
            best_assignments = assignments.copy()
    
    tprint(f"✅ Best CV Ratio from {n_runs} runs: {best_cv_ratio:.4f}", "SUCCESS")
    
    return best_assignments, best_cv_ratio
```

**Expected Gain**: +10-20% CV Ratio improvement

### 6.2 Cross-Method Consensus

```python
def cross_method_consensus(features, n_clusters):
    """
    Combine multiple clustering methods via consensus.
    Methods: K-Means, Hierarchical, Spectral, GMM
    """
    
    from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
    from sklearn.mixture import GaussianMixture
    
    # Run multiple methods
    kmeans_labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=20).fit_predict(features)
    hier_labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(features)
    spectral_labels = SpectralClustering(n_clusters=n_clusters, random_state=42).fit_predict(features)
    gmm_labels = GaussianMixture(n_components=n_clusters, random_state=42).fit_predict(features)
    
    # Consensus matrix: co-occurrence
    n_samples = len(features)
    consensus_matrix = np.zeros((n_samples, n_samples))
    
    all_labels = [kmeans_labels, hier_labels, spectral_labels, gmm_labels]
    
    for labels in all_labels:
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                if labels[i] == labels[j]:
                    consensus_matrix[i, j] += 1
                    consensus_matrix[j, i] += 1
    
    # Normalize
    consensus_matrix /= len(all_labels)
    
    # Use consensus as similarity, cluster again
    final_labels = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        linkage='average'
    ).fit_predict(1 - consensus_matrix)  # Convert similarity to distance
    
    return final_labels
```

**Expected Gain**: +8-15% CV Ratio improvement

---

## 📊 Summary of Strategies & Expected Gains

| Strategy | Difficulty | Expected CV Gain | Priority |
|----------|-----------|-----------------|----------|
| **Fix Balance Constraint** ✅ | Easy | +30-50% | CRITICAL (DONE) |
| **Enhanced Feature Engineering** | Medium | +20-40% | HIGH |
| **Supervised PCA** | Medium | +15-30% | HIGH |
| **Ensemble Clustering** | Medium | +10-20% | MEDIUM-HIGH |
| **Kernel PCA** | Easy | +10-20% | MEDIUM-HIGH |
| **Adaptive Weights** | Easy | +8-12% | MEDIUM |
| **Cross-Method Consensus** | Hard | +8-15% | MEDIUM |
| **Add Calinski-Harabasz** | Easy | +5-10% | MEDIUM |
| **Hierarchical Init** | Easy | +5-10% | MEDIUM |
| **Custom Distance Metric** | Medium | +10-15% | MEDIUM |
| **Regime Merging** | Easy | +5-8% | LOW-MEDIUM |
| **Regime Splitting** | Easy | +3-7% | LOW-MEDIUM |
| **Multi-Timeframe Features** | Medium | +15-25% | HIGH |

---

## 🚀 Recommended Implementation Plan

### Phase 1: Quick Wins (Week 1) ✅
1. ✅ **Fix Balance Constraint** - COMPLETED
2. 🔄 **Enhanced Feature Engineering** - Add regime-discriminative features
3. 🔄 **Adaptive Weights** - Implement iteration-based weight adjustment

**Expected Gain**: +50-80% cumulative

### Phase 2: Medium Impact (Week 2-3)
4. **Supervised PCA** - Use forward returns as target
5. **Ensemble Clustering** - Multi-run consensus
6. **Multi-Timeframe Features** - Add cross-timeframe alignment

**Expected Gain**: +30-50% additional

### Phase 3: Advanced Techniques (Week 4+)
7. **Kernel PCA** - Non-linear relationships
8. **Custom Distance Metric** - Regime-aware distances
9. **Cross-Method Consensus** - Combine clustering algorithms

**Expected Gain**: +20-35% additional

---

## 🎯 Overall Expected Improvement

**Baseline** (before fixes): CV Ratio ~0.8-1.2  
**After Balance Fix** ✅: CV Ratio ~1.2-1.8 (+30-50%)  
**After Phase 1**: CV Ratio ~1.8-2.5 (+50-80% from fixed baseline)  
**After Phase 2**: CV Ratio ~2.2-3.5 (+30-50% additional)  
**After Phase 3**: CV Ratio ~2.5-4.0 (+20-35% additional)

**TOTAL POTENTIAL**: CV Ratio of **2.5-4.0** (150-250% improvement from original)

---

## ✅ Action Items

### Immediate
- [x] Fix balance constraint (COMPLETED)
- [ ] Add regime-discriminative features
- [ ] Implement adaptive weights
- [ ] Test balance fix on real data

### Short-term
- [ ] Implement supervised PCA
- [ ] Add multi-timeframe features
- [ ] Set up ensemble clustering

### Long-term
- [ ] Kernel PCA integration
- [ ] Custom distance metrics
- [ ] Cross-method consensus

---

*Document Created: 2025-10-03*  
*Status: Balance Fix Implemented ✅, Strategies Documented*
