# Enhanced Feature Selection Metrics Proposal

## Problem Statement

Current feature selection shows concerning instability:
- **CV Consistency:** 14% (only 3/60 features consistent)
- **Stability:** 56.82% (below 58.37% threshold)
- **Risk:** Poor generalization, overfitting to noise

## Proposed Enhanced Metrics

### **Tier 1: Critical Metrics (Implement First)**

#### 1. Null Importance Distribution
**Purpose:** Statistical significance testing for feature importance

```python
def calculate_null_importance(X, y, n_permutations=50):
    """
    Measure importance of features when target is permuted (noise baseline).

    Returns:
        - null_importance_distribution: dict[feature -> list[float]]
        - p_values: dict[feature -> float]
        - significant_features: list[str] (p < 0.05)
    """
    null_importances = defaultdict(list)

    for i in range(n_permutations):
        y_permuted = np.random.permutation(y)
        importances = calculate_permutation_importance(X, y_permuted)
        for feature, importance in importances.items():
            null_importances[feature].append(importance)

    # Calculate p-values
    p_values = {}
    for feature in X.columns:
        true_importance = actual_importances[feature]
        null_dist = null_importances[feature]
        # P-value: proportion of null importances >= true importance
        p_values[feature] = np.mean(null_dist >= true_importance)

    return {
        'null_importance_distribution': dict(null_importances),
        'p_values': p_values,
        'significant_features': [f for f, p in p_values.items() if p < 0.05],
        'false_discovery_rate': calculate_fdr(p_values)
    }
```

**Threshold:** Keep features with p < 0.05 (statistically significant)

---

#### 2. Permutation Importance Confidence Intervals
**Purpose:** Distinguish signal from noise with statistical rigor

```python
def calculate_importance_with_ci(X, y, n_repeats=30, confidence=0.95):
    """
    Calculate permutation importance with confidence intervals.

    Returns:
        - importances_mean: dict[feature -> float]
        - importances_ci_lower: dict[feature -> float]
        - importances_ci_upper: dict[feature -> float]
        - significant_features: list[str] (CI doesn't include 0)
    """
    from scipy.stats import t as t_dist

    # Run permutation importance multiple times
    all_importances = defaultdict(list)

    for repeat in range(n_repeats):
        perm_result = permutation_importance(
            model, X, y,
            n_repeats=1,
            random_state=42 + repeat
        )
        for idx, feature in enumerate(X.columns):
            all_importances[feature].append(perm_result.importances[idx, 0])

    # Calculate confidence intervals
    alpha = 1 - confidence
    results = {}

    for feature in X.columns:
        importances = all_importances[feature]
        mean = np.mean(importances)
        std = np.std(importances, ddof=1)
        n = len(importances)

        # t-distribution for CI
        t_critical = t_dist.ppf(1 - alpha/2, n - 1)
        margin = t_critical * (std / np.sqrt(n))

        ci_lower = mean - margin
        ci_upper = mean + margin

        results[feature] = {
            'mean': mean,
            'std': std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'cv': std / mean if mean > 0 else np.inf,  # Coefficient of variation
            'significant': ci_lower > 0  # CI doesn't include zero
        }

    return results
```

**Threshold:** Keep features where `ci_lower > 0` (statistically significant)

---

#### 3. Forward Selection with Walk-Forward Validation
**Purpose:** Validate that features actually improve OOS performance

```python
def walk_forward_feature_validation(X, y, selected_features, n_splits=10):
    """
    Validate features using walk-forward analysis on time series.

    Returns:
        - feature_contributions: dict[feature -> float] (marginal R²)
        - cumulative_performance: list[dict] (R² as features added)
        - optimal_feature_count: int
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import r2_score

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Sort features by importance (descending)
    sorted_features = sorted(
        selected_features,
        key=lambda f: feature_importances[f],
        reverse=True
    )

    cumulative_performance = []
    feature_contributions = {}

    # Incrementally add features and measure OOS performance
    for n_features in range(1, len(sorted_features) + 1):
        current_features = sorted_features[:n_features]

        # Walk-forward validation
        r2_scores = []

        for train_idx, test_idx in tscv.split(X):
            X_train = X[current_features].iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X[current_features].iloc[test_idx]
            y_test = y.iloc[test_idx]

            model = ExtraTreesRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            r2_scores.append(r2)

        avg_r2 = np.mean(r2_scores)
        std_r2 = np.std(r2_scores)

        # Calculate marginal contribution
        if n_features > 1:
            marginal_contribution = avg_r2 - cumulative_performance[-1]['avg_r2']
        else:
            marginal_contribution = avg_r2

        feature_contributions[sorted_features[n_features - 1]] = marginal_contribution

        cumulative_performance.append({
            'n_features': n_features,
            'features': current_features.copy(),
            'avg_r2': avg_r2,
            'std_r2': std_r2,
            'marginal_contribution': marginal_contribution
        })

    # Find optimal feature count (elbow in R² curve)
    optimal_idx = find_elbow_point([p['avg_r2'] for p in cumulative_performance])
    optimal_feature_count = optimal_idx + 1

    return {
        'feature_contributions': feature_contributions,
        'cumulative_performance': cumulative_performance,
        'optimal_feature_count': optimal_feature_count,
        'max_r2': max(p['avg_r2'] for p in cumulative_performance)
    }
```

**Threshold:** Keep features with `marginal_contribution > 0.01` (1% R² improvement)

---

#### 4. Selection Frequency Distribution Analysis
**Purpose:** Understand feature selection patterns

```python
def analyze_selection_frequency_distribution(cv_results):
    """
    Analyze distribution of feature selection frequencies.

    Returns:
        - frequency_histogram: dict[bin -> count]
        - selection_mode: "bimodal" | "uniform" | "concentrated"
        - unstable_features_ratio: float
    """
    frequencies = list(cv_results['selection_consistency'].values())

    # Create histogram bins
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    histogram = {}

    for i in range(len(bins) - 1):
        bin_name = f"{int(bins[i]*100)}-{int(bins[i+1]*100)}%"
        count = sum(1 for f in frequencies if bins[i] <= f < bins[i+1])
        histogram[bin_name] = count

    # Add 100% bin (inclusive)
    histogram["100%"] = sum(1 for f in frequencies if f == 1.0)

    # Detect distribution mode
    if histogram["0-20%"] + histogram["80-100%"] > 0.7 * len(frequencies):
        mode = "bimodal"  # Good: clear separation
    elif all(count < len(frequencies) * 0.3 for count in histogram.values()):
        mode = "uniform"  # Bad: no clear winners
    else:
        mode = "concentrated"  # Depends on where concentration is

    unstable_ratio = (histogram["0-20%"] + histogram["20-40%"]) / len(frequencies)

    return {
        'frequency_histogram': histogram,
        'selection_mode': mode,
        'unstable_features_ratio': unstable_ratio,
        'highly_stable_features': histogram.get("80-100%", 0)
    }
```

**Warning Threshold:** `unstable_features_ratio > 0.6` → Too many unreliable features

---

### **Tier 2: Important Metrics (Implement Soon)**

#### 5. Temporal Drift Detection (Already Implemented, Not Reported)
**Location:** `/home/user/Ares/src/training/utils/feature_selection/stability_analysis.py:208-313`

**Action Required:** Enable in final report

```python
# In final_feature_selection_step.py, add:
temporal_results = stability_analyzer.analyze_temporal_stability(
    X=X_train,
    y=y_train,
    feature_names=selected_features,
    selection_method=selection_method,
    method_params=method_params,
    temporal_indices=X_train.index
)

# Report these metrics:
- temporal_consistency (per feature)
- temporal_drift_slope (trend over time)
- mean_jaccard_similarity (between time windows)
```

**Threshold:** `temporal_consistency > 0.7` for stable features

---

#### 6. Feature Redundancy with Clustering
**Purpose:** Remove correlated features, keep best representative

```python
def cluster_redundant_features(X, selected_features, corr_threshold=0.85):
    """
    Cluster highly correlated features and select best from each cluster.

    Returns:
        - feature_clusters: dict[cluster_id -> list[features]]
        - representative_features: list[str]
        - redundant_features: dict[feature -> representative]
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    # Calculate correlation matrix
    X_selected = X[selected_features]
    corr_matrix = X_selected.corr().abs()

    # Convert correlation to distance (1 - correlation)
    distance_matrix = 1 - corr_matrix

    # Hierarchical clustering
    linkage_matrix = linkage(squareform(distance_matrix), method='average')

    # Cut tree at threshold
    cluster_labels = fcluster(linkage_matrix, 1 - corr_threshold, criterion='distance')

    # Group features by cluster
    feature_clusters = defaultdict(list)
    for feature, cluster_id in zip(selected_features, cluster_labels):
        feature_clusters[cluster_id].append(feature)

    # Select best feature from each cluster (highest importance)
    representative_features = []
    redundant_features = {}

    for cluster_id, cluster_features in feature_clusters.items():
        # Sort by importance
        cluster_features_sorted = sorted(
            cluster_features,
            key=lambda f: feature_importances[f],
            reverse=True
        )

        representative = cluster_features_sorted[0]
        representative_features.append(representative)

        # Mark others as redundant
        for feature in cluster_features_sorted[1:]:
            redundant_features[feature] = representative

    return {
        'feature_clusters': dict(feature_clusters),
        'representative_features': representative_features,
        'redundant_features': redundant_features,
        'n_clusters': len(feature_clusters),
        'redundancy_ratio': len(redundant_features) / len(selected_features)
    }
```

**Action:** Remove redundant features, use only representatives

---

#### 7. Mutual Information Stability
**Purpose:** Robust measure of feature-target relationship

```python
def calculate_mi_stability(X, y, selected_features, cv_folds=10):
    """
    Calculate mutual information stability across CV folds.

    Returns:
        - mi_scores: dict[feature -> list[float]]
        - mi_mean: dict[feature -> float]
        - mi_std: dict[feature -> float]
        - mi_cv: dict[feature -> float] (coefficient of variation)
        - stable_mi_features: list[str] (CV < 0.3)
    """
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=cv_folds)

    mi_scores = defaultdict(list)

    for train_idx, _ in tscv.split(X):
        X_fold = X.iloc[train_idx][selected_features]
        y_fold = y.iloc[train_idx]

        # Calculate MI for this fold
        mi = mutual_info_regression(X_fold, y_fold, random_state=42)

        for feature, mi_value in zip(selected_features, mi):
            mi_scores[feature].append(mi_value)

    # Calculate stability metrics
    mi_mean = {f: np.mean(scores) for f, scores in mi_scores.items()}
    mi_std = {f: np.std(scores) for f, scores in mi_scores.items()}
    mi_cv = {
        f: (mi_std[f] / mi_mean[f] if mi_mean[f] > 0 else np.inf)
        for f in selected_features
    }

    # Features with stable MI (low CV)
    stable_mi_features = [f for f in selected_features if mi_cv[f] < 0.3]

    return {
        'mi_scores': dict(mi_scores),
        'mi_mean': mi_mean,
        'mi_std': mi_std,
        'mi_cv': mi_cv,
        'stable_mi_features': stable_mi_features
    }
```

**Threshold:** Keep features with `mi_cv < 0.3` (stable MI)

---

### **Tier 3: Advanced Metrics (Nice to Have)**

#### 8. SHAP Interaction Stability
**Purpose:** Validate that feature interactions are stable

```python
def analyze_shap_interaction_stability(X, y, selected_features, cv_folds=5):
    """
    Measure stability of SHAP interaction values across CV folds.

    Returns:
        - interaction_matrices: list[np.ndarray] (per fold)
        - interaction_consistency: float (average pairwise correlation)
        - stable_interactions: list[tuple[str, str]]
    """
    import shap
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=cv_folds)

    interaction_matrices = []

    for train_idx, _ in tscv.split(X):
        X_fold = X.iloc[train_idx][selected_features]
        y_fold = y.iloc[train_idx]

        # Train model
        model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        model.fit(X_fold, y_fold)

        # Calculate SHAP interaction values
        explainer = shap.TreeExplainer(model)
        shap_interaction_values = explainer.shap_interaction_values(X_fold.sample(min(500, len(X_fold))))

        # Average across samples
        avg_interaction = np.mean(np.abs(shap_interaction_values), axis=0)
        interaction_matrices.append(avg_interaction)

    # Calculate consistency (correlation between matrices)
    correlations = []
    for i in range(len(interaction_matrices)):
        for j in range(i + 1, len(interaction_matrices)):
            corr = np.corrcoef(
                interaction_matrices[i].flatten(),
                interaction_matrices[j].flatten()
            )[0, 1]
            correlations.append(corr)

    interaction_consistency = np.mean(correlations)

    # Identify stable interactions
    # Average interaction matrix
    avg_matrix = np.mean(interaction_matrices, axis=0)
    std_matrix = np.std(interaction_matrices, axis=0)
    cv_matrix = std_matrix / (avg_matrix + 1e-10)

    stable_interactions = []
    for i in range(len(selected_features)):
        for j in range(i + 1, len(selected_features)):
            if cv_matrix[i, j] < 0.3 and avg_matrix[i, j] > 0.01:
                stable_interactions.append((selected_features[i], selected_features[j]))

    return {
        'interaction_matrices': interaction_matrices,
        'interaction_consistency': interaction_consistency,
        'stable_interactions': stable_interactions,
        'avg_interaction_matrix': avg_matrix,
        'cv_interaction_matrix': cv_matrix
    }
```

---

## Implementation Priority

### **Phase 1: Immediate (This Week)**
1. ✅ Null Importance Distribution → Filter statistically significant features
2. ✅ Permutation Importance CI → Validate feature significance
3. ✅ Selection Frequency Distribution → Diagnose instability patterns
4. ✅ Enable Temporal Drift Detection → Already implemented, just enable reporting

**Expected Impact:** Remove 30-50% of noisy features, improve stability to 40-50%

### **Phase 2: Short-term (Next Sprint)**
5. ✅ Walk-Forward Feature Validation → Validate OOS performance
6. ✅ Feature Redundancy Clustering → Remove correlated features
7. ✅ Mutual Information Stability → Robust feature-target relationship

**Expected Impact:** Reduce feature set to 20-30 truly predictive features, stability > 60%

### **Phase 3: Long-term (Future Enhancement)**
8. ⏭️ SHAP Interaction Stability → Validate complex interactions
9. ⏭️ Regime-Aware Stability → Separate bull/bear/sideways market features
10. ⏭️ Ensemble Feature Selection → Combine multiple selection methods

---

## Integration Plan

### **Where to Add New Metrics**

**File:** `/home/user/Ares/src/training/steps/pre_training/components/final_feature_selection.py`

**Method:** `get_enhanced_analysis()` at line 1199

```python
def get_enhanced_analysis(self) -> Dict[str, Any]:
    """Get comprehensive enhanced feature analysis."""

    enhanced_analysis = {
        # Existing metrics
        'correlation_analysis': self.correlation_analysis,
        'redundancy_analysis': self.redundancy_analysis,
        'stability_analysis': self.stability_analysis,
        'cv_analysis': self.cv_analysis,
        'baseline_comparison': self.baseline_comparison,

        # NEW: Add these
        'null_importance_analysis': self.null_importance_analysis,  # Phase 1
        'importance_ci_analysis': self.importance_ci_analysis,  # Phase 1
        'frequency_distribution_analysis': self.frequency_distribution_analysis,  # Phase 1
        'temporal_drift_analysis': self.temporal_drift_analysis,  # Phase 1
        'walk_forward_validation': self.walk_forward_validation,  # Phase 2
        'redundancy_clustering': self.redundancy_clustering,  # Phase 2
        'mi_stability_analysis': self.mi_stability_analysis,  # Phase 2
    }

    return enhanced_analysis
```

### **Report Template Enhancement**

**File:** `/home/user/Ares/src/training/steps/pre_training/feature_generation_final_feature_selection_step.py`

Add new section to report:

```markdown
## Statistical Validation

### Null Importance Analysis
- **Significant Features:** {n_significant} / {total_features}
- **Mean P-Value:** {mean_p_value:.4f}
- **False Discovery Rate:** {fdr:.2%}

### Permutation Importance Confidence
- **Features with CI > 0:** {n_significant_ci}
- **Average CV:** {avg_cv:.3f}
- **High Confidence Features:** {n_high_confidence}

### Selection Frequency Distribution
- **0-20% selection:** {freq_0_20} features
- **20-40% selection:** {freq_20_40} features
- **40-60% selection:** {freq_40_60} features
- **60-80% selection:** {freq_60_80} features
- **80-100% selection:** {freq_80_100} features
- **Distribution Mode:** {mode}

### Walk-Forward Validation
- **Optimal Feature Count:** {optimal_n_features}
- **Maximum OOS R²:** {max_r2:.4f}
- **Features with Positive Contribution:** {n_positive_contrib}

### Temporal Drift Analysis
- **Mean Temporal Consistency:** {mean_temporal_consistency:.3f}
- **Stable Temporal Features:** {n_stable_temporal}
- **Max Drift Slope:** {max_drift_slope:.4f}
```

---

## Success Criteria

After implementing Phase 1 + Phase 2 metrics, aim for:

✅ **CV Consistency:** > 40% (up from 14%)
✅ **Stability Score:** > 70% (up from 56.82%)
✅ **Significant Features (p < 0.05):** > 80%
✅ **Walk-Forward R²:** > 0.1 (positive OOS performance)
✅ **Redundancy Ratio:** < 0.3 (less than 30% redundant)

---

## References

- Nogueira, S., Sechidis, K., & Brown, G. (2017). "On the Stability of Feature Selection Algorithms"
- Bommert, A., Sun, X., Bischl, B., Rahnenführer, J., & Lang, M. (2020). "Benchmark for filter methods for feature selection in high-dimensional classification data"
- Altmann, A., Toloşi, L., Sander, O., & Lengauer, T. (2010). "Permutation importance: a corrected feature importance measure"
- Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions" (SHAP)

---

**Document Version:** 1.0
**Date:** 2025-11-13
**Author:** Claude (Feature Selection Analysis)
