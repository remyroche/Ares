# Regime Clustering Feature Generation & Selection Review

**Date**: 2025-10-28  
**Reviewer**: AI Assistant  
**Focus**: Feature pipeline before `regime_clustering` step

---

## Executive Summary

This document reviews the feature generation and selection pipeline that runs prior to regime clustering. The system has a well-structured multi-stage approach, but there are several issues and opportunities for improvement that could impact clustering quality and efficiency.

### Key Findings

🔴 **Critical Issues**:
1. Feature selection may happen **after** initial clustering (circular dependency)
2. Inconsistent feature set between HDBSCAN discovery and regime clustering
3. Light mode may over-filter features needed for clustering quality

🟡 **Important Issues**:
1. Feature categorization system not fully leveraged in selection process
2. Memory optimization may discard important features too early
3. Feature importance calculation relies on cluster labels (chicken-egg problem)

🟢 **Strengths**:
1. Comprehensive feature categorization framework
2. Multiple feature families (regime, entropy, spectral, statistical)
3. Hardware-optimized feature generation

---

## Pipeline Architecture

### Current Flow

```
1. HDBSCAN Regime Discovery
   ├── Feature Generation (300+ features)
   │   ├── Entropy Features
   │   ├── Spectral Features
   │   ├── Regime Features
   │   └── Normalization Features
   ├── Feature Selection (→ 30-50 features)
   │   └── Using MRMR/LASSO/Mutual Info
   └── HDBSCAN Clustering
       └── Creates regime_labels

2. Regime Feature Selection Step
   ├── Load regime_labels from HDBSCAN
   ├── Load clustering_features from HDBSCAN
   ├── Feature Selection using TreeSHAP
   │   └── Uses regime_labels as TARGET
   └── Save selected_features

3. Regime Clustering Step
   ├── Load selected_features from step 2
   ├── Load HDBSCAN artifacts
   └── Refine clusters using economic validation
```

### Issue: Circular Dependency

The current flow has a logical issue:
- **HDBSCAN** clusters data to create `regime_labels`
- **Feature Selection** uses `regime_labels` to select features
- **Regime Clustering** uses selected features to refine clusters

This creates a circular dependency where feature selection depends on clustering results, but clustering quality depends on feature selection.

---

## Feature Generation Analysis

### 1. Feature Categories (from `regime_feature_categorization.py`)

The system defines clear feature categories with intended use cases:

#### Core Regime Features (Priority: 10)
- **Purpose**: Essential features for regime identification
- **Use Cases**: HDBSCAN clustering, Regime clustering, Models training, Ensemble training
- **Features**: 
  - `regime_persistence`, `vol_regime_strength`, `vol_clustering`
  - `vol_regime_change`, `volume_regime_strength`, `volume_clustering`
  - `statistical_persistence`, `distribution_stability`
- **Characteristics**: Stable, lookahead-safe

#### Advanced Regime Features (Priority: 8)
- **Purpose**: Complex regime analysis
- **Use Cases**: HDBSCAN clustering, Regime clustering, Models training
- **Features**:
  - `regime_entropy`, `regime_complexity`, `regime_fractal_dimension`
  - `regime_hurst_exponent`, `regime_memory_strength`
- **Characteristics**: Stable, lookahead-safe

#### Clustering-Only Features (Priority: 9)
- **Purpose**: Designed specifically for clustering algorithms
- **Use Cases**: HDBSCAN clustering, Regime clustering **ONLY**
- **Features**:
  - `price_distance`, `volume_distance`, `cluster_compactness`
  - `separation_strength`, `cluster_consistency`, `temporal_stability`
- **Characteristics**: **NEVER for live trading**, stable, lookahead-safe

#### Structural Trend Features (Priority: 8)
- **Purpose**: Structural trend regime analysis
- **Features**:
  - `structural_persistence`, `trend_regime_persistence`
  - `market_structure_strength`, `trend_transition_prob`

#### Cross-Asset Features (Priority: 6)
- **Purpose**: Multi-asset regime analysis
- **Features**:
  - `cross_timeframe_corr`, `regime_persistence_score`
  - `price_volume_sync`, `regime_sync_strength`

#### Transition Features (Priority: 8)
- **Purpose**: Regime change detection
- **Features**:
  - `cusum_change_point`, `change_point_prob`, `regime_change_intensity`
  - `transition_prob`, `regime_persistence_prob`
- **Characteristics**: **Not stable** (designed for change detection)

### 2. Feature Generators

The system uses multiple feature generator families:

```python
# From optimized_hdbscan_regime_discovery.py
1. Entropy Features (if enabled)
   └── create_default_entropy_generators()

2. Spectral Features (if enabled)
   └── create_default_spectral_wavelet_generators()

3. Regime Features (if enabled)
   └── create_default_regime_generators()
```

**Execution Mode Impact**:

| Mode  | Max Features | Entropy | Spectral | Regime | Normalization |
|-------|--------------|---------|----------|--------|---------------|
| Full  | 300          | ✅      | ✅       | ✅     | ✅            |
| Light | 30           | ❌      | ❌       | ✅     | ✅            |
| Blank | 40           | ❌      | ❌       | ❌     | ❌            |

### 3. Feature Generation Issues

#### Issue 1: Light Mode Over-Reduction
```python
# From OptimizedHDBSCANRegimeDiscoveryConfig.__post_init__
if self.execution_mode == "light":
    self.max_features = 30
    self.enable_entropy_features = False
    self.enable_spectral_features = False
```

**Problem**: Light mode reduces features to 30 and disables entropy/spectral features, which may be critical for regime differentiation.

**Impact**: 
- Missing entropy features (regime complexity indicators)
- Missing spectral features (frequency domain information)
- Only 30 features may not capture regime diversity

**Recommendation**: Increase light mode features to 40-50 and enable entropy features.

#### Issue 2: Blank Mode Too Minimal
```python
elif self.execution_mode == "blank":
    self.max_features = 40
    self.enable_entropy_features = False
    self.enable_spectral_features = False
    self.enable_regime_features = False  # ❌ Critical!
    self.enable_normalization_features = False
```

**Problem**: Blank mode disables regime features entirely, which are the most important for regime clustering.

**Impact**: Blank mode cannot properly identify regimes without regime-specific features.

**Recommendation**: Always enable regime features, even in blank mode.

#### Issue 3: Feature Selection Method Not Aligned

```python
# From OptimizedHDBSCANRegimeDiscoveryConfig
feature_selection_method: str = 'mrmr'  # 'mrmr', 'lasso', 'mutual_info'
```

HDBSCAN uses MRMR (Maximum Relevance Minimum Redundancy), but:
- MRMR requires **labels** to calculate relevance
- For unsupervised clustering, mutual information or variance-based methods may be more appropriate

**Recommendation**: Use unsupervised feature selection methods before clustering:
- Variance-based filtering
- Correlation-based filtering
- PCA/UMAP dimensionality reduction

---

## Feature Selection Analysis

### 1. Two-Stage Feature Selection

The system has two distinct feature selection stages:

#### Stage 1: HDBSCAN Internal Selection
```python
# In optimized_hdbscan_regime_discovery.py
if self.config.enable_feature_selection:
    # Select features using MRMR/LASSO/Mutual Info
    selected_features = self._select_features(
        features_df, 
        method=self.config.feature_selection_method,
        max_features=self.config.max_features
    )
```

**Timing**: Before clustering  
**Method**: MRMR/LASSO/Mutual Info  
**Target**: None (unsupervised)  
**Output**: clustering_features artifact

#### Stage 2: Regime Feature Selection Step
```python
# In regime_feature_selector.py
def select_features(
    self,
    features_df: pd.DataFrame,
    regime_labels: pd.Series,  # ❌ Uses labels from clustering!
    feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    # Use TreeSHAP with regime labels as target
    selection_results = self._run_treeshap_selection(
        features_df, regime_labels, feature_names
    )
```

**Timing**: After HDBSCAN clustering  
**Method**: TreeSHAP  
**Target**: regime_labels (from HDBSCAN)  
**Output**: selected_features artifact

### 2. Feature Selection Issues

#### Issue 1: Circular Dependency
```
HDBSCAN Clustering → regime_labels
       ↓
Feature Selection (TreeSHAP) → selected_features
       ↓
Regime Clustering (uses selected_features)
```

**Problem**: Feature selection depends on clustering results, creating a feedback loop.

**Why This Matters**:
1. Features selected to discriminate between HDBSCAN clusters may not be optimal for economic regimes
2. If HDBSCAN produces poor clusters, feature selection will optimize for poor clusters
3. Cannot iterate on feature selection without re-running clustering

**Recommendation**: Consider two approaches:
1. **Iterative Approach**: 
   - Initial clustering with all features
   - Feature selection based on clusters
   - Re-cluster with selected features
   - Iterate 2-3 times
2. **Hybrid Approach**:
   - Use unsupervised feature selection (variance, correlation) first
   - Apply clustering
   - Use supervised selection (TreeSHAP) for refinement

#### Issue 2: Feature Importance Calculation
```python
# From enhanced_regime_clustering_integration.py
def get_feature_importance_for_regime_clustering(
    self, data: pd.DataFrame, 
    clustering_result: Dict[str, Any]
) -> Dict[str, float]:
    # Calculate feature importance based on regime separation
    # Uses F-ratio (between-cluster variance / within-cluster variance)
    for i, feature_name in enumerate(feature_names):
        # Between-cluster variance
        between_var = sum([size * (mean - overall_mean)**2 
                          for mean, size in zip(cluster_means, cluster_sizes)])
        
        # Within-cluster variance
        within_var = sum([(values - cluster_mean)**2 
                         for cluster_id in clusters])
        
        # F-ratio
        f_ratio = between_var / within_var
        importance_scores[feature_name] = f_ratio
```

**Problem**: This calculates feature importance **after** clustering, which means:
- Cannot use importance for feature selection before clustering
- Importance scores are biased toward features that separate the **found** clusters
- May miss features that could lead to **better** clusters

#### Issue 3: TreeSHAP Uses Regime Labels as Target
```python
# From regime_feature_selector.py
def _run_treeshap_selection(
    self,
    features_df: pd.DataFrame,
    regime_labels: pd.Series,  # ❌ Labels as target
    feature_names: Optional[List[str]]
) -> Dict[str, Any]:
    # Run TreeSHAP selection with regime labels as target
    selection_results = self.treeshap_selector.select_features(
        optimized_features,
        regime_labels,  # Using labels as supervised target
        feature_names=feature_names,
        max_features=self.config.max_features,
        min_importance=self.config.min_feature_importance
    )
```

**Problem**: TreeSHAP is a supervised feature selection method that uses regime labels as the target variable.

**Why This Is Problematic**:
1. **Overfitting to initial clusters**: Features are selected to match the initial HDBSCAN clustering, which may not be economically meaningful
2. **Loss of exploration**: Features that could lead to better regime definitions are discarded
3. **Confirmation bias**: The system reinforces the initial clustering rather than improving it

**Example Scenario**:
```
Initial HDBSCAN: Finds 3 clusters based on volatility patterns
Feature Selection: Selects features that best discriminate these 3 clusters
Final Clustering: Can only refine the 3 volatility-based clusters
Missed: A 4th cluster based on volume patterns (feature discarded)
```

#### Issue 4: Feature Categorization Not Fully Leveraged
```python
# The categorization system exists but isn't used in selection
categorizer = RegimeFeatureCategorizer()
hdbscan_features = categorizer.get_priority_features(
    FeatureUseCase.HDBSCAN_CLUSTERING, 100
)
```

**Problem**: The comprehensive feature categorization system defines which features should be used for which purposes, but this information is not fully utilized in the selection process.

**Current Behavior**:
- Feature selection uses generic methods (MRMR, TreeSHAP)
- Ignores the priority and use-case information from categorizer
- May select features not intended for clustering

**Recommendation**: Integrate categorization into selection:
```python
# Proposed approach
def select_features_with_categories(
    features_df: pd.DataFrame,
    use_case: FeatureUseCase,
    max_features: int
) -> List[str]:
    # Get priority features for use case
    priority_features = categorizer.get_priority_features(use_case, max_features * 2)
    
    # Filter features_df to priority features
    available_priority_features = [f for f in priority_features if f in features_df.columns]
    
    # Apply selection within priority features
    selected = apply_selection_method(
        features_df[available_priority_features],
        method='variance_threshold'  # Unsupervised
    )
    
    return selected[:max_features]
```

### 3. Feature Selection in Enhanced Integration

The `enhanced_regime_clustering_integration.py` provides a more sophisticated approach:

```python
class EnhancedRegimeClusteringIntegration:
    def get_enhanced_regime_features(
        self, 
        data: pd.DataFrame, 
        cluster_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        # Get comprehensive features first
        comprehensive_result = self.get_comprehensive_regime_features(data)
        
        # If cluster labels provided, use enhanced selection
        if cluster_labels is not None:
            enhanced_selector.select_optimal_features(
                features, cluster_labels, feature_categories, max_features
            )
```

**Good**: This allows iterative refinement with cluster labels  
**Issue**: Still relies on having cluster labels first

---

## Recommended Improvements

### 1. Implement Three-Stage Feature Selection

```
Stage 1: Unsupervised Pre-filtering
├── Variance threshold (remove low-variance features)
├── Correlation filtering (remove highly correlated features)
└── Output: 100-150 features

Stage 2: Initial Clustering
├── Use pre-filtered features
├── HDBSCAN clustering
└── Output: regime_labels_v1

Stage 3: Supervised Refinement (OPTIONAL)
├── Use regime_labels_v1 to identify important features
├── Re-cluster with refined features
└── Output: regime_labels_v2
```

**Benefits**:
- Reduces dimensionality before clustering
- Avoids circular dependency for initial clustering
- Allows optional refinement with labeled data

### 2. Respect Feature Categorization

```python
def get_features_for_regime_clustering(
    data: pd.DataFrame,
    execution_mode: str
) -> Dict[str, np.ndarray]:
    """Get features specifically designed for regime clustering."""
    
    # Get priority features from categorization system
    categorizer = RegimeFeatureCategorizer()
    priority_features = categorizer.get_priority_features(
        FeatureUseCase.REGIME_CLUSTERING, 
        max_features=80  # Always use category recommendations
    )
    
    # Filter by availability and quality
    features = {}
    for feature_name in priority_features:
        if feature_name in data.columns:
            feature_values = data[feature_name].values
            
            # Quality checks
            if not has_sufficient_variance(feature_values):
                continue
            if has_too_many_nans(feature_values):
                continue
                
            features[feature_name] = feature_values
    
    # Always include high-priority categories
    required_categories = [
        'core_regime',        # Essential regime features
        'structural_trend',   # Trend patterns
        'clustering_only'     # Clustering-specific features
    ]
    
    # Ensure minimum representation from each category
    for category in required_categories:
        category_features = categorizer.get_features_for_category(category)
        included = [f for f in category_features if f in features]
        
        if len(included) < 5:  # Minimum 5 features per critical category
            logger.warning(f"Insufficient {category} features: {len(included)}/5")
    
    return features
```

### 3. Improve Light Mode Configuration

```python
# Current light mode (too restrictive)
if self.execution_mode == "light":
    self.max_features = 30               # ❌ Too few
    self.enable_entropy_features = False  # ❌ May be important
    self.enable_spectral_features = False # ❌ May be important

# Recommended light mode
if self.execution_mode == "light":
    self.max_features = 50                # ✅ More features
    self.enable_entropy_features = True   # ✅ Enable (cheap to compute)
    self.enable_spectral_features = False # ⚠️ Keep disabled (expensive)
    self.enable_regime_features = True    # ✅ Always enable
    
    # Use faster feature generation methods
    self.use_fast_entropy = True          # Use approximate entropy
    self.use_cached_features = True       # Cache computed features
```

### 4. Separate Clustering Features from Live Trading Features

```python
# Create separate feature sets
def prepare_features_for_task(
    data: pd.DataFrame,
    task: FeatureUseCase
) -> Dict[str, np.ndarray]:
    """Prepare features appropriate for the specific task."""
    
    if task == FeatureUseCase.HDBSCAN_CLUSTERING:
        # Include clustering-only features
        features = get_all_clustering_features(data)
        
    elif task == FeatureUseCase.LIVE_TRADING:
        # NEVER include clustering-only features
        features = get_live_trading_safe_features(data)
        
        # Validate no clustering features leaked
        clustering_features = get_clustering_only_features()
        leakage = [f for f in features if f in clustering_features]
        if leakage:
            raise ValueError(f"Clustering features leaked to live trading: {leakage}")
    
    return features
```

### 5. Add Feature Selection Validation

```python
def validate_feature_selection(
    selected_features: List[str],
    use_case: FeatureUseCase,
    expected_categories: List[str]
) -> Dict[str, Any]:
    """Validate that feature selection aligns with use case requirements."""
    
    categorizer = RegimeFeatureCategorizer()
    
    # Check feature use case alignment
    valid_features, invalid_features = categorizer.validate_feature_usage(
        selected_features, use_case
    )
    
    # Check category representation
    category_breakdown = {}
    for feature in selected_features:
        category = categorizer.get_feature_category(feature)
        category_breakdown[category] = category_breakdown.get(category, 0) + 1
    
    # Validate minimum representation
    issues = []
    for expected_category in expected_categories:
        count = category_breakdown.get(expected_category, 0)
        if count < 3:  # Minimum 3 features per expected category
            issues.append(f"Insufficient {expected_category} features: {count}")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'category_breakdown': category_breakdown,
        'valid_features': valid_features,
        'invalid_features': invalid_features
    }
```

### 6. Implement Iterative Feature Selection

```python
def iterative_feature_selection_clustering(
    data: pd.DataFrame,
    max_iterations: int = 3
) -> Dict[str, Any]:
    """Iterative feature selection and clustering."""
    
    # Stage 1: Initial unsupervised feature selection
    features = unsupervised_feature_selection(
        data, 
        max_features=100,
        methods=['variance', 'correlation']
    )
    
    best_result = None
    best_score = -np.inf
    
    for iteration in range(max_iterations):
        # Cluster with current features
        cluster_result = hdbscan_clustering(features)
        
        # Evaluate clustering quality
        score = evaluate_clustering_quality(cluster_result)
        
        if score > best_score:
            best_score = score
            best_result = cluster_result
        
        # Refine features based on clustering
        if iteration < max_iterations - 1:
            features = supervised_feature_refinement(
                features,
                cluster_result['labels'],
                max_features=80
            )
    
    return best_result
```

---

## Specific Code Issues

### 1. Regime Clustering Step - Feature Loading
**File**: `regime_clustering_step.py`  
**Lines**: 114-120

```python
# Load selected features from regime_feature_selection step
tprint("📥 Loading selected features from regime_feature_selection...", "INFO")
selected_features = self._load_selected_features(config)
if selected_features:
    tprint(f"✅ Loaded {len(selected_features)} selected features", "SUCCESS")
else:
    tprint("⚠️ No selected features found - proceeding without filtering", "WARNING")
```

**Issue**: The step loads features selected by TreeSHAP using regime labels, but doesn't verify:
- Whether selected features are appropriate for clustering
- Whether critical feature categories are represented
- Whether the feature selection was circular

**Recommendation**: Add validation:
```python
if selected_features:
    # Validate feature selection quality
    validation = validate_feature_selection(
        selected_features,
        FeatureUseCase.REGIME_CLUSTERING,
        expected_categories=['core_regime', 'structural_trend', 'clustering_only']
    )
    
    if not validation['valid']:
        tprint(f"⚠️ Feature selection issues: {validation['issues']}", "WARNING")
        # Use fallback feature set
        selected_features = get_fallback_regime_features()
```

### 2. Enhanced Regime Feature Selector - Regime Labels as Target
**File**: `regime_feature_selector.py`  
**Lines**: 584-618

```python
def _run_treeshap_selection(
    self,
    features_df: pd.DataFrame,
    regime_labels: pd.Series,  # ❌
    feature_names: Optional[List[str]]
) -> Dict[str, Any]:
    # Run TreeSHAP selection with regime labels as target
    selection_results = self.treeshap_selector.select_features(
        optimized_features,
        regime_labels,  # ❌ Using labels from clustering
        feature_names=feature_names
    )
```

**Issue**: This creates a circular dependency and may reinforce poor initial clustering.

**Recommendation**: Add an unsupervised mode:
```python
def _run_treeshap_selection(
    self,
    features_df: pd.DataFrame,
    regime_labels: Optional[pd.Series] = None,
    feature_names: Optional[List[str]] = None,
    use_supervised: bool = True
) -> Dict[str, Any]:
    if use_supervised and regime_labels is not None:
        # Supervised selection (for refinement only)
        selection_results = self.treeshap_selector.select_features(
            optimized_features, regime_labels, feature_names
        )
    else:
        # Unsupervised selection (for initial clustering)
        selection_results = self._unsupervised_feature_selection(
            optimized_features, feature_names
        )
    
    return selection_results

def _unsupervised_feature_selection(
    self,
    features_df: pd.DataFrame,
    feature_names: Optional[List[str]]
) -> Dict[str, Any]:
    """Unsupervised feature selection using variance and correlation."""
    
    # 1. Remove low-variance features
    variances = features_df.var()
    high_variance_features = variances[variances > variances.quantile(0.1)].index
    
    # 2. Remove highly correlated features
    corr_matrix = features_df[high_variance_features].corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_drop = [column for column in upper_triangle.columns 
               if any(upper_triangle[column] > 0.95)]
    
    selected_features = [f for f in high_variance_features if f not in to_drop]
    
    return {
        'selected_features': selected_features[:self.config.max_features],
        'selection_method': 'unsupervised_variance_correlation',
        'feature_importance': dict(zip(selected_features, variances[selected_features]))
    }
```

### 3. Optimized HDBSCAN - Execution Mode Feature Limits
**File**: `optimized_hdbscan_regime_discovery.py`  
**Lines**: 149-167

```python
def __post_init__(self):
    """Apply execution mode-based optimizations."""
    if self.execution_mode == "light":
        self.max_features = 30               # ❌
        self.enable_entropy_features = False  # ❌
        self.enable_spectral_features = False # ⚠️
    elif self.execution_mode == "blank":
        self.max_features = 40
        self.enable_regime_features = False  # ❌ Critical!
```

**Issue**: Feature limits are too restrictive and disable important features.

**Recommendation**:
```python
def __post_init__(self):
    """Apply execution mode-based optimizations."""
    if self.execution_mode == "light":
        self.max_features = 50                # ✅ Increased
        self.enable_entropy_features = True   # ✅ Enable (important)
        self.enable_spectral_features = False # ⚠️ OK to disable (expensive)
        self.enable_regime_features = True    # ✅ Always enable
        self.use_fast_methods = True          # ✅ Speed up computation
        
    elif self.execution_mode == "blank":
        self.max_features = 50                # ✅ Increased
        self.enable_entropy_features = False  # ⚠️ Can disable for speed
        self.enable_spectral_features = False # ⚠️ Can disable for speed
        self.enable_regime_features = True    # ✅ NEVER disable
        self.enable_normalization_features = True  # ✅ Re-enable
```

---

## Testing Recommendations

### 1. Feature Selection Quality Tests

```python
def test_feature_selection_quality():
    """Test that feature selection produces high-quality features."""
    
    # Load test data
    data = load_test_market_data()
    
    # Run feature selection
    selected_features = select_features_for_regime_clustering(data)
    
    # Validate results
    assert len(selected_features) >= 40, "Too few features selected"
    assert len(selected_features) <= 80, "Too many features selected"
    
    # Check category representation
    categorizer = RegimeFeatureCategorizer()
    categories = [categorizer.get_feature_category(f) for f in selected_features]
    
    assert 'core_regime' in categories, "Missing core regime features"
    assert 'structural_trend' in categories, "Missing structural trend features"
    assert categories.count('core_regime') >= 5, "Insufficient core regime features"
```

### 2. Circular Dependency Test

```python
def test_no_circular_dependency():
    """Test that feature selection doesn't depend on clustering results."""
    
    data = load_test_market_data()
    
    # Track dependencies
    with DependencyTracker() as tracker:
        # Generate features
        features = generate_features(data)
        
        # Select features (should not depend on clustering)
        selected_features = select_features(features)
        
        # Verify no clustering dependency
        assert not tracker.has_dependency('clustering', 'feature_selection')
```

### 3. Feature Quality Test

```python
def test_feature_quality_for_clustering():
    """Test that selected features are suitable for clustering."""
    
    data = load_test_market_data()
    selected_features = select_features_for_regime_clustering(data)
    feature_matrix = data[selected_features].values
    
    # Test 1: Features have sufficient variance
    variances = np.var(feature_matrix, axis=0)
    assert np.all(variances > 0.01), "Features have insufficient variance"
    
    # Test 2: Features are not too correlated
    corr_matrix = np.corrcoef(feature_matrix.T)
    max_corr = np.max(np.abs(corr_matrix - np.eye(len(selected_features))))
    assert max_corr < 0.95, "Features are too highly correlated"
    
    # Test 3: Features enable cluster separation
    # Run clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    labels = clusterer.fit_predict(feature_matrix)
    
    # Calculate silhouette score
    if len(set(labels)) > 1:
        score = silhouette_score(feature_matrix, labels)
        assert score > 0.2, f"Poor cluster separation: {score}"
```

---

## Conclusion

The regime clustering feature generation and selection pipeline is comprehensive but has several critical issues:

### Critical Issues to Address:
1. **Circular Dependency**: Feature selection depends on clustering results, creating a feedback loop
2. **Overly Restrictive Light Mode**: Disables important features (entropy, reduces to 30 features)
3. **Blank Mode Critical Error**: Disables regime features entirely
4. **Supervised Selection with Clusters**: TreeSHAP uses cluster labels as target, reinforcing initial clustering

### Recommended Priority Actions:
1. **Immediate**: Fix blank mode to always enable regime features
2. **High Priority**: Implement unsupervised pre-filtering before clustering
3. **High Priority**: Increase light mode feature limit to 50 and re-enable entropy
4. **Medium Priority**: Add feature selection validation using categorization system
5. **Medium Priority**: Implement iterative feature selection/clustering
6. **Low Priority**: Add comprehensive testing suite

### Expected Impact:
- **Better Cluster Quality**: Unsupervised pre-filtering will allow exploration of better feature combinations
- **More Robust Regimes**: Increased feature diversity will capture more regime characteristics
- **Faster Execution**: Proper feature filtering will reduce dimensionality while maintaining quality
- **Better Alignment**: Feature categorization integration will ensure appropriate features for each task

---

## Next Steps

1. Review this document with the team
2. Prioritize issues based on impact and effort
3. Create implementation plan for fixes
4. Update configuration defaults
5. Add validation tests
6. Document feature selection best practices
7. Monitor clustering quality improvements

---

**End of Review**
