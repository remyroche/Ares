# Regime Feature Selection Analysis

**Date:** 2025-10-28  
**Purpose:** Analyze the effectiveness and appropriateness of `regime_feature_selection` for regime clustering

---

## Executive Summary

The `regime_feature_selection` step uses **EconomicRegimeFeatureSelector** (registered in the pipeline) with a sophisticated multi-target approach. The implementation is **appropriate but has conceptual issues** for regime clustering purposes due to a potential circular dependency concern.

### Key Findings:

✅ **Strengths:**
- Well-designed multi-target scoring system
- Comprehensive feature quality metrics
- Good filtering pipeline (category → statistical → variance → correlation)
- Optional TreeSHAP integration for advanced feature importance
- Computational optimizations with VectorBT

❌ **Critical Concerns:**
1. **Circular dependency risk**: Uses economic targets (e.g., `close_return`, `volatility_20`) instead of regime labels
2. **Misalignment with clustering goal**: Selects features good at predicting returns, not necessarily good at discovering regimes
3. **TreeSHAP underutilized**: Config shows TreeSHAP enabled but code suggests it may not be the primary path

---

## Implementation Details

### 1. Active Implementation

**File:** `src/training/steps/market_analysis/economic_regime_feature_selector.py`  
**Class:** `EconomicRegimeFeatureSelector(BaseStep)`  
**Registration:** Line 33 in `__init__.py` as `"regime_feature_selection"`

### 2. Feature Selection Pipeline

The selection process follows a **four-stage filtering approach**:

```
Input Features (all)
    ↓
[1] Category Filtering (exclude microstructure, support_resistance)
    ↓
[2] Statistical Pruning (remove constant, infinite, extreme outliers)
    ↓
[3] Variance-based Pruning (remove low-variance features)
    ↓
[4] Correlation-based Pruning (remove highly correlated features)
    ↓
[5] Multi-target Scoring
    ↓
[6] Optimal Selection (adaptive thresholds)
    ↓
Selected Features (25 target)
```

### 3. Multi-Target Scoring System

**The core issue:** Features are scored against **10 economic targets**, not regime labels:

```yaml
Target Columns & Weights:
  - close_return: 8%           # Price movements
  - volume_log_return: 5%      # Volume patterns
  - price_range_pct: 30%       # Relative volatility ⚠️ HIGHEST
  - body_size_pct: 5%          # Price efficiency
  - volume_return: 5%          # Volume momentum
  - close_log_return: 5%       # Log price movements
  - price_range: 25%           # Absolute volatility ⚠️ HIGH
  - trades: 0%                 # Trade patterns (disabled)
  - volatility_20: 20%         # Realized volatility ⚠️ HIGH
  - cmf: 2%                    # Volume imbalance/order flow
```

**Analysis:** 75% of weight is on volatility-related targets (price_range_pct, price_range, volatility_20). This creates features optimized for **volatility prediction**, not **regime discovery**.

### 4. Feature Scoring Metrics

Each feature receives 5 component scores (code lines 1561-1568):

```python
composite_score = (
    economic_significance * 0.35 +      # Correlation with economic targets
    regime_discrimination * 0.25 +      # F-ratio between regime groups
    clustering_quality * 0.20 +         # Silhouette score approximation
    stability_score * 0.15 +            # Cross-validation consistency
    regime_transition_score * 0.05      # Transition detection ability
)
```

**Issue:** `economic_significance` (35% weight) measures correlation with economic targets like `price_range_pct`, NOT regime separability.

### 5. Regime Discrimination Calculation

```python
def _calculate_regime_discrimination(self, feature_data: pd.Series, labels: pd.Series) -> float:
    """Calculate regime discrimination using F-ratio."""
    # Lines 2120-2160
    
    # Between-regime variance
    between_var = np.var(regime_means)
    
    # Within-regime variance
    within_var = np.mean(regime_vars)
    
    # F-ratio
    f_ratio = between_var / within_var
```

**Critical Finding:** This method expects `labels` to be regime labels, but in multi-target mode, it receives **economic targets** (e.g., `close_return` values), not regime IDs!

This means the "regime discrimination" score is actually measuring how well the feature discriminates different **return levels**, not different **regimes**.

---

## Circular Dependency Issue

### The Problem

```
┌─────────────────────────────────────────────────┐
│  1. Regime Clustering                           │
│     - Discovers regimes from market conditions  │
│     - Output: regime_labels                     │
└─────────────────────────────────────────────────┘
                    ↓ should come BEFORE
┌─────────────────────────────────────────────────┐
│  2. Regime Feature Selection                    │
│     - Should use regime_labels to score         │
│     - But actually uses economic targets!       │
└─────────────────────────────────────────────────┘
```

### Current Flow (Problematic)

From `regime_clustering_step.py` lines 131-135:

```python
# Load selected features from regime_feature_selection step
tprint("📥 Loading selected features from regime_feature_selection...", "INFO")
selected_features = self._load_selected_features(config)
if selected_features:
    tprint(f"✅ Loaded {len(selected_features)} selected features", "SUCCESS")
```

**This implies:**
1. `regime_feature_selection` runs BEFORE `regime_clustering`
2. It selects features WITHOUT knowing what the regimes are
3. It uses economic targets as proxies for regimes
4. Clustering then uses these pre-selected features

### Why This Is Problematic

1. **Optimization Mismatch:** Features are optimized for predicting returns/volatility, not for separating market regimes
2. **Assumption Risk:** Assumes high-volatility-predictive features = good regime-separating features (not always true)
3. **Loss of Information:** May exclude features that are excellent for regime separation but poor for return prediction
4. **Regime Concept Confusion:** Conflates "regime" (market state) with "target" (economic outcome)

---

## TreeSHAP Integration

### Configuration

```yaml
treeshap_config:
  enable: true
  n_estimators: 100
  max_depth: 8
  learning_rate: 0.1
  correlation_threshold: 0.85
  diversity_weight: 0.2
  treeshap_weight: 0.6
  correlation_weight: 0.2
  target_feature_count: 25
```

### Status

From code analysis (lines 1359-1584), the TreeSHAP path exists but the multi-target scoring appears to be the primary execution path. The config shows `primary_method: "treeshap"` but the execute method (line 561) calls `_score_features_multi_target` directly when `multi_target_enabled: true`.

**Finding:** TreeSHAP integration exists but may be bypassed in favor of multi-target approach.

---

## Effectiveness Assessment

### For Return Prediction: ✅ **Highly Effective**

The selected features would be excellent for:
- Volatility regime prediction
- Return forecasting models
- Risk-adjusted trading strategies
- Economic state classification

### For Regime Clustering: ⚠️ **Moderately Appropriate**

**Pros:**
- High-quality features (statistically sound)
- Good coverage of volatility dynamics
- Computationally optimized
- Comprehensive validation

**Cons:**
- Not optimized for regime discovery
- May miss important regime-separating features
- Biased toward volatility (75% weight)
- Circular dependency with clustering

---

## Recommendations

### 1. **Unsupervised Pre-Selection** (Best Practice)

Add an unsupervised feature selection phase BEFORE clustering:

```python
def _unsupervised_regime_feature_selection(self, features_df: pd.DataFrame) -> List[str]:
    """
    Select features for regime clustering without using regime labels.
    
    Uses:
    - Variance filtering (keep high-variance features)
    - Correlation pruning (remove redundant features)
    - Diversity maximization (ensure feature coverage)
    """
    # Step 1: High variance filtering
    variances = features_df.var()
    high_var_features = variances[variances > variances.quantile(0.10)].index
    
    # Step 2: Correlation-based diversity
    decorrelated_features = self._remove_highly_correlated(high_var_features)
    
    # Step 3: Category balance (ensure diverse feature types)
    balanced_features = self._balance_feature_categories(decorrelated_features)
    
    return balanced_features
```

**Note:** This method already exists in `EnhancedRegimeFeatureSelector` (lines 542-640) but is not the registered implementation!

### 2. **Two-Stage Selection** (Hybrid Approach)

```
Stage 1: Unsupervised pre-selection for clustering
    ↓
Clustering (discovers regimes)
    ↓
Stage 2: Supervised refinement using regime labels
    ↓
Final feature set for regime-specific models
```

### 3. **Fix Multi-Target Scoring** (Quick Fix)

Change the `regime_discrimination` calculation to use actual regime labels:

```python
# Current (problematic):
regime_discrimination = self._calculate_regime_discrimination(
    feature_aligned, target_aligned  # target_aligned = close_return values
)

# Fixed:
regime_discrimination = self._calculate_regime_discrimination(
    feature_aligned, regime_labels  # regime_labels = actual regime IDs
)
```

But this still requires regime labels to exist first, creating the circular dependency.

### 4. **Use the Other Implementation**

Consider switching to `EnhancedRegimeFeatureSelector` (regime_feature_selector.py):

```python
# In __init__.py, change line 33 from:
step_registry.register("regime_feature_selection", EconomicRegimeFeatureSelector)

# To:
from .regime_feature_selector import EnhancedRegimeFeatureSelector
step_registry.register("regime_feature_selection", EnhancedRegimeFeatureSelector)
```

This implementation has proper unsupervised mode (lines 542-640).

---

## Conclusion

### Is it effective?

**For economic feature selection:** ✅ Yes, very effective  
**For regime clustering:** ⚠️ Partially - works but suboptimal

### Is it appropriate?

**Conceptually:** ❌ No - has circular dependency issue  
**Practically:** ⚠️ Works but could be better

### Key Issue

The fundamental problem is that `regime_feature_selection` is trying to select features for regime **clustering** (unsupervised) but is using **supervised** methods with economic targets. This creates:

1. **Optimization mismatch:** Features good for predicting returns ≠ features good for discovering regimes
2. **Circular logic:** Can't use regime labels to select features before regimes are discovered
3. **Bias risk:** Heavy volatility focus (75%) may miss other regime-defining characteristics

### Recommended Action

**Option A (Quick):** Switch to `EnhancedRegimeFeatureSelector` which has proper unsupervised mode

**Option B (Best):** Implement two-stage selection:
- Stage 1: Unsupervised selection for clustering
- Stage 2: Supervised refinement after clustering

**Option C (Current):** Keep using `EconomicRegimeFeatureSelector` if the goal is to discover **volatility-based regimes** specifically, as the current implementation is heavily optimized for volatility features.

---

## Additional Notes

### Code Quality
- Well-structured and modular ✅
- Comprehensive logging with tprint ✅
- Good error handling ✅
- Extensive configuration options ✅
- Hardware optimizations present ✅

### Documentation
- Inline documentation is good ✅
- Config file is well-commented ✅
- Missing: High-level architecture docs ⚠️

### Testing
- Test files exist (`test_regime_feature_selector_integration.py`) ✅
- Coverage appears limited to integration, not unit tests ⚠️

---

**End of Analysis**
