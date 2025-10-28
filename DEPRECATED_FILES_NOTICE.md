# Deprecated Files Notice

The following files have been deprecated due to the regime feature selection refactoring:

## Deprecated Files

### 1. `src/training/steps/market_analysis/economic_regime_feature_selector.py.deprecated`
**Original:** `economic_regime_feature_selector.py`  
**Reason:** 
- Circular dependency on regime labels
- Optimization mismatch (optimized for return prediction, not regime separation)
- Lacks unsupervised mode for pre-clustering use
- Used economic targets (75% volatility-based) as proxy

**Replacement:** Use `EnhancedRegimeFeatureSelector` from `regime_feature_selector.py`

### 2. `test_treeshap_simple.py.deprecated`
**Original:** `test_treeshap_simple.py`  
**Reason:** Tests the deprecated `EconomicRegimeFeatureSelector`

**Replacement:** Use `test_regime_feature_selection_fix.py`

### 3. `test_treeshap_integration.py.deprecated`
**Original:** `test_treeshap_integration.py`  
**Reason:** Tests the deprecated `EconomicRegimeFeatureSelector`

**Replacement:** Use `test_regime_feature_selection_fix.py`

## Migration Guide

### Old Code (Deprecated):
```python
from src.training.steps.market_analysis.economic_regime_feature_selector import EconomicRegimeFeatureSelector

selector = EconomicRegimeFeatureSelector()
result = selector.select_features(features_df, labels_df)  # Required labels!
```

### New Code (Recommended):
```python
from src.training.steps.market_analysis.regime_feature_selector import EnhancedRegimeFeatureSelector

selector = EnhancedRegimeFeatureSelector()

# Unsupervised mode (for pre-clustering) - NO regime labels needed
result = selector.select_features(
    features_df=features_df,
    regime_labels=None,
    use_supervised=False
)

# OR Supervised mode (for post-clustering refinement) - WITH regime labels
result = selector.select_features(
    features_df=features_df,
    regime_labels=discovered_regimes,
    use_supervised=True
)
```

## Why These Changes?

1. **No Circular Dependency:** Feature selection now works BEFORE clustering, not after
2. **Better Features:** Optimized for regime separation using `regime_feature_categorization`
3. **Flexibility:** Both unsupervised (pre-clustering) and supervised (post-clustering) modes
4. **Domain Knowledge:** Integrates regime-specific feature prioritization

## What to Do

### If you have references to deprecated code:
1. Update imports to use `EnhancedRegimeFeatureSelector`
2. Change to unsupervised mode for pre-clustering feature selection
3. Remove `regime_labels` parameter (unless doing post-clustering refinement)

### If you need help:
- See: `REGIME_FEATURE_SELECTION_FIX_SUMMARY.md` for detailed migration guide
- See: `REGIME_FEATURE_SELECTION_ANALYSIS.md` for problem explanation
- See: `test_regime_feature_selection_fix.py` for usage examples

## Files Kept for Reference

The deprecated files are kept with `.deprecated` extension for reference purposes only.

**DO NOT USE DEPRECATED FILES IN PRODUCTION**

---

**Deprecated:** 2025-10-28  
**Reason:** Circular dependency and optimization mismatch fixes
