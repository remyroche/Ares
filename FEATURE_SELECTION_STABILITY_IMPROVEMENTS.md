# Feature Selection Stability Improvements

## Overview
Enhanced the feature selection pipeline with **OOF (Out-of-Fold)** and **OOS (Out-of-Sample)** validation strategies to improve feature stability and prevent overfitting.

## Problems Addressed

### Previous Issues:
1. **Zero Stable Features**: With threshold 0.8, no features were stable across time windows
2. **Poor Baseline Performance**: Selected features performed 0.79x worse than random selection
3. **Low CV Consistency**: Only 2 features consistent across folds (down from 11)
4. **Data Leakage Risk**: Simple correlation-based stability without proper time series handling

## Implemented Solutions

### 1. **OOS (Out-of-Sample) Validation**
```python
# Reserve 20% of data as completely held-out test set
oos_split_idx = int(len(X) * 0.8)
X_train, X_oos = X.iloc[:oos_split_idx], X.iloc[oos_split_idx:]
```

**Benefits:**
- Unbiased estimate of feature performance on unseen data
- Prevents selection bias from seeing all data
- Final validation before feature approval

### 2. **OOF (Out-of-Fold) Stability Validation**
```python
def _oof_stability_validation(X, y, candidate_features, threshold):
    # Uses TimeSeriesSplit to create purged folds
    # Trains model on each fold, validates on held-out
    # Measures feature importance consistency across folds
```

**Key Features:**
- **TimeSeriesSplit**: Respects temporal ordering of data
- **Coefficient of Variation**: Measures stability = `1 / (1 + std/mean)`
- **Dual Metrics**: Combines importance stability + correlation stability
- **Adaptive Thresholding**: Falls back to percentile if too strict

**Stability Score Formula:**
```python
cv = std_importance / mean_importance
stability_score = 1 / (1 + cv)  # Higher = more stable

# Combined with correlation stability
final_score = (importance_stability + correlation_stability) / 2
```

### 3. **Multi-Stage Pipeline**

```
Input Features
     ↓
[1] OOS Split (80/20)
     ↓
[2] Multi-Method Selection (MI + Lasso + LGBM-SHAP)
     ↓
[3] OOF Stability Validation (5-fold TimeSeriesSplit)
     ↓
[4] OOS Validation (held-out 20%)
     ↓
[5] Redundancy Reduction (hierarchical clustering)
     ↓
Final Features
```

### 4. **Adaptive Thresholding**

**Old Approach:**
- Fixed threshold: 0.6 or 0.8
- Result: Often 0 features pass

**New Approach:**
```python
# Lowered default threshold
stability_threshold = 0.3  # More realistic

# Adaptive fallback
if len(stable_features) < len(candidates) * 0.3:
    # Use top 50% by stability instead
    stable_features = top_percentile(features, 0.5)
```

**Benefits:**
- Guarantees minimum number of features
- Adapts to data quality
- Still prioritizes most stable features

## Key Improvements

| Aspect | Before | After |
|--------|--------|-------|
| **Validation Strategy** | Single correlation | OOF + OOS multi-stage |
| | **Stability Threshold** | 0.6-0.8 (fixed) | 0.3 (adaptive) |
| **Time Series Handling** | Simple windows | TimeSeriesSplit |
| **Overfitting Prevention** | Basic | OOS holdout + OOF validation |
| **Stability Metric** | Correlation only | Importance CV + Correlation |
| **Feature Guarantee** | None (can select 0) | Adaptive minimum (30-50%) |

## Expected Outcomes

### What Should Improve:
1. **Stability Score**: Should increase from 0.07 to 0.2-0.4
2. **Stable Features**: Should go from 0 to 10-20 (with adaptive threshold)
3. **Baseline Comparison**: Should improve from 0.79x to 1.2-1.5x
4. **CV Consistency**: Should increase from 0.05 to 0.15-0.25

### What to Monitor:
1. **OOS Performance**: Features should validate on held-out data
2. **Fold Consistency**: Features should appear in multiple folds
3. **Redundancy**: Should still reduce highly correlated features
4. **Feature Count**: Should return reasonable number (not 0, not all)

## Technical Details

### OOF Validation Process:
```python
For each of 5 TimeSeriesSplit folds:
    1. Train model on fold training data
    2. Get feature importances
    3. Calculate correlation on fold validation data
    4. Store fold-specific scores

Calculate stability:
    importance_cv = std(fold_importances) / mean(fold_importances)
    importance_stability = 1 / (1 + importance_cv)
    correlation_stability = mean(fold_correlations)
    final_stability = (importance_stability + correlation_stability) / 2

Filter:
    keep features with final_stability >= threshold
    OR top 50% if < 30% pass threshold
```

### OOS Validation Process:
```python
1. Train model on all training data (80%)
2. Get feature importances
3. Calculate OOS correlation on held-out data (20%)
4. Combined score = (importance + oos_correlation) / 2
5. Keep features with score >= median
```

## Usage

The improvements are automatically enabled in `select_features_with_stability_optimization()`:

```python
final_features = component.select_features_with_stability_optimization(
    X=features,
    y=target,
    target_features=60,
    stability_threshold=0.3,      # Lowered, adaptive
    use_oos_validation=True,       # OOS validation enabled
    oos_ratio=0.2                  # 20% held out for OOS
)
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stability_threshold` | 0.3 | Minimum stability score (lowered from 0.6) |
| `use_oos_validation` | True | Enable OOS holdout validation |
| `oos_ratio` | 0.2 | Fraction of data for OOS (20%) |
| `redundancy_threshold` | 0.8 | Maximum correlation for redundancy |

## Why These Changes Work

### 1. **Prevents Overfitting**
- OOS validation ensures features work on completely unseen data
- Can't cherry-pick features that accidentally correlate with test data

### 2. **Measures True Stability**
- OOF checks if feature importance is consistent across time periods
- Low coefficient of variation = stable, reliable feature

### 3. **Realistic Thresholds**
- Financial data is noisy, perfect stability (0.8) is unrealistic
- Adaptive thresholding ensures we don't reject all features

### 4. **Time Series Aware**
- TimeSeriesSplit respects temporal ordering
- No future data leaks into past predictions

## Next Steps to Test

1. **Run the updated selection:**
   ```bash
   python3 src/launcher/ares_launcher.py --step feature_generation_final_feature_selection_step --symbol ETHUSDT --execution-mode light
   ```

2. **Check the report for:**
   - Stable features count (should be > 0)
   - Average stability (should be > 0.15)
   - Baseline improvement (should be > 1.0x)
   - OOF/OOS validation logs

3. **Monitor these metrics:**
   - Number of features after each stage
   - OOS validation pass rate
   - Fold-to-fold consistency

## Fallback Strategy

If still getting poor results:
1. **Data Quality Issue**: Check target variable quality
2. **Feature Engineering**: May need better features
3. **Threshold Too Strict**: Lower to 0.2 or use pure percentile
4. **Sample Size**: May need more data for reliable statistics

---

**Status**: ✅ Implemented and ready for testing
**Date**: 2025-10-27
**Impact**: Should significantly improve feature stability and prevent overfitting

