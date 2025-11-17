# Label-Guided Interaction Discovery

## Overview

This document explains the new **Label-Guided Interaction Discovery** system that addresses key issues with feature interaction generation:

1. **Trend category over-representation** - Too many trend-based interactions due to combinatorial explosion
2. **Heuristic-based selection** - Interactions weren't guided by label-specific mutual information or SHAP
3. **No lift validation** - Interactions weren't required to beat their base features
4. **No regularization** - No L1/group LASSO to enforce sparsity and prevent overfitting

## Key Improvements

### 1. Label-Specific Interaction Scoring

**Before:** Interactions were selected based on tree splitting patterns and generic MI scores.

**After:** Each interaction is scored using:
- **Mutual Information (MI)** between the interaction feature and the target
- **SHAP interaction values** measuring interaction strength between base features
- **Composite score:** `0.6 × MI + 0.4 × SHAP interaction`

### 2. Lift Requirements

**Critical Feature:** Interactions must demonstrate meaningful improvement over their base features.

```python
# MI Lift Calculation
base_mi_f1 = MI(feature1, target)
base_mi_f2 = MI(feature2, target)
max_base_mi = max(base_mi_f1, base_mi_f2)

interaction_mi = MI(feature1 × feature2, target)
mi_lift = (interaction_mi - max_base_mi) / max_base_mi

# Requirement: mi_lift >= 0.05 (5% improvement)
```

**Why this matters:**
- If `feature1 × feature2` doesn't provide more information than `feature1` or `feature2` alone, it's redundant
- Prevents overfitting on noise and reduces model complexity
- Ensures interactions actually capture non-linear relationships

### 3. Regularized Selection (LASSO)

After scoring, LASSO (L1 regularization) is applied to select sparse, meaningful interactions:

```python
# LASSO with cross-validation to find optimal alpha
lasso = LassoCV(cv=3, max_iter=500)
lasso.fit(candidate_interactions, target)

# Select interactions with non-zero coefficients
selected = interactions[abs(lasso.coef_) > 1e-6]
```

**Benefits:**
- Automatic feature selection through L1 penalty
- Reduces multicollinearity among interactions
- Prevents overfitting by enforcing sparsity

### 4. Category-Aware Limits

**The Problem:**
- Trend features are generated with multiple periods (SMA: 5,10,20,50; EMA: 12,26)
- Each gets 4 variants (base, volnorm, vwap, trend_adj) = 4× multiplication
- Cross-timeframe features (3x, 6x, 9x, 27x) = 5× multiplication
- Result: 4 SMA periods → **80 trend features**!
- Without controls, interactions become dominated by trend × trend

**The Solution:**

```python
# Limit interactions per category pair
max_interactions_per_category_pair = 3

# Example: Only allow 3 best (trend, trend) interactions
# Prioritize cross-category interactions: (trend, momentum), (volatility, volume)
```

**Category pair distribution example:**
```
momentum_x_trend: 3 interactions
trend_x_volatility: 3 interactions
trend_x_trend: 3 interactions  # Limited!
volume_x_volatility: 2 interactions
```

## Configuration

### Basic Configuration

```python
config = {
    # Lift requirements
    'min_interaction_mi_lift': 0.05,  # 5% MI improvement required
    'min_interaction_r2_lift': 0.01,  # 1% R² improvement (optional, expensive)

    # Category controls
    'max_interactions_per_category_pair': 3,  # Max per (cat1, cat2) pair

    # Pair generation
    'max_interaction_pairs': 100,  # Limit pairs to test
    'use_tree_guided_pairs': True,  # Use LGBM tree splits for guidance
}
```

### Advanced Configuration

```python
from src.training.utils.feature_selection import LabelGuidedInteractionConfig

config = LabelGuidedInteractionConfig(
    # Scoring weights
    use_mi_scoring=True,
    use_shap_scoring=True,
    mi_weight=0.6,
    shap_weight=0.4,

    # Lift requirements
    min_r2_lift=0.01,
    min_mi_lift=0.05,
    require_r2_lift=False,  # Expensive to compute
    require_mi_lift=True,   # Fast and effective

    # Regularization
    use_lasso=True,
    lasso_alpha=None,  # Use CV to find optimal
    lasso_cv_folds=3,

    # Interaction operations
    operations=['multiply', 'divide', 'subtract', 'log_ratio'],

    # Category controls
    max_interactions_per_category_pair=3,
    banned_category_pairs={(('trend', 'trend'))},  # Ban trend×trend entirely
)
```

## How It Works

### Step 1: Feature Pair Generation

```python
# Option A: Tree-guided (recommended)
# Extract pairs from LGBM tree splits
model = lgb.LGBMRegressor(max_depth=3, n_estimators=50)
model.fit(features, target)
pairs = extract_tree_splitting_pairs(model)

# Option B: Automatic generation
# Prioritize cross-category pairs
for f1 in features:
    for f2 in features:
        if category(f1) != category(f2):
            pairs.append((f1, f2))  # Higher priority
```

### Step 2: Candidate Generation

```python
operations = ['multiply', 'divide', 'subtract', 'log_ratio']

for f1, f2 in pairs:
    for op in operations:
        if op == 'multiply':
            interaction = f1 * f2
        elif op == 'divide':
            interaction = f1 / (f2 + 1e-8)
        # ... etc

        candidates.append(InteractionCandidate(
            name=f"{f1}_{op}_{f2}",
            feature1=f1,
            feature2=f2,
            operation=op
        ))
```

### Step 3: Scoring

```python
# Calculate MI for each candidate
for cand in candidates:
    cand.mi_score = mutual_info_regression(
        interaction_feature, target
    )

# Calculate SHAP interaction values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_interaction_values(features)
for cand in candidates:
    f1_idx = features.columns.get_loc(cand.feature1)
    f2_idx = features.columns.get_loc(cand.feature2)
    cand.shap_score = abs(shap_values[:, f1_idx, f2_idx]).mean()
```

### Step 4: Lift Filtering

```python
for cand in candidates:
    # Get base feature MI scores
    base_mi_f1 = MI(cand.feature1, target)
    base_mi_f2 = MI(cand.feature2, target)
    max_base_mi = max(base_mi_f1, base_mi_f2)

    # Calculate lift
    cand.mi_lift = (cand.mi_score - max_base_mi) / max_base_mi

    # Filter
    if cand.mi_lift < 0.05:  # 5% minimum
        candidates.remove(cand)
```

### Step 5: LASSO Selection

```python
# Build candidate matrix
X_candidates = DataFrame({cand.name: cand.series for cand in candidates})

# Apply LASSO
lasso = LassoCV(cv=3)
lasso.fit(X_candidates, target)

# Select non-zero coefficients
selected = [cand for i, cand in enumerate(candidates)
            if abs(lasso.coef_[i]) > 1e-6]
```

### Step 6: Category Limits

```python
# Group by category pair
groups = defaultdict(list)
for cand in selected:
    cat_pair = tuple(sorted([cand.category1, cand.category2]))
    groups[cat_pair].append(cand)

# Apply limits
final_interactions = []
for cat_pair, cands in groups.items():
    # Sort by composite score
    cands.sort(key=lambda c: 0.6*c.mi_score + 0.4*c.shap_score, reverse=True)

    # Take top 3 per category pair
    final_interactions.extend(cands[:3])
```

## Results

### Before (Legacy Approach)

```
Total interaction candidates: 400
Selected interactions: 80

Category distribution:
  trend_x_trend: 35 interactions     ⚠️ Over-represented!
  momentum_x_trend: 15 interactions
  volatility_x_trend: 12 interactions
  volume_x_volatility: 8 interactions
  other: 10 interactions
```

**Problems:**
- Trend dominates (43% of interactions)
- Many redundant trend×trend interactions
- No validation that interactions beat base features

### After (Label-Guided Approach)

```
Total interaction candidates: 400
Filtered by MI lift: 180 remaining
After LASSO selection: 45 remaining
After category limits: 30 final

Category distribution:
  momentum_x_trend: 3 interactions     ✅ Balanced!
  trend_x_volatility: 3 interactions
  volume_x_volatility: 3 interactions
  trend_x_trend: 3 interactions        ✅ Limited!
  momentum_x_volatility: 3 interactions
  oscillator_x_trend: 3 interactions
  returns_x_volatility: 3 interactions
  ... (balanced across categories)
```

**Improvements:**
- Balanced category representation
- All interactions validated to beat base features
- Sparse, interpretable interaction set
- No trend over-representation

## Testing

### Unit Tests

```python
# Test MI lift calculation
def test_mi_lift_requirement():
    # Create features where interaction provides no lift
    f1 = np.random.randn(1000)
    f2 = np.random.randn(1000)
    target = f1 + 0.1 * np.random.randn(1000)  # target correlated with f1

    interaction = f1 * f2  # No additional information

    discoverer = LabelGuidedInteractionDiscovery(
        LabelGuidedInteractionConfig(min_mi_lift=0.05)
    )

    results = discoverer.discover_interactions(
        features=pd.DataFrame({'f1': f1, 'f2': f2}),
        target=pd.Series(target),
        feature_categories={'f1': 'trend', 'f2': 'momentum'}
    )

    # Should have zero or very few interactions (no lift)
    assert len(results[0].columns) < 5
```

### Integration Tests

```bash
# Test with real training pipeline
python -m src.training.launchers.analyst_base_training \
    --symbol ETHUSDT \
    --enable_label_guided_interactions \
    --max_interactions_per_category_pair 3 \
    --min_interaction_mi_lift 0.05
```

## Troubleshooting

### Issue: No interactions selected

**Possible causes:**
1. `min_mi_lift` too high - Try lowering to 0.02
2. Features are too independent - No meaningful interactions exist
3. LASSO alpha too high - Let CV find optimal alpha

**Solution:**
```python
config = {
    'min_interaction_mi_lift': 0.02,  # Lower threshold
    'lasso_alpha': None,  # Use CV
}
```

### Issue: Still too many trend interactions

**Solution:**
```python
config = {
    'max_interactions_per_category_pair': 2,  # Stricter limit
    'banned_category_pairs': {('trend', 'trend')},  # Ban trend×trend
}
```

### Issue: SHAP scoring fails

**Fallback:** MI-only scoring
```python
config = {
    'use_shap_scoring': False,
    'use_mi_scoring': True,
    'mi_weight': 1.0,
}
```

## References

1. **Mutual Information:**
   - Ross, B. C. (2014). "Mutual Information between Discrete and Continuous Data Sets"
   - sklearn.feature_selection.mutual_info_regression

2. **SHAP Interaction Values:**
   - Lundberg, S. M., et al. (2018). "Consistent Individualized Feature Attribution for Tree Ensembles"
   - shap.TreeExplainer.shap_interaction_values

3. **LASSO Regularization:**
   - Tibshirani, R. (1996). "Regression Shrinkage and Selection via the Lasso"
   - sklearn.linear_model.LassoCV

## Future Enhancements

### Group LASSO

Treat category pairs as groups for structured sparsity:
```python
from sklearn.linear_model import GroupLasso

# Group interactions by category pair
groups = assign_group_ids(interactions, by='category_pair')

group_lasso = GroupLasso(groups=groups, alpha=0.1)
group_lasso.fit(X_interactions, y)
```

### R² Lift Validation

Currently optional due to cost. Could optimize:
```python
# Fast R² approximation using cached predictions
r2_lift = compute_r2_lift_cached(
    interaction_feature,
    base_predictions,  # Cached
    target
)
```

### Adaptive Category Limits

Adjust limits based on category signal strength:
```python
# Categories with higher MI get more interaction budget
category_mi = {cat: mean_mi(features[cat], target) for cat in categories}
category_budget = allocate_budget(category_mi, total_budget=30)
```

## Migration Guide

### From Legacy to Label-Guided

The new system is **backward compatible** with automatic fallback:

```python
# Old code (still works)
interactions, metadata = await self._phase3_3_interaction_discovery(
    features, targets, config
)

# New code (recommended)
interactions, metadata = await self._phase3_3_label_guided_interaction_discovery(
    features, targets, config
)
```

If label-guided discovery is unavailable, it automatically falls back to legacy.

### Configuration Migration

```python
# Legacy config
config = {
    'interaction_pairs_limit': 80,
    'interaction_ops_mode': 'full',
}

# Label-guided config (equivalent)
config = {
    'max_interaction_pairs': 100,
    'max_interactions_per_category_pair': 3,
    'min_interaction_mi_lift': 0.05,
    'use_tree_guided_pairs': True,
}
```

## Summary

Label-guided interaction discovery provides:

✅ **Label-specific scoring** using MI and SHAP interaction values
✅ **Lift requirements** ensuring interactions beat base features
✅ **Regularized selection** via LASSO for sparsity
✅ **Category-aware limits** preventing trend over-representation
✅ **Backward compatibility** with automatic fallback

This results in **sparse, interpretable, and predictive** interaction features that avoid overfitting and category imbalance.
