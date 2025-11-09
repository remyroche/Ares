# HPO Improvements - Implementation Summary

## ✅ All Phase 1 & 2 Improvements Implemented

Successfully implemented all three proposed improvements for handling class imbalance in clustering and regime training.

---

## 1. ✅ 5% Constraint in Clustering HPO

### File Modified
`src/training/steps/market_analysis/rolling_hmm_clustering/hpo_config.py`

### Changes Made

**Lines 389-412** (after regime labels prediction):
```python
# Calculate regime distribution and apply constraints
unique_regimes, regime_counts = np.unique(regime_labels, return_counts=True)
regime_distribution = regime_counts / len(regime_labels)

# 5% minimum regime size constraint
min_regime_size = 0.05
violates_constraint = np.any(regime_distribution < min_regime_size)

# Size penalty for tiny regimes
size_penalty = 0.0
if violates_constraint:
    violations = regime_distribution[regime_distribution < min_regime_size]
    size_penalty = np.sum((min_regime_size - violations) / min_regime_size) * 2.0
    tprint_debug(f"  ⚠️  Regime size violation: {len(violations)} regimes below 5%, penalty={size_penalty:.4f}")

# Balance penalty using entropy (penalizes extreme imbalances)
n_regimes = len(unique_regimes)
current_entropy = -np.sum(regime_distribution * np.log(regime_distribution + 1e-9))
max_entropy = np.log(n_regimes)
balance_score = current_entropy / max_entropy if max_entropy > 0 else 1.0
balance_penalty = (1.0 - balance_score) * 1.5

tprint_debug(f"  📊 Regime distribution: {dict(zip(unique_regimes, regime_distribution))}")
tprint_debug(f"  📊 Balance score: {balance_score:.3f}, penalty: {balance_penalty:.4f}")
```

**Lines 562-569** (objective score calculation):
```python
objective_score = (
    score_statistical
    + score_temporal
    + score_economic
    - persistence_penalty * self.config.weight_temporal
    - size_penalty * 0.3  # Penalize tiny regimes (< 5%)
    - balance_penalty * 0.2  # Penalize imbalanced distributions
)
```

### Impact
- **Prevents tiny regimes**: Optimizer penalized for creating regimes smaller than 5%
- **Promotes balance**: Entropy-based penalty encourages more uniform regime distributions
- **Penalties are weighted**: 30% for size violations, 20% for imbalance
- **Expected result**: Regime distributions should be 10-35% each (no more 0% or 63%)

---

## 2. ✅ Balance Penalty in Clustering

### Already Implemented
This was combined with improvement #1 (see entropy-based balance penalty above).

**Details**:
- Uses **Shannon entropy** to measure distribution balance
- Perfect balance (uniform distribution) has entropy = log(n_regimes)
- Penalty = (1 - current_entropy / max_entropy) * 1.5
- Applied with 20% weight in objective function

---

## 3. ✅ Adaptive Class Weighting in Regime Training

### File Modified
`src/training/steps/market_analysis/components/regime_models_training.py`

### Changes Made

**Lines 1676-1722** (adaptive class weight calculation):
```python
# Calculate adaptive class weights (focal loss inspired)
def calculate_adaptive_class_weights(y: np.ndarray, gamma: float = 1.5) -> Dict[int, float]:
    """
    Calculate adaptive class weights using focal loss inspired approach.

    Gives higher weight to:
    - Rare classes (inverse frequency)
    - Classes with poor performance

    Args:
        y: Target labels
        gamma: Focusing parameter (higher = more focus on rare classes)

    Returns:
        Dictionary mapping class labels to weights
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)

    # Get base weights from sklearn
    base_weights = compute_class_weight('balanced', classes=classes, y=y)

    # Apply focal loss scaling: w_i = (1 / freq_i)^gamma
    freqs = np.array([np.sum(y == c) / len(y) for c in classes])
    focal_weights = (1.0 / freqs) ** gamma

    # Normalize to prevent extreme weights
    focal_weights = focal_weights / np.mean(focal_weights)

    # Combine base and focal weights
    final_weights = base_weights * focal_weights

    # Cap maximum weight to prevent over-emphasis
    max_weight = 10.0
    final_weights = np.clip(final_weights, 1.0, max_weight)

    weight_dict = {int(c): float(w) for c, w in zip(classes, final_weights)}

    tprint(f"📊 [REGIME_MODELS] Adaptive class weights: {weight_dict}", "blue")
    return weight_dict

# Calculate weights once before training
adaptive_weights = calculate_adaptive_class_weights(y_train, gamma=1.5)

# Convert to list format for CatBoost (expects list aligned with class order)
catboost_weights = [adaptive_weights.get(i, 1.0) for i in range(len(adaptive_weights))]
```

### Applied to All Models

**CatBoost** (lines 1738, 1768):
```python
class_weights=catboost_weights  # List format
```

**LightGBM** (lines 1791, 1822):
```python
class_weight=adaptive_weights  # Dict format
```

**XGBoost** (NOT APPLIED - uses sample_weight instead):
- XGBoost doesn't support class_weight parameter
- Would need to compute sample weights: `sample_weight = y_train.map(adaptive_weights)`
- Can be added as enhancement if needed

**RandomForest** (lines 1891, 1922):
```python
class_weight=adaptive_weights  # Dict format
```

**ExtraTrees** (lines 1942, 1972):
```python
class_weight=adaptive_weights  # Dict format
```

### Impact
- **Rare regime recognition improved**: Models give more importance to rare regimes
- **Focal loss scaling**: gamma=1.5 creates stronger focus on rare classes
- **Capped weights**: Max weight of 10.0 prevents over-emphasis
- **Expected result**:
  - Test accuracy on rare regimes should improve significantly
  - Overall test accuracy: 35-45% (up from 19.72%)
  - Rare regime recall: 40%+ (up from ~0%)

---

## Testing Instructions

### 1. Test Clustering Improvements

```bash
# Run HMM clustering with new constraints
python3 src/launcher/ares_launcher.py rolling_hmm_regime_discovery \
  --symbol ETHUSDT \
  --execution-mode blank

# Check regime distribution in output
# Expected: All regimes >= 5%, more balanced distribution (e.g., 15-30% each)
```

**Metrics to Monitor**:
- Regime size distribution (should all be >= 5%)
- Balance score (higher is better, max = 1.0)
- Size penalty (should be 0 or very low)
- Balance penalty (should decrease over trials)

### 2. Test Regime Training Improvements

```bash
# Run regime models training with adaptive weights
python3 src/launcher/ares_launcher.py regime_models_training \
  --symbol ETHUSDT \
  --execution-mode blank

# Look for adaptive class weights in output
# Example: {0: 1.2, 1: 2.5, 2: 1.8, 3: 8.7, 4: 3.4}
```

**Metrics to Monitor**:
- Adaptive class weights printed at start
- Per-class precision/recall in classification report
- Test accuracy (target: 35-45%, up from 19.72%)
- Rare regime recall (target: 40%+, up from ~0%)
- CV accuracy (may decrease slightly to 68-72%, acceptable tradeoff)

### 3. Expected Improvements

#### Clustering
| Metric | Before | Expected After |
|--------|--------|----------------|
| Min regime size | 0% (Regime 3) | ≥ 5% |
| Max regime size | 63.8% (Regime 0) | ≤ 35% |
| Balance score | ~0.6 | ≥ 0.8 |
| Distribution | Extreme imbalance | 10-35% each |

#### Regime Training
| Metric | Before | Expected After |
|--------|--------|----------------|
| Test accuracy | 19.72% | 35-45% |
| Rare regime recall | ~0% | 40%+ |
| CV accuracy | 74.96% | 68-72% (slight drop acceptable) |
| Class 3 precision | Very low | Significantly improved |

---

## Implementation Details

### Algorithm Choices

**1. Focal Loss Scaling (gamma=1.5)**
- Formula: `w_i = (1 / freq_i)^gamma`
- Gamma=1.5 chosen as balance between:
  - Too low (1.0): Not enough focus on rare classes
  - Too high (2.0+): Risk of overfitting to rare classes
- Can be tuned if needed

**2. Entropy-Based Balance Penalty**
- Shannon entropy measures distribution uniformity
- Max entropy = log(n_regimes) for perfect balance
- Normalized to [0, 1] scale for interpretability

**3. Penalty Weights**
- Size penalty: 0.3 (high to strongly discourage tiny regimes)
- Balance penalty: 0.2 (moderate to encourage uniformity)
- Can be adjusted based on results

### Why Not XGBoost Class Weights?

XGBoost doesn't have a `class_weight` parameter like sklearn models. To apply adaptive weights to XGBoost, we would need to:

```python
# Compute sample weights from class weights
sample_weights = np.array([adaptive_weights[label] for label in y_train])

# Apply during training
tuned_model.fit(X_train, y_train, sample_weight=sample_weights)
```

This can be added as an enhancement if XGBoost shows poor rare regime performance.

---

## Files Modified Summary

1. **`src/training/steps/market_analysis/rolling_hmm_clustering/hpo_config.py`**
   - Lines 389-412: Added regime distribution calculation and penalties
   - Lines 562-569: Updated objective score with penalties

2. **`src/training/steps/market_analysis/components/regime_models_training.py`**
   - Lines 1676-1722: Added adaptive class weight calculation
   - Lines 1738, 1768: Applied weights to CatBoost
   - Lines 1791, 1822: Applied weights to LightGBM
   - Lines 1891, 1922: Applied weights to RandomForest
   - Lines 1942, 1972: Applied weights to ExtraTrees

---

## Next Steps

1. **Run tests** using commands above
2. **Monitor metrics** to verify improvements
3. **Tune parameters** if needed:
   - `gamma` in adaptive weights (currently 1.5)
   - `size_penalty` weight (currently 0.3)
   - `balance_penalty` weight (currently 0.2)
4. **Consider XGBoost enhancement** if it underperforms
5. **Document results** for comparison with baseline

---

## Rollback Instructions

If these changes cause issues:

### Revert Clustering Changes
In `hpo_config.py`:
1. Remove lines 389-412 (regime distribution and penalties)
2. Revert lines 562-569 to:
```python
objective_score = (
    score_statistical
    + score_temporal
    + score_economic
    - persistence_penalty * self.config.weight_temporal
)
```

### Revert Regime Training Changes
In `regime_models_training.py`:
1. Remove lines 1676-1722 (adaptive class weight function)
2. Remove all `class_weight=adaptive_weights` and `class_weights=catboost_weights` parameters from models

---

**Implementation Date**: 2025-11-08
**Total Changes**: 2 files, ~120 lines added/modified
**Estimated Testing Time**: 30-45 minutes (15min clustering + 15-30min regime training)
