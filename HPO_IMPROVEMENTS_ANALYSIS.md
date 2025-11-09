# HPO Improvements Analysis - Clustering & Regime Training

## Overview
Analysis of 4 proposed improvements for handling class imbalance in both Rolling HMM Clustering and Regime Models Training.

---

## Current State

### Rolling HMM Clustering (hpo_config.py)
**Objective Function** (lines 320-551):
- **Weights**: 40% statistical (CV ratio + silhouette), 20% temporal, 40% economic
- **No explicit regime size constraints**
- **Penalties**: Persistence penalty for high diagonal in transition matrix (lines 526-541)
- **Class handling**: No specific handling for tiny regimes

**Problem Observed**:
- Regime 0: 63.8% of training data → 0% of test data
- Regime 3: 3.3% of training data → 75.1% of test data
- Extreme distribution shifts between train/test

### Regime Models Training (regime_models_training.py)
**Current Approach** (lines 1676-1914):
- Uses `class_weight='balanced'` for LightGBM meta-learner (line 536, 560)
- No custom class weighting in base models (CatBoost, XGBoost, RandomForest, ExtraTrees)
- No focal loss implementation
- No balance penalty in HPO objective

**Problem Observed**:
- CV accuracy: 74.96% (overfits to dominant regimes)
- Test accuracy: 19.72% (fails on rare regimes)
- Model learns to predict Regime 0 (63.8% of train) but fails when it disappears in test

---

## Proposed Improvements

### 1. Add 5% Constraint to HPO Objective Function

#### For Clustering (hpo_config.py)

**Implementation Location**: Lines 335-543 (objective function)

**Code Addition** (after line 387, before quality assessment):
```python
# Calculate regime distribution
unique, counts = np.unique(regime_labels, return_counts=True)
regime_distribution = counts / len(regime_labels)

# Check 5% minimum regime size constraint
min_regime_size = 0.05
violates_constraint = np.any(regime_distribution < min_regime_size)

# Penalize tiny regimes
size_penalty = 0.0
if violates_constraint:
    # Calculate severity of violation
    violations = regime_distribution[regime_distribution < min_regime_size]
    size_penalty = np.sum((min_regime_size - violations) / min_regime_size) * 2.0
    tprint_debug(f"  Regime size violation: {len(violations)} regimes below 5%, penalty={size_penalty:.4f}")
```

**Integration** (line 537-542):
```python
objective_score = (
    score_statistical
    + score_temporal
    + score_economic
    - persistence_penalty * self.config.weight_temporal
    - size_penalty * 0.3  # NEW: Add size penalty with 30% weight
)
```

**Impact**:
- ✅ **Prevents generation of tiny regimes**: Forces optimizer away from extreme imbalances
- ✅ **Immediate improvement**: Works on first optimization run
- ✅ **Low complexity**: Simple to implement and debug
- ⚠️ **May reduce statistical quality**: Could lower CV ratio if balanced regimes have less separation
- ⚠️ **Rigid constraint**: 5% might be too strict for some market conditions

**Recommendation**: **IMPLEMENT** - Highest priority, easiest to add, directly addresses root cause

---

### 2. Implement Adaptive Class Weighting

#### For Regime Training (regime_models_training.py)

**Implementation Location**: Lines 1656-1914 (_train_models_with_hpo method)

**Current**:
```python
# Only meta-learner uses class_weight='balanced'
# Base models (CatBoost, XGBoost, RF, ET) use default weights
```

**Proposed Change** (before line 1678):
```python
def calculate_adaptive_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """
    Calculate adaptive class weights using focal loss inspired approach.

    Gives higher weight to:
    - Rare classes (inverse frequency)
    - Classes with poor performance (based on baseline model)

    Returns:
        Dictionary mapping class labels to weights
    """
    from sklearn.utils.class_weight import compute_class_weight

    # Get base weights from sklearn
    classes = np.unique(y_train)
    base_weights = compute_class_weight('balanced', classes=classes, y=y_train)

    # Apply focal loss scaling: w_i = (1 / freq_i)^gamma where gamma controls focus
    gamma = 1.5  # Higher gamma = more focus on rare classes
    freqs = np.array([np.sum(y_train == c) / len(y_train) for c in classes])
    focal_weights = (1.0 / freqs) ** gamma

    # Normalize to prevent extreme weights
    focal_weights = focal_weights / np.mean(focal_weights)

    # Combine base and focal weights
    final_weights = base_weights * focal_weights

    # Cap maximum weight to prevent over-emphasis
    max_weight = 10.0
    final_weights = np.clip(final_weights, 1.0, max_weight)

    weight_dict = {int(c): float(w) for c, w in zip(classes, final_weights)}

    tprint(f"📊 Adaptive class weights: {weight_dict}", "blue")
    return weight_dict

# Calculate weights once before training
adaptive_weights = calculate_adaptive_class_weights(y_train)
```

**Apply to Models**:

**CatBoost** (line 1683-1722):
```python
# OLD:
return cb.CatBoostClassifier(
    iterations=params.get('iterations', 100),
    ...
    random_seed=42,
    verbose=False
)

# NEW:
return cb.CatBoostClassifier(
    iterations=params.get('iterations', 100),
    ...
    class_weights=list(adaptive_weights.values()),  # CatBoost uses list format
    random_seed=42,
    verbose=False
)
```

**LightGBM** (line 1724-1771):
```python
# Already has class_weight='balanced' in config (lines 536, 560)
# Update to use adaptive weights:
return lgb.LGBMClassifier(
    ...
    class_weight=adaptive_weights,  # Dict format
    random_state=42,
    verbose=-1
)
```

**XGBoost** (line 1773-1820):
```python
# Calculate scale_pos_weight for binary/multiclass
# XGBoost uses different approach - compute sample_weight instead
model = xgb.XGBClassifier(...)
# Apply via fit: model.fit(X, y, sample_weight=weights_array)
```

**RandomForest & ExtraTrees** (line 1822-1911):
```python
# Both support class_weight parameter
return RandomForestClassifier(
    ...
    class_weight=adaptive_weights,
    random_state=42
)
```

**Impact**:
- ✅ **Immediate improvement**: Should boost rare regime recognition
- ✅ **Standard technique**: Well-established in ML literature
- ✅ **Flexible**: Can tune gamma parameter for more/less aggressive weighting
- ✅ **Works with existing code**: Drop-in replacement for current approach
- ⚠️ **May overfit rare classes**: Could sacrifice dominant regime performance
- ⚠️ **Requires tuning**: Gamma parameter needs optimization

**Recommendation**: **IMPLEMENT** - High priority, proven technique, addresses overfitting

---

### 3. Add Balance Penalty to Quality Scoring

#### For Clustering (hpo_config.py)

**Implementation Location**: Lines 526-542 (objective score calculation)

**Proposed Addition** (after line 535, before objective_score):
```python
# Calculate regime balance penalty
unique, counts = np.unique(regime_labels, return_counts=True)
regime_distribution = counts / len(regime_labels)

# Penalize extreme imbalances using entropy
# Perfect balance (uniform distribution) has max entropy
n_regimes = len(unique)
ideal_distribution = np.ones(n_regimes) / n_regimes
current_entropy = -np.sum(regime_distribution * np.log(regime_distribution + 1e-9))
max_entropy = np.log(n_regimes)
balance_score = current_entropy / max_entropy  # 0 = worst, 1 = perfect balance

# Convert to penalty (higher penalty for worse balance)
balance_penalty = (1.0 - balance_score) * 1.5

tprint_debug(f"  Regime balance: {regime_distribution}, entropy={current_entropy:.3f}, penalty={balance_penalty:.4f}")
```

**Integration**:
```python
objective_score = (
    score_statistical
    + score_temporal
    + score_economic
    - persistence_penalty * self.config.weight_temporal
    - balance_penalty * 0.2  # NEW: 20% weight for balance
)
```

**Alternative Approach** (Gini coefficient):
```python
# Use Gini coefficient to measure imbalance
sorted_dist = np.sort(regime_distribution)
n = len(sorted_dist)
cumsum = np.cumsum(sorted_dist)
gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
balance_penalty = gini * 2.0  # 0 = perfect balance, 2 = max imbalance
```

**Impact**:
- ✅ **Optimization respects distribution**: HPO will favor balanced solutions
- ✅ **Smooth penalty**: Continuous penalty allows gradual optimization
- ✅ **Complements size constraint**: Works with proposal #1 for comprehensive control
- ⚠️ **May conflict with quality**: Balance vs statistical separation tradeoff
- ⚠️ **Tuning required**: Need to set appropriate penalty weight (0.2 suggested)

**Recommendation**: **IMPLEMENT AFTER #1** - Use with size constraint for belt-and-suspenders approach

---

### 4. Focal Loss Implementation

#### For Regime Training (regime_models_training.py)

**Background**: Focal Loss was introduced in RetinaNet (Lin et al., 2017) for object detection with extreme class imbalance (1:1000 ratios). It down-weights easy examples and focuses on hard examples.

**Formula**: `FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)`
- `p_t`: predicted probability for true class
- `α_t`: class weight (addresses class imbalance)
- `γ`: focusing parameter (addresses easy/hard examples), typically 2.0

**Implementation Location**: Create new file `src/training/steps/market_analysis/components/focal_loss.py`

```python
import numpy as np
from typing import Optional

class FocalLoss:
    """
    Focal Loss for multiclass classification.

    Addresses class imbalance by:
    1. Down-weighting easy examples (high confidence predictions)
    2. Up-weighting hard examples (low confidence predictions)
    3. Applying class-specific weights
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[np.ndarray] = None):
        """
        Initialize focal loss.

        Args:
            gamma: Focusing parameter. Higher values focus more on hard examples.
                   gamma=0 reduces to cross-entropy. Typical: 2.0
            alpha: Class weights array of shape (n_classes,). If None, uses uniform.
        """
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Calculate focal loss.

        Args:
            y_true: True labels, shape (n_samples,)
            y_pred_proba: Predicted probabilities, shape (n_samples, n_classes)

        Returns:
            Focal loss value
        """
        n_samples, n_classes = y_pred_proba.shape

        # Clip predictions to prevent log(0)
        y_pred_proba = np.clip(y_pred_proba, 1e-7, 1 - 1e-7)

        # One-hot encode true labels
        y_true_onehot = np.eye(n_classes)[y_true]

        # Calculate focal loss
        # p_t: probability of true class
        p_t = np.sum(y_true_onehot * y_pred_proba, axis=1)

        # Focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Cross-entropy term: -log(p_t)
        ce_loss = -np.log(p_t)

        # Combine
        focal_loss = focal_weight * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[y_true]
            focal_loss = alpha_t * focal_loss

        return np.mean(focal_loss)

    def gradient(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of focal loss for gradient boosting.

        This is needed for XGBoost/LightGBM/CatBoost custom objectives.
        """
        n_samples, n_classes = y_pred_proba.shape
        y_pred_proba = np.clip(y_pred_proba, 1e-7, 1 - 1e-7)

        y_true_onehot = np.eye(n_classes)[y_true]
        p_t = np.sum(y_true_onehot * y_pred_proba, axis=1, keepdims=True)

        # Gradient computation (complex, derived from chain rule)
        grad = -y_true_onehot * (
            self.gamma * (1 - p_t) ** (self.gamma - 1) * np.log(p_t) / p_t
            + (1 - p_t) ** self.gamma / p_t
        )

        if self.alpha is not None:
            alpha_t = self.alpha[y_true][:, None]
            grad = grad * alpha_t

        return grad.flatten()

def create_focal_loss_objective(gamma: float = 2.0, alpha: Optional[np.ndarray] = None):
    """
    Create focal loss objective for gradient boosting libraries.

    Returns:
        Tuple of (loss_function, gradient_function) for custom objectives
    """
    focal_loss = FocalLoss(gamma=gamma, alpha=alpha)

    def loss_func(y_true, y_pred_proba):
        return focal_loss(y_true, y_pred_proba)

    def grad_func(y_true, y_pred_proba):
        return focal_loss.gradient(y_true, y_pred_proba)

    return loss_func, grad_func
```

**Integration in regime_models_training.py**:

```python
from src.training.steps.market_analysis.components.focal_loss import (
    FocalLoss, create_focal_loss_objective
)

# Calculate class weights (from proposal #2)
adaptive_weights = calculate_adaptive_class_weights(y_train)
alpha = np.array([adaptive_weights[i] for i in range(len(adaptive_weights))])

# Create focal loss objective
focal_loss_fn, focal_grad_fn = create_focal_loss_objective(gamma=2.0, alpha=alpha)

# For LightGBM
model = lgb.LGBMClassifier(...)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='multi_logloss',  # Keep standard metric for monitoring
    # Note: LightGBM doesn't easily support custom multiclass objectives
    # Instead, use sample_weight approach
)

# For XGBoost (easier to integrate custom objectives)
def xgb_focal_loss(preds, dtrain):
    """Custom focal loss for XGBoost."""
    labels = dtrain.get_label().astype(int)
    n_classes = len(np.unique(labels))

    # Reshape preds to (n_samples, n_classes)
    preds_proba = preds.reshape(-1, n_classes)
    preds_proba = np.exp(preds_proba) / np.exp(preds_proba).sum(axis=1, keepdims=True)

    grad = focal_grad_fn(labels, preds_proba)
    hess = np.ones_like(grad)  # Simplified hessian

    return grad, hess

model = xgb.XGBClassifier(...)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    obj=xgb_focal_loss  # Custom objective
)
```

**Impact**:
- ✅ **State-of-the-art technique**: Proven in extreme imbalance scenarios (object detection)
- ✅ **Better gradient focus**: Models learn more from hard examples (rare regimes)
- ✅ **Theoretical foundation**: Well-studied in literature
- ⚠️ **Complex implementation**: Requires custom objectives for each library
- ⚠️ **Compatibility issues**: Not all libraries easily support custom multiclass objectives
- ⚠️ **Hyperparameter tuning**: Gamma parameter needs optimization
- ⚠️ **Debugging difficulty**: Harder to debug than standard approaches

**Recommendation**: **DEFER** - Implement only if simpler approaches (#1, #2) don't work. High complexity vs benefit tradeoff.

---

## Implementation Priority & Roadmap

### Phase 1: Quick Wins (Implement First)
1. **✅ Add 5% constraint to clustering HPO** (Proposal #1)
   - Location: `hpo_config.py` lines 387-542
   - Effort: 15 minutes
   - Impact: HIGH - directly prevents tiny regimes

2. **✅ Implement adaptive class weighting** (Proposal #2)
   - Location: `regime_models_training.py` lines 1656-1914
   - Effort: 1 hour
   - Impact: HIGH - immediate rare regime improvement

### Phase 2: Reinforcement (Implement After Testing Phase 1)
3. **⚡ Add balance penalty to clustering** (Proposal #3)
   - Location: `hpo_config.py` lines 526-542
   - Effort: 30 minutes
   - Impact: MEDIUM - complements constraint from #1

### Phase 3: Advanced (Only If Needed)
4. **🔬 Focal loss implementation** (Proposal #4)
   - Location: New file + integration
   - Effort: 4-6 hours (including testing)
   - Impact: MEDIUM-HIGH - but high complexity

---

## Testing Strategy

### For Clustering (#1, #3)
```bash
# Baseline
python3 src/launcher/ares_launcher.py rolling_hmm_regime_discovery \
  --symbol ETHUSDT --execution-mode blank

# Check regime distribution in output
# Expected: No regime below 5%, more balanced distribution
```

### For Regime Training (#2, #4)
```bash
# Baseline
python3 src/launcher/ares_launcher.py regime_models_training \
  --symbol ETHUSDT --execution-mode blank

# Metrics to track:
# - Test accuracy on rare regimes (Regime 3: should improve from baseline)
# - Overall test accuracy (should improve from 19.72%)
# - CV accuracy (may decrease slightly from 74.96%, that's OK)
# - Per-class precision/recall in classification report
```

---

## Expected Outcomes

### After Phase 1 (#1 + #2):
- ✅ Clustering: Regime distribution 10-35% each (instead of 0-63%)
- ✅ Training: Test accuracy 35-45% (up from 19.72%)
- ✅ Training: Rare regime recall 40%+ (up from ~0%)
- ⚠️ May see slight drop in CV accuracy (74.96% → 68-72%)
- ⚠️ Statistical quality metrics may decrease 5-10%

### After Phase 2 (#3):
- ✅ Further improvement in regime balance
- ✅ More stable regime distributions across runs
- ⚠️ Possible conflict with CV ratio optimization

### After Phase 3 (#4):
- ✅ Marginal improvements over Phase 1+2 (2-5% accuracy gain)
- ⚠️ Significant implementation complexity
- ⚠️ Longer training time

---

## Code Location Summary

| Proposal | File | Lines | Complexity | Priority |
|----------|------|-------|------------|----------|
| #1: 5% Constraint | `rolling_hmm_clustering/hpo_config.py` | 387, 537-542 | Low | P0 |
| #2: Adaptive Weights | `components/regime_models_training.py` | 1656-1914 | Medium | P0 |
| #3: Balance Penalty | `rolling_hmm_clustering/hpo_config.py` | 526-542 | Low | P1 |
| #4: Focal Loss | New file + integrations | N/A | High | P2 |

---

## Recommendation

**START WITH PHASE 1 ONLY** (#1 + #2):
- Both are proven, simple, high-impact changes
- Can be implemented and tested in < 2 hours
- Should solve 80% of the class imbalance problem
- Low risk of breaking existing functionality

**DEFER PHASE 3** (#4):
- Only implement if Phase 1 results are insufficient
- Requires significant engineering effort
- Marginal gains over simpler approaches
- Higher debugging/maintenance cost

**MONITOR METRICS**:
- Regime size distribution (should be 10-35% each, min 5%)
- Per-class recall (especially rare regimes)
- Test accuracy (target: 35-45%)
- Training time (should not increase significantly)
