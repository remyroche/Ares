# Model Improvement Plan for Extreme Price Movements

## Executive Summary

Based on analysis of the training gate report and codebase, this document outlines targeted improvements for three main issues:
1. **PR-AUC gaps** - Models not achieving ≥0.54 target
2. **Fold robustness issues** - fold_ratio < 0.70, worst_fold < -0.5%
3. **Lift@k issues** - lift@20% < 1.2 for some models

---

## Problem Analysis by Model

### Critical Failures (Multiple Issues)

| Model | PR-AUC | Fold Ratio | Worst Fold | Lift@20% | Priority |
|-------|--------|------------|------------|----------|----------|
| long_mr_H2:lightgbm | 0.468 | 0.25 | -8.6% | 1.26 | HIGH |
| long_tf_H2:lightgbm | 0.411 | 0.50 | -5.1% | 1.14 | HIGH |
| long_mr_H8:lightgbm | 0.510 | 0.25 | -15.1% | 1.25 | HIGH |
| long_tf_H4:catboost | 0.531 | 0.50 | -7.5% | 1.39 | MEDIUM |

### PR-AUC Only Failures

| Model | PR-AUC | Gap | Fold Ratio | Lift@20% |
|-------|--------|-----|------------|----------|
| short_mr_H2:xgboost | 0.399 | -0.141 | 1.0 | 1.30 |
| short_tf_H2:xgboost | 0.468 | -0.072 | 1.0 | 1.28 |
| short_mr_H4:xgboost | 0.490 | -0.050 | 0.75 | 1.20 |
| short_mr_H8:extratrees | 0.454 | -0.086 | 0.75 | 1.26 |
| long_tf_H8:extratrees | 0.448 | -0.092 | 1.0 | 1.26 |
| short_tf_H8:lightgbm | 0.538 | -0.002 | 0.75 | 1.26 |

### Lift@k Only Failures

| Model | PR-AUC | Lift@20% | Fold Ratio |
|-------|--------|----------|------------|
| long_mr_H4:xgboost | 0.564 ✅ | 1.04 ❌ | 1.0 |

### Passing Model (Reference)

| Model | PR-AUC | Fold Ratio | Worst Fold | Lift@20% |
|-------|--------|------------|------------|----------|
| short_tf_H4:extratrees | 0.592 ✅ | 1.0 ✅ | +1.26% ✅ | 1.32 ✅ |

---

## Root Cause Analysis

### 1. PR-AUC Gaps

**Causes:**
- **Class imbalance**: Extreme price movements are rare events (~30% prevalence)
- **Feature signal dilution**: MDI feature selection optimizes for overall impurity reduction, not ranking quality
- **Model capacity**: Current hyperparameters (max_depth=4, heavy regularization) may be too conservative
- **Calibration mismatch**: Isotonic calibration optimizes Brier score, not PR-AUC

**Evidence from code:**
```python
# model_race.py line 174-181: ExtraTrees params
et_params = {
    "max_depth": 7,           # Could be deeper for better separation
    "min_samples_leaf": 50,   # High regularization
    ...
}

# model_race.py line 205-218: LightGBM params  
lgb_params = {
    "max_depth": 4,           # Very shallow
    "lambda_l2": 5.0,         # Heavy L2 regularization
    ...
}
```

### 2. Fold Robustness Issues

**Causes:**
- **Regime sensitivity**: Models trained on one market regime fail on another
- **Insufficient regime features**: Current features don't capture market state transitions
- **Purge window too small**: `purge=5` may not be enough for 1h data with overlapping labels
- **Sample weight concentration**: Weights may over-emphasize certain periods

**Evidence from code:**
```python
# purged_cv.py line 29: Purge window
purge: int = 5,  # Only 5 samples purge

# model_race.py line 330: CV setup
tscv = PurgedKFold(n_splits=5, purge=5, embargo=2)
```

### 3. Lift@k Issues

**Causes:**
- **Loss function mismatch**: Log loss optimization doesn't guarantee top-k concentration
- **Feature selection not top-k aware**: MDI doesn't optimize for precision at top
- **Probability calibration**: Isotonic regression can flatten top predictions

---

## Proposed Improvements

### Phase 1: Fold Robustness (Highest Priority for Critical Models)

#### 1.1 Regime-Aware Cross-Validation

**File:** `extreme_price_movements/purged_cv.py`

Add regime-stratified purged CV:

```python
class RegimeStratifiedPurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold that ensures each fold has similar regime distribution.
    
    Regimes are identified by:
    - Volatility regime (high/low vol)
    - Trend regime (trending/ranging)
    - Liquidity regime
    """
    def __init__(self, n_splits=5, purge=10, embargo=5, 
                 regime_labels=None, min_regime_ratio=0.7):
        self.n_splits = n_splits
        self.purge = purge
        self.embargo = embargo
        self.regime_labels = regime_labels
        self.min_regime_ratio = min_regime_ratio
    
    def split(self, X, y=None, groups=None):
        # Ensure each validation fold has similar regime distribution
        # to training data
        ...
```

#### 1.2 Increase Purge/Embargo Windows

**File:** `extreme_price_movements/model_race.py`

```python
# Line 330: Increase purge for 1h data
# OLD:
tscv = PurgedKFold(n_splits=5, purge=5, embargo=2)

# NEW:
tscv = PurgedKFold(n_splits=5, purge=10, embargo=5)
```

**Rationale:** With 1h data and label horizons up to 8h, a purge of 5 samples (5h) is insufficient. Labels at t can overlap with labels at t+8h.

#### 1.3 Add Regime Features

**File:** `extreme_price_movements/features.py`

Add regime indicator features:

```python
def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute market regime features for robustness.
    """
    # Volatility regime (rolling z-score of realized vol)
    rv_24h = df['ret'].rolling(24).std()
    rv_mean_7d = rv_24h.rolling(7*24).mean()
    rv_std_7d = rv_24h.rolling(7*24).std()
    df['vol_regime_z'] = (rv_24h - rv_mean_7d) / (rv_std_7d + 1e-9)
    df['is_high_vol_regime'] = (df['vol_regime_z'] > 1.0).astype(float)
    df['is_low_vol_regime'] = (df['vol_regime_z'] < -1.0).astype(float)
    
    # Trend regime (ADXR or simple trend strength)
    df['trend_regime'] = df['trend_pct'].rolling(48).mean().abs()
    df['is_trending'] = (df['trend_regime'] > 0.02).astype(float)
    df['is_ranging'] = (df['trend_regime'] < 0.005).astype(float)
    
    # Liquidity regime
    df['liq_regime'] = df['vol_z'].rolling(24).mean()
    
    return df
```

#### 1.4 Regime-Conditional Sample Weights

**File:** `extreme_price_movements/sample_weights.py`

Add regime-aware weighting:

```python
def compute_regime_robust_weights(
    base_weights: np.ndarray,
    regime_labels: np.ndarray,
    regime_balance_factor: float = 0.3
) -> np.ndarray:
    """
    Adjust sample weights to ensure regime balance.
    
    Down-weights over-represented regimes to prevent model from
    overfitting to dominant market conditions.
    """
    unique_regimes, counts = np.unique(regime_labels, return_counts=True)
    target_count = len(regime_labels) / len(unique_regimes)
    
    regime_weights = {}
    for r, c in zip(unique_regimes, counts):
        # Down-weight over-represented regimes
        regime_weights[r] = min(1.0, target_count / c)
    
    regime_adjustment = np.array([regime_weights[r] for r in regime_labels])
    
    # Blend with base weights
    adjusted = base_weights * (1 - regime_balance_factor + regime_balance_factor * regime_adjustment)
    
    return adjusted
```

---

### Phase 2: PR-AUC Improvements

#### 2.1 Ranking-Optimized Loss Function

**File:** `extreme_price_movements/model_race.py`

Add LambdaRank-style pairwise loss for LightGBM/XGBoost:

```python
def get_ranking_optimized_params(model_type: str, for_pr_auc: bool = True):
    """
    Get model parameters optimized for PR-AUC / ranking.
    """
    if model_type == "lightgbm":
        return {
            "n_estimators": 400,
            "max_depth": 6,           # Increased from 4
            "learning_rate": 0.03,
            "subsample": 0.7,
            "feature_fraction": 0.7,
            "lambda_l2": 2.0,         # Reduced from 5.0
            "lambda_l1": 0.1,         # Add some L1 for sparsity
            "min_child_samples": 30,  # Reduced from implicit 50
            # Ranking-specific
            "objective": "binary",
            "metric": ["auc", "average_precision"],  # Track PR-AUC
            "is_unbalance": True,      # Handle class imbalance
        }
    elif model_type == "xgboost":
        return {
            "n_estimators": 400,
            "max_depth": 5,           # Increased from 4
            "learning_rate": 0.03,
            "reg_lambda": 2.0,        # Reduced from 5.0
            "reg_alpha": 0.1,
            "min_child_weight": 10,   # Reduced from 20
            "scale_pos_weight": 2.5,  # Explicit class balance
            "eval_metric": ["auc", "aucpr"],  # PR-AUC metric
        }
    elif model_type == "catboost":
        return {
            "iterations": 400,
            "depth": 5,               # Increased from 4
            "learning_rate": 0.03,
            "l2_leaf_reg": 5.0,       # Reduced from 10.0
            "auto_class_weights": "Balanced",
            "eval_metric": "PRAUC",   # Direct PR-AUC optimization
        }
    elif model_type == "extratrees":
        return {
            "n_estimators": 400,
            "max_depth": 8,           # Increased from 7
            "min_samples_leaf": 30,   # Reduced from 50
            "max_features": 0.7,      # More features per split
            "class_weight": "balanced",
        }
```

#### 2.2 PR-AUC Aware Feature Selection

**File:** `extreme_price_movements/feature_selection_extreme_events.py`

Add PR-AUC based feature ranking:

```python
def compute_pr_auc_feature_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    n_bins: int = 10
) -> pd.Series:
    """
    Compute feature importance based on PR-AUC contribution.
    
    For each feature, compute how much it improves PR-AUC when
    used for binning/ranking.
    """
    from sklearn.metrics import average_precision_score
    
    base_pr_auc = average_precision_score(y, np.full_like(y, y.mean()))
    
    importance = {}
    for col in X.columns:
        # Use feature as prediction score
        try:
            feat_pr_auc = average_precision_score(y, X[col].values)
            importance[col] = feat_pr_auc - base_pr_auc
        except:
            importance[col] = 0.0
    
    return pd.Series(importance).sort_values(ascending=False)


def mdi_feature_selection_v4_pr_auc(
    X: pd.DataFrame,
    y: np.ndarray,
    base_model,
    sample_weight: np.ndarray = None,
    pr_auc_weight: float = 0.3,
    **kwargs
) -> SelectionResult:
    """
    MDI feature selection v4 with PR-AUC awareness.
    
    Combines MDI importance with PR-AUC contribution.
    """
    # Get standard MDI importance
    mdi_result = mdi_feature_selection_v3(X, y, base_model, sample_weight, **kwargs)
    
    # Get PR-AUC importance
    pr_auc_imp = compute_pr_auc_feature_importance(X, y)
    
    # Normalize both
    mdi_norm = mdi_result.importance / mdi_result.importance.sum()
    pr_norm = pr_auc_imp / pr_auc_imp.abs().sum()
    
    # Combine
    combined = (1 - pr_auc_weight) * mdi_norm + pr_auc_weight * pr_norm
    combined = combined.sort_values(ascending=False)
    
    # Select top features
    selected = combined.head(len(mdi_result.selected_features)).index.tolist()
    
    return SelectionResult(
        selected_features=selected,
        importance=combined,
        method="mdi_v4_pr_auc"
    )
```

#### 2.3 Class Balance Handling

**File:** `extreme_price_movements/model_race.py`

Improve class balance handling:

```python
def _compute_sample_weights_for_pr_auc(
    y: np.ndarray,
    sample_weight: np.ndarray = None,
    method: str = "effective"
) -> np.ndarray:
    """
    Compute sample weights optimized for PR-AUC.
    
    Methods:
    - "effective": Use effective number of samples (inverse class freq)
    - "focal": Focal loss style weighting (more weight on hard examples)
    - "smote": Synthetic minority oversampling (not recommended for time series)
    """
    if method == "effective":
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        w_pos = len(y) / (2 * n_pos)
        w_neg = len(y) / (2 * n_neg)
        weights = np.where(y >= 0.5, w_pos, w_neg)
        
    elif method == "focal":
        # Base weights
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        w_pos = len(y) / (2 * n_pos)
        w_neg = len(y) / (2 * n_neg)
        base_weights = np.where(y >= 0.5, w_pos, w_neg)
        
        # Focal adjustment (would need model predictions)
        # For now, just use base weights
        weights = base_weights
        
    if sample_weight is not None:
        weights = weights * sample_weight
    
    return weights.astype(np.float32)
```

---

### Phase 3: Lift@k Improvements

#### 3.1 Top-K Aware Feature Selection

**File:** `extreme_price_movements/feature_selection_extreme_events.py`

```python
def compute_topk_feature_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    k_frac: float = 0.20,
    n_bootstrap: int = 50
) -> pd.Series:
    """
    Compute feature importance based on precision@k contribution.
    
    For each feature, measure how well it ranks positive examples
    in the top-k.
    """
    k = max(1, int(len(y) * k_frac))
    n_pos = y.sum()
    base_prec_at_k = n_pos / len(y)  # Random baseline
    
    importance = {}
    rng = np.random.RandomState(42)
    
    for col in X.columns:
        feat_vals = X[col].values
        
        # Skip if feature has no variance
        if np.std(feat_vals) < 1e-9:
            importance[col] = 0.0
            continue
        
        prec_samples = []
        for _ in range(n_bootstrap):
            idx = rng.choice(len(y), size=len(y), replace=True)
            feat_sample = feat_vals[idx]
            y_sample = y[idx]
            
            # Top-k by feature
            top_k_idx = np.argsort(feat_sample)[-k:]
            prec_at_k = y_sample[top_k_idx].mean()
            prec_samples.append(prec_at_k)
        
        # Lift over random
        avg_prec = np.mean(prec_samples)
        lift = avg_prec / base_prec_at_k
        importance[col] = lift - 1.0  # Lift above baseline
    
    return pd.Series(importance).sort_values(ascending=False)
```

#### 3.2 Two-Stage Model for Top-K

**File:** `extreme_price_movements/model_race.py`

Add a two-stage model that first identifies candidates, then refines ranking:

```python
class TwoStageTopKModel(BaseEstimator, ClassifierMixin):
    """
    Two-stage model for improved top-k precision.
    
    Stage 1: Broad classifier to identify candidates
    Stage 2: Fine-grained ranker for top-k refinement
    """
    
    def __init__(self, stage1_model, stage2_model, candidate_threshold=0.3):
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.candidate_threshold = candidate_threshold
    
    def fit(self, X, y, sample_weight=None):
        # Stage 1: Train on all data
        self.stage1_model.fit(X, y, sample_weight=sample_weight)
        
        # Get stage 1 predictions
        probs1 = self.stage1_model.predict_proba(X)[:, 1]
        
        # Stage 2: Train only on candidates (top predictions)
        candidates = probs1 >= self.candidate_threshold
        X_cand = X[candidates]
        y_cand = y[candidates]
        w_cand = sample_weight[candidates] if sample_weight is not None else None
        
        if len(np.unique(y_cand)) > 1:
            self.stage2_model.fit(X_cand, y_cand, sample_weight=w_cand)
        else:
            self.stage2_model = None  # Fall back to stage 1 only
        
        return self
    
    def predict_proba(self, X):
        probs1 = self.stage1_model.predict_proba(X)[:, 1]
        
        if self.stage2_model is not None:
            # Refine predictions for candidates
            candidates = probs1 >= self.candidate_threshold
            if candidates.any():
                probs2 = probs1.copy()
                probs2[candidates] = self.stage2_model.predict_proba(X[candidates])[:, 1]
                return np.column_stack([1 - probs2, probs2])
        
        return np.column_stack([1 - probs1, probs1])
```

#### 3.3 Calibration for Top-K

**File:** `extreme_price_movements/calibration.py`

Add top-k preserving calibration:

```python
def calibrate_preserving_topk(
    probs: np.ndarray,
    y: np.ndarray,
    k_frac: float = 0.20,
    method: str = "isotonic_topk"
) -> Tuple[Callable, dict]:
    """
    Calibrate probabilities while preserving top-k ordering.
    
    Methods:
    - "isotonic_topk": Isotonic regression with top-k constraint
    - "platt_topk": Platt scaling with top-k constraint
    - "rank_preserving": Monotonic calibration that preserves rank
    """
    from sklearn.isotonic import IsotonicRegression
    
    k = max(1, int(len(y) * k_frac))
    
    if method == "isotonic_topk":
        # Standard isotonic but track top-k indices
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        calibrated = iso.fit_transform(probs, y)
        
        # Verify top-k preserved
        top_k_raw = np.argsort(probs)[-k:]
        top_k_cal = np.argsort(calibrated)[-k:]
        
        overlap = len(set(top_k_raw) & set(top_k_cal)) / k
        if overlap < 0.8:
            # Top-k not well preserved, use rank-preserving instead
            return calibrate_preserving_topk(probs, y, k_frac, "rank_preserving")
        
        return iso, {"method": method, "topk_overlap": overlap}
    
    elif method == "rank_preserving":
        # Monotonic transformation that strictly preserves rank
        # Use quantile matching
        from scipy.stats import rankdata
        
        ranks = rankdata(probs)
        # Map ranks to calibrated probabilities using empirical CDF
        # For each rank, use the actual positive rate at that rank
        n_bins = min(100, len(y) // 10)
        bin_edges = np.linspace(0, len(y), n_bins + 1).astype(int)
        
        calibrated = np.zeros_like(probs)
        for i in range(n_bins):
            mask = (ranks >= bin_edges[i]) & (ranks < bin_edges[i + 1])
            if mask.any():
                calibrated[mask] = y[mask].mean() if y[mask].any() else 0.5
        
        # Store the mapping function
        def calibrator(p):
            r = rankdata(p)
            return np.interp(r, np.arange(len(calibrated)), np.sort(calibrated))
        
        return calibrator, {"method": method}
```

---

### Phase 4: Model-Specific Recommendations

#### long_mr_H2:lightgbm (CRITICAL)
- **Issues**: PR-AUC 0.468, fold_ratio 0.25, worst_fold -8.6%
- **Root cause**: Severe regime sensitivity
- **Actions**:
  1. Implement regime-stratified CV (Phase 1.1)
  2. Add regime features (Phase 1.3)
  3. Increase purge window to 10 (Phase 1.2)
  4. Use ranking-optimized params (Phase 2.1)

#### long_tf_H2:lightgbm (CRITICAL)
- **Issues**: PR-AUC 0.411, fold_ratio 0.50, worst_fold -5.1%, lift@20% 1.14
- **Root cause**: Weak signal + regime sensitivity
- **Actions**:
  1. All Phase 1 improvements
  2. PR-AUC feature selection (Phase 2.2)
  3. Two-stage model for lift@k (Phase 3.2)

#### long_mr_H8:lightgbm (CRITICAL)
- **Issues**: PR-AUC 0.510, fold_ratio 0.25, worst_fold -15.1%
- **Root cause**: Severe regime sensitivity at longer horizon
- **Actions**:
  1. Increase purge to 20 (8h horizon needs more buffer)
  2. Regime-stratified CV
  3. Consider separate models per regime

#### long_tf_H4:catboost (MEDIUM)
- **Issues**: PR-AUC 0.531, fold_ratio 0.50, worst_fold -7.5%
- **Root cause**: Regime sensitivity
- **Actions**:
  1. Regime features
  2. Regime-conditional weights
  3. Increase purge to 15

#### short_mr_H2:xgboost (PR-AUC focus)
- **Issues**: PR-AUC 0.399 (far from target)
- **Root cause**: Class imbalance + weak signal
- **Actions**:
  1. Ranking-optimized params
  2. PR-AUC feature selection
  3. Class balance handling

#### long_mr_H4:xgboost (Lift@k focus)
- **Issues**: Lift@20% 1.04 (no edge at top)
- **Root cause**: Model doesn't concentrate predictions
- **Actions**:
  1. Top-k feature selection (Phase 3.1)
  2. Two-stage model (Phase 3.2)
  3. Top-k preserving calibration (Phase 3.3)

---

## Implementation Priority

### Week 1: Fold Robustness (Critical for 3 models)
1. Increase purge/embargo windows in `model_race.py`
2. Add regime features to `features.py`
3. Implement regime-stratified CV in `purged_cv.py`

### Week 2: PR-AUC Improvements
1. Update model hyperparameters in `model_race.py`
2. Implement PR-AUC feature selection in `feature_selection_extreme_events.py`
3. Improve class balance handling

### Week 3: Lift@k Improvements
1. Implement top-k feature selection
2. Add two-stage model option
3. Implement top-k preserving calibration

### Week 4: Validation & Tuning
1. Re-run training pipeline
2. Compare against baseline
3. Fine-tune parameters

---

## Expected Impact

| Model | Current PR-AUC | Target | Expected Improvement |
|-------|---------------|--------|---------------------|
| long_mr_H2 | 0.468 | 0.54 | +0.05-0.07 |
| long_tf_H2 | 0.411 | 0.54 | +0.08-0.12 |
| long_mr_H8 | 0.510 | 0.54 | +0.03-0.05 |
| short_mr_H2 | 0.399 | 0.54 | +0.06-0.10 |
| short_mr_H4 | 0.490 | 0.54 | +0.03-0.05 |

| Model | Current Fold Ratio | Target | Expected Improvement |
|-------|-------------------|--------|---------------------|
| long_mr_H2 | 0.25 | 0.70 | +0.30-0.45 |
| long_tf_H2 | 0.50 | 0.70 | +0.15-0.25 |
| long_mr_H8 | 0.25 | 0.70 | +0.30-0.50 |

| Model | Current Lift@20% | Target | Expected Improvement |
|-------|-----------------|--------|---------------------|
| long_tf_H2 | 1.14 | 1.20 | +0.06-0.10 |
| long_mr_H4 | 1.04 | 1.20 | +0.10-0.18 |

---

## Files to Modify

1. `extreme_price_movements/model_race.py` - Model hyperparameters, CV setup
2. `extreme_price_movements/purged_cv.py` - Add regime-stratified CV
3. `extreme_price_movements/features.py` - Add regime features
4. `extreme_price_movements/feature_selection_extreme_events.py` - PR-AUC and top-k feature selection
5. `extreme_price_movements/sample_weights.py` - Regime-aware weights
6. `extreme_price_movements/calibration.py` - Top-k preserving calibration
7. `extreme_price_movements/config.py` - Add new feature keys
