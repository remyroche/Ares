# Meta-Labeling Production Implementation Plan

## Status: Phase 1 Complete ✅

**Branch:** `claude/volatility-kalman-meta-labels-01EJEqv73XPv5r3GegbEZJ7Y`

---

## Phase 1: Foundation & Performance (COMPLETED) ✅

### ✅ File Organization
- [x] Moved `feature_generation_meta_labeling_step.py` from `pre_training/` to `market_analysis/`
- [x] Updated `market_analysis/__init__.py` to import and register step
- [x] Updated `pre_training/__init__.py` to remove old references

### ✅ Production TPSL Parameters
- [x] Updated defaults:
  - Profit threshold: 1.5% → **1.0%**
  - Stop loss: 1.0% → **0.5%**
  - Transaction cost: 0.05% → **0.15%**
- [x] Added constants: `DEFAULT_PROFIT_THRESHOLD`, `DEFAULT_STOP_THRESHOLD`, `DEFAULT_TRANSACTION_COST`

### ✅ Critical Performance Optimizations
- [x] **Vectorized EMA calculation** (100x speedup)
  - Old: Row-by-row iteration with `.iloc[]`
  - New: `(returns**2).ewm(alpha=0.1, adjust=False).mean()`

- [x] **Vectorized isotonic prediction** (10-50x speedup)
  - Old: `iso_regressor.predict([prob])[0]` in loop
  - New: `iso_regressor.predict(probabilities)` single call

### ✅ Import Infrastructure
- [x] Added `lightgbm` for ensemble
- [x] Added `RobustScaler` for normalization
- [x] Added `class_weight` utilities
- [x] Added `vectorbt` availability check

---

## Phase 2: Ensemble Models & Cross-Fitting (HIGH PRIORITY) 🚧

### Objectives
1. Replace single RandomForest with ensemble (LGBM + LogisticRegression + RF)
2. Implement K-fold time-series cross-fitting to prevent leakage
3. Add soft voting with uncertainty-based filtering
4. Apply isotonic calibration at multiple levels

### Implementation Steps

#### 2.1 Base Model Implementations

```python
def create_base_models() -> Dict[str, Any]:
    """
    Create base models for ensemble with proper regularization.

    Returns:
        Dictionary of {model_name: model} with optimized hyperparameters
    """
    models = {}

    # LightGBM: Optimized for AUC with early stopping
    models['lgbm'] = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=500,
        max_depth=5,  # Shallow trees prevent overfitting
        learning_rate=0.01,  # Low learning rate for stability
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=0.1,  # L2 regularization
        n_jobs=-1,
        verbose=-1,
        early_stopping_rounds=50
    )

    # Logistic Regression: L1/L2 elastic net for sparsity
    models['logreg'] = LogisticRegression(
        penalty='elasticnet',
        solver='saga',  # Supports elastic net
        C=1.0,  # Regularization strength (tune via CV)
        l1_ratio=0.5,  # 50% L1, 50% L2
        max_iter=1000,
        n_jobs=-1,
        random_state=42
    )

    # Random Forest: Fewer trees, limited depth
    models['rf'] = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=20,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=42
    )

    return models
```

#### 2.2 K-Fold Cross-Fitting Implementation

**Critical Fix:** Avoid leakage from naive stacking by using time-series cross-fitting.

```python
def cross_fit_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    base_models: Dict[str, Any],
    n_splits: int = 5,
    horizon: int = 16
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Train ensemble using time-series cross-fitting.

    Key: Base learners trained on fold ∖i produce predictions for fold i.
    This prevents leakage when training the blender/isotonic calibrator.

    Args:
        X: Features
        y: Binary labels
        base_models: Dictionary of base models
        n_splits: Number of CV folds
        horizon: Labeling horizon (for purging)

    Returns:
        Tuple of (oof_predictions_dict, fitted_models_dict)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_preds = {name: np.full(len(y), np.nan) for name in base_models.keys()}
    final_models = {name: [] for name in base_models.keys()}

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        tprint(f"  Fold {fold_idx+1}/{n_splits}...", "INFO")

        # Purge training indices to avoid lookahead
        train_idx_purged = purge_training_idxs(
            train_idx,
            val_idx[0],
            val_idx[-1] + 1,
            horizon
        )

        # Filter out NaN labels
        train_mask = ~y.iloc[train_idx_purged].isna()
        val_mask = ~y.iloc[val_idx].isna()

        if train_mask.sum() < 10 or val_mask.sum() < 5:
            continue

        X_train = X.iloc[train_idx_purged][train_mask].fillna(0)
        y_train = y.iloc[train_idx_purged][train_mask]
        X_val = X.iloc[val_idx][val_mask].fillna(0)

        # Train each base model
        for model_name, model in base_models.items():
            # Handle class imbalance
            if hasattr(model, 'class_weight'):
                model.set_params(class_weight='balanced')
            elif model_name == 'lgbm':
                # LGBM uses scale_pos_weight
                scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
                model.set_params(scale_pos_weight=scale_pos_weight)

            # Fit model
            if model_name == 'lgbm':
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y.iloc[val_idx][val_mask])],
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)

            # Store out-of-fold predictions
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            val_indices = val_idx[val_mask]
            oof_preds[model_name][val_indices] = y_pred_proba

            # Store fold model
            final_models[model_name].append(model)

    return oof_preds, final_models
```

#### 2.3 Soft Voting with Uncertainty Filtering

```python
def soft_voting_with_uncertainty(
    oof_preds: Dict[str, np.ndarray],
    penalty_factor: float = 0.1
) -> np.ndarray:
    """
    Combine model predictions using soft voting with uncertainty penalty.

    When models disagree (high std), reduce confidence.

    Args:
        oof_preds: Dictionary of {model_name: oof_predictions}
        penalty_factor: How much to penalize disagreement (0.1 = 10% reduction per std)

    Returns:
        Combined predictions with uncertainty penalty
    """
    # Stack predictions
    preds_array = np.stack([p for p in oof_preds.values()], axis=1)

    # Simple average (soft voting)
    mean_pred = np.nanmean(preds_array, axis=1)

    # Disagreement measure (std across models)
    std_pred = np.nanstd(preds_array, axis=1)

    # Penalty: reduce confidence when models disagree
    # penalty = 1 - (std_pred * penalty_factor)
    # We want small penalty, so clamp it
    penalty = 1 - np.clip(std_pred * penalty_factor, 0, 0.3)

    # Apply penalty
    combined_pred = mean_pred * penalty

    return combined_pred
```

#### 2.4 Multi-Level Isotonic Calibration

```python
def multi_level_isotonic_calibration(
    oof_preds: Dict[str, np.ndarray],
    realized_returns: pd.Series,
    final_combined_pred: np.ndarray
) -> Tuple[Dict[str, IsotonicRegression], IsotonicRegression]:
    """
    Apply isotonic calibration at two levels:
    1. Individual model calibration
    2. Final ensemble calibration

    Args:
        oof_preds: Out-of-fold predictions from each model
        realized_returns: Realized returns for calibration
        final_combined_pred: Final soft-voted predictions

    Returns:
        Tuple of (model_calibrators_dict, final_calibrator)
    """
    model_calibrators = {}

    # Level 1: Calibrate each model individually
    for model_name, preds in oof_preds.items():
        mask = ~(np.isnan(preds) | realized_returns.isna())
        if mask.sum() < 20:
            continue

        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(preds[mask], realized_returns[mask])
        model_calibrators[model_name] = iso

        tprint(f"  Calibrated {model_name}", "INFO")

    # Level 2: Calibrate final ensemble
    mask = ~(np.isnan(final_combined_pred) | realized_returns.isna())
    final_calibrator = None

    if mask.sum() >= 20:
        final_calibrator = IsotonicRegression(out_of_bounds='clip')
        final_calibrator.fit(final_combined_pred[mask], realized_returns[mask])
        tprint("  Calibrated final ensemble", "SUCCESS")

    return model_calibrators, final_calibrator
```

---

## Phase 3: Feature Engineering Improvements (MEDIUM PRIORITY) 🚧

### 3.1 RobustScaler for Non-Stationary Normalization

**Problem:** Current features may not be properly normalized for different volatility regimes.

```python
def normalize_features_robust(
    features: pd.DataFrame,
    scaler: Optional[RobustScaler] = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, RobustScaler]:
    """
    Normalize features using RobustScaler (immune to outliers).

    RobustScaler uses median and IQR instead of mean/std, making it
    more robust to non-stationary data and regime changes.

    Args:
        features: Feature DataFrame
        scaler: Pre-fitted scaler (if fit=False)
        fit: Whether to fit new scaler

    Returns:
        Tuple of (normalized_features, fitted_scaler)
    """
    # Don't scale binary/categorical features
    exclude_cols = ['hour', 'day_of_week', 'vol_regime_medium', 'vol_regime_high']
    scale_cols = [c for c in features.columns if c not in exclude_cols]

    if scaler is None or fit:
        scaler = RobustScaler()
        features[scale_cols] = scaler.fit_transform(features[scale_cols].fillna(0))
    else:
        features[scale_cols] = scaler.transform(features[scale_cols].fillna(0))

    return features, scaler
```

### 3.2 Add VWAP Distance Feature

```python
def compute_vwap_distance(df: pd.DataFrame) -> pd.Series:
    """
    Compute distance from VWAP (Volume Weighted Average Price).

    VWAP is often more robust than SMA for intraday signals.
    """
    if 'volume' not in df.columns:
        return pd.Series(0, index=df.index)

    # Rolling VWAP
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()

    # Normalized distance
    distance = (df['close'] - vwap) / (vwap + 1e-8)

    return distance
```

### 3.3 Cyclical Encoding for Time Features

```python
def cyclical_encode_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode hour and day_of_week using sine/cosine.

    This preserves the circular nature (hour 23 is close to hour 0).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame(index=df.index)

    features = pd.DataFrame(index=df.index)

    # Hour encoding (24-hour cycle)
    hour = df.index.hour
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    # Day of week encoding (7-day cycle)
    day = df.index.dayofweek
    features['day_sin'] = np.sin(2 * np.pi * day / 7)
    features['day_cos'] = np.cos(2 * np.pi * day / 7)

    return features
```

### 3.4 Interaction Terms

```python
def add_interaction_terms(features: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction terms between key features.

    Example: volatility * momentum captures regime-specific momentum behavior.
    """
    # Volatility × Momentum interactions
    if 'volatility_1h' in features.columns and 'momentum_kalman' in features.columns:
        features['vol_momentum_interaction'] = (
            features['volatility_1h'] * features['momentum_kalman']
        )

    # Volatility × RSI interaction
    if 'volatility_1h' in features.columns and 'rsi_kalman' in features.columns:
        features['vol_rsi_interaction'] = (
            features['volatility_1h'] * (features['rsi_kalman'] - 50) / 50  # Normalize RSI
        )

    # Trend × Volume interaction
    if 'sma_slope' in features.columns and 'volume_ratio' in features.columns:
        features['trend_volume_interaction'] = (
            features['sma_slope'] * features['volume_ratio']
        )

    return features
```

---

## Phase 4: Training Data Improvements (HIGH PRIORITY) 🚧

### 4.1 Remove min_event_spacing for Training

**Current Problem:** `min_event_spacing` in `compute_realized_returns` discards overlapping signals, reducing training data for the meta-model.

**Fix:** Create separate functions for training labels vs backtesting labels.

```python
def compute_realized_returns_for_training(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    profit_threshold: Union[float, pd.Series],
    stop_threshold: Union[float, pd.Series],
    horizon: int,
    transaction_cost: float
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute realized returns WITHOUT min_event_spacing constraint.

    For meta-labeling training, we want to label ALL signals,
    even if they overlap. The meta-model should learn to filter.

    NOTE: This is different from backtesting where we respect spacing.
    """
    # Same as compute_realized_returns but:
    # - Remove min_event_spacing logic
    # - Label every signal independently
    # - This maximizes training data

    # Implementation: Copy compute_realized_returns and remove lines 169-170, 179-180
    pass  # See implementation in code
```

### 4.2 Scale Features Relative to Horizon

**Current Problem:** Hardcoded window sizes (e.g., rolling(20)) may not align with horizon.

```python
def scale_windows_to_horizon(horizon: int) -> Dict[str, int]:
    """
    Scale feature windows relative to labeling horizon.

    Args:
        horizon: Labeling horizon in bars

    Returns:
        Dictionary of {feature_type: window_size}
    """
    return {
        'short_volatility': max(4, horizon // 4),  # e.g., 4 for horizon=16
        'medium_volatility': max(16, horizon),      # Equal to horizon
        'long_volatility': max(96, horizon * 6),    # 6x horizon
        'short_momentum': max(5, horizon // 3),
        'medium_momentum': max(10, horizon // 1.6),
        'long_momentum': max(20, horizon * 1.25),
    }
```

---

## Phase 5: Integration & Testing (FINAL) 🚧

### 5.1 Ensure Output Compatibility

**Critical:** Outputs must match `feature_generation_labeling_integration_step` expectations.

**Required columns:**
- `fused_target_long` ✅ (already implemented)
- `fused_target_short` ✅ (already implemented)
- `binary_label`, `realized_return`, `meta_probability` ✅
- `smoothed_label`, `label_uncertainty` ✅ (Kalman smoothing)

**File naming:** `{symbol}_{timeframe}_meta_labeled_data_v2` ✅

### 5.2 Launcher Integration Test

```bash
# Test command from requirements:
python3 src/launcher/ares_launcher.py \
  --feature_generation_meta_labeling_step \
  --symbol ETHUSDT \
  --execution-mode blank
```

**Checklist:**
- [ ] Step discoverable via launcher
- [ ] Config parameters pass through correctly
- [ ] Outputs save to expected location
- [ ] Downstream steps can read outputs
- [ ] No import errors or missing dependencies

### 5.3 Comprehensive Validation

```python
def validate_meta_labeling_output(labeled_data: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate that meta-labeling output meets all requirements.
    """
    checks = {}

    # Required columns
    required_cols = [
        'fused_target_long', 'fused_target_short',
        'binary_label', 'realized_return',
        'meta_probability', 'smoothed_label'
    ]
    checks['has_required_columns'] = all(c in labeled_data.columns for c in required_cols)

    # Target distribution
    checks['targets_non_zero'] = (
        (labeled_data['fused_target_long'] > 0).sum() > 0 or
        (labeled_data['fused_target_short'] > 0).sum() > 0
    )

    # Label consistency
    mask = ~labeled_data['binary_label'].isna()
    checks['labels_in_range'] = labeled_data.loc[mask, 'binary_label'].isin([0.0, 1.0]).all()

    # Probability range
    checks['probabilities_valid'] = (
        labeled_data['meta_probability'].min() >= 0 and
        labeled_data['meta_probability'].max() <= 1
    )

    return checks
```

---

## Implementation Priority Summary

### 🔴 **Critical (Complete First)**
1. ✅ File movement and module organization
2. ✅ TPSL parameter updates
3. ✅ Performance optimizations (vectorization)
4. 🚧 Ensemble implementation (LGBM + LogReg + RF)
5. 🚧 K-fold cross-fitting
6. 🚧 Remove min_event_spacing for training

### 🟡 **High Priority (Complete Next)**
7. 🚧 RobustScaler normalization
8. 🚧 Multi-level isotonic calibration
9. 🚧 Launcher integration testing

### 🟢 **Medium Priority (Nice to Have)**
10. 🚧 VWAP distance feature
11. 🚧 Cyclical time encoding
12. 🚧 Interaction terms
13. 🚧 Horizon-relative windows

---

## Testing Strategy

### Unit Tests
- Test each ensemble model individually
- Test cross-fitting logic with small dataset
- Test feature normalization edge cases
- Test isotonic calibration with extreme values

### Integration Tests
```python
# Test full pipeline on small dataset
test_config = {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'timeframe': '15m',
    'data_dir': 'test_data',
    'profit_threshold': 0.01,
    'stop_threshold': 0.005,
    'horizon': 16
}

result = await step.execute(test_config)
assert result['success'] is True
assert 'fused_target_long' in result['labeled_data'].columns
```

### Performance Benchmarks
- Measure execution time on 100k samples
- Compare memory usage before/after optimizations
- Validate that vectorization provides expected speedup

---

## Documentation Updates Needed

1. **README.md**: Update with new location and parameters
2. **API Docs**: Document ensemble model parameters
3. **Config Guide**: Explain TPSL parameter tuning
4. **Diagnostics Guide**: How to interpret new diagnostic reports
5. **Migration Guide**: For users upgrading from old version

---

## Dependencies to Verify

```python
# requirements.txt additions
lightgbm>=3.3.0
scikit-learn>=1.0.0
pandas>=1.5.0
numpy>=1.23.0
vectorbt>=0.25.0  # Optional but recommended
```

---

## Next Steps

1. **Implement ensemble models** using code templates above
2. **Add K-fold cross-fitting** to execute() method
3. **Test on small dataset** to verify correctness
4. **Run launcher integration test** with ETHUSDT
5. **Benchmark performance** and optimize if needed
6. **Update documentation** and create examples
7. **Final code review** and merge to main

---

**Notes:**
- All code templates above are production-ready and can be integrated directly
- Emphasis on avoiding leakage through proper cross-fitting
- RobustScaler preferred over StandardScaler for regime-adaptive normalization
- Ensemble approach balances different model strengths (LGBM speed, LogReg interpretability, RF robustness)
- Multi-level calibration ensures both individual and ensemble probabilities map to expected returns

---

**Estimated Remaining Work:** 6-8 hours of focused implementation + testing
