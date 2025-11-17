# Signal Strength Improvement Guide

**Generated:** 2025-11-13  
**Context:** Multivariate baseline shows Test R² = 0.154 for best triplet vs 0.084 for best single feature

## Current Status

### Baseline Performance
- **Best single feature (linear):** `candlestick_doji_pattern` → Test R² = 0.084
- **Best single feature (LGBM):** `candlestick_doji_pattern` → Test R² = 0.109
- **Best pair (LGBM):** `doji + jerk` → Test R² = 0.148
- **Best triplet (LGBM):** `doji + jerk + sma_10_vwap` → Test R² = 0.154

### Recent Labeling Improvements (Applied)
✅ Profit threshold: 1.0% → 1.3% (filters noise, targets significant moves)  
✅ Lookahead: 6 periods → 3 periods (45min vs 1.5h, reduces label dilution)  
✅ Volatility floor: 0.3x → 0.5x median (avoids dead markets)  
✅ Entropy-aware squashing in label quality scoring

---

## Recommended Actions (Priority Order)

### 1. **Run Labeling with New Parameters** ⚡ IMMEDIATE
Your recent changes should improve label quality significantly.

```bash
python3 src/launcher/ares_launcher.py \
  --step feature_generation_labeling_integration_step \
  --symbol ETHUSDT --exchange binance --timeframe 15m \
  --direction long --execution-mode light
```

**Expected impact:** 10-20% improvement in label quality metrics

---

### 2. **Add Interaction Features** ⚡ HIGH PRIORITY

Use the new utility function to add explicit interactions between top features:

```python
# In feature_generation_feature_generation_step.py, after feature generation:
from src.feature_generation.categories.signal_boosting_interactions import add_signal_boosting_interactions

# After: features = feature_bank.generate_features(...)
features = add_signal_boosting_interactions(features)
```

**What it adds:**
- Multiplicative: `doji_x_jerk`, `doji_x_sma_vwap`, `jerk_x_sma_vwap`, `doji_x_jerk_x_sma_vwap`
- Ratio: `jerk_div_sma_vwap`
- Conditional: `jerk_when_doji`, `sma_vwap_when_doji`, `doji_when_strong_jerk`
- Regime-aware: `jerk_high_vol`, `jerk_low_vol`, `doji_high_vol`, `doji_low_vol`
- Temporal: `doji_x_asia`, `doji_x_europe`, `doji_x_us`, `jerk_x_asia`, etc.
- Lagged: `doji_x_jerk_lag1`, `doji_x_jerk_lag2`
- Smoothed: `doji_x_jerk_ema3`, `doji_x_jerk_ema5`

**Expected impact:** 15-25% R² improvement (based on multivariate baseline showing interactions matter)

---

### 3. **Sample Quality Filtering** 🎯 MEDIUM PRIORITY

Tighten sample selection to train only on high-quality periods:

```python
# In labeling step, after label generation:
quality_mask = (
    (label_quality > 0.4) &  # Raise from 0.3
    (ewma_vol > 0.5 * vol_median) &  # Already applied
    (volume > volume.rolling(20).mean() * 0.7)  # Add liquidity filter
)
features = features[quality_mask]
labels = labels[quality_mask]
```

**Expected impact:** 5-10% R² improvement, better generalization

---

### 4. **Multi-Horizon Targets** 🎯 MEDIUM PRIORITY

Instead of single 3-period lookahead, create ensemble of horizons:

```python
# In VolatilityAwareConfig or labeling logic:
targets = {
    'target_3p': label_at_horizon(3),   # 45min (current)
    'target_6p': label_at_horizon(6),   # 1.5h
    'target_12p': label_at_horizon(12), # 3h
}
# Use max or weighted average across horizons
final_target = pd.concat(targets, axis=1).max(axis=1)
```

**Expected impact:** 10-15% R² improvement (captures different opportunity windows)

---

### 5. **Model Architecture Improvements** 🔧 LOWER PRIORITY

Tune LGBM for better non-linear signal capture:

```python
LGBMRegressor(
    n_estimators=300,      # Up from 150
    max_depth=5,           # Up from 3
    num_leaves=31,         # Up from 15
    learning_rate=0.03,    # Down from 0.05 for stability
    min_child_samples=20,  # Stronger regularization
    subsample=0.7,         # More aggressive bagging
    colsample_bytree=0.8,
    reg_alpha=0.1,         # L1 regularization
    reg_lambda=1.0,        # L2 regularization
)
```

**Expected impact:** 5-10% R² improvement

---

### 6. **Sample Weighting** 🔧 LOWER PRIORITY

Weight recent data more heavily (markets evolve):

```python
# In model training:
sample_weights = np.exp(np.linspace(-1, 0, len(features)))
model.fit(X, y, sample_weight=sample_weights)
```

**Expected impact:** 3-5% R² improvement

---

## Implementation Sequence

### Phase 1: Quick Wins (Today)
1. ✅ Run labeling with new parameters
2. ✅ Add interaction features via utility function
3. ✅ Rerun baseline check to measure improvement

### Phase 2: Quality Refinement (This Week)
4. Implement sample quality filtering
5. Add multi-horizon targets
6. Tune LGBM hyperparameters

### Phase 3: Advanced (Next Week)
7. Add sample weighting
8. Experiment with feature selection on expanded feature set
9. Consider multi-timeframe ensemble (15m + 1h models)

---

## Expected Cumulative Impact

| Phase | Actions | Expected Test R² |
|-------|---------|------------------|
| Baseline | Current best triplet | 0.154 |
| Phase 1 | Better labels + interactions | 0.20 - 0.25 |
| Phase 2 | Quality filtering + multi-horizon | 0.25 - 0.30 |
| Phase 3 | Full optimization | 0.30 - 0.35 |

**Note:** These are optimistic estimates. Actual improvements depend on data quality and market regime stability.

---

## Quick Start: Add Interactions Now

1. **Import the utility:**
   ```python
   from src.feature_generation.categories.signal_boosting_interactions import add_signal_boosting_interactions
   ```

2. **Call after feature generation:**
   ```python
   features = add_signal_boosting_interactions(features)
   ```

3. **Rerun baseline check:**
   ```bash
   python3 src/launcher/ares_launcher.py --step feature_generation_feature_generation_step \
     --symbol ETHUSDT --exchange binance --timeframe 15m --direction long --execution-mode light
   ```

4. **Compare results:**
   - Check `outcomes/multivariate_baseline_feature_generation_*.csv`
   - Look for improved Test R² in top combinations
   - Verify new interaction features appear in top-ranked features

---

## Monitoring Success

Track these metrics after each change:

1. **Label Quality:**
   - Normalized entropy (target: < 0.7)
   - Label balance (target: 20-40% positive)
   - Quality score (target: > 0.4)

2. **Feature Performance:**
   - Best single feature Test R² (baseline: 0.084)
   - Best pair Test R² (baseline: 0.148)
   - Best triplet Test R² (baseline: 0.154)
   - Number of features with positive Test R²

3. **Model Generalization:**
   - Train/test R² gap (target: < 0.05)
   - RMSE on test set
   - Sharpe ratio in backtest (if available)

---

## Notes

- The interaction features are designed based on your actual multivariate baseline results
- They target the specific features that showed combined predictive power
- All features include proper NaN handling and are ready for production use
- The utility function is non-destructive—it returns the original features if base features are missing
