# Implementation Summary: Trade Frequency Enhancement

**Date:** 2025-12-04  
**File Modified:** `src/training/steps/labeling/feature_generation_meta_labeling_step.py`

---

## Changes Implemented

### 1. ✅ Transaction Cost & Thresholds Updated

```python
DEFAULT_TRANSACTION_COST = 0.003        # 0.30% (was 0.15%)
DEFAULT_PROBABILITY_THRESHOLD = 0.55    # New constant
DEFAULT_EXPECTED_RETURN_THRESHOLD = 0.003  # 0.30% (was 0.45%)
```

**Impact:** More realistic transaction costs, lower gating thresholds for more trades.

---

### 2. ✅ Vol-Aware Dual-Mode Consensus (Linear Formula)

Replaced threshold-based weighting with linear interpolation:

```python
# Linear vol-aware weighting:
# vol_ratio < 0.7: favor mean-reversion (low vol, ranging market)
# vol_ratio > 1.3: favor momentum (high vol, trending market)
momentum_weight = np.clip((vol_ratio - 0.7) / 0.6, 0.2, 0.9)
mr_weight = 1.0 - momentum_weight

# Weighted consensus: blend momentum and mean-reversion based on vol regime
weighted_score = momentum_weight * momentum_score + mr_weight * mr_score
```

**Impact:** 
- Low vol (ratio < 0.7): 20% momentum + 80% mean-reversion
- Normal vol (ratio ≈ 1.0): 55% momentum + 45% mean-reversion
- High vol (ratio > 1.3): 90% momentum + 10% mean-reversion

---

### 3. ✅ New Primary Signals Added (7 New Signals)

| Signal | Type | Logic |
|--------|------|-------|
| `bb_fade` | Mean-reversion | Long at lower Bollinger Band, short at upper |
| `atr_breakout` | Momentum | Long on strong upward ATR breakout, short on downward |
| `volume_spike` | Momentum | Signal in direction of price when volume > 2x mean |
| `range_fade` | Mean-reversion | Long at bottom 15% of range, short at top 85% |
| `rsi_mr` | Mean-reversion | RSI < 35 → long, RSI > 65 → short (tighter thresholds) |
| `mtf_trend` | Momentum | Multi-timeframe trend (1-hour momentum) |
| `mtf_confluence` | Momentum | Agreement between short-term and MTF momentum |

**Signal Categories for Consensus:**
- Momentum: `rsi, rsi_long, macd, macd_long, ma, mom, atr_breakout, volume_spike, mtf_trend, mtf_confluence`
- Mean-Reversion: `bb_fade, range_fade, rsi_mr`

---

### 4. ✅ target_trades_per_day Increased

```python
target_trades_per_day: float = 4.0  # Was 2.0
```

**Impact:** Looser momentum thresholds in dynamic tuning → more raw signals.

---

### 5. ✅ CatBoost Added to Ensemble with Stacked Generalization

**CatBoost Model:**
```python
if CATBOOST_AVAILABLE:
    models['catboost'] = CatBoostClassifier(
        iterations=800,
        depth=6,
        learning_rate=0.01,
        l2_leaf_reg=3.0,
        auto_class_weights='Balanced',
        eval_metric='AUC',
        ...
    )
```

**Stacked Generalization:**
```python
# Instead of simple averaging, use a meta-learner to learn optimal weights
meta_learner = LogisticRegression(max_iter=1000, solver='lbfgs')
meta_learner.fit(stack_features_valid, y_valid)
ensemble_probs = meta_learner.predict_proba(stack_features_all)[:, 1]
```

**Impact:** Better ensemble calibration, learned optimal model weights.

---

### 6. ✅ New Features Added

#### Time-of-Day Patterns (5 new features - selective)
| Feature | Description |
|---------|-------------|
| `hour_sin`, `hour_cos` | Cyclical encoding of hour (24-hour cycle in 2 features) |
| `is_good_hour` | Hours 3, 5, 10 (>56% win rate) |
| `is_bad_hour` | Hours 0, 13, 19 (<45% win rate) |
| `is_sunday` | Sunday indicator (worst day at 39.5%)

#### Order Flow Imbalance (OFI) Proxy (5 new features)
| Feature | Description |
|---------|-------------|
| `cvd_proxy` | Cumulative Volume Delta proxy |
| `volume_pressure` | Volume-weighted price pressure |
| `ofi_proxy` | Order flow imbalance approximation |
| `volume_imbalance` | Buy vs sell volume ratio |
| `absorption_ratio` | Volume at price extremes vs mid-range |

---

### 7. ✅ Gating Grid Updated

```python
# Updated prob thresholds (lower for more trades)
prob_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]  # Was [0.55, 0.60, 0.65, 0.70, 0.75]

# Updated ER multipliers (lower for 0.30% target)
er_multipliers = [0.5, 1.0, 1.5, 2.0]  # Was [1.0, 2.0, 3.0]

# Fallback uses new defaults
prob_threshold: 0.55        # Was 0.60
expected_return_threshold: 0.003  # Was transaction_cost * 2.0
```

---

## Expected Impact

| Metric | Before | Expected After |
|--------|--------|----------------|
| Trades/day | 0.20-0.45 | 0.8-1.5 |
| Low-vol win rate | 37.9% | 45-50% |
| Signal sources | 6 | 13 |
| Features | ~80 | ~90 (+10 selective) |
| Ensemble models | 4 (LGBM, XGB, RF, LogReg) | 5 (+CatBoost) |

---

## Files Modified

1. `src/training/steps/labeling/feature_generation_meta_labeling_step.py`
   - Lines 266-276: New constants
   - Lines 69-73: CatBoost import
   - Lines 555-566: New signal parameters
   - Lines 720-855: New signals and vol-aware consensus
   - Lines 1755-1880: New time/OFI features
   - Lines 4397-4410: CatBoost model
   - Lines 4957-4990: Stacked generalization
   - Lines 6652-6732: Updated gating grid

---

## Next Steps

1. **Run HPO:** Re-run `meta_labeling_hpo_experiment` to find optimal parameters with new signals
2. **Run diagnostics:** Execute `snr_diagnostics` to verify improved low-vol performance
3. **Backtest:** Run `meta_gated_backtest` to measure actual trade frequency improvement
4. **Monitor:** Check that new features have reasonable importance in model output

---

*Implementation completed 2025-12-04*
