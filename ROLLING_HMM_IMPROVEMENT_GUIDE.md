# Rolling HMM Improvement Guide

## Problem Summary

**Current Performance:**
- Cross-Validation Accuracy: 50% (random guessing level for 4 regimes = 25%)
- CV Ratio (Between/Within): 1.02 (target: 2.0+)
- Economic Performance: -89% vs -10.65% buy-and-hold

**Root Causes:**
1. ✅ **FIXED**: Inverted position mapping (Fix #1)
2. ✅ **FIXED**: Low-confidence assignments (Fix #2 - bottom 10% filter)
3. ✅ **FIXED**: Missing momentum features (Fix #5)
4. ⚠️ **NEEDS FIXING**: Low regime stickiness (kappa too low)
5. ⚠️ **NEEDS FIXING**: Poor regime separation (CV Ratio 1.02)

---

## How to Improve Cross-Validation Accuracy (50% → 70%+)

### Issue: Regimes are not predictable from features

When CV accuracy is ~50%, it means the HMM can't reliably predict which regime you're in based on features. This causes:
- Noisy regime assignments
- Excessive transitions
- Poor economic performance

### Solutions:

#### **1. Increase Regime Stickiness (Kappa) - CRITICAL**

**Current kappa search space:** `[0.5, 1.0, 2.0, 3.0, 4.0]`
**Problem:** These are too low for stable regimes

**Solution:** Update `/src/training/steps/market_analysis/rolling_hmm_clustering_iterative/hpo_config.py:880`:

```python
# OLD (line 880)
'kappa': [0.5, 1.0, 2.0, 3.0, 4.0]

# NEW - Higher values for more persistent regimes
'kappa': [2.0, 5.0, 10.0, 20.0, 50.0]
```

**Impact:** Higher kappa makes regimes "sticky" - the model is less likely to transition between regimes. This:
- Increases persistence from 6.36 bars to 15-30+ bars
- Reduces noisy transitions
- Improves CV accuracy (regimes become more stable)
- Reduces transaction costs

**Trade-off:** Too high kappa (>100) makes regimes too sticky and slow to adapt to real market changes.

---

#### **2. Reduce Number of Regimes (5 → 3-4)**

**Current n_components search:** `[4, 5, 6]`
**Problem:** More regimes = harder to distinguish = lower CV accuracy

**Solution:** Update `/src/training/steps/market_analysis/rolling_hmm_clustering_iterative/hpo_config.py:878`:

```python
# OLD (line 878)
'n_components': [4, 5, 6]

# NEW - Fewer regimes are easier to distinguish
'n_components': [3, 4, 5]
```

**Impact:**
- 3 regimes: "Bear/Neutral/Bull" or "Low/Med/High Vol"
- Easier for model to distinguish
- Better CV accuracy
- Higher CV Ratio (regimes more separated)

---

#### **3. Use Better Economic Features (Already Implemented in Fix #5)**

✅ You already added momentum_1h and momentum_2h
✅ These give direct signal about price direction

**Additional features to consider:**
- Market microstructure (bid-ask spread, order flow)
- Volatility regime indicators (VIX-like)
- Trend strength (ADX-like)

---

## How to Improve CV Ratio (1.02 → 2.0+)

### Issue: Regimes don't differentiate returns well

CV Ratio = 1.02 means:
- Between-regime variance ≈ Within-regime variance
- All regimes look the same economically
- No trading edge

**Target:** CV Ratio ≥ 2.0 (regimes 2x more different from each other than internally)

### Solutions:

#### **1. Optimize HPO Objective for Economic Separation**

Currently, HPO might be optimizing for statistical metrics (silhouette, Davies-Bouldin) rather than economic separability.

**Location:** `/src/training/steps/market_analysis/rolling_hmm_clustering_iterative/hpo_config.py`

**Check the objective weights (around line 584-589):**

```python
weight_between_within_cv=0.40  # Statistical separation
weight_temporal=0.20           # Temporal stability
weight_economic=0.40           # Economic relevance
```

**Solution:** Increase `weight_economic` to prioritize economic separation:

```python
weight_between_within_cv=0.30  # Reduce statistical weight
weight_temporal=0.20           # Keep temporal weight
weight_economic=0.50           # Increase economic weight (from 0.40)
```

---

#### **2. Use Forward-Looking Features**

**Problem:** Current features are mostly backward-looking (EWMA of past returns)
**Solution:** Add more forward-predictive features

**In `/src/training/steps/market_analysis/rolling_hmm_clustering_iterative/feature_engineering.py`:**

Add to `_generate_returns_features()`:

```python
# Add trend persistence indicator
returns_sign = np.sign(returns)
features['trend_persistence'] = returns_sign.rolling(10).sum()  # Count consecutive up/down

# Add volatility breakout indicator
vol_percentile = vol_short.rolling(100).rank(pct=True)
features['vol_breakout'] = (vol_percentile > 0.90).astype(float)

# Add return acceleration
returns_momentum = returns.rolling(5).mean()
features['return_acceleration'] = returns_momentum.diff()
```

These features capture **regime characteristics** rather than just levels.

---

#### **3. Filter Training Data to Distinct Periods**

**Problem:** If your training data is mostly one regime (e.g., all sideways), HMM can't learn distinct regimes

**Solution:** Check data diversity before training

Add to `rolling_hmm_regime_discovery_step.py` before clustering:

```python
# Check data diversity
returns_std = market_data['close'].pct_change().std()
returns_skew = market_data['close'].pct_change().skew()

tprint_info(f"📊 Data characteristics:")
tprint_info(f"   - Volatility: {returns_std:.4f}")
tprint_info(f"   - Skewness: {returns_skew:.2f}")

if returns_std < 0.01:
    tprint_warning("⚠️ Very low volatility period - regimes may not be well-separated")
```

---

#### **4. Post-Process: Force Regime Separation**

After HMM fitting, you can manually enhance separation:

**In `rolling_hmm_regime_discovery_step.py` after line 700:**

```python
# Post-process: Identify and merge similar regimes
regime_returns = {}
for regime_id in range(n_components):
    mask = (regime_labels == regime_id)
    regime_returns[regime_id] = forward_returns[mask].mean()

# Check if any regimes are too similar
regime_pairs = []
for i in range(n_components):
    for j in range(i+1, n_components):
        diff = abs(regime_returns[i] - regime_returns[j])
        if diff < 0.0001:  # Less than 0.01% difference
            regime_pairs.append((i, j))
            tprint_warning(f"⚠️ Regimes {i} and {j} are very similar (diff={diff:.4%})")

# Optionally merge similar regimes or re-run with fewer components
```

---

## Implementation Priority

### Immediate (High Impact, Low Effort):

1. **✅ DONE:** Fix #1 (position mapping)
2. **✅ DONE:** Fix #2 (confidence filter - adjusted to 10%)
3. **✅ DONE:** Fix #5 (momentum features)
4. **🔧 TODO:** Update kappa range to `[2.0, 5.0, 10.0, 20.0, 50.0]`
5. **🔧 TODO:** Reduce n_components to `[3, 4, 5]`

### Next Steps (Medium Impact, Medium Effort):

6. **🔧 TODO:** Increase `weight_economic` to 0.50
7. **🔧 TODO:** Add trend persistence and volatility breakout features
8. **🔧 TODO:** Add data diversity checks

### Advanced (High Impact, High Effort):

9. **💡 CONSIDER:** Try different clustering methods (Gaussian Mixture Models, K-Means on economic features)
10. **💡 CONSIDER:** Add market microstructure features
11. **💡 CONSIDER:** Use ensemble of multiple regime models

---

## Testing Protocol

After implementing fixes, test with:

```bash
# Run Rolling HMM on ETHUSDT
python -m src.training.pipelines.regime_discovery_pipeline \
    --symbol ETHUSDT \
    --exchange binance \
    --timeframe 1h \
    --execution_mode light \
    --enable_auto_tuning True
```

**Success Metrics:**
- ✅ CV Accuracy: 60%+ (up from 50%)
- ✅ CV Ratio: 1.5+ (up from 1.02)
- ✅ Regime Persistence: 15+ bars (up from 6.36)
- ✅ Economic Performance: Beat buy-and-hold or at least within 10%

---

## Configuration Example

```python
rolling_hmm_params = {
    # Fix #2: Confidence filtering
    'regime_confidence_filter_pct': 0.10,  # Filter bottom 10%

    # Recommended: Increase stickiness
    'kappa': 20.0,  # Much higher than default 2.0

    # Recommended: Fewer regimes
    'n_components': 4,  # Down from 5

    # Optional: Increase regularization
    'min_covar': 0.01,  # Higher minimum covariance
}

hpo_config = {
    # Prioritize economic separation
    'weight_economic': 0.50,  # Up from 0.40
    'weight_between_within_cv': 0.30,  # Down from 0.40
    'weight_temporal': 0.20,  # Keep same
}
```

---

## Expected Results After Full Implementation

| Metric | Before | After Target |
|--------|---------|--------------|
| Total Return | -89% | -5% to +10% |
| Sharpe Ratio | -1.55 | 0.3 to 0.8 |
| CV Accuracy | 50% | 65%+ |
| CV Ratio | 1.02 | 2.0+ |
| Regime Persistence | 6.36 bars | 20+ bars |
| Turnover | 0.1131 | 0.04-0.06 |

---

## Next Steps

1. Update `hpo_config.py` with new kappa and n_components ranges
2. Re-run HPO to find new optimal parameters
3. Validate on out-of-sample data
4. Compare to previous version

Let me know if you'd like me to implement any of these changes!
