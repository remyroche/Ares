# Low-Volatility Signal Enhancement Proposal

**Problem:** The current system struggles in calm markets because:
1. Volatility weighting **dampens** signals when vol is low
2. All signal sources are **momentum-based** (trend-following)
3. No **mean-reversion** signals exist for range-bound conditions

## Current Behavior (Why Low-Vol Fails)

```python
# Current code (line 709):
vol_weight = (vol_short / vol_long).clip(0.5, 2.0)
```

| Market Condition | vol_short/vol_long | Weight | Effect |
|------------------|-------------------|--------|--------|
| High volatility | > 1.0 | 1.0-2.0 | Signals boosted ✓ |
| Normal volatility | ≈ 1.0 | 1.0 | No change |
| **Low volatility** | < 1.0 | **0.5-1.0** | **Signals dampened ✗** |

This is **backwards** for low-vol trading. In calm markets:
- Price tends to **mean-revert** (not trend)
- Momentum signals are **wrong** (they expect trends)
- We need to **fade extremes**, not follow momentum

---

## Proposed Solution: Vol-Aware Dual-Mode Signal Generation

### 1. Add Mean-Reversion Signals

Add these to `generate_primary_signals()`:

```python
# ===== MEAN-REVERSION SIGNALS (FOR LOW-VOL REGIMES) =====

# Bollinger Band fade signals
bb_window = 20
bb_std = 2.0
bb_mid = df_local['close'].rolling(bb_window).mean()
bb_upper = bb_mid + bb_std * df_local['close'].rolling(bb_window).std()
bb_lower = bb_mid - bb_std * df_local['close'].rolling(bb_window).std()

signals['bb_fade'] = 0
# Long when price touches lower band (fade to mean)
signals.loc[df_local['close'] <= bb_lower, 'bb_fade'] = 1
# Short when price touches upper band (fade to mean)  
signals.loc[df_local['close'] >= bb_upper, 'bb_fade'] = -1

# Range fade signals (mean-reversion at range extremes)
range_high = df_local['high'].rolling(48).max()  # 12-hour range
range_low = df_local['low'].rolling(48).min()
range_mid = (range_high + range_low) / 2
range_position = (df_local['close'] - range_low) / (range_high - range_low + 1e-8)

signals['range_fade'] = 0
# Long at bottom of range
signals.loc[range_position < 0.15, 'range_fade'] = 1
# Short at top of range
signals.loc[range_position > 0.85, 'range_fade'] = -1

# RSI mean-reversion (tighter thresholds for low-vol)
signals['rsi_mr'] = 0
signals.loc[df_local['rsi'] < 35, 'rsi_mr'] = 1   # Less extreme than momentum RSI
signals.loc[df_local['rsi'] > 65, 'rsi_mr'] = -1

# Price-to-VWAP reversion (if available)
if 'vwap' in df_local.columns:
    vwap_dist = (df_local['close'] - df_local['vwap']) / df_local['vwap']
    signals['vwap_fade'] = 0
    signals.loc[vwap_dist < -0.005, 'vwap_fade'] = 1   # Below VWAP → long
    signals.loc[vwap_dist > 0.005, 'vwap_fade'] = -1   # Above VWAP → short
```

### 2. Vol-Aware Consensus Logic

Replace the current simple weighting with regime-conditional consensus:

```python
# ===== VOL-AWARE DUAL-MODE CONSENSUS =====

# Momentum signals (for high vol)
momentum_cols = ['rsi', 'macd', 'ma', 'mom']
momentum_score = signals[momentum_cols].sum(axis=1)

# Mean-reversion signals (for low vol)  
mr_cols = ['bb_fade', 'range_fade', 'rsi_mr']
mr_score = signals[mr_cols].sum(axis=1)

# Volatility regime detection
vol_ratio = vol_short / (vol_long + 1e-8)
is_low_vol = vol_ratio < 0.8
is_high_vol = vol_ratio > 1.2

# Blend signals based on regime
# In low vol: weight mean-reversion higher
# In high vol: weight momentum higher
# In normal: equal weight

low_vol_weight = np.where(is_low_vol, 0.3, 0.7)   # Momentum weight (low in calm)
high_vol_weight = np.where(is_high_vol, 0.9, 0.7)  # Momentum weight (high in volatile)

momentum_weight = np.where(is_low_vol, 0.3, np.where(is_high_vol, 0.9, 0.6))
mr_weight = 1.0 - momentum_weight

# Weighted consensus
weighted_score = momentum_weight * momentum_score + mr_weight * mr_score
signals['consensus'] = np.sign(weighted_score)

# Store regime info for diagnostics
signals['vol_regime'] = np.where(is_low_vol, 'low', np.where(is_high_vol, 'high', 'normal'))
signals['momentum_weight_used'] = momentum_weight
```

### 3. Vol-Adjusted Triple Barrier Geometry

Different profit/stop targets for different regimes:

```python
# In compute_realized_returns():

# Current adaptive thresholds already scale with volatility, but could be more aggressive:

# For LOW VOL: Smaller targets (price moves less), shorter horizon
# For HIGH VOL: Larger targets (price moves more), longer horizon

if vol_regime == 'low':
    # Tighter targets for mean-reversion
    effective_profit = profit_threshold * 0.6  # Smaller profit target
    effective_stop = stop_threshold * 0.8      # Tighter stop
    effective_horizon = horizon * 0.7          # Shorter horizon (quick MR)
else:
    # Standard momentum targets
    effective_profit = profit_threshold
    effective_stop = stop_threshold
    effective_horizon = horizon
```

---

## Implementation Plan

### Phase 1: Add Mean-Reversion Signals (Low Risk)
1. Add `bb_fade`, `range_fade`, `rsi_mr` to `generate_primary_signals()`
2. Include them in signal_cols for consensus
3. **No change** to weighting logic yet

**Expected Impact:** 
- +20-30% more raw signals
- Signals in previously "quiet" periods

### Phase 2: Vol-Aware Consensus (Medium Risk)
1. Replace flat vol_weight with regime-conditional weighting
2. Boost mean-reversion signals in low-vol
3. Maintain momentum dominance in high-vol

**Expected Impact:**
- Better signal quality in low-vol regimes
- AUC improvement in low-vol bucket (currently 0.556 → target 0.62+)

### Phase 3: Regime-Specific Labeling (Higher Risk)
1. Different triple-barrier geometry per regime
2. Smaller targets in low-vol
3. May require HPO re-run

**Expected Impact:**
- Better calibrated win rates across regimes
- Reduce the 37.9% → 70.7% win rate gap

---

## Code Changes Required

### File: `src/training/steps/labeling/feature_generation_meta_labeling_step.py`

#### 1. In `generate_primary_signals()` (around line 615):

Add after the MACD signals section:

```python
# ===== MEAN-REVERSION SIGNALS (FOR LOW-VOL CONDITIONS) =====

# Bollinger Band fade
bb_window = 20
bb_mid = df_local['close'].rolling(bb_window).mean()
bb_std_series = df_local['close'].rolling(bb_window).std()
bb_upper = bb_mid + 2.0 * bb_std_series
bb_lower = bb_mid - 2.0 * bb_std_series

signals['bb_fade'] = 0
signals.loc[df_local['close'] <= bb_lower, 'bb_fade'] = 1
signals.loc[df_local['close'] >= bb_upper, 'bb_fade'] = -1

# Range fade (48-bar range = 12 hours on 15m)
range_high = df_local['high'].rolling(48).max()
range_low = df_local['low'].rolling(48).min()
range_pos = (df_local['close'] - range_low) / (range_high - range_low + 1e-8)

signals['range_fade'] = 0
signals.loc[range_pos < 0.15, 'range_fade'] = 1
signals.loc[range_pos > 0.85, 'range_fade'] = -1

# RSI mean-reversion (tighter thresholds)
signals['rsi_mr'] = 0
signals.loc[df_local['rsi'] < 35, 'rsi_mr'] = 1
signals.loc[df_local['rsi'] > 65, 'rsi_mr'] = -1
```

#### 2. Replace consensus logic (around line 698):

```python
# ===== VOL-AWARE DUAL-MODE CONSENSUS =====

# Separate signal types
momentum_cols = ['rsi', 'rsi_long', 'macd', 'macd_long', 'ma', 'mom']
mr_cols = ['bb_fade', 'range_fade', 'rsi_mr']

momentum_score = signals[momentum_cols].sum(axis=1)
mr_score = signals[mr_cols].sum(axis=1)

# Regime detection
vol_ratio = vol_short / (vol_long + 1e-8)
is_low_vol = (vol_ratio < 0.8).astype(float)
is_high_vol = (vol_ratio > 1.2).astype(float)
is_normal = 1.0 - is_low_vol - is_high_vol

# Regime-conditional weights
# Low vol: 30% momentum, 70% mean-reversion
# High vol: 85% momentum, 15% mean-reversion  
# Normal: 60% momentum, 40% mean-reversion
mom_weight = is_low_vol * 0.30 + is_high_vol * 0.85 + is_normal * 0.60
mr_weight = 1.0 - mom_weight

# Weighted consensus
raw_consensus = signals[momentum_cols + mr_cols].sum(axis=1).apply(np.sign)
funnel['raw_signals'] = (raw_consensus != 0).sum()

weighted_score = mom_weight * momentum_score + mr_weight * mr_score
signals['consensus'] = weighted_score.apply(np.sign)

# Store diagnostic info
signals['momentum_weight'] = mom_weight
signals['mr_weight'] = mr_weight
signals['vol_ratio_signal'] = vol_ratio
```

---

## Expected Results

| Regime | Current Win Rate | Current AUC | Target Win Rate | Target AUC |
|--------|-----------------|-------------|-----------------|------------|
| Low | 37.9% | 0.556 | 48-52% | 0.60+ |
| Medium | 50.1% | 0.676 | 52-55% | 0.68+ |
| High | 70.7% | 0.731 | 70-72% | 0.73+ |

**Trade Frequency Impact:**
- Current: ~0.45 trades/day (at threshold 0.65)
- Expected: ~0.7-0.9 trades/day (more signals in low-vol periods)

---

## Risk Mitigation

1. **A/B Test First**: Run the new signal generation alongside old, compare OOF metrics
2. **Feature Not Gate**: Add mean-reversion signals as features, not hard gates
3. **Gradual Rollout**: Start with Phase 1 only, validate, then Phase 2
4. **HPO Re-validation**: After changes, run HPO to find new optimal thresholds

---

## Validation Metrics to Track

1. **Per-regime AUC** (currently in SNR diagnostics)
2. **Per-regime trade count**
3. **Per-regime Sharpe ratio**
4. **Signal type attribution** (what % of trades came from MR vs momentum signals)
5. **Holding period distribution by regime**
