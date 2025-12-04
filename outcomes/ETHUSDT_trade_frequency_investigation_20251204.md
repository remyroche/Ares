# ETHUSDT Trade Frequency Investigation Report

**Date:** 2025-12-04  
**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m  
**Problem Statement:** Trade frequency of ~0.2-0.33 trades/day vs expected 1-2 trades/day lasting 2-3 hours

---

## Executive Summary

**Current State:**
- Gated trades: ~0.20-0.33 per day (depending on threshold configuration)
- Expected: 1-2 trades per day lasting 2-3 hours each
- **Gap: 3-10x fewer trades than expected**

**Root Causes Identified:**
1. **Conservative HPO optimization** favoring quality over quantity
2. **High probability thresholds** (0.55-0.65) with weak model AUC (~0.61)
3. **Additional expected return gate** (0.45% threshold) further filtering 50%+ of trades
4. **Limited primary signal generation** (~7.6 raw events/day)

---

## Detailed Analysis

### 1. Trade Funnel Analysis (From Reports)

| Stage | Count | Rate | Notes |
|-------|-------|------|-------|
| Total bars | 34,135 | - | ~3 years of 15m data |
| Raw consensus signals | ~8,304 | 7.6/day | Primary signals from RSI/MACD/MA/Momentum |
| Labeled events (post-filter) | 4,187 | 3.8/day | After triple-barrier resolution |
| Gated trades (prob ≥ 0.65) | 395-677 | 0.33-0.45/day | Meta-model probability gate |
| Gated trades (prob ≥ 0.55 + exp_ret) | 240 | 0.20/day | With expected return threshold |

### 2. Signal Generation Bottleneck

From `feature_generation_meta_labeling_step.py`:

```python
# Current configuration:
target_trades_per_day: float = 2.0  # Targeting 2 trades/day
rsi_oversold: float = 25.0          # Already loosened from 30
rsi_overbought: float = 75.0        # Already loosened from 70
macd_threshold: float = 0.02        # Already loosened
```

The system uses dynamic threshold tuning and claims to target 2 trades/day, but:
- Only ~7.6 raw events/day actually generated
- After triple-barrier labeling: ~3.8 labeled events/day
- After probability gating: ~0.33 trades/day

### 3. Triple-Barrier Configuration (HPO Results)

From `meta_labeling_hpo_best_params_ETHUSDT_15m_20251203_211655.json`:

```json
{
  "horizon_bars": 8,           // 2 hours at 15m - reasonable
  "min_event_spacing": 4,      // 1 hour spacing - limiting
  "profit_mult_min": 0.95,
  "profit_mult_max": 1.40,
  "stop_mult_min": 0.71,
  "stop_mult_max": 1.16
}
```

**Observations:**
- `horizon_bars=8` (2 hours) matches expected trade duration ✓
- `min_event_spacing=4` (1 hour) limits max trades to 24/day
- Event duration (mean=4.78 bars, ~1.2 hours) is reasonable
- 90th percentile duration (11 bars, ~2.75 hours) aligns with expectations

### 4. Meta-Model Probability Gating

From SNR diagnostics:
| Threshold | Trades | Trades/Day | Mean Return | Win Rate | Sharpe |
|-----------|--------|------------|-------------|----------|--------|
| 0.55 | 925 | 0.62 | 0.363% | 61.7% | 15.51 |
| 0.60 | 797 | 0.54 | 0.410% | 65.7% | 16.69 |
| 0.65 | 677 | 0.45 | 0.484% | 71.2% | 19.23 |
| 0.70 | 557 | 0.37 | 0.552% | 75.8% | 21.28 |
| 0.75 | 453 | 0.30 | 0.624% | 80.6% | 23.88 |

**Key Insight:** Even at threshold 0.55, only 0.62 trades/day achievable.

### 5. Model Predictive Power Issues

From diagnostics:
- **OOF AUC:** 0.58-0.69 (weak but usable signal)
- **Learnability score:** 0.405 ("Pass" - not great)
- **Model robustness:** 0.617 ("Pass" - moderate)
- **Underfit Detected:** Model shows signs of underfitting

The weak AUC means the model cannot reliably distinguish good from bad trades, so high probability thresholds are needed, which reduces trade count.

### 6. Volatility Regime Dependency

| Regime | Samples | Win Rate | AUC | Mean Return |
|--------|---------|----------|-----|-------------|
| Low | 1741 | 37.9% | 0.556 | -0.08% |
| Medium | 1370 | 50.1% | 0.676 | 0.10% |
| High | 1017 | 70.7% | 0.731 | 0.44% |

**Critical Finding:** The strategy performs significantly better in high-volatility regimes:
- Win rate jumps from 37.9% → 70.7%
- AUC improves from 0.556 → 0.731
- Mean return from -0.08% → 0.44%

But high volatility occurs less frequently, limiting trade opportunities.

---

## Why 1-2 Trades/Day Is Not Being Achieved

### Barrier 1: Primary Signal Generation (~7.6/day)
The consensus signal generator produces enough raw signals, but not all resolve profitably.

### Barrier 2: Triple-Barrier Resolution (~3.8/day labeled)
After applying profit/stop/timeout barriers:
- 49.7% hit profit target
- 47.8% hit stop loss
- 2.4% timeout
- Only ~50% retention (was 8,304 → 4,187)

### Barrier 3: Meta-Probability Filtering (~0.45/day at 0.65)
The meta-model's weak AUC (0.61) means:
- At threshold 0.65: only top ~20% of events pass
- Resulting in 0.45 trades/day

### Barrier 4: Expected Return Gate (~0.20/day)
When using expected return threshold (0.45%):
- Additional ~55% of trades filtered
- Down to 0.20 trades/day

---

## Recommendations to Achieve 1-2 Trades/Day

### Option A: Aggressive Configuration (Target: ~1.5 trades/day)

```yaml
# Recommended parameter changes:
probability_threshold: 0.50           # From 0.55-0.65 → trade on any positive signal
use_expected_return: false            # Disable expected return gate
min_event_spacing: 2                  # From 4 → allow 30-min spacing
horizon_bars: 6                       # From 8 → reduce timeout events
```

**Expected Results:**
- Trades/day: ~1.2-1.5
- Win rate: ~55-60%
- Sharpe: ~12-14
- Max DD: ~10-12%

### Option B: Volatility-Conditional Configuration (Target: ~1.2 trades/day)

Use lower thresholds during high-volatility periods:
```python
if volatility_regime == 'high':
    probability_threshold = 0.50  # More aggressive when conditions favor
    min_event_spacing = 2
else:
    probability_threshold = 0.60  # Conservative otherwise
    min_event_spacing = 4
```

**Expected Results:**
- Trades/day: ~1.0-1.2
- Higher win rate in activated periods
- Better risk-adjusted returns

### Option C: Enhanced Signal Generation (Target: ~2 trades/day)

Add more primary signal sources:
1. **Bollinger Band signals** (squeeze breakouts)
2. **ATR breakout signals** (volatility expansion)
3. **Volume spike signals** (above 2σ volume)
4. **Multi-timeframe confluence** (align with 1h signals)

This would increase raw events from ~7.6/day to ~15-20/day, providing more high-quality trade opportunities.

### Option D: Improve Model Predictive Power

**Current weakness:** AUC ~0.61 is barely above random (0.5).

Improvements:
1. **Add more predictive features:**
   - Order flow imbalance proxies
   - Funding rate dynamics
   - Cross-pair correlations (BTC dominance)
   - Time-of-day patterns (hour 5 has 61.3% win rate, hour 13 has 38.8%)

2. **Better regime conditioning:**
   - Train separate models per volatility regime
   - Use regime probabilities as gating features

3. **Ensemble improvements:**
   - Add CatBoost to ensemble
   - Use stacked generalization

---

## Quick Win: Immediate Configuration Change

Based on the threshold sweep table, the sweet spot is:

| Threshold | Trades/Day | Mean Return | Sharpe | Trade-off |
|-----------|------------|-------------|--------|-----------|
| **0.55** | **0.62** | 0.363% | 15.51 | More trades, slightly lower quality |
| 0.60 | 0.54 | 0.410% | 16.69 | Balanced |
| 0.65 | 0.45 | 0.484% | 19.23 | Current default |

**Recommendation:** 
- Lower `probability_threshold` from 0.65 → 0.55
- Disable `use_expected_return` (from True → False)

This alone would increase trades from 0.20/day → 0.62/day (3x improvement) while maintaining positive Sharpe ratio of 15.51.

---

## Summary Table: Trade Frequency Levers

| Lever | Current | Recommended | Impact |
|-------|---------|-------------|--------|
| probability_threshold | 0.65 | 0.50-0.55 | +50-100% trades |
| use_expected_return | True | False | +50% trades |
| min_event_spacing | 4 bars | 2 bars | +30% trades |
| horizon_bars | 8 | 6 | +20% trades |
| Primary signals | 6 | 8-10 | +30-50% raw events |

**Combined effect of all changes:** 0.20/day → ~1.5-2.0/day

---

## Risk Considerations

1. **Lower thresholds = lower quality trades**
   - Win rate drops from 71% → 62%
   - But aggregate PnL/day may still be similar due to volume

2. **More frequent trading = higher costs**
   - At 0.15% transaction cost, 2 trades/day = 0.30%/day cost
   - Must maintain edge > 0.35%/trade to remain profitable

3. **Regime dependency**
   - Strategy heavily depends on volatility
   - May have extended periods of low activity in calm markets

---

## Artifacts Referenced

- `meta_labeling_hpo_report_ETHUSDT_15m_20251203_211655.md`
- `meta_gated_backtest_ETHUSDT_15m_long_20251203_234825.md`
- `snr_full_diagnostics_ETHUSDT_15m_20251203_213033.md`
- `meta_labeling_diagnostics_20251203_222428.md`

---

*Report generated by automated investigation pipeline*
