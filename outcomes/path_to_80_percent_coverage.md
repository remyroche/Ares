# Path to 80-90% Regime Coverage

## Current State: Multi-Dimensional Analysis

Your diagnostics reveal **5 conditioning dimensions** that affect profitability:

### 1. Volatility Regime
| Regime | Win Rate | Gap from 50% |
|--------|----------|--------------|
| High | 70.7% | +20.7% ✓ |
| Medium | 50.1% | +0.1% |
| Low | 37.9% | -12.1% ✗ |

### 2. Time of Day (UTC)
| Hours | Win Rate | Gap from 50% |
|-------|----------|--------------|
| Best (3,5,10) | 59.0% avg | +9.0% ✓ |
| Worst (0,13,19) | 42.3% avg | -7.7% ✗ |

**22.5 percentage point spread** - this is huge unexploited alpha!

### 3. Day of Week
| Day | Win Rate | Gap from 50% |
|-----|----------|--------------|
| Thursday (3) | 55.2% | +5.2% ✓ |
| Sunday (6) | 39.5% | -10.5% ✗ |

### 4. Trend State
| State | Win Rate | Gap from 50% |
|-------|----------|--------------|
| Strong uptrend | 51.2% | +1.2% |
| Strong downtrend | 45.8% | -4.2% |

Trend alone is weak signal.

### 5. Year (Macro Regime)
| Year | Win Rate | AUC | Notes |
|------|----------|-----|-------|
| 2024 | **72.2%** | **0.854** | Bull market, high vol |
| 2022 | 50.7% | 0.689 | Bear market, high vol |
| 2023 | 46.2% | 0.718 | Low vol, ranging |
| 2025 | 44.1% | 0.738 | Current |
| 2021 | 47.0% | 0.523 | Late bull |

**2024 achieved 72.2% win rate** - proof that 80% is reachable under right conditions!

---

## The Math: How to Get to 80%

### Current Coverage (Single Dimension)
- High vol alone: 70.7% → covers ~30% of time
- Best hours alone: 59.0% → covers ~12.5% of time (3 hours / 24)
- Best day alone: 55.2% → covers 14.3% of time

**Single dimensions don't get us there.**

### Cross-Conditioning (Multiple Dimensions)

If we require **High Vol + Best Hours**:
- P(High Vol) ≈ 30%
- P(Best Hours) ≈ 12.5%
- P(High Vol ∩ Best Hours) ≈ 3.75% of bars

But win rate would be approximately:
- Base (High Vol): 70.7%
- Hour boost: +9% (from 59% vs 50% baseline)
- **Combined estimate: ~75-78%**

Still not 80%. We need **additional edges**.

---

## What's Missing for 80-90%

### Gap 1: External Data (Not Currently Used)

Your system has **zero** external data features:

| Data Source | Predictive Value | Currently Used |
|-------------|------------------|----------------|
| Funding Rate | High (sentiment) | ❌ No |
| Open Interest | High (positioning) | ❌ No |
| Liquidation Cascades | Very High | ❌ No |
| BTC Dominance | Medium (rotation) | ❌ No |
| Bid-Ask Spread | Medium (liquidity) | ❌ No |
| Order Book Imbalance | High | ❌ No |
| CVD (Cumulative Volume Delta) | High | ❌ No |

**Each of these could add 3-8% edge in specific conditions.**

### Gap 2: Negative Selection (When NOT to Trade)

Current system tries to find good trades. It should also **explicitly avoid bad conditions**:

```
Avoid signals when:
- Sunday (39.5% win rate)
- Hour 13 (38.8% win rate)  
- Low vol + downtrend combination
- Post-major-news volatility spikes (first 2 bars)
- Funding rate extreme (>0.1% or <-0.1%)
```

**Negative selection could eliminate 20-30% of losing trades.**

### Gap 3: Specialist Models per Regime

Current: One model for all conditions.

Better: **Ensemble of specialists**:
- `model_high_vol_momentum`: Trend-following for expansion
- `model_low_vol_meanreversion`: Range-fading for compression
- `model_transition_detector`: Catch regime changes early
- `model_time_conditional`: Hour/day-specific patterns

Each specialist could achieve 70%+ in its domain.

### Gap 4: Cross-Asset Signals

ETH doesn't move in isolation:

| Signal | Use Case |
|--------|----------|
| BTC lead/lag | BTC often leads ETH by 1-5 bars |
| ETH/BTC ratio momentum | Rotation signal |
| DXY / macro correlation | Risk-on/off |
| Altcoin breadth | Market health |

---

## Concrete Roadmap to 80%

### Phase 1: Multi-Dimensional Conditioning (60% → 70%)
*Estimated lift: +10% coverage*

```python
# Combine existing dimensions:
is_good_setup = (
    (vol_regime == 'high') |  # OR
    (vol_regime == 'medium' & hour in [3,5,10]) |  # AND
    (vol_regime == 'low' & use_mean_reversion_signals)
)

is_bad_setup = (
    (day_of_week == 6) |  # Sunday
    (hour == 13) |  # Worst hour
    (vol_regime == 'low' & trend_strength > 0.5)  # Low vol + trending = bad
)

# Only trade when good and not bad
trade_allowed = is_good_setup & ~is_bad_setup
```

### Phase 2: Add Funding Rate & OI (70% → 75%)
*Estimated lift: +5% coverage*

```python
# Fetch from exchange API:
funding_rate = get_funding_rate('ETHUSDT')
open_interest = get_open_interest('ETHUSDT')
oi_change_1h = open_interest.pct_change(4)  # 4 bars = 1h

# Signals:
# - Extreme negative funding + price near support = long
# - Rising OI + rising price = momentum continuation
# - Falling OI + price rise = distribution (bearish)
```

**Implementation:** Add to data aggregation pipeline.

### Phase 3: Order Flow Approximation (75% → 78%)
*Estimated lift: +3% coverage*

Without direct order book access, approximate:
```python
# CVD proxy (Cumulative Volume Delta)
cvd = (volume * np.where(close > open, 1, -1)).cumsum()
cvd_divergence = (price_trend > 0) & (cvd_trend < 0)  # Bearish divergence

# Volume at price extremes
vol_at_high = volume.where(close == high).rolling(20).sum()
vol_at_low = volume.where(close == low).rolling(20).sum()
absorption_ratio = vol_at_low / (vol_at_high + 1e-8)
```

### Phase 4: Specialist Ensemble (78% → 82%)
*Estimated lift: +4% coverage*

```python
# Train separate models:
model_high_vol = train_on_subset(data[vol_regime == 'high'])
model_low_vol = train_on_subset(data[vol_regime == 'low'])
model_transition = train_on_subset(data[vol_changing])

# Route predictions:
if vol_regime == 'high':
    prediction = model_high_vol.predict(features)
elif vol_regime == 'low':
    prediction = model_low_vol.predict(features)
else:
    prediction = weighted_average([model_high_vol, model_low_vol])
```

### Phase 5: Cross-Asset Integration (82% → 85%)
*Estimated lift: +3% coverage*

```python
# BTC lead signal
btc_return_lag1 = btc_close.pct_change().shift(1)
eth_predicted_direction = np.sign(btc_return_lag1)

# ETH/BTC rotation
eth_btc_ratio = eth_close / btc_close
ratio_zscore = (eth_btc_ratio - ratio_ma) / ratio_std
eth_outperform_signal = ratio_zscore < -2  # ETH underperforming, mean-revert
```

### Phase 6: Negative Selection Gate (85% → 88%)
*Estimated lift: +3% coverage from avoiding losses*

```python
# Hard filters (never trade):
NEVER_TRADE = (
    (day_of_week == 6) &  # Sunday
    (hour in [12, 13, 14]) &  # Lunch hours
    (funding_rate.abs() > 0.1)  # Extreme funding
)

# Soft filters (reduce position size):
REDUCE_SIZE = (
    (vol_regime == 'low') |
    (hour in [0, 19]) |
    (oi_change_1h < -5%)  # Deleveraging
)
```

---

## Summary: Path to 80%+

| Phase | Change | Coverage | Win Rate |
|-------|--------|----------|----------|
| Current | Baseline | ~24% | 50.0% avg |
| Phase 1 | Multi-dim conditioning | ~40% | 60%+ |
| Phase 2 | Funding + OI | ~50% | 65%+ |
| Phase 3 | Order flow proxy | ~55% | 68%+ |
| Phase 4 | Specialist ensemble | ~65% | 72%+ |
| Phase 5 | Cross-asset | ~75% | 76%+ |
| Phase 6 | Negative selection | ~85% | 80%+ |

**Key Insight:** The path to 80% is not about finding a magic signal. It's about:
1. **Combining multiple weak signals** (stacking edges)
2. **Avoiding known bad conditions** (negative selection)
3. **Using external data** (funding, OI, cross-asset)
4. **Specialization** (different strategies for different regimes)

---

## Quick Wins (Implementable This Week)

1. **Add time-of-day filter:**
   ```python
   # In meta_gating_config:
   avoid_hours: [12, 13, 14, 0]
   avoid_days: [6]  # Sunday
   ```
   Expected impact: +3-5% win rate by avoiding worst times

2. **Implement mean-reversion signals** (from previous proposal)
   Expected impact: Low-vol win rate 37.9% → 48%+

3. **Add funding rate feature** (if API available)
   Expected impact: +2-3% edge in extreme funding conditions

---

## The 90% Question

To get to **90%**, you'd need:
- Real-time order flow data (not just OHLCV)
- Sub-minute execution capability
- Market making component (capture spread)
- Or: Accept fewer trades at much higher quality

90% coverage with profitable trades is possible but requires **institutional-grade infrastructure**.

**Realistic target: 75-82%** with the improvements above.
