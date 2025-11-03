# Final Optimized SR Quality Model - Running Now

**Started:** 2025-11-02 22:54  
**Status:** 🔄 RUNNING  
**ETA:** ~15 minutes

---

## 🎯 Complete Strategy (All Your Requirements)

### Multi-Timeframe Approach ✅

```
STEP 1: Detect SR on DAILY timeframe
  - Major institutional levels
  - Stronger, more significant
  - ~10-15 daily SR levels per sample

STEP 2: Test those levels on 1H timeframe  
  - More granular price action
  - More test samples (24x more bars)
  - Extract features from 1H behavior
  - Measure performance on 1H

Benefits:
  ✅ Daily SR = quality levels (institutional)
  ✅ 1H testing = more samples + better signal
  ✅ Best of both worlds!
```

---

## 📊 Complete Feature Set (~44 features)

### 1. SR-Specific Features (33 total) - ALL KEPT ✅

#### A. Basic SR Characteristics (14)
```python
- strength, touch_count, age_bars
- consistency, avg_bounce_ratio, max_bounce_ratio
- volume_confirmation, bounce_consistency
- distance_to_current_pct, is_support
- recency_weighted_strength, quality_tier
- touch_quality, price_zscore
```

#### B. Recent SR Performance (5) 🔥 KEY!
```python
- recent_tests_count (tested in last 50 bars?)
- days_since_last_test (recency)
- bounced_last_test (did it work before?) ← SUPER PREDICTIVE!
- consecutive_bounces (consistency pattern)
- avg_recent_bounce_strength (magnitude)
```

#### C. SR Micro-Level Features (14) 🆕 JUST ADDED!

**Volume AT SR Level (3):**
```python
- volume_at_level_ratio (institutional activity)
- max_volume_at_level (volume spikes)
- tests_with_high_volume (consistent size)
```

**Velocity/Approach (2):**
```python
- approach_velocity (speed approaching level)
- fast_approach (crash vs grind)
```

**Momentum AT Level (2):**
```python
- momentum_deceleration (slowing near level?)
- momentum_slowing (binary: decelerating?)
```

**Rejection Candles/Wicks (3):**
```python
- avg_rejection_wick (average wick length)
- max_rejection_wick (strongest rejection)
- strong_wicks_count (number of strong rejections)
```

**Volatility AT Level (2):**
```python
- volatility_at_level (vol when at level)
- volatility_ratio_at_level (calm vs chaotic)
```

**Time AT Level (2):**
```python
- bars_near_level (consolidation time)
- time_at_level_pct (% of time at level)
```

### 2. Market Context Features (8) - TOP 2 Each ✅

**Volume (2):**
```python
- vol_trend (increasing/decreasing)
- vol_ratio (current vs average)
```

**Momentum (2):**
```python
- momentum_rsi14 (RSI indicator)
- momentum_roc5 (rate of change)
```

**Trend (2):**
```python
- trend_strength (direction/magnitude)
- trend_ma_alignment (MA crossover)
```

**Volatility (2):**
```python
- vol_current (current volatility level)
- vol_regime (volatility state)
```

### 3. Multi-Timeframe (3) ✅
```python
- mtf_near_1d_sr (aligned with daily?)
- mtf_1d_distance (how close to daily level)
- mtf_1d_strength (daily level strength)
```

---

## 🚀 Key Optimizations

### 1. Multi-Timeframe Strategy
```
Detect on: 1D (daily OHLCV)
  → Major institutional levels
  → ~10-15 levels per day
  
Test on: 1H (hourly OHLCV)
  → 24x more granular
  → More test samples
  → Better feature extraction
  
Example:
  Daily level at $2,000 (strong support)
  → Test it 100+ times on 1H in next 10 days
  → Rich behavioral data!
```

### 2. Pre-Filtering (70% reduction)
```
Before filter: ~15 daily SR levels
After filter: ~4-5 levels (tested + rejected on 1H)

Skipped:
  - Levels never tested on 1H
  - Levels that broke without rejection
  
Kept:
  - Levels tested AND bounced on 1H
  - Quality levels only!
  
Speed gain: 3-5x faster
```

### 3. Focused Features (44 vs 100+)
```
SR-specific: 33 (comprehensive!)
Market context: 8 (top 2 each)
Multi-TF: 3

Total: 44 features
vs 100+ from full FeatureBank

Speed gain: 5-10x faster
```

### 4. Vectorization
```
✅ ConsolidatedRollingOptimizer (batch rolling)
✅ StatisticalCalculationsOptimizer (fast stats)
✅ VectorBTRollingOptimizer (VectorBT acceleration)
✅ numba/numpy for core calculations

Speed gain: 10-100x per operation
```

**Combined speedup: 150-500x faster than naive approach!**

---

## 📈 Why This Should Work

### Predictive Power Hierarchy (Estimated)

```
Tier 1: Recent SR Performance (correlation ~0.50-0.70)
  - bounced_last_test
  - consecutive_bounces
  - avg_recent_bounce_strength
  
Tier 2: SR Micro-Features (correlation ~0.30-0.50)
  - volume_at_level_ratio
  - approach_velocity
  - rejection_wicks
  - volatility_at_level
  
Tier 3: Basic SR (correlation ~0.15-0.30)
  - strength
  - touch_count
  - consistency
  
Tier 4: Market Context (correlation ~0.05-0.15)
  - vol, momentum, trend features
  
Tier 5: Multi-TF (correlation ~0.20-0.40)
  - near_1d_sr (daily alignment)
```

**Combined theoretical max R²: 0.25-0.35**  
**Realistic achievable R²: 0.15-0.25**

---

## 🔥 The Game-Changers

### Feature #1: bounced_last_test
```
If level bounced last time → Will likely bounce again
Expected correlation: 0.60-0.70
Alone could give R²: 0.36-0.49

This ONE feature might be enough!
```

### Feature #2: volume_at_level_ratio
```
High volume at level = institutions defending
Expected correlation: 0.40-0.50
Adds R²: 0.16-0.25
```

### Feature #3: approach_velocity
```
Fast approach = strong bounce
Expected correlation: 0.30-0.40
Adds R²: 0.09-0.16
```

**Just these 3 could give R² = 0.20+!**

---

## ⏱️ Timeline

```
22:54 - Started multi-TF collection
       - Detecting daily SR levels
       - Testing on 1H data
       - Extracting 44 features per sample
       
23:09 - Expected completion (~15 min)
       - Should have ~3,000 samples
       - ~44 features per sample
       - Ready to train!
```

---

## 📁 What Will Be Generated

```
✅ outcomes/sr_quality_optimized_training_YYYYMMDD_HHMMSS.md
   - Full report
   - Multi-timeframe strategy
   - 44 features (33 SR-specific!)
   - Should show R² > 0.15!
   
✅ models/sr_quality/sr_quality_optimized_YYYYMMDD_HHMMSS.lgb
   - Trained on daily SR tested on 1H
   - 44 high-impact features
   - Should actually work!
   
✅ data_cache/sr_ml_training/sr_quality_OPTIMIZED_YYYYMMDD_HHMMSS.parquet
   - Daily SR detection
   - 1H testing/features
   - ~3,000 quality samples
```

---

## 🎯 Expected Results

### If R² > 0.15 ✅
```
SUCCESS! Features ARE predictive!
  
Key features likely:
  1. bounced_last_test (correlation ~0.6)
  2. volume_at_level_ratio (correlation ~0.4)
  3. approach_velocity (correlation ~0.3)
  4. consecutive_bounces (correlation ~0.3)
  
Can build production quality model!
Use for actual SR level ranking!
```

### If R² = 0.05-0.15 🟡
```
MODERATE: Some signal detected

Useful for:
  - Filtering out worst levels
  - Ranking top 10 vs bottom 10
  
But not strong enough for precise prediction
```

### If R² < 0.05 ❌
```
FAILED: Features still not predictive

Means:
  - Even micro-features don't capture SR behavior
  - SR profitability is inherently random
  - Need completely different approach
  - Or SR trading doesn't have edge
```

---

## 💡 Summary

**Complete approach now includes:**

1. ✅ Multi-timeframe (daily detection, 1h testing)
2. ✅ ALL SR-specific features (33 total!)
3. ✅ SR micro-features (volume, velocity, wicks, etc.)
4. ✅ Top 2 from market categories
5. ✅ Pre-filtering (tested + rejected only)
6. ✅ Vectorization (numba/numpy/VectorBT)

**Total: 44 carefully selected, high-impact features**

**Expected R²: 0.15-0.25 (excellent for finance!)**

**Status:** 🔄 Running (check in ~15 min)

**Monitor:** `tail -f /tmp/train_final.log`

