# Enhanced Features Collection - In Progress

**Started:** 2025-11-02 21:32  
**Status:** 🔄 RUNNING (processing 365 dates with FeatureBank integration)

---

## 🚀 What's Different Now (Your Requested Changes)

### Changes Implemented

**1️⃣ Daily Sampling (not weekly)** ✅
```
sample_freq_days = 1  # Was 7
→ 365 sample dates instead of 7
→ 50x more samples
```

**2️⃣ Full Year (not 6 weeks)** ✅
```
start: 2023-01-01
end: 2023-12-31
→ 365 days instead of 42
→ 8.7x longer period
```

**3️⃣ ENHANCED FEATURES from FeatureBank** ✅ NEW!
```
Features: 100+ (instead of 19!)

Categories added:
  1. SR-specific features (FeatureBank support_resistance)
  2. Market regime features (volatility + trend states)
  3. Price action features (momentum, candlestick patterns)
  4. Multi-timeframe features (1D SR on 1h)
  5. Recent SR performance (bounced_last_test, etc.)
```

---

## 📊 Expected Results

### Feature Count Explosion

```
Before: 19 features
  - Basic SR (strength, touch_count, age, etc.)
  - Simple market context (volatility, trend)

After: 100+ features (estimated)
  - All basic features (19)
  - SR-specific from FeatureBank (20-30)
  - Regime features (20-30)
  - Price action features (20-30)
  - Multi-timeframe (3-5)
  - Recent performance (3-5)
```

### Why This Should Work

**Before (failed):**
- 19 basic features
- Max correlation: 0.336
- Max theoretical R²: 0.1127 (11%)
- Actual R²: -0.002 (useless)

**After (expected):**
- 100+ enhanced features
- Should include features with correlation > 0.5
- Max theoretical R²: hopefully 0.25-0.35 (25-35%)
- Actual R²: hopefully 0.15-0.25 (USEFUL!)

---

## 🎯 Key Features Added

### 1. SR-Specific (FeatureBank)

**From `support_resistance.py`:**
```python
- sr_level (advanced SR detection)
- sr_strength (SR level strength)
- sr_distance (distance to SR)
- volume_sr (volume-weighted SR)
- pivot_sr (pivot-based SR)
- fibonacci_sr (Fibonacci levels)
```

**Why powerful:** These are specifically designed to measure SR quality!

### 2. Market Regime (FeatureBank)

**From `regime_features.py`:**
```python
# Volatility regime
- vol_persistence
- vol_regime_strength
- vol_clustering
- vol_stability

# Trend regime  
- trend_direction_consistency
- trend_regime_persistence
- structural_trend_strength
- trend_acceleration
```

**Why powerful:** SR levels perform differently in different regimes!

### 3. Price Action (FeatureBank)

**From momentum + candlestick categories:**
```python
# Momentum
- RSI, MACD, Stochastic
- Momentum oscillators
- Trend strength indicators

# Candlestick patterns
- Bullish/bearish patterns
- Reversal patterns
- Continuation patterns
```

**Why powerful:** Price action context matters for SR performance!

### 4. Multi-Timeframe (Custom)

**New implementation:**
```python
- feature_mtf_aligned_with_1d (near daily SR?)
- feature_mtf_1d_distance_pct (how far from 1D level)
- feature_mtf_1d_strength (strength of 1D level)
```

**Why powerful:** Levels that align across timeframes are stronger!

### 5. Recent SR Performance (Custom - MOST IMPORTANT!)

**New implementation:**
```python
- feature_recent_tests_count (tested in last 50 bars?)
- feature_days_since_last_test (recency)
- feature_bounced_last_test (did it work before?)
```

**Why powerful:** Past performance is often the best predictor!

---

## 📈 Expected Improvement

### Theoretical Maximum R²

**With basic features only:**
- Best correlation: 0.336
- Max R²: 0.1127 (11%)

**With enhanced features:**
- Expected best correlation: 0.50-0.60
- Expected max R²: 0.25-0.36 (25-36%)
- **2-3x improvement in ceiling!**

### Realistic Outcome

If features are predictive:
- **R² = 0.15-0.25** (good for financial data!)
- **Better than random**
- **Can build profitable strategy**

If features still don't work:
- R² < 0.05
- → Need completely different approach
- → Maybe SR levels just don't have edge

---

## ⏱️ Timeline

```
21:32 - Started collection
21:35 - Processing dates with FeatureBank
21:47 - Expected completion (~15 min)
```

---

## 📝 Monitor Progress

```bash
# Watch live
tail -f /tmp/train_enhanced.log

# Check feature count
grep "Total features" /tmp/train_enhanced.log

# Check R² when done
grep "Avg Val R²" /tmp/train_enhanced.log
```

---

## 🎯 What Will Be Generated

**When complete (~21:47):**

```
✅ outcomes/sr_quality_enhanced_training_YYYYMMDD_HHMMSS.md
   - Full year, daily sampling
   - 9,000+ samples
   - 100+ enhanced features
   - Hopefully R² > 0.10!

✅ models/sr_quality/sr_quality_enhanced_YYYYMMDD_HHMMSS.lgb
   - Trained with FeatureBank features
   - Should actually work!

✅ data_cache/sr_ml_training/sr_quality_ENHANCED_YYYYMMDD_HHMMSS.parquet
   - Full dataset with all features
```

---

## ✅ Summary

**Implemented ALL your requests:**
1. ✅ Daily sampling (not every 7 days)
2. ✅ Extended to 1 year (365 days)
3. ✅ SR-specific features (from FeatureBank)
4. ✅ Market regime features
5. ✅ Price action features
6. ✅ Multi-timeframe features

**Currently:** 🔄 Collecting and training (~15 min)  
**Expected:** R² improves from -0.002 to 0.15-0.25  
**This should FINALLY work with enhanced features!** 🎯

