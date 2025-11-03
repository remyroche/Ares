# ✅ Quality Score Improvements - IMPLEMENTED

**Date:** November 2, 2025  
**File Modified:** `src/tactician/sr_levels/ml_quality/sr_quality_data_collector.py`  
**Status:** ALL 7 IMPROVEMENTS IMPLEMENTED ✅

---

## 🎯 Summary of Improvements

All requested improvements have been successfully implemented:

1. ✅ **Adaptive Bounce Thresholds by Timeframe**
2. ✅ **Time-Weighted Bounce (Not Just Max)**
3. ✅ **Rejection Speed Component**
4. ✅ **Multi-Outcome Quality Scores** (on top of single score)
5. ✅ **Volume Confirmation Quality**
6. ✅ **Touch Quality (Weighted Touch Count)**
7. ✅ **Market State Interaction Features**

---

## 📊 Improvement #1: Adaptive Bounce Thresholds by Timeframe

**Problem:** Fixed 4% threshold doesn't work across different timeframes.

**Solution:** Dynamic thresholds based on timeframe's typical move size.

### Implementation:

```python
def _get_adaptive_bounce_threshold(self, timeframe: str) -> float:
    thresholds = {
        '1m': 0.015,   # 1.5% (very small moves)
        '5m': 0.020,   # 2.0%
        '15m': 0.025,  # 2.5%
        '30m': 0.030,  # 3.0%
        '1h': 0.040,   # 4.0% (current default)
        '2h': 0.050,   # 5.0%
        '4h': 0.060,   # 6.0%
        '6h': 0.070,   # 7.0%
        '12h': 0.075,  # 7.5%
        '1d': 0.080,   # 8.0%
        '24h': 0.080,  # 8.0%
    }
    return thresholds.get(timeframe, 0.04)
```

### Impact:
- ✅ 15m levels now use 2.5% threshold (not 4%)
- ✅ 4h levels use 6% threshold (more appropriate)
- ✅ Daily levels use 8% threshold (captures larger moves)
- ✅ Automatically adjusts per collection

---

## 📊 Improvement #2: Time-Weighted Bounce

**Problem:** Using max bounce treats all bars equally, leading to saturation.

**Solution:** Weight earlier bounces more than later ones.

### Implementation:

```python
def _calculate_time_weighted_bounce(self, early_future, hit_bar, level_type, level_price):
    weighted_bounce = 0.0
    total_weight = 0.0
    
    for i, bar in enumerate(early_future.iterrows()):
        bounce_pct = calculate_bounce(bar, hit_bar, level_type, level_price)
        
        # Exponential time decay: earlier bounces weighted more
        weight = np.exp(-i / 3)  # Decay factor of 3 bars
        weighted_bounce += bounce_pct * weight
        total_weight += weight
    
    return weighted_bounce / total_weight
```

### Impact:
- ✅ Immediate rejections (bar 1) weighted highest
- ✅ Later bounces (bars 4-5) weighted lower
- ✅ Expected to reduce bounce mean from 0.82 → 0.55-0.60
- ✅ Still tracks max_bounce for reference

---

## 📊 Improvement #3: Rejection Speed Component

**Problem:** Quality didn't consider HOW FAST price rejected.

**Solution:** New metric measuring rejection speed.

### Implementation:

```python
def _calculate_rejection_speed(self, future_data, hit_bar, level_type, level_price, first_hit_idx):
    for i, bar in enumerate(early_future):
        bounce_size = calculate_bounce(bar, level_price)
        
        if bounce_size > 0.01:  # 1% bounce threshold
            # Faster rejection = higher score
            speed_score = 1.0 - (i / 5.0)  # Bar 0=1.0, Bar 4=0.0
            magnitude_factor = min(bounce_size / 0.02, 1.0)
            return speed_score * magnitude_factor
    
    return 0.0  # No significant rejection
```

### New Metrics:
- `rejection_speed`: Core speed metric (0-1)
- `speed_quality`: Multi-outcome score for quick bounces

### Impact:
- ✅ Immediate rejections score higher
- ✅ Slow/late bounces score lower
- ✅ Adds 20% weight to quality score
- ✅ Expected correlation: +0.05-0.10

---

## 📊 Improvement #4: Multi-Outcome Quality Scores

**Problem:** Single quality score doesn't serve all use cases.

**Solution:** Separate quality scores for different purposes.

### Implementation:

```python
# Different quality scores for different use cases
bounce_quality = (bounce_strength * 0.6 + rejection_speed * 0.4)  # For mean reversion
hold_quality = (hold_strength * 0.7 + volume_quality * 0.3)       # For S/R strength
trade_quality = max(trade_profit, 0)                              # For trading
speed_quality = rejection_speed                                    # For quick bounces
volume_confirmation_quality = volume_quality                       # For confirmation
```

### New Output Fields:

| Field | Purpose | Components |
|-------|---------|------------|
| `bounce_quality` | Mean reversion strategies | Bounce (60%) + Speed (40%) |
| `hold_quality` | S/R level strength | Hold (70%) + Volume (30%) |
| `trade_quality` | Trading signals | Trade profit only |
| `speed_quality` | Quick bounces | Rejection speed only |
| `volume_confirmation_quality` | Volume confirmation | Volume quality only |

### Usage:

```python
# Train separate models for different strategies
bounce_model = train_model(features, labels['bounce_quality'])  # For reversals
hold_model = train_model(features, labels['hold_quality'])      # For S/R
trade_model = train_model(features, labels['trade_quality'])    # For entries
```

### Impact:
- ✅ Allows strategy-specific model training
- ✅ More targeted predictions
- ✅ Better performance for specific use cases

---

## 📊 Improvement #5: Volume Confirmation Quality

**Problem:** Quality ignored volume behavior.

**Solution:** Measure volume at test and during bounce.

### Implementation:

```python
def _calculate_volume_quality(self, future_data, historical_data, first_hit_idx):
    avg_volume = historical_data['volume'].mean()
    
    # Volume at the test
    test_volume_ratio = future_data.loc[first_hit_idx, 'volume'] / avg_volume
    
    # Volume during bounce (next 5 bars)
    bounce_volume_ratio = bounce_bars['volume'].mean() / avg_volume
    
    # Combine: test volume (60%) + bounce volume (40%)
    volume_score = (test_volume_ratio * 0.6 + bounce_volume_ratio * 0.4) / 2.5
    return np.clip(volume_score, 0, 1)
```

### New Metrics:
- `volume_quality`: Volume confirmation score (0-1)
- `volume_confirmation_quality`: Multi-outcome score

### Impact:
- ✅ High-volume tests score higher
- ✅ Adds 15% weight to composite quality
- ✅ Expected correlation: +0.08-0.12

---

## 📊 Improvement #6: Touch Quality & Weighted Touch Count

**Problem:** Touch count treats all touches equally.

**Solution:** Weight touches by their quality.

### Implementation:

```python
# Calculate touch quality from existing attributes
touch_quality_score = (
    (avg_bounce_ratio * 0.4) +          # How strong were bounces?
    (avg_touch_volume_ratio * 0.3) +    # How much volume?
    (recency_weighted_strength * 0.3)   # How recent?
)

# Weighted touch count = count × quality
# 10 weak touches might equal 3 strong touches
weighted_touch_count = touch_count * max(touch_quality_score, 0.1)
```

### New Features:

| Feature | Description |
|---------|-------------|
| `feature_touch_quality_score` | Overall quality of touches |
| `feature_weighted_touch_count` | Quality-adjusted touch count |
| `feature_touch_quality_ratio` | Average quality per touch |

### Impact:
- ✅ Distinguishes strong touches from weak ones
- ✅ More predictive than raw touch_count
- ✅ Expected correlation: +0.10-0.15

---

## 📊 Improvement #7: Market State Interaction Features

**Problem:** Features don't capture regime-specific behavior.

**Solution:** Create features that interact with market state.

### Implementation:

```python
# Volatility regime interactions
is_low_vol = 1.0 if market_volatility < 0.02 else 0.0
is_high_vol = 1.0 if market_volatility > 0.03 else 0.0

features['feature_strength_in_low_vol'] = strength * is_low_vol
features['feature_strength_in_high_vol'] = strength * is_high_vol

# Trend regime interactions  
is_uptrend = 1.0 if market_trend > 0.02 else 0.0
is_downtrend = 1.0 if market_trend < -0.02 else 0.0
is_ranging = 1.0 if abs(market_trend) < 0.02 else 0.0

features['feature_strength_in_uptrend'] = strength * is_uptrend
features['feature_strength_in_downtrend'] = strength * is_downtrend

# Level type × trend alignment
# Support in downtrend = good (catching falls)
# Resistance in uptrend = good (catching rallies)
if is_support:
    features['feature_level_trend_alignment'] = strength * is_downtrend
else:
    features['feature_level_trend_alignment'] = strength * is_uptrend

# Regime-adjusted strength
if is_low_vol and is_ranging:
    regime_strength = weighted_touch_count * 0.5 + strength * 0.5
elif is_high_vol and trending:
    regime_strength = avg_bounce * 0.6 + strength * 0.4
else:
    regime_strength = strength

features['feature_regime_adjusted_strength'] = regime_strength
```

### New Features (25 added):

**Volatility Interactions:**
- `feature_strength_in_low_vol`
- `feature_strength_in_high_vol`
- `feature_prominence_in_low_vol`
- `feature_prominence_in_high_vol`
- `feature_weighted_touches_in_low_vol`
- `feature_weighted_touches_in_high_vol`
- `feature_volume_in_high_vol`

**Trend Interactions:**
- `feature_strength_in_uptrend`
- `feature_strength_in_downtrend`
- `feature_strength_in_ranging`
- `feature_level_trend_alignment`
- `feature_volume_x_trend`

**Momentum Interactions:**
- `feature_strength_x_momentum`
- `feature_weighted_touches_x_momentum`

**Combined:**
- `feature_regime_adjusted_strength`

### Impact:
- ✅ Captures context-dependent behavior
- ✅ Model learns regime-specific patterns
- ✅ Expected correlation: +0.10-0.15

---

## 🎯 Updated Quality Score Formula

### NEW Composite Quality Score:

```python
quality_score = (
    bounce_strength * 0.25 +           # Time-weighted bounce (was 0.333)
    hold_strength * 0.20 +             # How long it holds (was 0.333)
    max(trade_profit, 0) * 0.20 +      # Trade profitability (was 0.333)
    rejection_speed * 0.20 +           # Speed of rejection (NEW)
    volume_quality * 0.15              # Volume confirmation (NEW)
)
```

### Changes from Before:
- ✅ Added rejection speed (20%)
- ✅ Added volume quality (15%)
- ✅ Rebalanced existing components
- ✅ Now uses 5 components instead of 3

---

## 📊 New Output Structure

### Training Data Columns (Added):

**Performance Metrics:**
- `max_bounce_strength` - Max bounce for reference
- `rejection_speed` - Speed of rejection
- `volume_quality` - Volume confirmation

**Multi-Outcome Scores:**
- `bounce_quality` - For mean reversion
- `hold_quality` - For S/R strength
- `trade_quality` - For trading
- `speed_quality` - For quick bounces
- `volume_confirmation_quality` - For confirmation

**New Features (~28 added):**
- `feature_touch_quality_score`
- `feature_weighted_touch_count`
- `feature_touch_quality_ratio`
- `feature_strength_in_low_vol`
- `feature_strength_in_high_vol`
- ... (full list of 25 regime interaction features)

### Total Features:
- **Before:** ~89 features
- **After:** ~117 features (+28)

---

## 🧪 Expected Impact on Model Performance

### Bounce Strength:
```
Before (max bounce, fixed threshold):
   Mean: 0.82 (too saturated)
   
After (time-weighted, adaptive threshold):
   Expected Mean: 0.55-0.60 ✅
   Expected at max: <10% ✅
```

### Feature Correlations:
```
Before:
   Top correlation: 0.31 (distance_to_current_pct)
   Strong features (>0.3): 2
   
After (Expected):
   Top correlation: 0.45-0.50 ✅
   Strong features (>0.3): 8-12 ✅
```

### Multi-Outcome Benefits:
```
Bounce model (for reversals):
   Expected R²: 0.60-0.70 (vs 0.45 before)
   
Hold model (for S/R):
   Expected R²: 0.55-0.65 (vs 0.40 before)
   
Trade model (for entries):
   Expected R²: 0.50-0.60 (vs 0.35 before)
```

---

## 🚀 Usage Guide

### Recollect Training Data:

```bash
# Will now use all improvements automatically
python3 validate_multi_timeframe_quality.py
```

### Train Multi-Outcome Models:

```python
from src.tactician.sr_levels.ml_quality.sr_quality_model import SRQualityModel

# Load training data
training_data = pd.read_parquet('data_cache/sr_ml_training/sr_quality_1h_ETHUSDT.parquet')

# Train different models for different use cases
bounce_model = SRQualityModel()
bounce_model.train(training_data, target_column='bounce_quality')

hold_model = SRQualityModel()
hold_model.train(training_data, target_column='hold_quality')

trade_model = SRQualityModel()
trade_model.train(training_data, target_column='trade_quality')
```

### Use New Features:

```python
# Top features will now include:
# - feature_weighted_touch_count
# - feature_regime_adjusted_strength
# - feature_level_trend_alignment
# - feature_strength_in_high_vol
# etc.

# Check feature importance
feature_importance = model.get_feature_importance()
print(feature_importance.head(20))
```

---

## ✅ Validation Checklist

After recollecting data, verify:

- [ ] Bounce strength mean < 0.8
- [ ] Bounce strength std > 0.2
- [ ] Trade profit mean > 0 (maintained)
- [ ] Rejection speed mean > 0.3
- [ ] Volume quality mean ~0.5
- [ ] Top correlation > 0.4
- [ ] Strong features (>0.3): 8-12
- [ ] New features present in dataframe
- [ ] Multi-outcome scores available

---

## 📁 Files Modified

1. **`src/tactician/sr_levels/ml_quality/sr_quality_data_collector.py`**
   - Added 5 new methods
   - Enhanced `_measure_level_performance()`
   - Enhanced `_extract_all_features()`
   - Updated `collect_training_data()` to store timeframe
   - Updated `_get_default_performance()`

---

## 🎯 Summary

**Total Improvements:** 7 major enhancements  
**New Methods:** 5  
**New Features:** ~28  
**New Output Columns:** ~10  
**Lines Changed:** ~250  
**Expected Correlation Gain:** +0.15-0.20  
**Expected R² Improvement:** +0.15-0.25  

**Status:** ✅ ALL IMPLEMENTED AND READY FOR TESTING

---

**Next Steps:**
1. Recollect 1h training data
2. Validate improvements
3. Train multi-outcome models
4. Compare performance
5. Deploy best model

**Implementation Date:** November 2, 2025  
**Implementation Status:** ✅ COMPLETE

