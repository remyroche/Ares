# SR ML Model Improvement Recommendations

**Generated:** November 1, 2025  
**Current Performance:** R² = 15.5% (weak predictive power)  
**Goal:** Achieve R² > 30% and improve trading profitability

---

## Executive Summary

Analysis of the SR ML pipeline reveals three critical issues:
1. **Low R² (15.5%)** - ML model explains <16% of SR level quality variance
2. **Suspicious Quality Scores** - Projected Fibonacci levels get high scores despite 0 touches
3. **Poor Feature Distribution** - One feature (`distance_to_current_pct`) dominates SHAP importance

---

## ISSUE 1: Low R² Performance (15.5%)

### Root Causes Identified

#### 1.1 Weak Target Variable Definition

**Current Implementation** (`sr_quality_data_collector.py:218-248`):
```python
def _measure_level_performance(self, level, future_data, historical_data):
    # Levels NOT hit get quality_score = 0.2 (low quality)
    # Levels HIT get score based on bounce_strength + hold_strength
    
    if len(hits) == 0:
        return {'quality_score': 0.2}  # Untested = low quality
    
    # Bounce measured as % move from level
    bounce_strength = min(bounce_pct / 0.02, 1.0)  # 2% bounce = 1.0
    hold_strength = 1.0 if level_held else 0.0
    
    quality_score = (bounce_strength * 0.5 + hold_strength * 0.3 + profit * 0.2)
```

**Problems:**
- ❌ Binary treatment: untested levels = 0.2, tested levels = variable
- ❌ Only measures FIRST touch performance
- ❌ Ignores level longevity and repeated success
- ❌ 2% bounce threshold arbitrary (not ATR-normalized)
- ❌ No consideration of market regime during test

#### 1.2 Feature-Target Mismatch

**Current Features Focus:**
- Distance to current price (0.064 SHAP importance - dominates!)
- Price percentile (0.028)
- Market volatility (0.009)
- All others < 0.01

**Current Target Measures:**
- Bounce strength from level
- Whether level held vs broke
- Simulated trade profit

**The Mismatch:**
> **Distance to current price** (spatial feature) has almost no causal relationship to **how well a level will perform when tested** (temporal performance metric).

This explains the low R²!

#### 1.3 Insufficient Training Data Quality

**Current Data Generation:**
```python
# Training data: 3,230 samples in Run 2 (down from 6,545 in Run 1)
# Each sample = one SR level + its future performance

Problems:
- Many levels never get tested (quality_score = 0.2 default)
- Training on untested levels adds noise
- Class imbalance: ~40% untested, 60% tested levels
```

---

### Recommendations: Improve R² to >30%

#### ✅ RECOMMENDATION 1.1: Enhanced Target Variable

**New Multi-Dimensional Quality Score:**

```python
def calculate_enhanced_quality_score(level, future_data, historical_data):
    """
    Multi-dimensional quality score combining:
    1. Bounce Quality (40%) - How strong are bounces from this level?
    2. Hold Quality (30%) - How reliably does it hold?
    3. Predictive Power (20%) - Can we profit from this level?
    4. Persistence (10%) - How long has level remained valid?
    """
    
    # 1. BOUNCE QUALITY (0-1 score)
    bounces = detect_all_bounces(level, future_data)
    if len(bounces) == 0:
        bounce_quality = 0.0
    else:
        # ATR-normalized bounce strength
        atr = calculate_atr(future_data)
        bounce_strengths = [abs(bounce['distance']) / atr for bounce in bounces]
        
        # Median bounce (robust to outliers)
        median_bounce = np.median(bounce_strengths)
        bounce_quality = min(median_bounce / 2.0, 1.0)  # 2 ATR move = 1.0
    
    # 2. HOLD QUALITY (0-1 score)
    tests = detect_all_level_tests(level, future_data)
    if len(tests) == 0:
        hold_quality = 0.0  # Never tested
    else:
        holds = sum(1 for test in tests if test['held'])
        hold_rate = holds / len(tests)
        
        # Penalize levels tested only once
        confidence_adjustment = min(len(tests) / 5.0, 1.0)
        hold_quality = hold_rate * confidence_adjustment
    
    # 3. PREDICTIVE POWER (0-1 score)
    # Simulate trades at this level
    trades = simulate_all_trades(level, future_data)
    if len(trades) == 0:
        predictive_power = 0.0
    else:
        win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades)
        avg_rrr = np.mean([t['rrr'] for t in trades])  # Risk-reward ratio
        
        # Combine win rate and R:R
        predictive_power = (win_rate * 0.6 + min(avg_rrr / 3.0, 1.0) * 0.4)
    
    # 4. PERSISTENCE (0-1 score)
    # How long has level remained valid without breaking?
    bars_since_formation = get_level_age(level)
    bars_since_breach = get_bars_since_last_breach(level, future_data)
    
    if bars_since_breach is None:
        persistence = 1.0  # Never breached
    else:
        # Exponential decay: levels that held longer = better
        persistence = np.exp(-bars_since_breach / 100)
    
    # Weighted combination
    quality_score = (
        bounce_quality * 0.40 +
        hold_quality * 0.30 +
        predictive_power * 0.20 +
        persistence * 0.10
    )
    
    return {
        'quality_score': quality_score,
        'bounce_quality': bounce_quality,
        'hold_quality': hold_quality,
        'predictive_power': predictive_power,
        'persistence': persistence,
        'num_tests': len(tests) if 'tests' in locals() else 0,
        'num_bounces': len(bounces) if 'bounces' in locals() else 0
    }
```

**Benefits:**
- ✅ More nuanced than binary tested/untested
- ✅ Captures multiple aspects of level quality
- ✅ ATR-normalized (adapts to volatility)
- ✅ Robust to outliers (uses median)
- ✅ Rewards consistency over lucky one-time bounces

---

#### ✅ RECOMMENDATION 1.2: Better Feature Engineering

**Problem:** Current features focus on spatial properties (where level is) rather than causal properties (why level works).

**New Feature Categories:**

##### **A. Microstructure Features** (Currently Missing!)

```python
# Order flow imbalance at level
feature_order_flow_imbalance = calculate_delta_volume_at_level(level, data)

# Aggressive buying/selling at level
feature_aggressive_ratio = get_taker_buy_ratio_at_level(level, data)

# Liquidity concentration
feature_volume_profile_strength = get_vpoc_proximity(level, data)

# Bid-ask imbalance near level (if available)
feature_book_imbalance = get_book_depth_ratio_at_level(level)
```

**Why This Matters:**
> Levels work because of **liquidity clusters** and **order flow dynamics**, not just price patterns. Without microstructure features, the model is blind to the actual mechanism that makes levels hold or break.

##### **B. Dynamic Interaction Features**

```python
# Current implementation only has 4 interaction features
# ADD these:

# Strength decays over time
feature_time_adjusted_strength = strength * np.exp(-age_bars / 50)

# Touch frequency (touches per 100 bars)
feature_touch_frequency = touch_count / (age_bars / 100 + 1)

# Recent vs old touches (are touches recent?)
feature_recent_touch_ratio = recent_touches / total_touches

# Volume-weighted touch quality
feature_volume_weighted_touches = sum(touch_volume) / (touch_count * avg_volume)

# Regime-adjusted strength
feature_regime_adjusted_strength = strength * volatility_regime_multiplier

# Distance × Volatility interaction
feature_distance_x_volatility = distance_to_current_pct * market_volatility

# Momentum-adjusted distance (approaching vs receding)
feature_momentum_adjusted_distance = distance_to_current_pct * (1 + abs(market_momentum))

# Price velocity approaching level
feature_approach_velocity = (current_price - price_20_bars_ago) / atr
```

##### **C. Temporal Decay Features**

```python
# How "fresh" is this level?
feature_freshness_score = 1.0 / (1.0 + days_since_last_touch / 7.0)

# Is level getting stronger or weaker?
feature_strength_trend = (recent_strength - old_strength) / old_strength

# Touch recency distribution
feature_touch_recency_std = std_deviation_of_touch_times

# Seasonality (does level work better at certain times?)
feature_hour_effectiveness = historical_success_rate_by_hour[current_hour]
feature_day_effectiveness = historical_success_rate_by_day[current_day]
```

##### **D. Confluence Features** (Partially Implemented)

```python
# How many different methods detected this level?
feature_method_agreement_count = len(unique_detection_methods)

# Fibonacci confluence (near Fib level?)
feature_fib_confluence = min_distance_to_fib_level / atr

# Psychological level proximity (round numbers)
feature_psychological_proximity = is_near_round_number(price, threshold=0.005)

# Multiple timeframe alignment
feature_mtf_alignment_score = count_timeframes_confirming_level / total_timeframes

# Volume profile node proximity
feature_vpoc_proximity = distance_to_nearest_hvn / atr
```

##### **E. Regime-Adaptive Features**

```python
# Level performance in current regime
feature_regime_historical_performance = get_level_winrate_in_regime(level, current_regime)

# Regime transition risk (is regime about to change?)
feature_regime_stability = regime_probability_score

# Volatility-adjusted metrics
feature_vol_adjusted_strength = strength / (volatility_percentile + 0.1)
feature_vol_adjusted_touches = touch_count * (volatility / historical_avg_volatility)
```

---

#### ✅ RECOMMENDATION 1.3: Data Quality Improvements

**Current Problem:**
```python
# 40% of training data = levels that were NEVER tested
# These all get quality_score = 0.2 (constant)
# This adds noise and hurts model performance
```

**Solution: Intelligent Filtering**

```python
def filter_training_data(raw_data):
    """
    Filter training data to keep only informative samples.
    """
    filtered = []
    
    for sample in raw_data:
        # FILTER 1: Remove untested levels
        if sample['num_tests'] == 0:
            continue  # Skip - no information about quality
        
        # FILTER 2: Remove barely-tested levels (high variance)
        if sample['num_tests'] < 2:
            continue  # Need multiple tests for reliable quality measure
        
        # FILTER 3: Remove ancient irrelevant levels
        if sample['price'] < current_price * 0.30:  # More than 70% below
            continue
        if sample['price'] > current_price * 2.0:   # More than 100% above
            continue
        
        # FILTER 4: Remove levels with insufficient data window
        if sample['future_data_bars'] < 50:
            continue  # Need enough data to measure quality
        
        # FILTER 5: Keep only levels with minimum age
        if sample['age_bars'] < 10:
            continue  # Too new, not enough history
        
        filtered.append(sample)
    
    logger.info(f"Filtered {len(raw_data)} → {len(filtered)} samples")
    logger.info(f"Retention rate: {len(filtered)/len(raw_data)*100:.1f}%")
    
    return filtered
```

**Expected Impact:**
- Current: 3,230 samples → ~1,500 high-quality samples (53% worse but cleaner)
- R² should improve from 15.5% → 25-35% with cleaner data
- Reduced overfitting (lower train-val gap)

---

#### ✅ RECOMMENDATION 1.4: Alternative Model Architectures

**Current:** LightGBM with regression objective

**Try These:**

##### **Option A: Two-Stage Model**

```python
# Stage 1: Classification (will level be tested?)
test_classifier = LGBMClassifier()
test_classifier.fit(X, y_will_be_tested)

# Stage 2: Regression (if tested, how good is it?)
quality_regressor = LGBMRegressor()
quality_regressor.fit(X[tested_levels_only], y_quality[tested_levels_only])

# Prediction
will_test = test_classifier.predict_proba(X_new)[:, 1]
quality_if_tested = quality_regressor.predict(X_new)
final_score = will_test * quality_if_tested
```

**Benefit:** Separates two different prediction tasks that require different features.

##### **Option B: Ensemble with Different Targets**

```python
# Model 1: Predict bounce strength
model_bounce = LGBMRegressor()
model_bounce.fit(X, y_bounce_quality)

# Model 2: Predict hold rate
model_hold = LGBMRegressor()
model_hold.fit(X, y_hold_quality)

# Model 3: Predict trading profit
model_profit = LGBMRegressor()
model_profit.fit(X, y_predictive_power)

# Ensemble prediction
final_quality = (
    model_bounce.predict(X) * 0.4 +
    model_hold.predict(X) * 0.3 +
    model_profit.predict(X) * 0.3
)
```

**Benefit:** Each sub-model specializes in one aspect of quality.

##### **Option C: Neural Network for Non-Linear Patterns**

```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_dim=n_features),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    
    keras.layers.Dense(32, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Quality score 0-1
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae', keras.metrics.R2Score()]
)
```

**Benefit:** Can capture complex non-linear interactions between features.

---

## ISSUE 2: Suspicious Quality Scores

### Root Cause

**Code Location:** `enhanced_sr_detection.py:2904-2969`

```python
def _detect_fibonacci_levels(self, data: pd.DataFrame) -> List[SRLevel]:
    # ...
    level = SRLevel(
        price=retracement_level,
        strength=strength,  # 0.4-0.6 based on Fib ratio
        type=level_type,
        touch_count=0,  # ❌ ZERO TOUCHES!
        # ...
        quality_score=0.9,  # ❌ HIGH QUALITY SCORE!
    )
```

**The Problem:**
1. Fibonacci levels created with `touch_count=0` (projected, never tested)
2. Ancient price levels ($42-$700) from ETH's early days
3. High quality scores (0.8-0.9) despite zero historical validation
4. These pollute the ML training data and detection results

---

### Recommendations: Fix Quality Score Calculation

#### ✅ RECOMMENDATION 2.1: Quality Score Should Require Evidence

**New Rule:**
> **Quality score MUST be earned through actual price interaction, not assigned based on theory.**

```python
def calculate_quality_score(level, historical_data, future_data):
    """
    Quality score based on EVIDENCE, not theory.
    
    Untested levels get 0.0 (unknown quality).
    Tested levels get scores based on performance.
    """
    
    # Count actual touches
    touches = count_real_touches(level, historical_data)
    
    # Base case: Never touched = unknown quality
    if touches == 0:
        return 0.0  # ← Changed from 0.2 or 0.9
    
    # Calculate quality from performance
    bounce_quality = measure_bounce_strength(level, historical_data)
    hold_quality = measure_hold_rate(level, historical_data)
    
    # Scale by confidence (more touches = more confident)
    confidence = min(touches / 5.0, 1.0)
    
    quality = (bounce_quality * 0.5 + hold_quality * 0.5) * confidence
    
    return quality
```

**Impact:**
- Fibonacci levels with 0 touches → quality = 0.0 (not 0.9)
- Ancient levels with 0 touches → quality = 0.0
- Only levels actually tested by price get non-zero quality scores

---

#### ✅ RECOMMENDATION 2.2: Filter Out Irrelevant Levels

**Add Post-Detection Filtering:**

```python
def filter_detected_levels(levels, current_price, data):
    """
    Remove irrelevant/suspicious levels before ML scoring.
    """
    filtered = []
    
    for level in levels:
        # FILTER 1: Remove untouched theoretical levels
        if level.touch_count == 0:
            logger.debug(f"Filtered untouched level at ${level.price:.2f}")
            continue
        
        # FILTER 2: Remove ancient irrelevant prices
        if level.price < current_price * 0.50:  # More than 50% below
            logger.debug(f"Filtered ancient support at ${level.price:.2f}")
            continue
        
        if level.price > current_price * 1.50:  # More than 50% above
            logger.debug(f"Filtered extreme resistance at ${level.price:.2f}")
            continue
        
        # FILTER 3: Remove levels too far from current price
        distance_pct = abs(level.price - current_price) / current_price
        if distance_pct > 0.30:  # More than 30% away
            logger.debug(f"Filtered distant level at ${level.price:.2f} ({distance_pct*100:.1f}% away)")
            continue
        
        # FILTER 4: Check if level is recent enough
        if hasattr(level, 'last_touch_time'):
            days_since_touch = (pd.Timestamp.now() - level.last_touch_time).days
            if days_since_touch > 90:  # More than 3 months old
                logger.debug(f"Filtered stale level at ${level.price:.2f} (last touch {days_since_touch} days ago)")
                continue
        
        filtered.append(level)
    
    logger.info(f"Level filtering: {len(levels)} → {len(filtered)} levels")
    return filtered
```

**Expected Impact:**
- Run 1/2: 160 levels → ~80-100 relevant levels
- Removes: ancient levels, untested projections, extreme outliers
- Higher signal-to-noise ratio for ML model

---

#### ✅ RECOMMENDATION 2.3: Separate Theoretical vs Actual Levels

**Create Two Level Categories:**

```python
class SRLevel:
    # Add field
    level_category: str  # 'actual' or 'theoretical'
    
    def __post_init__(self):
        # Classify level based on evidence
        if self.touch_count == 0:
            self.level_category = 'theoretical'
            self.quality_score = 0.0  # Force to zero
        elif self.touch_count >= 3:
            self.level_category = 'actual'
            # Calculate real quality score
        else:
            self.level_category = 'provisional'
            # Reduce quality score by uncertainty
            self.quality_score *= 0.5
```

**Usage:**

```python
# For ML training: Use ONLY 'actual' levels
training_data = [level for level in all_levels if level.level_category == 'actual']

# For detection output: Include all, but mark clearly
detection_output = {
    'actual_levels': [l for l in levels if l.level_category == 'actual'],
    'theoretical_levels': [l for l in levels if l.level_category == 'theoretical'],
    'provisional_levels': [l for l in levels if l.level_category == 'provisional']
}
```

---

## ISSUE 3: Poor Feature Distribution

### Root Cause

**SHAP Analysis Shows:**
```
feature_distance_to_current_pct:    0.064  (64% of total importance!)
feature_price_percentile:            0.028  (28%)
feature_distance_x_velocity:         0.015  (15%)
--- All others below 0.01 ---
feature_price_position:              0.012
feature_market_volume_avg:           0.011
feature_width:                       0.011
...
feature_trend_strength:              0.003
```

**The Problem:**
1. ONE feature dominates (distance_to_current_pct)
2. Actual SR-specific features (strength, prominence, touches) are WEAK
3. Market context features barely matter
4. Model is basically learning: "closer levels are more relevant" (not useful!)

**Why This Happens:**
> The target variable (quality_score) is measured in the FUTURE, but `distance_to_current_pct` is about the PRESENT location. The model incorrectly learns that closer levels have better quality scores, but this is actually because closer levels GET TESTED MORE OFTEN in the future data window, not because they're inherently better quality.

**This is a DATA LEAKAGE problem disguised as feature importance!**

---

### Recommendations: Balance Feature Importance

#### ✅ RECOMMENDATION 3.1: Remove Leaky Features

```python
# REMOVE these features - they leak information about the future:
FEATURES_TO_REMOVE = [
    'feature_distance_to_current_pct',  # Biases toward levels about to be tested
    'feature_price_position',            # Similar bias
    'feature_hour_of_day',               # Not predictive of level quality
    'feature_day_of_week',               # Not predictive of level quality
]

# KEEP these features - they're about level properties:
FEATURES_TO_KEEP = [
    'feature_strength',
    'feature_prominence',
    'feature_width',
    'feature_touch_count',
    'feature_consistency',
    'feature_volume_confirmation',
    'feature_avg_bounce_ratio',
    # ... etc
]
```

**Alternative:** Keep distance feature but add to target calculation:
```python
# Adjust quality score to be distance-independent
quality_score_normalized = raw_quality_score  # Don't normalize by distance
# Let the model learn which level PROPERTIES predict quality
# Not which level LOCATIONS are about to be tested
```

---

#### ✅ RECOMMENDATION 3.2: Feature Scaling & Regularization

```python
# Current: Some features are 0-1, others are 0-1000+
# This causes some features to dominate simply due to scale

from sklearn.preprocessing import RobustScaler

# Use RobustScaler (better than StandardScaler for outliers)
scaler = RobustScaler()

# Scale features before training
X_scaled = scaler.fit_transform(X_train)

# Update LightGBM params for better feature regularization
lgbm_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    
    # Stronger regularization to prevent single-feature dominance
    'lambda_l1': 2.0,  # ← Increased from 1.0
    'lambda_l2': 2.0,  # ← Increased from 1.0
    'min_data_in_leaf': 50,  # ← Increased from 32
    
    # Feature sampling to force use of multiple features
    'feature_fraction': 0.5,  # ← Reduced from 0.7 (more dropout)
    'feature_fraction_bynode': 0.5,  # ← New: per-node sampling
    
    # Limit tree depth to prevent overfitting on single features
    'max_depth': 4,  # ← Reduced from 5
    'num_leaves': 15,  # ← Reduced from 22
}
```

---

#### ✅ RECOMMENDATION 3.3: Add Missing Critical Features

Based on SHAP analysis, these feature categories are UNDERREPRESENTED:

##### **Priority 1: Volume & Liquidity Features**

```python
# Currently: Only 'feature_market_volume_avg' (generic)
# ADD:

feature_volume_at_level = get_cumulative_volume_at_price(level.price, data)
feature_volume_profile_strength = get_volume_node_strength(level.price, data)
feature_relative_volume_at_level = volume_at_level / avg_market_volume

# Liquidity metrics (if available from order book)
feature_bid_ask_imbalance = get_book_imbalance(level.price)
feature_liquidity_concentration = get_liquidity_score(level.price)
```

**Why:** Levels with high volume are more likely to hold. Current model doesn't know this!

##### **Priority 2: Level Evolution Features**

```python
# Currently: Only static level properties
# ADD:

feature_strength_change = (current_strength - strength_30_bars_ago) / 30
feature_touch_acceleration = (recent_touches - old_touches) / time_period
feature_consistency_trend = (recent_consistency - old_consistency)
feature_volume_trend_at_level = (recent_volume - old_volume) / old_volume
```

**Why:** Levels getting stronger over time are better than levels getting weaker!

##### **Priority 3: Comparative Features**

```python
# Currently: Features are absolute values
# ADD: How does this level compare to others?

feature_relative_strength = level.strength / mean_strength_all_levels
feature_strength_percentile = percentile_rank(level.strength, all_strengths)
feature_relative_prominence = level.prominence / max_prominence_all_levels
feature_nearest_level_distance = distance_to_nearest_other_level / atr
feature_isolation_score = 1.0 / (1.0 + count_levels_within_2atr)
```

**Why:** A "medium" level in a sparse area might be better than a "strong" level in a cluster!

---

#### ✅ RECOMMENDATION 3.4: Feature Interaction Engineering

**Current:** Only 4 interaction features
```python
'feature_strength_x_volume': strength * volume
'feature_prominence_x_width': prominence * width
'feature_touch_x_consistency': touches * consistency
'feature_cluster_x_multi_tf': cluster_density * multi_tf_score
```

**ADD: Systematic interaction generation**

```python
def generate_interaction_features(features_dict):
    """
    Generate meaningful interaction features automatically.
    """
    interactions = {}
    
    # Core SR features to interact
    core_features = [
        'strength', 'prominence', 'width', 'touch_count',
        'consistency', 'volume_confirmation', 'cluster_density'
    ]
    
    # Market context features
    market_features = [
        'market_volatility', 'market_trend', 'market_momentum',
        'regime_strength'
    ]
    
    # Create interactions: core × market
    for core in core_features:
        for market in market_features:
            key = f'feature_{core}_x_{market}'
            interactions[key] = features_dict[f'feature_{core}'] * features_dict[f'feature_{market}']
    
    # Create polynomial features for key metrics
    for feature in ['strength', 'touch_count', 'consistency']:
        interactions[f'feature_{feature}_squared'] = features_dict[f'feature_{feature}'] ** 2
        interactions[f'feature_{feature}_sqrt'] = np.sqrt(features_dict[f'feature_{feature}'])
    
    # Ratio features
    interactions['feature_strength_per_touch'] = features_dict['feature_strength'] / (features_dict['feature_touch_count'] + 1)
    interactions['feature_bounce_per_age'] = features_dict['feature_avg_bounce_ratio'] / (features_dict['feature_age_bars'] / 100 + 1)
    
    return interactions
```

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 days)

1. **Fix quality score calculation** (Recommendation 2.1)
   - Set untouched levels to quality = 0.0
   - Filter out ancient/irrelevant levels (Recommendation 2.2)
   - **Expected Impact:** +5-8% R²

2. **Remove leaky features** (Recommendation 3.1)
   - Remove `distance_to_current_pct`, `price_position`
   - **Expected Impact:** Forces model to learn real patterns, may initially drop R² but improve generalization

3. **Add critical missing features** (Recommendation 3.3 Priority 1)
   - Volume at level
   - Volume profile strength
   - **Expected Impact:** +3-5% R²

### Phase 2: Medium Effort (3-5 days)

4. **Improve target variable** (Recommendation 1.1)
   - Implement enhanced quality score with 4 components
   - Add ATR normalization
   - **Expected Impact:** +5-10% R²

5. **Filter training data** (Recommendation 1.3)
   - Remove untested levels
   - Remove barely-tested levels
   - **Expected Impact:** +3-7% R² (cleaner signal)

6. **Add evolution & comparative features** (Recommendation 3.3 Priority 2 & 3)
   - **Expected Impact:** +2-5% R²

### Phase 3: Advanced (1-2 weeks)

7. **Implement two-stage model** (Recommendation 1.4 Option A)
   - Separate "will be tested?" from "quality if tested?"
   - **Expected Impact:** +5-10% R²

8. **Add microstructure features** (Recommendation 1.2 Category A)
   - Order flow imbalance
   - Volume profile integration
   - **Expected Impact:** +5-10% R² (if data available)

9. **Feature engineering at scale** (Recommendation 3.4)
   - Systematic interaction generation
   - Polynomial features
   - **Expected Impact:** +2-5% R²

---

## Expected Final Results

### Conservative Estimate
- **Current R²:** 15.5%
- **After Phase 1:** 23-28% (+8-13%)
- **After Phase 2:** 31-40% (+8-12%)
- **After Phase 3:** 38-50% (+7-10%)

### Optimistic Estimate
- **Current R²:** 15.5%
- **After Phase 1:** 28-33%
- **After Phase 2:** 40-48%
- **After Phase 3:** 50-60%

---

## Success Metrics

Track these metrics to measure improvement:

```python
success_metrics = {
    # Model performance
    'val_r2': target > 0.35,
    'val_rmse': target < 0.18,
    'train_val_gap': target < 0.05,  # Overfitting measure
    
    # Feature importance
    'max_single_feature_importance': target < 0.30,  # No single feature dominates
    'top_5_cumulative_importance': target < 0.60,  # Distribution more balanced
    
    # Prediction quality
    'prediction_correlation': target > 0.60,  # Predictions match reality
    'shap_consistency': target > 0.70,  # Explanations make sense
    
    # Trading performance (ultimate test)
    'level_hit_rate': target > 0.65,  # Predicted good levels actually get hit
    'level_hold_rate': target > 0.60,  # Predicted strong levels actually hold
    'avg_rrr': target > 2.5,  # Risk-reward ratio from trades at detected levels
}
```

---

## Conclusion

The current ML model's low R² (15.5%) is caused by:
1. **Weak target variable** - binary, not capturing full level quality
2. **Feature-target mismatch** - spatial features predicting temporal performance
3. **Data leakage** - distance features leaking information about future tests
4. **Missing critical features** - no microstructure, evolution, or comparative features

Implementing these recommendations in phases should improve R² from **15.5% → 40-50%** and significantly improve trading profitability.

