# SR ML Improvement - Action Plan

**Goal:** Improve R² from 15.5% → 40%+ in 3 phases

---

## 📊 Current State Analysis

### Problems Identified

| Issue | Current State | Impact on R² |
|-------|---------------|--------------|
| **Weak Target** | Binary tested/untested, only first bounce | -10% |
| **Data Leakage** | `distance_to_current_pct` dominates (64% SHAP) | -8% |
| **Missing Features** | No microstructure, evolution, or volume profile | -12% |
| **Suspicious Levels** | Fibonacci/ancient levels with 0 touches get quality=0.9 | -5% |
| **Noisy Training Data** | 40% untested levels with quality=0.2 | -5% |

**Total R² Lost:** ~40% → Current achievable R² ≈ 15-20%

---

## 🎯 Phase 1: Quick Wins (1-2 days, +8-13% R²)

### Task 1.1: Fix Quality Score for Untouched Levels ⚡

**File:** `src/training/steps/market_analysis/components/sr_parameter_optimization.py`

**Line:** 2809-2821 (`_calculate_level_quality`)

```python
def _calculate_level_quality(self, touches: int, strength: float, volume_confirmation: float) -> float:
    """Calculate quality score for an SR level."""
    
    # NEW: Untouched levels get zero quality (not theoretical scores)
    if touches == 0:
        return 0.0  # Changed from assigning 0.4-0.9 based on Fib ratios
    
    # Require minimum touches for confidence
    if touches < 2:
        return 0.0
    
    # Existing calculation for tested levels
    touches_score = min(1.0, (touches - 2) / 8.0)
    strength_score = min(1.0, strength / 0.1)
    volume_score = min(1.0, max(0.0, (volume_confirmation - 0.5) / 1.5))
    
    # Weighted combination
    return (touches_score * 0.4 + strength_score * 0.4 + volume_score * 0.2)
```

**File:** `src/tactician/sr_levels/enhanced_sr_detection.py`

**Line:** 2944 (Fibonacci level creation)

```python
level = SRLevel(
    price=retracement_level,
    strength=0.0,  # Changed from 0.4-0.6
    type=level_type,
    touch_count=0,
    # ...
    confidence_score=0.0,  # Changed from strength
    quality_score=0.0,  # Changed from high default
    # ...
)
```

---

### Task 1.2: Add Level Filtering Before ML ⚡

**File:** `src/training/steps/market_analysis/sr_detection.py`

**Line:** After 639 (after existing filtering)

```python
# NEW: Enhanced filtering for ML quality
def _filter_levels_for_ml(self, levels: List[Dict], current_price: float) -> List[Dict]:
    """Filter out irrelevant levels before ML scoring."""
    filtered = []
    
    for level in levels:
        # Filter 1: Must have touches
        if level.get('touch_count', 0) == 0:
            continue
        
        # Filter 2: Price range filter
        price = level.get('price', 0)
        if price < current_price * 0.50 or price > current_price * 1.50:
            continue
        
        # Filter 3: Distance filter
        distance_pct = abs(price - current_price) / current_price
        if distance_pct > 0.30:  # More than 30% away
            continue
        
        filtered.append(level)
    
    self.logger.info(f"ML filtering: {len(levels)} → {len(filtered)} levels")
    return filtered

# Apply filter
current_price = clean_data['close'].iloc[-1]
levels_dict = self._filter_levels_for_ml(levels_dict, current_price)
```

---

### Task 1.3: Remove Leaky Features ⚡

**File:** `src/tactician/sr_levels/ml_quality/sr_quality_data_collector.py`

**Line:** 350-398 (`_extract_all_features`)

```python
# REMOVE these features (comment out or delete):
# 'feature_distance_to_current_pct': abs(...),  # LEAKY!
# 'feature_price_position': (...),               # LEAKY!
# 'feature_hour_of_day': ...,                   # Not predictive
# 'feature_day_of_week': ...,                   # Not predictive

# KEEP all other features
```

**Expected Impact:**
- R² may initially drop 2-3% but generalization will improve
- SHAP distribution will become more balanced

---

### Task 1.4: Add Volume Features ⚡

**File:** `src/tactician/sr_levels/ml_quality/sr_quality_data_collector.py`

**Line:** After 393 (in `_extract_all_features`)

```python
# NEW: Volume-based features
def _calculate_volume_at_level(level_price, data, tolerance=0.005):
    """Calculate cumulative volume near level."""
    level_min = level_price * (1 - tolerance)
    level_max = level_price * (1 - tolerance)
    
    mask = (data['low'] <= level_max) & (data['high'] >= level_min)
    volume_at_level = data.loc[mask, 'volume'].sum()
    
    return volume_at_level

volume_at_level = _calculate_volume_at_level(get_attr('price', current_price), data)
avg_volume = data['volume'].mean()

features.update({
    'feature_volume_at_level': volume_at_level / avg_volume if avg_volume > 0 else 0,
    'feature_volume_concentration': volume_at_level / data['volume'].sum() if data['volume'].sum() > 0 else 0,
})
```

---

## 🚀 Phase 2: Enhanced Target & Data Quality (3-5 days, +8-12% R²)

### Task 2.1: Implement Multi-Dimensional Quality Score

**File:** `src/tactician/sr_levels/ml_quality/sr_quality_data_collector.py`

**Line:** 218-336 (Replace `_measure_level_performance`)

```python
def _measure_level_performance(self, level, future_data: pd.DataFrame,
                               historical_data: pd.DataFrame) -> Dict[str, float]:
    """
    Multi-dimensional quality score.
    
    Components:
    1. Bounce Quality (40%) - strength of bounces
    2. Hold Quality (30%) - how reliably it holds
    3. Predictive Power (20%) - trade profitability
    4. Persistence (10%) - how long it remains valid
    """
    tolerance = level.price * 0.005
    level_type = level.type if hasattr(level, 'type') else 'unknown'
    level_price = level.price if hasattr(level, 'price') else 0
    
    # ATR for normalization
    atr = self._calculate_atr(future_data)
    
    # Detect all level tests
    tests = self._detect_all_level_tests(level, future_data, tolerance)
    
    if len(tests) == 0:
        # Never tested = unknown quality
        return {
            'quality_score': 0.0,
            'bounce_quality': 0.0,
            'hold_quality': 0.0,
            'predictive_power': 0.0,
            'persistence': 0.0,
            'num_tests': 0
        }
    
    # 1. BOUNCE QUALITY
    bounces = [test for test in tests if test['bounced']]
    if len(bounces) > 0:
        bounce_strengths_atr = [abs(b['bounce_distance']) / atr for b in bounces]
        median_bounce = np.median(bounce_strengths_atr)
        bounce_quality = min(median_bounce / 2.0, 1.0)  # 2 ATR = 1.0
    else:
        bounce_quality = 0.0
    
    # 2. HOLD QUALITY
    holds = sum(1 for test in tests if test['held'])
    hold_rate = holds / len(tests)
    confidence_adj = min(len(tests) / 5.0, 1.0)  # Need 5+ tests for full confidence
    hold_quality = hold_rate * confidence_adj
    
    # 3. PREDICTIVE POWER
    trades = self._simulate_trades_at_level(level, future_data, atr)
    if len(trades) > 0:
        win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades)
        avg_rrr = np.mean([t['rrr'] for t in trades])
        predictive_power = win_rate * 0.6 + min(avg_rrr / 3.0, 1.0) * 0.4
    else:
        predictive_power = 0.0
    
    # 4. PERSISTENCE
    bars_since_last_breach = self._get_bars_since_breach(level, future_data)
    if bars_since_last_breach is None:
        persistence = 1.0  # Never breached
    else:
        persistence = np.exp(-bars_since_last_breach / 100)
    
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
        'num_tests': len(tests),
        'num_bounces': len(bounces)
    }

def _detect_all_level_tests(self, level, data, tolerance):
    """Detect all times price tested this level."""
    tests = []
    level_price = level.price
    level_type = level.type
    
    i = 0
    while i < len(data):
        if level_type == 'support':
            if data['low'].iloc[i] <= level_price * (1 + tolerance):
                # Level tested
                bounce_distance = data['high'].iloc[i:i+10].max() - data['low'].iloc[i]
                held = data['close'].iloc[i:i+5].min() >= level_price * 0.99
                
                tests.append({
                    'index': i,
                    'bounced': bounce_distance > 0,
                    'bounce_distance': bounce_distance,
                    'held': held
                })
                i += 10  # Skip ahead
        else:  # resistance
            if data['high'].iloc[i] >= level_price * (1 - tolerance):
                bounce_distance = data['high'].iloc[i] - data['low'].iloc[i:i+10].min()
                held = data['close'].iloc[i:i+5].max() <= level_price * 1.01
                
                tests.append({
                    'index': i,
                    'bounced': bounce_distance > 0,
                    'bounce_distance': bounce_distance,
                    'held': held
                })
                i += 10
        i += 1
    
    return tests

def _calculate_atr(self, data, period=14):
    """Calculate Average True Range."""
    high = data['high']
    low = data['low']
    close = data['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    return atr.iloc[-1] if len(atr) > 0 else 1.0
```

---

### Task 2.2: Filter Training Data

**File:** `src/tactician/sr_levels/ml_quality/sr_quality_data_collector.py`

**Line:** After collecting training samples, before saving

```python
def _filter_training_samples(self, samples):
    """Keep only high-quality training samples."""
    filtered = []
    
    for sample in samples:
        features = sample['features']
        target = sample['target']
        
        # Filter 1: Must be tested
        if target.get('num_tests', 0) < 2:
            continue
        
        # Filter 2: Reasonable price range
        price = features.get('feature_price', 0)
        current_price = features.get('feature_current_price', price)
        if price < current_price * 0.5 or price > current_price * 2.0:
            continue
        
        # Filter 3: Minimum age
        if features.get('feature_age_bars', 0) < 10:
            continue
        
        filtered.append(sample)
    
    self.logger.info(f"Training data filtering: {len(samples)} → {len(filtered)} samples ({len(filtered)/len(samples)*100:.1f}% retained)")
    
    return filtered

# Apply before saving
training_samples = self._filter_training_samples(training_samples)
```

---

## 🏆 Phase 3: Advanced Features & Models (1-2 weeks, +7-10% R²)

### Task 3.1: Add Interaction Features

**File:** `src/tactician/sr_levels/ml_quality/sr_quality_data_collector.py`

**Line:** After 450 (in `_extract_all_features`)

```python
# Systematic interaction generation
interactions = {}

# Core × Market interactions
core_features = ['strength', 'prominence', 'touch_count', 'consistency']
market_features = ['market_volatility', 'market_trend', 'market_momentum']

for core in core_features:
    for market in market_features:
        core_val = features.get(f'feature_{core}', 0)
        market_val = features.get(f'feature_{market}', 0)
        interactions[f'feature_{core}_x_{market}'] = core_val * market_val

# Polynomial features
for feat in ['strength', 'touch_count']:
    val = features.get(f'feature_{feat}', 0)
    interactions[f'feature_{feat}_squared'] = val ** 2
    interactions[f'feature_{feat}_sqrt'] = np.sqrt(max(0, val))

# Ratio features
interactions['feature_strength_per_touch'] = (
    features.get('feature_strength', 0) / (features.get('feature_touch_count', 0) + 1)
)
interactions['feature_bounce_per_age'] = (
    features.get('feature_avg_bounce_ratio', 0) / (features.get('feature_age_bars', 0) / 100 + 1)
)

features.update(interactions)
```

---

### Task 3.2: Two-Stage Model Architecture

**File:** Create new file `src/tactician/sr_levels/ml_quality/two_stage_model.py`

```python
import lightgbm as lgb
from typing import Dict, Any
import numpy as np

class TwoStageQualityModel:
    """
    Stage 1: Predict if level will be tested
    Stage 2: Predict quality if tested
    """
    
    def __init__(self):
        self.test_classifier = None
        self.quality_regressor = None
    
    def train(self, X, y_quality, y_will_test):
        """
        Train both stages.
        
        Args:
            X: Features
            y_quality: Quality scores (0-1)
            y_will_test: Binary labels (0 = not tested, 1 = tested)
        """
        # Stage 1: Classification
        self.test_classifier = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            feature_fraction=0.7,
            bagging_fraction=0.7,
            lambda_l1=1.0,
            lambda_l2=1.0
        )
        self.test_classifier.fit(X, y_will_test)
        
        # Stage 2: Regression (only on tested levels)
        tested_mask = y_will_test == 1
        X_tested = X[tested_mask]
        y_tested = y_quality[tested_mask]
        
        self.quality_regressor = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=100,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=22,
            feature_fraction=0.7,
            bagging_fraction=0.7,
            lambda_l1=1.0,
            lambda_l2=1.0
        )
        self.quality_regressor.fit(X_tested, y_tested)
    
    def predict(self, X):
        """Predict final quality score."""
        # Probability of being tested
        test_prob = self.test_classifier.predict_proba(X)[:, 1]
        
        # Quality if tested
        quality_if_tested = self.quality_regressor.predict(X)
        
        # Combine
        final_score = test_prob * quality_if_tested
        
        return final_score
    
    def get_feature_importance(self):
        """Get feature importance from both models."""
        return {
            'test_classifier': self.test_classifier.feature_importances_,
            'quality_regressor': self.quality_regressor.feature_importances_
        }
```

---

## 📈 Success Tracking

### Metrics to Monitor

Create a tracking script to compare before/after:

```python
# File: scripts/evaluate_ml_improvements.py

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluate_model(y_true, y_pred, model_name="Model"):
    """Evaluate and print model performance."""
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Prediction correlation
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    
    print(f"\n{model_name} Performance:")
    print(f"  R² Score:     {r2:.4f}")
    print(f"  RMSE:         {rmse:.4f}")
    print(f"  MAE:          {mae:.4f}")
    print(f"  Correlation:  {corr:.4f}")
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'correlation': corr
    }

# Compare improvements
baseline_metrics = evaluate_model(y_val, y_pred_baseline, "Baseline (R²=15.5%)")
phase1_metrics = evaluate_model(y_val, y_pred_phase1, "Phase 1")
phase2_metrics = evaluate_model(y_val, y_pred_phase2, "Phase 2")

improvement = (phase2_metrics['r2'] - baseline_metrics['r2']) / baseline_metrics['r2'] * 100
print(f"\n✅ Total R² improvement: +{improvement:.1f}%")
```

---

## 🎯 Expected Timeline & Results

| Phase | Duration | Tasks | Expected R² | Effort |
|-------|----------|-------|-------------|--------|
| **Baseline** | - | - | 15.5% | - |
| **Phase 1** | 1-2 days | Fix quality scores, filter levels, remove leaky features | 23-28% | Low |
| **Phase 2** | 3-5 days | Multi-dim quality, filter training data | 31-40% | Medium |
| **Phase 3** | 1-2 weeks | Interactions, two-stage model | 38-50% | High |

---

## ✅ Checklist

### Phase 1 (Quick Wins)
- [ ] Task 1.1: Set untouched levels quality = 0.0
- [ ] Task 1.2: Add level filtering before ML
- [ ] Task 1.3: Remove leaky features
- [ ] Task 1.4: Add volume features
- [ ] Run training and verify R² improvement
- [ ] Generate new SHAP plot to verify balanced features

### Phase 2 (Enhanced Target)
- [ ] Task 2.1: Implement multi-dimensional quality score
- [ ] Task 2.2: Filter training data
- [ ] Run training with new quality metric
- [ ] Verify R² > 30%
- [ ] Backtest on recent data

### Phase 3 (Advanced)
- [ ] Task 3.1: Add interaction features
- [ ] Task 3.2: Implement two-stage model
- [ ] Compare single-stage vs two-stage
- [ ] Final R² verification (target: 40-50%)
- [ ] Production deployment

---

## 🚨 Common Pitfalls to Avoid

1. **Don't skip Phase 1** - Quick wins build momentum and validate approach
2. **Test after each phase** - Don't implement everything at once
3. **Watch for new data leakage** - New features might introduce new leaks
4. **Keep SHAP analysis** - Monitor feature importance distribution
5. **Validate on out-of-sample data** - Don't overfit to validation set

---

## 📞 Need Help?

If you encounter issues:
1. Check that feature names match exactly between training and inference
2. Verify data types (floats vs ints)
3. Look for NaN/Inf values in new features
4. Use `read_lints` to catch errors early
5. Test each component individually before integration

