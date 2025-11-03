# SR ML Model - Quick Summary

## 🎯 The Problem in One Sentence

**The ML model has 15.5% R² because it's learning "closer levels get tested more" instead of "which level properties predict quality."**

---

## 🔍 Root Causes

### 1. **Data Leakage** 👻
```
feature_distance_to_current_pct = 64% of SHAP importance

Why it's bad: This feature tells the model which levels are ABOUT TO BE TESTED
               in the future window, not which levels are INHERENTLY GOOD.

Fix: Remove this feature, force model to learn from level properties.
```

### 2. **Weak Target Variable** 🎯
```
Current: quality_score = 0.2 (if not tested) OR 0.2-1.0 (if tested)

Problems:
  - Binary treatment (tested vs untested)
  - Only measures FIRST touch
  - Not ATR-normalized
  - Ignores persistence over time

Fix: Multi-dimensional score with 4 components (bounce, hold, profit, persistence)
```

### 3. **Noisy Training Data** 🗑️
```
Current training data:
  - 40% untested levels (quality = 0.2 constant)
  - Ancient levels ($42-$700 ETH prices)
  - Fibonacci projections with 0 touches but quality = 0.9

Fix: Filter out untested, ancient, and suspicious levels
```

### 4. **Missing Critical Features** 📊
```
Missing:
  ✗ Volume at level (liquidity)
  ✗ Order flow imbalance
  ✗ Level evolution (getting stronger/weaker?)
  ✗ Comparative features (vs other levels)
  ✗ Interaction features (strength × volatility)

Fix: Add these feature categories
```

---

## 💡 The Solution (3 Phases)

### Phase 1: Quick Wins (1-2 days) → R² = 23-28%

```python
# 1. Fix quality scores
if touch_count == 0:
    quality_score = 0.0  # Not 0.9!

# 2. Filter out garbage
- Remove levels with 0 touches
- Remove ancient prices (<50% or >150% of current)
- Remove levels >30% away from current price

# 3. Remove leaky features
DELETE: feature_distance_to_current_pct
DELETE: feature_price_position

# 4. Add volume features
ADD: feature_volume_at_level
ADD: feature_volume_concentration
```

**Effort:** LOW | **Impact:** +8-13% R²

---

### Phase 2: Better Target & Data (3-5 days) → R² = 31-40%

```python
# 1. Multi-dimensional quality score
quality = (
    bounce_quality      * 0.40 +  # How strong are bounces? (ATR-normalized)
    hold_quality        * 0.30 +  # Does it hold reliably?
    predictive_power    * 0.20 +  # Can we profit?
    persistence         * 0.10    # Does it last?
)

# 2. Filter training data
KEEP ONLY:
  - Tested at least 2 times
  - Within reasonable price range
  - Minimum age of 10 bars
  
RESULT: 3,230 samples → ~1,500 high-quality samples
```

**Effort:** MEDIUM | **Impact:** +8-12% R²

---

### Phase 3: Advanced (1-2 weeks) → R² = 38-50%

```python
# 1. Add interactions
strength × volatility
touch_count × consistency
strength² (polynomial)

# 2. Two-stage model
Stage 1: Will level be tested? (classifier)
Stage 2: If tested, how good? (regressor)

Final = P(tested) × Quality(if_tested)
```

**Effort:** HIGH | **Impact:** +7-10% R²

---

## 📊 Visual: Feature Importance Problem

### Current (BAD)
```
feature_distance_to_current_pct  ████████████████████████████████ 64%
feature_price_percentile          ██████████████ 28%
feature_distance_x_velocity       ███████ 15%
─────────────────────────────────────────────────
All others combined               ████ 8%
```
☠️ **ONE FEATURE DOMINATES - This is data leakage!**

### Target (GOOD)
```
feature_volume_at_level           ████████████ 15%
feature_strength                  ███████████ 14%
feature_touch_frequency           ██████████ 12%
feature_consistency               █████████ 11%
feature_bounce_quality            ████████ 10%
feature_hold_rate                 ████████ 10%
feature_volatility_adjusted       ███████ 9%
All others distributed            ███████████████████ 19%
```
✅ **BALANCED - Model learns from multiple signals**

---

## 🎯 Success Criteria

| Metric | Current | Phase 1 | Phase 2 | Phase 3 | Target |
|--------|---------|---------|---------|---------|--------|
| **Val R²** | 15.5% | 25% | 35% | 45% | >40% ✅ |
| **Val RMSE** | 0.229 | 0.21 | 0.19 | 0.17 | <0.18 ✅ |
| **Max Feature Importance** | 64% | 35% | 25% | 20% | <30% ✅ |
| **Train-Val Gap** | 0.22 | 0.15 | 0.10 | 0.08 | <0.10 ✅ |

---

## 🚀 Start Here

1. **Open:** `SR_ML_ACTION_PLAN.md`
2. **Do:** Phase 1, Task 1.1 (5 minutes)
3. **Run:** ML training
4. **Check:** Did R² improve?
5. **Continue:** Next task

---

## 📁 Files to Edit

### Phase 1 (Quick)
```
✏️ src/training/steps/market_analysis/components/sr_parameter_optimization.py
   Line 2809: _calculate_level_quality()

✏️ src/tactician/sr_levels/enhanced_sr_detection.py
   Line 2944: Fibonacci level creation

✏️ src/training/steps/market_analysis/sr_detection.py
   Line 639: Add filtering function

✏️ src/tactician/sr_levels/ml_quality/sr_quality_data_collector.py
   Line 350: Remove leaky features
   Line 393: Add volume features
```

### Phase 2 (Medium)
```
✏️ src/tactician/sr_levels/ml_quality/sr_quality_data_collector.py
   Line 218: Replace _measure_level_performance()
   Add: _detect_all_level_tests()
   Add: _filter_training_samples()
```

### Phase 3 (Advanced)
```
✏️ src/tactician/sr_levels/ml_quality/sr_quality_data_collector.py
   Line 450: Add interaction features

🆕 src/tactician/sr_levels/ml_quality/two_stage_model.py
   Create new file with TwoStageQualityModel class
```

---

## 🎓 Key Insights

### Why Distance Feature is Bad
```
❌ WRONG THINKING:
   "Levels close to current price have better quality scores"
   
✅ REALITY:
   "Levels close to current price GET TESTED MORE in the future window"
   
The model learns correlation, not causation!
```

### Why Multi-Dimensional Quality Works
```
Current: quality = bounce_once()
         → Noisy, high variance

Better:  quality = bounce_quality + hold_quality + profit + persistence
         → Robust, low variance, captures multiple aspects
```

### Why Filtering Helps
```
Garbage in  → Garbage out
Good data   → Good model

3,230 samples with 40% noise → R² = 15%
1,500 samples with 10% noise → R² = 35%

Less data, better quality = BETTER RESULTS
```

---

## 💻 Quick Test

After Phase 1 changes, run this:

```bash
# Retrain model
python ares_launcher.py step2.5 --force-rerun

# Check new metrics
cat outcomes/sr_workflow_ETHUSDT_15m/ml_model_training_*.md | grep "avg_val_r2"

# Compare
# Before: "avg_val_r2": 0.15512310840914006
# After:  "avg_val_r2": 0.25-0.30 (expected)
```

If R² improved by 8-10%, Phase 1 worked! ✅

---

## 🆘 Troubleshooting

### "R² didn't improve"
- Check that quality scores actually changed (log them)
- Verify leaky features were removed (check feature list)
- Ensure filtering is working (check sample counts)

### "R² got worse"
- Normal if you removed distance feature (short-term drop)
- Should recover after Phase 2 with better target
- Check for bugs in new code

### "Training fails"
- Feature name mismatch (training vs inference)
- NaN/Inf values in new features
- Data type mismatch (float vs int)

---

## 📚 Full Documentation

- **Detailed Analysis:** `SR_ML_IMPROVEMENT_RECOMMENDATIONS.md`
- **Step-by-Step Guide:** `SR_ML_ACTION_PLAN.md`
- **This Summary:** `SR_ML_QUICK_SUMMARY.md`

**Read the Action Plan for exact code snippets!**

