# SR Feature Engineering Investigation & Enhancement Guide

## Overview

This guide explains the SR (Support/Resistance) quality prediction feature engineering improvements, including:

1. **Feature Investigation Tools** - Analyze current features and identify gaps
2. **New High-Impact Features** - 30+ new features added for better predictions
3. **Usage Instructions** - How to investigate, train, and validate improvements

---

## 🚀 Quick Start

### 1. Investigate Current Features

```bash
# Analyze training data features
python scripts/investigate_sr_features.py \
    --training-data data_cache/sr_ml_training/sr_quality_training_data.parquet \
    --analyze-missing

# Analyze feature importance from trained model
python scripts/investigate_sr_features.py \
    --model models/sr_quality_model.lgb \
    --top-n 30

# Full analysis with plots and report
python scripts/investigate_sr_features.py \
    --training-data data_cache/sr_ml_training/sr_quality_training_data.parquet \
    --model models/sr_quality_model.lgb \
    --analyze-missing \
    --generate-plots \
    --generate-report
```

### 2. Retrain Model with New Features

```bash
# Run full SR workflow (includes ML training with new features)
python scripts/run_sr_workflow.py \
    --symbol BTCUSDT \
    --exchange binance \
    --timeframe 1h \
    --lookback-days 180 \
    --ml-sample-freq-days 7 \
    --ml-forward-days 10
```

### 3. Compare Performance

```bash
# Before/after comparison (save old model first!)
cp models/sr_quality_model.lgb models/sr_quality_model_old.lgb

# Train new model with enhanced features
python scripts/run_sr_workflow.py --symbol BTCUSDT --timeframe 1h

# Compare metrics in the outcomes/ reports
```

---

## 📊 New Features Added (30+ Features)

### Category 1: Enhanced Temporal Features (6 features)

**Why:** Recency and time dynamics are critical for SR quality

- `feature_touch_frequency` - Touches per day (activity level)
- `feature_avg_time_between_touches` - Average spacing between touches
- `feature_recent_touch_rate` - Touch rate in last 30 bars
- `feature_bars_since_last_touch` - Recency metric
- `feature_level_age_days` - Age of level in days
- `feature_time_decay_*` - Exponential decay (30-bar and 100-bar windows)

**Impact:** High - Identifies active vs stale levels

### Category 2: Market Regime Features (4 features)

**Why:** Context matters - strong levels in volatile markets are more impressive

- `feature_regime_volatility` - Current vol / average vol (ratio)
- `feature_regime_trend_strength` - Trend strength (SMA divergence)
- `feature_distance_to_price_atr` - Distance in ATR units (scale-invariant)
- `feature_volume_regime` - Current volume / average volume

**Impact:** High - Captures market context for level evaluation

### Category 3: Statistical Significance Features (4 features)

**Why:** Measure how unusual/significant a level is

- `feature_volume_spike_ratio` - Volume at level / average volume
- `feature_price_reaction_strength` - Bounce strength vs typical moves
- `feature_volume_profile_score` - Volume confirmation quality
- `feature_price_action_quality` - Composite bounce quality metric

**Impact:** Medium-High - Identifies statistically significant levels

### Category 4: Advanced Interaction Features (4 features)

**Why:** Feature combinations reveal non-linear relationships

- `feature_touches_x_recency` - Many recent touches = active level
- `feature_volume_x_proximity` - High volume near price = important
- `feature_strength_x_volatility_regime` - Strong in volatile = impressive
- `feature_quality_composite` - Unified quality score

**Impact:** Medium - Captures complex relationships

### Category 5: Relative Ranking Features (4 features)

**Why:** Compare levels to each other, not in isolation

- `feature_strength_percentile` - Strength rank vs other levels
- `feature_touches_percentile` - Touch count rank
- `feature_level_density_nearby` - Crowding/clustering metric
- `feature_distance_to_nearest_level` - Proximity to other levels

**Impact:** Medium - Provides relative context

### Category 6: Level Quality Tiers (4 features)

**Why:** Categorical indicators for tier-based filtering

- `feature_is_top_10_pct` - Top 10% indicator
- `feature_is_top_20_pct` - Top 20% indicator
- `feature_quality_tier` - 0-3 quality tier (weak/medium/strong/critical)
- `feature_relative_strength_rank` - Normalized strength ranking

**Impact:** Low-Medium - Useful for filtering and interpretability

### Category 7: Additional Enhancements

Already existed but improved:

- Time decay features (exponential and linear)
- Method confluence (multiple detection methods agree)
- Regime-adjusted metrics (volatility and trend alignment)
- Enhanced interaction features

---

## 🔍 Feature Investigation Tool

### Purpose

The investigation script (`investigate_sr_features.py`) helps you:

1. **List all current features** - See what features exist in training data
2. **Analyze feature importance** - Find which features matter most
3. **Identify missing features** - Discover opportunities for improvement
4. **Generate visualizations** - Feature importance plots and correlations

### Usage Examples

**Basic feature listing:**
```bash
python scripts/investigate_sr_features.py \
    --training-data data_cache/sr_ml_training/sr_quality_training_data.parquet
```

**Feature importance analysis:**
```bash
python scripts/investigate_sr_features.py \
    --model models/sr_quality_model.lgb \
    --top-n 30
```

**Missing feature detection:**
```bash
python scripts/investigate_sr_features.py \
    --training-data data_cache/sr_ml_training/sr_quality_training_data.parquet \
    --analyze-missing
```

**Generate plots:**
```bash
python scripts/investigate_sr_features.py \
    --model models/sr_quality_model.lgb \
    --generate-plots
```

**Full comprehensive report:**
```bash
python scripts/investigate_sr_features.py \
    --training-data data_cache/sr_ml_training/sr_quality_training_data.parquet \
    --model models/sr_quality_model.lgb \
    --analyze-missing \
    --generate-plots \
    --generate-report
```

### Outputs

The investigation tool generates:

- **Console output** - Feature lists, importance rankings, missing features
- **Feature importance plot** - `outcomes/feature_importance.png`
- **Correlation heatmap** - `outcomes/feature_correlation.png`
- **Comprehensive report** - `outcomes/feature_engineering_report.md`

---

## 📈 Expected Performance Improvements

### Before Enhancement

**Typical metrics (with ~60 features):**
- CV R² Score: 0.30-0.45
- Precision@10: 60-70%
- Spearman ρ: 0.50-0.65

### After Enhancement

**Expected metrics (with ~90+ features):**
- CV R² Score: 0.40-0.55 (+10-15% improvement)
- Precision@10: 70-80% (+10% improvement)
- Spearman ρ: 0.60-0.75 (+10-15% improvement)

**Key improvements:**
- Better temporal understanding (recency/decay)
- Market context awareness (regime features)
- Statistical significance (volume/bounce spikes)
- Non-linear interactions (feature combinations)

---

## 🔧 Implementation Details

### Files Modified

1. **`src/tactician/sr_levels/ml_quality/sr_quality_data_collector.py`**
   - Enhanced `_extract_all_features()` method
   - Added 30+ new features
   - Added `_calculate_atr()` helper method

2. **`src/tactician/sr_levels/enhanced_sr_detection.py`**
   - Updated `_extract_all_ml_features()` method
   - Ensured feature parity with training data

3. **`scripts/investigate_sr_features.py`** (NEW)
   - Feature investigation and analysis tool
   - Generates reports and visualizations

### Feature Naming Convention

All features follow this pattern:
- Prefix: `feature_`
- Descriptive name: `{category}_{metric}`
- Examples:
  - `feature_regime_volatility`
  - `feature_touches_x_recency`
  - `feature_is_top_10_pct`

### Feature Consistency

**Critical:** Training and prediction features MUST match exactly!

- Training features: `sr_quality_data_collector.py::_extract_all_features()`
- Prediction features: `enhanced_sr_detection.py::_extract_all_ml_features()`

Both methods now generate the same feature set.

---

## 🎯 Best Practices

### 1. Feature Engineering Workflow

```bash
# Step 1: Investigate current state
python scripts/investigate_sr_features.py \
    --training-data <path> \
    --model <path> \
    --analyze-missing \
    --generate-report

# Step 2: Review report
cat outcomes/feature_engineering_report.md

# Step 3: Add new features (if needed)
# Edit sr_quality_data_collector.py and enhanced_sr_detection.py

# Step 4: Retrain model
python scripts/run_sr_workflow.py \
    --symbol BTCUSDT \
    --timeframe 1h

# Step 5: Evaluate improvements
# Compare before/after metrics in outcomes/ reports
```

### 2. Feature Validation

After adding new features:

1. **Check feature count** - Ensure training and prediction match
2. **Validate feature names** - No typos, consistent naming
3. **Check for NaN** - Ensure proper default values
4. **Test prediction** - Run model on test data

### 3. Monitoring Feature Importance

Regularly check which features are most important:

```bash
# After retraining
python scripts/investigate_sr_features.py \
    --model models/sr_quality_model.lgb \
    --top-n 30
```

Look for:
- **Temporal features in top 20?** - Good, recency matters
- **Regime features in top 30?** - Good, context matters
- **Interaction features in top 40?** - Expected, non-linear relationships
- **Duplicate features?** - Consider removing redundancy

### 4. Avoiding Overfitting

With 90+ features, overfitting risk increases. Mitigations:

1. **Strong regularization** - L1/L2 in LGBM config
2. **Cross-validation** - 5-fold time series CV
3. **Early stopping** - Monitor validation loss
4. **Feature selection** - Remove low-importance features
5. **HPO** - Hyperparameter optimization finds optimal regularization

---

## 📚 Additional Resources

### Related Files

- `src/tactician/sr_levels/ml_quality/sr_quality_model.py` - Model training
- `ML_PURE_SCORING_DETAILED_EXPLANATION.md` - ML scoring approach
- `scripts/run_sr_workflow.py` - Full SR workflow with ML training

### Key Concepts

**Temporal Features:** Age, recency, touch frequency
- **Why:** Recent activity matters more than old touches
- **Example:** A level touched 5 times in last 10 days > touched 5 times over 100 days

**Market Regime Features:** Volatility, trend, volume context
- **Why:** Strong level in volatile market > strong level in calm market
- **Example:** Holding during 5% volatility > holding during 1% volatility

**Statistical Significance:** Volume spikes, bounce strength
- **Why:** Unusual reactions signal important levels
- **Example:** 3x average volume at level = high significance

**Interaction Features:** Non-linear combinations
- **Why:** Features work together, not independently
- **Example:** Touches × Recency = active level score

---

## 🐛 Troubleshooting

### Feature Mismatch Error

**Error:** `KeyError: Missing features: ['feature_xyz']`

**Cause:** Training and prediction features don't match

**Fix:**
```python
# Check training features
df = pd.read_parquet('data_cache/sr_ml_training/sr_quality_training_data.parquet')
training_features = [c for c in df.columns if c.startswith('feature_')]

# Check model features
from src.tactician.sr_levels.ml_quality import SRQualityModel
model = SRQualityModel()
model.load('models/sr_quality_model.lgb')
model_features = model.feature_names

# Find differences
missing = set(model_features) - set(training_features)
extra = set(training_features) - set(model_features)
```

### NaN Features

**Error:** Model predictions are NaN or inf

**Cause:** Missing data handling in feature extraction

**Fix:** Add default values and NaN checks:
```python
# In feature extraction
features['feature_xyz'] = get_attr('xyz', default=0.0)

# Handle division by zero
features['feature_ratio'] = numerator / (denominator + 1e-8)

# Fill NaN
X = X.fillna(0.0)
```

### Low Feature Importance

**Issue:** New features have near-zero importance

**Possible causes:**
1. Feature not informative
2. Feature redundant with existing features
3. Feature scale issues (too small/large)
4. Feature always constant

**Fix:**
- Check feature distribution
- Normalize/scale if needed
- Remove if truly redundant

---

## 📝 Summary

**What we did:**
- Added 30+ high-impact features across 6 categories
- Created investigation tool for feature analysis
- Ensured training/prediction feature consistency
- Documented usage and best practices

**Expected outcomes:**
- 10-15% improvement in model performance
- Better temporal understanding
- Better market context awareness
- More robust level quality predictions

**Next steps:**
1. Run investigation tool on your data
2. Retrain model with new features
3. Compare before/after performance
4. Iterate on low-performing features

---

## 💡 Feature Ideas for Future Enhancement

**Level Clustering Features:**
- Distance to nearest level (actual, not proxy)
- Number of levels within 1% range
- Cluster quality score

**Cross-Symbol Features:**
- BTC correlation (for altcoins)
- Market-wide level strength
- Sector/category level density

**Advanced Temporal:**
- Fourier components (cyclical patterns)
- Momentum indicators at level
- Volatility forecast at level

**Multi-Timeframe:**
- Level confirmation across TFs
- Alignment score (all TFs agree)
- TF-specific strength weights

**Order Flow (if data available):**
- Order book depth at level
- Large order activity
- Bid/ask imbalance

---

**Last Updated:** 2025-11-02
**Author:** SR Feature Engineering Team
**Version:** 1.0

