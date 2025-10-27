# Cross-Timeframe Feature Pruning Analysis

## Problem Summary
Generated 460 cross-timeframe features, but **0 retained** after Phase 2 pruning.

## Detailed Findings

### 1. Generation Success
✅ **460 cross-timeframe features created** (66.3% of all features)
- Uses proper recalculation from scratch (not rolling windows)
- Calls `_recalculate_feature_with_period` with extended lookback
- Generates ratios between base and extended timeframe versions

### 2. Critical Issues

#### Issue 1: 75% of Cross-Timeframe Features are ALL NaN
- **345 out of 460 features** have all NaN values
- These are removed as "problematic features" before clustering
- Sample problematic features:
  - `volume_price_trend_base_3x_ratio` (all NaN)
  - `volume_price_trend_volnorm_3x_ratio` (all NaN)
  - `volume_price_trend_vwap_3x_ratio` (all NaN)

**Root Cause**: The `_generate_extended_timeframe_feature` method is failing to recalculate features with extended periods, returning None or NaN values.

**Why This Happens**:
1. `_extract_period_from_feature_name` may not extract periods correctly for all feature types
2. `_recalculate_feature_with_period` may not support all feature types
3. Extended periods (3x, 6x, 9x, 27x) may exceed available data length

#### Issue 2: Composite Scores are All 1.0 (Not Calculated)
- **All 694 features have composite score = 1.0**
- This means MI (Mutual Information) and stability scores are not being calculated
- Features are ranked alphabetically when scores are tied
- Cross-timeframe features rank #232-#691 (all below cutoff of #104)

**Root Cause**: In `_phase2_cheap_pruning`, line 2006 sets:
```python
composite_scores = {col: 1.0 for col in variant_features.columns}
```
This bypasses actual score calculation!

### 3. Pruning Metrics

**Entry Point**:
- Total features: 694
- Cross-timeframe features: 460 (66.3%)
- All have score: 1.0

**After Problematic Removal**:
- Removed: 345 cross-timeframe features (all NaN)
- Remaining: 115 cross-timeframe features
- But removed as "problematic": 115 more somewhere

**Final Selection**:
- Target retention: 15% (104 features)
- Cross-timeframe features above cutoff: 0
- Cross-timeframe features below cutoff: 460
- Best cross-timeframe rank: #232 (vs cutoff #104)
- **Survival rate: 0%**

## Solutions Needed

### High Priority: Fix NaN Generation (75% of features)

1. **Improve `_extract_period_from_feature_name`**:
   - Add more pattern matching for different feature types
   - Handle features without explicit periods
   - Add logging to track failed extractions

2. **Enhance `_recalculate_feature_with_period`**:
   - Support all feature types in the feature bank
   - Add fallback methods for unknown features
   - Handle extended periods that exceed data length

3. **Add NaN handling in ratio calculation**:
   - Better handling of division by zero
   - Improved NaN filling strategies
   - Validation before returning features

### Medium Priority: Calculate Proper Composite Scores

1. **Enable MI Calculation**:
   - Calculate actual Mutual Information scores for features
   - Use target variables from labeled_data

2. **Enable Stability Calculation**:
   - Calculate feature stability across time windows
   - Combine with MI for composite score

3. **Category-Specific Scoring**:
   - Add explicit boost for cross-timeframe features
   - Protect cross-timeframe category with min_features_per_category

### Low Priority: Adjust Pruning Thresholds

1. **Increase Retention Rate**:
   - Change from 15% to 30-40% retention
   - This would allow more features through

2. **Add Category Protection**:
   - Set `min_features_per_category` for cross_timeframe category
   - Ensure at least some cross-timeframe features survive

## Recommended Immediate Actions

1. **Fix NaN generation**:
   - Debug `_generate_extended_timeframe_feature` 
   - Add comprehensive error handling and logging
   - Validate recalculated features before returning

2. **Enable composite scoring**:
   - Remove hardcoded `composite_scores = {col: 1.0 for col in ...}`
   - Calculate actual MI and stability scores
   - This will give cross-timeframe features fair ranking

3. **Add explicit protection**:
   - Detect cross-timeframe features by pattern matching
   - Add to protected categories
   - Set minimum retention count

## Testing Commands

```bash
# Run with detailed tracking
python3 src/launcher/ares_launcher.py \
  --step feature_generation_interaction_generation_step \
  --symbol ETHUSDT \
  --execution-mode light \
  2>&1 | grep -A 30 "CROSS-TIMEFRAME FEATURE TRACKING"

# Check for NaN features
python3 -c "
import pandas as pd
df = pd.read_parquet('artifacts/.../variant_features_...parquet')
ct_features = [c for c in df.columns if '_ratio' in c]
for f in ct_features[:10]:
    nan_pct = df[f].isna().sum() / len(df)
    print(f'{f}: {nan_pct:.1%} NaN')
"
```

## Expected Outcomes After Fixes

1. **Reduce NaN features**: From 75% to <10%
2. **Proper scoring**: Features ranked by actual predictive value
3. **Better retention**: 10-20% of cross-timeframe features retained
4. **Final count**: 10-30 cross-timeframe features in final selection

