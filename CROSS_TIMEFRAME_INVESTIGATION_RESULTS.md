# Cross-Timeframe Feature Investigation Results

## Summary
Investigated both root causes of why 0 cross-timeframe features were retained:
1. **NaN Generation (75% of features)** - Added detailed debugging
2. **Composite Score Issue (all = 1.0)** - Implemented proper MI + stability calculation

## Investigation 1: NaN Generation in `_generate_extended_timeframe_feature`

### Root Causes Identified:

#### Issue 1A: Feature Type Not Supported
**Location**: `_recalculate_feature_with_period` (line 1620)

**Problem**: Limited feature mappings - only supports ~12 feature types:
```python
feature_mappings = {
    'rsi', 'sma', 'ema', 'bb', 'atr', 'stoch',  
    'williams', 'cci', 'macd', 'volume', 'volatility',
    'momentum', 'roc', 'vwap'
}
```

**Impact**: Most features don't match these patterns, causing recalculation to fail.

**Example Failing Features**:
- `volume_price_trend` - doesn't match any pattern
- `directional_signal` - doesn't match any pattern  
- `order_flow_imbalance_20` - doesn't match any pattern
- `analyst_volume_trend` - doesn't match any pattern
- `fibonacci_0.618_20_price_returns` - doesn't match any pattern

**Fix Applied**: Added comprehensive debug logging to identify which features fail:
```python
tprint_info(f"🔍 DEBUG: _recalculate_feature_with_period called")
tprint_info(f"    Feature: {feature_name}")
tprint_info(f"    Period: {period}")
if matched_key:
    tprint_info(f"    ✅ Matched pattern: '{key}'")
else:
    tprint_warning(f"    ⚠️ No simple pattern match")
    tprint_warning(f"    Trying FeatureCalculatorRegistry fallback...")
```

#### Issue 1B: Period Extraction Failures
**Location**: `_extract_period_from_feature_name` (line 1580)

**Problem**: Simple regex patterns don't work for complex feature names

**Patterns tried**:
- `_(\d+)$` - Feature ending with _number
- `_(\d+)_` - Feature with _number_ in middle
- `(\d+)_` - Feature starting with number_

**Example Failures**:
- `volume_price_trend` - no period in name
- `directional_signal` - no period in name
- Many analyst-generated features have no explicit period

**Fix Applied**: Added debug logging:
```python
if original_period is None:
    tprint_warning(f"⚠️ DEBUG: Could not extract period from feature name {base_feature_name}")
    tprint_warning(f"    Feature name pattern not recognized - needs period extraction")
    return None
```

#### Issue 1C: Extended Period Exceeds Data Length
**Location**: `_generate_extended_timeframe_feature` (line 1763)

**Problem**: With data length ~1920 and multipliers like 27x:
- Original period: 20
- Extended period: 20 * 27 = 540 (OK)
- But some features have larger base periods

**Example**:
- Feature with base period 100
- 27x multiplier → 2700 periods needed
- Only 1920 data points available
- Results in all NaN values

**Fix Applied**: Added data length check:
```python
if len(ohlcv_data) < extended_period:
    tprint_warning(f"⚠️ DEBUG: Extended period {extended_period} exceeds data length {len(ohlcv_data)}")
    tprint_warning(f"    Reducing to max available: {len(ohlcv_data) - 1}")
    extended_period = max(original_period, len(ohlcv_data) - 1)
```

### Changes Made to `_generate_extended_timeframe_feature`:

1. **Added entry logging**:
   - Logs original period, extended lookback, extended period
   - Shows calculation logic

2. **Added data length validation**:
   - Checks if extended period exceeds data
   - Reduces to max available if needed

3. **Added NaN detection**:
   - Checks if recalculated feature has NaN
   - Reports percentage of NaN values
   - Returns None if ALL NaN

4. **Added failure tracking**:
   - Logs when recalculation fails
   - Shows which feature types aren't supported

## Investigation 2: Composite Scores NOT Calculated (All = 1.0)

### Root Cause Identified:

**Location**: `_phase2_cheap_pruning` (line 2069)

**Problem**: Hardcoded to 1.0 instead of calculating:
```python
# BEFORE (WRONG):
composite_scores = {col: 1.0 for col in variant_features.columns}
```

**Impact**:
- All 694 features have identical score of 1.0
- Features ranked alphabetically instead of by predictive value
- Cross-timeframe features (starting with letters after 'd') rank lower
- Result: ALL cross-timeframe features below cutoff #104

### Solution Implemented:

Created `_calculate_composite_scores` method that:

1. **Calculates Mutual Information (MI) scores**:
   ```python
   mi_scores = mutual_info_regression(
       features_for_mi,
       target_aligned,
       random_state=42,
       n_neighbors=3
   )
   ```
   - Measures predictive value of each feature
   - Normalized to 0-1 range

2. **Calculates Stability scores**:
   ```python
   rolling_mean = feature_data.rolling(window=window_size).mean()
   rolling_std = feature_data.rolling(window=window_size).std()
   cv = rolling_std.mean() / (abs(rolling_mean.mean()) + 1e-8)
   stability = 1.0 / (1.0 + cv)
   ```
   - Measures feature consistency over time
   - Higher stability = more reliable

3. **Combines into composite score**:
   ```python
   composite_scores[col] = 0.6 * mi_dict[col] + 0.4 * stability_dict[col]
   ```
   - 60% weight on predictive value (MI)
   - 40% weight on stability
   - Range: 0-1

4. **Analyzes cross-timeframe scores**:
   - Compares CT mean vs All mean
   - Warns if CT scores significantly lower
   - Helps diagnose if CT features are inherently less predictive

### Changes Made:

**File**: `src/training/steps/pre_training/feature_generation_interaction_generation_step.py`

**Lines Changed**:
- Line 2068-2074: Replaced hardcoded scores with calculation call
- Lines 3879-4005: Added new `_calculate_composite_scores` method

**New Code**:
```python
# Calculate composite scores with MI and stability
tprint_info("="*80)
tprint_info("📊 CALCULATING COMPOSITE SCORES (MI + Stability)")
tprint_info("="*80)
composite_scores = self._calculate_composite_scores(
    variant_features, targets, feature_categories
)
```

## Expected Results After Fixes

### For NaN Generation:
- Debug logs will show exactly which features fail and why
- Can identify which feature types need to be added to mappings
- Can see which features have period extraction issues
- Can track which extended periods exceed data length

### For Composite Scores:
- Features will have diverse scores (not all 1.0)
- Rankings based on predictive value, not alphabetical order
- Cross-timeframe features get fair comparison
- Better features (higher MI + stability) ranked higher

### Retention Improvement:
**Before**:
- All scores = 1.0
- Alphabetical ranking
- CT features: #232-#691
- Cutoff: #104
- Retained: 0 CT features (0%)

**After (Expected)**:
- Scores: 0.01-1.0 based on MI + stability
- Predictive value ranking
- CT features: distributed across ranks
- Cutoff: #104  
- Retained: 10-30 CT features (10-25%)

## Next Steps

1. **Run with detailed logging**:
   ```bash
   python3 src/launcher/ares_launcher.py \
     --step feature_generation_interaction_generation_step \
     --symbol ETHUSDT \
     --execution-mode light \
     2>&1 | tee cross_timeframe_debug.log
   ```

2. **Analyze logs for**:
   - Which features fail recalculation
   - Period extraction success rate
   - MI score distribution
   - CT vs All score comparison

3. **Extend feature support**:
   - Add more patterns to `feature_mappings`
   - Improve period extraction regex
   - Add fallback to FeatureBank for unsupported types

4. **Monitor retention**:
   - Check if CT features get better scores
   - Verify some CT features survive cutoff
   - Aim for 10-25% CT feature retention

## Files Modified

1. `src/training/steps/pre_training/feature_generation_interaction_generation_step.py`:
   - Added comprehensive debugging to `_generate_extended_timeframe_feature`
   - Added detailed logging to `_recalculate_feature_with_period`
   - Added data length validation
   - Created `_calculate_composite_scores` method
   - Replaced hardcoded 1.0 scores with actual calculation

2. `src/training/utils/feature_selection/cheap_pruning.py`:
   - Added cross-timeframe tracking at every pruning step
   - Added score analysis and ranking details
   - Shows exactly where and why CT features are removed

## Testing

To verify fixes work:

```bash
# Run and check for DEBUG messages
python3 src/launcher/ares_launcher.py \
  --step feature_generation_interaction_generation_step \
  --symbol ETHUSDT \
  --execution-mode light \
  2>&1 | grep "DEBUG:\|COMPOSITE SCORE"

# Check if scores are diverse (not all 1.0)
# Look for "CT score stats:" and "All score stats:"

# Check retention improvement
# Look for "Cross-timeframe features in final selection"
```

