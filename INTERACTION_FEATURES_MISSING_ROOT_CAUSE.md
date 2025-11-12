# Root Cause: Interaction Features Not Included in Selection

## Problem
The selected features in the final feature selection report don't include any interaction features (no `_x_`, `_div_`, `_minus_`, cross-timeframe ratios, etc.).

## Root Cause Analysis

### Issue 1: Interaction Generation Step Not Run ❌
The `feature_generation_interaction_generation_step` has NOT been executed for the current pipeline run.

**Evidence:**
- Latest interaction artifacts are from October 27-30, 2025
- No recent `analyst_interaction_features` artifacts in the artifacts directory
- The command run was: `--feature_generation_final_feature_selection_step` (single step only)

### Issue 2: Pipeline Order Requirements
The feature generation pipeline has a specific order:

```python
FEATURE_GENERATION_STEPS = [
    'feature_generation_labeling_integration_step',           # Step 1
    'feature_generation_feature_generation_step',             # Step 2
    'feature_generation_period_lookback_optimization_step',   # Step 3
    'feature_generation_interaction_generation_step',         # Step 4 ⚠️ MISSING
    'feature_generation_final_feature_selection_step',        # Step 5
    'feature_generation_final_validation_step',               # Step 6
]
```

**The final feature selection step (Step 5) requires the interaction generation step (Step 4) to have run first.**

## What Happens When Interaction Step is Skipped

1. **Final feature selection loads artifacts:**
   - ✅ `generated_features` from Step 2 (feature_generation_step)
   - ❌ `analyst_interaction_features` from Step 4 (NOT AVAILABLE)

2. **Feature selection proceeds with only base features:**
   - Only ~327 base features available
   - No interaction features
   - No cross-timeframe ratios
   - No variant features

3. **Result:**
   - Selected features are all base features
   - Missing the engineered interaction features that could improve model performance

## Solution

### Option 1: Run Full Pipeline (Recommended)
```bash
python3 src/launcher/ares_launcher.py \
  --symbol ETHUSDT \
  --exchange binance \
  --timeframe 15m \
  --execution-mode blank
```

This runs all steps in order:
1. Labeling integration
2. Feature generation
3. Lookback optimization
4. **Interaction generation** ✅
5. Final feature selection ✅
6. Final validation

### Option 2: Run Interaction Generation + Final Selection
```bash
# Step 1: Run interaction generation
python3 src/launcher/ares_launcher.py \
  --run_analyst_interaction \
  --symbol ETHUSDT \
  --execution-mode blank

# Step 2: Run final feature selection
python3 src/launcher/ares_launcher.py \
  --feature_generation_final_feature_selection_step \
  --symbol ETHUSDT \
  --execution-mode blank
```

### Option 3: Run From Interaction Generation Onwards
```bash
python3 src/launcher/ares_launcher.py \
  --feature_generation_interaction_generation_step \
  --feature_generation_final_feature_selection_step \
  --symbol ETHUSDT \
  --execution-mode blank
```

## Code Fix Status

✅ **The code fix is correct** - The `_collect_features_from_previous_steps()` method now properly loads `analyst_interaction_features` when available.

❌ **The prerequisite step wasn't run** - The interaction generation step must be executed before final feature selection.

## Verification Steps

After running the interaction generation step, verify:

1. **Check for interaction artifacts:**
   ```bash
   find artifacts -name "*analyst_interaction_features*" -mtime -1
   ```

2. **Check artifact timestamp:**
   ```bash
   ls -lt artifacts/*analyst_interaction_features* | head -3
   ```

3. **Run final feature selection:**
   ```bash
   python3 src/launcher/ares_launcher.py \
     --feature_generation_final_feature_selection_step \
     --symbol ETHUSDT \
     --execution-mode blank
   ```

4. **Verify selected features include interactions:**
   ```bash
   grep -E "_x_|_div_|_minus_|_3x_ratio|_6x_ratio" \
     outcomes/final_feature_selection_outcome_report_*.md
   ```

## Expected Feature Types After Fix

Once interaction generation runs, the final selection should include:

### Base Features (from feature_generation_step):
- trend_score_14
- directional_signal
- enhanced_volatility_50
- support/resistance levels
- volume features

### Interaction Features (from interaction_generation_step):
- `feature1_x_feature2` (multiplications)
- `feature1_div_feature2` (divisions)
- `feature1_minus_feature2` (subtractions)
- `feature_3x_ratio` (3x cross-timeframe)
- `feature_6x_ratio` (6x cross-timeframe)
- `feature_volnorm` (volume normalized variants)
- `feature_vwap` (VWAP variants)
- `feature_trend_adj` (trend adjusted variants)

## Summary

The fix to load interaction features is **working correctly**, but the interaction generation step must be run first to create the artifacts. Running only the final feature selection step in isolation will not include interaction features because they don't exist yet.

**Action Required:** Run the full pipeline or at least run the interaction generation step before final feature selection.
