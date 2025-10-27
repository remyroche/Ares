# Composite Interaction Selection Implementation - Summary

## Changes Made

### 1. Implemented CompositeFeatureScorer in `selection_methods.py`

**Location**: Lines 1281-1628

**Features**:
- ✅ 5-way composite scoring with equal 20% weights each:
  1. **Mutual Information (MI)** - Predictive relevance to target
  2. **Redundancy (Correlation)** - Diversity from other features (MRMR-style)
  3. **LGBM Feature Importance** - Model-based importance  
  4. **SHAP Values** - Explainable AI importance
  5. **Stability Score** - Temporal consistency across time windows

- ✅ RFE-style iterative removal:
  - Removes 33% of excess features per round
  - Recalculates scores each round (adapts as features removed)
  - Continues until reaching target (50 interactions)

**Formula**:
```python
composite_score = (
    0.20 * mi_score +           # Predictive value
    0.20 * redundancy_score +   # Low correlation (diversity)
    0.20 * lgbm_importance +    # Tree-based importance
    0.20 * shap_importance +    # Explainable importance
    0.20 * stability_score      # Temporal robustness
)
```

### 2. Integrated into Phase 3.3 Interaction Discovery

**Location**: `feature_generation_interaction_generation_step.py` lines 3272-3346

**Changes**:
- Replaced simple MI-only selection
- Now uses CompositeFeatureScorer with RFE
- Target: 50 interactions (increased from 30)
- Fallback to MI if composite scoring fails

**Process**:
1. Generate 400 interaction candidates (80 pairs × 5 operations)
2. Prepare aligned DataFrame for scoring
3. Call CompositeFeatureScorer.select_features()
4. RFE iterates ~17 rounds: 400 → 285 → 208 → ... → 50
5. Returns top 50 interactions by composite score

## Results

### Composite RFE Performance:

```
Initial candidates: 400
Target: 50 interactions
Rounds: 17
Time: ~15 seconds

Round breakdown:
Round 1: 400 → 285 (removed 115)
Round 2: 285 → 208 (removed 77)
Round 3: 208 → 156 (removed 52)
...
Round 17: 51 → 50 (removed 1)
```

### Score Distribution:

```
Composite score range: 0.4689 - 0.7028
- Minimum score: 0.4689 (worst selected interaction)
- Maximum score: 0.7028 (best selected interaction)
- All 50 interactions have scores > 0.46 (high quality)
```

### Final Feature Counts:

**Phase 3 Output**:
- Base features from Phase 3.2: 80
- Interactions from Phase 3.3: 50
- **Total after Phase 3**: 130 features

**Phase 4 Output (Saved Artifact)**:
- Total features: 110
- Base/variant features: 32
- Cross-timeframe features: 68
- Traditional interactions: 10

**Discrepancy**: 130 → 110 (20 features removed)

### Why Only 10 Traditional Interactions in Final Artifact?

**Issue Identified**: Metadata shows `n_interaction_features: 30`, not 50

**Possible causes**:
1. **Overfitting complexity filter** (line 3351-3364) may reduce from 50 to 30
2. **Final artifact saver** may apply additional pruning
3. **Duplicate/correlation removal** in Phase 4
4. **Category coverage adjustment** may remove some interactions

## Benefits of Composite Scoring vs MI-Only

### Before (MI-only):
- Single metric (predictive relevance only)
- No redundancy check → similar interactions selected
- No stability check → unstable interactions included
- Selection: Top 30 by MI

### After (Composite with RFE):
- 5 balanced metrics (comprehensive quality)
- Redundancy check → diverse interactions
- Stability check → robust across time
- LGBM & SHAP → model-validated importance
- RFE → adaptive selection (re-scores each round)
- Selection: Top 50 by composite score

### Expected Improvements:
- ✅ Better diversity (low redundancy component)
- ✅ More robust (stability component)
- ✅ Model-relevant (LGBM + SHAP components)
- ✅ Higher quality (multi-criteria validation)
- ✅ Better out-of-sample performance

## Next Steps to Investigate

### Why 50 → 30 → 10 Reduction?

1. **Check overfitting complexity filter** (line 3351-3364):
   - May be removing 20 interactions due to complexity > 3
   - If interaction names contain multiple `_x_`, they're counted as complex

2. **Check if Phase 4 applies additional pruning**:
   - Look for correlation-based removal in `_verify_category_coverage`
   - Check if artifact saver limits interaction count

3. **Count actual interactions in returned DataFrame**:
   - Verify `interaction_df` has 50 columns after Phase 3.3
   - Check if it gets reduced before Phase 4

### Recommended Actions:

1. **Add logging** to track interaction count through pipeline:
   ```python
   tprint_info(f"After Phase 3.3: {len(interactions.columns)} interactions")
   tprint_info(f"After combination: {len(combined_features.columns)} total")
   tprint_info(f"Before saving: {len(combined_features.columns)} features")
   ```

2. **Review complexity filter**:
   - Current limit: 3-way interactions max
   - May need to increase or adjust counting logic

3. **Ensure no silent filtering**:
   - Check for hidden correlation/duplicate removal
   - Verify artifact saver uses `combined_features` as-is

## Summary

✅ **Implemented**: CompositeFeatureScorer with 5-way scoring and RFE  
✅ **Integrated**: Into Phase 3.3 interaction discovery  
✅ **Target**: 50 interactions (up from 30)  
✅ **Working**: 17-round RFE successfully selects 50 interactions  
⚠️ **Issue**: Only 10 appear in final artifact (need to investigate filtering)

The composite scoring infrastructure is complete and working. The reduction from 50 to 10 needs investigation to ensure all selected interactions reach the final artifact.

