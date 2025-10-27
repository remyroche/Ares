# Cross-Timeframe Feature Generation Fix Summary

## Problem
Cross-timeframe features were not being generated in the interaction generation step. The issue was that Phase 2 was pruning all cross-timeframe features because they were never being created in the first place.

## Root Cause
The `_phase1_generate_variants_optimized` function was returning the original features directly without generating:
1. Variant features (base, volnorm, vwap, trend_adj)
2. Cross-timeframe features (with 3x, 6x, 9x, 27x lookback ratios)

This occurred because when hardware optimization was enabled, the code took the "optimized" path which was actually a shortcut that skipped all feature generation.

## Solution
Modified `_phase1_generate_variants_optimized` to delegate to the full `_phase1_generate_variants` function, which properly:
1. Generates variant features using `generate_all_variants_optimized`
2. Calls `_generate_cross_timeframe_features` to create cross-timeframe ratio features
3. Combines both into the final feature set

## Changes Made
**File**: `src/training/steps/pre_training/feature_generation_interaction_generation_step.py`

**Function**: `_phase1_generate_variants_optimized` (lines 1155-1185)

**Change**: Replaced the shortcut logic with a delegation to the full variant generation:

```python
async def _phase1_generate_variants_optimized(
    self, 
    generated_features: pd.DataFrame,
    top_features_by_category: Dict,
    lookback_optimization: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Optimized Phase 1: Generate normalized variants with hardware optimization.
    
    Uses chunked processing, parallel feature generation, and VectorBT optimization.
    Note: For now, we delegate to the full variant generation to ensure cross-timeframe features are created.
    """
    tprint_info("🚀 Starting optimized variant generation")
    
    # Delegate to the full variant generation to ensure cross-timeframe features are created
    # TODO: Optimize this later for hardware acceleration while maintaining cross-timeframe generation
    return await self._phase1_generate_variants(
        generated_features, top_features_by_category, lookback_optimization, config
    )
```

## Debug Logging Added
Also added enhanced debug logging to track cross-timeframe feature generation and pruning:
- Logs cross-timeframe feature count before and after pruning
- Shows sample feature names for debugging
- Warns when all cross-timeframe features are pruned

## Expected Outcome
After this fix, the pipeline should now:
1. Generate ~200-300 variant features (63 base features × ~3-4 variants each)
2. Generate cross-timeframe features with names like `{feature_name}_3x_ratio`, `{feature_name}_6x_ratio`, etc.
3. Combine to create 500-1000+ total features before pruning
4. Phase 2 pruning should retain some cross-timeframe features

## Testing
Run the pipeline to verify:
```bash
python3 src/launcher/ares_launcher.py --step feature_generation_interaction_generation_step --symbol ETHUSDT --execution-mode light
```

Look for log messages showing:
- "🔄 CROSS-TIMEFRAME FEATURES GENERATION"
- "Cross-timeframe features found before pruning: X"
- "Sample cross-timeframe features: [...]"

## Next Steps
1. ✅ Verify cross-timeframe features are being generated - **DONE** (460 generated)
2. ✅ Monitor Phase 2 pruning to ensure reasonable retention - **DONE** (0 retained - found issues)
3. Optimize the hardware acceleration path while maintaining cross-timeframe generation
4. **URGENT**: Fix NaN generation issue (75% of cross-timeframe features are all NaN)
5. **URGENT**: Enable proper composite score calculation (currently all scores = 1.0)
6. Add explicit protection for cross-timeframe features during pruning

## Update: Pruning Analysis Completed

Detailed tracking revealed two critical issues:

### Issue 1: 75% of Cross-Timeframe Features are ALL NaN (345 out of 460)
- Root cause: `_generate_extended_timeframe_feature` fails to recalculate features properly
- These are removed as "problematic features" before clustering
- Affects features like: `volume_price_trend_base_3x_ratio`, etc.

### Issue 2: Composite Scores Not Calculated (All = 1.0)
- All 694 features have composite score = 1.0
- Features ranked alphabetically when scores tied
- Cross-timeframe features rank #232-#691 (all below cutoff #104)
- Root cause: Hardcoded `composite_scores = {col: 1.0 for col in ...}` in Phase 2

### Detailed Analysis
See `CROSS_TIMEFRAME_PRUNING_ANALYSIS.md` for comprehensive analysis and solutions.

### Verification
Cross-timeframe features ARE calculated from scratch (not rolling windows):
- Uses `_recalculate_feature_with_period` with extended lookback
- Properly applies variant transformations (volnorm, vwap, trend_adj)
- Creates ratio between base and extended timeframe versions
