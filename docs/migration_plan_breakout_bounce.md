# ml_breakout_bounce_regime_step Migration Plan

## Current State (After Cleanup)
- Deleted `_train_regime_classifier` (old non-OOF version)
- Deleted `_train_breakout_classifier` (old non-OOF version)  
- `_train_regime_classifier_oof` exists and uses StandardizedXGBTrainer
- `_train_2stage_model` exists but is NOT currently being called

## Required Changes

### 1. Architecture Change
Current: Uses `_train_regime_classifier_oof` for multi-class regime classification
Target: Use `_train_2stage_model` for resistance/support scalar prediction

### 2. Training Migration
**Stage 1: Binary Classification (break vs bounce)**
- Replace `xgb.XGBClassifier` with `StandardizedXGBTrainer`
- Config: `objective="binary:logistic"`, `task_type="classification"`
- Use OOF predictions instead of train/val/test split
- Remove manual time-series CV (handled by OOF windows)

**Stage 2: Regression (quality/forward returns)**
- Replace `xgb.XGBRegressor` with `StandardizedXGBTrainer`
- Config: `objective="reg:squarederror"`, `task_type="regression"`
- Use OOF predictions for quality scores
- Train on confidence-weighted samples

### 3. Output Simplification
Current outputs: `breakout_scalar_resistance`, `breakout_scalar_support` (derived from 3-class model)
Target outputs:
- `resistance_scalar`: 0 = certain breakout, 0.5 = trap/chop, 1 = certain bounce
- `support_scalar`: same logic
- Values closer to 0 or 1 = higher confidence/stronger magnitude
- NaN if on opposite side (not within x2 ATR)

### 4. Integration Steps
1. Update main execute() flow to call `_train_2stage_model` instead of `_train_regime_classifier_oof`
2. Migrate `_train_2stage_model` to use StandardizedXGBTrainer for both stages
3. Update scalar calculation logic to match new output format
4. Remove `_train_regime_classifier_oof` function
5. Update reporting to reflect new scalar outputs
6. Update artifact saving for new model structure

## Implementation Complexity
- File size: ~8000 lines
- Multiple interdependencies between functions
- Requires careful testing to ensure backward compatibility
- Need to preserve all feature engineering logic

## Recommendation
This migration requires:
1. Dedicated testing environment
2. Side-by-side comparison of old vs new outputs
3. Performance validation on historical data
4. Gradual rollout with feature flags
