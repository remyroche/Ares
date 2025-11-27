# ml_breakout_bounce_regime_step Migration Plan

## Status: IMPLEMENTED ✅

Migration completed on 2024-11-27.

## Changes Made

### 1. New `_train_breakout_classifier` Method ✅
Implemented a new 2-stage classifier using StandardizedXGBTrainer:

**Stage 1: Binary Classification (break vs bounce)**
- Uses `StandardizedXGBTrainer` with OOF predictions
- Config: `objective="binary:logistic"`, `task_type="classification"`
- Early stopping rounds: 20
- Retrain every 10 days, HPO every 30 days

**Stage 2: Regression (quality/forward returns)**
- Uses `StandardizedXGBTrainer` with OOF predictions
- Config: `objective="reg:squarederror"`, `task_type="regression"`
- Confidence-weighted samples based on Stage 1 predictions
- Early stopping rounds: 20

### 2. Output Format Updated ✅
New scalar outputs:
- `resistance_scalar`: 0 = certain breakout, 0.5 = trap/chop, 1 = certain bounce
- `support_scalar`: same logic
- Values closer to 0 or 1 = higher confidence/stronger magnitude
- NaN if on opposite side (not within x2 ATR)

Legacy columns maintained for backward compatibility:
- `breakout_scalar_resistance` → mirrors `resistance_scalar`
- `breakout_scalar_support` → mirrors `support_scalar`

### 3. Split Handling Fixed ✅
- Robust TZ-naive timestamp conversion for split_config
- Automatic fallback to percentage-based splits if temporal splits fail or are too small
- Minimum sample checks: train >= 100, val >= 50

### 4. Artifact Updates ✅
Core columns now include:
- `resistance_scalar`, `support_scalar` (new)
- `breakout_scalar_resistance`, `breakout_scalar_support` (legacy)
- All probability columns maintained

## Notes
- `_train_regime_classifier_oof` in MLPathRegimeStep is separate and unaffected
- `_train_2stage_model` exists for reference but the main path uses `_train_breakout_classifier`
- File size increased to ~8000+ lines due to comprehensive implementation

## Testing Checklist
- [ ] Run with temporal split_config
- [ ] Run with fallback to percentage splits
- [ ] Verify scalar outputs are in correct range [0, 1]
- [ ] Verify NaN on opposite sides
- [ ] Check backward compatibility with downstream consumers
