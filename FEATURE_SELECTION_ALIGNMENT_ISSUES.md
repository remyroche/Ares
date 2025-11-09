# Feature Selection & Training Alignment Issues

## Problem Summary

The analyst base training is failing because of data misalignment between selected features and targets.

## Issues Identified

### 1. ✅ FIXED: Tactician Targets Loading in Analyst Mode
**Problem**: Code was loading tactician targets even in analyst mode.
**Fix Applied**: Modified `unified_models_training_step.py` line 1911-1932 to only load tactician targets when `training_type` contains "tactician".

### 2. 🔴 CRITICAL: Feature-Target Index Mismatch
**Problem**: 
- Selected features: 300 rows (May 30 - Aug 31, 2025)
- Training uses: 75 rows (after temporal splits/filtering)
- Result: Only 60 overlapping samples instead of expected 75

**Root Cause**:
- Feature selection saves 300 rows from light mode
- Training applies additional filtering (temporal splits, data validation, NaN removal)
- This reduces 300 rows → 75 rows, but with a different time range
- Only 60 rows overlap between the two

**Evidence**:
```
Selected features index: 2025-05-30 22:00:00 to 2025-08-31 22:00:00 (300 rows)
Training filtered to: 75 samples
LightGBM trains on: 60 samples (the intersection)
```

**The Real Issue**: Feature selection and training apply different data filters, causing index misalignment.

### 3. 🔴 HPO Failing with 0 Samples
**Problem**: `Found array with 0 sample(s)` during HPO cross-validation.
**Root Cause**: When splitting data for cross-validation, some folds have no overlapping indices between features and targets.

### 4. 🔴 Empty Performance Metrics in Report
**Problem**: The markdown report shows empty performance metrics sections.
**Root Cause**: Models aren't returning proper metrics, likely due to training failures from data misalignment.

## Solution

### Option 1: Run Complete Pipeline in Same Mode (RECOMMENDED)
Run the entire pipeline in the same execution mode:
```bash
# 1. Feature generation in light mode
python3 src/launcher/ares_launcher.py --feature-generation --symbol ETHUSDT --execution-mode light

# 2. Labeling in light mode  
python3 src/launcher/ares_launcher.py --labeling --symbol ETHUSDT --execution-mode light

# 3. Feature selection in light mode
python3 src/launcher/ares_launcher.py --feature_generation_final_feature_selection_step --symbol ETHUSDT --execution-mode light

# 4. Training in light mode
python3 src/launcher/ares_launcher.py --train-analyst-base --symbol ETHUSDT --execution-mode light
```

### Option 2: Modify Feature Selection to Save Targets
Modify `feature_generation_final_feature_selection_step.py` to also save the targets/labeled_data alongside the selected features, ensuring they have the same index.

### Option 3: Fix Training Data Loading
Modify `unified_models_training_step.py` to load targets from the SAME store as the selected features, ensuring index alignment.

## Current State

- ✅ Tactician target loading fixed
- 🔴 Feature-target alignment still broken
- 🔴 Only CatBoost generates predictions (LightGBM and DepthwiseCNN fail)
- 🔴 Training uses 60/75 samples due to index mismatch

## Next Steps

1. Run complete pipeline in light mode to ensure all artifacts have matching indices
2. Verify all 3 models (LightGBM, DepthwiseCNN, CatBoost) generate predictions
3. Check that performance metrics are properly generated in reports
