# Regime Ensemble Performance Analysis

**Date:** November 9, 2025  
**Current Accuracy:** 30.13% (Ensemble) vs Expected: >50% (Base Models)

## Critical Issues

### 1. Severe Class Imbalance Bias
- **Problem**: Model predicts Regime 2 for 96.4% of test samples (1014/1052)
- **Impact**: Only 2 out of 6 regimes are ever predicted (Regime 2 and 4)
- **Balanced Accuracy**: 17.08% (vs 30.13% raw) - confirms severe imbalance

### 2. Low Prediction Confidence
- **Average Top-1 Confidence**: 32.45% (should be >50% for good predictions)
- **High Uncertainty**: 97% of samples classified as "low confidence"
- **Confidence Gap**: Only 13% difference between top-1 and top-2 predictions

### 3. Poor Per-Regime Performance
| Regime | Precision | Recall | F1-Score | Support | Issue |
|--------|-----------|--------|----------|---------|-------|
| 0 | 0.00% | 0.00% | 0.00% | 134 | Never predicted |
| 1 | 0.00% | 0.00% | 0.00% | 91 | Never predicted |
| 2 | 29.59% | 96.15% | 45.25% | 312 | Over-predicted (majority class) |
| 3 | 0.00% | 0.00% | 0.00% | 52 | Never predicted |
| 4 | 44.74% | 6.34% | 11.11% | 268 | Under-predicted |
| 5 | 0.00% | 0.00% | 0.00% | 195 | Never predicted |

## Root Causes

### 1. Base Model Quality
- **Only 2 base models actually used** (report says 3, but metrics show 2)
- Base models likely also biased toward Regime 2
- Need to verify base model individual accuracies

### 2. Feature Engineering Issues
- **53 enhanced features** created from 40 base meta-features
- Added features may be introducing noise rather than signal
- Class-specific features (section 8 in `_create_enhanced_meta_features`) depend on training labels

### 3. Calibration Problems
- **Isotonic calibration** applied but may be making predictions worse
- Calibration on imbalanced data can amplify majority class bias
- Low confidence scores suggest calibration isn't helping

### 4. Training Data Imbalance
Test set distribution:
- Regime 2: 312 samples (29.7%)
- Regime 4: 268 samples (25.5%)
- Regime 5: 195 samples (18.5%)
- Regime 0: 134 samples (12.7%)
- Regime 1: 91 samples (8.7%)
- Regime 3: 52 samples (4.9%)

## Recommended Fixes (Priority Order)

### Priority 1: Improve Base Models
```python
# Check individual base model accuracies
# If base models are <40% accurate, ensemble can't help
# Action: Investigate why base models are performing poorly
```

### Priority 2: Simplify Feature Engineering
```python
# Remove or reduce enhanced features
# Test with just base meta-features (40 features)
# Remove class-specific features that depend on training labels
```

### Priority 3: Adjust Class Weighting
```python
# Current: class_weight='balanced'
# Try: Custom weights that penalize majority class more
class_weights = {
    0: 3.0,  # Boost minority classes
    1: 3.0,
    2: 0.5,  # Penalize majority class
    3: 4.0,
    4: 1.0,
    5: 2.0
}
```

### Priority 4: Disable or Fix Calibration
```python
# Option A: Disable calibration entirely
# Option B: Use sigmoid calibration instead of isotonic
# Option C: Calibrate per-class separately
```

### Priority 5: Add Focal Loss
```python
# Use focal loss to handle class imbalance
# Focuses training on hard-to-classify examples
# Reduces over-confidence on majority class
```

## Diagnostic Commands

### Check Base Model Performance
```bash
# Find regime_models_training reports
ls -lt outcomes/regime_models_training_report_*.md | head -1
```

### Verify Training Data Distribution
```python
# Check if training data has same imbalance
# If yes, need better sampling strategy
```

### Test Without Enhanced Features
```python
# Modify _train_stacker_lgbm_calibrated to skip _create_enhanced_meta_features
# Train with just base meta-features (40 features)
```

## Next Steps

1. **Investigate base model performance** - If they're <40% accurate, fix them first
2. **Test simplified ensemble** - Remove enhanced features, test with just base predictions
3. **Implement better class balancing** - Custom weights or SMOTE
4. **Consider alternative meta-learners** - Try XGBoost with scale_pos_weight or CatBoost with auto_class_weights

## Expected Improvements

With proper fixes:
- **Target Accuracy**: >50% (should beat best base model)
- **Balanced Accuracy**: >40% (currently 17%)
- **All Regimes Predicted**: Each regime should have >0% recall
- **Confidence**: Average >50% for top-1 predictions
