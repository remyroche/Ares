# Regime Models Training Enhancements Summary

## Overview

This document summarizes the enhancements made to `regime_models_training` and `regime_ensemble_training` components to improve model stability and add comprehensive metrics for regime classification models.

## 1. Temporal Smoothness Penalty & Soft Labels

### Implementation
- **Location**: `src/utils/ml_common/evaluation/regime_temporal_metrics.py`
- **Features Added**:
  - `calculate_temporal_smoothness_penalty()`: Penalizes flipping predictions across consecutive bars
    - Formula: `L = L_CE + α * Σ_t 1[y_t != y_{t+1}]`
    - Default α: 0.1 (configurable)
  - `create_soft_labels()`: Creates probability vectors instead of 1-hot labels
    - Supports label smoothing (default: 0.1)
    - Can use cluster assignment confidence if available

### Usage
Both components now support:
- Temporal smoothness penalty calculation during evaluation
- Soft label creation (configuration flags: `enable_soft_labels`, `soft_label_smoothing`)

## 2. Smoothed Features & Rolling Aggregates

### Implementation
- **Location**: `src/utils/ml_common/feature_engineering/feature_smoothing.py`
- **Features Added**:
  - `add_smoothed_features()`: Adds rolling mean and std features
    - Window sizes: [3, 5, 7] (configurable)
    - Creates `feature_ma{window}` and `feature_std{window}` variants
  - `apply_ewm_smoothing()`: Applies exponential weighted moving average
    - Default alpha: 0.3

### Integration
- Both training components automatically add smoothed features when `enable_smoothed_features=True`
- Features are added after initial feature generation
- Window sizes configurable via `smoothing_window_sizes` parameter

## 3. Increased Tree Model Stability Parameters

### Changes Made
- **ExtraTrees**: 
  - `min_samples_split`: 2 → 5
  - `min_samples_leaf`: 1 → 5
- **LightGBM Meta-learner**:
  - `min_child_samples`: 20 → 50
  - Added `min_data_in_leaf`: 50
- **HPO Search Spaces**:
  - `min_child_samples`: Updated to [50, 150] range (was [20, 100])

### Rationale
Increased minimum leaf sizes force models to generalize rather than memorize noise, reducing jittery predictions.

## 4. Transition-Aware HPO Metrics

### Implementation
- **Location**: `src/utils/ml_common/optimization/transition_aware_scoring.py`
- **Features Added**:
  - `create_transition_aware_scorer()`: Composite scorer balancing accuracy and stability
    - Formula: `score = accuracy_weight * accuracy - stability_weight * (transition_rate + smoothness_penalty)`
    - Default weights: 0.7 accuracy, 0.3 stability
  - `create_multi_objective_scorer()`: Returns multiple metrics for Pareto optimization
    - Returns: accuracy, mean_episode_length, transition_rate, smoothness_penalty, etc.

### Usage
HPO can now use transition-aware metrics to optimize for both accuracy and stability:
```python
from src.utils.ml_common.optimization.transition_aware_scoring import create_transition_aware_scorer

scorer = create_transition_aware_scorer(alpha=0.1, accuracy_weight=0.7, stability_weight=0.3)
hpo_result = hpo_optimizer.optimize(
    model_factory=create_model,
    X=X_train,
    y=y_train,
    cv_folds=3,
    scoring=scorer,
    n_trials=20
)
```

## 5. Comprehensive Metrics Calculation

### Implementation
- **Location**: `src/utils/ml_common/evaluation/regime_temporal_metrics.py`
- **Class**: `RegimeTemporalMetricsCalculator`

### Metrics Provided

#### 1️⃣ Accuracy/Classification Metrics
- **Accuracy / Balanced Accuracy**: Fraction of correctly predicted bars
- **Precision / Recall per class**: Individual class reliability
- **F1-score**: Combines precision and recall (good for minority classes)
- **Log-loss / Cross-entropy**: Calibrated probability outputs

#### 2️⃣ Temporal/Stability Metrics
- **Mean Episode Length (MEL)**: Average consecutive bars in same state
- **Transition Rate (TR)**: Number of regime switches per unit time
- **Short Episode Count**: Episodes shorter than minimum desired length
- **Switch False Positive Rate (SFPR)**: Fraction of switches that immediately revert
- **Entropy / Confidence**: Average predicted class probability distribution entropy

#### 3️⃣ Regime-Persistence Metrics
- **Stability Index**: Fraction of time in persistent episodes (>N bars)
- **Persistence Ratio**: Average episode length / total episodes
- **Lag to Detection**: Average time to detect true regime change
- **Episode Purity**: Proportion of bars in episode matching true regime

### Integration
Both training components now calculate comprehensive metrics during evaluation:
```python
comprehensive_metrics = self.temporal_metrics_calc.calculate_comprehensive_metrics(
    y_test, y_pred, y_pred_proba
)

model_metrics = {
    'classification': comprehensive_metrics['classification'],
    'temporal': comprehensive_metrics['temporal'],
    'persistence': comprehensive_metrics['persistence'],
    'smoothness_penalty': calculate_temporal_smoothness_penalty(y_pred, alpha=0.1)
}
```

## 6. Enhanced Report Generation

### Updates
- **Location**: `src/training/steps/market_analysis/components/regime_models_training.py`
- **Method**: `_generate_text_report()`

### Report Sections Added
1. **Classification Metrics Section**:
   - Accuracy, Balanced Accuracy
   - Precision, Recall, F1-Score (weighted)
   - Log Loss (if available)

2. **Temporal/Stability Metrics Section**:
   - Mean Episode Length
   - Transition Rate
   - Short Episode Count
   - Switch False Positive Rate
   - Entropy and Confidence

3. **Regime-Persistence Metrics Section**:
   - Stability Index
   - Persistence Ratio
   - Lag to Detection
   - Episode Purity

### Report Format
Reports now include comprehensive metrics in both dictionary and human-readable text formats.

## Configuration Options

### New Configuration Flags
Both components support:
- `enable_temporal_smoothing`: Enable temporal smoothness penalty (default: True)
- `temporal_smoothing_alpha`: Smoothness penalty weight (default: 0.1)
- `enable_soft_labels`: Enable soft label creation (default: True)
- `soft_label_smoothing`: Label smoothing factor (default: 0.1)
- `enable_smoothed_features`: Enable smoothed feature generation (default: True)
- `smoothing_window_sizes`: Window sizes for smoothing (default: [3, 5, 7])

## Files Modified

1. **New Files Created**:
   - `src/utils/ml_common/evaluation/regime_temporal_metrics.py`
   - `src/utils/ml_common/feature_engineering/feature_smoothing.py`
   - `src/utils/ml_common/optimization/transition_aware_scoring.py`

2. **Files Modified**:
   - `src/training/steps/market_analysis/components/regime_models_training.py`
     - Added temporal metrics calculator initialization
     - Added smoothed feature generation
     - Updated model configurations (increased min_child_samples, etc.)
     - Enhanced evaluation with comprehensive metrics
     - Updated report generation
   
   - `src/training/steps/market_analysis/components/regime_ensemble_training.py`
     - Added temporal metrics calculator initialization
     - Added smoothed feature generation
     - Updated HPO search spaces
     - Enhanced evaluation with comprehensive metrics

## Usage Examples

### Using Transition-Aware HPO
```python
from src.utils.ml_common.optimization.transition_aware_scoring import create_transition_aware_scorer

# Create transition-aware scorer
scorer = create_transition_aware_scorer(
    alpha=0.1,
    accuracy_weight=0.7,
    stability_weight=0.3
)

# Use in HPO
hpo_result = hpo_optimizer.optimize(
    model_factory=create_model,
    X=X_train,
    y=y_train,
    cv_folds=3,
    scoring=scorer,
    n_trials=20
)
```

### Calculating Comprehensive Metrics
```python
from src.utils.ml_common.evaluation.regime_temporal_metrics import RegimeTemporalMetricsCalculator

calculator = RegimeTemporalMetricsCalculator(min_episode_length=3)
metrics = calculator.calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba)

print(f"Accuracy: {metrics['classification']['accuracy']:.4f}")
print(f"Mean Episode Length: {metrics['temporal']['mean_episode_length']:.2f}")
print(f"Stability Index: {metrics['persistence']['stability_index']:.4f}")
```

## Benefits

1. **Improved Stability**: Models produce more stable predictions with fewer unnecessary transitions
2. **Better Metrics**: Comprehensive assessment of model performance beyond accuracy
3. **Flexible Configuration**: Easy to tune stability vs. accuracy tradeoffs
4. **Enhanced Reporting**: Detailed reports help understand model behavior
5. **Transition-Aware Optimization**: HPO can optimize for both accuracy and stability

## Next Steps

1. **Tune Parameters**: Adjust `alpha`, `accuracy_weight`, and `stability_weight` based on validation results
2. **Monitor Metrics**: Track temporal and persistence metrics across training runs
3. **Experiment with Window Sizes**: Try different smoothing window sizes for your use case
4. **Use Multi-Objective HPO**: Consider Pareto optimization for complex tradeoffs

## Notes

- Temporal smoothness penalty is calculated during evaluation, not directly in tree model training (which optimizes per-sample accuracy)
- Soft labels can be used for models that support probability vector targets (e.g., neural networks)
- Smoothed features are additive and don't replace original features
- All new metrics are optional and can be disabled via configuration flags
