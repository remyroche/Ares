# Training Metrics Collection Implementation

## Summary

Implemented comprehensive metrics collection system for the unified training pipeline that tracks performance throughout the entire training process for all four training modes: `train_analyst_base`, `train_analyst_ensemble`, `train_tactician_base`, and `train_tactician_ensemble`.

## Changes Made

### 1. Created Training Metrics Collector (`training_metrics_collector.py`)

**Location:** `/workspace/src/training/steps/models_training/core/training_metrics_collector.py`

**Key Features:**
- **Pre-HPO Metrics Collection**: Captures baseline performance before hyperparameter optimization
  - Accuracy, Precision, Recall, F1-score (for classification)
  - MSE, MAE, RMSE, R² (for regression)
  - Cross-validation metrics across all folds

- **Fold Stability Tracking**: Measures performance consistency across CV folds
  - Coefficient of Variation (CV) for each metric
  - Min-max range across folds
  - Standard deviation tracking

- **Post-HPO Metrics Collection**: Captures improved performance after optimization
  - Same metrics as pre-HPO
  - Improvement calculations (absolute and relative)
  - Best hyperparameters tracking

- **Risk-Reward Metrics**:
  - Risk-Reward Ratio (RRR)
  - Sharpe Ratio
  - Sortino Ratio (downside risk-adjusted return)

- **Comprehensive Report Generation**:
  - Markdown format
  - Automatic saving to `outcomes/` directory
  - Includes all metrics, fold stability, and feature importance

### 2. Updated Model Trainer (`model_trainer.py`)

**Changes:**
- Integrated `TrainingMetricsCollector`
- Added 4-phase training process:
  1. **Phase 1**: Collect pre-HPO baseline metrics
  2. **Phase 2**: Run hyperparameter optimization (if enabled)
  3. **Phase 3**: Train final model with optimized parameters
  4. **Phase 4**: Collect post-HPO metrics

- Added `_optimize_hyperparameters()` method:
  - Supports LightGBM and CatBoost
  - Uses random search with configurable trials
  - Tracks HPO time and best parameters

### 3. Updated Ensemble Trainer (`ensemble_trainer.py`)

**Changes:**
- Integrated `TrainingMetricsCollector`
- Added 6-phase ensemble training process:
  1. **Phase 1**: Train individual base models
  2. **Phase 2**: Generate out-of-fold predictions
  3. **Phase 3**: Collect pre-HPO metrics for meta-learner
  4. **Phase 4**: Train meta-learner
  5. **Phase 5**: Collect post-HPO metrics for meta-learner
  6. **Phase 6**: Calculate ensemble metrics

- Added `_create_meta_learner_model()` helper method

## Metrics Collected

### Training Metrics
- **Pre-HPO Metrics**:
  - Mean accuracy/R² across folds
  - Standard deviation across folds
  - Fold stability (CV, range)
  
- **Post-HPO Metrics**:
  - Improved accuracy/R²
  - Reduced variance
  - Better fold stability

### Performance Metrics
- **Accuracy Metrics**:
  - Classification: Accuracy, Precision, Recall, F1-score
  - Regression: MSE, MAE, RMSE, R²

- **Stability Metrics**:
  - Coefficient of Variation per metric
  - Min-max range per metric
  - Standard deviation per metric

- **Risk-Reward Metrics**:
  - Risk-Reward Ratio (mean/std of scores)
  - Sharpe Ratio (excess return per unit risk)
  - Sortino Ratio (downside risk-adjusted return)

### HPO Tracking
- Number of trials
- Time spent on optimization
- Best hyperparameters found
- Improvement metrics (absolute and relative)

## Report Structure

Reports are saved to `outcomes/` directory with filename format:
```
{training_type}_{symbol}_{timestamp}_training_report.md
```

Example: `analyst_base_ETHUSDT_20250128_120000_training_report.md`

### Report Sections

1. **Header**
   - Session ID
   - Symbol, Timeframe
   - Total training time

2. **Data Quality**
   - Quality score
   - Number of samples and features

3. **Best Model Summary**
   - Best model name
   - Key metrics

4. **Model Training Details** (per model)
   - Pre-HPO metrics with fold stability
   - HPO information (trials, time, best params)
   - Post-HPO metrics with fold stability
   - Improvement metrics
   - Risk-Reward metrics
   - Top 10 important features

## Usage

The metrics collection is automatically integrated into the training pipeline. No additional configuration is needed. The system will:

1. Automatically collect metrics during training
2. Generate comprehensive reports
3. Save reports to `outcomes/` directory
4. Include report path in training results

### Configuration

HPO can be configured via `TrainingConfig`:

```python
config = TrainingConfig(
    role=TrainingRole.ANALYST,
    model_types=[ModelType.LIGHTGBM, ModelType.CATBOOST],
    enable_hyperparameter_optimization=True,  # Enable HPO
    cross_validation_folds=5,  # Number of CV folds
    custom_params={
        'hpo_n_trials': 50  # Number of HPO trials (default: 50)
    }
)
```

## Benefits

1. **Transparency**: Complete visibility into model performance at every stage
2. **Reproducibility**: All metrics, parameters, and configurations are tracked
3. **Performance Tracking**: Easy comparison between pre-HPO and post-HPO performance
4. **Stability Assessment**: Understand how consistent models are across folds
5. **Risk Management**: Risk-Reward metrics help assess model robustness
6. **Debugging**: Comprehensive metrics help identify training issues

## Example Report Output

```markdown
# Training Report: analyst_base

**Session ID:** analyst_base_ETHUSDT_20250128_120000
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-01-28T12:00:00
**Total Training Time:** 245.67s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 10,000
- **Features:** 150

---

## Best Model
**Name:** analyst_lightgbm

**Metrics:**
- r2_mean: 0.8245
- rmse_mean: 0.0153
...

---

## Model Training Details

### analyst_lightgbm (lightgbm)

#### Pre-HPO Metrics
- r2_mean: 0.7523
- r2_std: 0.0312
- rmse_mean: 0.0198
- rmse_std: 0.0021

**Fold Stability (Pre-HPO):**
- r2_cv: 0.0415
- r2_range: 0.0876
...

#### Hyperparameter Optimization
- **Trials:** 50
- **Time:** 123.45s
- **Best Parameters:** {'num_leaves': 45, 'learning_rate': 0.05, ...}

#### Post-HPO Metrics
- r2_mean: 0.8245
- r2_std: 0.0256
- rmse_mean: 0.0153
- rmse_std: 0.0015

**Fold Stability (Post-HPO):**
- r2_cv: 0.0310
- r2_range: 0.0654
...

**Improvement:**
- r2_abs_improvement: +0.0722
- r2_rel_improvement: +9.60%
...

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 32.15
- **Sharpe Ratio:** 28.67
- **Sortino Ratio:** 41.23

**Top 10 Important Features:**
- feature_1: 0.1234
- feature_2: 0.0987
...

---
```

## Future Enhancements

Potential improvements for future versions:

1. **Advanced HPO**: Integration with Optuna or Ray Tune
2. **Learning Curves**: Plot and track learning curves during training
3. **Ensemble Diversity**: Additional diversity metrics for ensemble models
4. **Model Comparison**: Side-by-side comparison of multiple models
5. **Interactive Reports**: HTML reports with charts and visualizations
6. **Metric Thresholds**: Configurable quality thresholds with alerts
7. **Time Series Metrics**: Specialized metrics for time series forecasting

## Notes

- Reports are automatically saved after each training session
- All metrics are computed on validation sets to avoid overfitting
- Risk-Reward metrics assume returns are normally distributed
- Fold stability metrics help identify overfitting or underfitting
- The system is backward compatible with existing training code
