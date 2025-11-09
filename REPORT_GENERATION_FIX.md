# Training Report Generation Fix

## Issue
Training reports (`analyst_base_ETHUSDT_15m_long_report_*.md` and `*_metrics_*.json`) showed:
- **Total Models: 0**
- **Model names: []**
- Missing per-model performance metrics

However, the JSON metrics showed aggregate metrics (avg_mse, avg_mae, avg_r2, etc.), indicating models were trained but not captured in the report.

## Root Cause

The issue was in the data flow from training to reporting:

1. **ModelTrainer.train()** (`model_trainer.py`):
   - Trains multiple models (LightGBM, CatBoost, DEPTHWISE_CNN)
   - Stores models in `self._model_instances` dict
   - Returns `TrainingResult` with only a single `model` attribute (best model)
   - **Missing:** No `models` dict attribute for all trained models

2. **PipelineOrchestrator._execute_analyst_base_training()** (`pipeline_orchestrator.py` line 588):
   ```python
   'models': result.models if hasattr(result, 'models') else {},
   ```
   - Checks for `result.models` attribute
   - Falls back to empty dict `{}` when attribute doesn't exist

3. **Report Generation** (`unified_models_training_step.py` line 378):
   ```python
   models_trained=result.get('models', {})
   ```
   - Receives empty dict
   - Reports show 0 models trained

## Solution

Modified `ModelTrainer.train()` to include trained models in the result:

```python
# Extract trained models dict for reporting
trained_models = {
    model_type.value: self._model_instances.get(model_type.value)
    for model_type in self.config.model_types
    if model_type.value in self._model_instances
}

result = TrainingResult(
    success=len(training_results) > 0,
    model=best_model,
    metrics=overall_metrics,
    training_time=training_time,
    metadata={
        'models_trained': len(training_results),
        'role': self.config.role.value,
        'timeframe': self.config.timeframe,
        'individual_results': training_results,
        'comprehensive_metrics': all_model_metrics,
        'report_path': str(report_path),
        'trained_models': trained_models  # Add models dict
    }
)

# Add models attribute for backward compatibility
result.models = trained_models
```

## Files Modified
- `src/training/steps/models_training/core/model_trainer.py` (lines 234-258)

## Expected Result

After this fix, training reports will show:
- **Total Models:** 3 (or actual count)
- **Model Types:** lightgbm, depthwise_cnn, catboost
- **Per-Model Metrics:** Individual performance metrics for each model
- **Model Details:** Training time, parameters, feature importance per model

## Testing
Run the training again:
```bash
python3 src/launcher/ares_launcher.py --train-analyst-base --symbol ETHUSDT --execution-mode light
```

Check the generated reports in `outcomes/` directory for populated model information.
