# Enhanced MLflow Integration Guide

## Overview

The Enhanced MLflow Integration ensures that all models in the `enhanced_training_manager` pipeline are properly associated with the required metadata throughout the entire training process. This provides complete traceability and reproducibility of all training runs.

## Required Metadata Associations

Every model and artifact logged to MLflow is automatically associated with:

1. **Asset** - The trading asset/symbol (e.g., "ETHUSDT")
2. **Exchange** - The trading exchange (e.g., "BINANCE") 
3. **Lookback Period** - The data lookback period used for training (e.g., "2_years")
4. **Project Version** - The current project version (from `ARES_VERSION`)
5. **Date** - The training date (automatically set to current timestamp)

## Key Components

### 1. Enhanced MLflow Utilities (`src/utils/mlflow_utils.py`)

Core utility functions that ensure all MLflow operations include the required metadata:

#### Core Functions

- `log_enhanced_training_metadata()` - Logs the core required metadata
- `log_model_with_metadata()` - Logs models with all required associations
- `log_metrics_with_metadata()` - Logs metrics with enhanced metadata
- `log_params_with_metadata()` - Logs parameters with enhanced metadata
- `log_artifacts_with_metadata()` - Logs artifacts with enhanced metadata

#### Helper Functions

- `extract_training_metadata()` - Extracts metadata from configuration
- `validate_run_metadata()` - Validates that a run has all required metadata
- `get_enhanced_run_metadata()` - Retrieves enhanced run information

### 2. Enhanced MLflow Manager (`src/utils/enhanced_mlflow_integration.py`)

A comprehensive manager class that provides a high-level interface for MLflow operations:

```python
from src.utils.enhanced_mlflow_integration import EnhancedMLflowManager

# Initialize the manager
mlflow_manager = EnhancedMLflowManager(config)

# Start a run with enhanced metadata
run_id = mlflow_manager.start_run(step_name="step6_hmm_based_training")

# Log models with metadata
mlflow_manager.log_model(
    model=trained_model,
    model_name="hmm_regime_model",
    model_type="hmm",
    additional_metadata={"n_components": 3}
)

# Log metrics with metadata
mlflow_manager.log_metrics(
    metrics={"accuracy": 0.85, "precision": 0.82},
    additional_metadata={"validation_split": "test"}
)

# End the run
mlflow_manager.end_run()
```

## Integration Points

### 1. Enhanced Training Manager Pipeline

The enhanced training manager automatically uses the enhanced MLflow integration in key steps:

#### Step 21: Saving (`src/training/steps/step21_saving.py`)

```python
# Enhanced MLflow saving with all required metadata
await self._save_to_mlflow(training_summary, symbol, exchange)

# This automatically logs:
# - Enhanced training metadata (asset, exchange, lookback_period, project_version, date)
# - Parameters with metadata
# - Metrics with metadata  
# - Artifacts with metadata
```

#### Model Trainer (`src/training/model_trainer.py`)

```python
# Enhanced metadata logging for model training
log_enhanced_training_metadata(
    asset=symbol,
    exchange=exchange,
    lookback_period=lookback_period,
    run_id=run.info.run_id,
    additional_metadata={
        "model_type": hpo_model_type,
        "timeframe": "1m",
        "pipeline_step": "model_training",
    }
)

# Enhanced parameter logging
log_params_with_metadata(
    params=best_params,
    asset=symbol,
    exchange=exchange,
    lookback_period=lookback_period,
    run_id=run.info.run_id,
    additional_metadata={
        "optimization_type": "optuna_hpo",
        "n_trials": hpo_trials,
    }
)
```

#### Enhanced LM Optimizer (`src/training/enhanced_lm_optimizer.py`)

```python
# Enhanced logging for hyperparameter optimization
log_params_with_metadata(
    params=all_params,
    asset=symbol,
    exchange=exchange,
    lookback_period=lookback_period,
    run_id=run.info.run_id,
    additional_metadata={
        "optimization_type": "enhanced_lm_optimizer",
        "trial_type": "hyperparameter_optimization",
    }
)
```

### 2. Step-by-Step Integration

Each step in the enhanced training manager can use the enhanced MLflow integration:

```python
from src.utils.enhanced_mlflow_integration import log_step_metadata, log_model_performance

# Log step metadata
log_step_metadata(
    config=config,
    step_name="step6_hmm_based_training",
    step_data=step_results,
    run_id=run_id
)

# Log model performance
log_model_performance(
    config=config,
    model_name="hmm_regime_model",
    model_type="hmm",
    performance_metrics={"accuracy": 0.85, "precision": 0.82},
    run_id=run_id
)
```

## Usage Examples

### Example 1: Basic Model Logging

```python
from src.utils.mlflow_utils import log_model_with_metadata

# Log a trained model with all required metadata
log_model_with_metadata(
    model=trained_model,
    model_name="analyst_model",
    asset="ETHUSDT",
    exchange="BINANCE", 
    lookback_period="2_years",
    additional_metadata={
        "model_type": "analyst",
        "training_algorithm": "lightgbm",
        "feature_count": 150
    }
)
```

### Example 2: Metrics Logging

```python
from src.utils.mlflow_utils import log_metrics_with_metadata

# Log performance metrics with enhanced metadata
log_metrics_with_metadata(
    metrics={
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.78,
        "f1_score": 0.80
    },
    asset="ETHUSDT",
    exchange="BINANCE",
    lookback_period="2_years",
    additional_metadata={
        "validation_split": "test",
        "model_type": "tactician",
        "cross_validation_folds": 5
    }
)
```

### Example 3: Artifact Logging

```python
from src.utils.mlflow_utils import log_artifacts_with_metadata

# Log training artifacts with enhanced metadata
log_artifacts_with_metadata(
    local_path="data/training/feature_importance.png",
    artifact_path="plots/feature_importance.png",
    asset="ETHUSDT",
    exchange="BINANCE",
    lookback_period="2_years",
    additional_metadata={
        "artifact_type": "plot",
        "plot_type": "feature_importance",
        "model_type": "analyst"
    }
)
```

### Example 4: Using the Enhanced MLflow Manager

```python
from src.utils.enhanced_mlflow_integration import EnhancedMLflowManager

# Initialize manager
mlflow_manager = EnhancedMLflowManager(config)

# Start run for a specific step
run_id = mlflow_manager.start_run(step_name="step7_analyst_enhancement")

# Log multiple models
for model_name, model in trained_models.items():
    mlflow_manager.log_model(
        model=model,
        model_name=model_name,
        model_type="analyst",
        additional_metadata={"ensemble_size": len(trained_models)}
    )

# Log performance metrics
mlflow_manager.log_metrics(
    metrics=performance_metrics,
    additional_metadata={"validation_method": "time_series_split"}
)

# Log training summary
mlflow_manager.log_training_summary(
    summary=training_summary,
    additional_metadata={"step_completion_time": datetime.now().isoformat()}
)

# Validate run has all required metadata
is_valid = mlflow_manager.validate_current_run()
print(f"Run validation: {'PASSED' if is_valid else 'FAILED'}")

# End run
mlflow_manager.end_run()
```

## Validation and Quality Assurance

### Metadata Validation

The system includes built-in validation to ensure all required metadata is present:

```python
from src.utils.mlflow_utils import validate_run_metadata

# Validate a run has all required metadata
is_valid = validate_run_metadata(run_id)
if is_valid:
    print("✅ Run has all required metadata")
else:
    print("❌ Run is missing required metadata")
```

### Enhanced Run Information

Retrieve comprehensive run information including all metadata:

```python
from src.utils.mlflow_utils import get_enhanced_run_metadata

# Get enhanced run metadata
run_info = get_enhanced_run_metadata(run_id)
print(f"Asset: {run_info['asset']}")
print(f"Exchange: {run_info['exchange']}")
print(f"Lookback Period: {run_info['lookback_period']}")
print(f"Project Version: {run_info['project_version']}")
print(f"Training Date: {run_info['training_date']}")
```

## Configuration

The enhanced MLflow integration automatically extracts metadata from the enhanced training manager configuration:

```python
# Configuration keys that are automatically extracted:
config = {
    "trading_symbol": "ETHUSDT",      # -> asset
    "exchange_name": "BINANCE",       # -> exchange  
    "lookback_years": 2,              # -> lookback_period (2_years)
    "project_version": "0.1.0",       # -> project_version
    # ... other config
}
```

## Benefits

1. **Complete Traceability** - Every model is associated with its training context
2. **Reproducibility** - All training parameters and metadata are preserved
3. **Quality Assurance** - Built-in validation ensures no missing metadata
4. **Easy Querying** - Models can be filtered by asset, exchange, lookback period, etc.
5. **Version Control** - Project version tracking for model lineage
6. **Compliance** - Full audit trail for regulatory requirements

## Best Practices

1. **Always use enhanced functions** - Use `log_model_with_metadata()` instead of `mlflow.log_model()`
2. **Validate runs** - Use `validate_run_metadata()` to ensure completeness
3. **Include step context** - Add `step_name` and `pipeline_step` to additional metadata
4. **Use the manager class** - For complex operations, use `EnhancedMLflowManager`
5. **Log comprehensive metadata** - Include model type, training algorithm, feature count, etc.

## Troubleshooting

### Common Issues

1. **Missing metadata fields** - Ensure configuration contains required keys
2. **Validation failures** - Check that all required fields are present and not "Unknown"
3. **MLflow not available** - Install MLflow: `poetry add mlflow`

### Debug Information

Enable debug logging to see detailed metadata extraction and validation:

```python
import logging
logging.getLogger("src.utils.mlflow_utils").setLevel(logging.DEBUG)
```

This enhanced MLflow integration ensures that every model in the enhanced training manager pipeline is properly associated with all required metadata, providing complete traceability and reproducibility for all training operations.