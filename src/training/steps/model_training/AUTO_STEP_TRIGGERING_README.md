# Automatic Step Triggering for Model Training Pipeline

This system ensures that when one model training step finishes, it automatically triggers the next step, creating a seamless execution flow.

## Overview

The model training pipeline consists of 5 steps that run in sequence:

1. **analyst_model_training** - Per-regime individual model training with HPO, saving, and metrics
2. **analyst_ensemble_training** - Per-regime ensemble training with HPO, saving, and metrics
3. **tactician_lookback_optimization** - Lookback optimization for tactician models
4. **tactician_models_training** - All-regime individual model training with HPO, saving, and metrics
5. **tactician_ensemble_training** - All-regime ensemble training with HPO, saving, and metrics

## How It Works

The automatic triggering system is built on top of the existing `sub_pipeline.py` infrastructure. When a step completes successfully, the system automatically calls the next step in the sequence, ensuring continuous execution without manual intervention.

## Usage

### Method 1: Using the Auto Step Trigger Module

```python
from src.training.steps.model_training.auto_step_trigger import (
    auto_execute_all_model_training_steps,
    auto_execute_from_step
)

# Execute all steps automatically from the beginning
result = await auto_execute_all_model_training_steps(
    symbol="ETHUSDT",  # Any symbol: BTCUSDT, ADAUSDT, etc.
    exchange="BINANCE",  # Any exchange: BYBIT, KRAKEN, etc.
    timeframe="1m",  # Any timeframe: 5m, 15m, 1h, etc.
    force_rerun=True
)

# Or execute from a specific step (will trigger all subsequent steps)
result = await auto_execute_from_step(
    step_name='tactician_models_training',  # Will trigger steps 4-5
    symbol="ETHUSDT",
    exchange="BINANCE",
    timeframe="1m",
    force_rerun=True
)
```

### Method 2: Using the Sub-Pipeline Directly

```python
from src.training.steps.model_training.sub_pipeline import (
    ModelTrainingSubPipeline,
    SubPipelineConfig,
    ExecutionMode
)

# Create configuration
config = SubPipelineConfig(
    mode=ExecutionMode.FULL,
    symbol="ETHUSDT",
    exchange="BINANCE",
    timeframe="1m",
    data_dir="historical_data",
    force_rerun=True,
    single_stage_only=False  # Enable automatic triggering
)

# Create and execute sub-pipeline
sub_pipeline = ModelTrainingSubPipeline(config)

# Execute all steps from the beginning
result = await sub_pipeline.execute_all_steps_from_start(config)

# Execute from a specific step
result = await sub_pipeline.execute_sub_pipeline_with_next('tactician_lookback_optimization', config)
```

## Key Features

### Automatic Triggering
- When one step completes successfully, the next step is automatically triggered
- No manual intervention required between steps
- Seamless execution flow

### Error Handling
- If a step fails, the execution stops and reports the failure
- Detailed error messages and execution summaries
- Comprehensive logging throughout the process

### Flexible Execution
- Can start from any step in the sequence
- Can execute all steps from the beginning
- Supports different execution modes (FULL, LIGHT, BLANK)

### Progress Monitoring
- Real-time progress updates
- Execution time tracking
- Success/failure statistics
- Detailed execution summaries

### Integration
- Works with existing pipeline infrastructure
- Compatible with ares_launcher parameters
- Supports different symbols, exchanges, and timeframes

## Configuration Options

### Basic Configuration
```python
config = {
    'force_rerun': True,           # Force rerun existing artifacts
    'parallel_processing': True,   # Enable parallel processing
    'validation_enabled': True,    # Enable validation
    'monitoring_enabled': True,    # Enable monitoring
    'fast_mode': False,            # Enable fast mode
}
```

### Advanced Configuration
```python
config = {
    'custom_params': {
        'analyst_model_training': {
            'hpo_iterations': 100,
            'cv_folds': 5,
            'models': ['xgboost', 'lightgbm', 'catboost']
        },
        'tactician_lookback_optimization': {
            'max_lookback': 50,
            'optimization_method': 'bayesian',
            'n_trials': 50
        },
        'ensemble_training': {
            'ensemble_methods': ['voting', 'stacking', 'bagging'],
            'meta_learner': 'xgboost'
        }
    }
}
```

## Example Output

```
🚀 Starting automatic execution of all 5 model training steps
================================================================================
📋 Steps to be executed automatically:
   1. analyst_model_training - Per-regime individual model training with HPO, saving, and metrics
   2. analyst_ensemble_training - Per-regime ensemble training with HPO, saving, and metrics
   3. tactician_lookback_optimization - Lookback optimization for tactician models
   4. tactician_models_training - All-regime individual model training with HPO, saving, and metrics
   5. tactician_ensemble_training - All-regime ensemble training with HPO, saving, and metrics
================================================================================

🎉 Automatic execution completed
✅ Successful steps: 5/5
⏱️ Total execution time: 1247.32 seconds
```

## Files Created/Modified

1. **sub_pipeline.py** - Enhanced with automatic triggering capabilities
2. **auto_step_trigger.py** - New module with convenience functions
3. **example_auto_triggering.py** - Example usage demonstrations

## Testing

Run the example script to test the automatic triggering:

```bash
python src/training/steps/model_training/example_auto_triggering.py
```

## Benefits

1. **Seamless Execution**: No manual intervention required between steps
2. **Error Recovery**: Clear failure reporting and error handling
3. **Flexibility**: Can start from any step in the sequence
4. **Monitoring**: Comprehensive progress tracking and logging
5. **Integration**: Works with existing pipeline infrastructure
6. **Scalability**: Supports different execution modes and configurations
7. **Agnostic**: Works with any symbol, exchange, and timeframe

## Integration with ares_launcher

The system is designed to work seamlessly with `ares_launcher`, accepting parameters like:

- **symbol**: Trading symbol (e.g., 'ETHUSDT', 'BTCUSDT')
- **exchange**: Exchange name (e.g., 'BINANCE', 'BYBIT')
- **timeframe**: Data timeframe (e.g., '1m', '5m', '1h')

These parameters are passed through without hardcoded defaults, making the system truly agnostic to the specific trading pair or analysis timeframe.

## Model Types and Timeframes

Different timeframes are optimized for different model types:

- **1m timeframe**: High-frequency models (Analyst focus) - Fast decision making
- **5m timeframe**: Medium-term models (Tactician focus) - Balanced approach
- **1h timeframe**: Long-term models (Strategic focus) - Long-term planning

The system automatically adapts to the chosen timeframe and optimizes the model training accordingly.

This system ensures that your model training pipeline runs smoothly from start to finish, automatically progressing through all 5 steps when each completes successfully.
