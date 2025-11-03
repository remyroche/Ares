# HPO Integration Guide

This guide explains how the Hierarchical Parameter Optimization (HPO) system is integrated into the training pipeline.

## Overview

The HPO system automatically optimizes hyperparameters for all models (base and ensemble, analyst and tactician) using:

1. **Parameter ranges defined in YAML files** - Search spaces are configured in the YAML config files
2. **Hierarchical optimization** - Parameters are optimized in groups with dependencies
3. **Custom balanced score** - Uses `custom_balanced_score` from `evaluation_metrics.py` as the optimization metric
4. **Auto-save results** - Optimal parameters are automatically saved back to YAML files

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                  YAML Configuration Files                   │
│  (analyst_base_config.yaml, tactician_base_config.yaml,   │
│   analyst_ensemble_config.yaml, tactician_ensemble_config.yaml) │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│                   HPO Orchestrator                          │
│                  (hpo_config.py)                            │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Model Parameter Groups                              │  │
│  │  - LGBM: structure + regularization                  │  │
│  │  - CatBoost: structure + regularization              │  │
│  │  - TCN: architecture + training                      │  │
│  │  - GRU: architecture + training                      │  │
│  │  - ExtraTrees: structure + sampling                  │  │
│  │  - Meta-learner: meta_structure + meta_regularization│  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Custom Balanced Score Objective                     │  │
│  │  - Uses evaluation_metrics.py                        │  │
│  │  - Combines financial + statistical metrics          │  │
│  │  - Returns score in [0, 1]                           │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│          Hierarchical Parameter Optimizer                   │
│  (hierarchical_parameter_optimizer.py)                      │
│                                                              │
│  Round 1: Exploration (full search space)                  │
│    Group 1 → Coarse Grid → Fine Grid → TPE                │
│    Group 2 → Coarse Grid → Fine Grid → TPE                │
│                                                              │
│  Round 2: Refinement (narrowed search space ±15%)          │
│    Group 1 → Coarse Grid → Fine Grid → TPE                │
│    Group 2 → Coarse Grid → Fine Grid → TPE                │
│                                                              │
│  Final Refinement: Joint optimization of all params        │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│                YAML Config Updater                          │
│                                                              │
│  1. Backup original config                                  │
│  2. Update model params with optimal values                │
│  3. Save HPO metadata (score, trials, time)                │
│  4. Write updated config back to YAML                       │
└────────────────────────────────────────────────────────────┘
```

## YAML Configuration Format

Each model in the YAML files has an `hpo` section:

```yaml
base_models:
  lgbm:
    model_type: "LGBMRegressor"
    params:
      n_estimators: 1000
      learning_rate: 0.1
      max_depth: 8
      # ... other default params
    
    # HPO Configuration
    hpo:
      enabled: true  # Enable/disable HPO for this model
      n_rounds: 2    # Number of optimization rounds
      enable_final_refinement: true
      final_refinement_trials: 50
      
      # Search space definition
      search_space:
        max_depth:
          type: "int"
          low: 3
          high: 10
        learning_rate:
          type: "float"
          low: 0.01
          high: 0.3
          log: true  # Use logarithmic scale
        # ... other parameters
      
      # Optimal parameters (auto-updated by HPO)
      optimal_params: {}
      
      # HPO metadata (auto-updated)
      last_optimization:
        timestamp: "2025-10-31T12:00:00"
        best_score: 0.8523
        total_trials: 250
        total_time_seconds: 145.3
        n_rounds: 2
```

## Usage in Training Pipeline

### 1. In `unified_models_training_step.py`

The HPO is triggered automatically during training if `enable_hpo` is True in the config:

```python
# In execute() method
if config.get('enable_hpo', True) and training_data is not None:
    tprint_info("🔍 Performing hyperparameter optimization...")
    
    # Get model config
    model_config = yaml_config.get('analyst_config') or yaml_config.get('tactician_config')
    
    # Run HPO
    optimized_config = await self._perform_hyperparameter_optimization(
        training_data=training_data,
        targets=targets,
        model_config=model_config,
        config=config
    )
    
    # Use optimized config for training
    yaml_config[model_config_key] = optimized_config
```

### 2. HPO for Base Models

For base models (analyst_base, tactician_base), HPO optimizes each individual model:

```python
# Example: Optimizing LGBM in analyst_base
from src.training.steps.model_training.hpo_config import HPOOrchestrator
import lightgbm as lgb

orchestrator = HPOOrchestrator(
    config_file='src/training/steps/model_training/analyst_base_config.yaml',
    execution_mode='full'
)

result = orchestrator.run_hpo(
    model_name='lgbm',
    model_type='LGBMRegressor',
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    model_class=lgb.LGBMRegressor,
    is_classification=False
)

# Result is automatically saved back to YAML
```

### 3. HPO for Ensemble Models

For ensemble models (analyst_ensemble, tactician_ensemble), HPO optimizes the meta-learner:

```python
# Example: Optimizing meta-learner in analyst_ensemble
orchestrator = HPOOrchestrator(
    config_file='src/training/steps/model_training/analyst_ensemble_config.yaml',
    execution_mode='full'
)

result = orchestrator.run_hpo(
    model_name='meta_learner',
    model_type='stacker_lgbm_calibrated',
    X_train=base_model_outputs_train,  # Outputs from base models
    y_train=y_train,
    X_val=base_model_outputs_val,
    y_val=y_val,
    model_class=lgb.LGBMRegressor,
    is_classification=False
)
```

## Optimization Metric: Custom Balanced Score

The HPO uses `custom_balanced_score` from `evaluation_metrics.py` which combines:

### Financial Metrics (weighted 0.75)
- **Sharpe Ratio** (45%): Risk-adjusted return
- **Max Drawdown** (20%): Maximum loss from peak
- **Profit Factor** (30%): Ratio of wins to losses  
- **Total Return** (5%): Overall return

### Statistical Metrics (weighted 0.25)
- **F1 Score** (60%): Harmonic mean of precision and recall
- **Accuracy** (25%): Correct predictions percentage
- **R² Score** (15%): Explained variance

The score is normalized to [0, 1] with sample count penalty for small datasets.

## Execution Modes

### Full Mode (default)
- **Stages**: Coarse Grid → Fine Grid → TPE
- **Rounds**: 2 (exploration + refinement)
- **Final Refinement**: Yes (50 trials)
- **Time**: ~5-15 minutes per model

### Light Mode
- **Stages**: Coarse Grid only
- **Rounds**: 1 (exploration only)
- **Final Refinement**: No
- **Time**: ~1-3 minutes per model

Set execution mode in config:
```python
config = {
    'execution_mode': 'light',  # or 'full'
    'enable_hpo': True
}
```

## Parameter Groups and Optimization Order

### LGBM / CatBoost
1. **Group 1**: Structure + Learning Rate (priority 1)
   - `max_depth`, `learning_rate`
2. **Group 2**: Regularization + Subsampling (priority 2, depends on Group 1)
   - `num_leaves`, `reg_alpha`, `reg_lambda`, `subsample`, `colsample_bytree`, `min_child_samples`

### TCN
1. **Group 1**: Architecture (priority 1)
   - `num_filters`, `num_layers`, `kernel_size`, `dilation_base`
2. **Group 2**: Training (priority 2, depends on Group 1)
   - `dropout`, `learning_rate`, `batch_size`

### GRU
1. **Group 1**: Architecture (priority 1)
   - `hidden_units`, `num_layers`, `sequence_length`
2. **Group 2**: Training (priority 2, depends on Group 1)
   - `dropout`, `learning_rate`, `batch_size`

### ExtraTrees
1. **Group 1**: Structure (priority 1)
   - `n_estimators`, `max_depth`, `max_features`
2. **Group 2**: Sampling (priority 2, depends on Group 1)
   - `min_samples_split`, `min_samples_leaf`

### Meta-Learner (Ensemble)
1. **Group 1**: Meta Structure (priority 1)
   - `max_depth`, `learning_rate`
2. **Group 2**: Meta Regularization (priority 2, depends on Group 1)
   - `num_leaves`, `reg_alpha`, `reg_lambda`, `subsample`, `colsample_bytree`, `min_child_samples`

## Benefits

1. **Automatic**: HPO runs automatically during training if enabled
2. **Persistent**: Results saved back to YAML files for future use
3. **Efficient**: Hierarchical optimization reduces search space complexity
4. **Comprehensive**: Uses custom_balanced_score combining financial + statistical metrics
5. **Transparent**: Full logging and backup of original configs
6. **Flexible**: Easy to enable/disable per model and adjust search spaces

## Disabling HPO

To disable HPO for a specific model, set `enabled: false` in the model's HPO section:

```yaml
base_models:
  lgbm:
    hpo:
      enabled: false  # Skip HPO for this model
```

Or disable globally:

```python
config = {
    'enable_hpo': False  # Skip HPO for all models
}
```

## Accessing HPO Results

After training, optimal parameters are available in:

1. **YAML file**: `hpo.optimal_params` section
2. **Backup directory**: `hpo_backups/` with timestamped backups
3. **Training logs**: Detailed HPO progress and results

## Example: Complete Training with HPO

```python
import asyncio
from src.training.steps.model_training.unified_models_training_step import UnifiedModelsTrainingStep

async def run_training_with_hpo():
    step = UnifiedModelsTrainingStep()
    
    config = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'direction': 'long',
        'training_type': 'analyst_base',
        'execution_mode': 'full',
        'enable_hpo': True  # Enable HPO
    }
    
    result = await step.execute(config)
    
    if result['success']:
        print(f"✅ Training completed successfully")
        print(f"   HPO optimized parameters saved to YAML")
        print(f"   Models saved to: {result['artifacts']}")
    else:
        print(f"❌ Training failed: {result['error']}")

# Run training
asyncio.run(run_training_with_hpo())
```

## Troubleshooting

### HPO Taking Too Long
- Switch to `execution_mode: 'light'`
- Reduce `n_rounds` to 1
- Disable `enable_final_refinement`
- Narrow search spaces in YAML

### HPO Not Finding Good Parameters
- Widen search spaces in YAML
- Increase `n_rounds` to 3
- Increase `final_refinement_trials` to 100
- Check data quality and size

### YAML Not Updating
- Check file permissions
- Look for backups in `hpo_backups/`
- Check logs for errors
- Verify YAML file structure

## Files

- `hpo_config.py`: HPO configuration and orchestration
- `hierarchical_parameter_optimizer.py`: Core optimization engine
- `evaluation_metrics.py`: Custom balanced score implementation
- `unified_models_training_step.py`: Training pipeline integration
- `*_config.yaml`: Model configurations with HPO settings

## References

- [Hierarchical Optimizer Guide](src/utils/ml_common/optimization/HIERARCHICAL_OPTIMIZER_GUIDE.md)
- [Evaluation Metrics](src/utils/ml_common/optimization/shared_utils/evaluation_metrics.py)
- [Unified Training Pipeline](src/training/steps/models_training/unified_training_pipeline.py)

