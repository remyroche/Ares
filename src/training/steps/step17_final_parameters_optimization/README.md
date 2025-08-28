# Step 12: Hyperparameter Optimization

## Overview

Step 12: Final Parameters Optimization is a comprehensive hyperparameter optimization system that uses Optuna to optimize trading strategy parameters. This step comes after confidence calibration (Step 11) and before walk-forward validation (Step 13).

## Key Features

### 🎯 **Multi-Objective Optimization**
- Optimizes multiple performance metrics simultaneously
- Uses Pareto front analysis for balanced solutions
- Supports weighted objective combinations

### 🔧 **Advanced Search Spaces**
- **Confidence Thresholds**: Analyst, tactician, and ensemble confidence levels
- **Volatility Parameters**: Target volatility, lookback periods, multipliers
- **Position Sizing**: Kelly criterion, confidence-based scaling
- **Risk Management**: Stop losses, trailing stops, drawdown limits
- **Ensemble Parameters**: Model weights, agreement thresholds
- **Regime-Specific**: Market regime adaptation parameters
- **Timing Parameters**: Cooldown periods, trade timing

### 📊 **Comprehensive Evaluation**
- **Performance Metrics**: Win rate, profit factor, Sharpe ratio, max drawdown
- **Risk Metrics**: VaR, CVaR, volatility, Sortino ratio
- **Trade Analysis**: Average win/loss, consecutive wins/losses
- **Composite Scoring**: Weighted combination of multiple metrics

### 🚀 **Advanced Features**
- **Warm Start**: Resume optimization from previous results
- **Parallel Execution**: Multi-core optimization support
- **Early Stopping**: Prune unpromising trials early
- **Persistence**: Save optimization progress to database
- **Validation**: Comprehensive result validation

## Implementation Guide

### 1. Basic Usage

```python
from src.training.steps.step12_final_parameters_optimization import FinalParametersOptimizationStep

# Create step instance
config = {
    "symbol": "ETHUSDT",
    "exchange": "BINANCE",
    "data_dir": "data/training"
}

step = FinalParametersOptimizationStep(config)
await step.initialize()

# Execute optimization
training_input = {
    "symbol": "ETHUSDT",
    "exchange": "BINANCE",
    "data_dir": "data/training"
}

pipeline_state = {}
result = await step.execute(training_input, pipeline_state)

print(f"Optimization completed: {result['status']}")
```

### 2. Configuration

#### Hyperparameter Configuration
```python
from src.training.steps.step12_final_parameters_optimization.hyperparameter_optimization_config import get_hyperparameter_config

config = get_hyperparameter_config()

# Get specific search space
confidence_space = config.get_search_space("confidence_thresholds")
print(f"Confidence parameters: {len(confidence_space.parameters)}")

# Get optimization plan
plan = config.get_optimization_plan()
print(f"Total trials: {plan['summary']['total_trials']}")
```

#### Evaluation Configuration
```python
evaluation_config = {
    "risk_free_rate": 0.02,
    "confidence_level": 0.95,
    "min_trades_for_evaluation": 10,
    "performance_thresholds": {
        "min_win_rate": 0.4,
        "min_profit_factor": 1.2,
        "max_drawdown": 0.25,
        "min_sharpe_ratio": 0.5,
    }
}
```

### 3. Search Space Configuration

#### Confidence Thresholds
```python
confidence_params = {
    "analyst_confidence_threshold": {
        "type": "float",
        "min": 0.5,
        "max": 0.95,
        "step": 0.02
    },
    "tactician_confidence_threshold": {
        "type": "float", 
        "min": 0.5,
        "max": 0.95,
        "step": 0.02
    },
    "ensemble_confidence_threshold": {
        "type": "float",
        "min": 0.5,
        "max": 0.95,
        "step": 0.02
    }
}
```

#### Position Sizing Parameters
```python
position_params = {
    "base_position_size": {
        "type": "float",
        "min": 0.01,
        "max": 0.2,
        "step": 0.01
    },
    "kelly_multiplier": {
        "type": "float",
        "min": 0.1,
        "max": 0.5,
        "step": 0.05
    },
    "confidence_based_scaling": {
        "type": "categorical",
        "choices": [True, False]
    }
}
```

### 4. Evaluation Engine

#### Basic Evaluation
```python
from src.training.steps.step12_final_parameters_optimization.evaluation_engine import create_evaluation_engine

# Create evaluation engine
engine = create_evaluation_engine(evaluation_config)

# Evaluate parameters
parameters = {
    "analyst_confidence_threshold": 0.7,
    "tactician_confidence_threshold": 0.65,
    "ensemble_confidence_threshold": 0.75,
    "base_position_size": 0.05,
    "stop_loss_atr_multiplier": 2.0,
}

calibration_results = {"calibration_data": "test"}
metrics = engine.evaluate_parameters(parameters, calibration_results)

print(f"Win Rate: {metrics.win_rate:.3f}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
print(f"Max Drawdown: {metrics.max_drawdown:.3f}")
```

#### Composite Scoring
```python
composite_score = engine.calculate_composite_score(metrics)
print(f"Composite Score: {composite_score:.3f}")
```

### 5. Multi-Objective Optimization

```python
# Multi-objective optimization for confidence thresholds
def objective(trial):
    params = {
        "analyst_confidence_threshold": trial.suggest_float("analyst_confidence_threshold", 0.5, 0.95, step=0.02),
        "tactician_confidence_threshold": trial.suggest_float("tactician_confidence_threshold", 0.5, 0.95, step=0.02),
        "ensemble_confidence_threshold": trial.suggest_float("ensemble_confidence_threshold", 0.5, 0.95, step=0.02),
    }
    
    # Evaluate multiple objectives
    win_rate = evaluate_win_rate(params)
    profit_factor = evaluate_profit_factor(params)
    sharpe_ratio = evaluate_sharpe_ratio(params)
    max_drawdown = evaluate_max_drawdown(params)
    
    return win_rate, profit_factor, sharpe_ratio, -max_drawdown

# Create multi-objective study
study = optuna.create_study(
    directions=["maximize", "maximize", "maximize", "maximize"],
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.HyperbandPruner()
)

study.optimize(objective, n_trials=100, timeout=1800)
```

### 6. Advanced Optuna Integration

#### Custom Sampler
```python
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler

# TPE Sampler (default)
sampler = TPESampler(seed=42)

# Random Sampler
sampler = RandomSampler(seed=42)

# CMA-ES Sampler
sampler = CmaEsSampler(seed=42)
```

#### Custom Pruner
```python
from optuna.pruners import HyperbandPruner, MedianPruner, PercentilePruner

# Hyperband Pruner (default)
pruner = HyperbandPruner(min_resource=1, max_resource=100)

# Median Pruner
pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

# Percentile Pruner
pruner = PercentilePruner(percentile=25.0, n_startup_trials=5, n_warmup_steps=10)
```

### 7. Storage and Persistence

#### SQLite Storage
```python
storage_url = "sqlite:///data/optimization_storage/optuna_studies.db"

study = optuna.create_study(
    storage=storage_url,
    study_name="confidence_thresholds_optimization",
    direction="maximize",
    load_if_exists=True
)
```

#### Resume Optimization
```python
# Load existing study
study = optuna.load_study(
    study_name="confidence_thresholds_optimization",
    storage=storage_url
)

# Continue optimization
study.optimize(objective, n_trials=50)
```

### 8. Validation and Monitoring

#### Result Validation
```python
# Validate optimization results
validation_passed = await step._validate_optimization_results(optimization_results)

if not validation_passed:
    print("⚠️ Optimization results validation failed")
    # Use fallback parameters
```

#### Performance Monitoring
```python
# Monitor optimization progress
for trial in study.trials:
    print(f"Trial {trial.number}: {trial.value}")
    print(f"Parameters: {trial.params}")
```

### 9. Integration with Training Pipeline

#### Step Orchestrator Integration
```python
from src.training.step_orchestrator import StepOrchestrator

orchestrator = StepOrchestrator("ETHUSDT", "BINANCE", "data/training")

# Execute Step 12
config = {"symbol": "ETHUSDT", "exchange": "BINANCE", "data_dir": "data/training"}
success = await orchestrator.execute_step("step12_final_parameters_optimization", config)

if success:
    print("✅ Step 12 completed successfully")
else:
    print("❌ Step 12 failed")
```

#### Pipeline State Management
```python
# Access optimization results in subsequent steps
pipeline_state = {
    "final_parameters": {
        "confidence_thresholds": {...},
        "volatility_parameters": {...},
        "position_sizing_parameters": {...},
        "risk_management_parameters": {...},
        "ensemble_parameters": {...},
        "regime_specific_parameters": {...},
        "timing_parameters": {...}
    },
    "optimization_report": {
        "optimization_metadata": {...},
        "optimization_summary": {...},
        "recommendations": [...]
    }
}
```

## Output Files

### 1. Optimization Results
- `data/training/BINANCE_ETHUSDT_final_parameters.pkl`: Pickle file with all optimization results
- `data/training/BINANCE_ETHUSDT_final_parameters_summary.json`: JSON summary
- `data/training/BINANCE_ETHUSDT_optimization_report.json`: Detailed report

### 2. Storage Database
- `data/optimization_storage/optuna_studies.db`: SQLite database with all studies

### 3. Example Output Structure
```json
{
  "confidence_thresholds": {
    "optimized_parameters": {
      "analyst_confidence_threshold": 0.72,
      "tactician_confidence_threshold": 0.68,
      "ensemble_confidence_threshold": 0.78
    },
    "pareto_front_size": 15,
    "best_objectives": [0.65, 1.8, 1.2, -0.15],
    "optimization_method": "multi_objective_optuna",
    "n_trials": 100,
    "optimization_date": "2024-01-15T10:30:00"
  },
  "volatility_parameters": {
    "optimized_parameters": {
      "target_volatility": 0.12,
      "volatility_lookback_period": 25,
      "volatility_multiplier": 1.2
    },
    "best_score": 0.85,
    "optimization_method": "optuna",
    "n_trials": 50
  }
}
```

## Performance Considerations

### 1. Optimization Time
- **Single Objective**: 30-60 minutes per parameter category
- **Multi-Objective**: 1-2 hours per parameter category
- **Total Time**: 4-8 hours for complete optimization

### 2. Resource Requirements
- **CPU**: Multi-core support for parallel trials
- **Memory**: 2-4 GB RAM for large datasets
- **Storage**: 1-2 GB for optimization database

### 3. Optimization Strategies
- **Fast Mode**: Reduce trials, use data subsampling
- **Thorough Mode**: More trials, full dataset
- **Balanced Mode**: Default settings

## Troubleshooting

### Common Issues

#### 1. Insufficient Trades for Evaluation
```python
# Increase minimum trades threshold
evaluation_config["min_trades_for_evaluation"] = 20
```

#### 2. Optimization Not Converging
```python
# Increase number of trials
search_space.n_trials = 200

# Adjust timeout
search_space.timeout_seconds = 3600
```

#### 3. Memory Issues
```python
# Reduce parallel trials
config.global_config["n_jobs"] = 2

# Use data subsampling
subsample_fraction = 0.5
```

#### 4. Database Connection Issues
```python
# Use file-based storage
storage_url = "sqlite:///./optuna_studies.db"

# Or use in-memory storage for testing
storage_url = "sqlite:///:memory:"
```

### Debug Mode
```python
from src.utils.logger import setup_logging
setup_logging()

# Enable Optuna logging
import optuna
optuna.logging.set_verbosity(optuna.logging.DEBUG)
```

## Best Practices

### 1. Parameter Selection
- Start with most impactful parameters
- Use domain knowledge to set reasonable ranges
- Consider parameter interactions

### 2. Evaluation Strategy
- Use out-of-sample data for evaluation
- Implement proper cross-validation
- Consider regime-specific optimization

### 3. Optimization Strategy
- Start with single-objective optimization
- Progress to multi-objective for fine-tuning
- Use warm start for incremental optimization

### 4. Validation Strategy
- Validate results on unseen data
- Check for overfitting
- Monitor parameter stability

## Advanced Features

### 1. Custom Evaluation Functions
```python
def custom_evaluation_function(parameters, calibration_results):
    # Implement custom evaluation logic
    performance = simulate_trading(parameters)
    return calculate_custom_metric(performance)
```

### 2. Parameter Constraints
```python
def objective_with_constraints(trial):
    params = {
        "param1": trial.suggest_float("param1", 0, 1),
        "param2": trial.suggest_float("param2", 0, 1),
    }
    
    # Check constraints
    if params["param1"] + params["param2"] > 1.5:
        raise optuna.TrialPruned()
    
    return evaluate_parameters(params)
```

### 3. Early Stopping
```python
from optuna.callbacks import EarlyStoppingCallback

callbacks = [EarlyStoppingCallback(patience=10)]
study.optimize(objective, n_trials=100, callbacks=callbacks)
```

## Integration Examples

### 1. Automated Pipeline
```python
async def run_optimization_pipeline():
    # Step 11: Confidence Calibration
    await run_step11()
    
    # Step 12: Hyperparameter Optimization
    await run_step12()
    
    # Step 13: Walk-Forward Validation
    await run_step13()
```

### 2. Continuous Optimization
```python
async def continuous_optimization():
    while True:
        # Run optimization
        await run_step12()
        
        # Wait for next cycle
        await asyncio.sleep(3600)  # 1 hour
```

### 3. A/B Testing Integration
```python
async def ab_testing_optimization():
    # Optimize parameters for variant A
    params_a = await optimize_parameters(variant="A")
    
    # Optimize parameters for variant B
    params_b = await optimize_parameters(variant="B")
    
    # Compare results
    comparison = compare_variants(params_a, params_b)
```

## Conclusion

Step 12: Hyperparameter Optimization provides a comprehensive framework for optimizing trading strategy parameters. By following this guide, you can implement robust hyperparameter optimization that improves strategy performance while maintaining risk management standards.

For more advanced usage, refer to the Optuna documentation and the individual module files in this directory.