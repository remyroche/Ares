# Configuration Usage Guide

## Overview

This guide explains how to use the new configuration sections in `src/config.py` for enhanced hyperparameter optimization and computational optimization.

## Table of Contents

1. [Hyperparameter Optimization Configuration](#hyperparameter-optimization-configuration)
2. [Computational Optimization Configuration](#computational-optimization-configuration)
3. [Integration Examples](#integration-examples)
4. [Best Practices](#best-practices)
5. [Troubleshooting](#troubleshooting)

---

## Hyperparameter Optimization Configuration

### 1. Multi-Objective Optimization

The multi-objective optimization section allows you to optimize multiple performance metrics simultaneously.

#### Basic Configuration
```python
from src.config import CONFIG

# Access multi-objective settings
multi_obj_config = CONFIG["hyperparameter_optimization"]["multi_objective"]

# Check if enabled
if multi_obj_config["enabled"]:
    objectives = multi_obj_config["objectives"]  # ["sharpe_ratio", "win_rate", "profit_factor"]
    weights = multi_obj_config["weights"]  # {"sharpe_ratio": 0.50, "win_rate": 0.30, "profit_factor": 0.20}
```

#### Usage Example
```python
from src.training.multi_objective_optimizer import MultiObjectiveOptimizer

# Initialize with configuration
optimizer = MultiObjectiveOptimizer(
    config=CONFIG["hyperparameter_optimization"],
    market_data=your_market_data
)

# Run optimization
results = await optimizer.optimize(
    n_trials=100,
    timeout=3600  # 1 hour
)
```

### 2. Bayesian Optimization

Bayesian optimization uses probabilistic models to guide the search for optimal hyperparameters.

#### Configuration Access
```python
bayesian_config = CONFIG["hyperparameter_optimization"]["bayesian_optimization"]

# Key settings
sampling_strategy = bayesian_config["sampling_strategy"]  # "tpe", "random", "cmaes", "nsga2"
max_trials = bayesian_config["max_trials"]  # 500
patience = bayesian_config["patience"]  # 50
```

#### Usage Example
```python
from src.training.bayesian_optimizer import AdvancedBayesianOptimizer

# Initialize Bayesian optimizer
bayesian_optimizer = AdvancedBayesianOptimizer(
    config=CONFIG["hyperparameter_optimization"]["bayesian_optimization"],
    search_space=CONFIG["hyperparameter_optimization"]["search_spaces"]
)

# Run optimization
best_params = await bayesian_optimizer.optimize(
    objective_function=your_objective_function,
    n_trials=100
)
```

### 3. Adaptive Optimization

Adaptive optimization adjusts hyperparameters based on detected market regimes.

#### Market Regime Configuration
```python
adaptive_config = CONFIG["hyperparameter_optimization"]["adaptive_optimization"]

# Available regimes: "bull", "bear", "sideways", "sr", "candle"
regime_constraints = adaptive_config["regime_specific_constraints"]

# Example: Bull market constraints
bull_constraints = regime_constraints["bull"]
tp_range = bull_constraints["tp_multiplier_range"]  # [2.5, 5.0]
sl_range = bull_constraints["sl_multiplier_range"]  # [1.2, 2.5]
position_range = bull_constraints["position_size_range"]  # [0.10, 0.25]
```

#### Usage Example
```python
from src.training.adaptive_optimizer import AdaptiveOptimizer

# Initialize adaptive optimizer
adaptive_optimizer = AdaptiveOptimizer(
    config=CONFIG["hyperparameter_optimization"]["adaptive_optimization"],
    market_data=your_market_data
)

# Run regime-aware optimization
results = await adaptive_optimizer.optimize_regime_specific(
    current_regime="bull",
    n_trials=50
)
```

### 4. Search Spaces

Define the parameter ranges for optimization.

#### Model Hyperparameters
```python
search_spaces = CONFIG["hyperparameter_optimization"]["search_spaces"]
model_params = search_spaces["model_hyperparameters"]

# Access specific parameter ranges
learning_rate_range = model_params["learning_rate"]  # {"type": "float", "low": 1e-4, "high": 1e-1, "log": True}
max_depth_range = model_params["max_depth"]  # {"type": "int", "low": 3, "high": 15}

# Model types available
model_types = model_params["model_type"]["choices"]  # ["xgboost", "lightgbm", "catboost", "random_forest", "gradient_boosting", "tabnet", "transformer"]
```

#### Trading Parameters
```python
trading_params = search_spaces["trading_parameters"]

# Access trading parameter ranges
tp_multiplier_range = trading_params["tp_multiplier"]  # {"type": "float", "low": 1.2, "high": 10.0}
confidence_threshold_range = trading_params["confidence_threshold"]  # {"type": "float", "low": 0.6, "high": 0.95}
```

### 5. Optimization Schedules

Configure automated optimization schedules.

#### Schedule Configuration
```python
schedules = CONFIG["hyperparameter_optimization"]["optimization_schedules"]

# Daily optimization
daily_schedule = schedules["daily"]
if daily_schedule["enabled"]:
    time = daily_schedule["time"]  # "02:00"
    max_trials = daily_schedule["max_trials"]  # 100
    focus = daily_schedule["focus"]  # "quick_adaptation"
```

#### Usage Example
```python
from src.training.enhanced_optimization_orchestrator import EnhancedOptimizationOrchestrator

# Initialize orchestrator with schedules
orchestrator = EnhancedOptimizationOrchestrator(
    config=CONFIG["hyperparameter_optimization"]
)

# Run scheduled optimization
await orchestrator.run_scheduled_optimization(schedule_type="daily")
```

---

## Computational Optimization Configuration

### 1. Caching

Enable caching to avoid redundant computations.

#### Configuration
```python
caching_config = CONFIG["computational_optimization"]["caching"]

if caching_config["enabled"]:
    max_cache_size = caching_config["max_cache_size"]  # 1000
    cache_ttl = caching_config["cache_ttl"]  # 3600 (1 hour)
```

#### Usage Example
```python
from src.training.optimized_backtester import OptimizedBacktester

# Initialize with caching
backtester = OptimizedBacktester(
    market_data=your_market_data,
    config=CONFIG["computational_optimization"]
)

# Run cached backtest
result = backtester.run_cached_backtest(parameters)
```

### 2. Parallelization

Enable parallel processing for faster optimization.

#### Configuration
```python
parallel_config = CONFIG["computational_optimization"]["parallelization"]

if parallel_config["enabled"]:
    max_workers = parallel_config["max_workers"]  # 8
    chunk_size = parallel_config["chunk_size"]  # 1000
```

#### Usage Example
```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Get configuration
max_workers = CONFIG["computational_optimization"]["parallelization"]["max_workers"]

# Use ProcessPoolExecutor for parallel processing
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(optimization_task, params) for params in parameter_list]
    results = [future.result() for future in futures]
```

### 3. Early Stopping

Configure early stopping to avoid wasting resources on unpromising trials.

#### Configuration
```python
early_stopping_config = CONFIG["computational_optimization"]["early_stopping"]

if early_stopping_config["enabled"]:
    patience = early_stopping_config["patience"]  # 10
    min_trials = early_stopping_config["min_trials"]  # 20
```

#### Usage Example
```python
import optuna

# Create study with early stopping
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=min_trials,
        n_warmup_steps=patience
    )
)
```

### 4. Memory Management

Configure memory monitoring and cleanup.

#### Configuration
```python
memory_config = CONFIG["computational_optimization"]["memory_management"]

if memory_config["enabled"]:
    memory_threshold = memory_config["memory_threshold"]  # 0.8 (80%)
    cleanup_frequency = memory_config["cleanup_frequency"]  # 100
```

#### Usage Example
```python
import psutil
import gc

def check_memory_usage():
    memory_percent = psutil.virtual_memory().percent / 100
    memory_threshold = CONFIG["computational_optimization"]["memory_management"]["memory_threshold"]
    
    if memory_percent > memory_threshold:
        gc.collect()  # Force garbage collection
        return True
    return False
```

### 5. Progressive Evaluation

Evaluate parameters on data subsets for faster initial screening.

#### Configuration
```python
progressive_config = CONFIG["computational_optimization"]["progressive_evaluation"]

if progressive_config["enabled"]:
    stages = progressive_config["stages"]
    # [
    #     {"data_ratio": 0.1, "weight": 0.3},
    #     {"data_ratio": 0.3, "weight": 0.5},
    #     {"data_ratio": 1.0, "weight": 1.0}
    # ]
```

#### Usage Example
```python
def progressive_evaluate(parameters, data, stages):
    for stage in stages:
        data_ratio = stage["data_ratio"]
        weight = stage["weight"]
        
        # Use subset of data
        subset_data = data.sample(frac=data_ratio)
        score = evaluate_parameters(parameters, subset_data)
        
        # Apply weight
        weighted_score = score * weight
        
        # Early stopping if score is too low
        if weighted_score < threshold:
            return weighted_score
    
    return weighted_score
```

---

## Integration Examples

### 1. Complete Optimization Pipeline

```python
from src.training.enhanced_optimization_orchestrator import EnhancedOptimizationOrchestrator
from src.config import CONFIG

async def run_complete_optimization():
    # Initialize orchestrator
    orchestrator = EnhancedOptimizationOrchestrator(
        config=CONFIG["hyperparameter_optimization"],
        computational_config=CONFIG["computational_optimization"]
    )
    
    # Run comprehensive optimization
    results = await orchestrator.run_comprehensive_optimization(
        market_data=your_market_data,
        optimization_types=["multi_objective", "bayesian", "adaptive"],
        max_trials=500
    )
    
    return results
```

### 2. Regime-Specific Optimization

```python
from src.training.adaptive_optimizer import AdaptiveOptimizer

async def optimize_for_regime(regime: str):
    # Get regime-specific configuration
    regime_config = CONFIG["hyperparameter_optimization"]["adaptive_optimization"]["regime_specific_constraints"][regime]
    
    # Initialize adaptive optimizer
    optimizer = AdaptiveOptimizer(
        config=CONFIG["hyperparameter_optimization"]["adaptive_optimization"]
    )
    
    # Run regime-specific optimization
    results = await optimizer.optimize_regime_specific(
        regime=regime,
        constraints=regime_config,
        n_trials=100
    )
    
    return results
```

### 3. Computational Optimization Integration

```python
from src.training.optimized_backtester import OptimizedBacktester
from src.training.multi_objective_optimizer import MultiObjectiveOptimizer

async def optimized_multi_objective_optimization():
    # Initialize optimized backtester
    backtester = OptimizedBacktester(
        market_data=your_market_data,
        config=CONFIG["computational_optimization"]
    )
    
    # Initialize multi-objective optimizer with optimized backtester
    optimizer = MultiObjectiveOptimizer(
        config=CONFIG["hyperparameter_optimization"]["multi_objective"],
        optimized_backtester=backtester
    )
    
    # Run optimization
    results = await optimizer.optimize(n_trials=200)
    
    return results
```

### 4. Scheduled Optimization

```python
import asyncio
from datetime import datetime, time

async def run_scheduled_optimization():
    schedules = CONFIG["hyperparameter_optimization"]["optimization_schedules"]
    
    while True:
        now = datetime.now()
        current_time = now.time()
        
        # Check daily schedule
        daily_schedule = schedules["daily"]
        if daily_schedule["enabled"]:
            schedule_time = time.fromisoformat(daily_schedule["time"])
            if current_time.hour == schedule_time.hour and current_time.minute == schedule_time.minute:
                await run_quick_optimization(daily_schedule["max_trials"])
        
        # Check weekly schedule
        weekly_schedule = schedules["weekly"]
        if weekly_schedule["enabled"] and now.weekday() == 6:  # Sunday
            schedule_time = time.fromisoformat(weekly_schedule["time"])
            if current_time.hour == schedule_time.hour and current_time.minute == schedule_time.minute:
                await run_comprehensive_optimization(weekly_schedule["max_trials"])
        
        await asyncio.sleep(60)  # Check every minute
```

---

## Best Practices

### 1. Configuration Management

```python
# Always validate configuration before use
def validate_config(config: dict) -> bool:
    required_sections = ["hyperparameter_optimization", "computational_optimization"]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    return True

# Use configuration with validation
if validate_config(CONFIG):
    # Proceed with optimization
    pass
```

### 2. Memory Management

```python
# Monitor memory usage during optimization
def monitor_memory():
    memory_config = CONFIG["computational_optimization"]["memory_management"]
    
    if memory_config["enabled"]:
        memory_percent = psutil.virtual_memory().percent / 100
        
        if memory_percent > memory_config["memory_threshold"]:
            # Trigger cleanup
            gc.collect()
            return True
    
    return False
```

### 3. Error Handling

```python
import logging

async def safe_optimization():
    try:
        # Run optimization with error handling
        results = await run_optimization()
        return results
    except Exception as e:
        logging.error(f"Optimization failed: {e}")
        # Fallback to default parameters
        return get_default_parameters()
```

### 4. Performance Monitoring

```python
import time
from contextlib import contextmanager

@contextmanager
def performance_monitor(operation_name: str):
    start_time = time.time()
    start_memory = psutil.virtual_memory().percent
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = psutil.virtual_memory().percent
        
        duration = end_time - start_time
        memory_change = end_memory - start_memory
        
        logging.info(f"{operation_name}: Duration={duration:.2f}s, Memory_change={memory_change:.1f}%")

# Usage
with performance_monitor("Hyperparameter Optimization"):
    results = await run_optimization()
```

---

## Troubleshooting

### 1. Common Issues

#### Memory Issues
```python
# Check if memory management is enabled
if CONFIG["computational_optimization"]["memory_management"]["enabled"]:
    # Reduce batch size or increase cleanup frequency
    CONFIG["computational_optimization"]["memory_management"]["cleanup_frequency"] = 50
```

#### Performance Issues
```python
# Check parallelization settings
if CONFIG["computational_optimization"]["parallelization"]["enabled"]:
    # Adjust number of workers based on CPU cores
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    CONFIG["computational_optimization"]["parallelization"]["max_workers"] = min(cpu_count, 8)
```

#### Convergence Issues
```python
# Adjust Bayesian optimization parameters
bayesian_config = CONFIG["hyperparameter_optimization"]["bayesian_optimization"]
bayesian_config["patience"] = 100  # Increase patience
bayesian_config["min_trials"] = 50  # Increase minimum trials
```

### 2. Debugging Configuration

```python
def debug_configuration():
    """Debug configuration settings"""
    
    # Check hyperparameter optimization
    hpo_config = CONFIG["hyperparameter_optimization"]
    print(f"Multi-objective enabled: {hpo_config['multi_objective']['enabled']}")
    print(f"Bayesian optimization enabled: {hpo_config['bayesian_optimization']['enabled']}")
    print(f"Adaptive optimization enabled: {hpo_config['adaptive_optimization']['enabled']}")
    
    # Check computational optimization
    comp_config = CONFIG["computational_optimization"]
    print(f"Caching enabled: {comp_config['caching']['enabled']}")
    print(f"Parallelization enabled: {comp_config['parallelization']['enabled']}")
    print(f"Memory management enabled: {comp_config['memory_management']['enabled']}")
```

### 3. Configuration Validation

```python
def validate_optimization_config():
    """Validate optimization configuration"""
    
    config = CONFIG["hyperparameter_optimization"]
    
    # Validate search spaces
    search_spaces = config["search_spaces"]
    for space_name, space_config in search_spaces.items():
        for param_name, param_config in space_config.items():
            if "low" in param_config and "high" in param_config:
                if param_config["low"] >= param_config["high"]:
                    raise ValueError(f"Invalid range for {param_name}: low >= high")
    
    # Validate regime constraints
    regime_constraints = config["adaptive_optimization"]["regime_specific_constraints"]
    for regime, constraints in regime_constraints.items():
        for constraint_name, constraint_range in constraints.items():
            if len(constraint_range) != 2 or constraint_range[0] >= constraint_range[1]:
                raise ValueError(f"Invalid constraint range for {regime}.{constraint_name}")
    
    return True
```

This comprehensive guide provides all the information needed to effectively use the new configuration sections for enhanced hyperparameter optimization and computational optimization in your Ares trading system. 