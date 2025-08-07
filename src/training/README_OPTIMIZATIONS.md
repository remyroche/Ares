# Enhanced Training Manager with Computational Optimizations

This document explains the comprehensive computational optimization strategies implemented in the enhanced training manager, based on the `computational_optimization_strategies.md` document.

## Overview

The optimized enhanced training manager implements multiple optimization strategies to significantly reduce computational demands while maintaining or improving training quality:

- **60-70% reduction** in overall optimization time
- **35-45% reduction** in memory usage
- **Maintained accuracy** with improved robustness and reliability

## Optimization Components

### 1. Enhanced Training Manager Optimized (`enhanced_training_manager_optimized.py`)

The main training manager with all optimization strategies integrated:

```python
from src.training.enhanced_training_manager_optimized import EnhancedTrainingManagerOptimized

# Initialize with optimization config
manager = EnhancedTrainingManagerOptimized(config)
await manager.initialize()

# Execute optimized training
results = await manager.execute_optimized_training(
    symbol="ETHUSDT", 
    exchange="BINANCE", 
    timeframe="1h"
)
```

**Key Features:**
- Cached backtesting to avoid redundant calculations
- Progressive evaluation to stop unpromising trials early
- Parallel backtesting for multiple parameter combinations
- Incremental training to reuse model states
- Streaming for large datasets
- Adaptive sampling to focus on promising regions
- Memory-efficient data structures
- Memory profiling and leak detection

### 2. Optimized Step Executor (`steps/optimized_step_executor.py`)

Optimized execution of all 16 training steps:

```python
from src.training.steps.optimized_step_executor import OptimizedStepExecutor

executor = OptimizedStepExecutor(config)
results = await executor.execute_optimized_pipeline(symbol, exchange, timeframe)
```

**Optimizations Applied:**
- **Step Caching**: Results cached to avoid re-computation
- **Parallel Processing**: Steps executed in parallel where possible
- **Memory Management**: Automatic cleanup between steps
- **Streaming Data**: Large datasets processed in chunks
- **Early Stopping**: Unpromising trials stopped early

### 3. Memory Profiler (`memory_profiler.py`)

Comprehensive memory monitoring and leak detection:

```python
from src.training.memory_profiler import MemoryProfiler, MemoryLeakDetector

# Create profiler with continuous monitoring
profiler = MemoryProfiler(enable_continuous_monitoring=True)

# Take snapshots
snapshot = profiler.take_snapshot("training_start")

# Detect memory leaks
leak_detector = MemoryLeakDetector(profiler)
leak_results = leak_detector.check_for_leaks()
```

**Features:**
- Real-time memory monitoring
- Automatic leak detection
- Memory usage trends analysis
- Optimization recommendations
- Garbage collection management

### 4. Configuration System (`config/computational_optimization_config.py`)

Centralized configuration for all optimizations:

```python
from src.config.computational_optimization_config import get_optimization_config

# Get default config
config = get_optimization_config()

# Override specific settings
custom_config = {
    "parallelization": {"max_workers": 12},
    "memory_management": {"memory_threshold": 0.75}
}
config = get_optimization_config(custom_config)
```

### 5. Factory System (`factory.py`)

Easy creation of optimized components:

```python
from src.training.factory import create_optimized_training_system

# Create complete system
training_system = create_optimized_training_system(config)

# Access components
training_manager = training_system["training_manager"]
memory_profiler = training_system["memory_profiler"]
step_executor = training_system["step_executor"]
```

## Performance Optimizations

### 1. Cached Backtesting

**Problem**: Full backtest for every trial is expensive
**Solution**: Cache results and precompute technical indicators

```python
# Technical indicators computed once
indicators = {
    'sma_20': data['close'].rolling(20).mean().values,
    'rsi': calculate_rsi(data['close']),
    'atr': calculate_atr(data)
}

# Backtest results cached by parameter hash
cache_key = hash(frozenset(params.items()))
if cache_key in cache:
    return cache[cache_key]
```

**Expected Improvement**: 70-80% reduction in backtesting time

### 2. Progressive Evaluation

**Problem**: Poor parameter sets evaluated on full dataset
**Solution**: Early stopping with progressive data evaluation

```python
evaluation_stages = [
    (0.1, 0.3),   # 10% data, 30% weight
    (0.3, 0.5),   # 30% data, 50% weight  
    (1.0, 1.0)    # 100% data, 100% weight
]

# Stop early if performance is poor
if data_ratio < 1.0 and score < -0.5:
    return -1.0  # Stop evaluation
```

**Expected Improvement**: 40-60% reduction in evaluation time

### 3. Parallel Processing

**Problem**: Sequential processing of independent tasks
**Solution**: Parallel execution across multiple cores

```python
# Parallel backtesting
with ProcessPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(evaluate_params, params) for params in param_batch]
    results = [future.result() for future in futures]

# Parallel feature engineering by regime
tasks = [engineer_features_for_regime(regime) for regime in regimes]
results = await asyncio.gather(*tasks)
```

**Expected Improvement**: 3-8x speedup depending on CPU cores

### 4. Incremental Training

**Problem**: Full model training for each trial
**Solution**: Reuse model states and continue training

```python
# Generate cache key based on core parameters
model_key = hash(frozenset(core_params.items()))

if model_key in model_cache:
    # Continue training from cached state
    model = model_cache[model_key]
    model.fit(X, y, xgb_model=model.get_booster())
```

**Expected Improvement**: 50-60% reduction in model training time

### 5. Streaming Data Processing

**Problem**: Large datasets don't fit in memory
**Solution**: Process data in chunks with efficient formats

```python
# Parquet streaming
for batch in parquet_file.iter_batches(batch_size=10000):
    chunk = batch.to_pandas()
    process_chunk(chunk)

# Memory-optimized DataFrames
df[col] = pd.to_numeric(df[col], downcast='float')
```

**Expected Improvement**: 40-50% reduction in memory usage

### 6. Adaptive Sampling

**Problem**: Random sampling wastes evaluations
**Solution**: Focus on promising parameter regions

```python
# Identify top 25% of trials
top_quartile = sorted(trial_history, key=lambda x: x['score'], reverse=True)[:len(sorted_trials)//4]

# Sample around promising regions with perturbation
reference_trial = random.choice(top_quartile)
new_params = perturb_parameters(reference_trial['params'])
```

**Expected Improvement**: 30-50% better parameter exploration efficiency

## Memory Management

### Automatic Memory Optimization

```python
# Monitor memory usage
memory_percent = psutil.virtual_memory().percent / 100
if memory_percent > 0.8:
    perform_cleanup()

# Optimize DataFrame memory
for col in df.select_dtypes(include=['float64']).columns:
    df[col] = pd.to_numeric(df[col], downcast='float')

# Use Parquet for efficient storage
df.to_parquet(file_path, compression='snappy')
```

### Memory Leak Detection

```python
# Analyze memory growth trends
rss_values = [snapshot['process_memory']['rss_mb'] for snapshot in recent_snapshots]
rss_growth = rss_values[-1] - rss_values[0]

# Detect growing object types
for obj_type in last_snapshot["object_counts"]:
    growth = last_count - first_count
    if growth > 1000:  # Flag significant growth
        potential_leak = True
```

## Usage Examples

### Basic Usage

```python
import asyncio
from src.training.factory import create_optimized_training_system

async def train_model():
    # Configuration
    config = {
        "training": {"n_trials": 200, "lookback_days": 30},
        "computational_optimization": {
            "parallelization": {"enabled": True, "max_workers": 8},
            "memory_management": {"enabled": True, "memory_threshold": 0.8},
            "caching": {"enabled": True, "max_cache_size": 1000}
        }
    }
    
    # Create optimized system
    system = create_optimized_training_system(config)
    
    # Execute training
    results = await system["training_manager"].execute_optimized_training(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1h"
    )
    
    return results

# Run training
results = asyncio.run(train_model())
```

### Advanced Usage with Memory Monitoring

```python
from src.training.enhanced_training_manager_optimized import EnhancedTrainingManagerOptimized
from src.training.memory_profiler import MemoryProfiler, MemoryLeakDetector

async def advanced_training():
    # Setup memory monitoring
    profiler = MemoryProfiler(enable_continuous_monitoring=True)
    leak_detector = MemoryLeakDetector(profiler)
    
    # Take initial snapshot
    profiler.take_snapshot("training_start")
    
    # Initialize training manager
    manager = EnhancedTrainingManagerOptimized(config)
    await manager.initialize()
    
    # Execute training with monitoring
    results = await manager.execute_optimized_training("ETHUSDT", "BINANCE")
    
    # Check for leaks
    leak_results = leak_detector.check_for_leaks()
    if leak_results["leak_detected"]:
        print("Memory leak detected!")
        for rec in leak_results["recommendations"]:
            print(f"  - {rec}")
    
    # Generate memory report
    report = profiler.generate_memory_report()
    
    # Cleanup
    await manager.cleanup()
    profiler.stop_continuous_monitoring()
    
    return results, report
```

## Configuration Options

### Computational Optimization Configuration

```python
COMPUTATIONAL_OPTIMIZATION_CONFIG = {
    "caching": {
        "enabled": True,
        "max_cache_size": 1000,
        "cache_ttl": 3600
    },
    "parallelization": {
        "enabled": True,
        "max_workers": 8,
        "chunk_size": 1000
    },
    "early_stopping": {
        "enabled": True,
        "patience": 10,
        "min_trials": 20,
        "performance_threshold": -0.5
    },
    "memory_management": {
        "enabled": True,
        "memory_threshold": 0.8,
        "cleanup_frequency": 100
    },
    "data_streaming": {
        "enabled": True,
        "chunk_size": 10000,
        "compression": "snappy"
    },
    "adaptive_sampling": {
        "enabled": True,
        "initial_samples": 100,
        "perturbation_factor": 0.1
    },
    "monitoring": {
        "continuous_monitoring": True,
        "monitoring_interval": 30,
        "memory_leak_detection": True
    }
}
```

## Expected Performance Improvements

Based on the computational optimization strategies document:

### Computational Time Reduction
- **Backtesting**: 70-80% reduction through caching and parallelization
- **Model Training**: 50-60% reduction through incremental training
- **Feature Engineering**: 90% reduction through precomputation
- **Overall**: 60-70% reduction in total optimization time

### Memory Usage Reduction
- **Data Storage**: 40-50% reduction through optimized data structures
- **Model Storage**: 30-40% reduction through model reuse
- **Overall**: 35-45% reduction in memory usage

### Quality Maintenance
- **Accuracy**: Maintained through surrogate model validation
- **Robustness**: Improved through adaptive sampling
- **Reliability**: Enhanced through progressive evaluation

## Integration with Existing System

The optimized training manager is designed to be a drop-in replacement for the existing enhanced training manager:

```python
# Old way
from src.training.enhanced_training_manager import EnhancedTrainingManager
manager = EnhancedTrainingManager(config)

# New optimized way
from src.training.enhanced_training_manager_optimized import EnhancedTrainingManagerOptimized
manager = EnhancedTrainingManagerOptimized(config)

# Same interface, better performance
results = await manager.execute_enhanced_training(symbol, exchange, timeframe)
```

## Monitoring and Debugging

### Memory Usage Monitoring

```python
# Get current memory profile
profile = manager.get_memory_profile()
print(f"Memory usage: {profile['percentage']:.1f}%")

# Get optimization statistics
stats = manager.get_optimization_stats()
print(f"Cache hit ratio: {stats.get('cache_hit_ratio', 0):.2%}")
```

### Performance Tracking

```python
# Execution statistics
if hasattr(step_executor, 'get_execution_stats'):
    stats = step_executor.get_execution_stats()
    print(f"Parallel execution: {stats['parallel_execution_enabled']}")
    print(f"Memory optimization: {stats['memory_optimization_enabled']}")
```

## Troubleshooting

### High Memory Usage
1. Enable memory management: `memory_management.enabled = True`
2. Lower memory threshold: `memory_threshold = 0.7`
3. Increase cleanup frequency: `cleanup_frequency = 50`
4. Enable data streaming: `data_streaming.enabled = True`

### Slow Performance
1. Enable parallelization: `parallelization.enabled = True`
2. Increase worker count: `max_workers = min(os.cpu_count(), 16)`
3. Enable caching: `caching.enabled = True`
4. Enable early stopping: `early_stopping.enabled = True`

### Memory Leaks
1. Enable leak detection: `memory_leak_detection = True`
2. Monitor continuously: `continuous_monitoring = True`
3. Check recommendations in memory reports
4. Force garbage collection: `profiler.force_garbage_collection()`

## Future Enhancements

Potential areas for further optimization:
1. GPU acceleration for model training
2. Distributed training across multiple machines
3. Advanced surrogate models (Gaussian Processes)
4. Dynamic resource allocation
5. Real-time performance monitoring dashboard