# Pipeline Performance Optimization Suggestions

Speed up meta_labeling_hpo_sample_weighted execution without disabling parameters or steps.

## Current Performance Analysis

### Bottlenecks Identified:
1. **Layer 2 Causal Discovery**: Multiple PC Algorithm + LiNGAM iterations
2. **Layer 2 Geometry Gates**: 100+ composite geometries with full statistical validation
3. **Memory Usage**: 582MB with potential for optimization
4. **CPU Utilization**: 57.9% (could be better utilized)

## Optimization Suggestions

### 1. Parallel Processing Enhancements

#### Causal Discovery Parallelization
```python
# Current: Sequential causal discovery iterations
# Suggestion: Parallel geometry evaluation

# In meta_labeling_hpo_experiment_step.py
async def run_layer2_parallel_geometries(config, market_data):
    """Run multiple causal discovery iterations in parallel"""
    
    # Create parameter grid for parallel execution
    param_combinations = [
        {'significance': 0.3, 'max_set': 2},
        {'significance': 0.4, 'max_set': 3},
        {'significance': 0.5, 'max_set': 3}
    ]
    
    # Execute in parallel using asyncio.gather
    tasks = [
        run_causal_discovery_async(params, market_data)
        for params in param_combinations
    ]
    results = await asyncio.gather(*tasks)
    return results
```

#### Geometry Gate Parallel Processing
```python
# Current: Sequential gate evaluation
# Suggestion: Batch gate processing

async def evaluate_geometry_gates_parallel(geometries, market_data):
    """Evaluate multiple geometry gates simultaneously"""
    
    batch_size = min(8, len(geometries))  # Process 8 at once
    batches = [geometries[i:i+batch_size] for i in range(0, len(geometries), batch_size)]
    
    all_results = []
    for batch in batches:
        tasks = [evaluate_single_gate(geom, market_data) for geom in batch]
        batch_results = await asyncio.gather(*tasks)
        all_results.extend(batch_results)
    
    return all_results
```

### 2. Memory Optimization

#### Chunked Data Processing
```python
# Current: Full 8000 samples processed at once
# Suggestion: Chunked processing for memory efficiency

def process_causal_discovery_chunked(market_data, chunk_size=4000):
    """Process causal discovery in chunks to reduce memory pressure"""
    
    chunks = []
    for i in range(0, len(market_data), chunk_size):
        chunk = market_data.iloc[i:i+chunk_size]
        chunks.append(chunk)
    
    # Process chunks and aggregate results
    chunk_results = []
    for chunk in chunks:
        result = run_causal_discovery(chunk)
        chunk_results.append(result)
    
    return aggregate_causal_results(chunk_results)
```

#### Efficient Data Structures
```python
# Use numpy arrays instead of pandas where possible
# Pre-allocate arrays to avoid dynamic resizing

def optimize_memory_usage(market_data):
    """Convert to memory-efficient formats"""
    
    # Convert pandas to numpy for computation
    price_array = market_data['close'].to_numpy(dtype=np.float32)
    volume_array = market_data['volume'].to_numpy(dtype=np.float32)
    
    # Use sparse matrices for causal graphs
    from scipy.sparse import csr_matrix
    causal_matrix = csr_matrix(causal_graph)
    
    return price_array, volume_array, causal_matrix
```

### 3. Algorithmic Optimizations

#### Early Termination for Poor Geometries
```python
# Current: Full evaluation of all geometries
# Suggestion: Early stopping for obviously poor candidates

def evaluate_gate_with_early_stopping(geometry, market_data):
    """Evaluate geometry with early termination"""
    
    # Quick pre-checks
    if not quick_quality_check(geometry):
        return {'status': 'failed_early', 'reason': 'pre_check_failed'}
    
    # Sample size check
    sample_count = estimate_sample_count(geometry)
    if sample_count < 50:  # Minimum threshold
        return {'status': 'failed_early', 'reason': 'insufficient_samples'}
    
    # Full evaluation only for promising candidates
    return full_gate_evaluation(geometry, market_data)
```

#### Caching Intermediate Results
```python
# Cache frequently computed values
from functools import lru_cache
import pickle
import os

@lru_cache(maxsize=1000)
def cached_statistical_test(data_hash, test_type):
    """Cache statistical test results"""
    cache_file = f"cache/stat_test_{data_hash}_{test_type}.pkl"
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    result = compute_statistical_test(data_hash, test_type)
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    
    return result
```

### 4. Hardware Optimization

#### CPU Affinity and Threading
```python
# Optimize CPU usage for the current hardware
import os
import psutil

def optimize_cpu_usage():
    """Optimize CPU utilization for current hardware"""
    
    # Get CPU count and set optimal thread pool
    cpu_count = psutil.cpu_count()
    optimal_threads = min(cpu_count - 1, 8)  # Leave 1 core free
    
    # Set environment variables for optimal threading
    os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
    os.environ['NUMBA_NUM_THREADS'] = str(optimal_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(optimal_threads)
    
    return optimal_threads
```

#### GPU Acceleration for Compatible Operations
```python
# Use GPU for matrix operations where possible
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def accelerate_matrix_operations(matrix):
    """Accelerate matrix operations using GPU when available"""
    
    if GPU_AVAILABLE:
        # Transfer to GPU
        gpu_matrix = cp.asarray(matrix)
        
        # Perform operations on GPU
        result = cp.linalg.svd(gpu_matrix)
        
        # Transfer back to CPU
        return cp.asnumpy(result)
    else:
        # Fallback to CPU
        return np.linalg.svd(matrix)
```

### 5. Configuration Optimizations

#### Adaptive Parameter Ranges
```python
# Current: Fixed parameter ranges
# Suggestion: Adaptive ranges based on data characteristics

def get_adaptive_parameters(market_data):
    """Determine optimal parameter ranges based on data"""
    
    data_size = len(market_data)
    volatility = market_data['close'].std()
    
    # Adjust ranges based on data characteristics
    if data_size > 10000:
        max_set_range = [2, 3, 4]
    else:
        max_set_range = [2, 3]
    
    if volatility > 0.02:  # High volatility
        significance_range = [0.3, 0.4, 0.5]
    else:
        significance_range = [0.4, 0.5]
    
    return {
        'significance_levels': significance_range,
        'max_set_sizes': max_set_range,
        'adaptive': True
    }
```

#### Smart Geometry Selection
```python
# Prioritize promising geometry types
def prioritize_geometries(all_geometries, historical_performance):
    """Prioritize geometries based on historical performance"""
    
    scored_geometries = []
    for geom in all_geometries:
        # Score based on type and historical success
        base_score = get_geometry_type_score(geom['type'])
        historical_score = historical_performance.get(geom['type'], 0.5)
        
        combined_score = 0.7 * base_score + 0.3 * historical_score
        scored_geometries.append((geom, combined_score))
    
    # Sort by score and prioritize top candidates
    scored_geometries.sort(key=lambda x: x[1], reverse=True)
    
    return [geom for geom, score in scored_geometries]
```

## Implementation Priority

### High Impact, Low Risk:
1. **CPU Optimization**: Better thread utilization
2. **Memory Optimization**: Chunked processing
3. **Caching**: Statistical test results

### Medium Impact, Medium Risk:
1. **Parallel Processing**: Geometry gate evaluation
2. **Early Termination**: Poor candidate filtering
3. **Adaptive Parameters**: Data-driven ranges

### High Impact, Higher Risk:
1. **GPU Acceleration**: Matrix operations
2. **Advanced Parallelization**: Causal discovery
3. **Smart Selection**: Historical performance weighting

## Expected Performance Gains

### Conservative Estimates:
- **CPU Optimization**: 20-30% speedup
- **Memory Optimization**: 15-25% speedup
- **Caching**: 10-20% speedup

### Aggressive Estimates:
- **Parallel Processing**: 40-60% speedup
- **Early Termination**: 25-35% speedup
- **GPU Acceleration**: 50-80% speedup (for compatible operations)

## Implementation Strategy

### Phase 1: Quick Wins (Immediate)
1. Optimize CPU thread settings
2. Add basic caching
3. Implement memory chunking

### Phase 2: Medium-term (Next run)
1. Parallel geometry evaluation
2. Early termination logic
3. Adaptive parameter ranges

### Phase 3: Advanced (Future)
1. GPU acceleration
2. Advanced parallelization
3. Smart geometry selection

## Validation Requirements

- **No functional changes**: All results must be identical
- **Statistical validity**: Causal relationships preserved
- **Quality maintained**: Gate standards not compromised
- **Reproducibility**: Same results on same data
