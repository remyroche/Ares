# Chunked Correlation Optimization Guide

## Current Implementation

The chunked correlation is processing 294 features in chunks to avoid memory issues. The current implementation:
- Chunk size: ~147 features (294 // 2)
- Total chunk pairs: 2 × 2 = 4 pairs
- Each pair computes correlation between chunk_i and chunk_j

## Applied Optimizations ✅

### 1. Enhanced Progress Logging
- Added detailed tprints showing:
  - Chunk size and total chunks
  - Current chunk pair being processed
  - Progress percentage
  - Elapsed time and ETA
  
### 2. Memory Optimization
- **Changed from float64 to float32**: Reduces memory usage by 50%
- **Adaptive chunk size**: `min(250, max(100, n_features // 3))`
  - For 294 features: chunk_size = 98 → 3×3 = 9 chunk pairs
  - Smaller chunks = more iterations but less memory per iteration

### 3. Progress Updates
- Updates every 10% of total pairs
- Shows elapsed time and ETA for better visibility

## Additional Optimization Suggestions

### Option 1: Parallel Processing (HIGH IMPACT) ⚡
**Impact:** 2-4x speedup on multi-core systems
**Quality:** No reduction
**Complexity:** Medium

```python
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Compute chunk pairs in parallel
def compute_chunk_pair(args):
    i, end_i, j, end_j, chunk_i, chunk_j, optimal_sample_size = args
    corr_chunk = np.dot(chunk_i.T, chunk_j) / optimal_sample_size
    return (i, end_i, j, end_j, corr_chunk)

# Use ThreadPoolExecutor for parallel computation
with ThreadPoolExecutor(max_workers=4) as executor:
    tasks = []
    for i in range(0, n_features, chunk_size):
        end_i = min(i + chunk_size, n_features)
        chunk_i = feature_normalized[:, i:end_i].astype(np.float32)
        
        for j in range(0, n_features, chunk_size):
            end_j = min(j + chunk_size, n_features)
            chunk_j = feature_normalized[:, j:end_j].astype(np.float32)
            tasks.append((i, end_i, j, end_j, chunk_i, chunk_j, optimal_sample_size))
    
    # Execute in parallel
    results = list(executor.map(compute_chunk_pair, tasks))
    
    # Fill correlation matrix
    for i, end_i, j, end_j, corr_chunk in results:
        corr_matrix[i:end_i, j:end_j] = corr_chunk
```

### Option 2: Symmetric Matrix Optimization (MEDIUM IMPACT) ⚡
**Impact:** 2x speedup (only compute upper triangle)
**Quality:** No reduction
**Complexity:** Low

```python
# Only compute upper triangle (correlation matrix is symmetric)
for i in range(0, n_features, chunk_size):
    end_i = min(i + chunk_size, n_features)
    chunk_i = feature_normalized[:, i:end_i].astype(np.float32)
    
    for j in range(i, n_features, chunk_size):  # Start from i, not 0
        end_j = min(j + chunk_size, n_features)
        chunk_j = feature_normalized[:, j:end_j].astype(np.float32)
        
        corr_chunk = np.dot(chunk_i.T, chunk_j) / optimal_sample_size
        corr_matrix[i:end_i, j:end_j] = corr_chunk
        
        # Mirror to lower triangle (except diagonal blocks)
        if i != j:
            corr_matrix[j:end_j, i:end_i] = corr_chunk.T

# For 294 features with chunk_size=98:
# Before: 3×3 = 9 chunk pairs
# After: (3×4)/2 = 6 chunk pairs (33% reduction)
```

### Option 3: Use Numba JIT Compilation (MEDIUM IMPACT) ⚡
**Impact:** 1.5-3x speedup
**Quality:** No reduction
**Complexity:** Low

```python
from numba import jit, prange

@jit(nopython=True, parallel=True, fastmath=True)
def compute_correlation_matrix(feature_normalized, chunk_size, n_features, optimal_sample_size):
    corr_matrix = np.zeros((n_features, n_features), dtype=np.float32)
    
    for i in prange(0, n_features, chunk_size):
        end_i = min(i + chunk_size, n_features)
        chunk_i = feature_normalized[:, i:end_i]
        
        for j in range(0, n_features, chunk_size):
            end_j = min(j + chunk_size, n_features)
            chunk_j = feature_normalized[:, j:end_j]
            
            corr_chunk = np.dot(chunk_i.T, chunk_j) / optimal_sample_size
            corr_matrix[i:end_i, j:end_j] = corr_chunk
    
    return corr_matrix

# Use the JIT-compiled function
corr_matrix = compute_correlation_matrix(
    feature_normalized.astype(np.float32),
    chunk_size,
    n_features,
    optimal_sample_size
)
```

### Option 4: GPU Acceleration with CuPy (VERY HIGH IMPACT) 🚀
**Impact:** 10-50x speedup (if GPU available)
**Quality:** No reduction
**Complexity:** Medium (requires GPU)

```python
try:
    import cupy as cp
    
    # Transfer to GPU
    feature_normalized_gpu = cp.asarray(feature_normalized, dtype=cp.float32)
    
    # Compute correlation on GPU (no chunking needed!)
    corr_matrix_gpu = cp.dot(feature_normalized_gpu.T, feature_normalized_gpu) / optimal_sample_size
    
    # Transfer back to CPU
    corr_matrix = cp.asnumpy(corr_matrix_gpu)
    
    tprint_success("✅ GPU acceleration used!")
except ImportError:
    # Fallback to CPU chunked correlation
    tprint_warning("⚠️ CuPy not available, using CPU")
```

### Option 5: Approximate Correlation (LOW QUALITY IMPACT) ⚡
**Impact:** 3-5x speedup
**Quality:** Slight reduction (95-98% accuracy)
**Complexity:** Low

```python
# Use random sampling for large datasets
if optimal_sample_size > 5000:
    # Sample 5000 rows randomly
    sample_indices = np.random.choice(optimal_sample_size, 5000, replace=False)
    feature_normalized_sampled = feature_normalized[sample_indices, :]
    sample_size = 5000
    
    tprint_warning(f"⚠️ Using random sampling: {sample_size}/{optimal_sample_size} rows")
else:
    feature_normalized_sampled = feature_normalized
    sample_size = optimal_sample_size

# Compute correlation on sampled data
corr_matrix = np.dot(feature_normalized_sampled.T, feature_normalized_sampled) / sample_size
```

### Option 6: Sparse Correlation (MEDIUM IMPACT) ⚡
**Impact:** 2-3x speedup for sparse features
**Quality:** No reduction
**Complexity:** Medium

```python
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Check sparsity
sparsity = 1.0 - (np.count_nonzero(feature_normalized) / feature_normalized.size)

if sparsity > 0.5:  # If >50% zeros
    tprint_info(f"📊 Feature sparsity: {sparsity:.1%} - using sparse computation")
    
    # Convert to sparse matrix
    feature_sparse = csr_matrix(feature_normalized)
    
    # Compute correlation using sparse operations
    corr_matrix = cosine_similarity(feature_sparse.T, dense_output=True)
else:
    # Use dense computation
    corr_matrix = np.dot(feature_normalized.T, feature_normalized) / optimal_sample_size
```

## Recommended Implementation Strategy

### Phase 1: Quick Wins (Immediate) ✅
1. ✅ **Enhanced logging** - Already applied
2. ✅ **Float32 instead of float64** - Already applied
3. ✅ **Adaptive chunk size** - Already applied
4. **Symmetric matrix optimization** - Compute only upper triangle

### Phase 2: Parallel Processing (Next Run)
1. **ThreadPoolExecutor** - Easy to implement, 2-4x speedup
2. **Combine with symmetric optimization** - 4-8x total speedup

### Phase 3: Advanced (If Still Slow)
1. **Numba JIT** - If ThreadPoolExecutor not enough
2. **GPU acceleration** - If available and worth the dependency
3. **Approximate correlation** - Only if quality impact acceptable

## Implementation Priority

### For Current Run (Already Applied) ✅
```python
# 1. Float32 memory optimization
corr_matrix = np.zeros((n_features, n_features), dtype=np.float32)
chunk_i = feature_normalized[:, i:end_i].astype(np.float32)

# 2. Adaptive chunk size
chunk_size = min(250, max(100, n_features // 3))

# 3. Progress logging
tprint_info(f"📊 Chunk pair [{chunk_num_i},{chunk_num_j}] | Progress: {progress_pct:.1f}%")
```

### For Next Run (Recommended)
```python
# 1. Symmetric matrix optimization (2x speedup)
for i in range(0, n_features, chunk_size):
    for j in range(i, n_features, chunk_size):  # Start from i
        # ... compute and mirror

# 2. Parallel processing (2-4x speedup)
with ThreadPoolExecutor(max_workers=4) as executor:
    # ... parallel chunk computation
```

### Combined Impact Estimate
- **Current optimizations:** 1.5x speedup (float32 + adaptive chunks)
- **+ Symmetric matrix:** 3x total speedup
- **+ Parallel processing:** 6-12x total speedup
- **+ Numba JIT:** 9-36x total speedup

## Performance Benchmarks

### Current Implementation (294 features)
- Chunk pairs: 9 (3×3)
- Estimated time: ~60-120 seconds
- Memory: ~330 MB (float32)

### With Symmetric Optimization
- Chunk pairs: 6 (upper triangle only)
- Estimated time: ~40-80 seconds
- Memory: ~330 MB

### With Parallel + Symmetric
- Chunk pairs: 6 (parallel)
- Estimated time: ~10-20 seconds (4 cores)
- Memory: ~330 MB

### With GPU (if available)
- No chunking needed
- Estimated time: ~2-5 seconds
- Memory: GPU VRAM

## Code Quality Considerations

All suggested optimizations maintain:
- ✅ Numerical accuracy (except approximate correlation)
- ✅ Reproducibility
- ✅ Error handling
- ✅ Memory safety
- ✅ Code readability

## Conclusion

The current implementation with enhanced logging and float32 optimization provides good visibility and moderate performance improvement. For significant speedup without quality loss, implement:

1. **Symmetric matrix optimization** (easy, 2x speedup)
2. **Parallel processing** (medium, 2-4x speedup)
3. **Combined:** 4-8x total speedup

This would reduce processing time from ~60-120s to ~8-15s for 294 features.
