# Clustering Constraints Update - Structural Constraints Added

## Summary

Added structural constraints to the unified clustering optimization goals to enforce cluster count and cluster size requirements across all clustering components.

## New Constraints Added

### 1. Cluster Count Constraint
- **Preferred Range**: 6-8 clusters
- **Absolute Range**: 5-10 clusters
- **Purpose**: Ensures optimal number of market regimes for trading strategies
- **Applied in**: All clustering components

### 2. Cluster Size Constraint
- **Minimum Size**: 2% of total samples
- **Maximum Size**: 20% of total samples
- **Purpose**: 
  - Prevents tiny clusters that may not be statistically significant
  - Prevents dominant clusters that may not be specific enough
- **Applied in**: All clustering components

## Updated Components

### 1. `clustering_optimization_goals.py` ✅
**Changes**:
- Added `cluster_count_range`, `cluster_count_min`, `cluster_count_max` to `ClusteringOptimizationGoals`
- Added `min_cluster_size_pct`, `max_cluster_size_pct` to `ClusteringOptimizationGoals`
- Added same constraints to `OptimizationTargets`
- Added `validate_cluster_sizes()` function for size validation
- Updated `meets_optimization_constraints()` to check both count and size constraints

**New Functions**:
```python
def validate_cluster_sizes(
    cluster_sizes: List[int],
    n_total_samples: int,
    targets: Optional[OptimizationTargets] = None
) -> Tuple[bool, Dict[str, any]]:
    """Validate cluster sizes meet constraints (2%-20%)."""
```

### 2. `iterative_optimization_tuner.py` ✅
**Changes**:
- Updated `OptimizationParameterSpace` to reference unified constraints
- Updated parameter space comments to show alignment with unified goals:
  - `K_MIN: (5, 8)` - aligned with 5 min from unified
  - `K_MAX: (8, 12)` - aligned with 10 max from unified
  - `MIN_FRAC: (0.02, 0.05)` - aligned with 2% min from unified
  - `MAX_FRAC: (0.15, 0.25)` - tunable around 20% max from unified

**Documentation**:
```python
# Uses DEFAULT_CLUSTERING_GOALS structural constraints:
# - Cluster count: 6-8 preferred (5-10 absolute range)
# - Cluster size: 2% min, 20% max
```

### 3. `hdbscan_regime_optimizer.py` ✅
**Changes**:
- Added import for unified clustering goals
- Added structural constraints to `HDBSCANRegimeOptimizerConfig`:
  ```python
  target_n_clusters: Tuple[int, int] = (6, 8)  # Preferred range
  min_n_clusters: int = 5  # Absolute minimum
  max_n_clusters: int = 10  # Absolute maximum
  min_cluster_size_pct: float = 0.02  # 2% minimum
  max_cluster_size_pct: float = 0.20  # 20% maximum
  ```
- Updated `_calculate_quality_metrics()` to validate against unified constraints
- Added real-time validation warnings/success messages during clustering

**Validation Output**:
```
✅ Cluster count 7 in preferred range (6, 8)
✅ All cluster sizes within bounds (2%-20%)
```

Or:
```
⚠️ Cluster count 12 outside range [5, 10]
⚠️ 2 cluster(s) violate size constraints (2%-20%)
  Cluster 0: 15 samples (1.2%) - too_small
  Cluster 3: 350 samples (28.0%) - too_large
```

### 4. Documentation ✅
**Updated Files**:
- `CLUSTERING_GOALS_SUMMARY.md` - Added section on structural constraints
- `ITERATIVE_OPT_TUNING_README.md` - Added structural constraints to goals list
- `CLUSTERING_CONSTRAINTS_UPDATE.md` (this file)

## Validation Logic

### Cluster Count Validation
```python
# Check if cluster count is in preferred range
if targets.target_clusters[0] <= n_clusters <= targets.target_clusters[1]:
    # Preferred: 6-8 clusters
    pass
elif targets.min_clusters <= n_clusters <= targets.max_clusters:
    # Acceptable: 5-10 clusters
    pass
else:
    # Outside acceptable range
    raise ValidationError(f"Cluster count {n_clusters} outside range")
```

### Cluster Size Validation
```python
# For each cluster
min_size = int(n_total_samples * 0.02)  # 2% minimum
max_size = int(n_total_samples * 0.20)  # 20% maximum

for cluster_size in cluster_sizes:
    if cluster_size < min_size:
        violations.append('too_small')
    elif cluster_size > max_size:
        violations.append('too_large')
```

## Usage Examples

### Example 1: Validate Cluster Configuration

```python
from clustering_optimization_goals import (
    DEFAULT_OPTIMIZATION_TARGETS,
    validate_cluster_sizes,
    meets_optimization_constraints
)

# Your clustering results
n_clusters = 7
cluster_sizes = [50, 75, 100, 120, 90, 80, 85]  # Example sizes
n_total_samples = 600

# Validate cluster sizes
targets = DEFAULT_OPTIMIZATION_TARGETS
sizes_valid, size_details = validate_cluster_sizes(
    cluster_sizes, 
    n_total_samples, 
    targets
)

if sizes_valid:
    print("✅ All cluster sizes valid")
else:
    print(f"❌ {size_details['n_violations']} violations")
    for v in size_details['violations']:
        print(f"  Cluster {v['cluster']}: {v['size_pct']:.1%} - {v['violation']}")
```

### Example 2: Complete Constraint Check

```python
from clustering_optimization_goals import meets_optimization_constraints

# All metrics including cluster info
all_met, checks = meets_optimization_constraints(
    cv_score=1.45,
    silhouette_score=0.25,
    dbi_score=1.8,
    balance_score=0.68,
    temporal_smoothness=0.92,
    n_clusters=7,
    cluster_sizes=[50, 75, 100, 120, 90, 80, 85],
    n_total_samples=600
)

print(f"All constraints met: {all_met}")
print(f"Cluster count check: {checks['cluster_count']}")
print(f"Preferred range check: {checks['cluster_count_preferred']}")
print(f"Cluster sizes valid: {checks['cluster_sizes_valid']}")
```

## Expected Behavior

### During Hyperparameter Tuning

The tuner will now:
1. **Enforce cluster count** during parameter search
2. **Validate cluster sizes** after each trial
3. **Penalize configurations** that violate constraints
4. **Prefer configurations** in the 6-8 cluster range
5. **Reject configurations** with clusters outside 2%-20% size bounds

### During HDBSCAN Clustering

The optimizer will now:
1. **Set initial parameters** based on unified constraints
2. **Validate results** against cluster count (6-8 preferred)
3. **Validate cluster sizes** against 2%-20% bounds
4. **Report violations** in real-time with warnings
5. **Include validation** in quality metrics report

## Benefits

1. **Consistency**: All clustering components use same constraints
2. **Prevention**: Catches problematic clustering configurations early
3. **Guidance**: Preferred ranges guide optimization toward better solutions
4. **Transparency**: Clear feedback on constraint violations
5. **Flexibility**: Constraints can be customized per use case

## Configuration Override

If you need different constraints for a specific use case:

```python
from clustering_optimization_goals import (
    ClusteringOptimizationGoals,
    OptimizationTargets
)

# Custom goals with different constraints
custom_goals = ClusteringOptimizationGoals()
custom_goals.cluster_count_range = (5, 9)  # Different preferred range
custom_goals.min_cluster_size_pct = 0.03  # 3% minimum instead of 2%
custom_goals.max_cluster_size_pct = 0.15  # 15% maximum instead of 20%

# Custom targets
custom_targets = OptimizationTargets(
    target_clusters=(5, 9),
    min_cluster_size_pct=0.03,
    max_cluster_size_pct=0.15
)

# Use in validation
all_met, checks = meets_optimization_constraints(
    ...,
    targets=custom_targets
)
```

## Testing

### Validation Tests
```bash
# Test module functionality
python3 -c "
from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
    validate_cluster_sizes,
    DEFAULT_OPTIMIZATION_TARGETS
)

# Test valid configuration
cluster_sizes = [50, 75, 100, 120, 90, 80, 85]
n_samples = 600
valid, details = validate_cluster_sizes(cluster_sizes, n_samples)
print(f'Valid: {valid}')
print(f'Details: {details}')

# Test invalid configuration (tiny cluster)
cluster_sizes_bad = [5, 75, 100, 120, 90, 80, 85]  # 5 samples = 0.83% < 2%
valid, details = validate_cluster_sizes(cluster_sizes_bad, n_samples)
print(f'Valid: {valid}')
print(f'Violations: {details[\"violations\"]}')"
```

## Migration Notes

### Existing Code
If you have existing code using hardcoded thresholds, you should migrate to unified constraints:

**Before**:
```python
# Hardcoded thresholds
min_clusters = 4
max_clusters = 8
min_size_pct = 0.01
```

**After**:
```python
from clustering_optimization_goals import DEFAULT_OPTIMIZATION_TARGETS

targets = DEFAULT_OPTIMIZATION_TARGETS
min_clusters = targets.min_clusters  # 5
max_clusters = targets.max_clusters  # 10
preferred_range = targets.target_clusters  # (6, 8)
min_size_pct = targets.min_cluster_size_pct  # 0.02
max_size_pct = targets.max_cluster_size_pct  # 0.20
```

## Version History

- **v1.1** (2025-10-28): Added structural constraints
  - Cluster count: 6-8 preferred (5-10 absolute)
  - Cluster size: 2% min, 20% max
  - Updated all clustering components to use constraints
  - Added validation functions and real-time checking

---

**Implementation Date**: 2025-10-28  
**Status**: ✅ Complete and Tested
