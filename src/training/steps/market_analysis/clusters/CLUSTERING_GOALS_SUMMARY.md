# Unified Clustering Optimization Goals

## Overview

All clustering optimization components now use a unified set of goals and targets defined in `clustering_optimization_goals.py`. This ensures consistency and makes hyperparameter tuning easier.

## Affected Components

1. **iterative_optimization_tuner.py** - Hyperparameter tuning for iterative optimization
2. **iterative_optimization.py** - The iterative clustering optimization loop
3. **hdbscan_clustering optimization** - HDBSCAN regime optimizer quality metrics
4. **regime_clustering_step.py** - Clustering validation and quality checks

## Common Optimization Goals

All components now optimize for these shared goals:

### Primary Goals (70% weight total)

1. **CV Score** (30% weight)
   - Metric: Between/Within Variance Ratio (Calinski-Harabasz)
   - Objective: **MAXIMIZE**
   - Target: ≥1.0 (excellent: ≥2.0)
   - Description: Measures cluster separation quality

2. **Silhouette Score** (25% weight)
   - Metric: Cluster cohesion and separation
   - Objective: **MAXIMIZE**
   - Range: -1 (worst) to 1 (best)
   - Target: ≥0.2 (excellent: ≥0.5)
   - Description: How well points fit in their assigned cluster

3. **DBI Score** (20% weight)
   - Metric: Davies-Bouldin Index
   - Objective: **MINIMIZE**
   - Target: ≤2.0 (excellent: ≤1.0)
   - Description: Average similarity ratio of clusters

### Secondary Goals - Soft Constraints (30% weight total)

4. **Balance Score** (15% weight)
   - Metric: Cluster size balance
   - Objective: **MAINTAIN**
   - Target: ≥0.5 (excellent: ≥0.7)
   - Constraint: Soft constraint, should be above 0.5
   - Description: Prevents overly imbalanced clusters

5. **Temporal Smoothness** (10% weight)
   - Metric: Temporal stability
   - Objective: **MAINTAIN**
   - Target: ≥0.85 (excellent: ≥0.95)
   - Constraint: Soft constraint, should be above 0.85
   - Description: Prevents excessive regime switching

### Structural Constraints (Hard Constraints)

6. **Cluster Count**
   - Preferred Range: **6-8 clusters**
   - Absolute Range: 5-10 clusters
   - Description: Optimal number of market regimes for trading strategies

7. **Cluster Size Bounds**
   - Minimum: **2%** of total samples
   - Maximum: **20%** of total samples
   - Description: Prevents tiny clusters and dominant clusters

## Composite Score Calculation

The weighted composite score is calculated as:

```python
from clustering_optimization_goals import calculate_composite_score

composite = calculate_composite_score(
    cv_score=1.45,
    silhouette_score=0.25,
    dbi_score=1.8,
    balance_score=0.68,
    temporal_smoothness=0.92
)
```

**Formula:**
```
composite = (
    0.30 * cv_score +
    0.25 * silhouette_score +
    0.20 * (1 / (1 + dbi_score)) +  # Inverted since lower is better
    0.15 * balance_score +
    0.10 * temporal_smoothness
)
```

## Optimization Targets

### Minimum Acceptable Values
- CV Score: ≥1.0
- Silhouette Score: ≥0.2
- DBI Score: ≤2.0
- Balance Score: ≥0.5
- Temporal Smoothness: ≥0.85
- Cluster Count: 5-10 clusters (preferred: 6-8)

### Aspirational Targets (Excellent Performance)
- CV Score: ≥1.5
- Silhouette Score: ≥0.3
- DBI Score: ≤1.5
- Balance Score: ≥0.7
- Temporal Smoothness: ≥0.95

## Usage Examples

### Example 1: Using Default Goals

```python
from clustering_optimization_goals import DEFAULT_CLUSTERING_GOALS, DEFAULT_OPTIMIZATION_TARGETS

# Get all goals
goals = DEFAULT_CLUSTERING_GOALS
for goal_name, goal_config in goals.get_all_goals().items():
    print(f"{goal_config.name}: weight={goal_config.weight}, objective={goal_config.objective.value}")

# Get optimization targets
targets = DEFAULT_OPTIMIZATION_TARGETS
print(f"Min CV Score: {targets.min_cv_score}")
print(f"Min Silhouette: {targets.min_silhouette_score}")
print(f"Max DBI: {targets.max_dbi_score}")
```

### Example 2: Calculating Composite Score

```python
from clustering_optimization_goals import calculate_composite_score

# Your clustering metrics
cv = 1.45
sil = 0.25
dbi = 1.8
bal = 0.68
temp = 0.92

# Calculate composite score (higher is better)
composite = calculate_composite_score(cv, sil, dbi, bal, temp)
print(f"Composite Score: {composite:.4f}")
```

### Example 3: Checking Constraints

```python
from clustering_optimization_goals import meets_optimization_constraints

# Check if metrics meet minimum constraints
all_met, checks = meets_optimization_constraints(
    cv_score=1.45,
    silhouette_score=0.25,
    dbi_score=1.8,
    balance_score=0.68,
    temporal_smoothness=0.92,
    n_clusters=7
)

print(f"All constraints met: {all_met}")
for metric, passed in checks.items():
    print(f"  {metric}: {'✅' if passed else '❌'}")
```

### Example 4: Generating Metrics Report

```python
from clustering_optimization_goals import format_metrics_report

report = format_metrics_report(
    cv_score=1.45,
    silhouette_score=0.25,
    dbi_score=1.8,
    balance_score=0.68,
    temporal_smoothness=0.92,
    n_clusters=7
)
print(report)
```

Output:
```
============================================================
CLUSTERING OPTIMIZATION METRICS REPORT
============================================================

Composite Score: 0.6234
Number of Clusters: 7

Primary Metrics:
  CV Score:          1.4500 (target: ≥1.00) ✅
  Silhouette:        0.2500 (target: ≥0.20) ✅
  DBI Score:         1.8000 (target: ≤2.00) ✅

Constraint Metrics:
  Balance:           0.6800 (target: ≥0.50) ✅
  Temporal:          0.9200 (target: ≥0.85) ✅

Overall Status: ✅ ALL CONSTRAINTS MET
============================================================
```

## Integration Guide

### For iterative_optimization_tuner.py

Already integrated! The tuner now imports and uses:
```python
from clustering_optimization_goals import (
    DEFAULT_CLUSTERING_GOALS,
    DEFAULT_OPTIMIZATION_TARGETS,
    calculate_composite_score,
    meets_optimization_constraints
)
```

### For regime_clustering_step.py

Already integrated! The step now uses unified targets:
```python
from clustering_optimization_goals import (
    DEFAULT_OPTIMIZATION_TARGETS,
    format_metrics_report
)

# Use unified targets
if UNIFIED_GOALS_AVAILABLE and DEFAULT_OPTIMIZATION_TARGETS:
    unified_targets = DEFAULT_OPTIMIZATION_TARGETS
    min_cv_score = config.get('min_cv_score', unified_targets.min_cv_score)
    # ... etc
```

### For New Components

To integrate unified goals in new clustering components:

```python
# Import the goals
from clustering_optimization_goals import (
    DEFAULT_CLUSTERING_GOALS,
    DEFAULT_OPTIMIZATION_TARGETS,
    calculate_composite_score,
    meets_optimization_constraints,
    format_metrics_report
)

# Use in your optimization logic
goals = DEFAULT_CLUSTERING_GOALS
weights = goals.get_weights_dict()

# Calculate composite score
composite = calculate_composite_score(
    cv_score, silhouette_score, dbi_score, 
    balance_score, temporal_smoothness
)

# Validate constraints
all_met, checks = meets_optimization_constraints(
    cv_score, silhouette_score, dbi_score,
    balance_score, temporal_smoothness, n_clusters
)
```

## Benefits

1. **Consistency**: All clustering components use the same optimization goals
2. **Easy Tuning**: Change weights/targets in one place, affects all components
3. **Clear Documentation**: Goals are well-documented in one location
4. **Type Safety**: Using dataclasses and enums for better code quality
5. **Flexibility**: Easy to customize goals for specific use cases

## Customization

To customize goals for your use case:

```python
from clustering_optimization_goals import ClusteringOptimizationGoals, OptimizationTargets

# Create custom goals with different weights
custom_goals = ClusteringOptimizationGoals()
custom_goals.cv_score.weight = 0.40  # Increase CV importance
custom_goals.silhouette_score.weight = 0.30  # Increase Silhouette importance
custom_goals.dbi_score.weight = 0.15
custom_goals.balance_score.weight = 0.10
custom_goals.temporal_smoothness.weight = 0.05
custom_goals.normalize_weights()  # Ensure weights sum to 1.0

# Create custom targets
custom_targets = OptimizationTargets(
    min_cv_score=1.2,  # More stringent
    min_silhouette_score=0.25,  # More stringent
    max_dbi_score=1.8,  # More stringent
    target_clusters=(5, 7)  # Different preferred range
)

# Use in calculations
composite = calculate_composite_score(
    cv, sil, dbi, bal, temp, 
    goals=custom_goals
)
```

## References

- `clustering_optimization_goals.py`: Main module with goals and utilities
- `iterative_optimization_tuner.py`: Hyperparameter tuning implementation
- `regime_clustering_step.py`: Clustering validation implementation
- `ITERATIVE_OPT_TUNING_README.md`: Tuning guide

## Version History

- **v1.0** (2025-10-28): Initial unified goals implementation
  - Centralized goals from iterative_optimization_tuner.py
  - Integrated with regime_clustering_step.py
  - Added utility functions for composite score and constraint checking
