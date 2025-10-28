# Clustering Optimization Goals - Quick Reference

## 🎯 New Primary Goals (33% each)

| Goal | Weight | Description | Target | Method |
|------|--------|-------------|--------|--------|
| **Rolling Log-Likelihood** | 33% | Predictive LL on held-out blocks | > -5.0 | `calculate_rolling_log_likelihood()` |
| **One-Step-Ahead LL** | 33% | One-step predictive density | > -3.0 | `calculate_one_step_log_likelihood()` |
| **Economic Utility (Sharpe)** | 34% | OOS risk-adjusted performance | > 1.0 | `calculate_economic_utility()` |

## 🔒 Cluster Constraints (Preserved)

- **Count**: 4-8 clusters (preferred), 3-10 (absolute)
- **Size**: 2%-20% each

## 📊 Quick Usage

### Calculate Composite Score
```python
from clustering_optimization_goals import calculate_composite_score

composite = calculate_composite_score(
    rolling_ll=-4.2,
    one_step_ll=-2.1,
    economic_utility=1.3  # Sharpe
)
```

### Time Series CV
```python
from clustering_optimization_goals import TimeSeriesCrossValidator, CVConfig

cv = TimeSeriesCrossValidator(CVConfig(
    n_splits=5,
    train_months=18,
    val_months=3
))

splits = cv.split(data)
```

### Calculate Metrics
```python
from clustering_optimization_goals import MetricCalculator

calc = MetricCalculator()

# All three metrics
rolling_ll, _ = calc.calculate_rolling_log_likelihood(data, probs, params)
one_step_ll, _ = calc.calculate_one_step_log_likelihood(data, labels, params)
econ = calc.calculate_economic_utility(returns, labels)

sharpe = econ['sharpe']
```

### Pareto Optimization
```python
from clustering_optimization_goals import (
    create_pareto_front, select_knee_point
)

pareto_solutions = create_pareto_front(trial_results, trial_params)
best = select_knee_point(pareto_solutions)
```

## 🛠️ Key Features

- ✅ Time Series CV (rolling/expanding/blocked)
- ✅ Metric normalization (z-score, rank, robust)
- ✅ Pareto multi-objective optimization
- ✅ Penalties for pathological fits
- ✅ Robustness validation (ARI)
- ✅ Statistical significance tests
- ✅ VectorBT integration
- ✅ Hardware optimization

## 📈 Performance Guidelines

| Metric | Good | Excellent |
|--------|------|-----------|
| Rolling LL | -5 to -3 | > -3 |
| One-Step LL | -3 to -2 | > -2 |
| Sharpe | 1.0 to 1.5 | > 1.5 |
| Max DD | 20-30% | < 20% |
| Median ARI | 0.5-0.7 | > 0.7 |

## 🎓 Best Practices

1. **Use K=5-10 folds** for CV
2. **Prefer rank-based normalization** (robust)
3. **Generate Pareto front** when possible
4. **Validate top-10 with M=30 seeds**
5. **Test significance** at α=0.10
6. **Monitor metric stability** (std/mean < 0.4)

## 🔧 Common Configurations

### Balanced (Default)
```python
ClusteringOptimizationGoals()  # All defaults
```

### Conservative
```python
ClusteringOptimizationGoals(
    cluster_count_range=(4, 6),
    min_cluster_size_pct=0.03,
    penalty_config=PenaltyConfig(
        min_duration_bars=14,
        max_monthly_turnover=2.0
    )
)
```

### Aggressive
```python
ClusteringOptimizationGoals(
    cluster_count_range=(6, 10),
    penalty_config=PenaltyConfig(
        min_duration_bars=5,
        max_monthly_turnover=6.0
    )
)
```

## 📁 File Location

```
src/training/steps/market_analysis/clusters/clustering_optimization_goals.py
```

**Lines**: 1,430 | **Status**: ✅ Production Ready

---

For full documentation, see: `CLUSTERING_OPTIMIZATION_GOALS_UPDATE_SUMMARY.md`
