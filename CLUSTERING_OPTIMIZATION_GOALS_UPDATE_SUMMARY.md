# Clustering Optimization Goals Update - Complete Summary

## ✅ Update Complete

Successfully updated `clustering_optimization_goals.py` with **predictive & economic focus**, implementing all requested features and integrating with existing utilities.

---

## 🎯 Main Goals Changed (33% each)

### Before (Old Goals)
- ❌ CV Score (30%) - Between/Within Variance Ratio
- ❌ Silhouette Score (25%) - Cluster cohesion
- ❌ DBI Score (20%) - Davies-Bouldin Index
- ⚠️ Balance (15%) - Constraint
- ⚠️ Temporal Smoothness (10%) - Constraint

### After (New Goals)
- ✅ **Rolling Predictive Log-Likelihood (33%)** - Preferred metric
- ✅ **One-Step-Ahead Log-Likelihood (33%)** - Predictive density
- ✅ **Economic Utility / OOS Sharpe (34%)** - Risk-adjusted performance

---

## 🔒 Cluster Constraints (Preserved)

✅ **Maintained as requested:**
- **Cluster count**: 4-8 (preferred), 3-10 (absolute limits)
- **Cluster size distribution**: 2%-20% each

---

## 📦 What Was Implemented

### 1. New Optimization Goals

```python
class OptimizationGoal(Enum):
    # PRIMARY GOALS (33% each)
    ROLLING_LOG_LIKELIHOOD = "rolling_log_likelihood"  # 33%
    ONE_STEP_LOG_LIKELIHOOD = "one_step_log_likelihood"  # 33%
    ECONOMIC_UTILITY = "economic_utility"  # 34% (OOS Sharpe)
```

**Key Features**:
- Weighted composite scoring (33%/33%/34%)
- Normalization support (z-score, rank, robust)
- Stability thresholds (std/mean < 0.4)
- Target ranges for each metric

### 2. Time Series Cross-Validation

**Class**: `TimeSeriesCrossValidator`

**Features**:
- **K=5-10 folds** (configurable)
- **Three strategies**:
  - `ROLLING`: train=18mo, val=3mo, slide forward (recommended for crypto)
  - `EXPANDING`: expanding window
  - `BLOCKED`: non-overlapping blocks
  - `PURGED`: purged CV for financial data (future extension)
- **Automatic frequency detection** from datetime index
- **Minimum sample requirements**: train≥200, val≥50
- **Robust statistics**: median & IQR across folds

**Example**:
```python
cv_config = CVConfig(
    n_splits=5,
    strategy=CVStrategy.ROLLING,
    train_months=18,
    val_months=3
)

cv = TimeSeriesCrossValidator(cv_config)
splits = cv.split(data)  # Returns [(train_idx, val_idx), ...]
```

### 3. Metric Calculators

**Class**: `MetricCalculator`

**Three Main Metrics**:

#### a) Rolling Predictive Log-Likelihood
```python
mean_ll, std_ll = calculator.calculate_rolling_log_likelihood(
    data=held_out_data,
    regime_probs=probabilities,
    regime_params=params
)
```
- Mixture model log-likelihood across held-out blocks
- Weighted by regime probabilities
- Numerical stability (clipping, regularization)

#### b) One-Step-Ahead Log-Likelihood
```python
mean_ll, std_ll = calculator.calculate_one_step_log_likelihood(
    data=data,
    regime_labels=labels,
    regime_params=params
)
```
- Predictive density using previous regime
- Rolling one-step forecasts
- Stable across regime switches

#### c) Economic Utility (Sharpe & More)
```python
metrics = calculator.calculate_economic_utility(
    returns=returns,
    regime_labels=labels,
    regime_signals=signals  # Optional
)
# Returns: sharpe, max_drawdown, monthly_turnover, win_rate, etc.
```
- **Annualized Sharpe ratio** (primary)
- **Maximum drawdown**
- **Monthly turnover** (regime switches)
- **Win rate**
- **Default regime signals**: long volatile regimes, short calm regimes

### 4. Metric Normalization

**Class**: `MetricNormalizer`

**Four Methods**:
1. **Z-Score**: `(value - mean) / std`
2. **Rank** (preferred): Rank-based to [0, 1], robust to outliers
3. **Robust Z-Score**: `(value - median) / MAD`
4. **Min-Max**: `(value - min) / (max - min)`

**Example**:
```python
normalizer = MetricNormalizer(method=NormalizationMethod.RANK)

metrics = {
    'rolling_ll': [-5.2, -3.8, -4.5, -6.1],
    'one_step_ll': [-2.1, -1.8, -2.5, -3.0],
    'sharpe': [1.2, 1.5, 0.9, 1.8]
}

objectives = {
    'rolling_ll': OptimizationObjective.MAXIMIZE,
    'one_step_ll': OptimizationObjective.MAXIMIZE,
    'sharpe': OptimizationObjective.MAXIMIZE
}

normalized = normalizer.normalize_metrics(metrics, objectives)
# All normalized to [0, 1], higher is better
```

### 5. Pareto Front Optimization

**Three Key Functions**:

#### a) Create Pareto Front
```python
pareto_solutions = create_pareto_front(
    trial_results=[{
        'rolling_ll': -4.2,
        'one_step_ll': -2.1,
        'sharpe': 1.3
    }, ...],
    trial_params=[{'n_clusters': 6, ...}, ...]
)
# Returns non-dominated solutions
```

#### b) Select Knee Point
```python
best_solution = select_knee_point(pareto_solutions)
# Automatically selects balanced tradeoff (closest to ideal)
```

#### c) Rank Solutions
```python
ranked = rank_pareto_solutions(pareto_solutions)
# Returns [(rank, composite_score, solution), ...] sorted
```

**Benefits**:
- No arbitrary normalization needed
- Preserves tradeoff frontier
- Multiple solution options
- Automatic knee point selection

### 6. Penalties & Soft Constraints

**Class**: `PenaltyConfig`

**Six Penalty Types**:

1. **Minimum Occupancy Penalty**
   - Threshold: 1% minimum (configurable)
   - Penalty: 10.0 * (threshold - actual)
   - Prevents tiny clusters

2. **Minimum Duration Penalty**
   - Threshold: 7 bars for daily data
   - Penalty: 5.0 * (threshold - actual)
   - Prevents excessive switching

3. **Turnover Penalty**
   - Threshold: 4 switches/month max
   - Penalty: 3.0 * (actual - threshold)
   - Keeps strategies realistic

4. **Stability Penalty (ARI)**
   - Threshold: median ARI ≥ 0.5 across restarts
   - Penalty: 5.0 * (threshold - actual)
   - Ensures reproducibility

5. **Calibration Penalty (CRPS/PIT)**
   - Threshold: error < 0.2
   - Penalty: 4.0 * (actual - threshold)
   - Ensures well-calibrated probabilities

6. **Metric Stability Penalty**
   - Threshold: std/mean < 0.4
   - Penalty: 3.0 * (actual - threshold)
   - Requires stable metrics across folds

**Example**:
```python
penalties = calculate_penalties(
    regime_labels=labels,
    n_total_samples=len(data),
    regime_durations=durations,
    monthly_turnover=2.5,
    ari_scores=[0.65, 0.72, 0.68],
    metric_cv_variation=0.35
)

# Apply to composite score
final_score = composite_score - sum(penalties.values())
```

### 7. Robustness & Statistical Checks

#### a) Robustness Validation
```python
is_robust, metrics = validate_robustness(
    candidate_params={'n_clusters': 6, ...},
    data=df,
    clustering_fn=hdbscan_cluster,
    n_seeds=30,
    min_ari=0.5
)

# Returns:
# - is_robust: True/False
# - metrics: {'median_ari': 0.65, 'mean_ari': 0.63, 'std_ari': 0.08}
```

**Features**:
- Tests M=30 random seeds (configurable)
- Computes pairwise ARI between runs
- Requires median ARI ≥ 0.5
- Returns full distribution statistics

#### b) Statistical Significance Test
```python
is_significant, p_value, metrics = statistical_significance_test(
    strategy_returns=regime_aware_returns,
    baseline_returns=buy_hold_returns,
    n_bootstrap=100,
    alpha=0.10
)

# Tests if Sharpe improvement is statistically significant
# Returns:
# - is_significant: True if p_value < alpha
# - p_value: block bootstrap p-value
# - metrics: {'strategy_sharpe': 1.3, 'baseline_sharpe': 0.8, 'sharpe_diff': 0.5}
```

**Features**:
- **Block bootstrap** to preserve autocorrelation
- **One-sided test**: strategy > baseline
- **Configurable significance level** (default: α=0.10)
- Block size: √n (automatic)

---

## 🛠️ Integration with Existing Tools

### 1. VectorBT
```python
# Imported for efficient rolling computations
from src.vectorbt import (
    vbt, rolling_mean, rolling_std, rolling_var,
    rolling_min, rolling_max, rolling_sum, rolling_apply,
    VECTORBT_AVAILABLE
)
```
- Used in `MetricCalculator` for fast computations
- Graceful fallback if not available

### 2. Pareto Front Utilities
```python
from src.utils.ml_common.optimization.pareto import ParetoFront, Solution
```
- Multi-objective optimization
- Non-dominated solution discovery
- Knee point selection

### 3. Matrix Cross-Validation
```python
from src.utils.ml_common.matrix_cross_validation import MatrixCrossValidator
```
- Optimized CV with GPU acceleration (optional)
- VectorBT portfolio-based evaluation

### 4. Feature Normalization/Scaling
```python
from src.features_common.transforms.vectorbt_scaler import VectorBTScaler
from src.features_common.transforms.scaling_normalization import (
    zscore_normalize, rank_normalize, robust_normalize
)
```
- Robust normalization methods
- Rank-based scaling (preferred)

### 5. Hardware Optimization
```python
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
from src.utils.hardware.m1_gpu_utils import M1GPUManager
```
- Optional GPU acceleration
- CPU optimization for parallel processing

---

## 📊 Usage Examples

### Example 1: Basic Composite Score

```python
from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
    calculate_composite_score, DEFAULT_CLUSTERING_GOALS
)

# Metrics from clustering evaluation
rolling_ll = -4.2
one_step_ll = -2.1
sharpe = 1.3

composite = calculate_composite_score(
    rolling_ll=rolling_ll,
    one_step_ll=one_step_ll,
    economic_utility=sharpe
)

print(f"Composite score: {composite:.4f}")
```

### Example 2: Time Series Cross-Validation

```python
from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
    TimeSeriesCrossValidator, CVConfig, CVStrategy
)

# Configure CV for crypto (daily data)
cv_config = CVConfig(
    n_splits=5,
    strategy=CVStrategy.ROLLING,
    train_months=18,
    val_months=3,
    min_train_samples=200,
    min_val_samples=50
)

cv = TimeSeriesCrossValidator(cv_config)

# Generate splits
splits = cv.split(market_data_df)

# Use splits for evaluation
for fold_idx, (train_idx, val_idx) in enumerate(splits):
    train_data = market_data_df.iloc[train_idx]
    val_data = market_data_df.iloc[val_idx]
    
    # Fit clustering on train, evaluate on val
    clustering_result = fit_clustering(train_data)
    metrics = evaluate_on_validation(clustering_result, val_data)
    
    print(f"Fold {fold_idx+1}: Sharpe = {metrics['sharpe']:.4f}")
```

### Example 3: Metric Calculation

```python
from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
    MetricCalculator
)

calculator = MetricCalculator(use_vectorbt=True)

# Calculate all three primary metrics
rolling_ll, rolling_std = calculator.calculate_rolling_log_likelihood(
    data=val_data,
    regime_probs=regime_probabilities,
    regime_params=regime_parameters
)

one_step_ll, one_step_std = calculator.calculate_one_step_log_likelihood(
    data=val_data,
    regime_labels=regime_labels,
    regime_params=regime_parameters
)

economic_metrics = calculator.calculate_economic_utility(
    returns=returns,
    regime_labels=regime_labels
)

print(f"Rolling LL: {rolling_ll:.4f} ± {rolling_std:.4f}")
print(f"One-Step LL: {one_step_ll:.4f} ± {one_step_std:.4f}")
print(f"Sharpe: {economic_metrics['sharpe']:.4f}")
print(f"Max DD: {economic_metrics['max_drawdown']:.2%}")
print(f"Turnover: {economic_metrics['monthly_turnover']:.2f}/month")
```

### Example 4: Pareto Optimization Workflow

```python
from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
    create_pareto_front, select_knee_point, rank_pareto_solutions
)

# After running multiple trials with different hyperparameters
trial_results = [
    {'rolling_ll': -4.2, 'one_step_ll': -2.1, 'sharpe': 1.3},
    {'rolling_ll': -3.8, 'one_step_ll': -2.5, 'sharpe': 1.1},
    {'rolling_ll': -5.1, 'one_step_ll': -1.8, 'sharpe': 1.6},
    # ... more trials
]

trial_params = [
    {'n_clusters': 6, 'min_cluster_size': 25},
    {'n_clusters': 7, 'min_cluster_size': 20},
    {'n_clusters': 5, 'min_cluster_size': 30},
    # ... corresponding params
]

# Create Pareto front
pareto_solutions = create_pareto_front(trial_results, trial_params)

print(f"Pareto front: {len(pareto_solutions)} non-dominated solutions")

# Option 1: Auto-select knee point (balanced tradeoff)
best_solution = select_knee_point(pareto_solutions)
print(f"\nKnee point solution:")
print(f"  Params: {best_solution['params']}")
print(f"  Metrics: {best_solution['metrics']}")

# Option 2: Rank all Pareto solutions by composite score
ranked = rank_pareto_solutions(pareto_solutions)

print(f"\nTop 3 Pareto solutions:")
for rank, score, solution in ranked[:3]:
    print(f"  Rank {rank}: Score = {score:.4f}")
    print(f"    Sharpe: {solution['metrics']['sharpe']:.4f}")
    print(f"    Rolling LL: {solution['metrics']['rolling_ll']:.4f}")
```

### Example 5: With Penalties

```python
from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
    calculate_composite_score, calculate_penalties, PenaltyConfig
)

# Custom penalty configuration
penalty_config = PenaltyConfig(
    min_occupancy_pct=0.02,  # 2% minimum
    min_duration_bars=10,  # 10 bars minimum duration
    max_monthly_turnover=3.0,  # 3 switches/month max
    min_ari_stability=0.6  # Stricter ARI requirement
)

# Calculate penalties
penalties = calculate_penalties(
    regime_labels=labels,
    n_total_samples=len(data),
    regime_durations=durations,
    monthly_turnover=2.5,
    ari_scores=[0.65, 0.72, 0.68],
    metric_cv_variation=0.35,
    penalty_config=penalty_config
)

# Calculate composite with penalties
composite = calculate_composite_score(
    rolling_ll=-4.2,
    one_step_ll=-2.1,
    economic_utility=1.3,
    penalties=penalties
)

print(f"Raw composite: {composite + sum(penalties.values()):.4f}")
print(f"Penalties: {penalties}")
print(f"Final composite: {composite:.4f}")
```

### Example 6: Robustness Validation

```python
from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
    validate_robustness, statistical_significance_test
)

# Validate top candidate
candidate_params = {
    'n_clusters': 6,
    'min_cluster_size': 25,
    'metric': 'euclidean'
}

is_robust, robustness_metrics = validate_robustness(
    candidate_params=candidate_params,
    data=market_data,
    clustering_fn=hdbscan_cluster_fn,
    n_seeds=30,
    min_ari=0.5
)

if is_robust:
    print(f"✅ Robust solution: median ARI = {robustness_metrics['median_ari']:.4f}")
else:
    print(f"❌ Unstable solution: median ARI = {robustness_metrics['median_ari']:.4f}")

# Test statistical significance of Sharpe improvement
is_significant, p_value, sig_metrics = statistical_significance_test(
    strategy_returns=regime_strategy_returns,
    baseline_returns=buy_hold_returns,
    n_bootstrap=100,
    alpha=0.10
)

if is_significant:
    print(f"✅ Significant Sharpe improvement: {sig_metrics['sharpe_diff']:.4f}")
    print(f"   p-value = {p_value:.4f} < 0.10")
else:
    print(f"⚠️ Not statistically significant: p-value = {p_value:.4f}")
```

---

## 📁 File Structure

```python
clustering_optimization_goals.py (1,430 lines)
├── Imports & Dependencies (50 lines)
│   ├── VectorBT (optional)
│   ├── Pareto optimization (optional)
│   ├── Matrix CV (optional)
│   ├── Feature scaling (optional)
│   └── Hardware optimization (optional)
│
├── Enums (30 lines)
│   ├── OptimizationGoal (8 goals)
│   ├── OptimizationObjective (3 types)
│   ├── NormalizationMethod (4 methods)
│   └── CVStrategy (4 strategies)
│
├── Configuration Classes (200 lines)
│   ├── GoalConfig
│   ├── CVConfig
│   ├── PenaltyConfig
│   ├── ClusteringOptimizationGoals
│   └── OptimizationTargets
│
├── Time Series Cross-Validation (150 lines)
│   └── TimeSeriesCrossValidator
│       ├── split()
│       ├── _rolling_split()
│       ├── _expanding_split()
│       └── _blocked_split()
│
├── Metric Calculators (250 lines)
│   └── MetricCalculator
│       ├── calculate_rolling_log_likelihood()
│       ├── calculate_one_step_log_likelihood()
│       ├── calculate_economic_utility()
│       └── Helper methods (_calculate_sharpe, _calculate_max_drawdown, etc.)
│
├── Normalization (100 lines)
│   └── MetricNormalizer
│       ├── normalize_metrics()
│       ├── _zscore_normalize()
│       ├── _rank_normalize()
│       ├── _robust_zscore_normalize()
│       └── _minmax_normalize()
│
├── Composite Score & Penalties (100 lines)
│   ├── calculate_composite_score()
│   └── calculate_penalties()
│
├── Validation Utilities (100 lines)
│   ├── validate_cluster_sizes()
│   ├── meets_optimization_constraints()
│   └── format_metrics_report()
│
├── Pareto Front Utilities (150 lines)
│   ├── create_pareto_front()
│   ├── select_knee_point()
│   └── rank_pareto_solutions()
│
├── Robustness Validation (200 lines)
│   ├── validate_robustness()
│   └── statistical_significance_test()
│
└── Example Usage (100 lines)
    └── Demonstration of all features
```

---

## 🔄 Migration Guide

### For Existing Code Using Old Goals

**Before**:
```python
from clustering_optimization_goals import (
    calculate_composite_score,
    DEFAULT_CLUSTERING_GOALS
)

# Old signature
composite = calculate_composite_score(
    cv_score=1.45,
    silhouette_score=0.25,
    dbi_score=1.8,
    balance_score=0.68,
    temporal_smoothness=0.92
)
```

**After**:
```python
from clustering_optimization_goals import (
    calculate_composite_score,
    DEFAULT_CLUSTERING_GOALS
)

# New signature (three primary metrics)
composite = calculate_composite_score(
    rolling_ll=-4.2,
    one_step_ll=-2.1,
    economic_utility=1.3  # Sharpe ratio
)
```

### Backward Compatibility

The old `OptimizationGoal` enum members are **still present** but deprecated:
- `CV_SCORE` (kept for compatibility)
- `SILHOUETTE` (kept for compatibility)
- `DBI` (kept for compatibility)

**Recommendation**: Migrate to new goals for better predictive and economic performance.

---

## ⚙️ Configuration Examples

### Conservative Configuration (Fewer Clusters, High Stability)

```python
goals = ClusteringOptimizationGoals(
    # Keep default weights (33%/33%/34%)
    cluster_count_range=(4, 6),  # Prefer 4-6 clusters
    min_cluster_size_pct=0.03,  # 3% minimum (stricter)
    max_cluster_size_pct=0.25,  # 25% maximum (relaxed)
    
    cv_config=CVConfig(
        n_splits=10,  # More folds for stability
        train_months=24,  # Longer training window
        val_months=3
    ),
    
    penalty_config=PenaltyConfig(
        min_occupancy_pct=0.03,
        min_duration_bars=14,  # 2 weeks minimum
        max_monthly_turnover=2.0,  # Lower turnover
        min_ari_stability=0.6  # Stricter stability
    )
)
```

### Aggressive Configuration (More Clusters, Higher Turnover)

```python
goals = ClusteringOptimizationGoals(
    cluster_count_range=(6, 10),  # Prefer 6-10 clusters
    min_cluster_size_pct=0.01,  # 1% minimum (relaxed)
    max_cluster_size_pct=0.15,  # 15% maximum (stricter)
    
    cv_config=CVConfig(
        n_splits=5,  # Fewer folds
        train_months=12,  # Shorter training
        val_months=2
    ),
    
    penalty_config=PenaltyConfig(
        min_occupancy_pct=0.01,
        min_duration_bars=5,  # Shorter duration OK
        max_monthly_turnover=6.0,  # Higher turnover OK
        min_ari_stability=0.4  # More relaxed
    )
)
```

### Economic-Focused Configuration

```python
goals = ClusteringOptimizationGoals(
    # Increase economic utility weight
    rolling_log_likelihood=GoalConfig(
        name="Rolling LL",
        objective=OptimizationObjective.MAXIMIZE,
        weight=0.25,  # Reduced
        # ... other settings
    ),
    one_step_log_likelihood=GoalConfig(
        name="One-Step LL",
        objective=OptimizationObjective.MAXIMIZE,
        weight=0.25,  # Reduced
        # ... other settings
    ),
    economic_utility=GoalConfig(
        name="Economic Utility",
        objective=OptimizationObjective.MAXIMIZE,
        weight=0.50,  # Increased! (50% weight on Sharpe)
        # ... other settings
    )
)
```

---

## 🎓 Best Practices

### 1. Metric Measurement

✅ **DO**:
- Use K=5-10 folds for crypto daily data
- Use rolling windows (train=18mo, val=3mo)
- Report median & IQR (robust statistics)
- Check metric stability (std/mean < 0.4)
- Test significance with block bootstrap

❌ **DON'T**:
- Use random splits for time series
- Use too few folds (K<3)
- Report only mean without std
- Ignore metric variability
- Trust improvements without significance test

### 2. Normalization

✅ **DO**:
- Use rank-based normalization (robust to outliers)
- Normalize across all trials
- Ensure all metrics point same direction (maximize)
- Check for outliers before normalization

❌ **DON'T**:
- Mix normalized and un-normalized metrics
- Normalize within each fold separately
- Ignore extreme outliers
- Use min-max on heavy-tailed distributions

### 3. Pareto Optimization

✅ **DO**:
- Generate Pareto front when compute allows
- Inspect tradeoffs between metrics
- Select knee point for balanced solution
- Consider top-K Pareto solutions

❌ **DON'T**:
- Force single composite score too early
- Ignore dominated solutions entirely (may be robust)
- Select extreme Pareto points (edge cases)

### 4. Penalties

✅ **DO**:
- Apply penalties consistently
- Tune penalty weights for your data
- Validate penalty thresholds
- Monitor penalty distributions

❌ **DON'T**:
- Over-penalize (kills exploration)
- Under-penalize (allows pathological fits)
- Use binary constraints (use soft penalties)
- Ignore penalty interactions

### 5. Robustness

✅ **DO**:
- Validate top-10 candidates with M=30 seeds
- Require median ARI ≥ 0.5
- Test significance at α=0.10
- Check posterior predictive quality

❌ **DON'T**:
- Accept unstable solutions (low ARI)
- Trust non-significant improvements
- Skip robustness checks
- Ignore failed restarts

---

## 📊 Expected Performance

### Metric Ranges (Guidelines)

| Metric | Poor | Acceptable | Good | Excellent |
|--------|------|------------|------|-----------|
| **Rolling LL** | < -10 | -10 to -5 | -5 to -3 | > -3 |
| **One-Step LL** | < -6 | -6 to -3 | -3 to -2 | > -2 |
| **Sharpe** | < 0.5 | 0.5 to 1.0 | 1.0 to 1.5 | > 1.5 |
| **Max Drawdown** | > 40% | 30-40% | 20-30% | < 20% |
| **Turnover** | > 8/mo | 4-8/mo | 2-4/mo | < 2/mo |
| **Median ARI** | < 0.3 | 0.3-0.5 | 0.5-0.7 | > 0.7 |

### Typical Composite Scores

- **Poor clustering**: -5 to 0
- **Acceptable clustering**: 0 to 0.5
- **Good clustering**: 0.5 to 1.0
- **Excellent clustering**: > 1.0

(After normalization and penalties)

---

## 🚀 Production Considerations

### 1. Retraining Cadence

**Recommendations**:
- **Weekly**: For highly dynamic markets
- **Monthly**: For stable markets (default)
- **Quarterly**: For long-term strategies

### 2. Warm Starts

```python
# Use previous best parameters as initialization
previous_best_params = load_best_params()

# Initialize optimizer with warm start
optimizer.set_initial_params(previous_best_params)
```

### 3. Monitoring

**Key Metrics to Track**:
- Out-of-sample predictive LL (drift detection)
- Regime occupancy distribution (drift)
- Sharpe ratio degradation
- ARI stability across retrain cycles

**Alert Thresholds**:
- Predictive LL drops > 20% → retrain
- Sharpe drops below 0.5 → retrain
- Median ARI < 0.4 → investigate instability

### 4. Ensemble

```python
# Average regime probabilities from top K models
top_k_models = select_top_k(pareto_solutions, k=5)

ensemble_probs = np.mean([
    model.predict_proba(data)
    for model in top_k_models
], axis=0)
```

**Benefits**:
- Reduces label-switching
- Improves stability
- Better calibration

---

## 🔧 Troubleshooting

### Issue: Metrics Too Variable (std/mean > 0.4)

**Solutions**:
1. Increase fold count (K=10 instead of K=5)
2. Use longer validation windows (4-6 months)
3. Apply robust statistics (median instead of mean)
4. Check for data quality issues

### Issue: No Significant Sharpe Improvement

**Solutions**:
1. Refine regime signals (not just vol-based)
2. Add transaction costs to baseline
3. Use longer evaluation period
4. Consider regime-conditional strategies

### Issue: Low ARI (Unstable Clustering)

**Solutions**:
1. Increase `min_cluster_size`
2. Reduce number of clusters
3. Use more stable distance metric
4. Add regularization to clustering

### Issue: Pareto Front Too Large

**Solutions**:
1. Use knee point selection
2. Filter dominated solutions earlier
3. Use stricter constraints
4. Rank by composite score

---

## ✅ Summary Checklist

### Implementation Complete

- ✅ **New Goals**: Rolling LL (33%), One-Step LL (33%), Economic Sharpe (34%)
- ✅ **Cluster Constraints**: 4-8 clusters, 2%-20% size distribution
- ✅ **Time Series CV**: Rolling/expanding/blocked strategies
- ✅ **Metric Calculators**: All three primary metrics implemented
- ✅ **Normalization**: Z-score, rank, robust methods
- ✅ **Pareto Optimization**: Front creation, knee selection, ranking
- ✅ **Penalties**: 6 penalty types for pathological fits
- ✅ **Robustness**: ARI validation, statistical significance tests
- ✅ **Integration**: VectorBT, hardware optimization, ML utilities
- ✅ **Documentation**: Comprehensive examples and best practices

### File Status

- ✅ **Syntax**: Valid Python 3.x
- ✅ **Line Count**: 1,430 lines (well-documented)
- ✅ **Backward Compatibility**: Old enum members preserved
- ✅ **Dependencies**: All optional with graceful fallback
- ✅ **Examples**: Complete example section included

---

## 📚 References

**Related Files**:
- `src/utils/ml_common/optimization/pareto.py` - Pareto front utilities
- `src/utils/ml_common/matrix_cross_validation.py` - Matrix CV
- `src/features_common/transforms/vectorbt_scaler.py` - Normalization
- `src/utils/hardware/` - Hardware optimization

**Related Documentation**:
- Time Series Cross-Validation Best Practices
- Financial Metrics Calculation Guide
- Pareto Multi-Objective Optimization
- Clustering Quality Assessment

---

## 🎉 Conclusion

The clustering optimization goals have been successfully updated to focus on **predictive log-likelihood and economic utility** while maintaining all requested cluster constraints. The implementation is **production-ready**, **well-documented**, and **integrated** with existing utilities.

**Key Achievements**:
- 🎯 Three new primary goals (33% each)
- 📊 Comprehensive time series CV
- 🔍 Robust metric calculation
- ⚖️ Pareto multi-objective optimization
- 🛡️ Penalties & robustness checks
- 🚀 Integration with VectorBT & hardware optimization
- 📚 Complete documentation & examples

**Ready for production use!** 🚀
