# Step 3: Automatic HMM Parameter Optimization

## Overview

Step 3 now includes **automatic parameter optimization** that runs before HMM regime discovery to find the best parameters for achieving 15-20 distinct, internally coherent regimes.

## 🎯 Key Features

### Automatic Optimization
- ✅ **Automatic execution** - runs automatically when needed
- ✅ **Smart caching** - reuses results unless outdated or forced
- ✅ **Configurable** - easily adjustable via configuration file
- ✅ **Comprehensive reporting** - generates detailed optimization reports
- ✅ **Visualization** - creates optimization progress charts
- ✅ **Vectorized operations** - uses matrix operations for better performance
- ✅ **Force rerun** - always runs optimization with `--force` flag

### Optimization Goals
- 🎯 **15-20 final regimes** - targets optimal regime count
- 🔄 **Regime differentiation** - ensures regimes capture distinct market conditions
- 📊 **Internal coherence** - maintains consistency within each regime
- ⚖️ **Regime balance** - prevents skewed regime sizes

## 🔧 Configuration

### Automatic Optimization Settings
```json
{
  "automatic_optimization": {
    "enabled": true,           // Enable/disable automatic optimization
    "max_trials": 50,          // Maximum optimization trials
    "timeout_minutes": 30,     // Maximum optimization time
    "force_rerun_days": 1,     // Re-run if results older than X days (default: 1 day)
    "vectorized_optimization": true  // Use vectorized operations for performance
  }
}
```

### Parameter Ranges
The optimizer searches through:

**HMM Parameters:**
- `n_components`: 2-10 states
- `covariance_type`: full, tied, diag, spherical
- `n_iter`: 100-300 iterations
- `tol`: 1e-4 to 1e-2 tolerance
- `reg_covar`: 1e-6 to 1e-3 regularization

**Clustering Parameters:**
- `clustering_method`: kmeans, gaussian_mixture
- `n_clusters`: 3-15 initial clusters

**Regime Merging Parameters:**
- `initial_clusters`: 25-50 starting clusters
- `target_regimes`: 15-20 final regimes
- `merging_method`: hierarchical, kmeans, dbscan, spectral
- `similarity_threshold`: 0.3-0.8 similarity for merging
- `coherence_threshold`: 0.6-0.9 internal consistency
- `differentiation_threshold`: 0.4-0.8 regime separation

## 🚀 Usage

### Automatic Execution
Step 3 **ALWAYS** runs optimization:
1. **Every execution** - optimization runs on every Step 3 execution
2. **Fresh parameters** - ensures optimal parameters for current data
3. **Vectorized operations** - uses matrix operations for better performance
4. **No caching** - always produces fresh optimization results

### Manual Control
```python
# Disable automatic optimization
config = {
    "automatic_optimization": {
        "enabled": False
    }
}

# Adjust optimization parameters
config = {
    "automatic_optimization": {
        "max_trials": 100,        # More thorough search
        "timeout_minutes": 60,    # Longer timeout
        "force_rerun_days": 1     # Re-run daily (default)
    }
}
```

## 📊 Output Files

### Optimization Results
- `{exchange}_{symbol}_{timeframe}_optimization_results.json` - Best parameters and scores
- `{exchange}_{symbol}_{timeframe}_optimization_report.md` - Detailed analysis report
- `optimization_visualizations/` - Progress charts and parameter importance

### Example Output Structure
```json
{
  "best_score": 0.8234,
  "best_params": {
    "n_components": 6,
    "covariance_type": "full",
    "target_regimes": 18,
    "merging_method": "hierarchical",
    "coherence_threshold": 0.75,
    "differentiation_threshold": 0.65
  },
  "optimization_history": [...],
  "timestamp": "2024-01-15T10:30:00",
  "symbol": "ETHUSDT",
  "exchange": "BINANCE",
  "timeframe": "1m"
}
```

## 🔍 Evaluation Criteria

The optimizer evaluates regimes using:

1. **Regime Differentiation** (40% weight)
   - How well regimes differ from each other
   - Based on market condition characteristics

2. **Internal Coherence** (30% weight)
   - Consistency within each regime
   - Lower coefficient of variation = better

3. **Regime Balance** (15% weight)
   - Balanced regime sizes
   - Prevents skewed distributions

4. **Target Count Penalty** (15% weight)
   - Penalty for not achieving 15-20 regimes
   - Additional penalty outside 15-20 range

## 📈 Optimization Process

### Step 1: Initial Clustering
- Generate 25-50 initial clusters using HMM + clustering
- Apply feature engineering (assumes Step 2 completed)

### Step 2: Regime Merging
- Merge similar clusters to achieve 15-20 final regimes
- Use hierarchical, kmeans, dbscan, or spectral methods
- Apply similarity and coherence thresholds

### Step 3: Evaluation
- Calculate differentiation, coherence, balance scores
- Apply target count penalties
- Generate weighted overall score

### Step 4: Parameter Selection
- Optuna selects best parameters based on scores
- Continues until max trials or timeout reached

## ⚡ Performance Optimizations

### Vectorized Operations
- **Matrix-based calculations** for regime differentiation
- **Broadcasting operations** for pairwise comparisons
- **Vectorized statistics** for coherence and balance scores
- **Pre-processed data structures** for faster access

### Always-On Optimization
- **Every execution**: Optimization runs on every Step 3 execution
- **Fresh parameters**: Ensures optimal parameters for current data
- **No caching**: Always produces fresh optimization results
- **Consistent quality**: Maintains high-quality regime discovery

## 🎛️ Integration with Step 3

### Automatic Integration
```python
# Step 3 automatically:
1. ALWAYS runs optimization
2. Applies optimized parameters
3. Performs HMM regime discovery
4. Generates comprehensive reports
```

### Parameter Application
Optimized parameters are automatically applied to:
- HMM model configuration
- Clustering algorithm settings
- Regime merging parameters
- Evaluation thresholds

## 📋 Monitoring and Logging

### Optimization Logs
```
🔧 No existing optimization results found - will run optimization
🚀 Starting automatic parameter optimization...
📊 Optimization data: 10000 samples, 25 features
📈 Market conditions: 8
✅ Parameter optimization completed successfully
🔧 Applying optimized parameters...
✅ Optimized parameters applied successfully
```

### Progress Tracking
- Trial number and score
- Parameter combinations tested
- Time elapsed
- Best score improvement

## 🔧 Troubleshooting

### Common Issues

**Optimization takes too long:**
```json
{
  "automatic_optimization": {
    "max_trials": 25,        // Reduce trials
    "timeout_minutes": 15    // Reduce timeout
  }
}
```

**Poor optimization results:**
- Check feature quality from Step 2
- Verify market condition columns
- Increase max_trials for better search

**Memory issues:**
- Reduce data sample size
- Use smaller parameter ranges
- Enable memory monitoring

### Debug Mode
```python
# Enable detailed logging
import logging
logging.getLogger("HMMRegimeOptimizer").setLevel(logging.DEBUG)
```

## 🎯 Best Practices

1. **Feature Quality First** - Ensure Step 2 produces good features
2. **Reasonable Timeouts** - Balance thoroughness vs. speed
3. **Monitor Results** - Check optimization reports regularly
4. **Validate Regimes** - Use cluster validation tools after optimization
5. **Update Periodically** - Re-run optimization when market conditions change

## 📚 Related Files

- `optimize_hmm_regime_parameters.py` - Main optimization engine
- `step03_optimization_config.json` - Configuration file
- `test_hmm_cluster_relevance.py` - Cluster validation tools
- `HMM_OPTIMIZATION_README.md` - Detailed optimization documentation