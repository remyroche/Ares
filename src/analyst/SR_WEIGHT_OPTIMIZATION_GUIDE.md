# S/R Weight Optimization Guide

## Overview

The S/R classifier now includes an advanced optimization system that automatically finds the best weight ratios for the 5 relevance scoring factors.

## Optimization Methods

### 1. **Optuna (Recommended)**
- Bayesian optimization with pruning
- Efficiently explores weight space
- Typically converges in 100-200 trials

### 2. **SciPy Differential Evolution**
- Global optimization algorithm
- Good for finding global optimum
- Slower but thorough

### 3. **Grid Search**
- Exhaustive search over grid
- Guaranteed to find best in grid
- Computationally expensive

## Weight Factors Being Optimized

| Factor | Default | Bounds | Description |
|--------|---------|--------|-------------|
| return_magnitude | 0.30 | [0.1, 0.5] | Size of price reversals at S/R |
| touch_count | 0.20 | [0.05, 0.3] | Number of times level tested |
| recency | 0.20 | [0.05, 0.3] | How recent the tests are |
| volume_confirmation | 0.15 | [0.05, 0.25] | Volume spikes at reversals |
| success_rate | 0.15 | [0.1, 0.4] | Historical accuracy of level |

## Optimization Process

### Initial Optimization
```python
# The optimizer will:
1. Generate S/R levels with current weights
2. Score each level using different weight combinations
3. Evaluate performance using chosen metric
4. Find optimal weight ratios
```

### Performance Metrics

**Sharpe Ratio** (Recommended):
- Risk-adjusted returns from trading S/R levels
- Accounts for both profit and consistency

**Accuracy**:
- How often S/R levels hold
- Simple but doesn't account for profit

**Profit**:
- Raw returns from trading signals
- Doesn't account for risk

## Dynamic Weight Adjustment

Weights are dynamically adjusted based on:

### Market Regime
```python
# Trending Market
- Recency ↑ 20%        # Recent levels matter more
- Success Rate ↑ 30%   # Reliability crucial
- Touch Count ↓ 20%    # Fewer touches in trends

# Ranging Market
- Touch Count ↑ 40%    # Many touches expected
- Return Magnitude ↑ 20%  # Strong reversals
- Recency ↓ 20%       # Old levels still valid

# Volatile Market
- Volume Confirmation ↑ 50%  # Volume crucial
- Return Magnitude ↑ 30%     # Larger moves
- Touch Count ↓ 30%         # Levels break often
```

### Volatility Percentile
```python
# High Volatility (>80th percentile)
- Volume Confirmation ↑ 20%
- Return Magnitude ↑ 10%

# Low Volatility (<20th percentile)
- Touch Count ↑ 20%
- Success Rate ↑ 10%
```

## Example Optimization Results

### Before Optimization (Default Weights)
```yaml
weights:
  return_magnitude: 0.30
  touch_count: 0.20
  recency: 0.20
  volume_confirmation: 0.15
  success_rate: 0.15
performance: 0.85 Sharpe
```

### After Optimization
```yaml
weights:
  return_magnitude: 0.35    # ↑ Increased importance
  touch_count: 0.12        # ↓ Reduced
  recency: 0.18           # ↓ Slightly reduced
  volume_confirmation: 0.22  # ↑ Increased
  success_rate: 0.13       # ↓ Slightly reduced
performance: 1.24 Sharpe   # 46% improvement
```

## Usage

### Automatic Optimization
```python
# Classifier automatically optimizes when:
1. First run (no saved weights)
2. Every 24 hours (configurable)
3. When manually triggered
```

### Manual Optimization
```python
# Force optimization
classifier = UnifiedRegimeClassifierSROptimized(config)
await classifier._optimize_weights_async(market_data)

# Get optimization report
report = classifier.get_optimization_report()
print(f"Best weights: {report['optimized_weights']}")
print(f"Performance improvement: {report['performance_improvement']:.1%}")
```

## Configuration

```yaml
# Enable/disable optimization
enable_weight_optimization: true
enable_dynamic_adjustment: true

# Optimization settings
optimization_method: "optuna"
n_trials: 100
validation_metric: "sharpe_ratio"

# Frequency
optimization_frequency_hours: 24
```

## Monitoring

The system tracks:
- Optimization history
- Performance improvements
- Weight evolution over time
- Market regime changes

## Best Practices

1. **Initial Period**: Let optimizer run for at least 1000 data points
2. **Re-optimization**: Daily is usually sufficient
3. **Validation**: Monitor out-of-sample performance
4. **Regime Changes**: Weights auto-adjust to market conditions

## Example Output

```python
{
    'weight_optimization': {
        'weights_used': {
            'return_magnitude': 0.342,
            'touch_count': 0.118,
            'recency': 0.225,
            'volume_confirmation': 0.187,
            'success_rate': 0.128
        },
        'market_regime': 'trending',
        'volatility_percentile': 0.72,
        'last_optimization': '2024-01-15T10:30:00',
        'optimization_enabled': true
    }
}
```

The optimization system ensures that S/R relevance scoring continuously adapts to market conditions, maximizing the value of the detected levels.