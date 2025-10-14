# Feature Acceleration and Window Dilation Guide

This guide explains how to use the feature acceleration and window dilation system to expose turning points and capture different regime/trend complements.

## Overview

The acceleration and dilation system compares features to their variants to identify additional signal that the base features might miss:

1. **Acceleration Features**: Δ over k or 2nd difference to expose turning points
2. **Window Dilation**: 3× lookback to capture slower regime/trend complements

## Why Use Acceleration and Dilation?

### Acceleration Features
- **Expose turning points** that base features miss
- **Detect inflection points** (reversals, exhaustion)
- **Amplify signal** when base shows persistence (autocorr > 0.2)

**Risks:**
- Amplifies noise and microstructure artifacts
- Requires careful handling of bounded features (e.g., RSI-50)

### Window Dilation Features
- **Capture different regimes** and trend complements
- **Test robustness** across different time horizons
- **Reduce turnover** by using longer-term signals

**When it's worth it:**
- Performance cliffs when sweeping window sizes
- Base horizon aligns with trading horizon
- Need to test robustness across time scales

## How to Construct Features

### Acceleration Features

```python
# 1. Center first if bounded (e.g., RSI-50)
if feature.min() >= 0 and feature.max() <= 100:
    centered = feature - 50
else:
    centered = feature

# 2. Winsorize to reduce noise
winsorized = feature.clip(
    lower=feature.quantile(0.01),
    upper=feature.quantile(0.99)
)

# 3. Calculate acceleration: accel_k = base_t - base_{t-k}
acceleration = winsorized - winsorized.shift(lag)

# 4. Optional re-scaling
acceleration = (acceleration - acceleration.mean()) / acceleration.std()
```

### Window Dilation Features

```python
# 1. Extract original window size from feature name
original_window = extract_window_size(feature_name)  # e.g., 20

# 2. Calculate new window size
new_window = int(original_window * dilation_factor)  # e.g., 60 for 3× dilation

# 3. Generate dilated feature based on type
if 'ma_' in feature_name:
    dilated = feature.rolling(new_window).mean()
elif 'ema_' in feature_name:
    span = int(original_window * dilation_factor)
    dilated = feature.ewm(span=span).mean()
elif 'std_' in feature_name:
    dilated = feature.rolling(new_window).std()
# ... etc.
```

## Acceptance Gates

### Signal Gates
- **Acceleration**: MI ≥ base × (1 - ε) and conditional MI ≥ threshold
- **Dilation**: 3× window beats base by ΔFQS ≥ 0.05 or offers orthogonal contribution

### Stability Gates
- **Rank consistency**: Rank SD across folds ≤ base + 25%
- **Scaling sensitivity**: Stable under different scaling methods

### Uniqueness Gates
- **Correlation**: |ρ|(variant, base) ≤ 0.9
- **Conditional MI**: Variant adds incremental value over base

### Practicality Gates
- **Latency**: Equal or better than base
- **Missingness**: Not more missing than base
- **Model improvement**: MSE improvement ≥ 1-2% when both included

## Usage Example

```python
from feature_comparison.feature_acceleration_dilation import FeatureAccelerationDilation

# Initialize system
system = FeatureAccelerationDilation(
    acceleration_lags=[1, 3],           # Lags for acceleration
    dilation_factors=[2.0, 3.0],       # Window dilation factors
    mi_threshold=0.6,                   # MI threshold for signal
    correlation_threshold=0.9,          # Correlation threshold for uniqueness
    conditional_mi_threshold=0.6,       # Conditional MI threshold
    enable_matrix_ops=True
)

# Generate acceleration features
acceleration_features = system.generate_acceleration_features(X)

# Generate dilation features
dilation_features = system.generate_dilation_features(X)

# Evaluate features
accel_results = system.evaluate_acceleration_features(X, y, acceleration_features, base_features)
dil_results = system.evaluate_dilation_features(X, y, dilation_features, base_features)

# Run complete evaluation
complete_results = system.run_complete_evaluation(X, y, base_features)
```

## Feature Selection Strategy

### 1. Pre-screen Base Features
- Run Phase A/B of pre-screening pipeline
- If base feature fails → don't generate variants

### 2. Build Small Variants
- **Acceleration**: `accel_k ∈ {1, 3}`
- **Dilation**: `window_dilation ∈ {2×, 3×}`

### 3. Pilot Slice Tests
- MI, permutation importance (shallow model)
- Correlation vs base feature

### 4. Joint Model Test
- Add variant on top of base
- Check Δloss / ΔAUC and conditional MI

### 5. Stability Check
- Rank SD across folds & regimes
- Scaling sensitivity

### 6. Decision
Keep only if:
- (a) Incremental value
- (b) Stable across conditions
- (c) Not redundant with base
- (d) Cost-effective

## Practical Checklist

### Fast to Run
1. **Pre-screen base** (Phase A/B)
2. **Build small variants** (accel_k∈{1,3}, window_dilation∈{2×,3×})
3. **Pilot slice tests** (MI, perm-imp, |ρ| vs base)
4. **Joint-model test** (add variant on top of base)
5. **Stability check** (rank SD, scaling sensitivity)

### Decision Criteria
- **Signal**: Accel's MI/perm-imp ≥ base × (1 - ε) and adds conditional MI
- **Stability**: Rank SD across folds ≤ base + 25%
- **Uniqueness**: |ρ|(accel, base) ≤ 0.9 or accel improves model loss ≥ 1-2%
- **Practicality**: Latency equal; not more missingness

### Keep Both?
Only if variant passes and remains incremental when base is already in the model.

## Configuration Parameters

### Acceleration Parameters
- `acceleration_lags`: List of lags for acceleration calculation (default: [1, 3])
- `mi_threshold`: MI threshold for signal acceptance (default: 0.6)
- `conditional_mi_threshold`: Conditional MI threshold for incremental value (default: 0.6)

### Dilation Parameters
- `dilation_factors`: Window dilation factors (default: [2.0, 3.0])
- `fqs_improvement_threshold`: FQS improvement threshold for acceptance (default: 0.05)

### General Parameters
- `correlation_threshold`: Correlation threshold for uniqueness (default: 0.9)
- `rank_std_threshold`: Rank standard deviation threshold for stability (default: 0.25)
- `enable_matrix_ops`: Whether to enable matrix operations (default: True)

## Output Structure

The system returns comprehensive evaluation results:

```python
{
    'acceleration_features': {
        'lag_1': DataFrame,  # Features with lag=1 acceleration
        'lag_3': DataFrame   # Features with lag=3 acceleration
    },
    'dilation_features': {
        'factor_2.0': DataFrame,  # Features with 2× dilation
        'factor_3.0': DataFrame   # Features with 3× dilation
    },
    'acceleration_evaluation': {
        'accepted_features': List[str],
        'rejected_features': List[str],
        'acceleration_evaluations': {
            'lag_1': {feature: evaluation_metrics},
            'lag_3': {feature: evaluation_metrics}
        }
    },
    'dilation_evaluation': {
        'accepted_features': List[str],
        'rejected_features': List[str],
        'dilation_evaluations': {
            'factor_2.0': {feature: evaluation_metrics},
            'factor_3.0': {feature: evaluation_metrics}
        }
    },
    'summary': {
        'total_acceleration_features': int,
        'total_dilation_features': int,
        'accepted_acceleration': int,
        'accepted_dilation': int,
        'rejected_acceleration': int,
        'rejected_dilation': int
    }
}
```

## Best Practices

1. **Start with high-quality base features** that show persistence
2. **Use small sets of variants** to avoid overfitting
3. **Test stability across different market regimes**
4. **Monitor computational cost** of variant generation
5. **Validate incremental value** in joint models
6. **Consider trading horizon** when selecting dilation factors
7. **Remove intraday seasonality** before calculating acceleration
8. **Use consistent normalization** across base and variant features

## Common Pitfalls

1. **Noise amplification**: Acceleration can amplify microstructure noise
2. **Overfitting**: Too many variants can lead to overfitting
3. **Redundancy**: Variants that are highly correlated with base features
4. **Instability**: Variants that are sensitive to scaling or regime changes
5. **Computational cost**: Generating too many variants can be expensive
6. **Lookahead bias**: Ensure variants don't use future information

## Integration with Pre-screening Pipeline

The acceleration and dilation system integrates seamlessly with the pre-screening pipeline:

```python
from feature_comparison.pre_screening_pipeline import PreScreeningPipeline
from feature_comparison.feature_acceleration_dilation import FeatureAccelerationDilation

# Run pre-screening first
pipeline = PreScreeningPipeline()
prescreening_results = pipeline.run_pre_screening(X, y)

# Get selected base features
base_features = prescreening_results['final_features']

# Generate and evaluate variants
accel_dil_system = FeatureAccelerationDilation()
variant_results = accel_dil_system.run_complete_evaluation(X, y, base_features)

# Combine results
final_features = base_features + variant_results['acceleration_evaluation']['accepted_features'] + variant_results['dilation_evaluation']['accepted_features']
```

This approach ensures that only high-quality base features are considered for variant generation, and only incremental variants are added to the final feature set.