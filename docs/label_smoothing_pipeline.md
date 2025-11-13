# Label Smoothing Pipeline

## Overview

The label smoothing pipeline implements a three-stage process to create robust, well-calibrated machine learning targets from raw trading labels. This approach reduces overfitting, improves model calibration, and stabilizes training by encoding sample confidence and removing high-frequency noise while preserving causality.

## Three-Stage Pipeline

### Stage 1: Classification/Probability Smoothing
**Purpose:** Prevent overconfidence and improve calibration

- **For binary labels:** Apply epsilon smoothing: `p_smooth = (1 - ε) * p + ε * 0.5`
- **For continuous labels:** Shrink towards neutral or apply temperature scaling
- **Default parameters:** ε = 0.08 (8% smoothing), T = 1.2 (temperature)

**Effect:** Softens hard labels to prevent the model from being overconfident

### Stage 2: Uncertainty-Weighted Shrinkage
**Purpose:** Encode sample reliability

- Shrinks uncertain labels towards baseline (0 for returns, 0.5 for probabilities)
- Uses quality scores from IC-based assessment or volatility
- Formula: `label_shrunk = α * label + (1 - α) * baseline`
  - where `α = 1 / (1 + γ * σ)` (higher uncertainty → lower α → more shrinkage)
- **Default parameters:** γ = 1.0, min_α = 0.12

**Effect:** Reduces the magnitude of uncertain labels while preserving high-confidence signals

### Stage 3: Causal EMA (Temporal Smoothing)
**Purpose:** Remove high-frequency noise while preserving causality

- Applies exponential moving average per instrument/regime
- **Strictly causal** - no lookahead bias
- Formula: `EMA[t] = decay * EMA[t-1] + (1 - decay) * value[t]`
- **Default parameters:** decay = 0.95 for 15-minute bars, 0.98 for hourly/daily

**Effect:** Stabilizes labels over time, reducing sensitivity to single-bar noise

## Usage

### Basic Usage

```python
from src.training.steps.pre_training.profit_labeling.volatility_aware_labeler import (
    VolatilityAwareConfig, VolatilityAwareMultiHorizonLabeler
)

# Configure with label smoothing enabled (enabled by default)
config = VolatilityAwareConfig()
config.label_smoothing.enabled = True
config.label_smoothing.epsilon = 0.08        # Classification smoothing strength
config.label_smoothing.gamma = 1.0           # Uncertainty shrinkage sensitivity
config.label_smoothing.ema_decay = 0.95      # Temporal smoothing (higher = more history)

# Create labeler and generate labels
labeler = VolatilityAwareMultiHorizonLabeler(config)
result = labeler.generate_labels(market_data, price_column='close')

# Access smoothed labels
labels = result.labels
smoothing_metadata = result.metadata['label_smoothing']
```

### Advanced Configuration

```python
from src.training.steps.pre_training.profit_labeling.label_smoother import LabelSmoothingConfig

config.label_smoothing = LabelSmoothingConfig(
    enabled=True,

    # Enable/disable individual stages
    apply_classification_smoothing=True,
    apply_uncertainty_shrinkage=True,
    apply_causal_ema=True,

    # Classification smoothing parameters
    epsilon=0.08,              # 0.05-0.15 range typical
    temperature=1.2,           # For probability scaling (>1 = smoother)

    # Uncertainty shrinkage parameters
    gamma=1.0,                 # Sensitivity to uncertainty
    min_alpha=0.12,            # Minimum weight on label (prevent over-shrinking)
    baseline=0.0,              # Shrink towards 0 (use 0.5 for probabilities)
    uncertainty_source='quality_inverse',  # 'quality_inverse', 'quality_score', 'volatility', 'custom'

    # Causal EMA parameters
    ema_decay=0.95,            # 0.9 for fast reaction, 0.98 for slow
    ema_group_by='instrument', # Group EMA per instrument
    ema_seed_method='first',   # 'first', 'mean', or 'zero'

    # Monitoring
    store_intermediate=False,  # Store intermediate results for debugging
    validate_causality=True    # Check for lookahead bias
)
```

### Ablation Testing

To determine which components help and tune hyperparameters:

```python
from src.training.steps.pre_training.profit_labeling.ablation_test_label_smoothing import (
    run_full_ablation_suite
)

results = run_full_ablation_suite(
    data=market_data,
    labeler_config=config,
    future_returns=actual_returns,  # For IC calculation
    output_dir='ablation_results',
    save_plots=True,
    save_csv=True
)

# Results contain:
# - component_ablation: Which stages help?
# - hyperparam_sensitivity: How sensitive to parameters?
# - comparison_summary: Overall best configuration
```

Command-line interface:
```bash
python src/training/steps/pre_training/profit_labeling/ablation_test_label_smoothing.py \
    --data_path data/market_data.parquet \
    --output_dir ablation_results \
    --future_returns_col forward_return_5d
```

## Recommended Hyperparameters by Timeframe

### High-Frequency (1m - 5m bars)
```python
epsilon = 0.12         # More smoothing for noisy signals
gamma = 1.5            # Stronger shrinkage for uncertain samples
ema_decay = 0.90       # Faster reaction to regime changes
```

### Medium-Frequency (15m - 1h bars) - **DEFAULT**
```python
epsilon = 0.08         # Moderate smoothing
gamma = 1.0            # Balanced shrinkage
ema_decay = 0.95       # Standard temporal smoothing
```

### Low-Frequency (4h - daily bars)
```python
epsilon = 0.05         # Lighter smoothing (data less noisy)
gamma = 0.5            # Gentler shrinkage
ema_decay = 0.98       # Slower reaction (preserve long-term signal)
apply_causal_ema = False  # May skip EMA for daily data
```

## Evaluation Metrics

When evaluating smoothing effectiveness, track:

1. **Information Coefficient (IC):** Spearman correlation with future returns
   - Should maintain or improve vs raw labels

2. **IC Stability:** Rolling IC standard deviation
   - Lower std → more consistent predictions

3. **Calibration:** Brier score, reliability diagrams
   - Smoothing should improve probability calibration

4. **Strategy Metrics:** Sharpe ratio, max drawdown, turnover
   - Should improve risk-adjusted returns

5. **Label Statistics:**
   - `mean_absolute_change`: How much labels changed
   - `correlation_raw_final`: Preservation of signal
   - `pct_changed`: Percentage of labels modified

## Ablation Test Recommendations

Always run ablation tests to confirm each component helps:

1. **Baseline:** Raw labels (no smoothing)
2. **Classification only**
3. **Classification + Uncertainty shrinkage**
4. **Full pipeline** (all three stages)
5. **Uncertainty + EMA** (no classification)
6. **EMA only**

Compare on:
- Walk-forward IC (mean & std)
- AUC / Precision-Recall
- Brier score
- Strategy Sharpe ratio
- Parameter sensitivity

**Choose the simplest stack that gives robust improvements.**

## Best Practices

### Do's ✅

- **Always keep raw labels:** Store both `label_raw` and `label_final` for debugging
- **Run ablation tests:** Validate each component helps on your data
- **Monitor over time:** Track distribution shifts (mean, variance) of labels
- **Tune per-instrument:** Use `ema_group_by='instrument'` for cross-instrument data
- **Validate causality:** Enable `validate_causality=True` during development
- **Test parameter sensitivity:** Small changes shouldn't drastically alter results

### Don'ts ❌

- **Don't double-downweight:** If shrinking uncertain labels, decide whether to also reduce sample weights
- **Don't use centered averages:** EMA must be causal (no lookahead)
- **Don't over-smooth:** Verify correlation_raw_final > 0.70 (preserves signal)
- **Don't skip validation:** Always test on hold-out period
- **Don't use same params everywhere:** Tune for your timeframe and market

## Integration with Existing Pipeline

### Sample Weights Integration

The pipeline now correctly exports sample weights to metadata:

```python
result = labeler.generate_labels(data)
sample_weights = result.metadata['sample_weights']  # Dict[str, pd.Series]

# Use in model training
model.fit(
    X=features,
    y=result.labels,
    sample_weight=sample_weights['default']  # or specific target name
)
```

### Quality Reporting

Smoothing effects are automatically reported in the comprehensive outcome report:

```
🎨 LABEL SMOOTHING ANALYSIS
----------------------------------------
  Configuration:
    • Ablation Mode: full
    • Stages Applied:
      ✓ Classification Smoothing (ε=0.080, T=1.20)
      ✓ Uncertainty Shrinkage (γ=1.00, min_α=0.12)
      ✓ Causal EMA (decay=0.950, group_by=instrument)

  Label Statistics:
    • Raw Labels:   mean=0.0234, std=0.4567
    • Final Labels: mean=0.0198, std=0.3892

  Smoothing Impact:
    • Mean Absolute Change: 0.0453
    • Max Absolute Change:  0.2341
    • Raw-Final Correlation: 0.9234
    • % Labels Changed:     87.34%

  Interpretation:
    ✅ Moderate smoothing - good balance of stability and signal
```

## Troubleshooting

### Labels barely changing (mean_abs_change < 0.01)
- **Cause:** Smoothing too weak
- **Fix:** Increase epsilon, gamma, or lower ema_decay

### Labels drastically altered (correlation < 0.70)
- **Cause:** Over-smoothing
- **Fix:** Reduce epsilon, gamma, or increase ema_decay

### High variance in IC across periods
- **Cause:** Insufficient temporal smoothing
- **Fix:** Enable causal EMA or increase ema_decay

### Poor calibration (Brier score high)
- **Cause:** Classification smoothing disabled or too weak
- **Fix:** Enable classification smoothing, increase epsilon or temperature

### EMA not applied
- **Cause:** Missing instrument column or grouping data
- **Fix:** Ensure data has 'instrument' column or set `ema_group_by=None` for global EMA

## References

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning* - Triple Barrier Method
- Müller, K. (1997). Label smoothing for improved calibration
- This implementation extends standard label smoothing with uncertainty weighting and temporal smoothing

## Files Modified

- `src/training/steps/pre_training/profit_labeling/label_smoother.py` - **NEW:** Core smoothing implementation
- `src/training/steps/pre_training/profit_labeling/volatility_aware_labeler.py` - **MODIFIED:** Integration into labeler
- `src/training/steps/pre_training/profit_labeling/ablation_test_label_smoothing.py` - **NEW:** Ablation testing utilities

## Changelog

### 2025-11-13: Initial Implementation
- Implemented three-stage label smoothing pipeline
- Added `LabelSmoother` and `LabelSmoothingConfig` classes
- Integrated smoothing into `VolatilityAwareMultiHorizonLabeler`
- Fixed sample weights bug (now properly added to metadata)
- Added comprehensive quality reporting for smoothing effects
- Created ablation testing utilities
- Added documentation and usage examples
