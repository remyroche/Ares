# Meta-Labeling Guide

## Overview

Meta-labeling is an alternative to the triple barrier method for generating trading labels. Instead of labeling based on price barriers, meta-labeling predicts whether **primary signals** (from technical indicators) will be profitable.

## Key Concepts

### What is Meta-Labeling?

Meta-labeling is a two-stage process:

1. **Primary Model**: Generates trading signals from technical indicators (RSI, MA crossovers, momentum)
2. **Meta-Model**: Learns which primary signals to act on (filters signals by predicted profitability)

### Advantages Over Triple Barrier

| Feature | Triple Barrier | Meta-Labeling |
|---------|---------------|---------------|
| Label Source | Price movements | Signal profitability |
| Signal Quality | All signals treated equally | Filters high-quality signals |
| Model Focus | Predicts direction | Predicts signal success |
| Interpretability | Price-based thresholds | Signal-based decisions |
| Leakage Risk | Moderate | Lower (with proper purging) |

### When to Use Meta-Labeling

Use meta-labeling when:
- You have strong primary signals but want to filter them
- You want to combine multiple indicators intelligently
- You need to adapt to changing market conditions
- You want to focus on signal quality over quantity

Use triple barrier when:
- You want pure price prediction
- You need symmetrical long/short labels
- You want simpler, more interpretable labels

## Implementation Details

### 1. Primary Signal Generation

Primary signals are generated from technical indicators:

```python
from src.training.steps.pre_training.feature_generation_meta_labeling_step import generate_primary_signals

# Generate signals from OHLCV data
signals = generate_primary_signals(
    df,
    rsi_period=14,           # RSI period
    sma_fast=10,             # Fast moving average
    sma_slow=30,             # Slow moving average
    momentum_period=10,      # Momentum lookback
    rsi_oversold=30,         # RSI oversold threshold
    rsi_overbought=70,       # RSI overbought threshold
    momentum_threshold=0.005 # Momentum threshold
)

# Signals DataFrame contains:
# - rsi: {-1, 0, 1} (bearish, neutral, bullish)
# - ma: {-1, 0, 1} (MA crossover signal)
# - mom: {-1, 0, 1} (momentum signal)
# - consensus: {-1, 0, 1} (majority vote)
```

**CRITICAL**: Primary signals must be **fixed** and not optimized during cross-validation to avoid leakage.

### 2. Meta-Label Creation

Meta-labels are binary: 1 = profitable signal, 0 = unprofitable signal.

```python
from src.training.steps.pre_training.feature_generation_meta_labeling_step import create_meta_labels

meta_labels = create_meta_labels(
    df,
    signals,
    profit_threshold=0.015,  # 1.5% profit target
    stop_threshold=0.010,    # 1.0% stop loss
    horizon=16               # Maximum bars to look ahead
)

# For each signal:
# - Look ahead up to 'horizon' bars
# - Check if profit_threshold is hit before stop_threshold
# - Label = 1 if profitable, 0 if stopped out
# - Label = NaN if no signal at that bar
```

### 3. Meta-Feature Creation

Meta-features help the model decide which signals to act on:

```python
from src.training.steps.pre_training.feature_generation_meta_labeling_step import create_meta_features

features = create_meta_features(
    df,
    signals,
    volume_available=True
)

# Features include:
# - Signal strength (confluence of indicators)
# - Volatility measures
# - Trend indicators
# - Volume patterns
# - Price momentum
# - Recent high/low distances
```

### 4. Time-Series CV with Purging

**CRITICAL**: Must purge training samples that create lookahead bias.

```python
from src.training.steps.pre_training.feature_generation_meta_labeling_step import purge_training_idxs
from sklearn.model_selection import TimeSeriesSplit

horizon = 16
outer_cv = TimeSeriesSplit(n_splits=5)

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(df)):
    # Purge training samples whose prediction horizon overlaps validation
    train_idx_purged = purge_training_idxs(
        train_idx,
        test_idx[0],        # Validation start
        test_idx[-1] + 1,   # Validation end (exclusive)
        horizon=horizon
    )

    # Now safe to train/validate without lookahead bias
    # ...
```

### 5. Label-to-Target Translation

Downstream optimization steps expect continuous targets, not binary labels:

```python
from src.training.steps.pre_training.feature_generation_meta_labeling_step import translate_metalabels_to_targets

# After training meta-model, get probabilities
probabilities = model.predict_proba(features)[:, 1]

# Translate to targets
target_long, target_short = translate_metalabels_to_targets(
    meta_labels,
    signals,
    probabilities,
    threshold=0.6  # Probability threshold for generating targets
)

# Result:
# - target_long: Positive values for long opportunities
# - target_short: Positive values for short opportunities
# - Both are compatible with downstream optimization steps
```

## Usage Examples

### Basic Usage

```python
import asyncio
from src.training.steps.pre_training.feature_generation_meta_labeling_step import FeatureGenerationMetaLabelingStep

# Initialize step
step = FeatureGenerationMetaLabelingStep()

# Configure
config = {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'timeframe': '15m',
    'profit_threshold': 0.015,  # 1.5%
    'stop_threshold': 0.010,    # 1.0%
    'horizon': 16,              # ~4 hours for 15m bars
    'data_dir': 'historical_data'
}

# Execute
result = await step.execute(config)

if result['success']:
    print(f"✅ Meta-labeling completed!")
    print(f"   Samples: {result['metrics']['n_samples']}")
    print(f"   Labeled: {result['metrics']['n_labeled']}")
    print(f"   Positive rate: {result['metrics']['positive_rate']:.1%}")
    print(f"   CV AUC: {result['metrics']['cv_mean_auc']:.3f}")
    print(f"   CV Precision: {result['metrics']['cv_mean_precision']:.3f}")
else:
    print(f"❌ Error: {result['error']}")
```

### Integration with Pipeline

Replace the labeling step in your pipeline:

```python
# OLD: Triple barrier labeling
pipeline_steps = [
    'feature_generation_data_validation_step',
    'feature_generation_labeling_integration_step',  # <-- Triple barrier
    'feature_generation_feature_generation_step',
    # ...
]

# NEW: Meta-labeling
pipeline_steps = [
    'feature_generation_data_validation_step',
    'feature_generation_meta_labeling_step',  # <-- Meta-labeling
    'feature_generation_feature_generation_step',
    # ...
]
```

### Advanced: Custom Primary Signals

You can customize primary signals for your strategy:

```python
import pandas as pd
from src.training.steps.pre_training.feature_generation_meta_labeling_step import (
    create_meta_labels, create_meta_features
)

# Define custom primary signals
def my_custom_signals(df):
    signals = pd.DataFrame(index=df.index)

    # Example: MACD-based signals
    ema_fast = df['close'].ewm(span=12).mean()
    ema_slow = df['close'].ewm(span=26).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=9).mean()

    signals['macd'] = 0
    signals.loc[macd > signal_line, 'macd'] = 1
    signals.loc[macd < signal_line, 'macd'] = -1

    # Example: Bollinger Band signals
    sma = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    upper = sma + 2 * std
    lower = sma - 2 * std

    signals['bb'] = 0
    signals.loc[df['close'] < lower, 'bb'] = 1   # Oversold
    signals.loc[df['close'] > upper, 'bb'] = -1  # Overbought

    # Consensus
    signals['consensus'] = signals[['macd', 'bb']].sum(axis=1).apply(np.sign)

    return signals

# Use custom signals
my_signals = my_custom_signals(market_data)
meta_labels = create_meta_labels(market_data, my_signals)
meta_features = create_meta_features(market_data, my_signals)
```

## Avoiding Common Pitfalls

### 1. Leakage from Optimizing Primary Signals

**WRONG**: Optimizing RSI period in inner CV and recomputing labels:

```python
# ❌ BAD: Recomputing labels with optimized parameters
for rsi_period in [7, 14, 21]:
    signals = generate_primary_signals(df, rsi_period=rsi_period)
    meta_labels = create_meta_labels(df, signals)  # LEAKAGE!
    # Train model...
```

**RIGHT**: Fix primary signals, only optimize meta-features:

```python
# ✅ GOOD: Fixed primary signals
signals = generate_primary_signals(df, rsi_period=14)  # FIXED
meta_labels = create_meta_labels(df, signals)

# Now you can optimize meta-features without leakage
for volatility_window in [5, 10, 20]:
    features = create_custom_features(df, signals, vol_window=volatility_window)
    # Train model...
```

### 2. Not Purging Training Data

**WRONG**: Using raw CV splits without purging:

```python
# ❌ BAD: Lookahead bias
for train_idx, val_idx in cv.split(df):
    X_train = features.iloc[train_idx]  # LEAKAGE!
    # Training sample at position i predicts i+horizon
    # which may overlap with validation
```

**RIGHT**: Purge overlapping samples:

```python
# ✅ GOOD: Purged training set
for train_idx, val_idx in cv.split(df):
    train_idx_purged = purge_training_idxs(
        train_idx, val_idx[0], val_idx[-1] + 1, horizon
    )
    X_train = features.iloc[train_idx_purged]
```

### 3. Insufficient Labeled Samples

**Issue**: Too few signals result in sparse labels:

```python
# Check label density
labeled_rate = (~meta_labels.isna()).sum() / len(meta_labels)
if labeled_rate < 0.1:  # Less than 10% labeled
    print("⚠️ Warning: Very sparse labels")
```

**Solutions**:
- Adjust signal thresholds (less strict)
- Use shorter horizon
- Combine more indicators for consensus
- Use lower profit/stop thresholds

### 4. Class Imbalance

Meta-labels are often imbalanced (more losses than wins):

```python
# Check imbalance
positive_rate = (meta_labels == 1.0).sum() / (~meta_labels.isna()).sum()
print(f"Positive rate: {positive_rate:.1%}")

# If very imbalanced (<20% or >80%), consider:
# - Adjusting profit/stop thresholds
# - Using class weights in model
# - Using stratified sampling
# - Using precision/recall instead of accuracy
```

## Performance Metrics

### Model Evaluation

```python
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# AUC: Overall ranking ability
auc = roc_auc_score(y_true, y_pred_proba)

# Precision: When model says "act", how often is it right?
precision = precision_score(y_true, y_pred)

# Recall: Of all good signals, how many did we catch?
recall = recall_score(y_true, y_pred)

# F1: Harmonic mean of precision and recall
f1 = f1_score(y_true, y_pred)

print(f"AUC: {auc:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1: {f1:.3f}")
```

### Trading Performance

After deployment, evaluate actual trading performance:

```python
# Signal quality metrics
win_rate = n_wins / n_trades
avg_win = total_profit / n_wins if n_wins > 0 else 0
avg_loss = total_loss / n_losses if n_losses > 0 else 0
profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')

# Filter effectiveness
signals_before_filter = n_primary_signals
signals_after_filter = n_meta_signals
filter_rate = 1 - (signals_after_filter / signals_before_filter)

print(f"Filter rate: {filter_rate:.1%}")
print(f"Win rate: {win_rate:.1%}")
print(f"Profit factor: {profit_factor:.2f}")
```

## Configuration Recommendations

### By Timeframe

| Timeframe | Horizon | Profit | Stop | Notes |
|-----------|---------|--------|------|-------|
| 1m        | 5-10    | 0.005  | 0.003| Very short term |
| 5m        | 10-20   | 0.008  | 0.005| Short term |
| 15m       | 16-32   | 0.015  | 0.010| Medium term (default) |
| 1h        | 8-16    | 0.020  | 0.015| Longer term |
| 4h        | 4-8     | 0.030  | 0.020| Swing trading |

### By Asset Volatility

| Volatility | Profit | Stop | Horizon |
|------------|--------|------|---------|
| Low (Forex)| 0.003  | 0.002| 20-40   |
| Medium (ETH)| 0.015 | 0.010| 16-32   |
| High (Altcoins)| 0.025| 0.015| 12-24  |

### By Strategy Type

**Scalping** (frequent, small profits):
- Horizon: 5-10
- Profit: 0.005-0.010
- Stop: 0.003-0.007
- Focus: High precision

**Swing Trading** (less frequent, larger profits):
- Horizon: 20-40
- Profit: 0.025-0.050
- Stop: 0.015-0.030
- Focus: High recall

**Balanced**:
- Horizon: 16-24
- Profit: 0.015-0.020
- Stop: 0.010-0.015
- Focus: F1 score

## Debugging and Monitoring

### Check Label Quality

```python
# After creating meta-labels
print(f"Total samples: {len(meta_labels)}")
print(f"Labeled samples: {(~meta_labels.isna()).sum()}")
print(f"Label density: {(~meta_labels.isna()).sum() / len(meta_labels):.1%}")
print(f"Positive rate: {(meta_labels == 1.0).sum() / (~meta_labels.isna()).sum():.1%}")

# Check signal distribution
print("\nSignal distribution:")
print(signals['consensus'].value_counts())

# Check feature quality
print("\nFeature statistics:")
print(features.describe())
```

### Visualize Meta-Labels

```python
import matplotlib.pyplot as plt

# Plot labels over time
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

# Price and signals
ax1.plot(df.index, df['close'], label='Close', alpha=0.7)
long_signals = signals['consensus'] > 0
short_signals = signals['consensus'] < 0
ax1.scatter(df.index[long_signals], df['close'][long_signals],
           color='green', marker='^', label='Long Signal', s=100)
ax1.scatter(df.index[short_signals], df['close'][short_signals],
           color='red', marker='v', label='Short Signal', s=100)
ax1.legend()
ax1.set_title('Price and Primary Signals')

# Meta-labels
profitable = meta_labels == 1.0
unprofitable = meta_labels == 0.0
ax2.scatter(df.index[profitable], [1]*profitable.sum(),
           color='green', marker='|', label='Profitable', s=100)
ax2.scatter(df.index[unprofitable], [0]*unprofitable.sum(),
           color='red', marker='|', label='Unprofitable', s=100)
ax2.set_ylim(-0.5, 1.5)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['Unprofitable', 'Profitable'])
ax2.legend()
ax2.set_title('Meta-Labels')

plt.tight_layout()
plt.show()
```

## Further Reading

### Academic References

1. **"The 10 Reasons Most Machine Learning Funds Fail"** - Marcos López de Prado
   - Discusses leakage and purging in financial ML

2. **"Advances in Financial Machine Learning"** - Marcos López de Prado
   - Chapter on meta-labeling and triple barrier method

3. **"Machine Learning for Asset Managers"** - Marcos López de Prado
   - Advanced techniques for financial ML

### Related Topics

- **Bet Sizing**: Use meta-model probabilities for position sizing
- **Ensemble Methods**: Combine multiple meta-models
- **Online Learning**: Update meta-model in real-time
- **Multi-Asset Meta-Labeling**: Share meta-model across assets

## Support

For issues or questions:
1. Check test file: `test_meta_labeling_step.py`
2. Review code: `src/training/steps/pre_training/feature_generation_meta_labeling_step.py`
3. Compare with triple barrier: `feature_generation_labeling_integration_step.py`
