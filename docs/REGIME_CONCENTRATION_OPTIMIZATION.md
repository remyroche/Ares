# Regime Concentration Optimization Guide

## Overview

This guide explains how to adjust the HMM regime discovery to achieve 70-80% concentration in the top 20 regimes. The enhanced regime merging functionality provides multiple strategies to consolidate low-frequency regimes into higher-frequency ones.

## Key Changes Made

### 1. Enhanced Configuration Parameters

The `REGIME_MERGING_CONFIG` has been updated with new parameters:

```python
REGIME_MERGING_CONFIG = {
    "min_frequency": 0.003,          # 0.3% minimum frequency (more aggressive)
    "similarity_threshold": 0.75,     # 75% similarity threshold (more aggressive)
    "max_regimes": 20,               # Target 20 regimes for 70-80% concentration
    "enable_merging": True,          # Enable regime merging
    "merge_strategy": "similarity",   # "similarity" or "frequency"
    "target_top_20_concentration": 0.75,  # Target 75% concentration in top 20
    "aggressive_merging": True,      # Enable aggressive merging strategies
}
```

### 2. Multi-Strategy Merging Algorithm

The enhanced `merge_similar_regimes` function implements three merging strategies:

#### Strategy 1: Basic Similarity Merging
- Merges low-frequency regimes into most similar high-frequency regimes
- Uses configurable similarity threshold (default: 75%)

#### Strategy 2: Aggressive Top-20 Merging
- Targets regimes outside the top 20 for merging into top-20 regimes
- Uses 20% lower similarity threshold for more aggressive merging
- Only applied when `aggressive_merging=True`

#### Strategy 3: Frequency-Based Merging
- Merges very low-frequency regimes (below 0.1%) into any similar regime
- Uses 50% similarity threshold for very low frequency regimes
- Only applied when `aggressive_merging=True`

### 3. Concentration Monitoring

The system now provides:
- Real-time concentration tracking
- Automatic guidance for achieving target concentrations
- Detailed logging of merging decisions
- Top 10 regime frequency visibility

## Usage Examples

### Command Line Usage

```bash
# Basic usage with 75% target concentration
python src/training/steps/step1_7_hmm_regime_discovery.py \
  --target-concentration 0.75 \
  --similarity-threshold 0.75 \
  --min-frequency 0.003 \
  --max-regimes 20 \
  --aggressive-merging

# More aggressive merging for 80% concentration
python src/training/steps/step1_7_hmm_regime_discovery.py \
  --target-concentration 0.80 \
  --similarity-threshold 0.70 \
  --min-frequency 0.002 \
  --max-regimes 20 \
  --aggressive-merging

# Conservative merging for 70% concentration
python src/training/steps/step1_7_hmm_regime_discovery.py \
  --target-concentration 0.70 \
  --similarity-threshold 0.80 \
  --min-frequency 0.005 \
  --max-regimes 25 \
  --aggressive-merging
```

### Programmatic Usage

```python
import asyncio
from src.training.steps.step1_7_hmm_regime_discovery import run_step

async def run_with_concentration_target():
    success = await run_step(
        symbol="ETHUSDT",
        exchange="BINANCE",
        data_dir="data/training",
        timeframe="1m",
        lookback_days=180,
        force_rerun=True,
        cluster_algorithm="kmeans",
        target_num_clusters=20,
        min_combination_frequency=0.003,
    )
    return success

# Run with enhanced configuration
asyncio.run(run_with_concentration_target())
```

## Configuration Recommendations

### For 70% Concentration
```python
REGIME_MERGING_CONFIG = {
    "min_frequency": 0.005,          # 0.5% minimum frequency
    "similarity_threshold": 0.80,     # 80% similarity threshold
    "max_regimes": 25,               # 25 regimes
    "target_top_20_concentration": 0.70,
    "aggressive_merging": True,
}
```

### For 75% Concentration (Recommended)
```python
REGIME_MERGING_CONFIG = {
    "min_frequency": 0.003,          # 0.3% minimum frequency
    "similarity_threshold": 0.75,     # 75% similarity threshold
    "max_regimes": 20,               # 20 regimes
    "target_top_20_concentration": 0.75,
    "aggressive_merging": True,
}
```

### For 80% Concentration
```python
REGIME_MERGING_CONFIG = {
    "min_frequency": 0.002,          # 0.2% minimum frequency
    "similarity_threshold": 0.70,     # 70% similarity threshold
    "max_regimes": 15,               # 15 regimes
    "target_top_20_concentration": 0.80,
    "aggressive_merging": True,
}
```

## Monitoring and Analysis

### Concentration Tracking

The system provides detailed logging of concentration levels:

```
🔍 Regime merging analysis:
   Total observations: 100,000
   Initial regimes: 45
   Current top 20 concentration: 65.2%
   Target top 20 concentration: 75.0%
   Aggressive merging: True

✅ Regime merging completed:
   Regimes merged: 12
   Final regimes: 33
   Final top 20 concentration: 76.8%
   Target achieved: True
```

### Guidance System

The system provides automatic guidance for achieving target concentrations:

```
📊 Concentration Analysis:
🎯 To achieve 75.0% concentration (current: 65.2%, gap: 9.8%):
   • Use moderate aggressive merging (similarity_threshold: 0.7-0.8)
   • Set min_frequency to 0.003 (0.3%)
   • Enable aggressive_merging=True
   • Target max_regimes around 20-25
   • Monitor top 10 regime frequencies for balance
```

## Testing

Use the provided test script to validate different concentration targets:

```bash
python test_regime_concentration.py
```

This script will test different concentration targets (70%, 75%, 80%) and provide recommendations.

## Troubleshooting

### Low Concentration Issues

If you're not achieving the target concentration:

1. **Lower the similarity threshold**: Try 0.70 instead of 0.75
2. **Lower the minimum frequency**: Try 0.002 instead of 0.003
3. **Enable aggressive merging**: Ensure `aggressive_merging=True`
4. **Reduce max_regimes**: Try 15-20 instead of 25+

### High Concentration Issues

If concentration is too high (regimes are too consolidated):

1. **Increase the similarity threshold**: Try 0.80 instead of 0.75
2. **Increase the minimum frequency**: Try 0.005 instead of 0.003
3. **Disable aggressive merging**: Set `aggressive_merging=False`
4. **Increase max_regimes**: Try 25-30 instead of 20

### Performance Considerations

- Aggressive merging may increase processing time
- Lower similarity thresholds may reduce regime quality
- Monitor regime descriptions to ensure meaningful consolidation
- Balance concentration targets with regime interpretability

## Expected Results

With the recommended configuration (75% target), you should see:

- **Top 20 regime concentration**: 70-80%
- **Total regimes**: 15-25
- **Top 10 regime frequencies**: 2-8% each
- **Regime quality**: Meaningful, interpretable market states
- **Processing time**: 10-30% increase due to aggressive merging

## Conclusion

The enhanced regime merging functionality provides flexible control over regime concentration while maintaining regime quality and interpretability. The multi-strategy approach ensures robust consolidation of low-frequency regimes into meaningful market archetypes.
