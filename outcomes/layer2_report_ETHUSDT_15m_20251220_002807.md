# Layer2 Report
- timestamp: 20251220_002807
- symbol: ETHUSDT
- timeframe: 15m
- n_bars: 34561
- n_events: 12614
- cache_hits: 0
- cache_misses: 595
- extracted_trials_per_family: {'Mean Reversion': 30, 'Trend Continuation': 30, 'Momentum': 30}
- production_geometries_by_family: {'Mean Reversion': 2, 'Trend Continuation': 4, 'Momentum': 4}
- production_geometries_n: 10
- oof_labeled_events: 8778
- oof_nonzero_weight_events: 8778
- oof_geometry_channels: 10

## Diagnostics
### 1. Signal Coverage (First-Order Test)
- **Coverage**: 2.24%
- **Diagnosis**:
  - < 5-10%: Under-hunting (over-regularised)
  - 20-50%: Healthy hunting regime
  - > 70%: Likely noise saturation

### 2. Prediction Entropy Distribution
- **Mean Entropy**: 0.3571 (Max ~0.693)
- **Entropy Std**: 0.3431
- **Diagnosis**:
  - Mass near 0 or 1: Over-confident / brittle
  - Mass near 0.5: Under-hunting
  - Wide distribution: Healthy

### 3. Feature Utilisation / Split Diversity
- **Avg Features Used**: 71.3
- **Avg Leaf Depth**: 6.52
- **Diagnosis**:
  - Few features, shallow: Over-regularised
  - Many features, deep: Expressive (desired)
