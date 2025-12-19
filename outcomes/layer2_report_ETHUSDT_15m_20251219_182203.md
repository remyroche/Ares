# Layer2 Report
- timestamp: 20251219_182203
- symbol: ETHUSDT
- timeframe: 15m
- n_bars: 34561
- n_events: 12614
- cache_hits: 647
- cache_misses: 2211
- extracted_trials_per_family: {'Mean Reversion': 15, 'Trend Continuation': 11, 'Momentum': 17}
- production_geometries_by_family: {'Momentum': 2, 'Trend Continuation': 2, 'Mean Reversion': 2}
- production_geometries_n: 6
- oof_labeled_events: 3499
- oof_nonzero_weight_events: 10241
- oof_geometry_channels: 10

## Diagnostics
### 1. Signal Coverage (First-Order Test)
- **Coverage**: 6.37%
- **Diagnosis**:
  - < 5-10%: Under-hunting (over-regularised)
  - 20-50%: Healthy hunting regime
  - > 70%: Likely noise saturation

### 2. Prediction Entropy Distribution
- **Mean Entropy**: 0.5906 (Max ~0.693)
- **Entropy Std**: 0.1528
- **Diagnosis**:
  - Mass near 0 or 1: Over-confident / brittle
  - Mass near 0.5: Under-hunting
  - Wide distribution: Healthy

### 3. Feature Utilisation / Split Diversity
- **Avg Features Used**: 49.3
- **Avg Leaf Depth**: 6.10
- **Diagnosis**:
  - Few features, shallow: Over-regularised
  - Many features, deep: Expressive (desired)
