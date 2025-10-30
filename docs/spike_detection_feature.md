# Spike Detection and Noise Filtering

## Overview

The spike detection and correction feature automatically identifies and removes price spikes (noise) from market data before opportunity labeling. This improves signal quality by filtering out false opportunities caused by data anomalies.

## How It Works

### Detection Criteria

A price spike is detected when **BOTH** conditions are met:

1. **Deviation from Baseline**: `|s_t - median(s_{t-1..t-N})| > threshold`
   - The current price deviates significantly from the rolling median baseline
   - `s_t`: Current bar's price
   - `median(s_{t-1..t-N})`: Rolling median over past N bars (excluding current)
   - `threshold`: k × recent std (k = threshold_multiplier, typically 3.0)

2. **Direction Reversal**: `sign(s_t - s_{t-1}) != sign(s_{t+1} - s_t)`
   - The price movement reverses direction (whipsaw pattern)
   - `s_t - s_{t-1}`: Movement from previous bar to current
   - `s_{t+1} - s_t`: Movement from current bar to next
   - If signs differ, it indicates a temporary spike rather than a sustained trend

### Correction Method

When a spike is detected:
1. Calculate corrected price: `corrected_price = (prev_price + spike_price + next_price) / 3`
2. Replace the spike with the 3-bar average (including the spike itself)
3. Track spike magnitude and statistics

**Why 3-bar average?**
- More conservative approach than 2-bar average
- Partially preserves potential signal in the spike (may not be 100% noise)
- Smooths the spike while retaining some of the price movement information

### Trend Preservation

**Important**: If the price movement continues in the same direction (no reversal), it's considered part of a genuine trend and is **NOT** clipped. This ensures real market movements are preserved while noise is filtered.

## Configuration

### Default Parameters

```python
spike_detection_config = {
    'lookback_window': 10,        # N bars for median baseline
    'threshold_multiplier': 3.0,  # k × std for threshold
    'volatility_window': 20       # Window for volatility calculation
}
```

### Customization

Pass configuration in the step config:

```python
config = {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'timeframe': '15m',
    'spike_detection': {
        'lookback_window': 15,      # Increase for more stable baseline
        'threshold_multiplier': 2.5, # Lower = more sensitive (detect more spikes)
        'volatility_window': 30      # Increase for smoother volatility estimate
    }
}
```

## Integration

### Execution Flow

1. **Load market data** from klines manager
2. **Detect and correct spikes** (this feature)
   - Calculate rolling median baseline
   - Identify deviations exceeding threshold
   - Check for direction reversals
   - Correct spikes to average of adjacent bars
3. **Generate volatility labels** on cleaned data
4. **Detect opportunities** (now more accurate due to noise filtering)

### Statistics Tracked

- `spikes_detected`: Total number of spikes identified
- `spikes_corrected`: Number successfully corrected
- `spike_correction_rate`: Percentage of spikes corrected
- `avg_spike_magnitude`: Average deviation (%)
- `max_spike_magnitude`: Maximum deviation (%)
- `spike_percentage`: Percentage of data points that were spikes

## Benefits

1. **Improved Signal Quality**: Removes false opportunities caused by data noise
2. **Better Labeling Accuracy**: Volatility labeler works with cleaner data
3. **Reduced False Positives**: Fewer spurious trading signals
4. **Trend Preservation**: Genuine market movements are not affected
5. **Transparent**: Full statistics reported for monitoring

## Example Output

```
🔍 Spike Detection Results:
   • Spikes detected: 142
   • Spikes corrected: 138
   • Correction rate: 97.2%
   • Avg spike magnitude: 0.34%
   • Max spike magnitude: 1.89%

📈 Labeling Results Summary:
   • Total samples: 10,000
   • Opportunities detected: 856 (8.6%)
   • Long opportunities: 856
   • Short opportunities: 0
```

## Technical Details

### Implementation

The spike detection is implemented in:
- **Function**: `detect_and_correct_price_spikes()`
- **File**: `src/training/steps/pre_training/feature_generation_labeling_integration_step.py`
- **Lines**: 84-224

### Performance

- **Complexity**: O(n) where n = number of data points
- **Memory**: O(n) for rolling window calculations
- **Execution Time**: Typically < 100ms for 10,000 samples

### Edge Cases Handled

- **Boundary conditions**: First and last bars cannot be corrected (no prev/next)
- **NaN values**: Automatically skipped
- **Missing data**: Rolling windows use `min_periods` fallback
- **Empty results**: Gracefully returns original data with zero stats

## Monitoring

### Metrics Reported

All spike detection metrics are included in:
1. **Console output**: Real-time feedback during execution
2. **Technical metrics**: `technical_metrics['spike_detection']`
3. **Process metrics**: `process_metrics['spike_detection_process']`
4. **Comprehensive report**: Full statistics in generated report

### Warning Thresholds

- **High spike rate** (>5%): May indicate data quality issues
- **Large magnitude spikes** (>5%): May indicate exchange API problems
- **Low correction rate** (<90%): May indicate boundary/NaN issues

## Best Practices

1. **Start with defaults**: Default parameters work well for most crypto pairs
2. **Monitor spike rates**: Track spike percentage over time
3. **Adjust sensitivity**: Lower `threshold_multiplier` for noisier data
4. **Review corrections**: Check `avg_spike_magnitude` for reasonableness
5. **Validate labels**: Compare opportunity detection with/without spike filtering

## Future Enhancements

Potential improvements:
- Adaptive threshold based on market regime
- Multi-column spike detection (OHLCV consistency)
- Spike pattern classification (flash crash, wick, etc.)
- Historical spike analysis and reporting
- Configurable correction methods (median, weighted average, etc.)

## References

- Triple barrier method: Marcos López de Prado, "Advances in Financial Machine Learning"
- Outlier detection: Tukey's method for identifying outliers
- Median absolute deviation (MAD): Robust statistical measure for outlier detection

