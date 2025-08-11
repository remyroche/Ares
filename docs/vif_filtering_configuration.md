# VIF Filtering Configuration

The VectorizedFeatureSelector now includes comprehensive VIF (Variance Inflation Factor) filtering to prevent multicollinearity issues that can cause infinite VIF scores.

## Configuration Options

Add these options to your `feature_selection` configuration section:

```yaml
feature_selection:
  # VIF Filtering Configuration
  vif_threshold: 5.0                    # VIF threshold for feature removal (default: 5.0)
  enable_vif_removal: true              # Enable VIF-based feature removal
  enable_redundant_price_filtering: true # Enable redundant price feature filtering
  
  # Emergency override settings
  emergency_override_infinite_vif: true  # Automatically remove features with infinite VIF
  emergency_override_perfect_correlation: true # Remove perfectly correlated features
```

## How It Works

### 1. Redundant Price Feature Filtering
The system automatically removes known redundant price features that cause multicollinearity:

```python
redundant_price_features = [
    'open', 'high', 'low', 'avg_price', 'min_price', 'max_price',
    'open_price_change', 'high_price_change', 'low_price_change',
    'avg_price_change', 'min_price_change', 'max_price_change',
    'trade_volume', 'volume_ratio',
    'close_returns',  # Redundant with price_change features
    'open_returns', 'high_returns', 'low_returns',  # Redundant return features
]
```

### 2. Perfect Correlation Detection
Before calculating VIF scores, the system detects and removes perfectly correlated features (correlation ≥ 0.999999):

```
🚨 Found 1 perfectly correlated feature pairs:
   close_returns <-> 1m_1m_price_change (correlation = 1.0)
🚨 Removing 1 perfectly correlated features: ['1m_1m_price_change']
```

### 3. VIF Score Calculation
The system calculates VIF scores safely with proper handling of infinite values:

```python
# Handle perfect correlation (r_squared = 1)
if r_squared >= 0.999999:
    vif_scores[col] = float('inf')
else:
    vif = 1 / (1 - r_squared) if r_squared != 1 else np.inf
    vif_scores[col] = vif
```

### 4. Extreme VIF Removal
Features with VIF > 1000 or infinite VIF are automatically removed:

```
🚨 Removing 2 features with extreme VIF:
   close_returns: VIF=inf
   1m_1m_price_change: VIF=inf
```

## Log Output Examples

### Successful Filtering
```
🔧 CRITICAL FIX: Filtering redundant price features...
🚨 Removing 8 redundant price features: ['open', 'high', 'low', 'avg_price', 'min_price', 'max_price', 'trade_volume', 'volume_ratio']
✅ Removed redundant features. Remaining features: 27

🔧 CRITICAL FIX: Checking for extreme VIF features...
🚨 Found 1 perfectly correlated feature pairs:
   close_returns <-> 1m_1m_price_change (correlation = 1.0)
🚨 Removing 1 perfectly correlated features: ['1m_1m_price_change']
✅ Removed perfectly correlated features. Remaining features: 26
✅ No extreme VIF features found
```

### No Issues Found
```
🔧 CRITICAL FIX: Filtering redundant price features...
✅ No redundant price features found to remove

🔧 CRITICAL FIX: Checking for extreme VIF features...
✅ No perfectly correlated features found
✅ No extreme VIF features found
```

## Benefits

1. **Prevents Infinite VIF**: Eliminates features that would cause infinite VIF scores
2. **Improves Model Stability**: Reduces multicollinearity issues
3. **Automatic Detection**: No manual intervention required
4. **Configurable**: Can be enabled/disabled as needed
5. **Comprehensive Logging**: Clear visibility into what's being removed and why

## Troubleshooting

If you still see infinite VIF errors:

1. **Check feature names**: Ensure problematic features are in the `redundant_price_features` list
2. **Enable perfect correlation detection**: Set `emergency_override_perfect_correlation: true`
3. **Lower VIF threshold**: Reduce `vif_threshold` to catch more multicollinearity issues
4. **Review logs**: Check the detailed logging output to understand what's being removed 