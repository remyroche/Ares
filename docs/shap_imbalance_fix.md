# SHAP Imbalance Fix

The AutoencoderFeatureGenerator now includes comprehensive handling of extreme label imbalance in SHAP computation to prevent issues with highly imbalanced datasets during feature importance calculation.

## Problem

The system was encountering issues with SHAP computation due to extreme label imbalance:
```
⚠️ Insufficient samples per class for stratification (min: 1)
```

This occurred because the SHAP computation was trying to use stratified sampling on extremely imbalanced labels, which failed when there were insufficient samples per class.

## Solution

### 1. Enhanced Label Distribution Analysis

The system now analyzes label distribution before SHAP computation:

```python
# Check if we have enough samples per class for stratification
unique_labels, label_counts = np.unique(y, return_counts=True)
min_class_count = label_counts.min()
max_class_count = label_counts.max()
imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')

self.logger.info(f"📊 Label distribution analysis:")
self.logger.info(f"   Unique labels: {unique_labels}")
self.logger.info(f"   Label counts: {label_counts}")
self.logger.info(f"   Min class count: {min_class_count}")
self.logger.info(f"   Imbalance ratio: {imbalance_ratio:.1f}")
```

### 2. Intelligent Sampling Strategy Selection

The system automatically chooses the best sampling strategy based on imbalance severity:

```python
# CRITICAL FIX: Handle extreme imbalance for SHAP computation
shap_imbalance_threshold = self.config.get("feature_filtering.shap_imbalance_threshold", 100.0)
enable_shap_imbalance_handling = self.config.get("feature_filtering.enable_shap_imbalance_handling", True)

if enable_shap_imbalance_handling and imbalance_ratio > shap_imbalance_threshold:
    # Use random sampling for extreme imbalance
    self.logger.warning(f"🚨 CRITICAL FIX: Extreme label imbalance detected (ratio={imbalance_ratio:.1f} > {shap_imbalance_threshold})")
    self.logger.info("🔄 Using random sampling for SHAP computation...")
    sample_indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X.iloc[sample_indices]
    y_sample = y[sample_indices]
elif min_class_count >= 10:
    # Use stratified sampling for acceptable imbalance
    # ... stratified sampling logic
else:
    # Use random sampling for insufficient samples
    # ... random sampling logic
```

### 3. Configuration Options

Add these options to your `feature_filtering` configuration section:

```yaml
feature_filtering:
  # SHAP imbalance handling
  shap_imbalance_threshold: 100.0              # Threshold for extreme imbalance
  enable_shap_imbalance_handling: true         # Enable extreme imbalance handling
  # ... other existing options
```

## Three Sampling Strategies

### A. Random Sampling (Default for Extreme Imbalance)
- **Trigger**: Imbalance ratio > 100.0
- **Method**: `np.random.choice(len(X), sample_size, replace=False)`
- **Use case**: Extreme imbalance (ratio > 100.0)
- **Benefits**: Avoids stratification issues with highly imbalanced data

### B. Stratified Sampling (Default for Acceptable Imbalance)
- **Trigger**: Min class count ≥ 10 AND imbalance ratio ≤ 100.0
- **Method**: `train_test_split` with `stratify=y`
- **Use case**: Acceptable imbalance levels with sufficient samples per class
- **Benefits**: Maintains class balance in the sample

### C. Random Sampling (Fallback for Insufficient Samples)
- **Trigger**: Min class count < 10
- **Method**: `np.random.choice(len(X), sample_size, replace=False)`
- **Use case**: Insufficient samples per class for stratification
- **Benefits**: Ensures SHAP computation can proceed

## Log Output Examples

### Extreme Imbalance Detected
```
📊 Label distribution analysis:
   Unique labels: [0. 1.]
   Label counts: [999   1]
   Min class count: 1
   Imbalance ratio: 999.0
🚨 CRITICAL FIX: Extreme label imbalance detected (ratio=999.0 > 100.0)
🔄 Using random sampling for SHAP computation...
📊 Random sample size: 25767 rows
```

### Acceptable Imbalance
```
📊 Label distribution analysis:
   Unique labels: [0. 1.]
   Label counts: [800 200]
   Min class count: 200
   Imbalance ratio: 4.0
✅ Stratified sampling successful!
📊 Original class distribution: {0: 800, 1: 200}
📊 Sample class distribution: {0: 80, 1: 20}
```

### Insufficient Samples
```
📊 Label distribution analysis:
   Unique labels: [0. 1.]
   Label counts: [15  5]
   Min class count: 5
   Imbalance ratio: 3.0
⚠️ Insufficient samples per class for stratification (min: 5)
🔄 Using random sampling...
📊 Random sample size: 25767 rows
```

## Benefits

1. **Prevents SHAP Failures**: Avoids stratification errors with imbalanced data
2. **Maintains Efficiency**: Ensures SHAP computation can proceed regardless of label distribution
3. **Configurable Thresholds**: Users can adjust imbalance thresholds based on their needs
4. **Comprehensive Logging**: Clear visibility into the decision-making process
5. **Fallback Safety**: Always has a safe fallback option (random sampling)

## Implementation Details

### Label Analysis
The system performs comprehensive label analysis:
- Calculates unique labels and their counts
- Determines minimum and maximum class counts
- Computes imbalance ratio (max_class_count / min_class_count)
- Logs detailed distribution information

### Decision Logic
The system uses a three-tier decision process:
1. **Extreme imbalance check**: If ratio > threshold, use random sampling
2. **Sufficient samples check**: If min_class_count ≥ 10, use stratified sampling
3. **Fallback**: Use random sampling for all other cases

### Configuration Integration
The fix integrates with existing configuration systems:
- Uses `feature_filtering` configuration section
- Respects existing SHAP computation settings
- Maintains compatibility with other system components

## Troubleshooting

If you still see SHAP issues:

1. **Check configuration**: Ensure `enable_shap_imbalance_handling: true`
2. **Adjust threshold**: Lower `shap_imbalance_threshold` if needed
3. **Review logs**: Check the detailed analysis output
4. **Verify data**: Ensure the label distribution is as expected
5. **Monitor performance**: Check if random sampling provides adequate results 