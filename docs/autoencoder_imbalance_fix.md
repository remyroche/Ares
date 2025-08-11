# Autoencoder Label Imbalance Fix

The VectorizedLabellingOrchestrator now includes comprehensive handling of extreme label imbalance in autoencoder generation to prevent issues with highly imbalanced datasets.

## Problem

The system was encountering extreme label imbalance in autoencoder generation:
```
⚠️ Extreme label imbalance in autoencoder_generation: ratio=167702.0
⚠️ Very small class in autoencoder_generation: 1 samples
```

This occurred because the autoencoder was using the same imbalanced labels as the main labeling process, which can cause training issues and poor feature generation.

## Solution

### 1. Intelligent Label Strategy Selection

The system now automatically chooses the best labeling strategy for autoencoder training:

```python
# Check for extreme label imbalance
label_validation = self._validate_labels_comprehensive(original_labels, "autoencoder_generation")
imbalance_ratio = label_validation.get("label_quality", {}).get("imbalance_ratio", 1.0)

# Choose strategy based on imbalance severity
if imbalance_ratio > self.autoencoder_imbalance_threshold:
    # Use unsupervised learning for extreme imbalance
    autoencoder_labels = None
elif self.enable_autoencoder_balanced_labels and imbalance_ratio > 2.0:
    # Create balanced labels for autoencoder
    autoencoder_labels = self._create_balanced_autoencoder_labels(original_labels)
else:
    # Use original labels if imbalance is acceptable
    autoencoder_labels = original_labels
```

### 2. Configuration Options

Add these options to your `autoencoder` configuration section:

```yaml
autoencoder:
  # Label imbalance handling
  enable_unsupervised_fallback: true    # Use unsupervised learning for extreme imbalance
  imbalance_threshold: 10.0             # Ratio threshold for using unsupervised learning
  enable_balanced_labels: false         # Enable balanced label creation
```

### 3. Three Labeling Strategies

#### A. Unsupervised Learning (Default for Extreme Imbalance)
- **Trigger**: Imbalance ratio > 10.0
- **Method**: `autoencoder_labels = None`
- **Use case**: Extreme imbalance (ratio > 10.0)
- **Benefits**: Avoids training issues with highly imbalanced data

#### B. Balanced Labels (Optional)
- **Trigger**: Imbalance ratio > 2.0 AND `enable_balanced_labels: true`
- **Method**: Undersampling majority classes to match minority class
- **Use case**: Moderate to high imbalance
- **Benefits**: Maintains supervised learning while reducing imbalance

#### C. Original Labels (Default for Acceptable Imbalance)
- **Trigger**: Imbalance ratio ≤ 10.0
- **Method**: Use original labels as-is
- **Use case**: Acceptable imbalance levels
- **Benefits**: Preserves original label distribution

## Log Output Examples

### Extreme Imbalance Detected
```
📊 Autoencoder label analysis: ratio=167702.0, severity=extreme
🚨 CRITICAL FIX: Extreme label imbalance detected for autoencoder (ratio=167702.0). Using unsupervised learning.
```

### Balanced Labels Created
```
📊 Autoencoder label analysis: ratio=15.2, severity=significant
🔄 Creating balanced labels for autoencoder training...
📊 Creating balanced autoencoder labels:
   Original distribution: {0: 1500, 1: 100}
   Target balanced count per class: 100
   Balanced distribution: {0: 100, 1: 100}
   New imbalance ratio: 1.0
✅ Successfully created balanced autoencoder labels
```

### Acceptable Imbalance
```
📊 Autoencoder label analysis: ratio=2.1, severity=moderate
✅ Using original labels for autoencoder (ratio=2.1)
```

## Benefits

1. **Prevents Training Issues**: Avoids problems with extremely imbalanced autoencoder training
2. **Flexible Strategy**: Automatically chooses the best approach based on data characteristics
3. **Configurable**: Users can control the behavior through configuration options
4. **Comprehensive Logging**: Clear visibility into the decision-making process
5. **Fallback Safety**: Always has a safe fallback option (unsupervised learning)

## Implementation Details

### Label Validation
The system uses comprehensive label validation to assess imbalance:
- Calculates imbalance ratio (max_class_count / min_class_count)
- Determines severity level (balanced, moderate, significant, extreme)
- Provides detailed distribution analysis

### Balanced Label Creation
When creating balanced labels:
- Undersamples majority classes to match minority class count
- Maintains random sampling to preserve data diversity
- Verifies the new balance ratio
- Falls back to unsupervised learning if balance cannot be achieved

### Configuration Integration
The fix integrates with existing configuration systems:
- Uses `autoencoder` configuration section
- Respects existing feature selection and data processing settings
- Maintains compatibility with other system components

## Troubleshooting

If you still see imbalance issues:

1. **Check configuration**: Ensure `enable_unsupervised_fallback: true`
2. **Adjust threshold**: Lower `imbalance_threshold` if needed
3. **Enable balanced labels**: Set `enable_balanced_labels: true` for moderate imbalance
4. **Review logs**: Check the detailed analysis output
5. **Verify data**: Ensure the original labeling process is working correctly 