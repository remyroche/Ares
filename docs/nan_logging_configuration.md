# NaN Logging Configuration

The VectorizedLabellingOrchestrator now includes configurable NaN logging to prevent excessive log output while still providing useful information about data quality issues.

## Configuration Options

Add these options to your `vectorized_labelling_orchestrator` configuration section:

```yaml
vectorized_labelling_orchestrator:
  # NaN Logging Configuration
  nan_logging_verbosity: "summary"  # Options: "minimal", "summary", "detailed"
  max_nan_ranges_to_log: 5          # Maximum number of ranges to show in logs
  enable_detailed_nan_logging: true # Master switch for detailed NaN logging
```

## Verbosity Levels

### Minimal (`"minimal"`)
Shows only the count of ranges and total NaN values:
```
📅 4 NaN ranges found (total 19 NaNs)
```

### Summary (`"summary"`) - Default
Shows a summary with limited details:
- For ≤3 ranges: Shows all ranges
- For >3 ranges: Shows first 2 and last 1 ranges with "..." indicator

Example:
```
📅 4 NaN ranges found (total 19 NaNs)
  Range 1: 2025-01-01 10:00:00 to 2025-01-01 10:00:00 (1 NaNs)
  Range 2: 2025-01-01 12:00:00 to 2025-01-01 12:05:00 (6 NaNs)
  ... (1 more ranges)
  Range 4: 2025-01-01 16:00:00 to 2025-01-01 16:10:00 (11 NaNs)
```

### Detailed (`"detailed"`)
Shows all ranges with full details (limited by `max_nan_ranges_to_log`):
```
📅 NaN timestamp ranges:
  Range 1: 2025-01-01 10:00:00 to 2025-01-01 10:00:00 (0.0 minutes, 1 NaNs)
  Range 2: 2025-01-01 12:00:00 to 2025-01-01 12:05:00 (5.0 minutes, 6 NaNs)
  ...
```

## Configuration Examples

### For Production (Minimal Logging)
```yaml
vectorized_labelling_orchestrator:
  nan_logging_verbosity: "minimal"
  enable_detailed_nan_logging: true
```

### For Development (Detailed Logging)
```yaml
vectorized_labelling_orchestrator:
  nan_logging_verbosity: "detailed"
  max_nan_ranges_to_log: 10
  enable_detailed_nan_logging: true
```

### To Disable NaN Logging Completely
```yaml
vectorized_labelling_orchestrator:
  enable_detailed_nan_logging: false
```

## What Was Fixed

### Before (Problematic)
The system was logging every individual NaN as a separate range:
```
Range 687: 253658 to 253659 (2 NaNs)
Range 688: 253660 to 253661 (2 NaNs)
Range 689: 253662 to 253663 (2 NaNs)
... (hundreds more lines)
```

### After (Fixed)
The system now groups consecutive NaN values into meaningful ranges:
```
📅 4 NaN ranges found (total 19 NaNs)
  Range 1: 2025-01-01 10:00:00 to 2025-01-01 10:00:00 (1 NaNs)
  Range 2: 2025-01-01 12:00:00 to 2025-01-01 12:05:00 (6 NaNs)
  Range 3: 2025-01-01 14:00:00 to 2025-01-01 14:00:00 (1 NaNs)
  Range 4: 2025-01-01 16:00:00 to 2025-01-01 16:10:00 (11 NaNs)
```

## Benefits

1. **Reduced Log Spam**: No more hundreds of individual NaN range entries
2. **Better Readability**: Consecutive NaNs are grouped into meaningful ranges
3. **Configurable**: Users can control verbosity based on their needs
4. **Performance**: Faster logging with less output
5. **Maintainable**: Clear, structured information about data quality issues 