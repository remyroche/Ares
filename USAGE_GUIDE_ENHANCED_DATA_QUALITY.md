# Enhanced Data Quality Checker - Usage Guide

## Overview

This guide explains how to use the enhanced data quality checker to automatically fix irregular interval issues that are causing data quality warnings in your feature engineering pipeline.

## The Problem

You're seeing warnings like:
```
⚠️ Moderate time interval variability (CV: 0.276, irregular: 0.6%) may affect multi-timeframe feature generation
⚠️ Scattered irregular timestamp intervals: 0.6% (threshold: 0.1%) - may affect multi-timeframe feature generation
```

These warnings indicate that your data has irregular time intervals, which can affect multi-timeframe feature generation and other time-sensitive operations.

## The Solution

The enhanced data quality checker implements an intelligent preprocessing strategy:

1. **Resample** to expected intervals
2. **Re-add original data** to preserve accuracy
3. **Forward-fill** if missing values are less than 10 seconds
4. **Download missing data** for gaps > 10 seconds using existing download functions

## Quick Fix: Using the Decorator

The easiest way to fix these issues is to use the `@auto_fix_data_quality_issues` decorator:

```python
from src.training.steps.raw_data_quality_checker import auto_fix_data_quality_issues

@auto_fix_data_quality_issues
def analyze_patterns(data, symbol, exchange):
    # Your existing analysis code here
    # The decorator will automatically fix irregular intervals before this runs
    pass

@auto_fix_data_quality_issues
def analyze_momentum_vectorized(data, symbol, exchange):
    # Your existing momentum analysis code here
    # The decorator will automatically fix irregular intervals before this runs
    pass
```

## Manual Usage

### Option 1: Fix Irregular Intervals Only

```python
from src.training.steps.raw_data_quality_checker import fix_irregular_intervals_automatically

# Fix irregular intervals in your data
fixed_data = fix_irregular_intervals_automatically(data, symbol, exchange)

# Use the fixed data for your analysis
results = analyze_patterns(fixed_data, symbol, exchange)
```

### Option 2: Comprehensive Validation and Fix

```python
from src.training.steps.raw_data_quality_checker import validate_and_fix_data_quality_issues

# Validate and fix all data quality issues
fixed_data, validation_results = validate_and_fix_data_quality_issues(data, symbol, exchange)

print(f"Quality improvement: {validation_results['preprocessing_summary']['quality_improvement']:.3f}")
print(f"Fixes applied: {validation_results['preprocessing_summary']['fixes_applied']}")

# Use the fixed data for your analysis
results = analyze_patterns(fixed_data, symbol, exchange)
```

### Option 3: Enhanced Preprocessing with Custom Settings

```python
from src.training.steps.raw_data_quality_checker import enhanced_preprocess_market_data

# Apply enhanced preprocessing with custom settings
preprocessed_data = enhanced_preprocess_market_data(
    data=data,
    symbol=symbol,
    exchange=exchange,
    expected_interval_seconds=60,  # 1-minute intervals
    max_forward_fill_seconds=10,   # Forward-fill gaps ≤ 10 seconds
    download_missing_data=True     # Download data for larger gaps
)

# Use the preprocessed data for your analysis
results = analyze_patterns(preprocessed_data, symbol, exchange)
```

## Integration with Existing Code

### Before (with warnings):
```python
def analyze_patterns(data, symbol, exchange):
    # This function triggers data quality warnings
    time_diffs = data.index.to_series().diff().dropna()
    # ... rest of your analysis code
```

### After (with auto-fix):
```python
from src.training.steps.raw_data_quality_checker import auto_fix_data_quality_issues

@auto_fix_data_quality_issues
def analyze_patterns(data, symbol, exchange):
    # This function will automatically fix irregular intervals before running
    time_diffs = data.index.to_series().diff().dropna()
    # ... rest of your analysis code (now with regular intervals)
```

## Configuration

You can customize the behavior by passing a configuration:

```python
from src.training.steps.raw_data_quality_checker import RawDataQualityChecker

config = {
    "preprocessing": {
        "max_forward_fill_seconds": 15,  # Increase forward-fill threshold
        "auto_fix_irregular_intervals": True,
        "download_missing_data": True,
        "preserve_original_data": True,
        "min_interval_seconds": 60,
    }
}

checker = RawDataQualityChecker(config)
fixed_data = checker.fix_irregular_intervals_automatically(data, symbol, exchange)
```

## Testing

Run the test script to see the enhanced data quality checker in action:

```bash
python test_enhanced_data_quality_fix.py
```

This will demonstrate:
- Creating test data with irregular intervals
- Detecting and analyzing interval issues
- Applying automatic fixes
- Measuring quality improvements

## Expected Results

After applying the fixes, you should see:

1. **Eliminated warnings** about irregular intervals
2. **Improved data quality scores**
3. **Regular time intervals** suitable for multi-timeframe feature generation
4. **Preserved data accuracy** through the re-add strategy

## Monitoring

The enhanced checker provides detailed logging:

```
🔧 Auto-fixing irregular intervals for analyze_patterns (ratio: 0.006, CV: 0.276)
🔧 Enhanced preprocessing for binance BTCUSDT
   Expected interval: 60s
   Max forward-fill: 10s
   Download missing: True
🔧 Step 1: Resampling to 60S intervals
🔧 Step 2: Re-adding original data to preserve accuracy
🔧 Step 3: Analyzing gaps and applying intelligent handling
✅ Enhanced preprocessing completed:
   Original shape: (1000, 5)
   Final shape: (1000, 5)
   Remaining large gaps: 0
   Data completeness: 1.000
```

## Compatibility

The enhanced data quality checker is fully compatible with:

- ✅ Existing data download functions
- ✅ CSV/Parquet/Pickle file formats
- ✅ Existing decorators and validation systems
- ✅ Multi-timeframe feature generation
- ✅ Wavelet and microstructure features

## Troubleshooting

### If fixes don't work:
1. Check that your data has a datetime index
2. Verify the symbol and exchange parameters
3. Ensure data download functions are available
4. Check the logs for specific error messages

### If you want to disable automatic fixing:
```python
config = {
    "preprocessing": {
        "auto_fix_irregular_intervals": False
    }
}
```

### If you want to disable data downloading:
```python
config = {
    "preprocessing": {
        "download_missing_data": False
    }
}
```

## Summary

The enhanced data quality checker provides a comprehensive solution to the irregular interval warnings you're experiencing. By using the `@auto_fix_data_quality_issues` decorator, you can automatically fix these issues without modifying your existing analysis code.

The intelligent preprocessing strategy ensures that:
- Data accuracy is preserved
- Irregular intervals are fixed
- Missing data is intelligently handled
- Multi-timeframe features work correctly

This should eliminate the warnings and improve the quality of your feature engineering pipeline.