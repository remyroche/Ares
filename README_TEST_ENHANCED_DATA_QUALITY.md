# Enhanced Data Quality Checker Test

## Overview

This test demonstrates the enhanced data quality checker's ability to automatically fix irregular interval issues that cause data quality warnings.

## Running the Test

### Prerequisites

1. Make sure you're in the project root directory
2. Ensure all dependencies are installed
3. The `src/` directory should be accessible

### Command

```bash
# From the project root directory
python test_enhanced_data_quality_fix.py
```

### What the Test Does

The test script:

1. **Creates test data** with irregular intervals (similar to real-world issues)
2. **Demonstrates detection** of irregular interval problems
3. **Shows automatic fixing** using the enhanced preprocessing strategy
4. **Tests the decorator** functionality
5. **Measures improvements** in data quality

### Expected Output

You should see output like:

```
🚀 Enhanced Data Quality Checker Test Suite
============================================================
This test suite demonstrates how to fix irregular interval issues
that cause data quality warnings in your feature engineering pipeline.

🧪 TEST 1: Basic Validation (No Fixing)
============================================================
🔧 Creating test data with irregular intervals...
✅ Created test data with 1000 records
   Date range: 2024-01-01 09:00:00 to 2024-01-02 06:40:00

🔍 Analyzing interval issues...
📊 Interval Analysis:
   Expected interval: 0 days 00:01:00
   Total intervals: 999
   Irregular intervals: 6 (0.006)
   Coefficient of variation: 0.276
   Tolerance: ±9.0s

📊 Validation Results:
   Validation passed: True
   Quality score: 0.950
   Warnings: 2
   Critical issues: 0

⚠️ Warnings:
   - Moderate time interval variability (CV: 0.276, irregular: 0.6%) may affect multi-timeframe feature generation
   - Scattered irregular timestamp intervals: 0.6% (threshold: 0.1%) - may affect multi-timeframe feature generation

🧪 TEST 2: Auto-Fix Irregular Intervals
============================================================
📊 Before fixing:
🔍 Analyzing interval issues...
📊 Interval Analysis:
   Expected interval: 0 days 00:01:00
   Total intervals: 999
   Irregular intervals: 6 (0.006)
   Coefficient of variation: 0.276
   Tolerance: ±9.0s

🔧 Auto-fixing irregular intervals for run_step (ratio: 0.006, CV: 0.276)
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

📊 After fixing:
🔍 Analyzing interval issues...
📊 Interval Analysis:
   Expected interval: 0 days 00:01:00
   Total intervals: 999
   Irregular intervals: 0 (0.000)
   Coefficient of variation: 0.000
   Tolerance: ±9.0s

✅ Improvement:
   Irregular ratio: 0.006 → 0.000
   CV: 0.276 → 0.000
   Records: 1000 → 1000

✅ All tests completed successfully!
============================================================

📋 Summary:
   - The enhanced data quality checker can automatically detect and fix irregular intervals
   - It uses intelligent gap handling: resample → re-add original → forward-fill small gaps → download large gaps
   - The @auto_fix_data_quality_issues decorator can be used to automatically fix issues in existing functions
   - This should eliminate the warnings you're seeing about irregular intervals

🔧 Usage in your code:
   from src.training.steps.raw_data_quality_checker import auto_fix_data_quality_issues
   @auto_fix_data_quality_issues
   def analyze_patterns(data, symbol, exchange):
       # Your existing code here
       pass

📝 Note: Run this test from the project root directory:
   python test_enhanced_data_quality_fix.py
```

## Troubleshooting

### Import Errors

If you get import errors, make sure:
1. You're running from the project root directory
2. The `src/` directory exists and contains the required modules
3. All dependencies are installed

### Data-Related Errors

If you get data-related errors:
1. Check that the project structure is correct
2. Ensure all required data directories exist
3. Verify that the test can create temporary test data

### Unexpected Errors

For unexpected errors:
1. Check the full traceback for details
2. Ensure all dependencies are properly installed
3. Verify Python version compatibility

## Integration

This test demonstrates how the enhanced data quality checker integrates with your existing pipeline:

1. **Automatic Detection**: Identifies irregular intervals automatically
2. **Intelligent Fixing**: Applies the enhanced preprocessing strategy
3. **Quality Improvement**: Measures and reports improvements
4. **Decorator Usage**: Shows how to use the `@auto_fix_data_quality_issues` decorator

The test validates that the solution works as intended and provides a working example for integration into your own code.