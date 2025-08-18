#!/usr/bin/env python3
"""
Test script for enhanced data quality checker
Demonstrates how to fix irregular interval issues that cause data quality warnings.

Note: Run this script from the project root directory:
    python test_enhanced_data_quality_fix.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Import the modules directly (run from project root)
from src.training.steps.raw_data_quality_checker import (
    RawDataQualityChecker,
    validate_raw_data_quality,
    fix_irregular_intervals_automatically,
    validate_and_fix_data_quality_issues,
    enhanced_preprocess_market_data,
    auto_fix_data_quality_issues
)


def create_test_data_with_irregular_intervals():
    """
    Create test data with irregular intervals similar to the issues you're experiencing.
    """
    print("🔧 Creating test data with irregular intervals...")
    
    # Create base timestamps with 1-minute intervals
    base_start = datetime(2024, 1, 1, 9, 0, 0)
    base_timestamps = [base_start + timedelta(minutes=i) for i in range(1000)]
    
    # Introduce irregular intervals (similar to your 0.6% irregular ratio)
    irregular_indices = np.random.choice(len(base_timestamps), size=int(len(base_timestamps) * 0.006), replace=False)
    
    # Create irregular timestamps
    timestamps = []
    for i, ts in enumerate(base_timestamps):
        if i in irregular_indices:
            # Add some irregularity (random offset between 30-90 seconds)
            offset = np.random.randint(30, 90)
            irregular_ts = ts + timedelta(seconds=offset)
            timestamps.append(irregular_ts)
        else:
            timestamps.append(ts)
    
    # Sort timestamps
    timestamps.sort()
    
    # Create OHLCV data
    np.random.seed(42)  # For reproducible results
    data = pd.DataFrame({
        'open': np.random.uniform(100, 200, len(timestamps)),
        'high': np.random.uniform(100, 200, len(timestamps)),
        'low': np.random.uniform(100, 200, len(timestamps)),
        'close': np.random.uniform(100, 200, len(timestamps)),
        'volume': np.random.uniform(1000, 10000, len(timestamps))
    }, index=timestamps)
    
    # Ensure OHLC consistency
    data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
    data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)
    
    print(f"✅ Created test data with {len(data)} records")
    print(f"   Date range: {data.index.min()} to {data.index.max()}")
    
    return data


def analyze_interval_issues(data):
    """
    Analyze interval issues in the data.
    """
    print("\n🔍 Analyzing interval issues...")
    
    time_diffs = data.index.to_series().diff().dropna()
    if len(time_diffs) == 0:
        print("❌ No time differences found")
        return
    
    expected_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
    tolerance_percentage = 0.15
    tolerance_seconds = expected_interval.total_seconds() * tolerance_percentage
    
    irregular_intervals = time_diffs[
        abs(time_diffs - expected_interval) > pd.Timedelta(seconds=tolerance_seconds)
    ]
    irregular_ratio = len(irregular_intervals) / len(time_diffs)
    
    # Calculate coefficient of variation
    time_diffs_seconds = time_diffs.dt.total_seconds()
    mean_interval = time_diffs_seconds.mean()
    std_interval = time_diffs_seconds.std()
    cv = std_interval / mean_interval if mean_interval > 0 else 0
    
    print(f"📊 Interval Analysis:")
    print(f"   Expected interval: {expected_interval}")
    print(f"   Total intervals: {len(time_diffs)}")
    print(f"   Irregular intervals: {len(irregular_intervals)} ({irregular_ratio:.3f})")
    print(f"   Coefficient of variation: {cv:.3f}")
    print(f"   Tolerance: ±{tolerance_seconds:.1f}s")
    
    return {
        'expected_interval': expected_interval,
        'irregular_ratio': irregular_ratio,
        'cv': cv,
        'total_intervals': len(time_diffs),
        'irregular_intervals': len(irregular_intervals)
    }


def test_basic_validation():
    """
    Test basic validation without fixing.
    """
    print("\n" + "="*60)
    print("🧪 TEST 1: Basic Validation (No Fixing)")
    print("="*60)
    
    data = create_test_data_with_irregular_intervals()
    analysis = analyze_interval_issues(data)
    
    # Run validation
    results = validate_raw_data_quality(data, "BTCUSDT", "binance")
    
    print(f"\n📊 Validation Results:")
    print(f"   Validation passed: {results['validation_passed']}")
    print(f"   Quality score: {results['data_quality_score']:.3f}")
    print(f"   Warnings: {len(results['warnings'])}")
    print(f"   Critical issues: {len(results['critical_issues'])}")
    
    if results['warnings']:
        print(f"\n⚠️ Warnings:")
        for warning in results['warnings']:
            print(f"   - {warning}")
    
    return data, results


def test_auto_fix_irregular_intervals():
    """
    Test automatic fixing of irregular intervals.
    """
    print("\n" + "="*60)
    print("🧪 TEST 2: Auto-Fix Irregular Intervals")
    print("="*60)
    
    data = create_test_data_with_irregular_intervals()
    print(f"\n📊 Before fixing:")
    before_analysis = analyze_interval_issues(data)
    
    # Apply auto-fix
    fixed_data = fix_irregular_intervals_automatically(data, "BTCUSDT", "binance")
    
    print(f"\n📊 After fixing:")
    after_analysis = analyze_interval_issues(fixed_data)
    
    print(f"\n✅ Improvement:")
    print(f"   Irregular ratio: {before_analysis['irregular_ratio']:.3f} → {after_analysis['irregular_ratio']:.3f}")
    print(f"   CV: {before_analysis['cv']:.3f} → {after_analysis['cv']:.3f}")
    print(f"   Records: {len(data)} → {len(fixed_data)}")
    
    return fixed_data


def test_comprehensive_validation_and_fix():
    """
    Test comprehensive validation and fixing.
    """
    print("\n" + "="*60)
    print("🧪 TEST 3: Comprehensive Validation and Fix")
    print("="*60)
    
    data = create_test_data_with_irregular_intervals()
    
    # Run comprehensive validation and fixing
    fixed_data, results = validate_and_fix_data_quality_issues(data, "BTCUSDT", "binance")
    
    print(f"\n📊 Results:")
    print(f"   Validation passed: {results['validation_passed']}")
    print(f"   Quality score: {results['data_quality_score']:.3f}")
    print(f"   Warnings: {len(results['warnings'])}")
    
    if 'preprocessing_summary' in results:
        summary = results['preprocessing_summary']
        print(f"\n🔧 Preprocessing Summary:")
        print(f"   Irregular ratio before: {summary.get('irregular_ratio_before', 0):.3f}")
        print(f"   CV before: {summary.get('cv_before', 0):.3f}")
        print(f"   Quality improvement: {summary.get('quality_improvement', 0):.3f}")
        print(f"   Fixes applied: {summary.get('fixes_applied', [])}")
        print(f"   Shape: {summary.get('original_shape', (0,0))} → {summary.get('fixed_shape', (0,0))}")
    
    return fixed_data, results


def test_enhanced_preprocessing():
    """
    Test enhanced preprocessing with intelligent gap handling.
    """
    print("\n" + "="*60)
    print("🧪 TEST 4: Enhanced Preprocessing")
    print("="*60)
    
    data = create_test_data_with_irregular_intervals()
    
    # Add some gaps to test gap handling
    print("🔧 Adding gaps to test gap handling...")
    
    # Remove some random rows to create gaps
    gap_indices = np.random.choice(len(data), size=int(len(data) * 0.02), replace=False)
    data_with_gaps = data.drop(data.index[gap_indices])
    
    print(f"   Created {len(gap_indices)} gaps")
    print(f"   Records: {len(data)} → {len(data_with_gaps)}")
    
    # Apply enhanced preprocessing
    preprocessed_data = enhanced_preprocess_market_data(
        data_with_gaps, 
        "BTCUSDT", 
        "binance",
        expected_interval_seconds=60,
        max_forward_fill_seconds=10,
        download_missing_data=False  # Disable download for testing
    )
    
    print(f"\n📊 Preprocessing Results:")
    print(f"   Original shape: {data_with_gaps.shape}")
    print(f"   Preprocessed shape: {preprocessed_data.shape}")
    print(f"   Records added: {len(preprocessed_data) - len(data_with_gaps)}")
    
    # Analyze intervals after preprocessing
    print(f"\n📊 After preprocessing:")
    analyze_interval_issues(preprocessed_data)
    
    return preprocessed_data


def test_decorator_usage():
    """
    Test the decorator for automatic fixing.
    """
    print("\n" + "="*60)
    print("🧪 TEST 5: Decorator Usage")
    print("="*60)
    
    @auto_fix_data_quality_issues
    def analyze_patterns(data, symbol, exchange):
        """
        Example function that would normally trigger data quality warnings.
        """
        print(f"🔍 Analyzing patterns for {symbol} on {exchange}")
        print(f"   Data shape: {data.shape}")
        
        # Simulate some analysis
        time_diffs = data.index.to_series().diff().dropna()
        if len(time_diffs) > 0:
            expected_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
            irregular_intervals = time_diffs[time_diffs != expected_interval]
            irregular_ratio = len(irregular_intervals) / len(time_diffs)
            print(f"   Irregular ratio: {irregular_ratio:.3f}")
        
        return {"analysis_complete": True}
    
    # Test with irregular data
    data = create_test_data_with_irregular_intervals()
    
    print("📊 Before decorator (should show irregular intervals):")
    analyze_interval_issues(data)
    
    print(f"\n🔧 Running analysis with decorator (should auto-fix):")
    result = analyze_patterns(data, "BTCUSDT", "binance")
    
    print(f"\n✅ Analysis result: {result}")


def main():
    """
    Run all tests.
    """
    print("🚀 Enhanced Data Quality Checker Test Suite")
    print("="*60)
    print("This test suite demonstrates how to fix irregular interval issues")
    print("that cause data quality warnings in your feature engineering pipeline.")
    print()
    
    try:
        # Test 1: Basic validation
        data, results = test_basic_validation()
        
        # Test 2: Auto-fix irregular intervals
        fixed_data = test_auto_fix_irregular_intervals()
        
        # Test 3: Comprehensive validation and fix
        comp_fixed_data, comp_results = test_comprehensive_validation_and_fix()
        
        # Test 4: Enhanced preprocessing
        preprocessed_data = test_enhanced_preprocessing()
        
        # Test 5: Decorator usage
        test_decorator_usage()
        
        print("\n" + "="*60)
        print("✅ All tests completed successfully!")
        print("="*60)
        print("\n📋 Summary:")
        print("   - The enhanced data quality checker can automatically detect and fix irregular intervals")
        print("   - It uses intelligent gap handling: resample → re-add original → forward-fill small gaps → download large gaps")
        print("   - The @auto_fix_data_quality_issues decorator can be used to automatically fix issues in existing functions")
        print("   - This should eliminate the warnings you're seeing about irregular intervals")
        print("\n🔧 Usage in your code:")
print("   from src.training.steps.raw_data_quality_checker import auto_fix_data_quality_issues")
print("   @auto_fix_data_quality_issues")
print("   def analyze_patterns(data, symbol, exchange):")
print("       # Your existing code here")
print("       pass")
print("\n📝 Note: Run this test from the project root directory:")
print("   python test_enhanced_data_quality_fix.py")
        
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"\n❌ A data-related error occurred: {e}")
        print("   This might be due to missing data files or import issues.")
        print("   Make sure you're running from the project root directory.")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        print("   Please check the error details below and ensure all dependencies are installed.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()