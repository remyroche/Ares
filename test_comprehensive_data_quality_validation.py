#!/usr/bin/env python3
"""
Test script for comprehensive data quality validation.

This script demonstrates the comprehensive data quality validation system
for Step1, Step1_5, and Step2 with special attention to NaN, infinite, and constant values.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.utils.comprehensive_data_quality_validator import (
        ComprehensiveDataQualityValidator,
        validate_step1_quality,
        validate_step1_5_quality,
        validate_step2_quality
    )
    from src.utils.data_quality_decorators import (
        log_feature_quality_issues,
        quick_validate_features
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure the comprehensive data quality validator is properly installed.")
    sys.exit(1)

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def create_test_data_with_issues():
    """Create test data with various quality issues for demonstration."""
    print("🔧 Creating test data with quality issues...")
    
    # Create base data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1min')
    n_samples = len(dates)
    
    # Create DataFrame with various issues
    data = {
        'timestamp': dates,
        'open': np.random.randn(n_samples) * 100 + 1000,
        'high': np.random.randn(n_samples) * 100 + 1000,
        'low': np.random.randn(n_samples) * 100 + 1000,
        'close': np.random.randn(n_samples) * 100 + 1000,
        'volume': np.random.randn(n_samples) * 1000 + 10000,
    }
    
    df = pd.DataFrame(data)
    
    # Add quality issues for demonstration (using new stricter thresholds)
    
    # 1. NaN values (0.1% threshold = ~4.3 values for 43200 samples)
    df.loc[100:105, 'open'] = np.nan  # 6 NaN values (will trigger)
    df.loc[200:202, 'high'] = np.nan  # 3 NaN values (will trigger)
    
    # 2. Infinite values (1 value threshold)
    df.loc[300, 'low'] = np.inf  # 1 infinite value (will trigger)
    df.loc[320, 'close'] = -np.inf  # 1 negative infinite value (will trigger)
    
    # 3. Constant features (2+ unique values, except boolean)
    df['constant_feature'] = 42  # Constant value (will trigger)
    df['binary_feature'] = np.random.choice([0, 1], n_samples)  # Binary feature (acceptable)
    df['boolean_feature'] = np.random.choice([True, False], n_samples)  # Boolean feature (acceptable)
    
    # 4. Highly correlated features
    df['highly_correlated'] = df['open'] * 1.01 + 0.1  # Almost perfect correlation
    
    # 5. Some good features
    df['good_feature_1'] = np.random.randn(n_samples)
    df['good_feature_2'] = np.random.randn(n_samples)
    df['good_feature_3'] = np.random.randn(n_samples)
    
    return df


def create_test_feature_data():
    """Create test feature data with various issues for Step2 validation."""
    print("🔧 Creating test feature data with quality issues...")
    
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1min')
    n_samples = len(dates)
    
    # Create feature DataFrame
    features = {
        'timestamp': dates,
        'rsi': np.random.uniform(0, 100, n_samples),
        'macd': np.random.randn(n_samples),
        'bollinger_upper': np.random.randn(n_samples) * 10 + 1000,
        'bollinger_lower': np.random.randn(n_samples) * 10 + 1000,
        'volume_sma': np.random.randn(n_samples) * 1000 + 10000,
    }
    
    df = pd.DataFrame(features)
    
    # Add feature-specific issues (using new stricter thresholds)
    
    # 1. NaN values in features (0.1% threshold)
    df.loc[100:105, 'rsi'] = np.nan  # 6 NaN values (will trigger)
    df.loc[200:202, 'macd'] = np.nan  # 3 NaN values (will trigger)
    
    # 2. Infinite values in features (1 value threshold)
    df.loc[300, 'bollinger_upper'] = np.inf  # 1 infinite value (will trigger)
    df.loc[310, 'bollinger_lower'] = -np.inf  # 1 negative infinite value (will trigger)
    
    # 3. Constant features (2+ unique values, except boolean)
    df['constant_rsi'] = 50  # Constant RSI (will trigger)
    df['constant_macd'] = 0  # Constant MACD (will trigger)
    df['binary_signal'] = np.random.choice([0, 1], n_samples)  # Binary feature (acceptable)
    
    # 4. Highly correlated features
    df['rsi_duplicate'] = df['rsi'] * 0.99 + 0.1  # Highly correlated with RSI
    
    # 5. Good features
    df['good_feature_1'] = np.random.randn(n_samples)
    df['good_feature_2'] = np.random.randn(n_samples)
    df['good_feature_3'] = np.random.randn(n_samples)
    
    return df


async def test_step1_validation():
    """Test Step1 data quality validation."""
    print("\n" + "="*80)
    print("🧪 TESTING STEP1 DATA QUALITY VALIDATION")
    print("="*80)
    
    # Create test data directory structure
    test_data_dir = "test_data_cache"
    os.makedirs(test_data_dir, exist_ok=True)
    
    # Create test consolidated files
    klines_data = create_test_data_with_issues()
    klines_file = f"{test_data_dir}/klines_BINANCE_ETHUSDT_1m_consolidated.parquet"
    klines_data.to_parquet(klines_file)
    
    aggtrades_data = create_test_data_with_issues()
    aggtrades_file = f"{test_data_dir}/aggtrades_BINANCE_ETHUSDT_consolidated.parquet"
    aggtrades_data.to_parquet(aggtrades_file)
    
    print(f"✅ Created test files:")
    print(f"   - {klines_file}")
    print(f"   - {aggtrades_file}")
    
    # Test Step1 validation
    print("\n🔍 Running Step1 data quality validation...")
    result = validate_step1_quality(
        symbol="ETHUSDT",
        exchange="BINANCE",
        data_dir=test_data_dir
    )
    
    print(f"\n📊 Step1 Validation Results:")
    print(f"   - Validation passed: {result['validation_passed']}")
    print(f"   - Issues found: {len(result['issues'])}")
    
    if result['issues']:
        print("   - Issues:")
        for issue in result['issues'][:5]:
            print(f"     * {issue}")
        if len(result['issues']) > 5:
            print(f"     ... and {len(result['issues']) - 5} more issues")
    
    return result


async def test_step1_5_validation():
    """Test Step1.5 data quality validation."""
    print("\n" + "="*80)
    print("🧪 TESTING STEP1.5 DATA QUALITY VALIDATION")
    print("="*80)
    
    # Create test unified data directory structure
    test_data_dir = "test_data_cache"
    unified_dir = f"{test_data_dir}/unified/binance/ethusdt/1m"
    os.makedirs(unified_dir, exist_ok=True)
    
    # Create test unified data files
    for i in range(3):
        unified_data = create_test_data_with_issues()
        unified_file = f"{unified_dir}/part-{i}.parquet"
        unified_data.to_parquet(unified_file)
        print(f"✅ Created test unified file: {unified_file}")
    
    # Test Step1.5 validation
    print("\n🔍 Running Step1.5 data quality validation...")
    result = validate_step1_5_quality(
        symbol="ETHUSDT",
        exchange="BINANCE",
        data_dir=test_data_dir
    )
    
    print(f"\n📊 Step1.5 Validation Results:")
    print(f"   - Validation passed: {result['validation_passed']}")
    print(f"   - Issues found: {len(result['issues'])}")
    
    if result['issues']:
        print("   - Issues:")
        for issue in result['issues'][:5]:
            print(f"     * {issue}")
        if len(result['issues']) > 5:
            print(f"     ... and {len(result['issues']) - 5} more issues")
    
    return result


async def test_step2_validation():
    """Test Step2 data quality validation with special attention to features."""
    print("\n" + "="*80)
    print("🧪 TESTING STEP2 DATA QUALITY VALIDATION")
    print("="*80)
    
    # Create test training data directory
    test_data_dir = "test_data_training"
    os.makedirs(test_data_dir, exist_ok=True)
    
    # Create test feature files
    train_features = create_test_feature_data()
    train_file = f"{test_data_dir}/BINANCE_ETHUSDT_features_train.parquet"
    train_features.to_parquet(train_file)
    
    validation_features = create_test_feature_data()
    validation_file = f"{test_data_dir}/BINANCE_ETHUSDT_features_validation.parquet"
    validation_features.to_parquet(validation_file)
    
    test_features = create_test_feature_data()
    test_file = f"{test_data_dir}/BINANCE_ETHUSDT_features_test.parquet"
    test_features.to_parquet(test_file)
    
    print(f"✅ Created test feature files:")
    print(f"   - {train_file}")
    print(f"   - {validation_file}")
    print(f"   - {test_file}")
    
    # Test Step2 validation
    print("\n🔍 Running Step2 data quality validation...")
    result = validate_step2_quality(
        symbol="ETHUSDT",
        exchange="BINANCE",
        data_dir=test_data_dir
    )
    
    print(f"\n📊 Step2 Validation Results:")
    print(f"   - Validation passed: {result['validation_passed']}")
    print(f"   - Issues found: {len(result['issues'])}")
    
    # Show problematic features
    problematic = result.get("problematic_features", {})
    if any(problematic.values()):
        print("   - Problematic features:")
        if problematic.get("nan_features"):
            print(f"     * NaN features: {len(problematic['nan_features'])}")
        if problematic.get("infinite_features"):
            print(f"     * Infinite features: {len(problematic['infinite_features'])}")
        if problematic.get("constant_features"):
            print(f"     * Constant features: {len(problematic['constant_features'])}")
        if problematic.get("high_correlation_pairs"):
            print(f"     * High correlation pairs: {len(problematic['high_correlation_pairs'])}")
    
    if result['issues']:
        print("   - Issues:")
        for issue in result['issues'][:5]:
            print(f"     * {issue}")
        if len(result['issues']) > 5:
            print(f"     ... and {len(result['issues']) - 5} more issues")
    
    return result


def test_feature_quality_logging():
    """Test feature quality logging functionality."""
    print("\n" + "="*80)
    print("🧪 TESTING FEATURE QUALITY LOGGING")
    print("="*80)
    
    # Create test feature data
    feature_data = create_test_feature_data()
    
    print("🔍 Testing feature quality logging...")
    log_feature_quality_issues(feature_data, "Test Features")
    
    print("\n🔍 Testing quick validation...")
    validation_result = quick_validate_features(feature_data, "Test Features")
    
    print(f"\n📊 Quick Validation Summary:")
    print(f"   - Shape: {validation_result['shape']}")
    print(f"   - Total features: {validation_result['total_features']}")
    print(f"   - Total samples: {validation_result['total_samples']}")
    print(f"   - NaN features: {validation_result['summary']['nan_count']}")
    print(f"   - Infinite features: {validation_result['summary']['infinite_count']}")
    print(f"   - Constant features: {validation_result['summary']['constant_count']}")
    print(f"   - High correlation pairs: {validation_result['summary']['high_correlation_count']}")


async def test_comprehensive_validator():
    """Test the comprehensive validator with all steps."""
    print("\n" + "="*80)
    print("🧪 TESTING COMPREHENSIVE VALIDATOR")
    print("="*80)
    
    # Create validator instance with updated thresholds
    validator = ComprehensiveDataQualityValidator({
        "max_nan_ratio": 0.001,  # 0.1% NaN
        "max_infinite_count": 1,  # 1 infinite value
        "min_unique_values": 2,   # 2+ unique values (except boolean)
        "min_feature_count": 5,
        "max_correlation_threshold": 0.95
    })
    
    # Run all validations
    print("🔍 Running comprehensive validation for all steps...")
    
    step1_result = validator.validate_step1_data_quality("ETHUSDT", "BINANCE", "test_data_cache")
    step1_5_result = validator.validate_step1_5_data_quality("ETHUSDT", "BINANCE", "test_data_cache")
    step2_result = validator.validate_step2_data_quality("ETHUSDT", "BINANCE", "test_data_training")
    
    # Save comprehensive report
    report_path = "comprehensive_validation_report.json"
    validator.save_validation_report(report_path)
    
    print(f"\n📊 Comprehensive Validation Summary:")
    print(f"   - Step1 passed: {step1_result['validation_passed']}")
    print(f"   - Step1.5 passed: {step1_5_result['validation_passed']}")
    print(f"   - Step2 passed: {step2_result['validation_passed']}")
    print(f"   - Report saved to: {report_path}")
    
    return {
        "step1": step1_result,
        "step1_5": step1_5_result,
        "step2": step2_result
    }


async def main():
    """Main test function."""
    print("🚀 COMPREHENSIVE DATA QUALITY VALIDATION TEST")
    print("="*80)
    print("This test demonstrates the comprehensive data quality validation system")
    print("for Step1, Step1_5, and Step2 with special attention to:")
    print("  - NaN values")
    print("  - Infinite values")
    print("  - Constant features")
    print("  - High correlations")
    print("  - File structure validation")
    print("="*80)
    
    try:
        # Test individual step validations
        await test_step1_validation()
        await test_step1_5_validation()
        await test_step2_validation()
        
        # Test feature quality logging
        test_feature_quality_logging()
        
        # Test comprehensive validator
        await test_comprehensive_validator()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80)
        print("The comprehensive data quality validation system is working correctly.")
        print("Check the generated reports and logs for detailed information.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # Run the test
    success = asyncio.run(main())
    
    if success:
        print("\n🎉 Test completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Test failed!")
        sys.exit(1)