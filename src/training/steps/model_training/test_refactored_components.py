"""Test script for refactored data preparation and quality components."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the refactored components
from data_preparation_components import DataFormatConverter, DataValidator, DataCleaner
from data_quality_components import QualityMetricsCalculator, DataIntegrityChecker, AnomalyDetector


def create_sample_data():
    """Create sample OHLCV data for testing."""
    # Generate timestamps
    start_date = datetime(2024, 1, 1)
    timestamps = pd.date_range(start=start_date, periods=1000, freq='1min')
    
    # Generate price data with some patterns
    np.random.seed(42)
    base_price = 100
    prices = base_price + np.cumsum(np.random.randn(1000) * 0.5)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'timestamp': timestamps.astype(np.int64) // 10**6,  # Convert to milliseconds
        'open': prices + np.random.randn(1000) * 0.1,
        'high': prices + np.abs(np.random.randn(1000) * 0.2),
        'low': prices - np.abs(np.random.randn(1000) * 0.2),
        'close': prices + np.random.randn(1000) * 0.1,
        'volume': np.random.exponential(1000, 1000)
    })
    
    # Ensure OHLC consistency
    data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
    data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)
    
    # Add some data quality issues for testing
    # Add missing values
    data.loc[50:55, 'volume'] = np.nan
    
    # Add duplicates
    data = pd.concat([data, data.iloc[100:105]], ignore_index=True)
    
    # Add an anomaly
    data.loc[500, 'close'] = data.loc[500, 'close'] * 10
    
    # Set datetime index
    data.index = pd.to_datetime(data['timestamp'], unit='ms')
    
    return data


def test_data_preparation_components():
    """Test the data preparation components."""
    print("=" * 80)
    print("Testing Data Preparation Components")
    print("=" * 80)
    
    # Create sample data
    data = create_sample_data()
    print(f"\nCreated sample data with shape: {data.shape}")
    print(f"Data has {data.isna().sum().sum()} missing values")
    print(f"Data has {data.index.duplicated().sum()} duplicate timestamps")
    
    # Test DataValidator
    print("\n1. Testing DataValidator:")
    validator = DataValidator()
    missing_info = validator.verify_missing_columns(data, data_type="klines")
    print(f"   - Verification passed: {missing_info['verification_passed']}")
    print(f"   - Missing required columns: {missing_info['missing_required']}")
    
    # Test DataCleaner
    print("\n2. Testing DataCleaner:")
    cleaner = DataCleaner()
    
    # Remove duplicates
    cleaned_data = cleaner.remove_duplicates(data, subset=['timestamp'])
    print(f"   - After removing duplicates: {len(cleaned_data)} rows")
    
    # Fill missing values
    cleaned_data = cleaner.fill_missing_values(cleaned_data)
    print(f"   - After filling missing values: {cleaned_data.isna().sum().sum()} missing")
    
    # Detect outliers
    outlier_data, outliers = cleaner.detect_outliers(cleaned_data, method="zscore", threshold=3.0)
    print(f"   - Detected outliers in columns: {list(outliers.keys())}")
    
    # Test DataFormatConverter
    print("\n3. Testing DataFormatConverter:")
    converter = DataFormatConverter()
    
    # Enforce schema
    formatted_data = converter.enforce_schema(cleaned_data, "klines")
    print(f"   - Schema enforced, data types:")
    for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
        if col in formatted_data.columns:
            print(f"     - {col}: {formatted_data[col].dtype}")
    
    return cleaned_data


def test_data_quality_components():
    """Test the data quality components."""
    print("\n" + "=" * 80)
    print("Testing Data Quality Components")
    print("=" * 80)
    
    # Create sample data with some issues
    data = create_sample_data()
    
    # Test QualityMetricsCalculator
    print("\n1. Testing QualityMetricsCalculator:")
    calculator = QualityMetricsCalculator()
    
    completeness = calculator.calculate_completeness_metrics(data)
    print(f"   - Overall completeness: {completeness['overall_completeness']:.2%}")
    
    consistency = calculator.calculate_consistency_metrics(data)
    print(f"   - Overall consistency: {consistency['overall_consistency']:.2%}")
    
    # Test DataIntegrityChecker
    print("\n2. Testing DataIntegrityChecker:")
    checker = DataIntegrityChecker()
    
    is_valid, results = checker.validate_data_integrity(data)
    print(f"   - Data integrity valid: {is_valid}")
    print(f"   - Critical issues: {len(results['critical_issues'])}")
    print(f"   - Warnings: {len(results['warnings'])}")
    
    # Test AnomalyDetector
    print("\n3. Testing AnomalyDetector:")
    detector = AnomalyDetector()
    
    anomaly_results = detector.detect_anomalies(data)
    print(f"   - Total anomalies detected: {anomaly_results['summary']['total_anomalies']}")
    print(f"   - Columns with anomalies: {anomaly_results['summary']['columns_with_anomalies']}")
    
    # Test volume anomaly detection
    volume_anomalies = detector.detect_volume_anomalies(data)
    print(f"   - Volume spikes: {len(volume_anomalies['volume_spikes'])}")
    print(f"   - Volume drops: {len(volume_anomalies['volume_drops'])}")
    
    # Generate quality report
    print("\n4. Testing Quality Report Generation:")
    report = calculator.generate_quality_report(data, "BTCUSDT", "BYBIT")
    print(f"   - Overall quality score: {report['overall_score']:.2f}")
    print(f"   - Recommendations: {len(report['recommendations'])}")
    for rec in report['recommendations']:
        print(f"     - {rec}")


def test_integration():
    """Test integration of components."""
    print("\n" + "=" * 80)
    print("Testing Component Integration")
    print("=" * 80)
    
    # Create sample data
    data = create_sample_data()
    
    # Clean data first
    cleaner = DataCleaner()
    cleaned_data = cleaner.remove_duplicates(data)
    cleaned_data = cleaner.fill_missing_values(cleaned_data)
    
    # Validate cleaned data
    validator = DataValidator()
    missing_info = validator.verify_missing_columns(cleaned_data, "klines")
    
    # Check integrity
    checker = DataIntegrityChecker()
    is_valid, integrity_results = checker.validate_data_integrity(cleaned_data)
    
    # Detect anomalies
    detector = AnomalyDetector()
    anomaly_results = detector.detect_anomalies(cleaned_data)
    
    # Calculate final quality score
    calculator = QualityMetricsCalculator()
    final_results = {
        "critical_issues": integrity_results.get("critical_issues", []),
        "warnings": integrity_results.get("warnings", []),
        "detailed_analysis": {
            "anomalies": anomaly_results
        }
    }
    
    quality_score = calculator.calculate_quality_score(final_results)
    
    print(f"\nIntegration Test Results:")
    print(f"  - Data cleaned: ✓")
    print(f"  - Data validated: {'✓' if missing_info['verification_passed'] else '✗'}")
    print(f"  - Integrity checked: {'✓' if is_valid else '✗'}")
    print(f"  - Anomalies detected: {anomaly_results['summary']['total_anomalies']}")
    print(f"  - Final quality score: {quality_score:.2f}")


if __name__ == "__main__":
    print("Testing Refactored Components")
    print("=" * 80)
    
    # Test data preparation components
    cleaned_data = test_data_preparation_components()
    
    # Test data quality components
    test_data_quality_components()
    
    # Test integration
    test_integration()
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)