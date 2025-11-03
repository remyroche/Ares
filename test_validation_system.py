"""
Test script for quality score validation system

Demonstrates error handling and data validation capabilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Test the validator
from src.tactician.sr_levels.ml_quality.quality_data_validator import (
    QualityDataValidator, 
    DataQualityMonitor,
    validate_training_data
)


def test_validation_on_existing_data():
    """Test validation on existing training data."""
    print("\n" + "="*80)
    print("🧪 TESTING VALIDATION ON EXISTING DATA")
    print("="*80)
    
    # Load existing data
    data_path = Path('data_cache/sr_ml_training/multi_timeframe/sr_quality_1h_ETHUSDT.parquet')
    
    if not data_path.exists():
        print(f"\n❌ Data not found at {data_path}")
        print("   Run: python3 validate_multi_timeframe_quality.py first")
        return None
    
    data = pd.read_parquet(data_path)
    print(f"\n✅ Loaded {len(data):,} samples from {data_path.name}")
    
    # Test validator
    validator = QualityDataValidator(strict_mode=False)
    report = validator.validate_training_data(data, timeframe='1h')
    
    # Save report
    output_dir = Path('analysis_output/validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    validator.save_validation_report(report, output_dir / 'validation_report_1h.json')
    
    # Generate quality report
    quality_report = validator.generate_quality_report(
        data, 
        output_path=output_dir / 'quality_report_1h.txt'
    )
    
    return report


def test_validation_with_bad_data():
    """Test validation with intentionally bad data."""
    print("\n" + "="*80)
    print("🧪 TESTING VALIDATION WITH BAD DATA")
    print("="*80)
    
    # Create synthetic bad data
    bad_data = pd.DataFrame({
        'quality_score': [0.5, 0.7, np.nan, 0.9, np.inf, -0.1, 1.5],  # NaN, Inf, out of range
        'bounce_strength': [1.0, 1.0, 1.0, 0.95, 0.98, 0.97, 0.99],  # Saturated
        'hold_strength': [0.5, 0.6, 0.4, 0.7, 0.5, 0.6, 0.5],
        'trade_profit': [0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.4],
        'feature_strength': [0.5, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7],
        'feature_prominence': [0.4, 0.5, 0.6, 0.7, 0.4, 0.5, 0.6]
    })
    
    print(f"\n📊 Created synthetic bad data with {len(bad_data)} samples")
    print("   Issues included:")
    print("   • NaN values")
    print("   • Inf values")
    print("   • Out-of-range values")
    print("   • Saturated distributions")
    print("   • Insufficient samples")
    
    # Test with strict mode OFF (should pass with warnings)
    print("\n🔍 Testing with strict_mode=False...")
    validator = QualityDataValidator(strict_mode=False)
    report = validator.validate_training_data(bad_data)
    
    print(f"\n   Critical issues: {len(report['critical_issues'])}")
    print(f"   Warnings: {len(report['warnings'])}")
    print(f"   Validation passed: {report['validation_passed']}")
    
    # Test with strict mode ON (should raise exception)
    print("\n🔍 Testing with strict_mode=True...")
    try:
        validator_strict = QualityDataValidator(strict_mode=True)
        report_strict = validator_strict.validate_training_data(bad_data)
        print("   ⚠️  Unexpectedly passed (should have raised exception)")
    except ValueError as e:
        print(f"   ✅ Correctly raised ValueError: {str(e)[:80]}...")
    
    return report


def test_drift_detection():
    """Test drift detection."""
    print("\n" + "="*80)
    print("🧪 TESTING DRIFT DETECTION")
    print("="*80)
    
    # Load baseline data
    data_path = Path('data_cache/sr_ml_training/multi_timeframe/sr_quality_1h_ETHUSDT.parquet')
    
    if not data_path.exists():
        print(f"\n❌ No data available for drift testing")
        return None
    
    baseline_data = pd.read_parquet(data_path)
    print(f"\n✅ Loaded baseline: {len(baseline_data):,} samples")
    
    # Create slightly drifted data
    drifted_data = baseline_data.copy()
    drifted_data['bounce_strength'] = drifted_data['bounce_strength'] * 0.9  # 10% reduction
    drifted_data['quality_score'] = drifted_data['quality_score'] * 0.95
    
    print(f"✅ Created drifted data (bounce -10%, quality -5%)")
    
    # Test drift detection
    monitor = DataQualityMonitor(baseline_data=baseline_data)
    drift_report = monitor.detect_drift(drifted_data)
    
    print(f"\n📊 Drift Detection Results:")
    print(f"   Drift detected: {drift_report['drift_detected']}")
    print(f"   Drifted metrics: {drift_report['drifted_metrics']}")
    
    if drift_report['statistics']:
        print(f"\n   Details:")
        for metric, stats in drift_report['statistics'].items():
            if stats['drifted']:
                print(f"      {metric}:")
                print(f"         Baseline mean: {stats['baseline_mean']:.4f}")
                print(f"         Current mean: {stats['current_mean']:.4f}")
                print(f"         Change: {stats['mean_change']:.4f}")
                print(f"         p-value: {stats['p_value']:.4f}")
    
    return drift_report


def test_collection_monitoring():
    """Test collection metrics monitoring."""
    print("\n" + "="*80)
    print("🧪 TESTING COLLECTION MONITORING")
    print("="*80)
    
    # Load existing data
    data_path = Path('data_cache/sr_ml_training/multi_timeframe/sr_quality_1h_ETHUSDT.parquet')
    
    if not data_path.exists():
        print(f"\n❌ No data available for monitoring test")
        return None
    
    data = pd.read_parquet(data_path)
    
    # Simulate collection metrics
    monitor = DataQualityMonitor()
    
    metrics = monitor.track_collection_metrics(
        training_df=data,
        duration=25.5,  # Simulated duration
        timeframe='1h'
    )
    
    print(f"\n📊 Collection Metrics Tracked:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    return metrics


def test_quick_validation():
    """Test quick validation before training."""
    print("\n" + "="*80)
    print("🧪 TESTING QUICK VALIDATION")
    print("="*80)
    
    # Load data
    data_path = Path('data_cache/sr_ml_training/multi_timeframe/sr_quality_1h_ETHUSDT.parquet')
    
    if data_path.exists():
        data = pd.read_parquet(data_path)
        
        # Quick validation (convenience function)
        result = validate_training_data(data, timeframe='1h', strict=False)
        
        print(f"\n✅ Quick validation completed")
        print(f"   Passed: {result['validation_passed']}")
        print(f"   Critical issues: {len(result['critical_issues'])}")
        print(f"   Warnings: {len(result['warnings'])}")
    else:
        print(f"\n❌ No data available")


def main():
    """Run all validation tests."""
    print("\n" + "="*80)
    print("🔍 QUALITY SCORE VALIDATION SYSTEM - TEST SUITE")
    print("="*80)
    
    # Test 1: Validation on existing data
    test_validation_on_existing_data()
    
    # Test 2: Validation with bad data
    test_validation_with_bad_data()
    
    # Test 3: Drift detection
    test_drift_detection()
    
    # Test 4: Collection monitoring
    test_collection_monitoring()
    
    # Test 5: Quick validation
    test_quick_validation()
    
    print("\n" + "="*80)
    print("✅ ALL VALIDATION TESTS COMPLETE")
    print("="*80)
    print("\n📁 Validation reports saved to: analysis_output/validation/")
    print("\nKey outputs:")
    print("   • validation_report_1h.json")
    print("   • quality_report_1h.txt")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()

