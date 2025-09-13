#!/usr/bin/env python3
"""
Test Enhanced Duplicate Analysis Integration

This script tests the comprehensive duplicate analyzer integration
with klines_downloading_processing.py and DataQualityFramework.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_test_data_with_duplicates():
    """Create test DataFrame with various types of duplicates."""
    print("🔧 Creating test data with duplicates...")

    # Create base data
    timestamps = []
    data = []

    # Generate 100 unique records
    base_timestamp = 1640995200000  # 2022-01-01 00:00:00 UTC in milliseconds
    for i in range(100):
        timestamp = base_timestamp + (i * 60000)  # 1 minute intervals
        record = {
            'timestamp': timestamp,
            'open': 50000 + np.random.normal(0, 100),
            'high': 50100 + np.random.normal(0, 100),
            'low': 49900 + np.random.normal(0, 100),
            'close': 50050 + np.random.normal(0, 100),
            'volume': 100 + np.random.uniform(0, 200)
        }
        timestamps.append(timestamp)
        data.append(record)

    # Add TRUE DUPLICATES (identical records)
    print("  📝 Adding true duplicates...")
    for i in range(5):
        idx = np.random.randint(0, 50)
        data.append(data[idx].copy())  # Exact copy

    # Add FALSE DUPLICATES (same timestamp, different values)
    print("  📝 Adding false duplicates...")
    for i in range(3):
        idx = np.random.randint(50, 80)
        duplicate_record = data[idx].copy()
        # Modify values to create conflicts
        duplicate_record['close'] = duplicate_record['close'] * 1.05  # 5% different
        duplicate_record['volume'] = duplicate_record['volume'] * 0.8  # Different volume
        data.append(duplicate_record)

    # Add MIXED DUPLICATES (some same timestamp with mixed identical/different values)
    print("  📝 Adding mixed duplicates...")
    for i in range(2):
        idx = np.random.randint(80, 95)
        duplicate_record = data[idx].copy()
        # Keep some values same, change others
        duplicate_record['high'] = duplicate_record['high']  # Same
        duplicate_record['low'] = duplicate_record['low'] * 0.95  # Different
        data.append(duplicate_record)

    df = pd.DataFrame(data)
    print(f"✅ Created test data: {len(df)} records ({len(df) - 100} duplicates added)")
    return df


def test_comprehensive_duplicate_analyzer():
    """Test the comprehensive duplicate analyzer directly."""
    print("\n" + "="*60)
    print("🧪 TESTING COMPREHENSIVE DUPLICATE ANALYZER")
    print("="*60)

    try:
        from src.utils.data.quality.comprehensive_duplicate_analyzer import (
            ComprehensiveDuplicateAnalyzer,
            analyze_duplicates_comprehensive
        )

        # Create test data
        df = create_test_data_with_duplicates()

        # Test direct analyzer
        print("\n🔍 Testing direct analyzer...")
        analyzer = ComprehensiveDuplicateAnalyzer()
        result = analyzer.analyze_duplicates(df)

        print("📊 ANALYSIS RESULTS:")
        print(f"  Total duplicates: {result.total_duplicates}")
        print(f"  Duplicate groups: {len(result.duplicate_groups)}")
        print(f"  True duplicates: {result.true_duplicate_groups}")
        print(f"  False duplicate groups: {result.false_duplicate_groups}")
        print(f"  Mixed duplicates: {result.mixed_duplicate_groups}")

        if result.recommendations:
            print("💡 RECOMMENDATIONS:")
            for rec in result.recommendations:
                print(f"  • {rec}")

        # Test convenience function
        print("\n🔍 Testing convenience function...")
        result2 = analyze_duplicates_comprehensive(df)
        assert result.total_duplicates == result2.total_duplicates, "Results should match"

        print("✅ Direct analyzer tests passed!")

    except Exception as e:
        print(f"❌ Direct analyzer test failed: {e}")
        return False

    return True


def test_klines_pipeline_integration():
    """Test integration with klines_downloading_processing.py"""
    print("\n" + "="*60)
    print("🧪 TESTING KLINES PIPELINE INTEGRATION")
    print("="*60)

    try:
        from src.training.steps.data_collection.klines_downloading_processing import (
            KlinesDataQualityChecker,
            run_duplicate_analysis
        )

        # Create test data
        df = create_test_data_with_duplicates()

        # Test the checker class
        print("\n🔍 Testing KlinesDataQualityChecker...")
        checker = KlinesDataQualityChecker()

        # Create a temporary parquet file for testing
        temp_file = Path("/tmp/test_duplicate_data.parquet")
        df.to_parquet(temp_file, index=False)

        # Test duplicate checking method
        duplicate_results = checker._check_duplicate_timestamps([temp_file])

        print("📊 DUPLICATE ANALYSIS RESULTS:")
        print(f"  Files analyzed: {duplicate_results['total_files_analyzed']}")
        print(f"  Total duplicates: {duplicate_results['total_duplicate_records']}")
        print(f"  Duplicate groups: {duplicate_results['duplicate_groups']}")
        print(f"  True duplicates: {duplicate_results['true_duplicates']}")
        print(f"  False duplicates: {duplicate_results['false_duplicates']}")

        if duplicate_results['duplicate_issues']:
            print("⚠️ ISSUES FOUND:")
            for issue in duplicate_results['duplicate_issues']:
                print(f"  • {issue}")

        # Clean up
        temp_file.unlink()

        print("✅ Klines pipeline integration tests passed!")

    except Exception as e:
        print(f"❌ Klines pipeline integration test failed: {e}")
        return False

    return True


def test_data_quality_framework_integration():
    """Test integration with DataQualityFramework"""
    print("\n" + "="*60)
    print("🧪 TESTING DATA QUALITY FRAMEWORK INTEGRATION")
    print("="*60)

    try:
        from src.utils.data.quality.data_quality import (
            DataQualityFramework,
            validate_with_duplicate_analysis,
            check_duplicate_quality,
            analyze_duplicates_enhanced
        )

        # Create test data
        df = create_test_data_with_duplicates()

        # Test framework integration
        print("\n🔍 Testing DataQualityFramework...")
        framework = DataQualityFramework()
        result = framework.validate_dataframe_quality(df, "test_duplicate_analysis")

        print("📊 QUALITY VALIDATION RESULTS:")
        print(f"  Overall quality: {'✅ PASSED' if result.passed else '❌ FAILED'}")
        print(f"  Quality score: {result.quality_score:.2f}")
        print(f"  Issues: {len(result.issues)}")
        print(f"  Warnings: {len(result.warnings)}")

        # Check duplicate metrics
        duplicate_count = result.metrics.get('total_duplicate_records', 0)
        print(f"  Duplicate records: {duplicate_count}")
        print(f"  Duplicate groups: {result.metrics.get('duplicate_groups', 0)}")
        print(f"  True duplicates: {result.metrics.get('true_duplicates', 0)}")
        print(f"  False duplicates: {result.metrics.get('false_duplicates', 0)}")

        # Test convenience functions
        print("\n🔍 Testing convenience functions...")

        # Test enhanced analysis
        analysis_result = analyze_duplicates_enhanced(df)
        print(f"  Enhanced analysis: {analysis_result.total_duplicates} duplicates")

        # Test duplicate-focused quality check
        duplicate_info = check_duplicate_quality(df)
        print(f"  Duplicate quality check: {duplicate_info['duplicate_count']} duplicates")

        print("✅ DataQualityFramework integration tests passed!")

    except Exception as e:
        print(f"❌ DataQualityFramework integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_duplicate_manual_review():
    """Test duplicate manual review functionality (only strategy available)."""
    print("\n" + "="*60)
    print("🧪 TESTING DUPLICATE MANUAL REVIEW (ONLY AVAILABLE STRATEGY)")
    print("="*60)

    try:
        from src.utils.data.quality.comprehensive_duplicate_analyzer import (
            ComprehensiveDuplicateAnalyzer
        )

        # Create test data
        df = create_test_data_with_duplicates()
        original_count = len(df)

        print(f"Original data: {original_count} records")

        # Test manual review strategy (only available)
        analyzer = ComprehensiveDuplicateAnalyzer()

        print("\n🔍 Testing manual review strategy (only available option)")
        try:
            # This should fail with other strategies
            cleaned_df, resolution_summary = analyzer.resolve_duplicates(df.copy(), 'highest_volume')
            print("❌ ERROR: Should have failed with unsupported strategy")
            return False
        except ValueError as e:
            print(f"✅ Correctly rejected unsupported strategy: {e}")

        # Test the only supported strategy
        original_df, resolution_summary = analyzer.resolve_duplicates(df.copy(), 'manual_review')

        print(f"  Records flagged: {resolution_summary['records_flagged']}")
        print(f"  Final count: {len(original_df)} (unchanged)")
        print(f"  Groups requiring review: {resolution_summary['groups_processed']}")
        print(f"  Manual review items: {len(resolution_summary['manual_review_needed'])}")

        # Verify original data is unchanged
        if len(original_df) != original_count:
            print("❌ ERROR: Original data was modified - this should not happen")
            return False

        # Verify manual review items are present
        if not resolution_summary['manual_review_needed']:
            print("❌ ERROR: No manual review items generated")
            return False

        print("✅ Manual review functionality tests passed!")

    except Exception as e:
        print(f"❌ Manual review test failed: {e}")
        return False

    return True


def main():
    """Run all integration tests."""
    print("🚀 STARTING ENHANCED DUPLICATE ANALYSIS INTEGRATION TESTS")
    print("="*80)

    tests = [
        ("Comprehensive Duplicate Analyzer", test_comprehensive_duplicate_analyzer),
        ("Klines Pipeline Integration", test_klines_pipeline_integration),
        ("DataQualityFramework Integration", test_data_quality_framework_integration),
        ("Duplicate Manual Review", test_duplicate_manual_review)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: FAILED with exception: {e}")

    print("\n" + "="*80)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ Enhanced duplicate analysis is ready for production use.")
    else:
        print("⚠️ Some tests failed. Please review the integration.")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
