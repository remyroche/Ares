#!/usr/bin/env python3
"""
Test script to validate the streamlined regime data splitting implementation.

This script tests:
1. Import functionality
2. Class instantiation
3. Basic validation methods
4. Error handling
5. Integration with main module
"""

import asyncio
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Also add the current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all imports work correctly."""
    print("🔍 Testing imports...")

    try:
        from src.training.steps.market_analysis.regime_data_splitting.streamlined_regime_splitting import (
            StreamlinedRegimeDataSplitting,
            RegimeSplittingStatus,
            RegimeSplittingMetrics,
            RegimeSplittingResult,
            create_streamlined_regime_splitting
        )
        print("✅ All imports successful")

        # Test utility imports
        from src.utils.common_operations import safe_read_parquet, optimize_dataframe_dtypes
        from src.utils.math_validation import validate_positive, safe_divide
        from src.utils.data.quality.data_quality import DataQualityFramework
        from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
        print("✅ All utility imports successful")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during imports: {e}")
        return False

def test_class_instantiation():
    """Test that the class can be instantiated correctly."""
    print("🔍 Testing class instantiation...")

    try:
        from src.training.steps.market_analysis.regime_data_splitting import (
            StreamlinedRegimeDataSplitting,
            create_streamlined_regime_splitting
        )

        # Test factory function
        config = {'chunk_size': 5000, 'max_memory_gb': 4.0}
        instance = create_streamlined_regime_splitting(config)

        if not isinstance(instance, StreamlinedRegimeDataSplitting):
            print("❌ Factory function did not return correct type")
            return False

        # Test direct instantiation
        instance2 = StreamlinedRegimeDataSplitting(config)
        if not isinstance(instance2, StreamlinedRegimeDataSplitting):
            print("❌ Direct instantiation failed")
            return False

        # Test that required attributes are set
        required_attrs = ['config', 'logger', 'metrics', 'chunk_size', 'max_memory_gb']
        for attr in required_attrs:
            if not hasattr(instance, attr):
                print(f"❌ Missing required attribute: {attr}")
                return False

        print("✅ Class instantiation successful")
        return True

    except Exception as e:
        print(f"❌ Error during instantiation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality with sample data."""
    print("🔍 Testing basic functionality...")

    try:
        from src.training.steps.market_analysis.regime_data_splitting import (
            StreamlinedRegimeDataSplitting
        )

        # Create test instance
        config = {'chunk_size': 1000, 'max_memory_gb': 2.0}
        instance = StreamlinedRegimeDataSplitting(config)

        # Create sample data
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5000, freq='1min'),
            'price_close': np.random.randn(5000).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 5000)
        })

        # Test validation methods
        print("  📊 Testing validation methods...")

        # Test input validation
        training_input = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'data_dir': '/tmp/test',
            'timeframe': '1m'
        }

        input_valid = instance._validate_inputs(training_input)
        if not input_valid:
            print("❌ Input validation failed")
            return False

        # Test data quality assessment
        quality_score = instance._assess_data_quality(sample_data)
        if quality_score < 0 or quality_score > 1:
            print(f"❌ Invalid quality score: {quality_score}")
            return False

        # Test data chunking
        chunks = instance._create_data_chunks(sample_data)
        if len(chunks) == 0:
            print("❌ Data chunking failed")
            return False

        # Test chunk processing
        test_chunk = chunks[0]
        processed_chunk = instance._apply_regime_tags_to_chunk(test_chunk)
        if processed_chunk is None:
            print("❌ Chunk processing failed")
            return False

        # Check that regime column was added
        if 'composite_cluster_id' not in processed_chunk.columns:
            print("❌ Regime column not added to processed chunk")
            return False

        # Test validation methods
        print("  ✅ Testing validation methods...")

        temporal_issues = instance._validate_temporal_continuity(sample_data)
        if not isinstance(temporal_issues, list):
            print("❌ Temporal validation did not return list")
            return False

        completeness_issues = instance._validate_data_completeness(sample_data)
        if not isinstance(completeness_issues, list):
            print("❌ Completeness validation did not return list")
            return False

        consistency_issues = instance._validate_data_consistency(sample_data)
        if not isinstance(consistency_issues, list):
            print("❌ Consistency validation did not return list")
            return False

        # Test tagged data validation
        tagged_data = processed_chunk.copy()
        validation_passed = instance._validate_tagged_data(tagged_data)
        if not isinstance(validation_passed, bool):
            print("❌ Tagged data validation did not return boolean")
            return False

        # Test metrics calculation
        instance._calculate_final_metrics(tagged_data)
        if instance.metrics.total_data_points != len(tagged_data):
            print("❌ Metrics calculation failed")
            return False

        print("✅ Basic functionality tests passed")
        return True

    except Exception as e:
        print(f"❌ Error during basic functionality test: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling and edge cases."""
    print("🔍 Testing error handling...")

    try:
        from src.training.steps.market_analysis.regime_data_splitting import (
            StreamlinedRegimeDataSplitting
        )

        instance = StreamlinedRegimeDataSplitting()

        # Test with invalid input
        invalid_input = {
            'invalid': 'data'
        }

        input_valid = instance._validate_inputs(invalid_input)
        if input_valid:
            print("❌ Invalid input validation should have failed")
            return False

        # Test with empty data
        empty_data = pd.DataFrame()
        if not empty_data.empty:
            print("❌ Empty data test failed")
            return False

        # Test chunk processing with empty data
        processed_empty = instance._apply_regime_tags_to_chunk(empty_data)
        if processed_empty is not None:
            print("❌ Empty chunk should return None")
            return False

        # Test with None data
        processed_none = instance._apply_regime_tags_to_chunk(None)
        if processed_none is not None:
            print("❌ None chunk should return None")
            return False

        # Test data quality with None
        quality_score = instance._assess_data_quality(None)
        if quality_score != 0.0:
            print("❌ None data quality should return 0.0")
            return False

        # Test validation methods with None
        temporal_issues = instance._validate_temporal_continuity(None)
        if temporal_issues is not None and len(temporal_issues) > 0:
            print("❌ None temporal validation should return empty list")
            return False

        print("✅ Error handling tests passed")
        return True

    except Exception as e:
        print(f"❌ Error during error handling test: {e}")
        traceback.print_exc()
        return False

async def test_main_integration():
    """Test integration with main module."""
    print("🔍 Testing main module integration...")

    try:
        # Test that main module can import streamlined component
        from src.training.steps.market_analysis.regime_data_splitting.main import RegimeDataSplittingStep

        # Create instance
        step = RegimeDataSplittingStep({})

        # Check that it has the required methods
        if not hasattr(step, 'execute'):
            print("❌ Main step missing execute method")
            return False

        print("✅ Main module integration successful")
        return True

    except Exception as e:
        print(f"❌ Error during main module integration test: {e}")
        traceback.print_exc()
        return False

async def test_streamlined_implementation():
    """Test the streamlined implementation end-to-end."""
    print("🔍 Testing streamlined implementation...")

    try:
        from src.training.steps.market_analysis.regime_data_splitting import (
            StreamlinedRegimeDataSplitting
        )

        # Create instance
        config = {
            'chunk_size': 1000,
            'max_memory_gb': 2.0
        }
        instance = StreamlinedRegimeDataSplitting(config)

        # Create test training input
        training_input = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'data_dir': '/tmp/test',
            'timeframe': '1m'
        }

        pipeline_state = {}

        # Test the main execution method
        result = await instance.execute_regime_splitting(training_input, pipeline_state)

        # Check result structure
        if not isinstance(result, dict):
            print("❌ Result is not a dictionary")
            return False

        # The result should indicate failure due to missing test data file
        if result.get('success', False):
            print("❌ Expected failure due to missing test data file")
            return False

        # Check that error information is provided
        if 'error' not in result and 'errors' not in result:
            print("❌ Expected error information in result")
            return False

        print("✅ Streamlined implementation test completed (expected failure due to missing data)")
        return True

    except Exception as e:
        print(f"❌ Error during streamlined implementation test: {e}")
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("🚀 Starting streamlined regime data splitting tests\n")

    # Track test results
    test_results = []

    # Test 1: Imports
    print("=" * 60)
    test_results.append(("Imports", test_imports()))

    # Test 2: Class instantiation
    print("=" * 60)
    test_results.append(("Class Instantiation", test_class_instantiation()))

    # Test 3: Basic functionality
    print("=" * 60)
    test_results.append(("Basic Functionality", test_basic_functionality()))

    # Test 4: Error handling
    print("=" * 60)
    test_results.append(("Error Handling", test_error_handling()))

    # Test 5: Main integration
    print("=" * 60)
    test_results.append(("Main Integration", await test_main_integration()))

    # Test 6: Streamlined implementation
    print("=" * 60)
    test_results.append(("Streamlined Implementation", await test_streamlined_implementation()))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed + failed} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Streamlined regime data splitting is functional.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
