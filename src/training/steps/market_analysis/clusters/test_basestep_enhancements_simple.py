"""
Simple test script for BaseStep enhancements in market analysis steps.

This script tests that the enhanced steps can be imported and instantiated correctly.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def test_imports():
    """Test that all enhanced steps can be imported."""
    print("🧪 Testing imports...")
    
    try:
        # Test BaseStep import
        from src.training.steps.base_step import BaseStep
        print("✅ BaseStep import - PASSED")
        
        # Test enhanced step imports
        from src.training.steps.market_analysis.clusters.step1_feature_preparation_data_driven import DataDrivenFeaturePreparationStep
        print("✅ Step1 import - PASSED")
        
        from src.training.steps.market_analysis.clusters.step2_initial_clustering import InitialClusteringStep
        print("✅ Step2 import - PASSED")
        
        from src.training.steps.market_analysis.clusters.step8_validation import ValidationStep
        print("✅ Step8 import - PASSED")
        
        from src.training.steps.market_analysis.clusters.step9_results_consolidation import ResultsConsolidationStep
        print("✅ Step9 import - PASSED")
        
        from src.training.steps.market_analysis.clusters.step10_comprehensive_reporting import ComprehensiveReporter
        print("✅ Step10 import - PASSED")
        
        from src.training.steps.market_analysis.clusters.shared_utils import MetricsCalculator
        print("✅ Shared utils import - PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test - FAILED: {e}")
        return False


def test_instantiation():
    """Test that all enhanced steps can be instantiated."""
    print("🧪 Testing instantiation...")
    
    try:
        # Test BaseStep instantiation
        from src.training.steps.base_step import BaseStep
        basestep = BaseStep("test_step")
        print("✅ BaseStep instantiation - PASSED")
        
        # Test enhanced step instantiation
        from src.training.steps.market_analysis.clusters.step1_feature_preparation_data_driven import DataDrivenFeaturePreparationStep
        step1 = DataDrivenFeaturePreparationStep(verbose=True)
        print("✅ Step1 instantiation - PASSED")
        
        from src.training.steps.market_analysis.clusters.step2_initial_clustering import InitialClusteringStep
        step2 = InitialClusteringStep(verbose=True)
        print("✅ Step2 instantiation - PASSED")
        
        from src.training.steps.market_analysis.clusters.step8_validation import ValidationStep
        step8 = ValidationStep(verbose=True)
        print("✅ Step8 instantiation - PASSED")
        
        from src.training.steps.market_analysis.clusters.step9_results_consolidation import ResultsConsolidationStep
        step9 = ResultsConsolidationStep(verbose=True)
        print("✅ Step9 instantiation - PASSED")
        
        from src.training.steps.market_analysis.clusters.step10_comprehensive_reporting import ComprehensiveReporter
        step10 = ComprehensiveReporter(verbose=True)
        print("✅ Step10 instantiation - PASSED")
        
        from src.training.steps.market_analysis.clusters.shared_utils import MetricsCalculator
        calculator = MetricsCalculator()
        print("✅ Shared utils instantiation - PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Instantiation test - FAILED: {e}")
        return False


def test_basestep_methods():
    """Test that BaseStep methods are available in enhanced steps."""
    print("🧪 Testing BaseStep methods...")
    
    try:
        from src.training.steps.market_analysis.clusters.step1_feature_preparation_data_driven import DataDrivenFeaturePreparationStep
        
        step = DataDrivenFeaturePreparationStep(verbose=True)
        
        # Test BaseStep methods
        assert hasattr(step, '_safe_json_save'), "Should have _safe_json_save method"
        assert hasattr(step, '_safe_json_load'), "Should have _safe_json_load method"
        assert hasattr(step, '_safe_divide'), "Should have _safe_divide method"
        assert hasattr(step, '_validate_finite'), "Should have _validate_finite method"
        assert hasattr(step, '_validate_dataframe_columns'), "Should have _validate_dataframe_columns method"
        assert hasattr(step, '_safe_dataframe_operation'), "Should have _safe_dataframe_operation method"
        assert hasattr(step, '_get_availability_status'), "Should have _get_availability_status method"
        assert hasattr(step, '_save_dataframe'), "Should have _save_dataframe method"
        assert hasattr(step, '_save_metadata'), "Should have _save_metadata method"
        assert hasattr(step, '_ensure_directory'), "Should have _ensure_directory method"
        
        print("✅ BaseStep methods - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ BaseStep methods test - FAILED: {e}")
        return False


def test_utility_availability():
    """Test that utility availability checking works."""
    print("🧪 Testing utility availability...")
    
    try:
        from src.training.steps.market_analysis.clusters.step1_feature_preparation_data_driven import DataDrivenFeaturePreparationStep
        
        step = DataDrivenFeaturePreparationStep(verbose=True)
        
        # Test utility availability checking
        availability = step._get_availability_status()
        assert isinstance(availability, dict), "Availability should be a dictionary"
        assert len(availability) > 0, "Should have some utility availability info"
        
        print(f"✅ Utility availability - PASSED (found {len(availability)} utilities)")
        return True
        
    except Exception as e:
        print(f"❌ Utility availability test - FAILED: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("🚀 Starting BaseStep Enhancement Tests...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_instantiation,
        test_basestep_methods,
        test_utility_availability
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("=" * 50)
    print("📊 Test Results Summary:")
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {i+1}. {test.__name__}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! BaseStep enhancements are working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    # Run the tests
    success = run_all_tests()
    sys.exit(0 if success else 1)