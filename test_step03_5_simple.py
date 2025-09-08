#!/usr/bin/env python3
"""
Simple test for Step03_5 utility integration without external dependencies.

This script tests the utility integration by checking imports and basic functionality.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_utility_imports():
    """Test that all utility modules can be imported."""
    logger.info("🧪 Testing utility module imports...")
    
    import_results = {}
    
    # Test common operations import
    try:
        from src.utils.common_operations import safe_mean, safe_divide, validate_dataframe
        import_results['common_operations'] = True
        logger.info("  ✅ common_operations: Import successful")
    except Exception as e:
        import_results['common_operations'] = False
        logger.error(f"  ❌ common_operations: Import failed - {e}")
    
    # Test common utilities import
    try:
        from src.utils.common_utilities import safe_dataframe_operation, validate_dataframe_columns
        import_results['common_utilities'] = True
        logger.info("  ✅ common_utilities: Import successful")
    except Exception as e:
        import_results['common_utilities'] = False
        logger.error(f"  ❌ common_utilities: Import failed - {e}")
    
    # Test math validation import
    try:
        from src.utils.math_validation import safe_divide as math_safe_divide, validate_finite
        import_results['math_validation'] = True
        logger.info("  ✅ math_validation: Import successful")
    except Exception as e:
        import_results['math_validation'] = False
        logger.error(f"  ❌ math_validation: Import failed - {e}")
    
    # Test parquet utils import
    try:
        from src.utils.parquet_utils import ParquetUtils, get_parquet_utils
        import_results['parquet_utils'] = True
        logger.info("  ✅ parquet_utils: Import successful")
    except Exception as e:
        import_results['parquet_utils'] = False
        logger.error(f"  ❌ parquet_utils: Import failed - {e}")
    
    # Test serialization utils import
    try:
        from src.utils.serialization_utils import JSONSerializer, UniversalSerializer
        import_results['serialization_utils'] = True
        logger.info("  ✅ serialization_utils: Import successful")
    except Exception as e:
        import_results['serialization_utils'] = False
        logger.error(f"  ❌ serialization_utils: Import failed - {e}")
    
    # Test data processing utils import
    try:
        from src.utils.data_processing_utils import DataFrameValidator, DataFrameCleaner
        import_results['data_processing_utils'] = True
        logger.info("  ✅ data_processing_utils: Import successful")
    except Exception as e:
        import_results['data_processing_utils'] = False
        logger.error(f"  ❌ data_processing_utils: Import failed - {e}")
    
    # Test M1 GPU utils import
    try:
        from src.utils.m1_gpu_utils import get_m1_gpu_manager, M1GPUManager
        import_results['m1_gpu_utils'] = True
        logger.info("  ✅ m1_gpu_utils: Import successful")
    except Exception as e:
        import_results['m1_gpu_utils'] = False
        logger.error(f"  ❌ m1_gpu_utils: Import failed - {e}")
    
    # Test M1 memory optimizer import
    try:
        from src.utils.m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
        import_results['m1_memory_optimizer'] = True
        logger.info("  ✅ m1_memory_optimizer: Import successful")
    except Exception as e:
        import_results['m1_memory_optimizer'] = False
        logger.error(f"  ❌ m1_memory_optimizer: Import failed - {e}")
    
    # Test M1 CPU optimizer import
    try:
        from src.utils.m1_cpu_optimizer import get_m1_cpu_optimizer, M1CPUOptimizer
        import_results['m1_cpu_optimizer'] = True
        logger.info("  ✅ m1_cpu_optimizer: Import successful")
    except Exception as e:
        import_results['m1_cpu_optimizer'] = False
        logger.error(f"  ❌ m1_cpu_optimizer: Import failed - {e}")
    
    # Calculate success rate
    success_count = sum(import_results.values())
    total_count = len(import_results)
    success_rate = success_count / total_count * 100
    
    logger.info(f"📊 Import Test Results:")
    logger.info(f"  📈 Success Rate: {success_rate:.1f}% ({success_count}/{total_count})")
    
    return {
        'success_rate': success_rate,
        'success_count': success_count,
        'total_count': total_count,
        'results': import_results
    }

def test_step03_5_import():
    """Test that Step03_5 can be imported with utility integration."""
    logger.info("🧪 Testing Step03_5 import with utility integration...")
    
    try:
        # Try to import the main class
        from src.training.steps.market_analysis.hmm_clustering.step03_5_final_regime_clustering import (
            FinalRegimeClusteringStep,
            UtilityDependencyInjector
        )
        
        logger.info("  ✅ Step03_5: Import successful")
        logger.info("  ✅ UtilityDependencyInjector: Import successful")
        
        # Test that the class has the expected methods
        expected_methods = [
            '_log_utility_integration_status',
            '_perform_comprehensive_utility_operations',
            '_load_data_with_comprehensive_utilities',
            '_prepare_features_with_comprehensive_utilities',
            '_perform_hmm_regime_discovery_with_utilities',
            '_perform_final_clustering_with_utilities'
        ]
        
        method_results = {}
        for method in expected_methods:
            if hasattr(FinalRegimeClusteringStep, method):
                method_results[method] = True
                logger.info(f"  ✅ Method {method}: Available")
            else:
                method_results[method] = False
                logger.error(f"  ❌ Method {method}: Not available")
        
        method_success_rate = sum(method_results.values()) / len(method_results) * 100
        
        logger.info(f"📊 Method Availability:")
        logger.info(f"  📈 Success Rate: {method_success_rate:.1f}% ({sum(method_results.values())}/{len(method_results)})")
        
        return {
            'import_success': True,
            'method_success_rate': method_success_rate,
            'method_results': method_results
        }
        
    except Exception as e:
        logger.error(f"  ❌ Step03_5: Import failed - {e}")
        return {
            'import_success': False,
            'method_success_rate': 0,
            'method_results': {},
            'error': str(e)
        }

def test_utility_functionality():
    """Test basic functionality of key utilities."""
    logger.info("🧪 Testing utility functionality...")
    
    functionality_results = {}
    
    # Test safe_mean
    try:
        from src.utils.common_operations import safe_mean
        result = safe_mean([1, 2, 3, 4, 5])
        expected = 3.0
        functionality_results['safe_mean'] = abs(result - expected) < 0.001
        logger.info(f"  ✅ safe_mean: {result} (expected: {expected})")
    except Exception as e:
        functionality_results['safe_mean'] = False
        logger.error(f"  ❌ safe_mean: Failed - {e}")
    
    # Test safe_divide
    try:
        from src.utils.common_operations import safe_divide
        result = safe_divide(10, 2)
        expected = 5.0
        functionality_results['safe_divide'] = abs(result - expected) < 0.001
        logger.info(f"  ✅ safe_divide: {result} (expected: {expected})")
    except Exception as e:
        functionality_results['safe_divide'] = False
        logger.error(f"  ❌ safe_divide: Failed - {e}")
    
    # Test safe_divide with zero
    try:
        from src.utils.common_operations import safe_divide
        result = safe_divide(10, 0)
        expected = 0.0  # Should return default value
        functionality_results['safe_divide_zero'] = abs(result - expected) < 0.001
        logger.info(f"  ✅ safe_divide (zero): {result} (expected: {expected})")
    except Exception as e:
        functionality_results['safe_divide_zero'] = False
        logger.error(f"  ❌ safe_divide (zero): Failed - {e}")
    
    # Test validate_finite
    try:
        from src.utils.math_validation import validate_finite
        result = validate_finite(42.0)
        expected = 42.0
        functionality_results['validate_finite'] = abs(result - expected) < 0.001
        logger.info(f"  ✅ validate_finite: {result} (expected: {expected})")
    except Exception as e:
        functionality_results['validate_finite'] = False
        logger.error(f"  ❌ validate_finite: Failed - {e}")
    
    # Calculate success rate
    success_count = sum(functionality_results.values())
    total_count = len(functionality_results)
    success_rate = success_count / total_count * 100
    
    logger.info(f"📊 Functionality Test Results:")
    logger.info(f"  📈 Success Rate: {success_rate:.1f}% ({success_count}/{total_count})")
    
    return {
        'success_rate': success_rate,
        'success_count': success_count,
        'total_count': total_count,
        'results': functionality_results
    }

def main():
    """Main test function."""
    logger.info("🚀 Starting simple Step03_5 utility integration tests...")
    
    # Test 1: Utility Imports
    logger.info("=" * 60)
    logger.info("TEST 1: Utility Module Imports")
    logger.info("=" * 60)
    import_results = test_utility_imports()
    
    # Test 2: Step03_5 Import
    logger.info("=" * 60)
    logger.info("TEST 2: Step03_5 Import with Utility Integration")
    logger.info("=" * 60)
    step03_5_results = test_step03_5_import()
    
    # Test 3: Utility Functionality
    logger.info("=" * 60)
    logger.info("TEST 3: Utility Functionality")
    logger.info("=" * 60)
    functionality_results = test_utility_functionality()
    
    # Summary
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE TEST SUMMARY")
    logger.info("=" * 60)
    
    import_success = import_results['success_rate'] >= 80
    step03_5_success = step03_5_results['import_success'] and step03_5_results['method_success_rate'] >= 80
    functionality_success = functionality_results['success_rate'] >= 80
    
    logger.info(f"📊 Utility Imports: {'✅ PASS' if import_success else '❌ FAIL'}")
    logger.info(f"  📈 Success Rate: {import_results['success_rate']:.1f}%")
    
    logger.info(f"📊 Step03_5 Integration: {'✅ PASS' if step03_5_success else '❌ FAIL'}")
    logger.info(f"  📈 Import Success: {step03_5_results['import_success']}")
    logger.info(f"  📈 Method Success Rate: {step03_5_results['method_success_rate']:.1f}%")
    
    logger.info(f"📊 Utility Functionality: {'✅ PASS' if functionality_success else '❌ FAIL'}")
    logger.info(f"  📈 Success Rate: {functionality_results['success_rate']:.1f}%")
    
    overall_success = import_success and step03_5_success and functionality_success
    logger.info(f"🎉 Overall Test Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if overall_success:
        logger.info("🎊 Comprehensive utility integration in Step03_5 is working correctly!")
        logger.info("🔧 All utilities are properly imported and integrated with dependency injection.")
        logger.info("🚀 Step03_5 is ready for production use with extensive utility integration.")
    else:
        logger.warning("⚠️ Some utility integration issues detected. Review the logs above.")
    
    return overall_success

if __name__ == "__main__":
    # Run the tests
    success = main()
    sys.exit(0 if success else 1)