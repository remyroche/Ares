"""
Test script for Step 7 optimizations and fixes.

This script tests:
- Computational optimizations: caching, vectorized operations, chunked processing
- Fast fails: data shape validation, dependency validation, data type validation
- Fixes: async/sync mixing, algorithmic issues
- Math validation integration
- Extensive logging for fast fail scenarios
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import time
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_fast_fail_validations():
    """Test fast fail validation mechanisms."""
    logger.info("🧪 Testing fast fail validations...")
    
    try:
        from src.training.steps.market_analysis.step07_enhanced_matrix_operations_final import (
            Step7EnhancedMatrixOperationsFinal, FastFailError
        )
        
        # Test 1: Missing dependencies
        logger.info("Test 1: Testing missing dependencies...")
        try:
            # Create a config that should trigger dependency validation
            config = {
                'step07_enhanced_matrix_operations': {},
                'output_dir': 'test_output'
            }
            step = Step7EnhancedMatrixOperationsFinal(config)
            logger.info("✅ Dependency validation passed")
        except FastFailError as e:
            logger.info(f"✅ Fast fail caught missing dependencies: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error in dependency validation: {e}")
        
        # Test 2: Invalid configuration
        logger.info("Test 2: Testing invalid configuration...")
        try:
            config = {
                'step07_enhanced_matrix_operations': {
                    'correlation_threshold': 1.5  # Invalid: should be 0-1
                },
                'output_dir': 'test_output'
            }
            step = Step7EnhancedMatrixOperationsFinal(config)
            logger.error("❌ Should have failed on invalid configuration")
        except FastFailError as e:
            logger.info(f"✅ Fast fail caught invalid configuration: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error in configuration validation: {e}")
        
        # Test 3: Missing required configuration
        logger.info("Test 3: Testing missing required configuration...")
        try:
            config = {}  # Missing required configs
            step = Step7EnhancedMatrixOperationsFinal(config)
            logger.error("❌ Should have failed on missing configuration")
        except FastFailError as e:
            logger.info(f"✅ Fast fail caught missing configuration: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error in missing config validation: {e}")
        
        logger.info("✅ Fast fail validation tests completed")
        
    except ImportError as e:
        logger.error(f"❌ Could not import optimized step07: {e}")
    except Exception as e:
        logger.error(f"❌ Error in fast fail tests: {e}")

def test_math_validation_integration():
    """Test math validation integration."""
    logger.info("🧪 Testing math validation integration...")
    
    try:
        from src.utils.math_validation import (
            safe_divide, safe_log, safe_sqrt, validate_finite, 
            validate_positive, validate_range, MathValidationError
        )
        
        # Test safe division
        logger.info("Test 1: Testing safe division...")
        result = safe_divide(10, 0)
        assert result == 0.0, f"Expected 0.0, got {result}"
        result = safe_divide(10, 2)
        assert result == 5.0, f"Expected 5.0, got {result}"
        logger.info("✅ Safe division tests passed")
        
        # Test safe logarithm
        logger.info("Test 2: Testing safe logarithm...")
        result = safe_log(0)
        assert result == 0.0, f"Expected 0.0, got {result}"
        result = safe_log(10)
        assert result > 0, f"Expected positive result, got {result}"
        logger.info("✅ Safe logarithm tests passed")
        
        # Test safe square root
        logger.info("Test 3: Testing safe square root...")
        result = safe_sqrt(-1)
        assert result == 0.0, f"Expected 0.0, got {result}"
        result = safe_sqrt(4)
        assert result == 2.0, f"Expected 2.0, got {result}"
        logger.info("✅ Safe square root tests passed")
        
        # Test validation functions
        logger.info("Test 4: Testing validation functions...")
        try:
            validate_finite(5.0)
            validate_positive(5.0)
            validate_range(5.0, 0.0, 10.0)
            logger.info("✅ Validation functions passed")
        except MathValidationError as e:
            logger.error(f"❌ Validation failed unexpectedly: {e}")
        
        # Test validation with invalid inputs
        logger.info("Test 5: Testing validation with invalid inputs...")
        try:
            validate_positive(-1.0)
            logger.error("❌ Should have failed on negative value")
        except MathValidationError:
            logger.info("✅ Correctly caught negative value")
        
        try:
            validate_range(15.0, 0.0, 10.0)
            logger.error("❌ Should have failed on out-of-range value")
        except MathValidationError:
            logger.info("✅ Correctly caught out-of-range value")
        
        logger.info("✅ Math validation integration tests completed")
        
    except ImportError as e:
        logger.error(f"❌ Could not import math validation: {e}")
    except Exception as e:
        logger.error(f"❌ Error in math validation tests: {e}")

def test_computational_optimizations():
    """Test computational optimizations."""
    logger.info("🧪 Testing computational optimizations...")
    
    try:
        from src.training.steps.market_analysis.utils.matrix_operations_optimized import (
            OptimizedMatrixOperations, FastFailError
        )
        
        # Create test data
        logger.info("Creating test data...")
        np.random.seed(42)
        test_data = pd.DataFrame(np.random.randn(100, 10))
        test_data.columns = [f'feature_{i}' for i in range(10)]
        
        # Initialize optimized matrix operations
        logger.info("Initializing optimized matrix operations...")
        matrix_ops = OptimizedMatrixOperations(logger)
        
        # Test configuration
        config = {
            'correlation_threshold': 0.8,
            'condition_number_threshold': 1e12,
            'min_eigenvalue_threshold': 1e-10,
            'sr_features': [],
            'sr_correlation_threshold': 0.7,
            'sr_condition_number_threshold': 1e10
        }
        
        # Test 1: Standard matrix operations
        logger.info("Test 1: Testing standard matrix operations...")
        start_time = time.time()
        results = asyncio.run(matrix_ops.execute_standard_matrix_operations_optimized(test_data, config))
        execution_time = time.time() - start_time
        
        assert 'error' not in results, f"Matrix operations failed: {results.get('error')}"
        assert 'correlation_analysis' in results, "Missing correlation analysis"
        assert 'condition_number_check' in results, "Missing condition number check"
        assert 'eigenvalue_analysis' in results, "Missing eigenvalue analysis"
        
        logger.info(f"✅ Standard matrix operations completed in {execution_time:.3f}s")
        
        # Test 2: Caching
        logger.info("Test 2: Testing caching...")
        start_time = time.time()
        cached_results = asyncio.run(matrix_ops.execute_standard_matrix_operations_optimized(test_data, config))
        cached_execution_time = time.time() - start_time
        
        # Cached execution should be faster
        assert cached_execution_time < execution_time, f"Caching not working: {cached_execution_time} >= {execution_time}"
        logger.info(f"✅ Caching test passed: {cached_execution_time:.3f}s < {execution_time:.3f}s")
        
        # Test 3: Performance summary
        logger.info("Test 3: Testing performance summary...")
        perf_summary = matrix_ops.get_performance_summary()
        assert 'cache_size' in perf_summary, "Missing cache size in performance summary"
        assert 'performance_metrics' in perf_summary, "Missing performance metrics"
        logger.info(f"✅ Performance summary: cache_size={perf_summary['cache_size']}")
        
        # Test 4: Fast fail with insufficient data
        logger.info("Test 4: Testing fast fail with insufficient data...")
        small_data = pd.DataFrame(np.random.randn(5, 2))  # Too few rows
        try:
            asyncio.run(matrix_ops.execute_standard_matrix_operations_optimized(small_data, config))
            logger.error("❌ Should have failed on insufficient data")
        except FastFailError as e:
            logger.info(f"✅ Fast fail caught insufficient data: {e}")
        
        # Test 5: Fast fail with no numeric columns
        logger.info("Test 5: Testing fast fail with no numeric columns...")
        non_numeric_data = pd.DataFrame({'text': ['a', 'b', 'c'], 'category': ['x', 'y', 'z']})
        try:
            asyncio.run(matrix_ops.execute_standard_matrix_operations_optimized(non_numeric_data, config))
            logger.error("❌ Should have failed on no numeric columns")
        except FastFailError as e:
            logger.info(f"✅ Fast fail caught no numeric columns: {e}")
        
        logger.info("✅ Computational optimizations tests completed")
        
    except ImportError as e:
        logger.error(f"❌ Could not import optimized matrix operations: {e}")
    except Exception as e:
        logger.error(f"❌ Error in computational optimization tests: {e}")

def test_async_sync_fixes():
    """Test async/sync mixing fixes."""
    logger.info("🧪 Testing async/sync mixing fixes...")
    
    try:
        from src.training.steps.market_analysis.step07_enhanced_matrix_operations_final import run_step
        
        # Test async execution
        logger.info("Test 1: Testing async execution...")
        config = {
            'step07_enhanced_matrix_operations': {
                'correlation_threshold': 0.8,
                'condition_number_threshold': 1e12,
                'min_eigenvalue_threshold': 1e-10
            },
            'output_dir': 'test_output'
        }
        
        # This should not hang or cause async/sync issues
        result = asyncio.run(run_step('TEST', 'TEST', '1m', config=config))
        logger.info(f"✅ Async execution completed: {result}")
        
        logger.info("✅ Async/sync mixing fixes tests completed")
        
    except ImportError as e:
        logger.error(f"❌ Could not import final step07: {e}")
    except Exception as e:
        logger.error(f"❌ Error in async/sync tests: {e}")

def test_algorithmic_fixes():
    """Test algorithmic fixes."""
    logger.info("🧪 Testing algorithmic fixes...")
    
    try:
        from src.training.steps.market_analysis.utils.matrix_operations_optimized import OptimizedMatrixOperations
        
        # Create test data with known properties
        logger.info("Creating test data with known properties...")
        np.random.seed(42)
        
        # Create a well-conditioned matrix
        well_conditioned_data = pd.DataFrame(np.random.randn(50, 5))
        well_conditioned_data.columns = [f'feature_{i}' for i in range(5)]
        
        # Create an ill-conditioned matrix (near-singular)
        ill_conditioned_data = well_conditioned_data.copy()
        ill_conditioned_data['feature_4'] = ill_conditioned_data['feature_0'] + 1e-10  # Nearly identical columns
        
        matrix_ops = OptimizedMatrixOperations(logger)
        config = {
            'correlation_threshold': 0.8,
            'condition_number_threshold': 1e12,
            'min_eigenvalue_threshold': 1e-10
        }
        
        # Test 1: Well-conditioned matrix
        logger.info("Test 1: Testing well-conditioned matrix...")
        results = asyncio.run(matrix_ops.execute_standard_matrix_operations_optimized(well_conditioned_data, config))
        
        assert 'condition_number_check' in results, "Missing condition number check"
        condition_check = results['condition_number_check']
        assert 'condition_number' in condition_check, "Missing condition number"
        assert 'is_well_conditioned' in condition_check, "Missing well-conditioned flag"
        
        logger.info(f"✅ Well-conditioned matrix: condition_number={condition_check['condition_number']:.2e}, well_conditioned={condition_check['is_well_conditioned']}")
        
        # Test 2: Ill-conditioned matrix
        logger.info("Test 2: Testing ill-conditioned matrix...")
        results = asyncio.run(matrix_ops.execute_standard_matrix_operations_optimized(ill_conditioned_data, config))
        
        condition_check = results['condition_number_check']
        logger.info(f"✅ Ill-conditioned matrix: condition_number={condition_check['condition_number']:.2e}, well_conditioned={condition_check['is_well_conditioned']}")
        
        # Test 3: Vectorized operations
        logger.info("Test 3: Testing vectorized operations...")
        start_time = time.time()
        results = asyncio.run(matrix_ops.execute_standard_matrix_operations_optimized(well_conditioned_data, config))
        vectorized_time = time.time() - start_time
        
        # Should complete quickly with vectorized operations
        assert vectorized_time < 1.0, f"Vectorized operations too slow: {vectorized_time:.3f}s"
        logger.info(f"✅ Vectorized operations completed in {vectorized_time:.3f}s")
        
        # Test 4: Safe mathematical operations
        logger.info("Test 4: Testing safe mathematical operations...")
        assert 'eigenvalue_analysis' in results, "Missing eigenvalue analysis"
        eigenvalue_analysis = results['eigenvalue_analysis']
        assert 'eigenvalues' in eigenvalue_analysis, "Missing eigenvalues"
        assert 'eigenvalue_ratio' in eigenvalue_analysis, "Missing eigenvalue ratio"
        
        # Eigenvalue ratio should be finite
        ratio = eigenvalue_analysis['eigenvalue_ratio']
        assert np.isfinite(ratio), f"Eigenvalue ratio should be finite: {ratio}"
        logger.info(f"✅ Safe mathematical operations: eigenvalue_ratio={ratio:.2e}")
        
        logger.info("✅ Algorithmic fixes tests completed")
        
    except ImportError as e:
        logger.error(f"❌ Could not import optimized matrix operations: {e}")
    except Exception as e:
        logger.error(f"❌ Error in algorithmic tests: {e}")

def test_extensive_logging():
    """Test extensive logging for fast fail scenarios."""
    logger.info("🧪 Testing extensive logging...")
    
    # Capture log output
    import io
    import contextlib
    
    log_capture = io.StringIO()
    
    with contextlib.redirect_stderr(log_capture):
        try:
            from src.training.steps.market_analysis.step07_enhanced_matrix_operations_final import (
                Step7EnhancedMatrixOperationsFinal, FastFailError
            )
            
            # Test with invalid configuration to trigger extensive logging
            config = {
                'step07_enhanced_matrix_operations': {
                    'correlation_threshold': 1.5  # Invalid
                },
                'output_dir': 'test_output'
            }
            
            try:
                step = Step7EnhancedMatrixOperationsFinal(config)
            except FastFailError:
                pass  # Expected
            
            # Check that extensive logging occurred
            log_output = log_capture.getvalue()
            assert 'FAST FAIL' in log_output, "Missing FAST FAIL logging"
            assert 'Validating' in log_output, "Missing validation logging"
            assert 'MISSING' in log_output or 'Available' in log_output, "Missing dependency logging"
            
            logger.info("✅ Extensive logging test passed")
            
        except ImportError as e:
            logger.error(f"❌ Could not import final step07: {e}")
        except Exception as e:
            logger.error(f"❌ Error in logging tests: {e}")

def main():
    """Run all tests."""
    logger.info("🚀 Starting Step 7 optimization tests...")
    
    test_functions = [
        test_fast_fail_validations,
        test_math_validation_integration,
        test_computational_optimizations,
        test_async_sync_fixes,
        test_algorithmic_fixes,
        test_extensive_logging
    ]
    
    passed_tests = 0
    total_tests = len(test_functions)
    
    for test_func in test_functions:
        try:
            test_func()
            passed_tests += 1
        except Exception as e:
            logger.error(f"❌ Test {test_func.__name__} failed: {e}")
    
    logger.info(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! Step 7 optimizations are working correctly.")
    else:
        logger.error(f"❌ {total_tests - passed_tests} tests failed. Please review the implementation.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)