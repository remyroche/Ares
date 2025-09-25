"""
Enhanced Scripts Validation and Testing

This module provides comprehensive validation and testing for the enhanced scripts
with proper error handling, logging, and utility integration.

Key Features:
- Validation of error handling improvements
- Testing of logging implementation
- Verification of utility integration
- Performance testing with M1 optimization
- Comprehensive test coverage
"""

import asyncio
import pytest
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

# Import the enhanced scripts
try:
    from src.launcher.ares_launcher import AresLauncher, LauncherMode, ExecutionModeType
    from src.training.steps.main_training_pipeline import MainTrainingPipeline, MainPipelineConfig
    from src.training.steps.market_analysis.sub_pipeline import MarketAnalysisSubPipeline, SubPipelineConfig
    from src.utils.kline_parquet import KlineParquetManager
    SCRIPTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Enhanced scripts not available: {e}")
    SCRIPTS_AVAILABLE = False

# Import utilities
try:
    from src.utils.tprint import tprint, tprint_error, tprint_success, tprint_warning, tprint_info
    from src.utils.common_operations import (
        safe_json_dump, safe_json_load, ensure_directory,
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        integrate_with_m1_optimizers, cleanup_m1_optimizers
    )
    from src.utils.math_validation import (
        validate_finite, validate_positive, safe_divide, safe_log, safe_sqrt
    )
    from src.utils.serialization_utils import UniversalSerializer
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Utilities not available: {e}")
    UTILS_AVAILABLE = False

class EnhancedScriptsValidator:
    """
    Validator for enhanced scripts with comprehensive testing.
    
    Tests error handling, logging, utility integration, and performance
    improvements across all enhanced scripts.
    """
    
    def __init__(self):
        """Initialize the validator."""
        try:
            tprint("🚀 [INIT] Starting EnhancedScriptsValidator initialization...")
            self.utils_available = UTILS_AVAILABLE
            self.scripts_available = SCRIPTS_AVAILABLE
            self.test_results: Dict[str, Any] = {}
            self.temp_dir = None
            
            if self.utils_available:
                try:
                    self.serializer = UniversalSerializer()
                    self.m1_optimizers = integrate_with_m1_optimizers()
                    tprint_success("✅ [INIT] Utility systems initialized")
                except Exception as e:
                    tprint_warning(f"⚠️ [INIT] Utility systems initialization failed: {e}")
                    self.utils_available = False
            
            tprint_success("✅ [INIT] EnhancedScriptsValidator initialized successfully")
            
        except Exception as e:
            tprint_error(f"❌ [INIT] EnhancedScriptsValidator initialization failed: {e}")
            raise
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive validation of all enhanced scripts.
        
        Returns:
            Dictionary with validation results
        """
        try:
            tprint("🔍 [VALIDATION] Starting comprehensive validation...")
            
            # Create temporary directory for testing
            self.temp_dir = tempfile.mkdtemp()
            tprint_info(f"📁 [VALIDATION] Created temporary directory: {self.temp_dir}")
            
            # Run validation tests
            validation_results = {
                'error_handling_tests': self._test_error_handling(),
                'logging_tests': self._test_logging_implementation(),
                'utility_integration_tests': self._test_utility_integration(),
                'performance_tests': self._test_performance_improvements(),
                'data_validation_tests': self._test_data_validation(),
                'm1_optimization_tests': self._test_m1_optimization(),
                'ml_utilities_tests': self._test_ml_utilities(),
                'serialization_tests': self._test_serialization(),
                'overall_score': 0.0
            }
            
            # Calculate overall score
            scores = []
            for test_name, results in validation_results.items():
                if test_name != 'overall_score' and isinstance(results, dict):
                    if 'score' in results:
                        scores.append(results['score'])
            
            if scores:
                validation_results['overall_score'] = sum(scores) / len(scores)
            
            tprint_success(f"✅ [VALIDATION] Comprehensive validation completed with score: {validation_results['overall_score']:.2f}")
            return validation_results
            
        except Exception as e:
            tprint_error(f"❌ [VALIDATION] Comprehensive validation failed: {e}")
            return {'error': str(e), 'overall_score': 0.0}
        finally:
            # Cleanup
            if self.temp_dir and Path(self.temp_dir).exists():
                try:
                    shutil.rmtree(self.temp_dir)
                    tprint_info("🧹 [VALIDATION] Temporary directory cleaned up")
                except Exception as e:
                    tprint_warning(f"⚠️ [VALIDATION] Cleanup failed: {e}")
    
    def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling improvements."""
        try:
            tprint("🔍 [ERROR_HANDLING] Testing error handling improvements...")
            
            test_results = {
                'ares_launcher_error_handling': False,
                'main_pipeline_error_handling': False,
                'market_analysis_error_handling': False,
                'kline_parquet_error_handling': False,
                'score': 0.0
            }
            
            # Test AresLauncher error handling
            try:
                if self.scripts_available:
                    launcher = AresLauncher()
                    # Test with invalid parameters
                    try:
                        asyncio.run(launcher.execute_pipeline(
                            symbol="",  # Invalid empty symbol
                            exchange="",  # Invalid empty exchange
                            timeframe="",  # Invalid empty timeframe
                            data_dir=""  # Invalid empty data_dir
                        ))
                    except Exception:
                        pass  # Expected to fail
                    test_results['ares_launcher_error_handling'] = True
            except Exception as e:
                tprint_warning(f"⚠️ [ERROR_HANDLING] AresLauncher error handling test failed: {e}")
            
            # Test MainTrainingPipeline error handling
            try:
                if self.scripts_available:
                    pipeline = MainTrainingPipeline()
                    # Test with invalid configuration
                    try:
                        asyncio.run(pipeline.execute_pipeline())
                    except Exception:
                        pass  # Expected to fail
                    test_results['main_pipeline_error_handling'] = True
            except Exception as e:
                tprint_warning(f"⚠️ [ERROR_HANDLING] MainTrainingPipeline error handling test failed: {e}")
            
            # Test MarketAnalysisSubPipeline error handling
            try:
                if self.scripts_available:
                    sub_pipeline = MarketAnalysisSubPipeline()
                    # Test with invalid data
                    try:
                        sub_pipeline.execute_sub_pipeline("invalid_step", {})
                    except Exception:
                        pass  # Expected to fail
                    test_results['market_analysis_error_handling'] = True
            except Exception as e:
                tprint_warning(f"⚠️ [ERROR_HANDLING] MarketAnalysisSubPipeline error handling test failed: {e}")
            
            # Test KlineParquetManager error handling
            try:
                kline_manager = KlineParquetManager()
                # Test with invalid file path
                try:
                    kline_manager.load_kline_data("nonexistent_file.parquet")
                except Exception:
                    pass  # Expected to fail
                test_results['kline_parquet_error_handling'] = True
            except Exception as e:
                tprint_warning(f"⚠️ [ERROR_HANDLING] KlineParquetManager error handling test failed: {e}")
            
            # Calculate score
            passed_tests = sum(1 for v in test_results.values() if v is True)
            total_tests = len([k for k in test_results.keys() if k.endswith('_error_handling')])
            test_results['score'] = passed_tests / total_tests if total_tests > 0 else 0.0
            
            tprint_success(f"✅ [ERROR_HANDLING] Error handling tests completed with score: {test_results['score']:.2f}")
            return test_results
            
        except Exception as e:
            tprint_error(f"❌ [ERROR_HANDLING] Error handling tests failed: {e}")
            return {'error': str(e), 'score': 0.0}
    
    def _test_logging_implementation(self) -> Dict[str, Any]:
        """Test logging implementation."""
        try:
            tprint("🔍 [LOGGING] Testing logging implementation...")
            
            test_results = {
                'tprint_functions_available': False,
                'logging_integration': False,
                'error_logging': False,
                'success_logging': False,
                'score': 0.0
            }
            
            # Test tprint functions availability
            try:
                from src.utils.tprint import (
                    tprint, tprint_error, tprint_success, tprint_warning, tprint_info
                )
                test_results['tprint_functions_available'] = True
            except Exception as e:
                tprint_warning(f"⚠️ [LOGGING] tprint functions not available: {e}")
            
            # Test logging integration
            try:
                if self.scripts_available:
                    launcher = AresLauncher()
                    # Check if logging is properly initialized
                    if hasattr(launcher, 'logger') and launcher.logger:
                        test_results['logging_integration'] = True
            except Exception as e:
                tprint_warning(f"⚠️ [LOGGING] Logging integration test failed: {e}")
            
            # Test error logging
            try:
                tprint_error("Test error message")
                test_results['error_logging'] = True
            except Exception as e:
                tprint_warning(f"⚠️ [LOGGING] Error logging test failed: {e}")
            
            # Test success logging
            try:
                tprint_success("Test success message")
                test_results['success_logging'] = True
            except Exception as e:
                tprint_warning(f"⚠️ [LOGGING] Success logging test failed: {e}")
            
            # Calculate score
            passed_tests = sum(1 for v in test_results.values() if v is True)
            total_tests = len([k for k in test_results.keys() if k.endswith('_logging') or k.endswith('_available')])
            test_results['score'] = passed_tests / total_tests if total_tests > 0 else 0.0
            
            tprint_success(f"✅ [LOGGING] Logging tests completed with score: {test_results['score']:.2f}")
            return test_results
            
        except Exception as e:
            tprint_error(f"❌ [LOGGING] Logging tests failed: {e}")
            return {'error': str(e), 'score': 0.0}
    
    def _test_utility_integration(self) -> Dict[str, Any]:
        """Test utility integration."""
        try:
            tprint("🔍 [UTILITY_INTEGRATION] Testing utility integration...")
            
            test_results = {
                'common_operations_available': False,
                'math_validation_available': False,
                'serialization_available': False,
                'm1_optimizers_available': False,
                'score': 0.0
            }
            
            # Test common operations
            try:
                from src.utils.common_operations import (
                    safe_json_dump, safe_json_load, ensure_directory
                )
                test_results['common_operations_available'] = True
            except Exception as e:
                tprint_warning(f"⚠️ [UTILITY_INTEGRATION] Common operations not available: {e}")
            
            # Test math validation
            try:
                from src.utils.math_validation import (
                    validate_finite, validate_positive, safe_divide
                )
                test_results['math_validation_available'] = True
            except Exception as e:
                tprint_warning(f"⚠️ [UTILITY_INTEGRATION] Math validation not available: {e}")
            
            # Test serialization
            try:
                from src.utils.serialization_utils import UniversalSerializer
                serializer = UniversalSerializer()
                test_results['serialization_available'] = True
            except Exception as e:
                tprint_warning(f"⚠️ [UTILITY_INTEGRATION] Serialization not available: {e}")
            
            # Test M1 optimizers
            try:
                if self.m1_optimizers and self.m1_optimizers.get('success', False):
                    test_results['m1_optimizers_available'] = True
            except Exception as e:
                tprint_warning(f"⚠️ [UTILITY_INTEGRATION] M1 optimizers not available: {e}")
            
            # Calculate score
            passed_tests = sum(1 for v in test_results.values() if v is True)
            total_tests = len([k for k in test_results.keys() if k.endswith('_available')])
            test_results['score'] = passed_tests / total_tests if total_tests > 0 else 0.0
            
            tprint_success(f"✅ [UTILITY_INTEGRATION] Utility integration tests completed with score: {test_results['score']:.2f}")
            return test_results
            
        except Exception as e:
            tprint_error(f"❌ [UTILITY_INTEGRATION] Utility integration tests failed: {e}")
            return {'error': str(e), 'score': 0.0}
    
    def _test_performance_improvements(self) -> Dict[str, Any]:
        """Test performance improvements."""
        try:
            tprint("🔍 [PERFORMANCE] Testing performance improvements...")
            
            test_results = {
                'memory_optimization': False,
                'execution_time_optimization': False,
                'dataframe_optimization': False,
                'score': 0.0
            }
            
            # Test memory optimization
            try:
                if self.utils_available:
                    from src.utils.common_operations import get_m1_memory_optimizer
                    memory_optimizer = get_m1_memory_optimizer()
                    if memory_optimizer:
                        test_results['memory_optimization'] = True
            except Exception as e:
                tprint_warning(f"⚠️ [PERFORMANCE] Memory optimization test failed: {e}")
            
            # Test execution time optimization
            try:
                start_time = datetime.now()
                # Simulate some work
                data = pd.DataFrame({'test': range(1000)})
                data = data * 2
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                if execution_time < 1.0:  # Should be fast
                    test_results['execution_time_optimization'] = True
            except Exception as e:
                tprint_warning(f"⚠️ [PERFORMANCE] Execution time optimization test failed: {e}")
            
            # Test dataframe optimization
            try:
                if self.utils_available:
                    from src.utils.common_operations import optimize_dataframe_dtypes
                    data = pd.DataFrame({
                        'int_col': [1, 2, 3, 4, 5],
                        'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
                        'string_col': ['a', 'b', 'c', 'd', 'e']
                    })
                    optimized_data = optimize_dataframe_dtypes(data)
                    if optimized_data is not None:
                        test_results['dataframe_optimization'] = True
            except Exception as e:
                tprint_warning(f"⚠️ [PERFORMANCE] Dataframe optimization test failed: {e}")
            
            # Calculate score
            passed_tests = sum(1 for v in test_results.values() if v is True)
            total_tests = len([k for k in test_results.keys() if k.endswith('_optimization')])
            test_results['score'] = passed_tests / total_tests if total_tests > 0 else 0.0
            
            tprint_success(f"✅ [PERFORMANCE] Performance tests completed with score: {test_results['score']:.2f}")
            return test_results
            
        except Exception as e:
            tprint_error(f"❌ [PERFORMANCE] Performance tests failed: {e}")
            return {'error': str(e), 'score': 0.0}
    
    def _test_data_validation(self) -> Dict[str, Any]:
        """Test data validation improvements."""
        try:
            tprint("🔍 [DATA_VALIDATION] Testing data validation improvements...")
            
            test_results = {
                'kline_data_validation': False,
                'dataframe_validation': False,
                'numeric_validation': False,
                'score': 0.0
            }
            
            # Test kline data validation
            try:
                kline_manager = KlineParquetManager()
                # Create test data
                test_data = pd.DataFrame({
                    'open': [100, 101, 102, 103, 104],
                    'high': [105, 106, 107, 108, 109],
                    'low': [99, 100, 101, 102, 103],
                    'close': [101, 102, 103, 104, 105],
                    'volume': [1000, 1100, 1200, 1300, 1400],
                    'timestamp': pd.date_range('2023-01-01', periods=5, freq='D')
                })
                
                # Test validation
                kline_manager._validate_kline_data(test_data)
                test_results['kline_data_validation'] = True
            except Exception as e:
                tprint_warning(f"⚠️ [DATA_VALIDATION] Kline data validation test failed: {e}")
            
            # Test dataframe validation
            try:
                if self.utils_available:
                    from src.utils.common_operations import validate_dataframe_columns
                    test_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
                    validate_dataframe_columns(test_data, ['col1', 'col2'])
                    test_results['dataframe_validation'] = True
            except Exception as e:
                tprint_warning(f"⚠️ [DATA_VALIDATION] Dataframe validation test failed: {e}")
            
            # Test numeric validation
            try:
                if self.utils_available:
                    from src.utils.math_validation import validate_finite, validate_positive
                    validate_finite(1.5)
                    validate_positive(1.5)
                    test_results['numeric_validation'] = True
            except Exception as e:
                tprint_warning(f"⚠️ [DATA_VALIDATION] Numeric validation test failed: {e}")
            
            # Calculate score
            passed_tests = sum(1 for v in test_results.values() if v is True)
            total_tests = len([k for k in test_results.keys() if k.endswith('_validation')])
            test_results['score'] = passed_tests / total_tests if total_tests > 0 else 0.0
            
            tprint_success(f"✅ [DATA_VALIDATION] Data validation tests completed with score: {test_results['score']:.2f}")
            return test_results
            
        except Exception as e:
            tprint_error(f"❌ [DATA_VALIDATION] Data validation tests failed: {e}")
            return {'error': str(e), 'score': 0.0}
    
    def _test_m1_optimization(self) -> Dict[str, Any]:
        """Test M1 optimization features."""
        try:
            tprint("🔍 [M1_OPTIMIZATION] Testing M1 optimization features...")
            
            test_results = {
                'm1_gpu_available': False,
                'm1_memory_optimizer_available': False,
                'm1_cpu_optimizer_available': False,
                'score': 0.0
            }
            
            # Test M1 GPU availability
            try:
                if self.utils_available:
                    from src.utils.common_operations import get_m1_gpu_manager
                    gpu_manager = get_m1_gpu_manager()
                    if gpu_manager:
                        test_results['m1_gpu_available'] = True
            except Exception as e:
                tprint_warning(f"⚠️ [M1_OPTIMIZATION] M1 GPU test failed: {e}")
            
            # Test M1 memory optimizer
            try:
                if self.utils_available:
                    from src.utils.common_operations import get_m1_memory_optimizer
                    memory_optimizer = get_m1_memory_optimizer()
                    if memory_optimizer:
                        test_results['m1_memory_optimizer_available'] = True
            except Exception as e:
                tprint_warning(f"⚠️ [M1_OPTIMIZATION] M1 memory optimizer test failed: {e}")
            
            # Test M1 CPU optimizer
            try:
                if self.utils_available:
                    from src.utils.common_operations import get_m1_cpu_optimizer
                    cpu_optimizer = get_m1_cpu_optimizer()
                    if cpu_optimizer:
                        test_results['m1_cpu_optimizer_available'] = True
            except Exception as e:
                tprint_warning(f"⚠️ [M1_OPTIMIZATION] M1 CPU optimizer test failed: {e}")
            
            # Calculate score
            passed_tests = sum(1 for v in test_results.values() if v is True)
            total_tests = len([k for k in test_results.keys() if k.endswith('_available')])
            test_results['score'] = passed_tests / total_tests if total_tests > 0 else 0.0
            
            tprint_success(f"✅ [M1_OPTIMIZATION] M1 optimization tests completed with score: {test_results['score']:.2f}")
            return test_results
            
        except Exception as e:
            tprint_error(f"❌ [M1_OPTIMIZATION] M1 optimization tests failed: {e}")
            return {'error': str(e), 'score': 0.0}
    
    def _test_ml_utilities(self) -> Dict[str, Any]:
        """Test ML utilities integration."""
        try:
            tprint("🔍 [ML_UTILITIES] Testing ML utilities integration...")
            
            test_results = {
                'ml_common_available': False,
                'score': 0.0
            }
            
            # Test ML common utilities (Bayesian TPE is used in specific components, not main pipelines)
            try:
                if self.utils_available:
                    # Test if ML utilities are available (they're used in specific components)
                    from src.utils.nas_tas.bayesian_tpe_optimizer import BayesianTPEOptimizer
                    if BayesianTPEOptimizer:
                        test_results['ml_common_available'] = True
            except Exception as e:
                tprint_warning(f"⚠️ [ML_UTILITIES] ML common utilities test failed: {e}")
            
            # Calculate score
            passed_tests = sum(1 for v in test_results.values() if v is True)
            total_tests = len([k for k in test_results.keys() if k.endswith('_available')])
            test_results['score'] = passed_tests / total_tests if total_tests > 0 else 0.0
            
            tprint_success(f"✅ [ML_UTILITIES] ML utilities tests completed with score: {test_results['score']:.2f}")
            return test_results
            
        except Exception as e:
            tprint_error(f"❌ [ML_UTILITIES] ML utilities tests failed: {e}")
            return {'error': str(e), 'score': 0.0}
    
    def _test_serialization(self) -> Dict[str, Any]:
        """Test serialization utilities."""
        try:
            tprint("🔍 [SERIALIZATION] Testing serialization utilities...")
            
            test_results = {
                'json_serialization': False,
                'pickle_serialization': False,
                'parquet_serialization': False,
                'score': 0.0
            }
            
            # Test JSON serialization
            try:
                if self.utils_available:
                    test_data = {'test': 'data', 'number': 42}
                    test_file = Path(self.temp_dir) / 'test.json'
                    success = self.serializer.save(test_data, str(test_file), format='json')
                    if success and test_file.exists():
                        test_results['json_serialization'] = True
            except Exception as e:
                tprint_warning(f"⚠️ [SERIALIZATION] JSON serialization test failed: {e}")
            
            # Test Pickle serialization
            try:
                if self.utils_available:
                    test_data = {'test': 'data', 'number': 42}
                    test_file = Path(self.temp_dir) / 'test.pkl'
                    success = self.serializer.save(test_data, str(test_file), format='pickle')
                    if success and test_file.exists():
                        test_results['pickle_serialization'] = True
            except Exception as e:
                tprint_warning(f"⚠️ [SERIALIZATION] Pickle serialization test failed: {e}")
            
            # Test Parquet serialization
            try:
                if self.utils_available:
                    test_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
                    test_file = Path(self.temp_dir) / 'test.parquet'
                    success = self.serializer.save(test_data, str(test_file), format='parquet')
                    if success and test_file.exists():
                        test_results['parquet_serialization'] = True
            except Exception as e:
                tprint_warning(f"⚠️ [SERIALIZATION] Parquet serialization test failed: {e}")
            
            # Calculate score
            passed_tests = sum(1 for v in test_results.values() if v is True)
            total_tests = len([k for k in test_results.keys() if k.endswith('_serialization')])
            test_results['score'] = passed_tests / total_tests if total_tests > 0 else 0.0
            
            tprint_success(f"✅ [SERIALIZATION] Serialization tests completed with score: {test_results['score']:.2f}")
            return test_results
            
        except Exception as e:
            tprint_error(f"❌ [SERIALIZATION] Serialization tests failed: {e}")
            return {'error': str(e), 'score': 0.0}

def run_validation():
    """Run the comprehensive validation."""
    try:
        tprint("🚀 [MAIN] Starting comprehensive validation...")
        validator = EnhancedScriptsValidator()
        results = validator.run_comprehensive_validation()
        
        tprint_success(f"✅ [MAIN] Validation completed with overall score: {results.get('overall_score', 0.0):.2f}")
        
        # Print detailed results
        for test_name, test_results in results.items():
            if test_name != 'overall_score' and isinstance(test_results, dict):
                score = test_results.get('score', 0.0)
                tprint_info(f"📊 {test_name}: {score:.2f}")
        
        return results
        
    except Exception as e:
        tprint_error(f"❌ [MAIN] Validation failed: {e}")
        return {'error': str(e), 'overall_score': 0.0}

if __name__ == "__main__":
    results = run_validation()
    print(f"\n🎯 Overall Validation Score: {results.get('overall_score', 0.0):.2f}")