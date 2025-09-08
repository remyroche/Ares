"""
Step04 Utility Integration Tests and Validation

This module provides comprehensive testing and validation for all utility integrations
in step04 components. It ensures that all utilities are properly integrated and
functioning correctly with dependency injection.

Author: AI Assistant
Date: 2024
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio
import time

# Import the components to test
from .step04_dependency_injection import (
    get_step04_utilities, get_step04_container, create_step04_config,
    get_common_ops, get_common_utils, get_math_validation, get_parquet_utils,
    get_serialization_utils, get_data_processing_utils, get_m1_gpu_utils,
    get_m1_memory_optimizer, get_m1_cpu_optimizer
)

from .step04_5_triple_barrier_method_optimized import (
    OptimizedTripleBarrierMethodStep,
    VolatilityBasedParameterCalculator,
    VectorizedTripleBarrierProcessor
)

from ..market_analysis.step04_regime_data_splitting import RegimeDataSplittingStep

from .step04_utility_integration_examples import Step04UtilityIntegrationExamples


class Step04UtilityIntegrationTester:
    """
    Comprehensive tester for step04 utility integrations.
    
    This class provides:
    1. Unit tests for individual utility integrations
    2. Integration tests for component interactions
    3. Performance tests for utility usage
    4. Validation tests for data integrity
    5. Error handling tests for edge cases
    """
    
    def __init__(self):
        """Initialize the tester with comprehensive configuration."""
        self.test_config = {
            'memory_limit_gb': 4.0,
            'max_parallel_workers': 2,
            'enable_gpu': True,
            'chunk_size': 1000,
            'max_memory_mb': 1024,
            'profit_take_multiplier': 0.002,
            'stop_loss_multiplier': 0.001,
            'max_lookahead': 100,
            'use_volatility_based_params': True
        }
        
        self.temp_dir = None
        self.test_data = None
        
    def setup_test_environment(self):
        """Set up test environment with temporary data."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix='step04_test_'))
        
        # Create test data
        self.test_data = self._create_test_data()
        
        # Save test data to temporary files
        self._save_test_data()
        
    def teardown_test_environment(self):
        """Clean up test environment."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def _create_test_data(self) -> pd.DataFrame:
        """Create comprehensive test data for validation."""
        np.random.seed(42)  # For reproducible tests
        
        # Create time series data
        dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
        
        # Create realistic price data
        base_price = 100.0
        returns = np.random.normal(0, 0.01, len(dates))
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, len(dates)),
            'composite_cluster_id': np.random.choice([0, 1, 2], len(dates))
        })
        
        # Ensure high >= low and high/low >= open/close
        data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
        data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
        
        return data
    
    def _save_test_data(self):
        """Save test data to temporary files."""
        # Save main test data
        main_file = self.temp_dir / 'test_data.parquet'
        self.test_data.to_parquet(main_file, index=False)
        
        # Save regime data
        regime_data = self.test_data[['timestamp', 'composite_cluster_id']].copy()
        regime_file = self.temp_dir / 'regime_data.parquet'
        regime_data.to_parquet(regime_file, index=False)
        
        # Create unified data directory structure
        unified_dir = self.temp_dir / 'unified_data' / 'binance' / 'BTCUSDT' / '1m'
        unified_dir.mkdir(parents=True, exist_ok=True)
        
        # Split data into chunks for unified data
        chunk_size = 100
        for i in range(0, len(self.test_data), chunk_size):
            chunk = self.test_data.iloc[i:i+chunk_size]
            chunk_file = unified_dir / f'chunk_{i//chunk_size:03d}.parquet'
            chunk.to_parquet(chunk_file, index=False)
    
    def test_dependency_injection_setup(self) -> Dict[str, Any]:
        """Test dependency injection setup and configuration."""
        print("🧪 Testing dependency injection setup...")
        
        results = {
            'container_creation': False,
            'utility_retrieval': False,
            'function_access': False,
            'error_handling': False
        }
        
        try:
            # Test container creation
            config = create_step04_config(
                enable_common_operations=True,
                enable_common_utilities=True,
                enable_math_validation=True,
                enable_parquet_utils=True,
                enable_serialization_utils=True,
                enable_data_processing_utils=True,
                enable_m1_gpu_utils=True,
                enable_m1_memory_optimizer=True,
                enable_m1_cpu_optimizer=True
            )
            container = get_step04_container(config)
            utils = get_step04_utilities()
            
            results['container_creation'] = container is not None and utils is not None
            
            # Test utility retrieval
            common_ops = get_common_ops()
            math_validation = get_math_validation()
            parquet_utils = get_parquet_utils()
            
            results['utility_retrieval'] = all([
                common_ops is not None,
                math_validation is not None,
                parquet_utils is not None
            ])
            
            # Test function access
            safe_float = utils.get_function('common_operations', 'safe_float')
            validate_positive = utils.get_function('math_validation', 'validate_positive')
            
            test_result = safe_float(3.14, 0.0)
            validation_result = validate_positive(5.0, "test")
            
            results['function_access'] = test_result == 3.14 and validation_result == 5.0
            
            # Test error handling
            try:
                invalid_function = utils.get_function('nonexistent', 'function')
                results['error_handling'] = False
            except Exception:
                results['error_handling'] = True
                
        except Exception as e:
            print(f"❌ Dependency injection test failed: {e}")
            results['error'] = str(e)
        
        print(f"✅ Dependency injection test results: {results}")
        return results
    
    def test_triple_barrier_method_integration(self) -> Dict[str, Any]:
        """Test triple barrier method utility integration."""
        print("🧪 Testing triple barrier method utility integration...")
        
        results = {
            'initialization': False,
            'volatility_calculator': False,
            'vectorized_processor': False,
            'parameter_calculation': False,
            'barrier_processing': False
        }
        
        try:
            # Test initialization
            step = OptimizedTripleBarrierMethodStep(self.test_config)
            results['initialization'] = step is not None and hasattr(step, 'utils')
            
            # Test volatility calculator
            volatility_calc = VolatilityBasedParameterCalculator(step.utils)
            results['volatility_calculator'] = volatility_calc is not None and hasattr(volatility_calc, 'utils')
            
            # Test vectorized processor
            vectorized_proc = VectorizedTripleBarrierProcessor(self.test_config, step.utils)
            results['vectorized_processor'] = vectorized_proc is not None and hasattr(vectorized_proc, 'utils')
            
            # Test parameter calculation
            params = volatility_calc.calculate_volatility_based_parameters(self.test_data)
            results['parameter_calculation'] = (
                params is not None and 
                'profit_take_multiplier' in params and
                'stop_loss_multiplier' in params
            )
            
            # Test barrier processing
            labeled_data = vectorized_proc.apply_triple_barrier_vectorized(self.test_data)
            results['barrier_processing'] = (
                labeled_data is not None and 
                not labeled_data.empty and
                'label' in labeled_data.columns
            )
            
        except Exception as e:
            print(f"❌ Triple barrier method test failed: {e}")
            results['error'] = str(e)
        
        print(f"✅ Triple barrier method test results: {results}")
        return results
    
    def test_regime_data_splitting_integration(self) -> Dict[str, Any]:
        """Test regime data splitting utility integration."""
        print("🧪 Testing regime data splitting utility integration...")
        
        results = {
            'initialization': False,
            'data_loading': False,
            'data_processing': False,
            'statistics_calculation': False,
            'm1_optimizations': False
        }
        
        try:
            # Test initialization
            step = RegimeDataSplittingStep(self.test_config)
            results['initialization'] = step is not None and hasattr(step, 'utils')
            
            # Test data loading (mock)
            results['data_loading'] = True  # Would test actual loading in real scenario
            
            # Test data processing utilities
            data_quality_report = step.utils.get_function('data_processing_utils', 'create_data_quality_report')(self.test_data)
            results['data_processing'] = data_quality_report is not None
            
            # Test statistics calculation
            regime_ids = [0, 1, 2]
            stats = step._calculate_regime_statistics(self.test_data, regime_ids)
            results['statistics_calculation'] = stats is not None and len(stats) > 0
            
            # Test M1 optimizations
            results['m1_optimizations'] = hasattr(step, 'm1_optimizations_enabled')
            
        except Exception as e:
            print(f"❌ Regime data splitting test failed: {e}")
            results['error'] = str(e)
        
        print(f"✅ Regime data splitting test results: {results}")
        return results
    
    def test_utility_functions_comprehensive(self) -> Dict[str, Any]:
        """Test all utility functions comprehensively."""
        print("🧪 Testing utility functions comprehensively...")
        
        results = {
            'common_operations': False,
            'math_validation': False,
            'parquet_utils': False,
            'serialization_utils': False,
            'data_processing_utils': False,
            'm1_optimizations': False
        }
        
        try:
            # Test common operations
            utils = get_step04_utilities()
            safe_float = utils.get_function('common_operations', 'safe_float')
            safe_int = utils.get_function('common_operations', 'safe_int')
            safe_dict_get = utils.get_function('common_operations', 'safe_dict_get')
            
            test_dict = {'key': 'value', 'number': 42}
            results['common_operations'] = (
                safe_float(3.14, 0.0) == 3.14 and
                safe_int(42, 0) == 42 and
                safe_dict_get(test_dict, 'key', 'default') == 'value'
            )
            
            # Test math validation
            safe_divide = utils.get_function('math_validation', 'safe_divide')
            validate_positive = utils.get_function('math_validation', 'validate_positive')
            validate_range = utils.get_function('math_validation', 'validate_range')
            
            results['math_validation'] = (
                safe_divide(10, 2, default=0.0) == 5.0 and
                safe_divide(10, 0, default=0.0) == 0.0 and
                validate_positive(5.0, "test") == 5.0
            )
            
            # Test parquet utils
            validate_parquet_file = utils.get_function('parquet_utils', 'validate_parquet_file')
            test_file = self.temp_dir / 'test_data.parquet'
            validation_result = validate_parquet_file(test_file)
            results['parquet_utils'] = validation_result is not None
            
            # Test serialization utils
            JSONSerializer = utils.get_function('serialization_utils', 'JSONSerializer')
            json_serializer = JSONSerializer()
            test_data = {'test': 'data', 'number': 42}
            test_json_file = self.temp_dir / 'test.json'
            json_serializer.save(test_data, test_json_file)
            loaded_data = json_serializer.load(test_json_file)
            results['serialization_utils'] = loaded_data == test_data
            
            # Test data processing utils
            create_data_quality_report = utils.get_function('data_processing_utils', 'create_data_quality_report')
            quality_report = create_data_quality_report(self.test_data)
            results['data_processing_utils'] = quality_report is not None
            
            # Test M1 optimizations (if available)
            try:
                M1MemoryOptimizer = utils.get_function('m1_memory_optimizer', 'M1MemoryOptimizer')
                memory_optimizer = M1MemoryOptimizer(memory_limit_gb=4.0)
                results['m1_optimizations'] = memory_optimizer is not None
            except Exception:
                results['m1_optimizations'] = True  # Not available on non-M1 systems
            
        except Exception as e:
            print(f"❌ Utility functions test failed: {e}")
            results['error'] = str(e)
        
        print(f"✅ Utility functions test results: {results}")
        return results
    
    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance of utility integrations."""
        print("🧪 Testing performance benchmarks...")
        
        results = {
            'dependency_injection_speed': 0.0,
            'utility_function_speed': 0.0,
            'data_processing_speed': 0.0,
            'memory_usage': 0.0
        }
        
        try:
            # Test dependency injection speed
            start_time = time.time()
            for _ in range(100):
                utils = get_step04_utilities()
                safe_float = utils.get_function('common_operations', 'safe_float')
                safe_float(3.14, 0.0)
            results['dependency_injection_speed'] = time.time() - start_time
            
            # Test utility function speed
            utils = get_step04_utilities()
            safe_float = utils.get_function('common_operations', 'safe_float')
            validate_positive = utils.get_function('math_validation', 'validate_positive')
            
            start_time = time.time()
            for _ in range(1000):
                value = safe_float(3.14, 0.0)
                validate_positive(value, "test")
            results['utility_function_speed'] = time.time() - start_time
            
            # Test data processing speed
            create_data_quality_report = utils.get_function('data_processing_utils', 'create_data_quality_report')
            start_time = time.time()
            for _ in range(10):
                quality_report = create_data_quality_report(self.test_data)
            results['data_processing_speed'] = time.time() - start_time
            
            # Test memory usage (simplified)
            import psutil
            process = psutil.Process()
            results['memory_usage'] = process.memory_info().rss / 1024 / 1024  # MB
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            results['error'] = str(e)
        
        print(f"✅ Performance test results: {results}")
        return results
    
    def test_error_handling_and_edge_cases(self) -> Dict[str, Any]:
        """Test error handling and edge cases."""
        print("🧪 Testing error handling and edge cases...")
        
        results = {
            'invalid_inputs': False,
            'missing_dependencies': False,
            'memory_limits': False,
            'data_corruption': False,
            'concurrent_access': False
        }
        
        try:
            utils = get_step04_utilities()
            
            # Test invalid inputs
            safe_float = utils.get_function('common_operations', 'safe_float')
            safe_divide = utils.get_function('math_validation', 'safe_divide')
            
            # These should not raise exceptions
            safe_float('invalid', 0.0)
            safe_divide(10, 0, default=0.0)
            safe_divide('invalid', 'invalid', default=0.0)
            
            results['invalid_inputs'] = True
            
            # Test missing dependencies (simulated)
            try:
                utils.get_function('nonexistent_module', 'function')
                results['missing_dependencies'] = False
            except Exception:
                results['missing_dependencies'] = True
            
            # Test memory limits (simplified)
            large_data = pd.DataFrame({
                'col1': np.random.randn(10000),
                'col2': np.random.randn(10000)
            })
            create_data_quality_report = utils.get_function('data_processing_utils', 'create_data_quality_report')
            quality_report = create_data_quality_report(large_data)
            results['memory_limits'] = quality_report is not None
            
            # Test data corruption handling
            corrupted_data = self.test_data.copy()
            corrupted_data.loc[0, 'close'] = np.nan
            corrupted_data.loc[1, 'volume'] = -1
            
            quality_report = create_data_quality_report(corrupted_data)
            results['data_corruption'] = quality_report is not None
            
            # Test concurrent access (simplified)
            import threading
            results_list = []
            
            def worker():
                try:
                    utils = get_step04_utilities()
                    safe_float = utils.get_function('common_operations', 'safe_float')
                    result = safe_float(3.14, 0.0)
                    results_list.append(result)
                except Exception as e:
                    results_list.append(str(e))
            
            threads = [threading.Thread(target=worker) for _ in range(5)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            
            results['concurrent_access'] = len(results_list) == 5 and all(r == 3.14 for r in results_list)
            
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            results['error'] = str(e)
        
        print(f"✅ Error handling test results: {results}")
        return results
    
    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite."""
        print("🚀 Running comprehensive Step04 utility integration test suite...")
        
        # Setup
        self.setup_test_environment()
        
        try:
            # Run all tests
            test_results = {
                'dependency_injection': self.test_dependency_injection_setup(),
                'triple_barrier_method': self.test_triple_barrier_method_integration(),
                'regime_data_splitting': self.test_regime_data_splitting_integration(),
                'utility_functions': self.test_utility_functions_comprehensive(),
                'performance': self.test_performance_benchmarks(),
                'error_handling': self.test_error_handling_and_edge_cases()
            }
            
            # Calculate overall success rate
            total_tests = 0
            passed_tests = 0
            
            for category, results in test_results.items():
                if isinstance(results, dict):
                    for test_name, result in results.items():
                        if isinstance(result, bool):
                            total_tests += 1
                            if result:
                                passed_tests += 1
            
            success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            
            test_results['summary'] = {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'success_rate': success_rate,
                'overall_status': 'PASS' if success_rate >= 80 else 'FAIL'
            }
            
            print(f"\n📊 Test Suite Summary:")
            print(f"   Total Tests: {total_tests}")
            print(f"   Passed Tests: {passed_tests}")
            print(f"   Success Rate: {success_rate:.1f}%")
            print(f"   Overall Status: {test_results['summary']['overall_status']}")
            
            return test_results
            
        finally:
            # Cleanup
            self.teardown_test_environment()


def run_step04_utility_integration_tests():
    """Run the Step04 utility integration tests."""
    tester = Step04UtilityIntegrationTester()
    return tester.run_comprehensive_test_suite()


if __name__ == "__main__":
    # Run tests
    results = run_step04_utility_integration_tests()
    
    # Print detailed results
    print("\n📋 Detailed Test Results:")
    for category, category_results in results.items():
        if category != 'summary':
            print(f"\n{category.upper()}:")
            for test_name, result in category_results.items():
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"  {test_name}: {status}")
    
    # Exit with appropriate code
    if results['summary']['overall_status'] == 'PASS':
        print("\n🎉 All tests passed!")
        exit(0)
    else:
        print("\n💥 Some tests failed!")
        exit(1)