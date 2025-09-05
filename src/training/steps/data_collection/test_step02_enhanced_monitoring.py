"""
Comprehensive Test Suite for Enhanced Step02 Function Monitoring

This script tests all the enhanced function checking mechanisms including:
- Function call validation
- Function-to-function call monitoring
- Function completion reporting
- Enhanced logging system
- Error handling improvements
- Performance monitoring
"""
import asyncio
import sys
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.training.steps.data_collection.step02_data_reading import DataReadingStep, function_monitor, comprehensive_function_monitoring, FunctionCallMonitor, FunctionInteractionReport
from src.training.steps.data_collection.step02_data_reading_validator import run_validator, generate_validation_function_report
import collections
import logging

class Step02EnhancedMonitoringTester:
    """Comprehensive test suite for enhanced Step02 function monitoring."""

    def __init__(self) -> None:
        self.test_results = {}
        self.temp_dir = None
        self.setup_test_environment()

    def setup_test_environment(self) -> None:
        """Setup test environment with temporary directories and mock data."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix='step02_test_'))
        unified_data_dir = self.temp_dir / 'unified' / 'BINANCE' / 'ETHUSDT' / '1m'
        unified_data_dir.mkdir(parents=True, exist_ok=True)
        mock_data = pd.DataFrame({'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'), 'open': np.random.uniform(100, 200, 1000), 'high': np.random.uniform(100, 200, 1000), 'low': np.random.uniform(100, 200, 1000), 'close': np.random.uniform(100, 200, 1000), 'volume': np.random.uniform(1000, 10000, 1000)})
        mock_data.to_parquet(unified_data_dir / 'test_data.parquet', index=False)
        print(f'✅ Test environment setup complete: {self.temp_dir}')

    def cleanup_test_environment(self) -> None:
        """Cleanup test environment."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f'✅ Test environment cleaned up: {self.temp_dir}')

    async def test_function_call_monitoring(self) -> Dict[str, Any]:
        """Test comprehensive function call monitoring."""
        print('\n🔍 Testing Function Call Monitoring...')
        try:
            function_monitor.active_calls.clear()
            function_monitor.completed_calls.clear()
            function_monitor.call_counter = 0
            config = {'SYMBOL': 'ETHUSDT', 'EXCHANGE': 'BINANCE', 'TIMEFRAME': '1m', 'DATA_DIR': str(self.temp_dir), 'step02_quality_thresholds': {'min_rows': 100, 'max_null_ratio': 0.1, 'min_quality_score': 0.5}}
            step = DataReadingStep(config)
            result = await step.execute(symbol='ETHUSDT', exchange='BINANCE', timeframe='1m', data_dir=str(self.temp_dir))
            function_report = function_monitor.get_function_interaction_report()
            test_passed = result.get('success', False) and function_report.total_calls > 0 and (function_report.successful_calls > 0) and ('function_interaction_report' in result)
            print(f'   - Total function calls: {function_report.total_calls}')
            print(f'   - Successful calls: {function_report.successful_calls}')
            print(f'   - Failed calls: {function_report.failed_calls}')
            print(f"   - Success rate: {function_report.performance_metrics.get('success_rate', 0):.1f}%")
            print(f'   - Total execution time: {function_report.total_execution_time:.3f}s')
            return {'test_name': 'function_call_monitoring', 'passed': test_passed, 'function_report': function_report, 'result': result}
        except Exception as e:
            print(f'❌ Function call monitoring test failed: {e}')
            return {'test_name': 'function_call_monitoring', 'passed': False, 'error': str(e)}

    async def test_function_interaction_tracking(self) -> Dict[str, Any]:
        """Test function-to-function call interaction tracking."""
        print('\n🔗 Testing Function Interaction Tracking...')
        try:
            function_monitor.active_calls.clear()
            function_monitor.completed_calls.clear()
            function_monitor.call_counter = 0

            @comprehensive_function_monitoring(validate_inputs=True, validate_outputs=True, track_performance=True, timeout_seconds=30, retry_attempts=1)
            async def test_parent_function(x: int) -> Dict[str, Any]:
                """Parent function that calls child functions."""
                result1 = await test_child_function_1(x)
                result2 = await test_child_function_2(x)
                return {'parent_result': x * 2, 'child1': result1, 'child2': result2}

            @comprehensive_function_monitoring(validate_inputs=True, validate_outputs=True, track_performance=True, timeout_seconds=30, retry_attempts=1)
            async def test_child_function_1(x: int) -> int:
                """Child function 1."""
                await asyncio.sleep(0.01)
                return x + 1

            @comprehensive_function_monitoring(validate_inputs=True, validate_outputs=True, track_performance=True, timeout_seconds=30, retry_attempts=1)
            async def test_child_function_2(x: int) -> int:
                """Child function 2."""
                await asyncio.sleep(0.01)
                return x * 2
            result = await test_parent_function(5)
            function_report = function_monitor.get_function_interaction_report()
            test_passed = function_report.total_calls >= 3 and len(function_report.call_hierarchy) > 0 and (function_report.performance_metrics.get('max_call_depth', 0) > 0)
            print(f'   - Total function calls: {function_report.total_calls}')
            print(f'   - Call hierarchy entries: {len(function_report.call_hierarchy)}')
            print(f"   - Maximum call depth: {function_report.performance_metrics.get('max_call_depth', 0)}")
            print(f"   - Function frequency: {function_report.performance_metrics.get('function_frequency', {})}")
            return {'test_name': 'function_interaction_tracking', 'passed': test_passed, 'function_report': function_report, 'result': result}
        except Exception as e:
            print(f'❌ Function interaction tracking test failed: {e}')
            return {'test_name': 'function_interaction_tracking', 'passed': False, 'error': str(e)}

    async def test_error_handling_and_recovery(self) -> Dict[str, Any]:
        """Test enhanced error handling and recovery mechanisms."""
        print('\n⚠️ Testing Error Handling and Recovery...')
        try:
            function_monitor.active_calls.clear()
            function_monitor.completed_calls.clear()
            function_monitor.call_counter = 0

            @comprehensive_function_monitoring(validate_inputs=True, validate_outputs=True, track_performance=True, timeout_seconds=30, retry_attempts=2)
            async def test_failing_function(x: int) -> int:
                """Function that will fail on first attempt but succeed on retry."""
                if x < 0:
                    raise ValueError('Negative value not allowed')
                return x * 2
            result1 = await test_failing_function(5)
            try:
                result2 = await test_failing_function(-1)
                error_test_passed = False
            except ValueError:
                error_test_passed = True
            function_report = function_monitor.get_function_interaction_report()
            test_passed = error_test_passed and function_report.failed_calls > 0 and (len(function_report.error_summary) > 0) and (result1 == 10)
            print(f'   - Successful calls: {function_report.successful_calls}')
            print(f'   - Failed calls: {function_report.failed_calls}')
            print(f'   - Error summary: {function_report.error_summary}')
            print(f'   - Error handling test passed: {error_test_passed}')
            return {'test_name': 'error_handling_and_recovery', 'passed': test_passed, 'function_report': function_report, 'error_test_passed': error_test_passed}
        except Exception as e:
            print(f'❌ Error handling test failed: {e}')
            return {'test_name': 'error_handling_and_recovery', 'passed': False, 'error': str(e)}

    async def test_performance_monitoring(self) -> Dict[str, Any]:
        """Test performance monitoring capabilities."""
        print('\n⚡ Testing Performance Monitoring...')
        try:
            function_monitor.active_calls.clear()
            function_monitor.completed_calls.clear()
            function_monitor.call_counter = 0

            @comprehensive_function_monitoring(validate_inputs=True, validate_outputs=True, track_performance=True, timeout_seconds=30, retry_attempts=1)
            async def test_performance_function(duration: float) -> Dict[str, Any]:
                """Function with controllable performance characteristics."""
                await asyncio.sleep(duration)
                return {'duration': duration, 'timestamp': time.time()}
            await test_performance_function(0.01)
            await test_performance_function(0.1)
            await test_performance_function(0.05)
            function_report = function_monitor.get_function_interaction_report()
            test_passed = function_report.total_calls == 3 and function_report.performance_metrics.get('fastest_call') is not None and (function_report.performance_metrics.get('slowest_call') is not None) and (function_report.performance_metrics.get('median_execution_time', 0) > 0)
            print(f'   - Total calls: {function_report.total_calls}')
            print(f"   - Fastest call: {function_report.performance_metrics.get('fastest_call')}")
            print(f"   - Slowest call: {function_report.performance_metrics.get('slowest_call')}")
            print(f"   - Median execution time: {function_report.performance_metrics.get('median_execution_time', 0):.3f}s")
            print(f'   - Average execution time: {function_report.average_execution_time:.3f}s')
            return {'test_name': 'performance_monitoring', 'passed': test_passed, 'function_report': function_report}
        except Exception as e:
            print(f'❌ Performance monitoring test failed: {e}')
            return {'test_name': 'performance_monitoring', 'passed': False, 'error': str(e)}

    async def test_validation_framework_integration(self) -> Dict[str, Any]:
        """Test integration with validation framework."""
        print('\n🔍 Testing Validation Framework Integration...')
        try:
            function_monitor.active_calls.clear()
            function_monitor.completed_calls.clear()
            function_monitor.call_counter = 0
            training_input = {'symbol': 'ETHUSDT', 'exchange': 'BINANCE', 'timeframe': '1m', 'data_dir': str(self.temp_dir)}
            pipeline_state = {}
            validation_result = await run_validator(training_input, pipeline_state)
            report_result = await generate_validation_function_report(training_input, validation_result, str(self.temp_dir))
            function_report = function_monitor.get_function_interaction_report()
            test_passed = validation_result.get('validation_passed', False) and report_result.get('success', False) and (function_report.total_calls > 0) and ('validation_monitoring' in str(report_result.get('report_path', '')))
            print(f"   - Validation passed: {validation_result.get('validation_passed', False)}")
            print(f"   - Report generation success: {report_result.get('success', False)}")
            print(f'   - Function calls during validation: {function_report.total_calls}')
            print(f"   - Report path: {report_result.get('report_path', 'N/A')}")
            return {'test_name': 'validation_framework_integration', 'passed': test_passed, 'validation_result': validation_result, 'report_result': report_result, 'function_report': function_report}
        except Exception as e:
            print(f'❌ Validation framework integration test failed: {e}')
            return {'test_name': 'validation_framework_integration', 'passed': False, 'error': str(e)}

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and generate comprehensive report."""
        print('🚀 Starting Comprehensive Step02 Enhanced Monitoring Tests...')
        print('=' * 80)
        test_methods = [self.test_function_call_monitoring, self.test_function_interaction_tracking, self.test_error_handling_and_recovery, self.test_performance_monitoring, self.test_validation_framework_integration]
        all_results = {}
        passed_tests = 0
        total_tests = len(test_methods)
        for test_method in test_methods:
            try:
                result = await test_method()
                all_results[result['test_name']] = result
                if result['passed']:
                    passed_tests += 1
                    print(f"✅ {result['test_name']}: PASSED")
                else:
                    print(f"❌ {result['test_name']}: FAILED")
                    if 'error' in result:
                        print(f"   Error: {result['error']}")
            except Exception as e:
                print(f'❌ {test_method.__name__}: EXCEPTION - {e}')
                all_results[test_method.__name__] = {'test_name': test_method.__name__, 'passed': False, 'error': str(e)}
        print('\n' + '=' * 80)
        print('📊 COMPREHENSIVE TEST RESULTS')
        print('=' * 80)
        print(f'Total Tests: {total_tests}')
        print(f'Passed: {passed_tests}')
        print(f'Failed: {total_tests - passed_tests}')
        print(f'Success Rate: {passed_tests / total_tests * 100:.1f}%')
        print('\n📋 Detailed Results:')
        for test_name, result in all_results.items():
            status = '✅ PASSED' if result['passed'] else '❌ FAILED'
            print(f'   {test_name}: {status}')
            if 'error' in result:
                print(f"     Error: {result['error']}")
        return {'total_tests': total_tests, 'passed_tests': passed_tests, 'failed_tests': total_tests - passed_tests, 'success_rate': passed_tests / total_tests * 100, 'detailed_results': all_results}

async def main() -> None:
    """Main test execution function."""
    tester = Step02EnhancedMonitoringTester()
    try:
        results = await tester.run_all_tests()
        print('\n' + '=' * 80)
        if results['success_rate'] == 100:
            print('🎉 ALL TESTS PASSED! Step02 Enhanced Monitoring is working perfectly!')
        elif results['success_rate'] >= 80:
            print('✅ MOSTLY SUCCESSFUL! Step02 Enhanced Monitoring is working well with minor issues.')
        else:
            print('⚠️ SOME ISSUES DETECTED! Step02 Enhanced Monitoring needs attention.')
        print(f"Overall Success Rate: {results['success_rate']:.1f}%")
        print('=' * 80)
        return results
    finally:
        tester.cleanup_test_environment()
if __name__ == '__main__':
    results = asyncio.run(main())
    if results['success_rate'] == 100:
        sys.exit(0)
    else:
        sys.exit(1)