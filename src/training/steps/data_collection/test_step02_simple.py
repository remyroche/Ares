"""
Simple Test Suite for Enhanced Step02 Function Monitoring

This script tests the core function monitoring mechanisms without external dependencies.
"""
import asyncio
import sys
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
try:
    from src.training.steps.data_collection.step02_data_reading import FunctionCallMonitor, FunctionCallStatus, FunctionCallContext, FunctionInteractionReport, comprehensive_function_monitoring
    print('✅ Successfully imported function monitoring components')
except ImportError as e:
    print(f'❌ Failed to import function monitoring components: {e}')
    sys.exit(1)

class SimpleStep02Tester:
    """Simple test suite for Step02 function monitoring."""

    def __init__(self) -> None:
        self.test_results = {}
        self.function_monitor = FunctionCallMonitor()

    async def test_basic_function_monitoring(self) -> Dict[str, Any]:
        """Test basic function call monitoring."""
        print('\n🔍 Testing Basic Function Call Monitoring...')
        try:
            self.function_monitor.active_calls.clear()
            self.function_monitor.completed_calls.clear()
            self.function_monitor.call_counter = 0

            @comprehensive_function_monitoring(validate_inputs=True, validate_outputs=True, track_performance=True, timeout_seconds=30, retry_attempts=1)
            async def test_function(x: int) -> int:
                """Simple test function."""
                await asyncio.sleep(0.01)
                return x * 2
            result = await test_function(5)
            report = self.function_monitor.get_function_interaction_report()
            test_passed = result == 10 and report.total_calls == 1 and (report.successful_calls == 1) and (report.failed_calls == 0)
            print(f'   - Function result: {result}')
            print(f'   - Total calls: {report.total_calls}')
            print(f'   - Successful calls: {report.successful_calls}')
            print(f'   - Failed calls: {report.failed_calls}')
            print(f"   - Success rate: {report.performance_metrics.get('success_rate', 0):.1f}%")
            return {'test_name': 'basic_function_monitoring', 'passed': test_passed, 'result': result, 'report': report}
        except Exception as e:
            print(f'❌ Basic function monitoring test failed: {e}')
            return {'test_name': 'basic_function_monitoring', 'passed': False, 'error': str(e)}

    async def test_function_interaction_tracking(self) -> Dict[str, Any]:
        """Test function-to-function call tracking."""
        print('\n🔗 Testing Function Interaction Tracking...')
        try:
            self.function_monitor.active_calls.clear()
            self.function_monitor.completed_calls.clear()
            self.function_monitor.call_counter = 0

            @comprehensive_function_monitoring(validate_inputs=True, validate_outputs=True, track_performance=True, timeout_seconds=30, retry_attempts=1)
            async def parent_function(x: int) -> Dict[str, Any]:
                """Parent function."""
                child1_result = await child_function_1(x)
                child2_result = await child_function_2(x)
                return {'parent': x, 'child1': child1_result, 'child2': child2_result}

            @comprehensive_function_monitoring(validate_inputs=True, validate_outputs=True, track_performance=True, timeout_seconds=30, retry_attempts=1)
            async def child_function_1(x: int) -> int:
                """Child function 1."""
                await asyncio.sleep(0.01)
                return x + 1

            @comprehensive_function_monitoring(validate_inputs=True, validate_outputs=True, track_performance=True, timeout_seconds=30, retry_attempts=1)
            async def child_function_2(x: int) -> int:
                """Child function 2."""
                await asyncio.sleep(0.01)
                return x * 2
            result = await parent_function(5)
            report = self.function_monitor.get_function_interaction_report()
            test_passed = report.total_calls >= 3 and len(report.call_hierarchy) > 0 and (report.performance_metrics.get('max_call_depth', 0) > 0)
            print(f'   - Total calls: {report.total_calls}')
            print(f'   - Call hierarchy entries: {len(report.call_hierarchy)}')
            print(f"   - Max call depth: {report.performance_metrics.get('max_call_depth', 0)}")
            print(f"   - Function frequency: {report.performance_metrics.get('function_frequency', {})}")
            return {'test_name': 'function_interaction_tracking', 'passed': test_passed, 'result': result, 'report': report}
        except Exception as e:
            print(f'❌ Function interaction tracking test failed: {e}')
            return {'test_name': 'function_interaction_tracking', 'passed': False, 'error': str(e)}

    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and recovery."""
        print('\n⚠️ Testing Error Handling...')
        try:
            self.function_monitor.active_calls.clear()
            self.function_monitor.completed_calls.clear()
            self.function_monitor.call_counter = 0

            @comprehensive_function_monitoring(validate_inputs=True, validate_outputs=True, track_performance=True, timeout_seconds=30, retry_attempts=1)
            async def failing_function(x: int) -> int:
                """Function that fails with negative input."""
                if x < 0:
                    raise ValueError('Negative value not allowed')
                return x * 2
            result1 = await failing_function(5)
            try:
                result2 = await failing_function(-1)
                error_handled = False
            except ValueError:
                error_handled = True
            report = self.function_monitor.get_function_interaction_report()
            test_passed = result1 == 10 and error_handled and (report.failed_calls > 0) and (len(report.error_summary) > 0)
            print(f'   - Successful call result: {result1}')
            print(f'   - Error handled correctly: {error_handled}')
            print(f'   - Failed calls: {report.failed_calls}')
            print(f'   - Error summary: {report.error_summary}')
            return {'test_name': 'error_handling', 'passed': test_passed, 'result1': result1, 'error_handled': error_handled, 'report': report}
        except Exception as e:
            print(f'❌ Error handling test failed: {e}')
            return {'test_name': 'error_handling', 'passed': False, 'error': str(e)}

    async def test_performance_monitoring(self) -> Dict[str, Any]:
        """Test performance monitoring."""
        print('\n⚡ Testing Performance Monitoring...')
        try:
            self.function_monitor.active_calls.clear()
            self.function_monitor.completed_calls.clear()
            self.function_monitor.call_counter = 0

            @comprehensive_function_monitoring(validate_inputs=True, validate_outputs=True, track_performance=True, timeout_seconds=30, retry_attempts=1)
            async def performance_function(duration: float) -> Dict[str, Any]:
                """Function with controllable duration."""
                await asyncio.sleep(duration)
                return {'duration': duration, 'timestamp': time.time()}
            await performance_function(0.01)
            await performance_function(0.05)
            await performance_function(0.02)
            report = self.function_monitor.get_function_interaction_report()
            test_passed = report.total_calls == 3 and report.performance_metrics.get('fastest_call') is not None and (report.performance_metrics.get('slowest_call') is not None) and (report.performance_metrics.get('median_execution_time', 0) > 0)
            print(f'   - Total calls: {report.total_calls}')
            print(f"   - Fastest call: {report.performance_metrics.get('fastest_call')}")
            print(f"   - Slowest call: {report.performance_metrics.get('slowest_call')}")
            print(f"   - Median execution time: {report.performance_metrics.get('median_execution_time', 0):.3f}s")
            print(f'   - Average execution time: {report.average_execution_time:.3f}s')
            return {'test_name': 'performance_monitoring', 'passed': test_passed, 'report': report}
        except Exception as e:
            print(f'❌ Performance monitoring test failed: {e}')
            return {'test_name': 'performance_monitoring', 'passed': False, 'error': str(e)}

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests."""
        print('🚀 Starting Simple Step02 Function Monitoring Tests...')
        print('=' * 60)
        test_methods = [self.test_basic_function_monitoring, self.test_function_interaction_tracking, self.test_error_handling, self.test_performance_monitoring]
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
        print('\n' + '=' * 60)
        print('📊 TEST RESULTS SUMMARY')
        print('=' * 60)
        print(f'Total Tests: {total_tests}')
        print(f'Passed: {passed_tests}')
        print(f'Failed: {total_tests - passed_tests}')
        print(f'Success Rate: {passed_tests / total_tests * 100:.1f}%')
        return {'total_tests': total_tests, 'passed_tests': passed_tests, 'failed_tests': total_tests - passed_tests, 'success_rate': passed_tests / total_tests * 100, 'detailed_results': all_results}

async def main() -> None:
    """Main test execution."""
    tester = SimpleStep02Tester()
    try:
        results = await tester.run_all_tests()
        print('\n' + '=' * 60)
        if results['success_rate'] == 100:
            print('🎉 ALL TESTS PASSED! Step02 Function Monitoring is working perfectly!')
        elif results['success_rate'] >= 75:
            print('✅ MOSTLY SUCCESSFUL! Step02 Function Monitoring is working well.')
        else:
            print('⚠️ SOME ISSUES DETECTED! Step02 Function Monitoring needs attention.')
        print(f"Overall Success Rate: {results['success_rate']:.1f}%")
        print('=' * 60)
        return results
    except Exception as e:
        print(f'❌ Test execution failed: {e}')
        return {'success_rate': 0, 'error': str(e)}
if __name__ == '__main__':
    results = asyncio.run(main())
    if results.get('success_rate', 0) == 100:
        sys.exit(0)
    else:
        sys.exit(1)