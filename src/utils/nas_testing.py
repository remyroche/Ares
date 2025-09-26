#!/usr/bin/env python3
"""
Comprehensive Testing Framework for NAS Components

This module provides a comprehensive testing framework with unit tests,
integration tests, performance tests, and stress tests for NAS components.
"""

import unittest
import time
import threading
import concurrent.futures
from typing import Any, Callable, Dict, List, Optional, Union, Type, TypeVar
from dataclasses import dataclass, field
from contextlib import contextmanager
import logging
import tempfile
import shutil
from pathlib import Path
import json
import pickle
import numpy as np
import psutil

from .nas_error_handling import (
    NASBaseException, ErrorContext, error_context, 
    safe_execute, get_error_handler
)
from .nas_threading import ThreadSafeCounter, ThreadSafeCache, ThreadSafeQueue
from .nas_resource_manager import ResourceManager, ResourceType
from .nas_performance import PerformanceProfiler, PerformanceMetrics

T = TypeVar('T')


@dataclass
class TestResult:
    """Result of a test execution."""
    test_name: str
    success: bool
    duration: float
    error_message: Optional[str] = None
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuite:
    """A collection of related tests."""
    name: str
    tests: List[Callable]
    setup_func: Optional[Callable] = None
    teardown_func: Optional[Callable] = None
    timeout: float = 300.0  # 5 minutes default timeout


class NASTestCase(unittest.TestCase):
    """Base test case for NAS components with enhanced functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self._error_handler = get_error_handler()
        self._logger = logging.getLogger(__name__)
        self._temp_dir = tempfile.mkdtemp()
        self._test_start_time = time.time()
        self._test_memory_before = self._get_memory_usage()
    
    def tearDown(self):
        """Clean up test environment."""
        try:
            # Clean up temporary directory
            if hasattr(self, '_temp_dir') and Path(self._temp_dir).exists():
                shutil.rmtree(self._temp_dir)
            
            # Log test performance
            test_duration = time.time() - self._test_start_time
            test_memory_after = self._get_memory_usage()
            memory_delta = test_memory_after - self._test_memory_before
            
            self._logger.info(
                f"Test {self._testMethodName}: {test_duration:.3f}s, "
                f"Memory delta: {memory_delta:.1f}MB"
            )
            
        except Exception as e:
            self._logger.error(f"Error in tearDown: {e}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def assert_within_range(
        self,
        value: float,
        min_val: float,
        max_val: float,
        message: str = None
    ):
        """Assert that a value is within a specified range."""
        if not (min_val <= value <= max_val):
            msg = f"{value} not in range [{min_val}, {max_val}]"
            if message:
                msg = f"{message}: {msg}"
            self.fail(msg)
    
    def assert_memory_usage(
        self,
        max_memory_mb: float,
        message: str = None
    ):
        """Assert that memory usage is within limits."""
        current_memory = self._get_memory_usage()
        if current_memory > max_memory_mb:
            msg = f"Memory usage {current_memory:.1f}MB exceeds limit {max_memory_mb}MB"
            if message:
                msg = f"{message}: {msg}"
            self.fail(msg)
    
    def assert_performance(
        self,
        operation: Callable,
        max_duration: float,
        *args,
        **kwargs
    ):
        """Assert that an operation completes within time limit."""
        start_time = time.time()
        try:
            result = operation(*args, **kwargs)
            duration = time.time() - start_time
            
            if duration > max_duration:
                self.fail(f"Operation took {duration:.3f}s, exceeds limit {max_duration}s")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.fail(f"Operation failed after {duration:.3f}s: {e}")
    
    def assert_thread_safe(
        self,
        operation: Callable,
        num_threads: int = 10,
        num_iterations: int = 100,
        *args,
        **kwargs
    ):
        """Assert that an operation is thread-safe."""
        results = []
        errors = []
        
        def worker():
            try:
                for _ in range(num_iterations):
                    result = operation(*args, **kwargs)
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        if errors:
            self.fail(f"Thread safety test failed with {len(errors)} errors: {errors[0]}")
        
        # Check for consistency in results
        if results and len(set(str(r) for r in results)) > 1:
            self.fail("Thread safety test failed: inconsistent results")
    
    def assert_resource_cleanup(
        self,
        operation: Callable,
        resource_type: ResourceType,
        *args,
        **kwargs
    ):
        """Assert that an operation properly cleans up resources."""
        resource_manager = ResourceManager()
        initial_stats = resource_manager.get_resource_stats()
        
        try:
            operation(*args, **kwargs)
        finally:
            final_stats = resource_manager.get_resource_stats()
            
            # Check that resources were cleaned up
            initial_count = initial_stats.get('tracker_stats', {}).get('total_resources', 0)
            final_count = final_stats.get('tracker_stats', {}).get('total_resources', 0)
            
            if final_count > initial_count:
                self.fail(f"Resource cleanup failed: {final_count - initial_count} resources not cleaned up")
    
    def assert_error_handling(
        self,
        operation: Callable,
        expected_exception: Type[Exception],
        *args,
        **kwargs
    ):
        """Assert that an operation raises the expected exception."""
        with self.assertRaises(expected_exception):
            operation(*args, **kwargs)
    
    def assert_no_memory_leaks(
        self,
        operation: Callable,
        num_iterations: int = 10,
        max_memory_growth_mb: float = 10.0,
        *args,
        **kwargs
    ):
        """Assert that an operation doesn't cause memory leaks."""
        initial_memory = self._get_memory_usage()
        
        for _ in range(num_iterations):
            operation(*args, **kwargs)
            # Force garbage collection
            import gc
            gc.collect()
        
        final_memory = self._get_memory_usage()
        memory_growth = final_memory - initial_memory
        
        if memory_growth > max_memory_growth_mb:
            self.fail(f"Memory leak detected: {memory_growth:.1f}MB growth over {num_iterations} iterations")


class PerformanceTestCase(NASTestCase):
    """Test case for performance testing."""
    
    def setUp(self):
        super().setUp()
        self._profiler = PerformanceProfiler()
    
    def assert_performance_improvement(
        self,
        original_func: Callable,
        optimized_func: Callable,
        test_data: List[Any],
        min_improvement_factor: float = 1.5
    ):
        """Assert that optimized function is faster than original."""
        # Profile original function
        original_time = self._profiler.profile_operation(
            "original", original_func, test_data
        ).duration
        
        # Profile optimized function
        optimized_time = self._profiler.profile_operation(
            "optimized", optimized_func, test_data
        ).duration
        
        improvement_factor = original_time / optimized_time
        
        if improvement_factor < min_improvement_factor:
            self.fail(
                f"Performance improvement insufficient: {improvement_factor:.2f}x "
                f"(expected >= {min_improvement_factor}x)"
            )
    
    def assert_scalability(
        self,
        operation: Callable,
        data_sizes: List[int],
        max_time_per_element: float = 0.001
    ):
        """Assert that operation scales linearly with data size."""
        times = []
        
        for size in data_sizes:
            test_data = list(range(size))
            start_time = time.time()
            operation(test_data)
            duration = time.time() - start_time
            times.append(duration)
        
        # Check that time per element doesn't exceed limit
        for size, duration in zip(data_sizes, times):
            time_per_element = duration / size
            if time_per_element > max_time_per_element:
                self.fail(
                    f"Scalability test failed: {time_per_element:.6f}s per element "
                    f"exceeds limit {max_time_per_element}s"
                )


class StressTestCase(NASTestCase):
    """Test case for stress testing."""
    
    def assert_stress_resistance(
        self,
        operation: Callable,
        stress_levels: List[int],
        max_failure_rate: float = 0.1,
        *args,
        **kwargs
    ):
        """Assert that operation can handle stress levels."""
        for stress_level in stress_levels:
            failures = 0
            total_attempts = 100
            
            for _ in range(total_attempts):
                try:
                    # Simulate stress by running multiple operations concurrently
                    with concurrent.futures.ThreadPoolExecutor(max_workers=stress_level) as executor:
                        futures = [
                            executor.submit(operation, *args, **kwargs)
                            for _ in range(stress_level)
                        ]
                        
                        # Wait for all operations to complete
                        for future in futures:
                            future.result(timeout=30.0)
                            
                except Exception:
                    failures += 1
            
            failure_rate = failures / total_attempts
            
            if failure_rate > max_failure_rate:
                self.fail(
                    f"Stress test failed at level {stress_level}: "
                    f"{failure_rate:.2%} failure rate exceeds {max_failure_rate:.2%}"
                )
    
    def assert_memory_stress(
        self,
        operation: Callable,
        max_memory_mb: float = 1000.0,
        *args,
        **kwargs
    ):
        """Assert that operation can handle memory stress."""
        initial_memory = self._get_memory_usage()
        
        try:
            operation(*args, **kwargs)
        except MemoryError:
            self.fail("Operation failed due to memory stress")
        
        final_memory = self._get_memory_usage()
        memory_usage = final_memory - initial_memory
        
        if memory_usage > max_memory_mb:
            self.fail(f"Memory usage {memory_usage:.1f}MB exceeds limit {max_memory_mb}MB")


class IntegrationTestCase(NASTestCase):
    """Test case for integration testing."""
    
    def assert_component_integration(
        self,
        components: List[Any],
        integration_test: Callable
    ):
        """Assert that components integrate correctly."""
        try:
            result = integration_test(components)
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"Integration test failed: {e}")
    
    def assert_data_flow(
        self,
        data_source: Callable,
        data_processor: Callable,
        data_sink: Callable,
        expected_output: Any
    ):
        """Assert that data flows correctly through the pipeline."""
        try:
            # Get data from source
            data = data_source()
            self.assertIsNotNone(data)
            
            # Process data
            processed_data = data_processor(data)
            self.assertIsNotNone(processed_data)
            
            # Send to sink
            output = data_sink(processed_data)
            self.assertIsNotNone(output)
            
            # Verify output
            if expected_output is not None:
                self.assertEqual(output, expected_output)
                
        except Exception as e:
            self.fail(f"Data flow test failed: {e}")


class TestRunner:
    """Runs tests and generates reports."""
    
    def __init__(self):
        self._error_handler = get_error_handler()
        self._logger = logging.getLogger(__name__)
        self._test_results: List[TestResult] = []
    
    def run_test_suite(self, test_suite: TestSuite) -> List[TestResult]:
        """Run a test suite and return results."""
        results = []
        
        try:
            # Setup
            if test_suite.setup_func:
                test_suite.setup_func()
            
            # Run tests
            for test_func in test_suite.tests:
                result = self._run_single_test(test_func, test_suite.timeout)
                results.append(result)
            
            # Teardown
            if test_suite.teardown_func:
                test_suite.teardown_func()
                
        except Exception as e:
            context = ErrorContext("run_test_suite", "test_runner")
            self._error_handler.handle_error(e, context, reraise=False)
        
        self._test_results.extend(results)
        return results
    
    def _run_single_test(self, test_func: Callable, timeout: float) -> TestResult:
        """Run a single test function."""
        test_name = test_func.__name__
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        try:
            # Run test with timeout
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(test_func)
                result = future.result(timeout=timeout)
            
            success = True
            error_message = None
            
        except Exception as e:
            success = False
            error_message = str(e)
            result = None
        
        end_time = time.time()
        memory_after = self._get_memory_usage()
        
        return TestResult(
            test_name=test_name,
            success=success,
            duration=end_time - start_time,
            error_message=error_message,
            memory_usage=memory_after - memory_before
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate test report."""
        if not self._test_results:
            return {}
        
        total_tests = len(self._test_results)
        successful_tests = sum(1 for r in self._test_results if r.success)
        failed_tests = total_tests - successful_tests
        
        avg_duration = np.mean([r.duration for r in self._test_results])
        avg_memory = np.mean([r.memory_usage for r in self._test_results])
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'success_rate': successful_tests / total_tests,
            'avg_duration': avg_duration,
            'avg_memory_usage': avg_memory,
            'test_results': [
                {
                    'name': r.test_name,
                    'success': r.success,
                    'duration': r.duration,
                    'memory_usage': r.memory_usage,
                    'error_message': r.error_message
                }
                for r in self._test_results
            ]
        }
    
    def save_report(self, file_path: str) -> None:
        """Save test report to file."""
        try:
            report = self.generate_report()
            
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2)
                
        except Exception as e:
            context = ErrorContext("save_report", "test_runner")
            self._error_handler.handle_error(e, context, reraise=False)


class MockDataGenerator:
    """Generates mock data for testing."""
    
    @staticmethod
    def generate_random_data(size: int, data_type: str = "float") -> List[Any]:
        """Generate random data of specified type and size."""
        if data_type == "float":
            return np.random.random(size).tolist()
        elif data_type == "int":
            return np.random.randint(0, 1000, size).tolist()
        elif data_type == "string":
            return [f"test_string_{i}" for i in range(size)]
        elif data_type == "dict":
            return [{"id": i, "value": np.random.random()} for i in range(size)]
        else:
            return list(range(size))
    
    @staticmethod
    def generate_test_model():
        """Generate a test model for testing."""
        # This would be implemented based on the actual model structure
        return {"layers": 3, "parameters": 1000, "accuracy": 0.95}
    
    @staticmethod
    def generate_test_dataset(size: int = 1000):
        """Generate a test dataset."""
        return {
            "features": np.random.random((size, 10)),
            "labels": np.random.randint(0, 2, size),
            "metadata": {"size": size, "features": 10, "classes": 2}
        }


class TestUtilities:
    """Utility functions for testing."""
    
    @staticmethod
    def create_temp_file(content: str = "test content") -> str:
        """Create a temporary file with content."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(content)
            return f.name
    
    @staticmethod
    def create_temp_directory() -> str:
        """Create a temporary directory."""
        return tempfile.mkdtemp()
    
    @staticmethod
    def cleanup_temp_file(file_path: str) -> None:
        """Clean up a temporary file."""
        try:
            Path(file_path).unlink()
        except Exception:
            pass
    
    @staticmethod
    def cleanup_temp_directory(dir_path: str) -> None:
        """Clean up a temporary directory."""
        try:
            shutil.rmtree(dir_path)
        except Exception:
            pass
    
    @staticmethod
    def assert_files_equal(file1: str, file2: str) -> bool:
        """Assert that two files have equal content."""
        try:
            with open(file1, 'r') as f1, open(file2, 'r') as f2:
                return f1.read() == f2.read()
        except Exception:
            return False
    
    @staticmethod
    def assert_directories_equal(dir1: str, dir2: str) -> bool:
        """Assert that two directories have equal content."""
        try:
            files1 = set(Path(dir1).rglob('*'))
            files2 = set(Path(dir2).rglob('*'))
            
            if files1 != files2:
                return False
            
            for file1 in files1:
                if file1.is_file():
                    file2 = Path(dir2) / file1.relative_to(Path(dir1))
                    if not TestUtilities.assert_files_equal(str(file1), str(file2)):
                        return False
            
            return True
        except Exception:
            return False


# Global test runner
_global_test_runner = TestRunner()


def run_tests(test_suite: TestSuite) -> List[TestResult]:
    """Run a test suite."""
    return _global_test_runner.run_test_suite(test_suite)


def generate_test_report() -> Dict[str, Any]:
    """Generate test report."""
    return _global_test_runner.generate_report()


def save_test_report(file_path: str) -> None:
    """Save test report to file."""
    _global_test_runner.save_report(file_path)


def create_test_suite(
    name: str,
    tests: List[Callable],
    setup_func: Optional[Callable] = None,
    teardown_func: Optional[Callable] = None,
    timeout: float = 300.0
) -> TestSuite:
    """Create a test suite."""
    return TestSuite(name, tests, setup_func, teardown_func, timeout)


# Export main classes and functions
__all__ = [
    'TestResult',
    'TestSuite',
    'NASTestCase',
    'PerformanceTestCase',
    'StressTestCase',
    'IntegrationTestCase',
    'TestRunner',
    'MockDataGenerator',
    'TestUtilities',
    'run_tests',
    'generate_test_report',
    'save_test_report',
    'create_test_suite'
]