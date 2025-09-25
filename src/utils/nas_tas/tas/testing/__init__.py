"""
Comprehensive Testing Framework Module for TAS to NAS Parity

This module provides comprehensive testing and benchmarking capabilities to ensure
TAS system achieves full parity with state-of-the-art NAS systems.
"""

from .comprehensive_testing_framework import (
    ComprehensiveTestingFramework,
    UnitTestSuite,
    IntegrationTestSuite,
    NASBenchmarkSuite,
    TestingConfig,
    TestResult,
    BenchmarkResult,
    create_comprehensive_testing_framework,
    create_unit_test_suite,
    create_integration_test_suite,
    create_nas_benchmark_suite
)

__all__ = [
    'ComprehensiveTestingFramework',
    'UnitTestSuite',
    'IntegrationTestSuite',
    'NASBenchmarkSuite',
    'TestingConfig',
    'TestResult',
    'BenchmarkResult',
    'create_comprehensive_testing_framework',
    'create_unit_test_suite',
    'create_integration_test_suite',
    'create_nas_benchmark_suite'
]