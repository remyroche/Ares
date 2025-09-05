#!/usr/bin/env python3
"""
Enhanced Testing Pipeline

This pipeline integrates all testing scripts and provides comprehensive testing capabilities
for the code quality system. It includes:

1. Common operations testing
2. Real subset testing
3. Subset testing
4. Simple testing
5. Mock-based testing
6. Integration testing
7. Pipeline testing
8. Tools testing
9. Enhanced analyzer testing
10. Import analysis testing
11. Dead code integration testing
12. Duplicate import fixer testing
13. Intelligent import fixer testing
14. Simple enhanced analyzer testing

All tests are executed with proper error handling, reporting, and integration with the
plugin system.
"""

import ast
import json
import sys
import time
import subprocess
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_pipeline import BasePipeline, PipelineConfig
from plugins import PluginManager, PluginContext, PluginResult


@dataclass
class TestResult:
    """Result of a test execution."""
    test_name: str
    status: str  # 'passed', 'failed', 'skipped', 'error'
    duration: float
    output: str
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestingConfig(PipelineConfig):
    """Configuration for testing pipeline."""
    test_timeout: int = 300
    parallel_tests: bool = True
    max_test_workers: int = 4
    include_mock_tests: bool = True
    include_integration_tests: bool = True
    include_pipeline_tests: bool = True
    include_tool_tests: bool = True
    test_coverage: bool = True
    verbose_output: bool = False


class EnhancedTestingPipeline(BasePipeline):
    """Enhanced testing pipeline with comprehensive test integration."""
    
    def __init__(self, config: TestingConfig):
        super().__init__(config)
        self.config = config
        self.test_results: List[TestResult] = []
        self.test_scripts = self._discover_test_scripts()
        
    def _discover_test_scripts(self) -> Dict[str, str]:
        """Discover all test scripts in the code_quality directory."""
        test_scripts = {}
        code_quality_dir = Path(__file__).parent.parent
        
        # Main test scripts
        main_tests = {
            'run_common_operations_tests': 'run_common_operations_tests.py',
            'run_real_subset_tests': 'run_real_subset_tests.py',
            'run_subset_tests': 'run_subset_tests.py',
            'run_tests_simple': 'run_tests_simple.py',
            'run_tests_with_mocks': 'run_tests_with_mocks.py',
            'test_integration': 'test_integration.py',
            'test_pipeline': 'test_pipeline.py',
            'test_pipeline_simple': 'test_pipeline_simple.py',
            'test_tools': 'test_tools.py',
            'test_enhanced_analyzer': 'test_enhanced_analyzer.py',
            'test_enhanced_import_analysis': 'test_enhanced_import_analysis.py',
            'test_dead_code_integration': 'test_dead_code_integration.py',
            'test_duplicate_import_fixer': 'test_duplicate_import_fixer.py',
            'test_intelligent_import_fixer': 'test_intelligent_import_fixer.py',
            'test_simple_enhanced_analyzer': 'test_simple_enhanced_analyzer.py',
        }
        
        # Verify scripts exist and add to test_scripts
        for test_name, script_name in main_tests.items():
            script_path = code_quality_dir / script_name
            if script_path.exists():
                test_scripts[test_name] = str(script_path)
            else:
                self.logger.warning(f"Test script not found: {script_path}")
        
        return test_scripts
    
    def run_test_script(self, test_name: str, script_path: str) -> TestResult:
        """Run a single test script and return the result."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Running test: {test_name}")
            
            # Run the test script
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=str(Path(__file__).parent.parent),
                capture_output=True,
                text=True,
                timeout=self.config.test_timeout
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                status = 'passed'
                error = None
            else:
                status = 'failed'
                error = result.stderr
            
            return TestResult(
                test_name=test_name,
                status=status,
                duration=duration,
                output=result.stdout,
                error=error,
                details={
                    'returncode': result.returncode,
                    'script_path': script_path
                }
            )
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return TestResult(
                test_name=test_name,
                status='error',
                duration=duration,
                output='',
                error=f'Test timed out after {self.config.test_timeout} seconds',
                details={'script_path': script_path}
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name=test_name,
                status='error',
                duration=duration,
                output='',
                error=str(e),
                details={'script_path': script_path}
            )
    
    def run_common_operations_tests(self) -> TestResult:
        """Run common operations tests."""
        if 'run_common_operations_tests' not in self.test_scripts:
            return TestResult(
                test_name='run_common_operations_tests',
                status='skipped',
                duration=0.0,
                output='Test script not found',
                error='Script not available'
            )
        
        return self.run_test_script(
            'run_common_operations_tests',
            self.test_scripts['run_common_operations_tests']
        )
    
    def run_real_subset_tests(self) -> TestResult:
        """Run real subset tests."""
        if 'run_real_subset_tests' not in self.test_scripts:
            return TestResult(
                test_name='run_real_subset_tests',
                status='skipped',
                duration=0.0,
                output='Test script not found',
                error='Script not available'
            )
        
        return self.run_test_script(
            'run_real_subset_tests',
            self.test_scripts['run_real_subset_tests']
        )
    
    def run_subset_tests(self) -> TestResult:
        """Run subset tests."""
        if 'run_subset_tests' not in self.test_scripts:
            return TestResult(
                test_name='run_subset_tests',
                status='skipped',
                duration=0.0,
                output='Test script not found',
                error='Script not available'
            )
        
        return self.run_test_script(
            'run_subset_tests',
            self.test_scripts['run_subset_tests']
        )
    
    def run_simple_tests(self) -> TestResult:
        """Run simple tests."""
        if 'run_tests_simple' not in self.test_scripts:
            return TestResult(
                test_name='run_tests_simple',
                status='skipped',
                duration=0.0,
                output='Test script not found',
                error='Script not available'
            )
        
        return self.run_test_script(
            'run_tests_simple',
            self.test_scripts['run_tests_simple']
        )
    
    def run_mock_tests(self) -> TestResult:
        """Run mock-based tests."""
        if not self.config.include_mock_tests:
            return TestResult(
                test_name='run_mock_tests',
                status='skipped',
                duration=0.0,
                output='Mock tests disabled',
                error=None
            )
        
        if 'run_tests_with_mocks' not in self.test_scripts:
            return TestResult(
                test_name='run_mock_tests',
                status='skipped',
                duration=0.0,
                output='Test script not found',
                error='Script not available'
            )
        
        return self.run_test_script(
            'run_mock_tests',
            self.test_scripts['run_tests_with_mocks']
        )
    
    def run_integration_tests(self) -> TestResult:
        """Run integration tests."""
        if not self.config.include_integration_tests:
            return TestResult(
                test_name='run_integration_tests',
                status='skipped',
                duration=0.0,
                output='Integration tests disabled',
                error=None
            )
        
        if 'test_integration' not in self.test_scripts:
            return TestResult(
                test_name='run_integration_tests',
                status='skipped',
                duration=0.0,
                output='Test script not found',
                error='Script not available'
            )
        
        return self.run_test_script(
            'run_integration_tests',
            self.test_scripts['test_integration']
        )
    
    def run_pipeline_tests(self) -> TestResult:
        """Run pipeline tests."""
        if not self.config.include_pipeline_tests:
            return TestResult(
                test_name='run_pipeline_tests',
                status='skipped',
                duration=0.0,
                output='Pipeline tests disabled',
                error=None
            )
        
        if 'test_pipeline' not in self.test_scripts:
            return TestResult(
                test_name='run_pipeline_tests',
                status='skipped',
                duration=0.0,
                output='Test script not found',
                error='Script not available'
            )
        
        return self.run_test_script(
            'run_pipeline_tests',
            self.test_scripts['test_pipeline']
        )
    
    def run_tool_tests(self) -> TestResult:
        """Run tool tests."""
        if not self.config.include_tool_tests:
            return TestResult(
                test_name='run_tool_tests',
                status='skipped',
                duration=0.0,
                output='Tool tests disabled',
                error=None
            )
        
        if 'test_tools' not in self.test_scripts:
            return TestResult(
                test_name='run_tool_tests',
                status='skipped',
                duration=0.0,
                output='Test script not found',
                error='Script not available'
            )
        
        return self.run_test_script(
            'run_tool_tests',
            self.test_scripts['test_tools']
        )
    
    def run_enhanced_analyzer_tests(self) -> TestResult:
        """Run enhanced analyzer tests."""
        if 'test_enhanced_analyzer' not in self.test_scripts:
            return TestResult(
                test_name='run_enhanced_analyzer_tests',
                status='skipped',
                duration=0.0,
                output='Test script not found',
                error='Script not available'
            )
        
        return self.run_test_script(
            'run_enhanced_analyzer_tests',
            self.test_scripts['test_enhanced_analyzer']
        )
    
    def run_import_analysis_tests(self) -> TestResult:
        """Run import analysis tests."""
        if 'test_enhanced_import_analysis' not in self.test_scripts:
            return TestResult(
                test_name='run_import_analysis_tests',
                status='skipped',
                duration=0.0,
                output='Test script not found',
                error='Script not available'
            )
        
        return self.run_test_script(
            'run_import_analysis_tests',
            self.test_scripts['test_enhanced_import_analysis']
        )
    
    def run_dead_code_tests(self) -> TestResult:
        """Run dead code integration tests."""
        if 'test_dead_code_integration' not in self.test_scripts:
            return TestResult(
                test_name='run_dead_code_tests',
                status='skipped',
                duration=0.0,
                output='Test script not found',
                error='Script not available'
            )
        
        return self.run_test_script(
            'run_dead_code_tests',
            self.test_scripts['test_dead_code_integration']
        )
    
    def run_import_fixer_tests(self) -> TestResult:
        """Run import fixer tests."""
        if 'test_duplicate_import_fixer' not in self.test_scripts:
            return TestResult(
                test_name='run_import_fixer_tests',
                status='skipped',
                duration=0.0,
                output='Test script not found',
                error='Script not available'
            )
        
        return self.run_test_script(
            'run_import_fixer_tests',
            self.test_scripts['test_duplicate_import_fixer']
        )
    
    def run_intelligent_import_fixer_tests(self) -> TestResult:
        """Run intelligent import fixer tests."""
        if 'test_intelligent_import_fixer' not in self.test_scripts:
            return TestResult(
                test_name='run_intelligent_import_fixer_tests',
                status='skipped',
                duration=0.0,
                output='Test script not found',
                error='Script not available'
            )
        
        return self.run_test_script(
            'run_intelligent_import_fixer_tests',
            self.test_scripts['test_intelligent_import_fixer']
        )
    
    def run_simple_enhanced_analyzer_tests(self) -> TestResult:
        """Run simple enhanced analyzer tests."""
        if 'test_simple_enhanced_analyzer' not in self.test_scripts:
            return TestResult(
                test_name='run_simple_enhanced_analyzer_tests',
                status='skipped',
                duration=0.0,
                output='Test script not found',
                error='Script not available'
            )
        
        return self.run_test_script(
            'run_simple_enhanced_analyzer_tests',
            self.test_scripts['test_simple_enhanced_analyzer']
        )
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all available tests."""
        start_time = time.time()
        
        self.logger.info("Starting comprehensive test execution")
        
        # Run all test categories
        test_methods = [
            self.run_common_operations_tests,
            self.run_real_subset_tests,
            self.run_subset_tests,
            self.run_simple_tests,
            self.run_mock_tests,
            self.run_integration_tests,
            self.run_pipeline_tests,
            self.run_tool_tests,
            self.run_enhanced_analyzer_tests,
            self.run_import_analysis_tests,
            self.run_dead_code_tests,
            self.run_import_fixer_tests,
            self.run_intelligent_import_fixer_tests,
            self.run_simple_enhanced_analyzer_tests,
        ]
        
        results = []
        for test_method in test_methods:
            try:
                result = test_method()
                results.append(result)
                self.test_results.append(result)
            except Exception as e:
                error_result = TestResult(
                    test_name=test_method.__name__,
                    status='error',
                    duration=0.0,
                    output='',
                    error=str(e)
                )
                results.append(error_result)
                self.test_results.append(error_result)
        
        # Calculate summary statistics
        total_tests = len(results)
        passed_tests = len([r for r in results if r.status == 'passed'])
        failed_tests = len([r for r in results if r.status == 'failed'])
        skipped_tests = len([r for r in results if r.status == 'skipped'])
        error_tests = len([r for r in results if r.status == 'error'])
        total_duration = sum(r.duration for r in results)
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'skipped_tests': skipped_tests,
            'error_tests': error_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'total_duration': total_duration,
            'execution_time': time.time() - start_time,
            'test_results': [
                {
                    'test_name': r.test_name,
                    'status': r.status,
                    'duration': r.duration,
                    'error': r.error,
                    'details': r.details
                }
                for r in results
            ]
        }
        
        self.logger.info(f"Test execution completed: {passed_tests}/{total_tests} passed")
        
        return summary
    
    def execute(self) -> Dict[str, Any]:
        """Execute the enhanced testing pipeline."""
        start_time = time.time()
        
        self.logger.info("Starting Enhanced Testing Pipeline")
        
        # Run all tests
        test_summary = self.run_all_tests()
        
        # Execute plugins if available
        plugin_results = {}
        if self.plugin_manager:
            try:
                context = PluginContext(
                    project_root=self.config.project_root,
                    output_dir=self.config.output_dir,
                    test_results=self.test_results
                )
                plugin_results = self.plugin_manager.execute_pipeline(
                    "enhanced_testing_pipeline",
                    context
                )
            except Exception as e:
                self.logger.warning(f"Plugin execution failed: {e}")
        
        # Generate final results
        results = {
            'pipeline_name': 'enhanced_testing_pipeline',
            'execution_time': time.time() - start_time,
            'test_summary': test_summary,
            'plugin_results': plugin_results,
            'configuration': {
                'test_timeout': self.config.test_timeout,
                'parallel_tests': self.config.parallel_tests,
                'max_test_workers': self.config.max_test_workers,
                'include_mock_tests': self.config.include_mock_tests,
                'include_integration_tests': self.config.include_integration_tests,
                'include_pipeline_tests': self.config.include_pipeline_tests,
                'include_tool_tests': self.config.include_tool_tests,
                'test_coverage': self.config.test_coverage,
                'verbose_output': self.config.verbose_output
            }
        }
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save test results to output directory."""
        output_file = self.config.output_dir / f"enhanced_testing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Test results saved to: {output_file}")


def main():
    """Main entry point for the enhanced testing pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Testing Pipeline")
    parser.add_argument("--project-root", type=str, default="/workspace", help="Project root directory")
    parser.add_argument("--output-dir", type=str, default="/workspace/code_quality/reports", help="Output directory")
    parser.add_argument("--test-timeout", type=int, default=300, help="Test timeout in seconds")
    parser.add_argument("--parallel-tests", action="store_true", help="Enable parallel test execution")
    parser.add_argument("--max-test-workers", type=int, default=4, help="Maximum test workers")
    parser.add_argument("--include-mock-tests", action="store_true", help="Include mock tests")
    parser.add_argument("--include-integration-tests", action="store_true", help="Include integration tests")
    parser.add_argument("--include-pipeline-tests", action="store_true", help="Include pipeline tests")
    parser.add_argument("--include-tool-tests", action="store_true", help="Include tool tests")
    parser.add_argument("--test-coverage", action="store_true", help="Enable test coverage")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    
    args = parser.parse_args()
    
    # Create configuration
    config = TestingConfig(
        project_root=Path(args.project_root),
        output_dir=Path(args.output_dir),
        test_timeout=args.test_timeout,
        parallel_tests=args.parallel_tests,
        max_test_workers=args.max_test_workers,
        include_mock_tests=args.include_mock_tests,
        include_integration_tests=args.include_integration_tests,
        include_pipeline_tests=args.include_pipeline_tests,
        include_tool_tests=args.include_tool_tests,
        test_coverage=args.test_coverage,
        verbose_output=args.verbose,
        dry_run=args.dry_run
    )
    
    # Create and run pipeline
    pipeline = EnhancedTestingPipeline(config)
    results = pipeline.execute()
    
    # Print summary
    test_summary = results['test_summary']
    print(f"\n{'='*60}")
    print("ENHANCED TESTING PIPELINE RESULTS")
    print(f"{'='*60}")
    print(f"Total Tests: {test_summary['total_tests']}")
    print(f"Passed: {test_summary['passed_tests']}")
    print(f"Failed: {test_summary['failed_tests']}")
    print(f"Skipped: {test_summary['skipped_tests']}")
    print(f"Errors: {test_summary['error_tests']}")
    print(f"Success Rate: {test_summary['success_rate']:.1f}%")
    print(f"Total Duration: {test_summary['total_duration']:.2f}s")
    print(f"Execution Time: {test_summary['execution_time']:.2f}s")
    print(f"{'='*60}")
    
    # Print individual test results
    if args.verbose:
        print("\nIndividual Test Results:")
        for result in test_summary['test_results']:
            status_icon = "✅" if result['status'] == 'passed' else "❌" if result['status'] == 'failed' else "⏭️" if result['status'] == 'skipped' else "⚠️"
            print(f"  {status_icon} {result['test_name']}: {result['status']} ({result['duration']:.2f}s)")
            if result['error']:
                print(f"    Error: {result['error']}")


if __name__ == "__main__":
    main()