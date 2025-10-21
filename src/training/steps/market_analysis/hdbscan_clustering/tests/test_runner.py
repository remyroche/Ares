"""
Comprehensive Test Runner for Data-Driven Clustering System

This module provides a unified test runner that combines:
- Automated regression tests
- Performance profiling
- Synthetic dataset testing
- Baseline comparison
- Performance monitoring
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import test modules
from test_economic_validation_regression import (
    EconomicValidationRegressionTester, run_regression_tests
)
from test_synthetic_datasets import (
    SyntheticDatasetTester, run_synthetic_tests
)

# Import performance profiler
sys.path.append(str(Path(__file__).parent.parent / "performance"))
from performance_profiler import (
    PerformanceProfiler, run_performance_profiling
)

logger = logging.getLogger(__name__)


class ComprehensiveTestRunner:
    """
    Comprehensive test runner that orchestrates all testing activities.
    
    Combines regression testing, performance profiling, and synthetic dataset testing
    to ensure the data-driven clustering system maintains quality and performance.
    """
    
    def __init__(self, 
                 test_dir: str = "comprehensive_tests",
                 tolerance: float = 0.05,
                 enable_performance_profiling: bool = True,
                 enable_regression_tests: bool = True,
                 enable_synthetic_tests: bool = True):
        """
        Initialize the comprehensive test runner.
        
        Args:
            test_dir: Directory to save test results
            tolerance: Maximum allowed difference in scores
            enable_performance_profiling: Whether to run performance profiling
            enable_regression_tests: Whether to run regression tests
            enable_synthetic_tests: Whether to run synthetic dataset tests
        """
        self.test_dir = Path(test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        self.tolerance = tolerance
        self.enable_performance_profiling = enable_performance_profiling
        self.enable_regression_tests = enable_regression_tests
        self.enable_synthetic_tests = enable_synthetic_tests
        
        # Test results storage
        self.test_results: Dict[str, Any] = {}
        self.performance_results: Dict[str, Any] = {}
        self.regression_results: Dict[str, Any] = {}
        self.synthetic_results: Dict[str, Any] = {}
        
        # Initialize testers
        self.regression_tester = EconomicValidationRegressionTester(tolerance=tolerance)
        self.synthetic_tester = SyntheticDatasetTester(tolerance=tolerance)
        self.performance_profiler = PerformanceProfiler()
        
    def run_performance_profiling(self, 
                                market_data: Optional[Any] = None,
                                n_iterations: int = 3) -> Dict[str, Any]:
        """
        Run performance profiling tests.
        
        Args:
            market_data: Market data to profile (generates synthetic if None)
            n_iterations: Number of iterations for averaging
            
        Returns:
            Dictionary containing performance results
        """
        if not self.enable_performance_profiling:
            logger.info("Performance profiling disabled")
            return {}
        
        logger.info("Starting performance profiling...")
        
        try:
            # Generate synthetic data if not provided
            if market_data is None:
                import pandas as pd
                import numpy as np
                
                np.random.seed(42)
                n_samples = 1000
                dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='1H')
                
                market_data = pd.DataFrame({
                    'timestamp': dates,
                    'open': 100 + np.cumsum(np.random.normal(0, 0.01, n_samples)),
                    'high': 100 + np.cumsum(np.random.normal(0, 0.01, n_samples)) + np.abs(np.random.normal(0, 0.01, n_samples)),
                    'low': 100 + np.cumsum(np.random.normal(0, 0.01, n_samples)) - np.abs(np.random.normal(0, 0.01, n_samples)),
                    'close': 100 + np.cumsum(np.random.normal(0, 0.01, n_samples)),
                    'volume': np.random.lognormal(5, 0.5, n_samples)
                })
                
                market_data['returns'] = market_data['close'].pct_change()
                market_data['volatility'] = market_data['returns'].rolling(20).std()
            
            # Run performance profiling
            performance_results = self.performance_profiler.run_comprehensive_profile(
                market_data=market_data,
                n_iterations=n_iterations
            )
            
            self.performance_results = performance_results
            logger.info("Performance profiling completed successfully")
            
            return performance_results
            
        except Exception as e:
            logger.error(f"Performance profiling failed: {e}")
            return {'error': str(e), 'success': False}
    
    def run_regression_tests(self, 
                           test_cases: Optional[List[Dict[str, Any]]] = None,
                           save_baseline: bool = False) -> Dict[str, Any]:
        """
        Run regression tests.
        
        Args:
            test_cases: Custom test cases (uses default if None)
            save_baseline: Whether to save results as new baseline
            
        Returns:
            Dictionary containing regression test results
        """
        if not self.enable_regression_tests:
            logger.info("Regression tests disabled")
            return {}
        
        logger.info("Starting regression tests...")
        
        try:
            # Run regression tests
            regression_results = self.regression_tester.run_regression_suite(
                test_cases=test_cases,
                save_baseline=save_baseline
            )
            
            self.regression_results = regression_results
            logger.info("Regression tests completed successfully")
            
            return regression_results
            
        except Exception as e:
            logger.error(f"Regression tests failed: {e}")
            return {'error': str(e), 'success': False}
    
    def run_synthetic_tests(self, 
                          test_scenarios: Optional[List[Dict[str, Any]]] = None,
                          save_baseline: bool = False) -> Dict[str, Any]:
        """
        Run synthetic dataset tests.
        
        Args:
            test_scenarios: Custom test scenarios (uses default if None)
            save_baseline: Whether to save results as new baseline
            
        Returns:
            Dictionary containing synthetic test results
        """
        if not self.enable_synthetic_tests:
            logger.info("Synthetic tests disabled")
            return {}
        
        logger.info("Starting synthetic dataset tests...")
        
        try:
            # Run synthetic tests
            synthetic_results = self.synthetic_tester.run_synthetic_test_suite(
                test_scenarios=test_scenarios,
                save_baseline=save_baseline
            )
            
            self.synthetic_results = synthetic_results
            logger.info("Synthetic dataset tests completed successfully")
            
            return synthetic_results
            
        except Exception as e:
            logger.error(f"Synthetic tests failed: {e}")
            return {'error': str(e), 'success': False}
    
    def run_comprehensive_tests(self, 
                              market_data: Optional[Any] = None,
                              test_cases: Optional[List[Dict[str, Any]]] = None,
                              test_scenarios: Optional[List[Dict[str, Any]]] = None,
                              save_baseline: bool = False,
                              n_iterations: int = 3) -> Dict[str, Any]:
        """
        Run all comprehensive tests.
        
        Args:
            market_data: Market data to profile (generates synthetic if None)
            test_cases: Custom regression test cases
            test_scenarios: Custom synthetic test scenarios
            save_baseline: Whether to save results as new baseline
            n_iterations: Number of iterations for performance profiling
            
        Returns:
            Dictionary containing all test results
        """
        logger.info("Starting comprehensive test suite...")
        
        start_time = datetime.now()
        
        # Run all tests
        all_results = {
            'timestamp': start_time.isoformat(),
            'tolerance': self.tolerance,
            'test_config': {
                'enable_performance_profiling': self.enable_performance_profiling,
                'enable_regression_tests': self.enable_regression_tests,
                'enable_synthetic_tests': self.enable_synthetic_tests,
                'n_iterations': n_iterations
            }
        }
        
        # Performance profiling
        if self.enable_performance_profiling:
            logger.info("Running performance profiling...")
            performance_results = self.run_performance_profiling(market_data, n_iterations)
            all_results['performance_profiling'] = performance_results
        
        # Regression tests
        if self.enable_regression_tests:
            logger.info("Running regression tests...")
            regression_results = self.run_regression_tests(test_cases, save_baseline)
            all_results['regression_tests'] = regression_results
        
        # Synthetic tests
        if self.enable_synthetic_tests:
            logger.info("Running synthetic dataset tests...")
            synthetic_results = self.run_synthetic_tests(test_scenarios, save_baseline)
            all_results['synthetic_tests'] = synthetic_results
        
        # Calculate overall summary
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        all_results['execution_time_seconds'] = execution_time
        all_results['summary'] = self._calculate_overall_summary(all_results)
        
        # Save results
        self._save_test_results(all_results)
        
        logger.info(f"Comprehensive test suite completed in {execution_time:.2f}s")
        
        return all_results
    
    def _calculate_overall_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall test summary."""
        summary = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'error_tests': 0,
            'overall_pass_rate': 0.0,
            'performance_metrics': {},
            'test_categories': {}
        }
        
        # Regression tests summary
        if 'regression_tests' in all_results and 'summary' in all_results['regression_tests']:
            reg_summary = all_results['regression_tests']['summary']
            summary['total_tests'] += reg_summary.get('total_tests', 0)
            summary['passed_tests'] += reg_summary.get('passed_tests', 0)
            summary['failed_tests'] += reg_summary.get('failed_tests', 0)
            summary['error_tests'] += reg_summary.get('error_tests', 0)
            summary['test_categories']['regression_tests'] = reg_summary
        
        # Synthetic tests summary
        if 'synthetic_tests' in all_results and 'summary' in all_results['synthetic_tests']:
            syn_summary = all_results['synthetic_tests']['summary']
            summary['total_tests'] += syn_summary.get('total_tests', 0)
            summary['passed_tests'] += syn_summary.get('passed_tests', 0)
            summary['failed_tests'] += syn_summary.get('failed_tests', 0)
            summary['error_tests'] += syn_summary.get('error_tests', 0)
            summary['test_categories']['synthetic_tests'] = syn_summary
        
        # Performance profiling summary
        if 'performance_profiling' in all_results:
            perf_summary = all_results['performance_profiling']
            summary['performance_metrics'] = {
                'feature_generation_time': perf_summary.get('feature_generation', {}).get('avg_execution_time', 0),
                'optimization_time': perf_summary.get('multi_objective_optimization', {}).get('execution_time', 0),
                'validation_time': perf_summary.get('economic_validation', {}).get('avg_execution_time', 0),
                'peak_memory_mb': perf_summary.get('memory_usage', {}).get('peak_memory_mb', 0),
                'optimal_workers': perf_summary.get('parallelization', {}).get('optimal_workers', 1),
                'caching_speedup': perf_summary.get('caching', {}).get('speedup', 1.0)
            }
            summary['test_categories']['performance_profiling'] = perf_summary
        
        # Calculate overall pass rate
        if summary['total_tests'] > 0:
            summary['overall_pass_rate'] = summary['passed_tests'] / summary['total_tests']
        
        return summary
    
    def _save_test_results(self, results: Dict[str, Any]) -> None:
        """Save test results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.test_dir / f"comprehensive_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Test results saved to {results_file}")
    
    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive test report."""
        report = f"""# Comprehensive Test Report

**Generated:** {results['timestamp']}
**Execution Time:** {results['execution_time_seconds']:.2f} seconds
**Tolerance:** {results['tolerance']:.1%}

## Test Configuration
- **Performance Profiling:** {'Enabled' if results['test_config']['enable_performance_profiling'] else 'Disabled'}
- **Regression Tests:** {'Enabled' if results['test_config']['enable_regression_tests'] else 'Disabled'}
- **Synthetic Tests:** {'Enabled' if results['test_config']['enable_synthetic_tests'] else 'Disabled'}
- **Iterations:** {results['test_config']['n_iterations']}

## Overall Summary
- **Total Tests:** {results['summary']['total_tests']}
- **Passed:** {results['summary']['passed_tests']}
- **Failed:** {results['summary']['failed_tests']}
- **Errors:** {results['summary']['error_tests']}
- **Pass Rate:** {results['summary']['overall_pass_rate']:.1%}

## Performance Metrics
"""
        
        if results['summary']['performance_metrics']:
            perf_metrics = results['summary']['performance_metrics']
            report += f"""
- **Feature Generation Time:** {perf_metrics.get('feature_generation_time', 0):.3f}s
- **Optimization Time:** {perf_metrics.get('optimization_time', 0):.3f}s
- **Validation Time:** {perf_metrics.get('validation_time', 0):.3f}s
- **Peak Memory:** {perf_metrics.get('peak_memory_mb', 0):.1f}MB
- **Optimal Workers:** {perf_metrics.get('optimal_workers', 1)}
- **Caching Speedup:** {perf_metrics.get('caching_speedup', 1.0):.2f}x
"""
        
        # Add detailed results for each test category
        for category, category_results in results['summary']['test_categories'].items():
            report += f"\n## {category.replace('_', ' ').title()}\n"
            
            if category == 'regression_tests':
                report += f"""
- **Total Tests:** {category_results.get('total_tests', 0)}
- **Passed:** {category_results.get('passed_tests', 0)}
- **Failed:** {category_results.get('failed_tests', 0)}
- **Pass Rate:** {category_results.get('pass_rate', 0):.1%}
"""
            elif category == 'synthetic_tests':
                report += f"""
- **Total Tests:** {category_results.get('total_tests', 0)}
- **Passed:** {category_results.get('passed_tests', 0)}
- **Failed:** {category_results.get('failed_tests', 0)}
- **Pass Rate:** {category_results.get('pass_rate', 0):.1%}
"""
            elif category == 'performance_profiling':
                report += f"""
- **Feature Generation:** {category_results.get('feature_generation', {}).get('avg_execution_time', 0):.3f}s
- **Multi-Objective Optimization:** {category_results.get('multi_objective_optimization', {}).get('execution_time', 0):.3f}s
- **Economic Validation:** {category_results.get('economic_validation', {}).get('avg_execution_time', 0):.3f}s
- **Peak Memory:** {category_results.get('memory_usage', {}).get('peak_memory_mb', 0):.1f}MB
"""
        
        report += "\n## Recommendations\n"
        
        # Add recommendations based on results
        if results['summary']['overall_pass_rate'] < 0.8:
            report += "- **Critical:** Overall pass rate is below 80%. Review failed tests.\n"
        
        if results['summary']['performance_metrics'].get('peak_memory_mb', 0) > 1000:
            report += "- **Memory:** Peak memory usage is high. Consider optimization.\n"
        
        if results['summary']['performance_metrics'].get('caching_speedup', 1.0) < 1.5:
            report += "- **Caching:** Caching effectiveness is low. Review cache strategy.\n"
        
        if results['summary']['performance_metrics'].get('optimal_workers', 1) > 1:
            report += f"- **Parallelization:** Use {results['summary']['performance_metrics'].get('optimal_workers', 1)} workers for optimal performance.\n"
        
        return report


def main():
    """Main function for running comprehensive tests."""
    parser = argparse.ArgumentParser(description='Run comprehensive tests for data-driven clustering system')
    parser.add_argument('--tolerance', type=float, default=0.05, help='Tolerance for score differences')
    parser.add_argument('--save-baseline', action='store_true', help='Save results as new baseline')
    parser.add_argument('--no-performance', action='store_true', help='Disable performance profiling')
    parser.add_argument('--no-regression', action='store_true', help='Disable regression tests')
    parser.add_argument('--no-synthetic', action='store_true', help='Disable synthetic tests')
    parser.add_argument('--iterations', type=int, default=3, help='Number of iterations for performance profiling')
    parser.add_argument('--output-dir', type=str, default='comprehensive_tests', help='Output directory for test results')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test runner
    test_runner = ComprehensiveTestRunner(
        test_dir=args.output_dir,
        tolerance=args.tolerance,
        enable_performance_profiling=not args.no_performance,
        enable_regression_tests=not args.no_regression,
        enable_synthetic_tests=not args.no_synthetic
    )
    
    # Run comprehensive tests
    print("Starting comprehensive test suite...")
    results = test_runner.run_comprehensive_tests(
        save_baseline=args.save_baseline,
        n_iterations=args.iterations
    )
    
    # Generate and print report
    report = test_runner.generate_test_report(results)
    print("\n" + "="*80)
    print(report)
    print("="*80)
    
    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = test_runner.test_dir / f"test_report_{timestamp}.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nTest report saved to {report_file}")
    
    # Return exit code based on results
    if results['summary']['overall_pass_rate'] < 0.8:
        print("\n❌ Tests failed - overall pass rate below 80%")
        sys.exit(1)
    else:
        print("\n✅ Tests passed - overall pass rate above 80%")
        sys.exit(0)


if __name__ == "__main__":
    main()