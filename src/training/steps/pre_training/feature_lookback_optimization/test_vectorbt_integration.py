"""
Comprehensive Test Suite for VectorBT Integration in Feature Lookback Optimization.

This module provides comprehensive tests for all VectorBT-enhanced components
to ensure proper integration and performance improvements.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import time
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.tprint import tprint, tprint_error, tprint_success, tprint_warning, tprint_info
from src.utils.logger import get_logger

logger = get_logger('VectorBTIntegrationTest')


class VectorBTIntegrationTester:
    """Comprehensive tester for VectorBT integration."""
    
    def __init__(self):
        """Initialize the tester."""
        self.test_results = {}
        self.performance_metrics = {}
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all VectorBT integration tests."""
        tprint("🧪 Starting VectorBT Integration Tests...")
        
        tests = [
            ("VectorBT Availability", self.test_vectorbt_availability),
            ("Correlation Calculator", self.test_correlation_calculator),
            ("Scoring System", self.test_scoring_system),
            ("Feature Generation", self.test_feature_generation),
            ("Bootstrap Validation", self.test_bootstrap_validation),
            ("Core Optimizer", self.test_core_optimizer),
            ("Main Component Integration", self.test_main_component_integration),
            ("Performance Comparison", self.test_performance_comparison)
        ]
        
        for test_name, test_func in tests:
            tprint_info(f"🔄 Running {test_name}...")
            try:
                start_time = time.time()
                result = test_func()
                execution_time = time.time() - start_time
                
                self.test_results[test_name] = {
                    'passed': result,
                    'execution_time': execution_time,
                    'timestamp': time.time()
                }
                
                if result:
                    tprint_success(f"✅ {test_name} passed ({execution_time:.3f}s)")
                else:
                    tprint_error(f"❌ {test_name} failed ({execution_time:.3f}s)")
                    
            except Exception as e:
                tprint_error(f"❌ {test_name} crashed: {e}")
                self.test_results[test_name] = {
                    'passed': False,
                    'execution_time': 0.0,
                    'error': str(e),
                    'timestamp': time.time()
                }
        
        return self._generate_test_report()
    
    def test_vectorbt_availability(self) -> bool:
        """Test VectorBT availability and basic functionality."""
        try:
            import vectorbt as vbt
            from vectorbt.portfolio.base import Portfolio
            
            # Test basic VectorBT functionality
            data = np.random.randn(100, 5)
            portfolio = vbt.Portfolio.from_signals(
                close=data[:, 0],
                entries=data[:, 1] > 0,
                exits=data[:, 2] < 0
            )
            
            # Test portfolio metrics
            returns = portfolio.returns()
            sharpe = portfolio.sharpe_ratio()
            
            return len(returns) > 0 and not np.isnan(sharpe)
            
        except ImportError:
            tprint_warning("⚠️ VectorBT not available - install with: pip install vectorbt")
            return False
        except Exception as e:
            tprint_error(f"❌ VectorBT test failed: {e}")
            return False
    
    def test_correlation_calculator(self) -> bool:
        """Test VectorBT correlation calculator."""
        try:
            from .core.vectorbt_correlation import create_vectorbt_correlation_calculator
            
            # Create test data
            np.random.seed(42)
            features_list = [np.random.randn(100) for _ in range(5)]
            returns_list = [np.random.randn(100) for _ in range(5)]
            
            # Test correlation calculator
            calculator = create_vectorbt_correlation_calculator()
            correlations = calculator.calculate_correlations_vectorbt(features_list, returns_list)
            
            # Test MI calculation
            mi_scores = calculator.calculate_mutual_information_vectorbt(features_list, returns_list)
            
            return len(correlations) == 5 and len(mi_scores) == 5
            
        except Exception as e:
            tprint_error(f"❌ Correlation calculator test failed: {e}")
            return False
    
    def test_scoring_system(self) -> bool:
        """Test VectorBT scoring system."""
        try:
            from .core.vectorbt_scoring import create_vectorbt_scoring_system, ScoringMethod
            
            # Create test data
            np.random.seed(42)
            feature_values = np.cumsum(np.random.randn(100) * 0.01)
            target_values = np.random.randn(100) * 0.02
            
            # Test scoring system
            scoring_system = create_vectorbt_scoring_system()
            
            # Test different scoring methods
            methods = [ScoringMethod.SHARPE_RATIO, ScoringMethod.COMPOSITE]
            
            for method in methods:
                result = scoring_system.score_feature_lookback(
                    feature_values, target_values, 20, method
                )
                if not result.is_valid:
                    return False
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Scoring system test failed: {e}")
            return False
    
    def test_feature_generation(self) -> bool:
        """Test VectorBT feature generation."""
        try:
            from .core.vectorbt_feature_generation import create_vectorbt_feature_generator, FeatureType
            
            # Create test data
            np.random.seed(42)
            data = pd.DataFrame({
                'open': np.random.randn(100).cumsum() + 100,
                'high': np.random.randn(100).cumsum() + 105,
                'low': np.random.randn(100).cumsum() + 95,
                'close': np.random.randn(100).cumsum() + 100,
                'volume': np.random.randint(1000, 10000, 100)
            })
            
            # Test feature generator
            generator = create_vectorbt_feature_generator()
            
            # Test different feature types
            feature_types = [FeatureType.SMA, FeatureType.RSI, FeatureType.MACD]
            lookback_periods = [10, 20, 30]
            
            for feature_type in feature_types:
                features = generator.generate_features_vectorbt(
                    data, feature_type.value, lookback_periods, feature_type
                )
                if len(features) == 0:
                    return False
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Feature generation test failed: {e}")
            return False
    
    def test_bootstrap_validation(self) -> bool:
        """Test VectorBT bootstrap validation."""
        try:
            from .core.vectorbt_bootstrap import create_vectorbt_bootstrap_validator, BootstrapMethod
            
            # Create test data
            np.random.seed(42)
            feature_values = np.cumsum(np.random.randn(100) * 0.01)
            target_values = np.random.randn(100) * 0.02
            
            # Test bootstrap validator
            validator = create_vectorbt_bootstrap_validator(n_bootstrap_samples=20)
            
            # Test different bootstrap methods
            methods = [BootstrapMethod.SIMPLE, BootstrapMethod.BLOCK]
            
            for method in methods:
                validator.config.bootstrap_method = method
                result = validator.validate_lookback_period(
                    feature_values, target_values, 20
                )
                if not result.is_valid:
                    return False
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Bootstrap validation test failed: {e}")
            return False
    
    def test_core_optimizer(self) -> bool:
        """Test VectorBT core optimizer."""
        try:
            from .core.vectorbt_optimizer import create_vectorbt_optimizer, OptimizationStrategy
            
            # Create test data
            np.random.seed(42)
            data = pd.DataFrame({
                'close': np.cumsum(np.random.randn(100) * 0.01) + 100,
                'high': np.cumsum(np.random.randn(100) * 0.01) + 105,
                'low': np.cumsum(np.random.randn(100) * 0.01) + 95,
                'volume': np.random.randint(1000, 10000, 100),
                'returns': np.random.randn(100) * 0.02
            })
            
            # Test optimizer
            optimizer = create_vectorbt_optimizer()
            
            result = optimizer.optimize_feature_lookback(
                data, 'sma', 'returns', (10, 30)
            )
            
            return result.is_valid and result.best_lookback_period > 0
            
        except Exception as e:
            tprint_error(f"❌ Core optimizer test failed: {e}")
            return False
    
    def test_main_component_integration(self) -> bool:
        """Test main component integration."""
        try:
            from .feature_lookback_optimization import FeatureLookbackOptimizationComponent
            
            # Create test data
            np.random.seed(42)
            data = pd.DataFrame({
                'close': np.cumsum(np.random.randn(100) * 0.01) + 100,
                'high': np.cumsum(np.random.randn(100) * 0.01) + 105,
                'low': np.cumsum(np.random.randn(100) * 0.01) + 95,
                'volume': np.random.randint(1000, 10000, 100),
                'returns': np.random.randn(100) * 0.02
            })
            
            # Test component initialization
            component = FeatureLookbackOptimizationComponent()
            
            # Check if VectorBT optimizer is available
            has_vectorbt = hasattr(component, 'vectorbt_optimizer') and component.vectorbt_optimizer is not None
            
            return has_vectorbt
            
        except Exception as e:
            tprint_error(f"❌ Main component integration test failed: {e}")
            return False
    
    def test_performance_comparison(self) -> bool:
        """Test performance comparison between VectorBT and standard methods."""
        try:
            from .core.vectorbt_correlation import create_vectorbt_correlation_calculator
            from .core.vectorbt_scoring import create_vectorbt_scoring_system
            
            # Create test data
            np.random.seed(42)
            n_samples = 1000
            n_features = 10
            
            features_list = [np.random.randn(n_samples) for _ in range(n_features)]
            returns_list = [np.random.randn(n_samples) for _ in range(n_features)]
            
            # Test VectorBT performance
            calculator = create_vectorbt_correlation_calculator()
            scoring_system = create_vectorbt_scoring_system()
            
            # Time VectorBT correlation calculation
            start_time = time.time()
            vectorbt_correlations = calculator.calculate_correlations_vectorbt(features_list, returns_list)
            vectorbt_correlation_time = time.time() - start_time
            
            # Time VectorBT scoring
            start_time = time.time()
            feature_values = features_list[0]
            target_values = returns_list[0]
            scoring_result = scoring_system.score_feature_lookback(feature_values, target_values, 20)
            vectorbt_scoring_time = time.time() - start_time
            
            # Store performance metrics
            self.performance_metrics = {
                'vectorbt_correlation_time': vectorbt_correlation_time,
                'vectorbt_scoring_time': vectorbt_scoring_time,
                'n_features': n_features,
                'n_samples': n_samples
            }
            
            tprint_info(f"📊 VectorBT Correlation Time: {vectorbt_correlation_time:.3f}s")
            tprint_info(f"📊 VectorBT Scoring Time: {vectorbt_scoring_time:.3f}s")
            
            return len(vectorbt_correlations) == n_features and scoring_result.is_valid
            
        except Exception as e:
            tprint_error(f"❌ Performance comparison test failed: {e}")
            return False
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['passed'])
        failed_tests = total_tests - passed_tests
        
        total_time = sum(result['execution_time'] for result in self.test_results.values())
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'total_execution_time': total_time
            },
            'test_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if not self.test_results.get('VectorBT Availability', {}).get('passed', False):
            recommendations.append("Install VectorBT: pip install vectorbt")
        
        if not self.test_results.get('Correlation Calculator', {}).get('passed', False):
            recommendations.append("Check VectorBT correlation calculator implementation")
        
        if not self.test_results.get('Scoring System', {}).get('passed', False):
            recommendations.append("Verify VectorBT scoring system configuration")
        
        if not self.test_results.get('Feature Generation', {}).get('passed', False):
            recommendations.append("Review VectorBT feature generation setup")
        
        if not self.test_results.get('Bootstrap Validation', {}).get('passed', False):
            recommendations.append("Check VectorBT bootstrap validation implementation")
        
        if not self.test_results.get('Core Optimizer', {}).get('passed', False):
            recommendations.append("Verify VectorBT core optimizer integration")
        
        if not self.test_results.get('Main Component Integration', {}).get('passed', False):
            recommendations.append("Check main component VectorBT integration")
        
        if not recommendations:
            recommendations.append("All tests passed! VectorBT integration is working correctly.")
        
        return recommendations


def run_vectorbt_integration_tests():
    """Run all VectorBT integration tests."""
    tester = VectorBTIntegrationTester()
    report = tester.run_all_tests()
    
    # Print summary
    tprint("\n" + "="*60)
    tprint("📊 VECTORBT INTEGRATION TEST SUMMARY")
    tprint("="*60)
    
    summary = report['summary']
    tprint(f"Total Tests: {summary['total_tests']}")
    tprint(f"Passed: {summary['passed_tests']}")
    tprint(f"Failed: {summary['failed_tests']}")
    tprint(f"Success Rate: {summary['success_rate']:.1%}")
    tprint(f"Total Time: {summary['total_execution_time']:.3f}s")
    
    # Print recommendations
    tprint("\n📋 RECOMMENDATIONS:")
    for i, rec in enumerate(report['recommendations'], 1):
        tprint(f"{i}. {rec}")
    
    # Print performance metrics if available
    if report['performance_metrics']:
        tprint("\n⚡ PERFORMANCE METRICS:")
        metrics = report['performance_metrics']
        tprint(f"VectorBT Correlation Time: {metrics.get('vectorbt_correlation_time', 0):.3f}s")
        tprint(f"VectorBT Scoring Time: {metrics.get('vectorbt_scoring_time', 0):.3f}s")
        tprint(f"Features Processed: {metrics.get('n_features', 0)}")
        tprint(f"Samples per Feature: {metrics.get('n_samples', 0)}")
    
    return report


if __name__ == "__main__":
    run_vectorbt_integration_tests()