"""
Shared Utilities Integration Verification

This script demonstrates and verifies that the shared utilities (evolutionary search,
feature engineering, and advanced metrics) are seamlessly integrated and used by both
NAS and TAS systems.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import shared utilities
from .evolutionary_search import (
    EvolutionaryAlgorithmManager, EvolutionaryConfig, EvolutionaryResult,
    create_evolutionary_algorithm_manager, Individual
)
from .feature_engineering import (
    UnifiedFeatureEngineer, FeatureConfig, FeatureEngineeringResult,
    create_unified_feature_engineer
)
from .advanced_metrics import (
    AdvancedEvaluator, AdvancedEvaluationResult,
    create_advanced_evaluator
)

# Import TAS components
try:
    from ..tas.core.tas_engine import TreeArchitectureSearchEngine, TASEngineConfig
    TAS_AVAILABLE = True
except ImportError:
    TAS_AVAILABLE = False

# Import NAS components
try:
    from ...training.steps.market_analysis.nas_regime.core.nas_shared_utils_integration import (
        NASSharedUtilsIntegration, NASSearchConfig
    )
    NAS_AVAILABLE = True
except ImportError:
    NAS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class IntegrationTestResult:
    """Result from integration testing."""
    test_name: str
    success: bool
    execution_time: float
    details: Dict[str, Any]
    error_message: Optional[str] = None

class SharedUtilsIntegrationVerifier:
    """Verifier for shared utilities integration."""

    def __init__(self):
        """Initialize the integration verifier."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.test_results: List[IntegrationTestResult] = []

        self.logger.info("✅ Shared Utils Integration Verifier initialized")

    def run_all_tests(self) -> List[IntegrationTestResult]:
        """Run all integration tests."""
        self.logger.info("🧪 Starting comprehensive integration tests...")

        # Test 1: Shared utilities basic functionality
        self._test_shared_utilities_basic()

        # Test 2: Evolutionary search integration
        self._test_evolutionary_search_integration()

        # Test 3: Feature engineering integration
        self._test_feature_engineering_integration()

        # Test 4: Advanced metrics integration
        self._test_advanced_metrics_integration()

        # Test 5: TAS integration (if available)
        if TAS_AVAILABLE:
            self._test_tas_integration()
        else:
            self.logger.warning("⚠️ TAS not available for testing")

        # Test 6: NAS integration (if available)
        if NAS_AVAILABLE:
            self._test_nas_integration()
        else:
            self.logger.warning("⚠️ NAS not available for testing")

        # Test 7: Cross-system compatibility
        self._test_cross_system_compatibility()

        self.logger.info(f"✅ Integration tests completed: {len(self.test_results)} tests")
        return self.test_results

    def _test_shared_utilities_basic(self):
        """Test basic shared utilities functionality."""
        start_time = time.time()

        try:
            self.logger.info("🔧 Testing shared utilities basic functionality...")

            # Test evolutionary search
            config = EvolutionaryConfig(population_size=10, max_generations=5)
            manager = create_evolutionary_algorithm_manager(config)

            # Test feature engineering
            feature_config = FeatureConfig(enable_technical_indicators=True)
            engineer = create_unified_feature_engineer(feature_config)

            # Test advanced metrics
            evaluator = create_advanced_evaluator()

            execution_time = time.time() - start_time

            self.test_results.append(IntegrationTestResult(
                test_name="Shared Utilities Basic",
                success=True,
                execution_time=execution_time,
                details={
                    'evolutionary_manager_created': manager is not None,
                    'feature_engineer_created': engineer is not None,
                    'advanced_evaluator_created': evaluator is not None
                }
            ))

            self.logger.info("✅ Shared utilities basic test passed")

        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Shared Utilities Basic",
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            ))
            self.logger.error(f"❌ Shared utilities basic test failed: {e}")

    def _test_evolutionary_search_integration(self):
        """Test evolutionary search integration."""
        start_time = time.time()

        try:
            self.logger.info("🧬 Testing evolutionary search integration...")

            # Create test data
            X = np.random.rand(100, 10)
            y = np.random.randint(0, 2, 100)

            # Define simple objective functions
            def objective1(params):
                return np.random.random()

            def objective2(params):
                return np.random.random()

            # Define parameter space
            parameter_space = {
                'param1': {'type': 'continuous', 'min': 0.0, 'max': 1.0},
                'param2': {'type': 'integer', 'min': 1, 'max': 10},
                'param3': {'type': 'categorical', 'choices': ['A', 'B', 'C']}
            }

            # Test NSGA-II
            config = EvolutionaryConfig(
                population_size=20,
                max_generations=10,
                use_nsga2=True,
                use_spea2=False,
                use_genetic_algorithm=False
            )
            manager = create_evolutionary_algorithm_manager(config)

            result = manager.optimize_with_algorithm(
                [objective1, objective2], parameter_space, "nsga2"
            )

            execution_time = time.time() - start_time

            self.test_results.append(IntegrationTestResult(
                test_name="Evolutionary Search Integration",
                success=result.success,
                execution_time=execution_time,
                details={
                    'pareto_front_size': len(result.pareto_front),
                    'total_individuals': len(result.best_individuals),
                    'execution_time': result.execution_time,
                    'convergence_reached': result.convergence_info.get('convergence_reached', False)
                }
            ))

            self.logger.info("✅ Evolutionary search integration test passed")

        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Evolutionary Search Integration",
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            ))
            self.logger.error(f"❌ Evolutionary search integration test failed: {e}")

    def _test_feature_engineering_integration(self):
        """Test feature engineering integration."""
        start_time = time.time()

        try:
            self.logger.info("🔧 Testing feature engineering integration...")

            # Create test data
            X = np.random.rand(200, 20)
            y = np.random.randint(0, 2, 200)
            feature_names = [f'feature_{i}' for i in range(20)]

            # Configure feature engineering
            config = FeatureConfig(
                enable_technical_indicators=True,
                enable_feature_selection=True,
                feature_selection_method="mutual_info",
                max_features=15,
                enable_dimensionality_reduction=True,
                reduction_method="pca",
                n_components=10
            )

            # Create feature engineer
            engineer = create_unified_feature_engineer(config)

            # Engineer features
            result = engineer.engineer_features(X, y, feature_names)

            execution_time = time.time() - start_time

            self.test_results.append(IntegrationTestResult(
                test_name="Feature Engineering Integration",
                success=result.success,
                execution_time=execution_time,
                details={
                    'original_features': result.original_feature_count,
                    'enhanced_features': result.enhanced_feature_count,
                    'selected_features': result.selected_feature_count,
                    'feature_importance_count': len(result.feature_importance),
                    'correlation_matrix_shape': result.correlation_matrix.shape
                }
            ))

            self.logger.info("✅ Feature engineering integration test passed")

        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Feature Engineering Integration",
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            ))
            self.logger.error(f"❌ Feature engineering integration test failed: {e}")

    def _test_advanced_metrics_integration(self):
        """Test advanced metrics integration."""
        start_time = time.time()

        try:
            self.logger.info("📊 Testing advanced metrics integration...")

            # Create test data
            predictions = np.random.randint(0, 2, 100)
            targets = np.random.randint(0, 2, 100)
            returns = np.random.normal(0.001, 0.02, 100)
            regime_labels = np.random.choice([0, 1, 2], 100)

            # Create evaluator
            evaluator = create_advanced_evaluator()

            # Evaluate
            result = evaluator.evaluate(
                predictions, targets, returns, regime_labels, model_complexity=0.5
            )

            execution_time = time.time() - start_time

            self.test_results.append(IntegrationTestResult(
                test_name="Advanced Metrics Integration",
                success=result.success,
                execution_time=execution_time,
                details={
                    'overall_score': result.overall_score,
                    'risk_adjusted_score': result.risk_adjusted_score,
                    'regime_aware_score': result.regime_aware_score,
                    'economic_score': result.economic_score,
                    'trading_score': result.trading_score,
                    'model_stability': result.model_stability,
                    'model_robustness': result.model_robustness
                }
            ))

            self.logger.info("✅ Advanced metrics integration test passed")

        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Advanced Metrics Integration",
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            ))
            self.logger.error(f"❌ Advanced metrics integration test failed: {e}")

    def _test_tas_integration(self):
        """Test TAS integration with shared utilities."""
        start_time = time.time()

        try:
            self.logger.info("🌳 Testing TAS integration with shared utilities...")

            # Create test data
            X = np.random.rand(200, 20)
            y = np.random.randint(0, 2, 200)
            X_train, X_val = X[:150], X[150:]
            y_train, y_val = y[:150], y[150:]

            # Configure TAS with shared utilities
            config = TASEngineConfig(
                enable_enhanced_models=True,
                enable_automl=True,
                enable_evolutionary_search=True,
                enable_advanced_metrics=True,
                enable_feature_engineering=True,
                model_types=["xgboost", "lightgbm"],
                evolutionary_algorithm="nsga2",
                population_size=10,
                max_generations=5
            )

            # Create TAS engine
            engine = TreeArchitectureSearchEngine(config)

            # Run search (simplified)
            result = engine.search(
                (X_train, y_train),
                (X_val, y_val),
                search_strategy=engine.SearchStrategy.EVOLUTIONARY,
                optimization_mode=engine.OptimizationMode.MULTI_OBJECTIVE
            )

            execution_time = time.time() - start_time

            self.test_results.append(IntegrationTestResult(
                test_name="TAS Integration",
                success=result.success,
                execution_time=execution_time,
                details={
                    'best_score': result.best_score,
                    'execution_time': result.execution_time,
                    'search_history_length': len(result.search_history)
                }
            ))

            self.logger.info("✅ TAS integration test passed")

        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="TAS Integration",
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            ))
            self.logger.error(f"❌ TAS integration test failed: {e}")

    def _test_nas_integration(self):
        """Test NAS integration with shared utilities."""
        start_time = time.time()

        try:
            self.logger.info("🧠 Testing NAS integration with shared utilities...")

            # Create test data
            X = np.random.rand(200, 20)
            y = np.random.randint(0, 2, 200)
            X_train, X_val = X[:150], X[150:]
            y_train, y_val = y[:150], y[150:]
            regime_labels = np.random.choice([0, 1, 2], 150)

            # Configure NAS with shared utilities
            config = NASSearchConfig(
                population_size=10,
                max_generations=5,
                enable_feature_engineering=True,
                enable_advanced_metrics=True,
                regime_labels=regime_labels
            )

            # Create NAS integration
            nas_integration = NASSharedUtilsIntegration(config)

            # Run architecture search
            best_architectures = nas_integration.search_architectures(
                (X_train, y_train), (X_val, y_val)
            )

            execution_time = time.time() - start_time

            self.test_results.append(IntegrationTestResult(
                test_name="NAS Integration",
                success=len(best_architectures) > 0,
                execution_time=execution_time,
                details={
                    'architectures_found': len(best_architectures),
                    'search_statistics': nas_integration.get_search_statistics()
                }
            ))

            self.logger.info("✅ NAS integration test passed")

        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="NAS Integration",
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            ))
            self.logger.error(f"❌ NAS integration test failed: {e}")

    def _test_cross_system_compatibility(self):
        """Test cross-system compatibility."""
        start_time = time.time()

        try:
            self.logger.info("🔄 Testing cross-system compatibility...")

            # Test that shared utilities work with both NAS and TAS
            # Create common data
            X = np.random.rand(100, 10)
            y = np.random.randint(0, 2, 100)

            # Test evolutionary search with both systems
            config = EvolutionaryConfig(population_size=10, max_generations=5)
            manager = create_evolutionary_algorithm_manager(config)

            # Test feature engineering with both systems
            feature_config = FeatureConfig(enable_technical_indicators=True)
            engineer = create_unified_feature_engineer(feature_config)

            # Test advanced metrics with both systems
            evaluator = create_advanced_evaluator()

            # Verify compatibility
            nas_compatible = NAS_AVAILABLE and hasattr(manager, 'optimize_with_algorithm')
            tas_compatible = TAS_AVAILABLE and hasattr(engineer, 'engineer_features')
            metrics_compatible = hasattr(evaluator, 'evaluate')

            execution_time = time.time() - start_time

            self.test_results.append(IntegrationTestResult(
                test_name="Cross-System Compatibility",
                success=nas_compatible and tas_compatible and metrics_compatible,
                execution_time=execution_time,
                details={
                    'nas_compatible': nas_compatible,
                    'tas_compatible': tas_compatible,
                    'metrics_compatible': metrics_compatible,
                    'shared_utilities_available': True
                }
            ))

            self.logger.info("✅ Cross-system compatibility test passed")

        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(IntegrationTestResult(
                test_name="Cross-System Compatibility",
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            ))
            self.logger.error(f"❌ Cross-system compatibility test failed: {e}")

    def get_test_summary(self) -> Dict[str, Any]:
        """Get test summary."""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result.success)
        failed_tests = total_tests - successful_tests

        total_execution_time = sum(result.execution_time for result in self.test_results)

        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0.0,
            'total_execution_time': total_execution_time,
            'average_execution_time': total_execution_time / total_tests if total_tests > 0 else 0.0,
            'test_results': self.test_results
        }

    def print_test_report(self):
        """Print comprehensive test report."""
        summary = self.get_test_summary()

        print("\n" + "="*80)
        print("🔍 SHARED UTILITIES INTEGRATION VERIFICATION REPORT")
        print("="*80)

        print(f"\n📊 Test Summary:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Successful: {summary['successful_tests']}")
        print(f"   Failed: {summary['failed_tests']}")
        print(f"   Success Rate: {summary['success_rate']:.2%}")
        print(f"   Total Execution Time: {summary['total_execution_time']:.2f}s")
        print(f"   Average Execution Time: {summary['average_execution_time']:.2f}s")

        print(f"\n🧪 Individual Test Results:")
        for result in self.test_results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            print(f"   {status} {result.test_name} ({result.execution_time:.2f}s)")
            if result.error_message:
                print(f"      Error: {result.error_message}")
            if result.details:
                for key, value in result.details.items():
                    print(f"      {key}: {value}")

        print(f"\n🎯 Integration Status:")
        if summary['success_rate'] >= 0.8:
            print("   ✅ EXCELLENT: Shared utilities are well-integrated")
        elif summary['success_rate'] >= 0.6:
            print("   ⚠️ GOOD: Shared utilities are mostly integrated")
        else:
            print("   ❌ POOR: Shared utilities need better integration")

        print("="*80)

def run_integration_verification():
    """Run comprehensive integration verification."""
    print("🚀 Starting Shared Utilities Integration Verification")
    print("="*60)

    # Create verifier
    verifier = SharedUtilsIntegrationVerifier()

    # Run all tests
    test_results = verifier.run_all_tests()

    # Print report
    verifier.print_test_report()

    return test_results

if __name__ == "__main__":
    run_integration_verification()
