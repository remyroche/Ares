"""
Comprehensive Test Suite for Data-Driven Lookback Optimization System

This module provides comprehensive tests for all components of the three-stage
Bayesian optimization system, ensuring reliability and correctness.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import tempfile
import os

# Import the optimization system
from .orchestrator import LookbackOptimizationOrchestrator
from .config import create_development_config, FamilyType
from .ic_surface import ICSurfaceEstimator, ICSurfaceResult
from .wf_stability import StabilityTester, StabilityResult
from .decision import LookbackDecisionMaker, DecisionResult
from .feature_families import MultiFamilyFeatureGenerator, FeatureResult

# Set up logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing


class TestDataGeneration(unittest.TestCase):
    """Test data generation utilities."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_days = 1000
        self.data = self._generate_test_data()
        self.target = self._generate_test_target()
    
    def _generate_test_data(self):
        """Generate test market data."""
        # Generate price data
        returns = np.random.normal(0.0001, 0.02, self.n_days)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        high_low_noise = np.random.uniform(0.001, 0.005, self.n_days)
        df = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.001, 0.001, self.n_days)),
            'high': prices * (1 + high_low_noise),
            'low': prices * (1 - high_low_noise),
            'close': prices,
            'volume': np.random.uniform(1000000, 5000000, self.n_days)
        })
        
        return df
    
    def _generate_test_target(self):
        """Generate test target variable."""
        future_returns = self.data['close'].pct_change(5).shift(-5)
        return future_returns.fillna(0).values
    
    def test_data_generation(self):
        """Test that test data is generated correctly."""
        self.assertEqual(len(self.data), self.n_days)
        self.assertEqual(len(self.target), self.n_days)
        self.assertIn('close', self.data.columns)
        self.assertIn('volume', self.data.columns)


class TestConfiguration(unittest.TestCase):
    """Test configuration system."""
    
    def test_default_config_creation(self):
        """Test default configuration creation."""
        from .config import create_default_config
        
        config = create_default_config()
        self.assertIsNotNone(config)
        self.assertIsNotNone(config.penalties)
        self.assertIsNotNone(config.search_grids)
        self.assertIsNotNone(config.hysteresis)
    
    def test_development_config_creation(self):
        """Test development configuration creation."""
        from .config import create_development_config
        
        config = create_development_config()
        self.assertIsNotNone(config)
        self.assertTrue(config.cv.n_folds <= 5)  # Development should have fewer folds
    
    def test_production_config_creation(self):
        """Test production configuration creation."""
        from .config import create_production_config
        
        config = create_production_config()
        self.assertIsNotNone(config)
        self.assertTrue(config.cv.n_folds >= 5)  # Production should have more folds
    
    def test_config_validation(self):
        """Test configuration validation."""
        from .config import create_default_config
        
        config = create_default_config()
        issues = config.validate()
        self.assertEqual(len(issues), 0)  # Default config should be valid


class TestICSurfaceEstimation(unittest.TestCase):
    """Test IC surface estimation (Stage 1)."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = TestDataGeneration()._generate_test_data()
        self.test_target = TestDataGeneration()._generate_test_target()
        self.config = create_development_config()
        self.ic_estimator = ICSurfaceEstimator(self.config)
    
    def test_ic_surface_estimation(self):
        """Test IC surface estimation for momentum family."""
        result = self.ic_estimator.estimate_surface(
            self.test_data, self.test_target, FamilyType.MOMENTUM, "momentum_feature"
        )
        
        self.assertIsInstance(result, ICSurfaceResult)
        self.assertEqual(result.family, FamilyType.MOMENTUM)
        self.assertGreater(len(result.lookbacks), 0)
        self.assertGreater(len(result.ic_values), 0)
        self.assertGreater(result.optimal_lookback, 0)
        self.assertIsNotNone(result.execution_time)
    
    def test_ic_surface_with_invalid_data(self):
        """Test IC surface estimation with invalid data."""
        # Test with empty data
        empty_data = pd.DataFrame()
        empty_target = np.array([])
        
        result = self.ic_estimator.estimate_surface(
            empty_data, empty_target, FamilyType.MOMENTUM, "momentum_feature"
        )
        
        self.assertIsInstance(result, ICSurfaceResult)
        self.assertEqual(len(result.lookbacks), 0)
    
    def test_cost_aware_scoring(self):
        """Test cost-aware scoring system."""
        from .ic_surface import CostAwareScorer
        
        scorer = CostAwareScorer(self.config)
        
        # Test CPU cost calculation
        cpu_cost = scorer.compute_cpu_cost(10, FamilyType.MOMENTUM)
        self.assertGreater(cpu_cost, 0)
        
        # Test staleness cost calculation
        stale_cost = scorer.compute_staleness_cost(10, FamilyType.MOMENTUM)
        self.assertGreater(stale_cost, 0)
        
        # Test uncertainty cost calculation
        unc_cost = scorer.compute_uncertainty_cost(0.1)
        self.assertGreater(unc_cost, 0)
        
        # Test adjusted score calculation
        adjusted_score = scorer.compute_adjusted_score(0.5, 10, FamilyType.MOMENTUM, 0.1)
        self.assertIsInstance(adjusted_score, float)


class TestStabilityTesting(unittest.TestCase):
    """Test walk-forward stability testing (Stage 2)."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = TestDataGeneration()._generate_test_data()
        self.test_target = TestDataGeneration()._generate_test_target()
        self.config = create_development_config()
        
        # Create a mock IC surface result
        self.ic_result = ICSurfaceResult(
            family=FamilyType.MOMENTUM,
            lookbacks=np.array([5, 10, 15, 20]),
            ic_values=np.array([0.1, 0.15, 0.12, 0.08]),
            ic_errors=np.array([0.05, 0.03, 0.04, 0.06]),
            optimal_lookback=10.0,
            optimal_ic=0.15,
            optimal_ic_error=0.03,
            execution_time=1.0
        )
        
        self.stability_tester = StabilityTester(self.config)
    
    def test_stability_testing(self):
        """Test stability testing."""
        result = self.stability_tester.test_stability(
            self.test_data, self.test_target, self.ic_result, "momentum_feature"
        )
        
        self.assertIsInstance(result, StabilityResult)
        self.assertEqual(result.family, FamilyType.MOMENTUM)
        self.assertGreaterEqual(result.match_rate, 0.0)
        self.assertLessEqual(result.match_rate, 1.0)
        self.assertIsNotNone(result.recommendation)
        self.assertIsNotNone(result.execution_time)
    
    def test_purged_cv_splitter(self):
        """Test purged cross-validation splitter."""
        from .wf_stability import PurgedTimeSeriesSplit
        
        splitter = PurgedTimeSeriesSplit(self.config.cv)
        X = np.arange(1000)
        
        splits = splitter.split(X)
        self.assertGreater(len(splits), 0)
        
        for train_idx, test_idx in splits:
            self.assertGreater(len(train_idx), 0)
            self.assertGreater(len(test_idx), 0)
            self.assertTrue(np.all(train_idx < test_idx))  # Train before test


class TestDecisionMaking(unittest.TestCase):
    """Test decision making logic."""
    
    def setUp(self):
        """Set up test data."""
        self.config = create_development_config()
        self.decision_maker = LookbackDecisionMaker(self.config)
        
        # Create mock results
        self.ic_result = ICSurfaceResult(
            family=FamilyType.MOMENTUM,
            lookbacks=np.array([5, 10, 15, 20]),
            ic_values=np.array([0.1, 0.15, 0.12, 0.08]),
            ic_errors=np.array([0.05, 0.03, 0.04, 0.06]),
            optimal_lookback=10.0,
            optimal_ic=0.15,
            optimal_ic_error=0.03,
            execution_time=1.0
        )
        
        self.stability_result = StabilityResult(
            family=FamilyType.MOMENTUM,
            global_optimal_lookback=10.0,
            global_optimal_ic=0.15,
            fold_results=[],
            match_rate=0.8,
            average_ic_penalty=0.02,
            average_lookback_difference=1.5,
            stability_score=0.7,
            recommendation="stable",
            execution_time=1.0
        )
    
    def test_decision_making(self):
        """Test decision making for stable case."""
        result = self.decision_maker.make_decision(
            "TEST_SYMBOL", FamilyType.MOMENTUM, self.ic_result, self.stability_result
        )
        
        self.assertIsInstance(result, DecisionResult)
        self.assertEqual(result.family, FamilyType.MOMENTUM)
        self.assertEqual(result.symbol, "TEST_SYMBOL")
        self.assertIsNotNone(result.lookback_spec)
        self.assertIsNotNone(result.execution_time)
    
    def test_hysteresis_manager(self):
        """Test hysteresis management."""
        from .decision import HysteresisManager
        
        hysteresis_manager = HysteresisManager(self.config.hysteresis)
        
        # First decision should be allowed
        should_change = hysteresis_manager.should_change_lookback(
            "TEST", FamilyType.MOMENTUM, 10.0, 0.1
        )
        self.assertTrue(should_change)
        
        # Update lookback
        hysteresis_manager.update_lookback("TEST", FamilyType.MOMENTUM, 10.0)
        
        # Small change should be rejected
        should_change = hysteresis_manager.should_change_lookback(
            "TEST", FamilyType.MOMENTUM, 11.0, 0.05
        )
        self.assertFalse(should_change)
        
        # Large change should be allowed
        should_change = hysteresis_manager.should_change_lookback(
            "TEST", FamilyType.MOMENTUM, 20.0, 0.3
        )
        self.assertTrue(should_change)


class TestFeatureGeneration(unittest.TestCase):
    """Test feature generation."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = TestDataGeneration()._generate_test_data()
        self.config = create_development_config()
        self.feature_generator = MultiFamilyFeatureGenerator(self.config)
        
        # Create mock decision
        from .decision import DecisionResult, LookbackSpec, DecisionType
        
        self.decision = DecisionResult(
            family=FamilyType.MOMENTUM,
            symbol="TEST_SYMBOL",
            lookback_spec=LookbackSpec(
                decision_type=DecisionType.DISCRETE,
                primary_lookback=10.0,
                effective_lookback=10.0,
                confidence_score=0.8,
                reasoning="Test decision"
            ),
            execution_time=1.0
        )
    
    def test_feature_generation(self):
        """Test feature generation."""
        decisions = {FamilyType.MOMENTUM: self.decision}
        feature_names = {FamilyType.MOMENTUM: "momentum_feature"}
        
        results = self.feature_generator.generate_features(
            self.test_data, decisions, feature_names
        )
        
        self.assertIn(FamilyType.MOMENTUM, results)
        result = results[FamilyType.MOMENTUM]
        
        self.assertIsInstance(result, FeatureResult)
        self.assertEqual(result.family, FamilyType.MOMENTUM)
        self.assertEqual(len(result.feature_values), len(self.test_data))
        self.assertIsNotNone(result.generation_time)
        self.assertIsNotNone(result.quality_score)
    
    def test_feature_family_builders(self):
        """Test individual feature family builders."""
        from .feature_families import (
            MomentumFeatureBuilder, VolatilityFeatureBuilder, RSIFeatureBuilder
        )
        
        # Test momentum builder
        momentum_builder = MomentumFeatureBuilder(self.config)
        result = momentum_builder.build_feature(
            self.test_data, self.decision.lookback_spec, "momentum_feature"
        )
        
        self.assertIsInstance(result, FeatureResult)
        self.assertEqual(result.family, FamilyType.MOMENTUM)
        self.assertEqual(len(result.feature_values), len(self.test_data))
        
        # Test volatility builder
        volatility_builder = VolatilityFeatureBuilder(self.config)
        result = volatility_builder.build_feature(
            self.test_data, self.decision.lookback_spec, "volatility_feature"
        )
        
        self.assertIsInstance(result, FeatureResult)
        self.assertEqual(result.family, FamilyType.VOLATILITY)


class TestOrchestrator(unittest.TestCase):
    """Test main orchestrator."""
    
    def setUp(self):
        """Set up test data."""
        self.config = create_development_config()
        self.orchestrator = LookbackOptimizationOrchestrator(self.config)
        
        # Generate test data
        test_data_gen = TestDataGeneration()
        self.data = {"TEST_SYMBOL": test_data_gen._generate_test_data()}
        self.targets = {"TEST_SYMBOL": test_data_gen._generate_test_target()}
        self.feature_names = {family: f"{family.value}_feature" for family in FamilyType}
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        self.assertIsNotNone(self.orchestrator.config)
        self.assertIsNotNone(self.orchestrator.ic_estimator)
        self.assertIsNotNone(self.orchestrator.stability_tester)
        self.assertIsNotNone(self.orchestrator.decision_maker)
        self.assertIsNotNone(self.orchestrator.feature_generator)
    
    def test_input_validation(self):
        """Test input validation."""
        # Test with empty data
        with self.assertRaises(ValueError):
            self.orchestrator._validate_inputs({}, {})
        
        # Test with mismatched data and targets
        with self.assertRaises(ValueError):
            self.orchestrator._validate_inputs({"A": self.data["TEST_SYMBOL"]}, {})
        
        # Test with insufficient data
        small_data = {"TEST": self.data["TEST_SYMBOL"].iloc[:50]}
        small_targets = {"TEST": self.targets["TEST_SYMBOL"][:50]}
        
        with self.assertRaises(ValueError):
            self.orchestrator._validate_inputs(small_data, small_targets)
    
    def test_full_optimization_pipeline(self):
        """Test the complete optimization pipeline."""
        # This is a comprehensive test that may take some time
        result = self.orchestrator.optimize_lookbacks(
            self.data, self.targets, self.feature_names
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.execution_time)
        self.assertIsNotNone(result.success)
        
        # Check that results are populated
        if result.success:
            self.assertIsNotNone(result.ic_surface_results)
            self.assertIsNotNone(result.stability_results)
            self.assertIsNotNone(result.decisions)
            self.assertIsNotNone(result.feature_results)
    
    def test_report_generation(self):
        """Test report generation."""
        # Create a mock result
        from .orchestrator import OptimizationResult
        
        mock_result = OptimizationResult(
            ic_surface_results={},
            stability_results={},
            hierarchical_results={},
            decisions={},
            feature_results={},
            execution_time=1.0,
            success=True
        )
        
        report = self.orchestrator.generate_comprehensive_report(mock_result)
        
        self.assertIsInstance(report, dict)
        self.assertIn('execution_summary', report)
        self.assertIn('stage_1_summary', report)
        self.assertIn('stage_2_summary', report)
        self.assertIn('stage_3_summary', report)
        self.assertIn('decision_summary', report)
        self.assertIn('feature_summary', report)
        self.assertIn('recommendations', report)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up test data."""
        self.config = create_development_config()
        self.orchestrator = LookbackOptimizationOrchestrator(self.config)
        
        # Generate test data for multiple symbols
        test_data_gen = TestDataGeneration()
        self.data = {
            "SYMBOL_1": test_data_gen._generate_test_data(),
            "SYMBOL_2": test_data_gen._generate_test_data()
        }
        self.targets = {
            "SYMBOL_1": test_data_gen._generate_test_target(),
            "SYMBOL_2": test_data_gen._generate_test_target()
        }
        self.feature_names = {family: f"{family.value}_feature" for family in FamilyType}
    
    def test_multi_symbol_optimization(self):
        """Test optimization with multiple symbols."""
        result = self.orchestrator.optimize_lookbacks(
            self.data, self.targets, self.feature_names
        )
        
        self.assertIsNotNone(result)
        
        if result.success:
            # Check that all symbols are processed
            self.assertEqual(len(result.ic_surface_results), 2)
            self.assertIn("SYMBOL_1", result.ic_surface_results)
            self.assertIn("SYMBOL_2", result.ic_surface_results)
    
    def test_configuration_persistence(self):
        """Test configuration saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "test_config.yaml")
            
            # Save configuration
            self.config.to_yaml(config_path)
            self.assertTrue(os.path.exists(config_path))
            
            # Load configuration
            from .config import LookbackOptimizationConfig
            loaded_config = LookbackOptimizationConfig.from_yaml(config_path)
            
            self.assertEqual(loaded_config.penalties.lambda_cost, self.config.penalties.lambda_cost)
            self.assertEqual(loaded_config.search_grids.momentum_bars, self.config.search_grids.momentum_bars)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDataGeneration,
        TestConfiguration,
        TestICSurfaceEstimation,
        TestStabilityTesting,
        TestDecisionMaking,
        TestFeatureGeneration,
        TestOrchestrator,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running Data-Driven Lookback Optimization System Tests")
    print("=" * 60)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    print("=" * 60)