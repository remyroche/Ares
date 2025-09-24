#!/usr/bin/env python3
"""
Unified Utilities Integration Example

This example demonstrates how to use the unified utilities in both TAS and NAS
regime detection systems, showing the common patterns and shared functionality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

# Import unified utilities
from ..shared_utils import (
    # Core evaluators
    UnifiedEconomicSignificanceEvaluator, EconomicEvaluationConfig,
    UnifiedTradingViabilityEvaluator, TradingViabilityConfig,
    UnifiedMultiObjectiveOptimizer, OptimizationConfig,
    UnifiedHardwareOptimizer, HardwareConfig,
    UnifiedRegimeAnalyzer, RegimeAnalysisConfig,
    UnifiedValidationSystem, ValidationConfig,
    UnifiedConfigManager, UnifiedRegimeConfig,
    
    # Convenience functions
    create_unified_economic_evaluator, quick_economic_evaluation,
    create_unified_trading_viability_evaluator, quick_trading_viability_evaluation,
    create_unified_multi_objective_optimizer, quick_multi_objective_optimization,
    create_unified_hardware_optimizer, quick_hardware_optimization,
    create_unified_regime_analyzer, quick_regime_analysis,
    create_unified_config_manager, load_config_from_file, create_environment_config,
    create_unified_validation_system, quick_validation
)

logger = logging.getLogger(__name__)


class UnifiedUtilitiesIntegrationExample:
    """
    Example showing how to integrate unified utilities with both TAS and NAS systems.
    """
    
    def __init__(self):
        """Initialize the integration example."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()
        
        # Initialize unified utilities
        self._initialize_unified_utilities()
        
        self.logger.info("✅ Unified Utilities Integration Example initialized")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _initialize_unified_utilities(self):
        """Initialize all unified utilities."""
        try:
            # Initialize unified configuration manager
            self.config_manager = create_unified_config_manager()
            
            # Initialize unified economic significance evaluator
            economic_config = EconomicEvaluationConfig(
                significance_threshold=0.6,
                price_impact_threshold=0.5,
                volume_threshold=0.4,
                volatility_threshold=0.5,
                trend_threshold=0.6,
                efficiency_threshold=0.5,
                enable_economic_indicators=True,
                enable_bootstrap_analysis=True,
                enable_position_aware_analysis=True
            )
            self.economic_evaluator = create_unified_economic_evaluator(economic_config)
            
            # Initialize unified trading viability evaluator
            trading_config = TradingViabilityConfig(
                viability_threshold=0.6,
                min_trading_frequency=0.1,
                max_trading_frequency=10.0,
                min_position_duration=5.0,
                max_position_duration=1440.0,
                min_model_confidence=0.6,
                min_risk_adjusted_return=0.1,
                enable_position_aware_analysis=True,
                enable_liquidity_analysis=True,
                enable_execution_analysis=True
            )
            self.trading_evaluator = create_unified_trading_viability_evaluator(trading_config)
            
            # Initialize unified multi-objective optimizer
            optimization_config = OptimizationConfig(
                objectives=['regime_accuracy', 'economic_significance', 'trading_viability', 'computational_efficiency'],
                objective_weights={
                    'regime_accuracy': 0.3,
                    'economic_significance': 0.25,
                    'trading_viability': 0.25,
                    'computational_efficiency': 0.2
                },
                max_iterations=100,
                population_size=50,
                algorithm='nsga2'
            )
            self.optimizer = create_unified_multi_objective_optimizer(optimization_config)
            
            # Initialize unified hardware optimizer
            hardware_config = HardwareConfig(
                enable_hardware_optimization=True,
                max_memory_usage_gb=8.0,
                enable_gpu_acceleration=True,
                enable_performance_monitoring=True,
                enable_adaptive_optimization=True
            )
            self.hardware_optimizer = create_unified_hardware_optimizer(hardware_config)
            
            # Initialize unified regime analyzer
            regime_config = RegimeAnalysisConfig(
                analysis_types=['stability', 'transitions', 'uncertainty', 'meta_learning'],
                stability_window=20,
                transition_window=10,
                uncertainty_method='entropy',
                enable_meta_learning=True,
                adaptation_rate=0.1,
                learning_threshold=0.05
            )
            self.regime_analyzer = create_unified_regime_analyzer(regime_config)
            
            # Initialize unified validation system
            validation_config = ValidationConfig(
                validation_type='time_series_validation',
                n_folds=5,
                test_size=0.2,
                metrics=['accuracy', 'precision', 'recall', 'f1_score', 'sharpe_ratio', 'max_drawdown'],
                enable_trading_metrics=True,
                enable_regime_metrics=True,
                enable_bootstrap=True,
                bootstrap_iterations=100
            )
            self.validator = create_unified_validation_system(validation_config)
            
            self.logger.info("✅ All unified utilities initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize unified utilities: {e}")
            raise
    
    def demonstrate_tas_integration(self):
        """Demonstrate unified utilities integration with TAS system."""
        self.logger.info("🌳 Demonstrating TAS Integration with Unified Utilities")
        
        try:
            # Generate sample TAS data
            market_data = self._generate_sample_market_data()
            regime_predictions = self._generate_sample_regime_predictions()
            regime_probabilities = self._generate_sample_regime_probabilities()
            
            # 1. Economic Significance Evaluation for TAS
            self.logger.info("1. Economic Significance Evaluation for TAS...")
            economic_result = self.economic_evaluator.evaluate(
                market_data=market_data,
                regime_predictions=regime_predictions,
                regime_probabilities=regime_probabilities
            )
            self.logger.info(f"   Economic significance score: {economic_result.overall_score:.3f}")
            self.logger.info(f"   Significance level: {economic_result.significance_level}")
            
            # 2. Trading Viability Assessment for TAS
            self.logger.info("2. Trading Viability Assessment for TAS...")
            trading_result = self.trading_evaluator.evaluate(
                market_data=market_data,
                regime_predictions=regime_predictions,
                regime_probabilities=regime_probabilities
            )
            self.logger.info(f"   Trading viability score: {trading_result.overall_score:.3f}")
            self.logger.info(f"   Viability level: {trading_result.viability_level}")
            
            # 3. Multi-Objective Optimization for TAS
            self.logger.info("3. Multi-Objective Optimization for TAS...")
            optimization_result = self.optimizer.optimize(
                market_data=market_data,
                regime_predictions=regime_predictions
            )
            self.logger.info(f"   Optimization success: {optimization_result.success}")
            self.logger.info(f"   Best score: {optimization_result.weighted_score:.3f}")
            self.logger.info(f"   Pareto solutions: {optimization_result.n_solutions}")
            
            # 4. Hardware Optimization for TAS
            self.logger.info("4. Hardware Optimization for TAS...")
            hardware_result = self.hardware_optimizer.optimize_regime_detection(
                data=market_data.values,
                regime_config={'n_regimes': 3, 'complexity_factor': 0.5}
            )
            self.logger.info(f"   Hardware optimization completed for TAS")
            
            # 5. Regime Analysis for TAS
            self.logger.info("5. Regime Analysis for TAS...")
            regime_result = self.regime_analyzer.analyze(
                regime_predictions=regime_predictions,
                regime_probabilities=regime_probabilities,
                market_data=market_data.values
            )
            self.logger.info(f"   Overall stability: {regime_result.overall_stability:.3f}")
            self.logger.info(f"   Number of regimes: {regime_result.n_regimes}")
            
            # 6. Validation for TAS
            self.logger.info("6. Validation for TAS...")
            # Create a mock model for validation
            class MockTASModel:
                def fit(self, X, y):
                    pass
                def predict(self, X):
                    return np.random.randint(0, 3, len(X))
            
            mock_model = MockTASModel()
            validation_result = self.validator.validate(
                model=mock_model,
                X=market_data.values,
                y=regime_predictions,
                market_data=market_data.values,
                regime_predictions=regime_predictions
            )
            self.logger.info(f"   Validation success: {validation_result.success}")
            if validation_result.metrics:
                self.logger.info(f"   Accuracy: {validation_result.metrics.get('accuracy', 0.0):.3f}")
                self.logger.info(f"   F1 Score: {validation_result.metrics.get('f1_score', 0.0):.3f}")
            
            self.logger.info("✅ TAS integration demonstration completed")
            
        except Exception as e:
            self.logger.error(f"TAS integration demonstration failed: {e}")
            raise
    
    def demonstrate_nas_integration(self):
        """Demonstrate unified utilities integration with NAS system."""
        self.logger.info("🧠 Demonstrating NAS Integration with Unified Utilities")
        
        try:
            # Generate sample NAS data
            market_data = self._generate_sample_market_data()
            regime_predictions = self._generate_sample_regime_predictions()
            regime_probabilities = self._generate_sample_regime_probabilities()
            
            # 1. Economic Significance Evaluation for NAS
            self.logger.info("1. Economic Significance Evaluation for NAS...")
            economic_result = quick_economic_evaluation(
                market_data=market_data,
                regime_predictions=regime_predictions
            )
            self.logger.info(f"   Economic significance score: {economic_result.overall_score:.3f}")
            self.logger.info(f"   Significance level: {economic_result.significance_level}")
            
            # 2. Trading Viability Assessment for NAS
            self.logger.info("2. Trading Viability Assessment for NAS...")
            trading_result = quick_trading_viability_evaluation(
                market_data=market_data,
                regime_predictions=regime_predictions
            )
            self.logger.info(f"   Trading viability score: {trading_result.overall_score:.3f}")
            self.logger.info(f"   Viability level: {trading_result.viability_level}")
            
            # 3. Multi-Objective Optimization for NAS
            self.logger.info("3. Multi-Objective Optimization for NAS...")
            optimization_result = quick_multi_objective_optimization(
                market_data=market_data,
                regime_predictions=regime_predictions
            )
            self.logger.info(f"   Optimization success: {optimization_result.success}")
            self.logger.info(f"   Best score: {optimization_result.weighted_score:.3f}")
            self.logger.info(f"   Pareto solutions: {optimization_result.n_solutions}")
            
            # 4. Hardware Optimization for NAS
            self.logger.info("4. Hardware Optimization for NAS...")
            hardware_result = quick_hardware_optimization(
                data=market_data.values,
                workload_type='neural_inference',
                config=None
            )
            self.logger.info(f"   Hardware optimization completed for NAS")
            
            # 5. Regime Analysis for NAS
            self.logger.info("5. Regime Analysis for NAS...")
            regime_result = quick_regime_analysis(
                regime_predictions=regime_predictions,
                regime_probabilities=regime_probabilities
            )
            self.logger.info(f"   Overall stability: {regime_result.overall_stability:.3f}")
            self.logger.info(f"   Number of regimes: {regime_result.n_regimes}")
            
            # 6. Validation for NAS
            self.logger.info("6. Validation for NAS...")
            # Create a mock model for validation
            class MockNASModel:
                def fit(self, X, y):
                    pass
                def predict(self, X):
                    return np.random.randint(0, 3, len(X))
            
            mock_model = MockNASModel()
            validation_result = quick_validation(
                model=mock_model,
                X=market_data.values,
                y=regime_predictions
            )
            self.logger.info(f"   Validation success: {validation_result.success}")
            if validation_result.metrics:
                self.logger.info(f"   Accuracy: {validation_result.metrics.get('accuracy', 0.0):.3f}")
                self.logger.info(f"   F1 Score: {validation_result.metrics.get('f1_score', 0.0):.3f}")
            
            self.logger.info("✅ NAS integration demonstration completed")
            
        except Exception as e:
            self.logger.error(f"NAS integration demonstration failed: {e}")
            raise
    
    def demonstrate_configuration_management(self):
        """Demonstrate unified configuration management."""
        self.logger.info("⚙️ Demonstrating Unified Configuration Management")
        
        try:
            # 1. Create environment-specific configurations
            self.logger.info("1. Creating environment-specific configurations...")
            
            dev_config = create_environment_config("development")
            prod_config = create_environment_config("production")
            test_config = create_environment_config("testing")
            
            self.logger.info(f"   Development config: {dev_config.config.environment}")
            self.logger.info(f"   Production config: {prod_config.config.environment}")
            self.logger.info(f"   Testing config: {test_config.config.environment}")
            
            # 2. Load configuration from file
            self.logger.info("2. Loading configuration from file...")
            try:
                # Save a sample configuration
                sample_config = UnifiedRegimeConfig(
                    system_name="sample_system",
                    version="1.0.0",
                    environment="development",
                    n_regimes=3,
                    economic_significance_threshold=0.6,
                    trading_viability_threshold=0.6
                )
                
                config_manager = create_unified_config_manager(sample_config)
                config_manager.save_to_file("/tmp/sample_config.json")
                
                # Load the configuration
                loaded_config = load_config_from_file("/tmp/sample_config.json")
                self.logger.info(f"   Loaded config system name: {loaded_config.config.system_name}")
                self.logger.info(f"   Loaded config version: {loaded_config.config.version}")
                
            except Exception as e:
                self.logger.warning(f"Configuration file operations not available: {e}")
            
            # 3. Validate configuration
            self.logger.info("3. Validating configuration...")
            is_valid = self.config_manager.validate_config()
            self.logger.info(f"   Configuration valid: {is_valid}")
            
            # 4. Get configuration summary
            self.logger.info("4. Getting configuration summary...")
            summary = self.config_manager.get_config_summary()
            self.logger.info(f"   System: {summary['system_name']}")
            self.logger.info(f"   Version: {summary['version']}")
            self.logger.info(f"   Environment: {summary['environment']}")
            self.logger.info(f"   N Regimes: {summary['n_regimes']}")
            
            self.logger.info("✅ Configuration management demonstration completed")
            
        except Exception as e:
            self.logger.error(f"Configuration management demonstration failed: {e}")
            raise
    
    def demonstrate_quick_functions(self):
        """Demonstrate quick convenience functions."""
        self.logger.info("🚀 Demonstrating Quick Convenience Functions")
        
        try:
            # Generate sample data
            market_data = self._generate_sample_market_data()
            regime_predictions = self._generate_sample_regime_predictions()
            
            # 1. Quick economic evaluation
            self.logger.info("1. Quick economic evaluation...")
            economic_result = quick_economic_evaluation(market_data, regime_predictions)
            self.logger.info(f"   Quick economic score: {economic_result.overall_score:.3f}")
            
            # 2. Quick trading viability evaluation
            self.logger.info("2. Quick trading viability evaluation...")
            trading_result = quick_trading_viability_evaluation(market_data, regime_predictions)
            self.logger.info(f"   Quick trading viability: {trading_result.overall_score:.3f}")
            
            # 3. Quick multi-objective optimization
            self.logger.info("3. Quick multi-objective optimization...")
            optimization_result = quick_multi_objective_optimization(market_data, regime_predictions)
            self.logger.info(f"   Quick optimization success: {optimization_result.success}")
            
            # 4. Quick hardware optimization
            self.logger.info("4. Quick hardware optimization...")
            hardware_result = quick_hardware_optimization(market_data.values, 'regime_detection')
            self.logger.info(f"   Quick hardware optimization completed")
            
            # 5. Quick regime analysis
            self.logger.info("5. Quick regime analysis...")
            regime_result = quick_regime_analysis(regime_predictions)
            self.logger.info(f"   Quick regime stability: {regime_result.overall_stability:.3f}")
            
            # 6. Quick validation
            self.logger.info("6. Quick validation...")
            class MockModel:
                def fit(self, X, y): pass
                def predict(self, X): return np.random.randint(0, 3, len(X))
            
            validation_result = quick_validation(MockModel(), market_data.values, regime_predictions)
            self.logger.info(f"   Quick validation success: {validation_result.success}")
            
            self.logger.info("✅ Quick functions demonstration completed")
            
        except Exception as e:
            self.logger.error(f"Quick functions demonstration failed: {e}")
            raise
    
    def _generate_sample_market_data(self) -> pd.DataFrame:
        """Generate sample market data for demonstration."""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate OHLCV data
        prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.01)
        volumes = np.random.randint(1000, 10000, n_samples)
        
        data = {
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
            'open': prices + np.random.randn(n_samples) * 0.1,
            'high': prices + np.abs(np.random.randn(n_samples) * 0.2),
            'low': prices - np.abs(np.random.randn(n_samples) * 0.2),
            'close': prices,
            'volume': volumes
        }
        
        return pd.DataFrame(data)
    
    def _generate_sample_regime_predictions(self) -> np.ndarray:
        """Generate sample regime predictions."""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate regime predictions with some structure
        regimes = np.zeros(n_samples, dtype=int)
        current_regime = 0
        
        for i in range(n_samples):
            if i > 0 and np.random.random() < 0.05:  # 5% chance of regime change
                current_regime = (current_regime + 1) % 3
            regimes[i] = current_regime
        
        return regimes
    
    def _generate_sample_regime_probabilities(self) -> np.ndarray:
        """Generate sample regime probabilities."""
        np.random.seed(42)
        n_samples = 1000
        n_regimes = 3
        
        # Generate probabilities that sum to 1
        probabilities = np.random.rand(n_samples, n_regimes)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        
        return probabilities
    
    def run_full_demonstration(self):
        """Run the full demonstration of unified utilities integration."""
        self.logger.info("🎯 Starting Full Unified Utilities Integration Demonstration")
        self.logger.info("=" * 80)
        
        try:
            # 1. TAS Integration
            self.demonstrate_tas_integration()
            self.logger.info("")
            
            # 2. NAS Integration
            self.demonstrate_nas_integration()
            self.logger.info("")
            
            # 3. Configuration Management
            self.demonstrate_configuration_management()
            self.logger.info("")
            
            # 4. Quick Functions
            self.demonstrate_quick_functions()
            self.logger.info("")
            
            self.logger.info("=" * 80)
            self.logger.info("🎉 Full Unified Utilities Integration Demonstration Completed!")
            self.logger.info("=" * 80)
            
            self.logger.info("\n📋 DEMONSTRATION SUMMARY:")
            self.logger.info("   ✅ TAS Integration with Unified Utilities")
            self.logger.info("   ✅ NAS Integration with Unified Utilities")
            self.logger.info("   ✅ Unified Configuration Management")
            self.logger.info("   ✅ Quick Convenience Functions")
            self.logger.info("   ✅ Economic Significance Evaluation")
            self.logger.info("   ✅ Trading Viability Assessment")
            self.logger.info("   ✅ Multi-Objective Optimization")
            self.logger.info("   ✅ Hardware Optimization")
            self.logger.info("   ✅ Regime Analysis")
            self.logger.info("   ✅ Validation and Metrics")
            
            self.logger.info("\n🚀 BENEFITS DEMONSTRATED:")
            self.logger.info("   • Code Reusability: Same utilities work for both TAS and NAS")
            self.logger.info("   • Consistency: Unified interfaces across systems")
            self.logger.info("   • Maintainability: Single source of truth for shared functionality")
            self.logger.info("   • Extensibility: Easy to add new features to both systems")
            self.logger.info("   • Performance: Optimized shared implementations")
            self.logger.info("   • Testing: Centralized testing of shared functionality")
            
        except Exception as e:
            self.logger.error(f"Full demonstration failed: {e}")
            raise


def main():
    """Main function to run the unified utilities integration example."""
    try:
        # Create and run the integration example
        example = UnifiedUtilitiesIntegrationExample()
        example.run_full_demonstration()
        
    except Exception as e:
        print(f"❌ Integration example failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())