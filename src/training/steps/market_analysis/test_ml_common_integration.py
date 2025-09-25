"""
Test script to verify ML Common utilities integration in TAS, NAS, and Hybrid regime models.

This script tests that all the ML common utilities are properly imported and functional
in the enhanced models.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def test_ml_common_imports():
    """Test that all ML common utilities can be imported successfully."""
    print("🧪 Testing ML Common utilities imports...")

    try:
        # Test basic ML common imports
        from src.utils.ml_common import (
            EnhancedModelTrainer, train_model_with_confidence_metrics,
            ModelEvaluator, ModelRegistry,
            UnifiedCrossValidator, perform_cross_validation, temporal_cross_validation,
            ConfigurationValidator, optimize_threshold, calibrate_probabilities,
            RegimeSpecificTPSLOptimizer,
            StackingEnsembleManager, create_analyst_ensemble,
            MemoryOptimizer, UnifiedCache, get_unified_cache,
            LookaheadProtection, MLTrainingSafeguards, RobustErrorHandler,
            setup_logger, get_logger
        )
        print("✅ All ML Common utilities imported successfully")

        # Test that we can create instances
        logger = get_logger(__name__)
        cache = get_unified_cache()
        safeguards = MLTrainingSafeguards()
        error_handler = RobustErrorHandler()
        memory_optimizer = MemoryOptimizer()
        lookahead_protection = LookaheadProtection()
        model_registry = ModelRegistry()
        regime_optimizer = RegimeSpecificTPSLOptimizer()
        config_validator = ConfigurationValidator()
        ensemble_manager = StackingEnsembleManager()

        print("✅ All ML Common utility instances created successfully")
        return True

    except Exception as e:
        print(f"❌ ML Common utilities import failed: {e}")
        return False

def test_tas_engine_integration():
    """Test TAS engine with ML common utilities integration."""
    print("\n🧪 Testing TAS Engine ML Common integration...")

    try:
        from src.training.steps.market_analysis.tas_regime.core.enhanced_tas_engine import (
            EnhancedTASEngine, TASConfig, TreeSearchStrategy, create_enhanced_tas_engine
        )

        # Create TAS config with ML utilities enabled
        config = TASConfig(
            search_strategy=TreeSearchStrategy.ENHANCED_BAYESIAN,
            population_size=10,
            max_generations=5,
            max_evaluations=20,
            enable_multi_objective=True,
            enable_constraint_validation=True,
            enable_performance_estimation=True,
            parallel_evaluation=True
        )

        # Create engine
        engine = EnhancedTASEngine(config)

        # Check that ML common components are initialized
        assert hasattr(engine, 'safeguards'), "TAS Engine missing safeguards"
        assert hasattr(engine, 'error_handler'), "TAS Engine missing error handler"
        assert hasattr(engine, 'memory_optimizer'), "TAS Engine missing memory optimizer"
        assert hasattr(engine, 'lookahead_protection'), "TAS Engine missing lookahead protection"
        assert hasattr(engine, 'cache'), "TAS Engine missing cache"
        assert hasattr(engine, 'model_registry'), "TAS Engine missing model registry"

        print("✅ TAS Engine ML Common components initialized correctly")
        return True

    except Exception as e:
        print(f"❌ TAS Engine integration test failed: {e}")
        return False

def test_nas_engine_integration():
    """Test NAS engine with ML common utilities integration."""
    print("\n🧪 Testing NAS Engine ML Common integration...")

    try:
        from src.training.steps.market_analysis.nas_regime.core.enhanced_nas_engine import (
            EnhancedNASEngine, NASSearchConfig, SearchStrategy, create_enhanced_nas_engine
        )

        # Create NAS config with ML utilities enabled
        config = NASSearchConfig(
            search_strategy=SearchStrategy.ENHANCED_BAYESIAN,
            population_size=10,
            max_generations=5,
            max_evaluations=20,
            enable_multi_objective=True,
            enable_constraint_validation=True,
            enable_performance_estimation=True,
            parallel_evaluation=True
        )

        # Create engine
        engine = EnhancedNASEngine(config)

        # Check that ML common components are initialized
        assert hasattr(engine, 'safeguards'), "NAS Engine missing safeguards"
        assert hasattr(engine, 'error_handler'), "NAS Engine missing error handler"
        assert hasattr(engine, 'memory_optimizer'), "NAS Engine missing memory optimizer"
        assert hasattr(engine, 'lookahead_protection'), "NAS Engine missing lookahead protection"
        assert hasattr(engine, 'cache'), "NAS Engine missing cache"
        assert hasattr(engine, 'model_registry'), "NAS Engine missing model registry"

        print("✅ NAS Engine ML Common components initialized correctly")
        return True

    except Exception as e:
        print(f"❌ NAS Engine integration test failed: {e}")
        return False

def test_hybrid_orchestrator_integration():
    """Test Hybrid Orchestrator with ML common utilities integration."""
    print("\n🧪 Testing Hybrid Orchestrator ML Common integration...")

    try:
        from src.utils.nas_tas.enhanced_hybrid_orchestrator import (
            EnhancedHybridOrchestrator, HybridRegimeConfig, create_enhanced_hybrid_orchestrator
        )
        from src.utils.nas_tas.config.hybrid_regime_config import (
            RegimeCombinationStrategy
        )

        # Create hybrid config with ML utilities enabled
        config = HybridRegimeConfig(
            combination_strategy=RegimeCombinationStrategy.WEIGHTED_AVERAGE,
            n_regimes=5,
            enable_multi_timeframe=True,
            use_unified_search=True,
            use_signal_generation=True
        )

        # Create orchestrator
        orchestrator = EnhancedHybridOrchestrator(config)

        # Check that ML common components are initialized
        assert hasattr(orchestrator, 'safeguards'), "Hybrid Orchestrator missing safeguards"
        assert hasattr(orchestrator, 'error_handler'), "Hybrid Orchestrator missing error handler"
        assert hasattr(orchestrator, 'memory_optimizer'), "Hybrid Orchestrator missing memory optimizer"
        assert hasattr(orchestrator, 'lookahead_protection'), "Hybrid Orchestrator missing lookahead protection"
        assert hasattr(orchestrator, 'cache'), "Hybrid Orchestrator missing cache"
        assert hasattr(orchestrator, 'ensemble_manager'), "Hybrid Orchestrator missing ensemble manager"

        print("✅ Hybrid Orchestrator ML Common components initialized correctly")
        return True

    except Exception as e:
        print(f"❌ Hybrid Orchestrator integration test failed: {e}")
        return False

def test_functionality_with_mock_data():
    """Test functionality with mock data to ensure utilities work end-to-end."""
    print("\n🧪 Testing functionality with mock data...")

    try:
        # Create mock market data
        dates = pd.date_range('2023-01-01', periods=100, freq='15min')
        mock_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(100, 200, 100),
            'low': np.random.uniform(100, 200, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })

        # Test TAS engine with mock data
        from src.training.steps.market_analysis.tas_regime import (
            quick_tas_search, TASConfig, TreeSearchStrategy
        )

        train_data = (mock_data[['open', 'high', 'low', 'close', 'volume']].values[:60],
                     np.random.randint(0, 3, 60))  # 3 regimes
        val_data = (mock_data[['open', 'high', 'low', 'close', 'volume']].values[60:80],
                   np.random.randint(0, 3, 20))

        config = TASConfig(
            search_strategy=TreeSearchStrategy.ENHANCED_BAYESIAN,
            population_size=5,
            max_generations=3,
            max_evaluations=10
        )

        result = quick_tas_search(train_data, val_data, config)

        # Check that result has ML utilities metadata
        assert result.metadata.get('ml_common_utilities_used', False), "TAS result missing ML utilities metadata"
        assert result.metadata.get('safeguards_applied', False), "TAS result missing safeguards metadata"
        assert result.metadata.get('cross_validation_performed', False), "TAS result missing CV metadata"

        print("✅ TAS functionality test completed successfully")
        print(f"   Best Score: {result.best_score:.4f}")
        print(f"   ML Common utilities used: {result.metadata.get('ml_common_utilities_used', False)}")

        return True

    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🚀 Starting ML Common Integration Tests")
    print("=" * 50)

    tests = [
        test_ml_common_imports,
        test_tas_engine_integration,
        test_nas_engine_integration,
        test_hybrid_orchestrator_integration,
        test_functionality_with_mock_data
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! ML Common utilities are properly integrated.")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed. Please check the integration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)