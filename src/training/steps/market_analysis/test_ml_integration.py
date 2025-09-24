"""
Test script for ML Common Utilities Integration

This script tests the comprehensive integration of ML utilities
across TAS, NAS, and Hybrid regime detection systems.
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import traceback

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

def test_ml_common_imports():
    """Test importing ML common utilities."""
    print("🧪 Testing ML Common Utilities Imports...")
    
    try:
        from src.utils.ml_common import (
            OverfittingPrevention, OverfittingPreventionConfig,
            HyperparameterOptimization,
            UnifiedCrossValidator,
            EnhancedModelFactory,
            EnsembleManager,
            FeatureImportanceAnalyzer,
            DataDriftDetector,
            LookaheadProtection,
            MLTrainingSafeguards,
            RobustErrorHandler
        )
        print("✅ All ML common utilities imported successfully")
        return True
    except ImportError as e:
        print(f"❌ ML common utilities import failed: {e}")
        return False

def test_target_system_imports():
    """Test importing target systems."""
    print("\n🧪 Testing Target System Imports...")
    
    try:
        # Test TAS engine import
        from tas_regime.core.tas_engine import TreeArchitectureSearchEngine, TASEngineConfig
        print("✅ TAS engine imported successfully")
        
        # Test NAS engine import
        from nas_regime.core.enhanced_nas_engine import EnhancedNASEngine, NASSearchConfig
        print("✅ NAS engine imported successfully")
        
        # Test hybrid orchestrator import
        from hybrid_nas_tas_regime.enhanced_hybrid_orchestrator import EnhancedHybridOrchestrator
        print("✅ Hybrid orchestrator imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Target system import failed: {e}")
        return False

def test_ml_common_initialization():
    """Test initializing ML common utilities."""
    print("\n🧪 Testing ML Common Utilities Initialization...")
    
    try:
        from src.utils.ml_common import (
            OverfittingPrevention, OverfittingPreventionConfig,
            HyperparameterOptimization,
            UnifiedCrossValidator,
            EnhancedModelFactory,
            EnsembleManager,
            FeatureImportanceAnalyzer,
            DataDriftDetector,
            LookaheadProtection,
            MLTrainingSafeguards,
            RobustErrorHandler
        )
        
        # Test overfitting prevention
        overfitting_config = OverfittingPreventionConfig(
            enable_early_stopping=True,
            enable_cross_validation=True,
            enable_regularization=True,
            enable_ensemble_diversity=True,
            early_stopping_patience=20,
            cv_folds=5
        )
        overfitting_prevention = OverfittingPrevention(overfitting_config)
        print("✅ OverfittingPrevention initialized")
        
        # Test hyperparameter optimization
        hpo_optimizer = HyperparameterOptimization({
            'enable_parallel': True,
            'max_workers': 4,
            'enable_monitoring': True,
            'use_nonlinear_optimization': True
        })
        print("✅ HyperparameterOptimization initialized")
        
        # Test validation framework
        validation_framework = UnifiedCrossValidator()
        print("✅ UnifiedCrossValidator initialized")
        
        # Test model factory
        model_factory = EnhancedModelFactory()
        print("✅ EnhancedModelFactory initialized")
        
        # Test ensemble manager
        ensemble_manager = EnsembleManager()
        print("✅ EnsembleManager initialized")
        
        # Test feature analyzer
        feature_analyzer = FeatureImportanceAnalyzer()
        print("✅ FeatureImportanceAnalyzer initialized")
        
        # Test drift detector
        drift_detector = DataDriftDetector()
        print("✅ DataDriftDetector initialized")
        
        # Test safeguards
        lookahead_protection = LookaheadProtection()
        training_safeguards = MLTrainingSafeguards()
        error_handler = RobustErrorHandler()
        print("✅ Safeguards initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ ML common utilities initialization failed: {e}")
        traceback.print_exc()
        return False

def test_target_system_initialization():
    """Test initializing target systems."""
    print("\n🧪 Testing Target System Initialization...")
    
    try:
        from tas_regime.core.tas_engine import TreeArchitectureSearchEngine, TASEngineConfig
        from nas_regime.core.enhanced_nas_engine import EnhancedNASEngine, NASSearchConfig
        from hybrid_nas_tas_regime.enhanced_hybrid_orchestrator import EnhancedHybridOrchestrator
        
        # Test TAS engine initialization
        tas_config = TASEngineConfig(
            enable_meta_learning=True,
            enable_hardware_optimization=True,
            enable_uncertainty_estimation=True,
            enable_regime_analysis=True,
            enable_real_time_adaptation=True
        )
        tas_engine = TreeArchitectureSearchEngine(tas_config)
        print("✅ TAS engine initialized")
        
        # Test NAS engine initialization
        nas_config = NASSearchConfig(
            search_strategy='enhanced_bayesian',
            population_size=50,
            max_generations=100,
            enable_multi_objective=True
        )
        nas_engine = EnhancedNASEngine(nas_config)
        print("✅ NAS engine initialized")
        
        # Test hybrid orchestrator initialization
        from hybrid_nas_tas_regime.config.hybrid_regime_config import HybridRegimeConfig
        hybrid_config = HybridRegimeConfig(
            enable_multi_timeframe=True,
            use_unified_search=True,
            use_signal_generation=True
        )
        hybrid_orchestrator = EnhancedHybridOrchestrator(hybrid_config)
        print("✅ Hybrid orchestrator initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Target system initialization failed: {e}")
        traceback.print_exc()
        return False

def test_comprehensive_integration():
    """Test comprehensive ML integration."""
    print("\n🧪 Testing Comprehensive ML Integration...")
    
    try:
        from comprehensive_ml_integration_example import ComprehensiveMLIntegration
        
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randint(0, 3, n_samples), name='target')
        
        # Initialize comprehensive ML integration
        ml_integration = ComprehensiveMLIntegration()
        
        # Run comprehensive analysis
        results = ml_integration.demonstrate_comprehensive_ml_usage(X, y)
        
        print("✅ Comprehensive ML integration test completed")
        print(f"📊 Results keys: {list(results.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive ML integration test failed: {e}")
        traceback.print_exc()
        return False

def test_ml_utilities_functionality():
    """Test ML utilities functionality."""
    print("\n🧪 Testing ML Utilities Functionality...")
    
    try:
        from src.utils.ml_common import (
            OverfittingPrevention, OverfittingPreventionConfig,
            HyperparameterOptimization,
            UnifiedCrossValidator,
            EnhancedModelFactory,
            EnsembleManager,
            FeatureImportanceAnalyzer,
            DataDriftDetector,
            LookaheadProtection,
            MLTrainingSafeguards,
            RobustErrorHandler
        )
        
        # Create sample data
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)
        
        # Test overfitting prevention
        overfitting_config = OverfittingPreventionConfig(
            enable_early_stopping=True,
            enable_cross_validation=True,
            enable_regularization=True,
            enable_ensemble_diversity=True,
            early_stopping_patience=20,
            cv_folds=5
        )
        overfitting_prevention = OverfittingPrevention(overfitting_config)
        
        # Test data drift detection
        drift_detector = DataDriftDetector()
        drift_result = drift_detector.detect_drift(X)
        print(f"✅ Data drift detection: {drift_result.drift_detected}")
        
        # Test lookahead protection
        lookahead_protection = LookaheadProtection()
        protection_result = lookahead_protection.protect_data(X, y)
        print(f"✅ Lookahead protection: {protection_result.protected}")
        
        # Test training safeguards
        training_safeguards = MLTrainingSafeguards()
        safeguard_result = training_safeguards.validate_training_setup(X, y)
        print(f"✅ Training safeguards: {safeguard_result.valid}")
        
        print("✅ ML utilities functionality test completed")
        return True
        
    except Exception as e:
        print(f"❌ ML utilities functionality test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests."""
    print("🚀 Starting ML Integration Tests")
    print("=" * 60)
    
    tests = [
        ("ML Common Imports", test_ml_common_imports),
        ("Target System Imports", test_target_system_imports),
        ("ML Common Initialization", test_ml_common_initialization),
        ("Target System Initialization", test_target_system_initialization),
        ("ML Utilities Functionality", test_ml_utilities_functionality),
        ("Comprehensive Integration", test_comprehensive_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results[test_name] = "✅ PASSED" if success else "❌ FAILED"
        except Exception as e:
            print(f"❌ Test {test_name} failed with exception: {e}")
            results[test_name] = "❌ FAILED"
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        print(f"{result} {test_name}")
    
    passed = sum(1 for result in results.values() if "✅" in result)
    total = len(results)
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! ML integration is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return results

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run all tests
    results = run_all_tests()