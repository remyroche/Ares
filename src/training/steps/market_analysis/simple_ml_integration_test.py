"""
Simple ML Integration Test

This script tests the basic integration of ML utilities
without requiring external dependencies.
"""

import sys
import os
import logging
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

def test_ml_common_imports():
    """Test importing ML common utilities."""
    print("🧪 Testing ML Common Utilities Imports...")
    
    try:
        # Test basic import
        from src.utils.ml_common import ML_COMMON_AVAILABLE
        print(f"✅ ML common utilities available: {ML_COMMON_AVAILABLE}")
        
        # Test specific imports
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
        return False

def test_integration_files():
    """Test that integration files exist and are properly structured."""
    print("\n🧪 Testing Integration Files...")
    
    try:
        # Check if integration files exist
        integration_files = [
            'comprehensive_ml_integration_example.py',
            'test_ml_integration.py',
            'simple_ml_integration_test.py',
            'ML_INTEGRATION_GUIDE.md'
        ]
        
        for file in integration_files:
            if os.path.exists(file):
                print(f"✅ {file} exists")
            else:
                print(f"❌ {file} missing")
                return False
        
        # Check if target system files have been modified
        target_files = [
            'tas_regime/core/tas_engine.py',
            'nas_regime/core/enhanced_nas_engine.py',
            'hybrid_nas_tas_regime/enhanced_hybrid_orchestrator.py'
        ]
        
        for file in target_files:
            if os.path.exists(file):
                print(f"✅ {file} exists")
                
                # Check if file contains ML common utilities
                with open(file, 'r') as f:
                    content = f.read()
                    if 'ml_common' in content and '_initialize_ml_common_utilities' in content:
                        print(f"✅ {file} contains ML common utilities integration")
                    else:
                        print(f"⚠️ {file} may not have ML common utilities integration")
            else:
                print(f"❌ {file} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Integration files test failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("🚀 Starting Simple ML Integration Tests")
    print("=" * 60)
    
    tests = [
        ("ML Common Imports", test_ml_common_imports),
        ("Target System Imports", test_target_system_imports),
        ("ML Common Initialization", test_ml_common_initialization),
        ("Target System Initialization", test_target_system_initialization),
        ("Integration Files", test_integration_files)
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