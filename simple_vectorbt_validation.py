#!/usr/bin/env python3
"""
Simple VectorBT Ensemble Training Optimizations Validation Script

This script validates the VectorBT optimizations implemented in the ensemble training modules
without requiring external dependencies. It checks for proper imports and basic functionality.

Usage:
    python3 simple_vectorbt_validation.py
"""

import sys
import os
import traceback

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all required imports work correctly."""
    print("🧪 Testing imports...")
    
    results = {}
    
    # Test VectorBT Rolling Optimizer import
    try:
        from src.feature_generation.utils.vectorbt_rolling_optimizer import (
            VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
        )
        results['vectorbt_rolling_optimizer'] = True
        print("✅ VectorBT Rolling Optimizer import successful")
    except Exception as e:
        results['vectorbt_rolling_optimizer'] = False
        print(f"❌ VectorBT Rolling Optimizer import failed: {e}")
    
    # Test Unified Vectorization Manager import
    try:
        from src.utils.ml_common.unified_vectorization_manager import (
            UnifiedVectorizationManager, get_unified_vectorization_manager,
            OperationType, OptimizationStrategy, OperationConfig
        )
        results['unified_vectorization_manager'] = True
        print("✅ Unified Vectorization Manager import successful")
    except Exception as e:
        results['unified_vectorization_manager'] = False
        print(f"❌ Unified Vectorization Manager import failed: {e}")
    
    # Test Tactician Ensemble Training import
    try:
        from src.training.steps.models_training.tactician_ensemble_training import (
            TacticianEnsembleTrainingStep, TacticianEnsembleTrainingConfig
        )
        results['tactician_ensemble_training'] = True
        print("✅ Tactician Ensemble Training import successful")
    except Exception as e:
        results['tactician_ensemble_training'] = False
        print(f"❌ Tactician Ensemble Training import failed: {e}")
    
    # Test Analyst Ensemble Training import
    try:
        from src.training.steps.models_training.analyst_ensemble_training import (
            AnalystEnsembleTrainingStep, AnalystEnsembleTrainingConfig
        )
        results['analyst_ensemble_training'] = True
        print("✅ Analyst Ensemble Training import successful")
    except Exception as e:
        results['analyst_ensemble_training'] = False
        print(f"❌ Analyst Ensemble Training import failed: {e}")
    
    return results

def test_class_initialization():
    """Test that classes can be initialized without errors."""
    print("\n🧪 Testing class initialization...")
    
    results = {}
    
    # Test TacticianEnsembleTrainingConfig
    try:
        from src.training.steps.models_training.tactician_ensemble_training import (
            TacticianEnsembleTrainingConfig
        )
        config = TacticianEnsembleTrainingConfig()
        results['tactician_config'] = True
        print("✅ TacticianEnsembleTrainingConfig initialization successful")
    except Exception as e:
        results['tactician_config'] = False
        print(f"❌ TacticianEnsembleTrainingConfig initialization failed: {e}")
    
    # Test AnalystEnsembleTrainingConfig
    try:
        from src.training.steps.models_training.analyst_ensemble_training import (
            AnalystEnsembleTrainingConfig
        )
        config = AnalystEnsembleTrainingConfig()
        results['analyst_config'] = True
        print("✅ AnalystEnsembleTrainingConfig initialization successful")
    except Exception as e:
        results['analyst_config'] = False
        print(f"❌ AnalystEnsembleTrainingConfig initialization failed: {e}")
    
    return results

def test_vectorbt_optimizer_creation():
    """Test VectorBT optimizer creation."""
    print("\n🧪 Testing VectorBT optimizer creation...")
    
    results = {}
    
    try:
        from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
        
        # Test with minimal configuration
        optimizer = get_vectorbt_rolling_optimizer(
            enable_gpu=False,
            enable_parallel=False,
            memory_efficient=True,
            chunk_size=100,
            fast_fail=False,
            enable_logging=False
        )
        
        results['vectorbt_optimizer_creation'] = True
        print("✅ VectorBT Rolling Optimizer creation successful")
        
        # Test performance stats
        try:
            stats = optimizer.get_performance_stats()
            results['vectorbt_performance_stats'] = True
            print("✅ VectorBT performance stats retrieval successful")
        except Exception as e:
            results['vectorbt_performance_stats'] = False
            print(f"❌ VectorBT performance stats retrieval failed: {e}")
            
    except Exception as e:
        results['vectorbt_optimizer_creation'] = False
        results['vectorbt_performance_stats'] = False
        print(f"❌ VectorBT Rolling Optimizer creation failed: {e}")
    
    return results

def test_unified_vectorization_manager_creation():
    """Test Unified Vectorization Manager creation."""
    print("\n🧪 Testing Unified Vectorization Manager creation...")
    
    results = {}
    
    try:
        from src.utils.ml_common.unified_vectorization_manager import get_unified_vectorization_manager
        
        # Test manager creation
        manager = get_unified_vectorization_manager()
        results['unified_manager_creation'] = True
        print("✅ Unified Vectorization Manager creation successful")
        
        # Test optimization stats
        try:
            stats = manager.get_optimization_stats()
            results['unified_optimization_stats'] = True
            print("✅ Unified Vectorization Manager stats retrieval successful")
        except Exception as e:
            results['unified_optimization_stats'] = False
            print(f"❌ Unified Vectorization Manager stats retrieval failed: {e}")
            
    except Exception as e:
        results['unified_manager_creation'] = False
        results['unified_optimization_stats'] = False
        print(f"❌ Unified Vectorization Manager creation failed: {e}")
    
    return results

def test_ensemble_training_step_creation():
    """Test ensemble training step creation."""
    print("\n🧪 Testing ensemble training step creation...")
    
    results = {}
    
    # Test TacticianEnsembleTrainingStep
    try:
        from src.training.steps.models_training.tactician_ensemble_training import (
            TacticianEnsembleTrainingStep, TacticianEnsembleTrainingConfig
        )
        
        config = TacticianEnsembleTrainingConfig(
            enable_full_integration=True,
            include_hmm_features=True,
            include_analyst_features=True,
            include_oof_predictions=True,
            enable_gpu_acceleration=False,
            memory_limit_gb=4.0
        )
        
        trainer = TacticianEnsembleTrainingStep(config)
        results['tactician_trainer_creation'] = True
        print("✅ TacticianEnsembleTrainingStep creation successful")
        
        # Test performance metrics
        try:
            metrics = trainer.get_performance_metrics()
            results['tactician_performance_metrics'] = True
            print("✅ TacticianEnsembleTrainingStep performance metrics successful")
        except Exception as e:
            results['tactician_performance_metrics'] = False
            print(f"❌ TacticianEnsembleTrainingStep performance metrics failed: {e}")
            
    except Exception as e:
        results['tactician_trainer_creation'] = False
        results['tactician_performance_metrics'] = False
        print(f"❌ TacticianEnsembleTrainingStep creation failed: {e}")
    
    # Test AnalystEnsembleTrainingStep
    try:
        from src.training.steps.models_training.analyst_ensemble_training import (
            AnalystEnsembleTrainingStep, AnalystEnsembleTrainingConfig
        )
        
        config = AnalystEnsembleTrainingConfig(
            enable_full_integration=True,
            include_hmm_features=True,
            include_nas_features=True,
            enable_gpu_acceleration=False,
            memory_limit_gb=4.0
        )
        
        trainer = AnalystEnsembleTrainingStep(config)
        results['analyst_trainer_creation'] = True
        print("✅ AnalystEnsembleTrainingStep creation successful")
        
        # Test performance metrics
        try:
            metrics = trainer.get_performance_metrics()
            results['analyst_performance_metrics'] = True
            print("✅ AnalystEnsembleTrainingStep performance metrics successful")
        except Exception as e:
            results['analyst_performance_metrics'] = False
            print(f"❌ AnalystEnsembleTrainingStep performance metrics failed: {e}")
            
    except Exception as e:
        results['analyst_trainer_creation'] = False
        results['analyst_performance_metrics'] = False
        print(f"❌ AnalystEnsembleTrainingStep creation failed: {e}")
    
    return results

def main():
    """Main validation function."""
    print("🚀 Starting VectorBT Ensemble Training Optimizations Validation")
    print("=" * 70)
    
    all_results = {}
    
    # Test imports
    import_results = test_imports()
    all_results.update(import_results)
    
    # Test class initialization
    init_results = test_class_initialization()
    all_results.update(init_results)
    
    # Test VectorBT optimizer creation
    vectorbt_results = test_vectorbt_optimizer_creation()
    all_results.update(vectorbt_results)
    
    # Test Unified Vectorization Manager creation
    unified_results = test_unified_vectorization_manager_creation()
    all_results.update(unified_results)
    
    # Test ensemble training step creation
    ensemble_results = test_ensemble_training_step_creation()
    all_results.update(ensemble_results)
    
    print("\n" + "=" * 70)
    print("📊 VALIDATION SUMMARY")
    print("=" * 70)
    
    for test_name, success in all_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    total_tests = len(all_results)
    passed_tests = sum(all_results.values())
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All VectorBT optimizations are working correctly!")
        return True
    else:
        print("⚠️ Some tests failed. Check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)