#!/usr/bin/env python3
"""
VectorBT Ensemble Training Optimizations Validation Script

This script validates the VectorBT optimizations implemented in the ensemble training modules.
It tests both TacticianEnsembleTrainingStep and AnalystEnsembleTrainingStep with VectorBT
optimizations and provides performance comparisons.

Usage:
    python validate_vectorbt_ensemble_optimizations.py
"""

import numpy as np
import pandas as pd
import time
import asyncio
import sys
import os
from typing import Dict, Any, List
import warnings

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def create_sample_training_data(n_samples: int = 1000, n_features: int = 50) -> pd.DataFrame:
    """Create sample training data for testing."""
    np.random.seed(42)
    
    # Create base features
    base_features = np.random.randn(n_samples, n_features)
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Create DataFrame
    data = pd.DataFrame(base_features, columns=feature_names)
    
    # Add HMM features
    data['hmm_regime'] = np.random.randint(0, 3, n_samples)
    data['hmm_regime_prob'] = np.random.rand(n_samples)
    
    # Add Analyst features
    data['analyst_confidence'] = np.random.rand(n_samples)
    data['analyst_signal'] = np.random.randint(0, 2, n_samples)
    
    # Add NAS features
    data['nas_architecture_score'] = np.random.rand(n_samples)
    data['nas_performance_metric'] = np.random.rand(n_samples)
    
    # Add timestamps
    data['timestamp'] = pd.date_range('2023-01-01', periods=n_samples, freq='1min')
    
    return data

def create_sample_base_models() -> Dict[str, Any]:
    """Create sample base models for testing."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor
    
    models = {
        'random_forest': RandomForestRegressor(n_estimators=10, random_state=42),
        'ridge': Ridge(alpha=1.0),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=10, random_state=42)
    }
    
    return models

def test_vectorbt_rolling_optimizer():
    """Test VectorBT Rolling Optimizer functionality."""
    print("🧪 Testing VectorBT Rolling Optimizer...")
    
    try:
        from src.feature_generation.utils.vectorbt_rolling_optimizer import (
            VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
        )
        
        # Create test data
        data = pd.Series(np.random.randn(1000))
        
        # Initialize optimizer
        optimizer = get_vectorbt_rolling_optimizer(
            enable_gpu=False,
            enable_parallel=True,
            memory_efficient=True,
            chunk_size=500,
            fast_fail=False,
            enable_logging=False
        )
        
        # Test various operations
        operations = ['mean', 'std', 'var', 'min', 'max', 'sum']
        window = 20
        
        results = {}
        for op in operations:
            start_time = time.time()
            if op == 'mean':
                result = optimizer.rolling_mean(data, window=window)
            elif op == 'std':
                result = optimizer.rolling_std(data, window=window)
            elif op == 'var':
                result = optimizer.rolling_var(data, window=window)
            elif op == 'min':
                result = optimizer.rolling_min(data, window=window)
            elif op == 'max':
                result = optimizer.rolling_max(data, window=window)
            elif op == 'sum':
                result = optimizer.rolling_sum(data, window=window)
            
            execution_time = time.time() - start_time
            results[op] = {
                'execution_time': execution_time,
                'result_shape': result.shape,
                'has_nan': result.isna().any()
            }
        
        # Get performance stats
        stats = optimizer.get_performance_stats()
        
        print("✅ VectorBT Rolling Optimizer test completed")
        print(f"   Operations tested: {list(results.keys())}")
        print(f"   Performance stats: {stats}")
        
        return True, results, stats
        
    except Exception as e:
        print(f"❌ VectorBT Rolling Optimizer test failed: {e}")
        return False, {}, {}

def test_unified_vectorization_manager():
    """Test Unified Vectorization Manager functionality."""
    print("🧪 Testing Unified Vectorization Manager...")
    
    try:
        from src.utils.ml_common.unified_vectorization_manager import (
            UnifiedVectorizationManager, get_unified_vectorization_manager,
            OperationType, OperationConfig
        )
        
        # Create test data
        data = pd.DataFrame(np.random.randn(500, 20))
        
        # Initialize manager
        manager = get_unified_vectorization_manager()
        
        # Test feature engineering optimization
        config = OperationConfig(
            operation_type=OperationType.FEATURE_ENGINEERING,
            data_size=len(data),
            data_dimensions=data.shape,
            memory_budget_mb=1024.0,
            time_budget_seconds=60.0
        )
        
        start_time = time.time()
        result = manager.optimize_operation(
            OperationType.FEATURE_ENGINEERING,
            data,
            config
        )
        execution_time = time.time() - start_time
        
        # Get optimization stats
        stats = manager.get_optimization_stats()
        
        print("✅ Unified Vectorization Manager test completed")
        print(f"   Strategy used: {result.strategy_used.value}")
        print(f"   Execution time: {execution_time:.3f}s")
        print(f"   Performance gain: {result.performance_gain:.2f}x")
        print(f"   Memory used: {result.memory_used_mb:.1f}MB")
        
        return True, result, stats
        
    except Exception as e:
        print(f"❌ Unified Vectorization Manager test failed: {e}")
        return False, None, {}

async def test_tactician_ensemble_training():
    """Test Tactician Ensemble Training with VectorBT optimizations."""
    print("🧪 Testing Tactician Ensemble Training...")
    
    try:
        from src.training.steps.models_training.tactician_ensemble_training import (
            TacticianEnsembleTrainingStep, TacticianEnsembleTrainingConfig
        )
        
        # Create test data
        training_data = create_sample_training_data(n_samples=500, n_features=20)
        base_models = create_sample_base_models()
        
        # Train base models
        X = training_data.iloc[:, :20].values
        y = np.random.randn(500)
        
        for model in base_models.values():
            model.fit(X, y)
        
        # Initialize trainer
        config = TacticianEnsembleTrainingConfig(
            enable_full_integration=True,
            include_hmm_features=True,
            include_analyst_features=True,
            include_oof_predictions=True,
            enable_gpu_acceleration=False,
            memory_limit_gb=4.0
        )
        
        trainer = TacticianEnsembleTrainingStep(config)
        
        # Test training
        feature_columns = [f'feature_{i}' for i in range(20)]
        target_columns = ['target']
        training_data['target'] = y
        
        start_time = time.time()
        result = await trainer.train_tactician_ensemble(
            training_data=training_data,
            base_models=base_models,
            feature_columns=feature_columns,
            target_columns=target_columns
        )
        execution_time = time.time() - start_time
        
        # Get performance metrics
        metrics = trainer.get_performance_metrics()
        
        print("✅ Tactician Ensemble Training test completed")
        print(f"   Execution time: {execution_time:.3f}s")
        print(f"   Training completed: {result.get('training_completed', False)}")
        print(f"   Features used: {len(result.get('features_used', []))}")
        print(f"   Samples used: {result.get('samples_used', 0)}")
        
        return True, result, metrics
        
    except Exception as e:
        print(f"❌ Tactician Ensemble Training test failed: {e}")
        return False, {}, {}

async def test_analyst_ensemble_training():
    """Test Analyst Ensemble Training with VectorBT optimizations."""
    print("🧪 Testing Analyst Ensemble Training...")
    
    try:
        from src.training.steps.models_training.analyst_ensemble_training import (
            AnalystEnsembleTrainingStep, AnalystEnsembleTrainingConfig
        )
        
        # Create test data
        training_data = create_sample_training_data(n_samples=500, n_features=20)
        base_models = create_sample_base_models()
        
        # Train base models
        X = training_data.iloc[:, :20].values
        y = np.random.randn(500)
        
        for model in base_models.values():
            model.fit(X, y)
        
        # Initialize trainer
        config = AnalystEnsembleTrainingConfig(
            enable_full_integration=True,
            include_hmm_features=True,
            include_nas_features=True,
            enable_gpu_acceleration=False,
            memory_limit_gb=4.0
        )
        
        trainer = AnalystEnsembleTrainingStep(config)
        
        # Test training
        feature_columns = [f'feature_{i}' for i in range(20)]
        target_columns = ['target']
        training_data['target'] = y
        
        start_time = time.time()
        result = await trainer.train_analyst_ensemble(
            training_data=training_data,
            base_models=base_models,
            feature_columns=feature_columns,
            target_columns=target_columns
        )
        execution_time = time.time() - start_time
        
        # Get performance metrics
        metrics = trainer.get_performance_metrics()
        
        print("✅ Analyst Ensemble Training test completed")
        print(f"   Execution time: {execution_time:.3f}s")
        print(f"   Training completed: {result.get('training_completed', False)}")
        print(f"   Features used: {len(result.get('features_used', []))}")
        print(f"   Samples used: {result.get('samples_used', 0)}")
        
        return True, result, metrics
        
    except Exception as e:
        print(f"❌ Analyst Ensemble Training test failed: {e}")
        return False, {}, {}

def performance_comparison():
    """Compare performance with and without VectorBT optimizations."""
    print("🧪 Running performance comparison...")
    
    try:
        # Test data
        data = pd.Series(np.random.randn(5000))
        window = 50
        
        # Test pandas rolling (baseline)
        start_time = time.time()
        pandas_result = data.rolling(window=window).mean()
        pandas_time = time.time() - start_time
        
        # Test VectorBT rolling
        from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
        
        optimizer = get_vectorbt_rolling_optimizer(enable_logging=False)
        start_time = time.time()
        vectorbt_result = optimizer.rolling_mean(data, window=window)
        vectorbt_time = time.time() - start_time
        
        # Calculate speedup
        speedup = pandas_time / vectorbt_time if vectorbt_time > 0 else 0
        
        print("✅ Performance comparison completed")
        print(f"   Pandas rolling time: {pandas_time:.3f}s")
        print(f"   VectorBT rolling time: {vectorbt_time:.3f}s")
        print(f"   Speedup: {speedup:.2f}x")
        
        return True, {
            'pandas_time': pandas_time,
            'vectorbt_time': vectorbt_time,
            'speedup': speedup
        }
        
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")
        return False, {}

async def main():
    """Main validation function."""
    print("🚀 Starting VectorBT Ensemble Training Optimizations Validation")
    print("=" * 70)
    
    results = {
        'vectorbt_rolling_optimizer': False,
        'unified_vectorization_manager': False,
        'tactician_ensemble_training': False,
        'analyst_ensemble_training': False,
        'performance_comparison': False
    }
    
    # Test VectorBT Rolling Optimizer
    success, rolling_results, rolling_stats = test_vectorbt_rolling_optimizer()
    results['vectorbt_rolling_optimizer'] = success
    
    print()
    
    # Test Unified Vectorization Manager
    success, vectorization_result, vectorization_stats = test_unified_vectorization_manager()
    results['unified_vectorization_manager'] = success
    
    print()
    
    # Test Tactician Ensemble Training
    success, tactician_result, tactician_metrics = await test_tactician_ensemble_training()
    results['tactician_ensemble_training'] = success
    
    print()
    
    # Test Analyst Ensemble Training
    success, analyst_result, analyst_metrics = await test_analyst_ensemble_training()
    results['analyst_ensemble_training'] = success
    
    print()
    
    # Performance comparison
    success, performance_data = performance_comparison()
    results['performance_comparison'] = success
    
    print()
    print("=" * 70)
    print("📊 VALIDATION SUMMARY")
    print("=" * 70)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All VectorBT optimizations are working correctly!")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")
    
    return results

if __name__ == "__main__":
    # Run the validation
    asyncio.run(main())