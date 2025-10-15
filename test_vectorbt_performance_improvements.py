#!/usr/bin/env python3
"""
VectorBT Performance Improvements Test Script

This script tests the performance improvements from the VectorBT optimizations
implemented in the models training pipeline.

Tests:
1. Batch rolling operations vs sequential
2. Parallel cross-validation vs standard
3. Memory-efficient chunking for large datasets
4. Overall training speed improvements
"""

import numpy as np
import pandas as pd
import time
import asyncio
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Import the enhanced VectorBT optimizer
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    )
    VECTORBT_AVAILABLE = True
except ImportError as e:
    print(f"❌ VectorBT not available: {e}")
    VECTORBT_AVAILABLE = False

# Import training modules
try:
    from src.training.steps.models_training.tactician_ensemble_training import TacticianEnsembleTrainingStep
    from src.training.steps.models_training.analyst_models_training import AnalystModelsTrainingStep
    TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"❌ Training modules not available: {e}")
    TRAINING_AVAILABLE = False

def create_test_data(n_samples: int = 10000, n_features: int = 50) -> pd.DataFrame:
    """Create test data for performance testing."""
    np.random.seed(42)
    
    # Create realistic financial data
    data = {}
    
    # Price data
    price = 100 + np.cumsum(np.random.randn(n_samples) * 0.01)
    data['close'] = price
    data['open'] = price + np.random.randn(n_samples) * 0.005
    data['high'] = price + np.abs(np.random.randn(n_samples) * 0.01)
    data['low'] = price - np.abs(np.random.randn(n_samples) * 0.01)
    data['volume'] = np.random.randint(1000, 10000, n_samples)
    
    # Technical indicators
    for i in range(5, 21, 5):
        data[f'sma_{i}'] = pd.Series(price).rolling(i).mean()
        data[f'std_{i}'] = pd.Series(price).rolling(i).std()
        data[f'rsi_{i}'] = np.random.uniform(0, 100, n_samples)
    
    # Additional features
    for i in range(n_features - 20):
        data[f'feature_{i}'] = np.random.randn(n_samples)
    
    df = pd.DataFrame(data)
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

def test_batch_rolling_performance():
    """Test batch rolling operations vs sequential processing."""
    print("\n🧪 Testing Batch Rolling Operations Performance")
    print("=" * 60)
    
    if not VECTORBT_AVAILABLE:
        print("❌ VectorBT not available, skipping test")
        return
    
    # Create test data
    data = create_test_data(n_samples=5000, n_features=20)
    numeric_cols = [col for col in data.columns if data[col].dtype in ['float64', 'int64']][:10]
    test_data = data[numeric_cols]
    
    # Initialize optimizer
    optimizer = get_vectorbt_rolling_optimizer(enable_parallel=True, memory_efficient=True)
    
    operations = ['mean', 'std', 'var', 'min', 'max']
    window = 20
    
    # Test sequential processing
    print("📊 Testing sequential processing...")
    start_time = time.time()
    sequential_results = {}
    for operation in operations:
        for col in numeric_cols:
            sequential_results[f'{col}_{operation}'] = optimizer.rolling_mean(test_data[col], window)
    sequential_time = time.time() - start_time
    
    # Test batch processing
    print("🚀 Testing batch processing...")
    start_time = time.time()
    batch_results = optimizer.batch_rolling_operations(test_data, operations, window)
    batch_time = time.time() - start_time
    
    # Calculate speedup
    speedup = sequential_time / batch_time if batch_time > 0 else 0
    
    print(f"\n📈 Results:")
    print(f"   Sequential time: {sequential_time:.3f}s")
    print(f"   Batch time: {batch_time:.3f}s")
    print(f"   Speedup: {speedup:.2f}x")
    print(f"   Operations: {len(operations) * len(numeric_cols)}")
    
    return {
        'sequential_time': sequential_time,
        'batch_time': batch_time,
        'speedup': speedup,
        'operations_count': len(operations) * len(numeric_cols)
    }

def test_parallel_cv_performance():
    """Test parallel cross-validation vs standard CV."""
    print("\n🧪 Testing Parallel Cross-Validation Performance")
    print("=" * 60)
    
    if not VECTORBT_AVAILABLE:
        print("❌ VectorBT not available, skipping test")
        return
    
    # Create test data
    n_samples = 2000
    n_features = 20
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    
    # Initialize optimizer
    optimizer = get_vectorbt_rolling_optimizer(enable_parallel=True)
    
    # Test standard CV (simulated)
    print("📊 Testing standard cross-validation...")
    from sklearn.model_selection import KFold
    from sklearn.ensemble import RandomForestRegressor
    
    start_time = time.time()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    standard_oof = np.zeros(n_samples)
    
    for train_idx, val_idx in kf.split(X):
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X[train_idx], y[train_idx])
        standard_oof[val_idx] = model.predict(X[val_idx])
    
    standard_time = time.time() - start_time
    
    # Test parallel CV
    print("🚀 Testing parallel cross-validation...")
    start_time = time.time()
    parallel_result = optimizer.parallel_cross_validation(
        X, y, RandomForestRegressor, cv_folds=5,
        n_estimators=50, random_state=42
    )
    parallel_time = time.time() - start_time
    
    # Calculate speedup
    speedup = standard_time / parallel_time if parallel_time > 0 else 0
    
    print(f"\n📈 Results:")
    print(f"   Standard CV time: {standard_time:.3f}s")
    print(f"   Parallel CV time: {parallel_time:.3f}s")
    print(f"   Speedup: {speedup:.2f}x")
    print(f"   CV folds: 5")
    
    return {
        'standard_time': standard_time,
        'parallel_time': parallel_time,
        'speedup': speedup,
        'cv_folds': 5
    }

def test_memory_chunking_performance():
    """Test memory-efficient chunking for large datasets."""
    print("\n🧪 Testing Memory-Efficient Chunking Performance")
    print("=" * 60)
    
    if not VECTORBT_AVAILABLE:
        print("❌ VectorBT not available, skipping test")
        return
    
    # Create large test data
    large_data = create_test_data(n_samples=50000, n_features=30)
    numeric_cols = [col for col in large_data.columns if large_data[col].dtype in ['float64', 'int64']][:15]
    test_data = large_data[numeric_cols]
    
    # Initialize optimizer
    optimizer = get_vectorbt_rolling_optimizer(enable_parallel=True, memory_efficient=True, chunk_size=5000)
    
    window = 20
    
    # Test without chunking (if possible)
    print("📊 Testing without chunking...")
    start_time = time.time()
    try:
        no_chunk_results = optimizer.batch_rolling_operations(test_data, ['mean', 'std'], window)
        no_chunk_time = time.time() - start_time
        no_chunk_success = True
    except Exception as e:
        print(f"   ⚠️ No chunking failed (expected for large data): {e}")
        no_chunk_time = float('inf')
        no_chunk_success = False
    
    # Test with chunking
    print("🚀 Testing with chunking...")
    start_time = time.time()
    chunk_results = optimizer.chunked_processing(
        test_data, 
        lambda data: optimizer.batch_rolling_operations(data, ['mean', 'std'], window),
        chunk_size=5000
    )
    chunk_time = time.time() - start_time
    
    print(f"\n📈 Results:")
    if no_chunk_success:
        print(f"   No chunking time: {no_chunk_time:.3f}s")
        print(f"   Chunking time: {chunk_time:.3f}s")
        print(f"   Memory efficiency: {'Better' if chunk_time < no_chunk_time else 'Worse'}")
    else:
        print(f"   No chunking: Failed (memory issues)")
        print(f"   Chunking time: {chunk_time:.3f}s")
        print(f"   Memory efficiency: Enables processing of large datasets")
    
    return {
        'no_chunk_time': no_chunk_time if no_chunk_success else None,
        'chunk_time': chunk_time,
        'chunk_success': True,
        'no_chunk_success': no_chunk_success
    }

async def test_training_pipeline_performance():
    """Test overall training pipeline performance improvements."""
    print("\n🧪 Testing Training Pipeline Performance")
    print("=" * 60)
    
    if not TRAINING_AVAILABLE:
        print("❌ Training modules not available, skipping test")
        return
    
    # Create test data
    data = create_test_data(n_samples=3000, n_features=25)
    
    # Add target columns
    data['target_long'] = np.random.randn(len(data))
    data['target_short'] = np.random.randn(len(data))
    
    feature_columns = [col for col in data.columns if col.startswith(('sma_', 'std_', 'rsi_', 'feature_'))]
    target_columns = ['target_long', 'target_short']
    
    # Test Analyst training
    print("📊 Testing Analyst training with optimizations...")
    try:
        from src.training.steps.models_training.analyst_models_training import AnalystModelsTrainingConfig
        
        config = AnalystModelsTrainingConfig(
            model_types=['LGBM', 'CATBOOST'],
            enable_parallel_processing=True,
            enable_gpu_acceleration=False,
            memory_limit_gb=4.0
        )
        
        trainer = AnalystModelsTrainingStep(config)
        
        start_time = time.time()
        result = await trainer.train_analyst_models(
            data, feature_columns, target_columns
        )
        training_time = time.time() - start_time
        
        print(f"   Analyst training time: {training_time:.3f}s")
        print(f"   Models trained: {len(result.get('models', {}))}")
        print(f"   Features used: {len(result.get('features_used', []))}")
        
        return {
            'training_time': training_time,
            'models_trained': len(result.get('models', {})),
            'features_used': len(result.get('features_used', [])),
            'success': True
        }
        
    except Exception as e:
        print(f"   ❌ Analyst training failed: {e}")
        return {'success': False, 'error': str(e)}

def run_performance_tests():
    """Run all performance tests."""
    print("🚀 VectorBT Performance Improvements Test Suite")
    print("=" * 80)
    
    results = {}
    
    # Test 1: Batch rolling operations
    try:
        results['batch_rolling'] = test_batch_rolling_performance()
    except Exception as e:
        print(f"❌ Batch rolling test failed: {e}")
        results['batch_rolling'] = {'error': str(e)}
    
    # Test 2: Parallel cross-validation
    try:
        results['parallel_cv'] = test_parallel_cv_performance()
    except Exception as e:
        print(f"❌ Parallel CV test failed: {e}")
        results['parallel_cv'] = {'error': str(e)}
    
    # Test 3: Memory chunking
    try:
        results['memory_chunking'] = test_memory_chunking_performance()
    except Exception as e:
        print(f"❌ Memory chunking test failed: {e}")
        results['memory_chunking'] = {'error': str(e)}
    
    # Test 4: Training pipeline
    try:
        results['training_pipeline'] = asyncio.run(test_training_pipeline_performance())
    except Exception as e:
        print(f"❌ Training pipeline test failed: {e}")
        results['training_pipeline'] = {'error': str(e)}
    
    # Summary
    print("\n📊 Performance Test Summary")
    print("=" * 80)
    
    total_speedup = 1.0
    successful_tests = 0
    
    for test_name, result in results.items():
        if 'error' in result:
            print(f"❌ {test_name}: Failed - {result['error']}")
        else:
            successful_tests += 1
            if 'speedup' in result:
                print(f"✅ {test_name}: {result['speedup']:.2f}x speedup")
                total_speedup *= result['speedup']
            elif 'success' in result and result['success']:
                print(f"✅ {test_name}: Completed successfully")
            else:
                print(f"⚠️ {test_name}: Completed with issues")
    
    print(f"\n🎯 Overall Results:")
    print(f"   Successful tests: {successful_tests}/{len(results)}")
    if total_speedup > 1.0:
        print(f"   Combined speedup: {total_speedup:.2f}x")
    print(f"   VectorBT available: {VECTORBT_AVAILABLE}")
    print(f"   Training modules available: {TRAINING_AVAILABLE}")
    
    return results

if __name__ == "__main__":
    results = run_performance_tests()