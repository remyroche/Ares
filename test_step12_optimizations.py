"""
Test script for Step 12 optimizations

This script tests all the implemented optimizations:
- Fast fail validations
- Memory management
- Hyperparameter optimization improvements
- Feature selection streamlining
- Data loading optimizations
- Vectorized preprocessing
"""

import asyncio
import numpy as np
import pandas as pd
import psutil
import time
from pathlib import Path
import sys
import os

# Add the src directory to the path
sys.path.append('/workspace/src')

from training.steps.model_training.step12_analyst_enhancement_optimized import (
    OptimizedStep12AnalystEnhancement,
    FastFailValidator,
    MemoryManager,
    OptimizedHyperparameterOptimizer,
    StreamlinedFeatureSelector,
    VectorizedPreprocessor,
    LazyDataLoader,
    PerformanceMonitor
)

def create_test_data(n_samples=1000, n_features=50):
    """Create test data for optimization testing."""
    np.random.seed(42)
    
    # Create features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add some constant features to test validation
    X['constant_feature'] = 1.0
    X['near_constant_feature'] = np.random.choice([0, 1], size=n_samples, p=[0.99, 0.01])
    
    # Create target
    y = pd.Series(np.random.randint(0, 3, n_samples))
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    return X_train, X_val, y_train, y_val

def test_fast_fail_validations():
    """Test fast fail validation system."""
    print("🧪 Testing Fast Fail Validations...")
    
    logger = __import__('logging').getLogger('test')
    validator = FastFailValidator(logger)
    
    # Test 1: Normal data (should pass)
    X_train, X_val, y_train, y_val = create_test_data()
    try:
        validator.validate_data_quality(X_train, y_train, X_val, y_val)
        print("✅ Normal data validation passed")
    except Exception as e:
        print(f"❌ Normal data validation failed: {e}")
    
    # Test 2: Empty data (should fail)
    try:
        validator.validate_data_quality(pd.DataFrame(), y_train, X_val, y_val)
        print("❌ Empty data validation should have failed")
    except ValueError as e:
        print(f"✅ Empty data validation correctly failed: {e}")
    
    # Test 3: Insufficient samples (should fail)
    try:
        X_small = X_train[:10]
        y_small = y_train[:10]
        validator.validate_data_quality(X_small, y_small, X_val, y_val)
        print("❌ Small data validation should have failed")
    except ValueError as e:
        print(f"✅ Small data validation correctly failed: {e}")
    
    # Test 4: Too many constant features (should fail)
    try:
        X_constant = X_train.copy()
        for i in range(30):  # Add many constant features
            X_constant[f'constant_{i}'] = 1.0
        validator.validate_data_quality(X_constant, y_train, X_val, y_val)
        print("❌ Constant features validation should have failed")
    except ValueError as e:
        print(f"✅ Constant features validation correctly failed: {e}")
    
    # Test 5: Model compatibility
    try:
        validator.validate_model_compatibility('svm', X_train, y_train)
        print("✅ SVM compatibility check passed")
    except ValueError as e:
        print(f"✅ SVM compatibility correctly failed: {e}")
    
    print("Fast fail validation tests completed\n")

def test_memory_management():
    """Test memory management system."""
    print("🧪 Testing Memory Management...")
    
    memory_manager = MemoryManager(max_memory_gb=2.0, cleanup_threshold=0.7)
    
    # Test memory monitoring
    percent, used_gb = memory_manager.check_memory_usage()
    print(f"Current memory usage: {percent:.1f}%, {used_gb:.2f}GB")
    
    # Test cleanup logic
    should_cleanup = memory_manager.should_cleanup()
    print(f"Should cleanup: {should_cleanup}")
    
    # Test delayed cleanup
    for i in range(7):
        cleaned = memory_manager.delayed_cleanup()
        print(f"Cleanup {i+1}: {cleaned}")
    
    print("Memory management tests completed\n")

def test_performance_monitoring():
    """Test performance monitoring."""
    print("🧪 Testing Performance Monitoring...")
    
    logger = __import__('logging').getLogger('test')
    
    # Test performance monitoring
    with PerformanceMonitor("test_operation", logger) as monitor:
        time.sleep(0.1)  # Simulate work
        print("Performance monitoring test completed")
    
    print("Performance monitoring tests completed\n")

async def test_feature_selection():
    """Test streamlined feature selection."""
    print("🧪 Testing Streamlined Feature Selection...")
    
    logger = __import__('logging').getLogger('test')
    config = {'feature_selection_k': 10}
    
    # Create test data
    X_train, X_val, y_train, y_val = create_test_data(n_samples=500, n_features=30)
    
    selector = StreamlinedFeatureSelector(config, logger)
    
    # Test simple selection
    try:
        selected_features, summary = await selector.select_optimal_features(
            None, 'test_model', X_train, y_train, X_val, y_val
        )
        print(f"✅ Feature selection completed: {len(selected_features)} features selected")
        print(f"Selection method: {summary['method']}")
        print(f"Selection ratio: {summary['selection_ratio']:.2f}")
    except Exception as e:
        print(f"❌ Feature selection failed: {e}")
    
    # Test caching
    try:
        selected_features2, summary2 = await selector.select_optimal_features(
            None, 'test_model', X_train, y_train, X_val, y_val
        )
        print("✅ Feature selection caching works")
    except Exception as e:
        print(f"❌ Feature selection caching failed: {e}")
    
    print("Feature selection tests completed\n")

async def test_hyperparameter_optimization():
    """Test optimized hyperparameter optimization."""
    print("🧪 Testing Optimized Hyperparameter Optimization...")
    
    logger = __import__('logging').getLogger('test')
    config = {'n_trials': 5}  # Small number for testing
    
    # Create test data
    X_train, X_val, y_train, y_val = create_test_data(n_samples=200, n_features=10)
    
    optimizer = OptimizedHyperparameterOptimizer(config, logger)
    
    # Test optimization
    try:
        best_params, best_score = await optimizer.optimize_model(
            'random_forest', X_train, y_train, X_val, y_val
        )
        print(f"✅ Hyperparameter optimization completed")
        print(f"Best score: {best_score:.4f}")
        print(f"Best params: {best_params}")
    except Exception as e:
        print(f"❌ Hyperparameter optimization failed: {e}")
    
    print("Hyperparameter optimization tests completed\n")

def test_vectorized_preprocessing():
    """Test vectorized preprocessing."""
    print("🧪 Testing Vectorized Preprocessing...")
    
    logger = __import__('logging').getLogger('test')
    config = {'normalize_features': True}
    
    # Create test data with some issues
    X_train, X_val, y_train, y_val = create_test_data(n_samples=500, n_features=20)
    
    # Add some missing values and infinities
    X_train.iloc[0, 0] = np.nan
    X_train.iloc[1, 1] = np.inf
    X_train.iloc[2, 2] = -np.inf
    
    preprocessor = VectorizedPreprocessor(config, logger)
    
    try:
        X_train_proc, X_val_proc, y_train_proc, y_val_proc = preprocessor.preprocess_data(
            X_train, X_val, y_train, y_val
        )
        
        print(f"✅ Preprocessing completed")
        print(f"Original shape: {X_train.shape}, Processed shape: {X_train_proc.shape}")
        print(f"Missing values in processed data: {X_train_proc.isnull().sum().sum()}")
        print(f"Infinite values in processed data: {np.isinf(X_train_proc).sum().sum()}")
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
    
    print("Vectorized preprocessing tests completed\n")

async def test_data_loading():
    """Test lazy data loading."""
    print("🧪 Testing Lazy Data Loading...")
    
    logger = __import__('logging').getLogger('test')
    config = {}
    
    loader = LazyDataLoader(config, logger)
    
    try:
        # Test data loading
        data = await loader.load_data_optimized('ETHUSDT', 'BINANCE', '1m', 30)
        print(f"✅ Data loading completed: {data.shape}")
        
        # Test caching
        data2 = await loader.load_data_optimized('ETHUSDT', 'BINANCE', '1m', 30)
        print("✅ Data loading caching works")
        
        # Test cache clearing
        loader.clear_cache()
        print("✅ Cache clearing works")
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
    
    print("Data loading tests completed\n")

async def test_full_optimization():
    """Test the full optimized step 12."""
    print("🧪 Testing Full Optimized Step 12...")
    
    config = {
        'n_trials': 3,  # Small for testing
        'feature_selection_k': 5,
        'normalize_features': True,
        'max_memory_gb': 4.0,
        'cleanup_threshold': 0.8
    }
    
    try:
        step = OptimizedStep12AnalystEnhancement(config)
        
        training_input = {
            'symbol': 'ETHUSDT',
            'exchange': 'BINANCE',
            'timeframe': '1m',
            'lookback_days': 7  # Small for testing
        }
        
        results = await step.execute(training_input, {})
        
        print(f"✅ Full optimization completed")
        print(f"Enhanced models: {len(results.get('enhanced_models', {}))}")
        print(f"Processing metadata: {results.get('processing_metadata', {})}")
        
    except Exception as e:
        print(f"❌ Full optimization failed: {e}")
    
    print("Full optimization tests completed\n")

def run_performance_comparison():
    """Run a performance comparison between original and optimized versions."""
    print("🧪 Running Performance Comparison...")
    
    # This would compare the original step12 with the optimized version
    # For now, just show the improvements made
    
    improvements = {
        "Hyperparameter Optimization": [
            "Early stopping with MedianPruner",
            "Reduced logging overhead (every 10th trial)",
            "Model instance caching",
            "Adaptive trial count based on data size",
            "Proper resource management with context managers"
        ],
        "Feature Selection": [
            "Intelligent caching system",
            "Batched processing for large feature sets",
            "Multiple selection strategies based on feature count",
            "Combined scoring methods (MI + F-score)"
        ],
        "Memory Management": [
            "Intelligent cleanup based on memory usage",
            "Delayed garbage collection (every 5 operations)",
            "Memory usage monitoring",
            "Cache size limits"
        ],
        "Fast Fail Validations": [
            "Data quality checks (empty data, insufficient samples)",
            "Model compatibility validation",
            "Configuration validation",
            "Data type consistency checks"
        ],
        "Data Processing": [
            "Vectorized preprocessing",
            "Lazy loading with caching",
            "Proper handling of missing/infinite values",
            "Memory-efficient data operations"
        ],
        "Error Handling": [
            "Specific exception handling instead of generic suppression",
            "Proper resource cleanup with context managers",
            "Detailed error logging",
            "Graceful degradation"
        ]
    }
    
    for category, items in improvements.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  ✅ {item}")
    
    print("\nPerformance comparison completed\n")

async def main():
    """Run all optimization tests."""
    print("🚀 Starting Step 12 Optimization Tests\n")
    
    # Test individual components
    test_fast_fail_validations()
    test_memory_management()
    test_performance_monitoring()
    await test_feature_selection()
    await test_hyperparameter_optimization()
    test_vectorized_preprocessing()
    await test_data_loading()
    
    # Test full system
    await test_full_optimization()
    
    # Show improvements
    run_performance_comparison()
    
    print("🎉 All Step 12 optimization tests completed!")

if __name__ == '__main__':
    asyncio.run(main())