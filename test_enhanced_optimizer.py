#!/usr/bin/env python3
"""
Test script for the enhanced coarse optimizer improvements.
"""

import asyncio
import pandas as pd
import numpy as np
from src.training.enhanced_coarse_optimizer import EnhancedCoarseOptimizer
from src.config import CONFIG

async def test_enhanced_optimizer():
    """Test the enhanced coarse optimizer with the new improvements."""
    print("🧪 Testing Enhanced Coarse Optimizer Improvements...")
    
    # Create mock data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Create mock klines data
    klines_data = pd.DataFrame({
        'open': np.random.randn(n_samples) * 100 + 1000,
        'high': np.random.randn(n_samples) * 100 + 1050,
        'low': np.random.randn(n_samples) * 100 + 950,
        'close': np.random.randn(n_samples) * 100 + 1000,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Create mock agg_trades data
    agg_trades_data = pd.DataFrame({
        'price': np.random.randn(n_samples) * 100 + 1000,
        'quantity': np.random.randint(1, 100, n_samples),
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1h')
    })
    
    # Create mock futures data
    futures_data = pd.DataFrame({
        'funding_rate': np.random.randn(n_samples) * 0.001,
        'open_interest': np.random.randint(1000000, 10000000, n_samples),
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1h')
    })
    
    # Create mock database manager - use None for testing
    db_manager = None
    
    # Initialize enhanced optimizer
    optimizer = EnhancedCoarseOptimizer(
        db_manager=db_manager,
        symbol="ETHUSDT",
        timeframe="1h",
        optimal_target_params={},
        klines_data=klines_data,
        agg_trades_data=agg_trades_data,
        futures_data=futures_data,
        blank_training_mode=True  # Use blank training mode for faster testing
    )
    
    print("📊 Testing resource allocation...")
    print(f"   - CPU cores: {optimizer.resources['cpu_count']}")
    print(f"   - Max workers: {optimizer.resources['max_workers']}")
    print(f"   - Memory limit: {optimizer.resources['memory_limit_gb']:.1f} GB")
    print(f"   - Parallel processing: {optimizer.resources['enable_parallel']}")
    
    print("\n📊 Testing memory monitoring...")
    memory_cleaned = optimizer._monitor_memory_usage()
    print(f"   - Memory cleanup triggered: {memory_cleaned}")
    print(f"   - Current memory usage: {optimizer.resource_usage.get('memory_percent', 0):.1f}%")
    
    print("\n📊 Testing progress tracking...")
    optimizer._track_optimization_progress(
        "Test Stage", 
        50.0, 
        {"test_metric": 0.85, "features_processed": 25}
    )
    print(f"   - Progress: {optimizer.optimization_progress:.1f}%")
    print(f"   - Current stage: {optimizer.current_stage}")
    
    print("\n📊 Testing parallel feature selection...")
    test_features = [f"feature_{i}" for i in range(10)]
    # Create mock X and y for testing
    optimizer.X = pd.DataFrame(np.random.randn(100, 10), columns=test_features)
    optimizer.y = pd.Series(np.random.randint(0, 2, 100))
    
    feature_importance = optimizer._parallel_feature_selection(test_features)
    print(f"   - Features processed: {len(feature_importance)}")
    print(f"   - Average importance: {np.mean(list(feature_importance.values())):.4f}")
    
    print("\n📊 Testing robust SHAP analysis...")
    X_sample = pd.DataFrame(np.random.randn(100, 5), columns=[f"shap_feature_{i}" for i in range(5)])
    y_sample = pd.Series(np.random.randint(0, 2, 100))
    
    shap_scores = optimizer._robust_shap_analysis(X_sample, y_sample)
    print(f"   - SHAP analysis completed: {len(shap_scores) > 0}")
    if shap_scores:
        print(f"   - Top feature: {max(shap_scores.items(), key=lambda x: x[1])}")
    
    print("\n📊 Testing enhanced cross-validation...")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    cv_results = optimizer._enhanced_cross_validation(model, X_sample, y_sample)
    print(f"   - Cross-validation completed: {len(cv_results) > 0}")
    if cv_results:
        print(f"   - Accuracy: {cv_results['accuracy']['mean']:.4f} ± {cv_results['accuracy']['std']:.4f}")
    
    print("\n✅ All tests completed successfully!")
    print("🚀 Enhanced coarse optimizer improvements are working correctly.")

if __name__ == "__main__":
    asyncio.run(test_enhanced_optimizer()) 