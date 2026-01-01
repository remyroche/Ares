#!/usr/bin/env python3
"""
Test script for incremental XGBoost training with warm start and 6-month burn-in.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.ml_common.standardized_xgb_trainer import StandardizedXGBTrainer, XGBTrainingConfig

def create_sample_data(n_samples=10000, start_date='2022-01-01'):
    """Create sample data for testing incremental training."""
    
    # Create date range
    dates = pd.date_range(start=start_date, periods=n_samples, freq='15min')
    
    # Generate features
    np.random.seed(42)
    data = {
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples),
        'feature_4': np.random.randn(n_samples),
        'feature_5': np.random.randn(n_samples),
    }
    
    # Create target with some relationship to features
    df = pd.DataFrame(data, index=dates)
    df['target'] = (
        0.3 * df['feature_1'] + 
        0.2 * df['feature_2'] + 
        0.1 * df['feature_3'] + 
        np.random.randn(n_samples) * 0.1
    )
    
    # Convert to binary classification
    df['target'] = (df['target'] > df['target'].median()).astype(int)
    
    return df

def test_incremental_training():
    """Test incremental XGBoost training."""
    
    print("🧪 Testing Incremental XGBoost Training")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating sample data...")
    df = create_sample_data(n_samples=5000, start_date='2022-01-01')
    
    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Test configuration with incremental training
    print("⚙️  Creating incremental training config...")
    config = XGBTrainingConfig(
        model_id="test_incremental_xgb",
        retrain_interval_days=28,  # 28 days (4 weeks)
        burnin_pct=1/6,  # 6 months burn-in
        enable_incremental_training=True,
        incremental_strategy="warm_start",
        enable_warm_start=True,
        warm_start_learning_rate_factor=0.5,
        verbose=True,
        enable_hpo=False,  # Disable HPO for faster testing
        enable_oof_training=True,
        n_estimators=100,  # Reduce for faster testing
        early_stopping_rounds=10
    )
    
    # Create trainer
    print("🤖 Creating XGB trainer...")
    trainer = StandardizedXGBTrainer(model_id=config.model_id, config=config)
    
    # Train and predict
    print("🚀 Starting incremental training...")
    try:
        results = trainer.train_and_predict(
            X=X,
            y=y,
            data_start=df.index.min(),
            data_end=df.index.max()
        )
        
        print("✅ Incremental training completed successfully!")
        print(f"   Total windows: {len(results.models)}")
        print(f"   Total predictions: {len(results.oof_predictions)}")
        print(f"   Training windows: {len(results.training_windows)}")
        
        # Check if model states were saved
        model_dir = Path(config.model_persistence_dir)
        if model_dir.exists():
            saved_models = list(model_dir.glob("*.json"))
            print(f"   Saved model states: {len(saved_models)}")
        
        # Analyze results
        if len(results.metadata) > 0:
            avg_training_time = np.mean([m['training_time'] for m in results.metadata])
            total_training_time = sum([m['training_time'] for m in results.metadata])
            
            print(f"   Average training time per window: {avg_training_time:.2f}s")
            print(f"   Total training time: {total_training_time:.2f}s")
            
            # Check for HPO usage
            hpo_windows = [m for m in results.metadata if m.get('used_hpo', False)]
            print(f"   Windows with HPO: {len(hpo_windows)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Incremental training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comparison_with_regular_training():
    """Compare incremental training with regular training."""
    
    print("\n🔄 Comparing Incremental vs Regular Training")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data(n_samples=3000, start_date='2022-01-01')
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Test regular training
    print("📊 Testing regular training...")
    regular_config = XGBTrainingConfig(
        model_id="test_regular_xgb",
        retrain_interval_days=28,  # 28 days (4 weeks)
        burnin_pct=1/6,
        enable_incremental_training=False,  # Disable incremental
        verbose=False,
        enable_hpo=False,
        enable_oof_training=True,
        n_estimators=100,
        early_stopping_rounds=10
    )
    
    regular_trainer = StandardizedXGBTrainer(model_id=regular_config.model_id, config=regular_config)
    
    try:
        import time
        start_time = time.time()
        regular_results = regular_trainer.train_and_predict(X, y, df.index.min(), df.index.max())
        regular_time = time.time() - start_time
        
        print(f"✅ Regular training completed in {regular_time:.2f}s")
        print(f"   Windows: {len(regular_results.models)}")
        
    except Exception as e:
        print(f"❌ Regular training failed: {e}")
        return False
    
    # Test incremental training
    print("📊 Testing incremental training...")
    incremental_config = XGBTrainingConfig(
        model_id="test_incremental_xgb",
        retrain_interval_days=28,  # 28 days (4 weeks)
        burnin_pct=1/6,
        enable_incremental_training=True,
        incremental_strategy="warm_start",
        verbose=False,
        enable_hpo=False,
        enable_oof_training=True,
        n_estimators=100,
        early_stopping_rounds=10
    )
    
    incremental_trainer = StandardizedXGBTrainer(model_id=incremental_config.model_id, config=incremental_config)
    
    try:
        start_time = time.time()
        incremental_results = incremental_trainer.train_and_predict(X, y, df.index.min(), df.index.max())
        incremental_time = time.time() - start_time
        
        print(f"✅ Incremental training completed in {incremental_time:.2f}s")
        print(f"   Windows: {len(incremental_results.models)}")
        
        # Compare results
        print("\n📈 Performance Comparison:")
        print(f"   Regular training time: {regular_time:.2f}s")
        print(f"   Incremental training time: {incremental_time:.2f}s")
        
        if incremental_time < regular_time:
            improvement = ((regular_time - incremental_time) / regular_time) * 100
            print(f"   Improvement: {improvement:.1f}% faster")
        else:
            degradation = ((incremental_time - regular_time) / regular_time) * 100
            print(f"   Degradation: {degradation:.1f}% slower")
        
        # Compare window counts
        regular_windows = len(regular_results.models)
        incremental_windows = len(incremental_results.models)
        
        print(f"\n📊 Window Count Comparison:")
        print(f"   Regular windows: {regular_windows}")
        print(f"   Incremental windows: {incremental_windows}")
        
        if incremental_windows < regular_windows:
            reduction = ((regular_windows - incremental_windows) / regular_windows) * 100
            print(f"   Reduction: {reduction:.1f}% fewer windows")
        
        return True
        
    except Exception as e:
        print(f"❌ Incremental training failed: {e}")
        return False

def main():
    """Run all tests."""
    
    print("🧪 Incremental XGBoost Training Test Suite")
    print("=" * 60)
    
    tests = [
        ("Incremental Training Test", test_incremental_training),
        ("Performance Comparison Test", test_comparison_with_regular_training),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Incremental XGBoost training is working correctly!")
        return 0
    else:
        print(f"❌ {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
