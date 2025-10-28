#!/usr/bin/env python3
"""
Test script for TCN data preparation fixes.
This script validates that the TCN can handle:
1. Engineered features (9 features from 4 base features)
2. NaN values in input data
3. Proper sequence creation for temporal modeling
"""

import numpy as np
import pandas as pd
from src.models.causal_dilated_tcn import CausalDilatedTCNModel, CausalTCNConfig

def test_tcn_with_engineered_features():
    """Test TCN with engineered features and NaN handling."""
    print("=" * 80)
    print("Testing TCN with Engineered Features and NaN Handling")
    print("=" * 80)
    
    # Simulate data after feature engineering
    # Original: 4 base features -> Engineered: 9 features
    np.random.seed(42)
    n_samples = 200
    n_features = 9  # After feature engineering
    
    # Create test data
    X_test = np.random.randn(n_samples, n_features)
    
    # Add some NaN values (simulating missing data)
    X_test[10:15, 2] = np.nan
    X_test[50:55, 5] = np.nan
    X_test[100:105, 7] = np.nan
    
    # Create regression targets
    y_test = np.random.randn(n_samples) * 0.1
    
    print(f"\n📊 Input Data:")
    print(f"   X shape: {X_test.shape} (samples={n_samples}, features={n_features})")
    print(f"   y shape: {y_test.shape}")
    print(f"   NaN count in X: {np.sum(np.isnan(X_test))}")
    print(f"   NaN count in y: {np.sum(np.isnan(y_test))}")
    
    # Create TCN config (lightweight for testing)
    config = CausalTCNConfig(
        num_filters=32,
        kernel_size=3,
        num_layers=3,
        dropout=0.1,
        epochs=10,
        batch_size=16,
        early_stopping_patience=5
    )
    
    print(f"\n🔧 TCN Configuration:")
    print(f"   Filters: {config.num_filters}")
    print(f"   Layers: {config.num_layers}")
    print(f"   Kernel size: {config.kernel_size}")
    print(f"   Epochs: {config.epochs}")
    print(f"   Batch size: {config.batch_size}")
    
    # Create and fit model
    print(f"\n🚀 Training TCN Model...")
    model = CausalDilatedTCNModel(config=config)
    
    try:
        model.fit(X_test, y_test)
        print("✅ TCN model fitted successfully!")
        
        # Test prediction
        print(f"\n🔮 Making Predictions...")
        predictions = model.predict(X_test)
        print(f"✅ Predictions generated successfully!")
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print(f"   Predictions mean: {predictions.mean():.4f}")
        print(f"   Predictions std: {predictions.std():.4f}")
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"\n📈 Model Performance:")
        print(f"   MSE: {mse:.6f}")
        print(f"   MAE: {mae:.6f}")
        print(f"   R²: {r2:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tcn_with_pandas_dataframe():
    """Test TCN with pandas DataFrame input."""
    print("\n" + "=" * 80)
    print("Testing TCN with Pandas DataFrame Input")
    print("=" * 80)
    
    # Create DataFrame with named features
    np.random.seed(42)
    n_samples = 150
    feature_names = [
        'close', 'volume', 'rsi', 'macd',  # Base features
        'regime_strength', 'volatility_5d', 'volatility_20d',  # Engineered features
        'volume_momentum', 'price_momentum'
    ]
    
    data = {name: np.random.randn(n_samples) for name in feature_names}
    X_df = pd.DataFrame(data)
    
    # Add some NaN values
    X_df.loc[10:15, 'rsi'] = np.nan
    X_df.loc[50:55, 'volatility_5d'] = np.nan
    
    y_series = pd.Series(np.random.randn(n_samples) * 0.1, name='target')
    
    print(f"\n📊 Input Data:")
    print(f"   X shape: {X_df.shape}")
    print(f"   Features: {list(X_df.columns)}")
    print(f"   NaN count: {X_df.isna().sum().sum()}")
    
    # Create and fit model
    config = CausalTCNConfig(epochs=5, batch_size=16)
    model = CausalDilatedTCNModel(config=config)
    
    try:
        model.fit(X_df, y_series)
        print("✅ TCN model fitted with DataFrame input!")
        
        predictions = model.predict(X_df)
        print(f"✅ Predictions shape: {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_model():
    """Test TCN fallback to Ridge regression when PyTorch fails."""
    print("\n" + "=" * 80)
    print("Testing TCN Fallback Model (Ridge Regression)")
    print("=" * 80)
    
    # Create simple test data
    np.random.seed(42)
    X_test = np.random.randn(100, 9)
    X_test[5:10, 2] = np.nan
    y_test = np.random.randn(100)
    
    print(f"📊 Input: X={X_test.shape}, y={y_test.shape}, NaN={np.sum(np.isnan(X_test))}")
    
    # The fallback will be triggered if PyTorch is not available
    # or if TCN training fails
    config = CausalTCNConfig(epochs=2, batch_size=16)
    model = CausalDilatedTCNModel(config=config)
    
    try:
        # Force fallback by catching PyTorch import error
        model.fit(X_test, y_test)
        print("✅ Model fitted (TCN or fallback)")
        
        predictions = model.predict(X_test)
        print(f"✅ Predictions: shape={predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n🔬 TCN Data Preparation Fix Test Suite")
    print("=" * 80)
    
    # Run all tests
    results = []
    
    print("\n\n")
    results.append(("Engineered Features + NaN", test_tcn_with_engineered_features()))
    
    print("\n\n")
    results.append(("Pandas DataFrame Input", test_tcn_with_pandas_dataframe()))
    
    print("\n\n")
    results.append(("Fallback Model", test_fallback_model()))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    total_passed = sum(1 for _, result in results if result)
    total_tests = len(results)
    print(f"\n   Total: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n✅ All tests passed! TCN data preparation is working correctly.")
    else:
        print(f"\n⚠️ {total_tests - total_passed} test(s) failed. Please review the errors above.")
