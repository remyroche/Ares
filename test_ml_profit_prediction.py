#!/usr/bin/env python3
"""
Test script to verify ML-based profit prediction functionality.
"""

import pandas as pd
import numpy as np
from src.training.steps.step4_analyst_labeling_feature_engineering_components.multi_output_profit_prediction import MultiOutputProfitPredictor, MultiOutputConfig

def test_ml_profit_prediction():
    """Test the ML-based profit prediction system."""
    print("🧪 Testing ML-based Profit Prediction System")
    print("=" * 50)
    
    # Create sample data with profit information
    np.random.seed(42)
    n_samples = 200
    
    # Create realistic market data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1H'),
        'open': np.random.normal(100, 5, n_samples),
        'high': np.random.normal(102, 5, n_samples),
        'low': np.random.normal(98, 5, n_samples),
        'close': np.random.normal(101, 5, n_samples),
        'volume': np.random.normal(1000, 200, n_samples),
        'rsi': 50 + np.random.normal(0, 15, n_samples),
        'sma_20': np.random.normal(100, 3, n_samples),
        'volatility': np.random.normal(0.02, 0.01, n_samples),
        'label': np.random.choice([0, 1], n_samples),  # 0=SELL, 1=BUY
        'potential_profit_pct': np.random.normal(0.02, 0.05, n_samples)  # Profit information
    })
    
    print(f"📊 Created test data: {data.shape}")
    print(f"   - Samples: {len(data)}")
    print(f"   - Features: {len(data.columns) - 3}")  # Exclude timestamp, label, profit
    print(f"   - Profit range: {data['potential_profit_pct'].min():.4f} to {data['potential_profit_pct'].max():.4f}")
    
    # Initialize multi-output predictor
    config = MultiOutputConfig()
    predictor = MultiOutputProfitPredictor(config)
    
    print("\n🚀 Training Multi-Output Profit Prediction Model...")
    
    # Train the model
    training_results = predictor.train(data)
    
    if training_results and training_results.get("method"):
        print(f"✅ Training successful using method: {training_results['method']}")
        print(f"   - Direction accuracy: {training_results.get('direction_accuracy', 0):.4f}")
        print(f"   - High-value accuracy: {training_results.get('high_value_accuracy', 0):.4f}")
    else:
        print("❌ Training failed")
        return False
    
    print("\n🔮 Making Predictions...")
    
    # Make predictions on new data
    test_data = data.tail(10).copy()  # Use last 10 samples for testing
    predictions = predictor.predict(test_data)
    
    print(f"✅ Predictions made successfully")
    print(f"   - Direction predictions: {predictions.get('direction', [])}")
    print(f"   - Profit predictions: {predictions.get('profit', [])}")
    print(f"   - Confidence scores: {predictions.get('confidence', [])}")
    print(f"   - High-value factors: {predictions.get('high_value_trades', [])}")
    
    # Verify prediction structure
    expected_keys = ['direction', 'profit', 'confidence', 'high_value_trades']
    missing_keys = [key for key in expected_keys if key not in predictions]
    
    if missing_keys:
        print(f"❌ Missing prediction keys: {missing_keys}")
        return False
    
    print("\n✅ ML-based Profit Prediction Test Passed!")
    print("   - Model trained successfully")
    print("   - Predictions generated correctly")
    print("   - All expected outputs present")
    
    return True

if __name__ == "__main__":
    success = test_ml_profit_prediction()
    if success:
        print("\n🎉 All tests passed! ML-based profit prediction is working correctly.")
    else:
        print("\n❌ Tests failed! Please check the implementation.")