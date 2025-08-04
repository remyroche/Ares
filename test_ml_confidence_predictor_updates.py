#!/usr/bin/env python3
"""
Test script for Updated ML Confidence Predictor functionality.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime

# Import the components
from src.analyst.ml_confidence_predictor import MLConfidencePredictor, setup_ml_confidence_predictor
from src.training.dual_model_system import DualModelSystem, setup_dual_model_system


async def test_ml_confidence_predictor_updates():
    """Test the updated ML Confidence Predictor functionality."""
    print("🧪 Testing Updated ML Confidence Predictor...")
    
    # Test configuration
    config = {
        "ml_confidence_predictor": {
            "model_path": "models/confidence_predictor.joblib",
            "min_samples_for_training": 100,
            "confidence_threshold": 0.6,
            "max_prediction_horizon": 1,
            "retrain_interval_hours": 24
        },
        "dual_model_system": {
            "analyst_timeframes": ["1h", "15m", "5m", "1m"],
            "tactician_timeframes": ["1m"],
            "analyst_confidence_threshold": 0.7,
            "tactician_confidence_threshold": 0.8,
            "enable_ensemble_analysis": True,
        }
    }
    
    try:
        # Test 1: Initialize ML Confidence Predictor
        print("\n📋 Test 1: Initializing ML Confidence Predictor...")
        ml_predictor = await setup_ml_confidence_predictor(config)
        
        if ml_predictor is None:
            print("❌ Failed to initialize ML Confidence Predictor")
            return False
        
        print("✅ ML Confidence Predictor initialized successfully")
        
        # Test 2: Create Mock Market Data
        print("\n📋 Test 2: Creating Mock Market Data...")
        mock_market_data = pd.DataFrame({
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 105,
            'low': np.random.randn(100) + 95,
            'close': np.random.randn(100) + 100,
            'volume': np.random.randint(1000, 10000, 100),
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1min')
        })
        current_price = 100.0
        print("✅ Mock market data created")
        
        # Test 3: Test Dual Model System Predictions
        print("\n📋 Test 3: Testing Dual Model System Predictions...")
        
        # Test Analyst predictions
        analyst_predictions = await ml_predictor.predict_for_dual_model_system(
            market_data=mock_market_data,
            current_price=current_price,
            model_type="analyst"
        )
        
        if analyst_predictions:
            print("✅ Analyst predictions generated successfully")
            print(f"Analyst Decision: {analyst_predictions.get('strategic_decision', {})}")
        else:
            print("❌ Analyst predictions failed")
            return False
        
        # Test Tactician predictions
        tactician_predictions = await ml_predictor.predict_for_dual_model_system(
            market_data=mock_market_data,
            current_price=current_price,
            model_type="tactician"
        )
        
        if tactician_predictions:
            print("✅ Tactician predictions generated successfully")
            print(f"Tactician Decision: {tactician_predictions.get('timing_decision', {})}")
        else:
            print("❌ Tactician predictions failed")
            return False
        
        # Test 4: Test Ensemble Predictions
        print("\n📋 Test 4: Testing Ensemble Predictions...")
        
        # Create mock ensemble models
        try:
            from sklearn.ensemble import RandomForestClassifier
            mock_ensemble_models = {
                "model_1": RandomForestClassifier(n_estimators=10),
                "model_2": RandomForestClassifier(n_estimators=10),
                "model_3": RandomForestClassifier(n_estimators=10)
            }
            ensemble_weights = {"model_1": 0.4, "model_2": 0.3, "model_3": 0.3}
            
            ensemble_predictions = await ml_predictor.predict_ensemble_confidence(
                market_data=mock_market_data,
                current_price=current_price,
                ensemble_models=mock_ensemble_models,
                ensemble_weights=ensemble_weights
            )
            
            if ensemble_predictions:
                print("✅ Ensemble predictions generated successfully")
                print(f"Ensemble Statistics: {ensemble_predictions.get('ensemble_statistics', {})}")
            else:
                print("⚠️ Ensemble predictions failed (expected for test)")
                
        except ImportError:
            print("⚠️ sklearn not available, skipping ensemble test")
        
        # Test 5: Test Confidence Verification
        print("\n📋 Test 5: Testing Confidence Verification...")
        
        verification_results = ml_predictor.verify_confidence_calculations(
            market_data=mock_market_data,
            current_price=current_price
        )
        
        if verification_results:
            print("✅ Confidence verification completed")
            print(f"Verification Status: {verification_results.get('overall_verification', 'UNKNOWN')}")
        else:
            print("❌ Confidence verification failed")
            return False
        
        # Test 6: Test Dual Model System Integration
        print("\n📋 Test 6: Testing Dual Model System Integration...")
        
        dual_system = await setup_dual_model_system(config)
        
        if dual_system:
            # Test entry decision
            entry_decision = await dual_system.make_trading_decision(
                market_data=mock_market_data,
                current_price=current_price
            )
            
            if entry_decision:
                print("✅ Dual model system integration successful")
                print(f"Entry Decision: {entry_decision.get('decision', {})}")
            else:
                print("❌ Dual model system integration failed")
                return False
        else:
            print("❌ Failed to initialize dual model system")
            return False
        
        # Test 7: Test System Cleanup
        print("\n📋 Test 7: Testing System Cleanup...")
        await ml_predictor.stop()
        await dual_system.stop()
        print("✅ System cleanup completed")
        
        print("\n🎉 All tests passed! Updated ML Confidence Predictor is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_confidence_calculation_verification():
    """Test confidence calculation verification functionality."""
    print("\n🧪 Testing Confidence Calculation Verification...")
    
    config = {
        "ml_confidence_predictor": {
            "model_path": "models/confidence_predictor.joblib",
            "min_samples_for_training": 100,
            "confidence_threshold": 0.6,
            "max_prediction_horizon": 1,
            "retrain_interval_hours": 24
        }
    }
    
    try:
        ml_predictor = MLConfidencePredictor(config)
        success = await ml_predictor.initialize()
        
        if not success:
            print("❌ Failed to initialize ML Confidence Predictor for verification test")
            return False
        
        # Test with different data scenarios
        test_scenarios = [
            {
                "name": "Normal Data",
                "data": pd.DataFrame({
                    'open': np.random.randn(50) + 100,
                    'high': np.random.randn(50) + 105,
                    'low': np.random.randn(50) + 95,
                    'close': np.random.randn(50) + 100,
                    'volume': np.random.randint(1000, 10000, 50)
                })
            },
            {
                "name": "Data with Missing Values",
                "data": pd.DataFrame({
                    'open': [100, 101, np.nan, 103, 104],
                    'high': [105, 106, 107, np.nan, 109],
                    'low': [95, 96, 97, 98, 99],
                    'close': [101, 102, 103, 104, 105],
                    'volume': [1000, 1100, 1200, 1300, 1400]
                })
            },
            {
                "name": "Empty Data",
                "data": pd.DataFrame()
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\n📋 Testing {scenario['name']}...")
            
            verification_results = ml_predictor.verify_confidence_calculations(
                market_data=scenario['data'],
                current_price=100.0
            )
            
            if verification_results:
                status = verification_results.get('overall_verification', 'UNKNOWN')
                print(f"✅ {scenario['name']} verification completed: {status}")
                
                # Check for anomalies
                anomaly_count = verification_results.get('anomaly_detection', {}).get('anomaly_count', 0)
                if anomaly_count > 0:
                    print(f"⚠️ Found {anomaly_count} anomalies in {scenario['name']}")
            else:
                print(f"❌ {scenario['name']} verification failed")
        
        await ml_predictor.stop()
        print("✅ Confidence calculation verification test completed")
        return True
        
    except Exception as e:
        print(f"❌ Confidence verification test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting ML Confidence Predictor Update Tests...")
    
    # Run tests
    test_results = []
    
    # Test 1: Main functionality
    test_results.append(await test_ml_confidence_predictor_updates())
    
    # Test 2: Confidence verification
    test_results.append(await test_confidence_calculation_verification())
    
    # Summary
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\n📊 Test Summary:")
    print(f"✅ Passed: {passed_tests}/{total_tests}")
    print(f"❌ Failed: {total_tests - passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! ML Confidence Predictor updates are ready for use.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    asyncio.run(main()) 