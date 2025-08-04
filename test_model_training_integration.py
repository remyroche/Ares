#!/usr/bin/env python3
"""
Test script for Model Training Integration functionality.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Import the training components
from src.training.model_training_integrator import ModelTrainingIntegrator, setup_model_training_integrator
from src.analyst.ml_confidence_predictor import MLConfidencePredictor, setup_ml_confidence_predictor
from src.training.dual_model_system import DualModelSystem, setup_dual_model_system


async def test_model_training_integration():
    """Test model training integration functionality."""
    print("🧪 Testing Model Training Integration...")
    
    # Test configuration
    config = {
        "model_training_integrator": {
            "models_path": "models/",
            "training_data_path": "data/training/",
            "test_size": 0.2,
            "random_state": 42
        },
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
        # Test 1: Initialize Model Training Integrator
        print("\n📋 Test 1: Initializing Model Training Integrator...")
        training_integrator = await setup_model_training_integrator(config)
        
        if training_integrator:
            print("✅ Model Training Integrator initialized successfully")
        else:
            print("❌ Model Training Integrator initialization failed")
            return False
        
        # Test 2: Generate training data
        print("\n📋 Test 2: Generating Training Data...")
        X, y = await training_integrator.generate_training_data(5000)
        
        if not X.empty and not y.empty:
            print(f"✅ Generated training data: {len(X)} samples, {X.shape[1]} features")
            print(f"Target distribution: {y.value_counts().to_dict()}")
        else:
            print("❌ Failed to generate training data")
            return False
        
        # Test 3: Train models
        print("\n📋 Test 3: Training Models...")
        training_results = await training_integrator.train_models(X, y)
        
        if training_results:
            print("✅ Model training completed successfully")
            print(f"Best model: {training_results['best_model']}")
            print(f"Best score: {training_results['best_score']:.4f}")
            
            # Print model scores
            print("\nModel Performance Summary:")
            for model_name, scores in training_results['model_scores'].items():
                print(f"  {model_name}: F1={scores['f1_score']:.4f}, "
                      f"Accuracy={scores['accuracy']:.4f}, "
                      f"CV={scores['cv_mean']:.4f}±{scores['cv_std']:.4f}")
        else:
            print("❌ Model training failed")
            return False
        
        # Test 4: Train ML Confidence Predictor
        print("\n📋 Test 4: Training ML Confidence Predictor...")
        ml_training_success = await training_integrator.train_ml_confidence_predictor()
        
        if ml_training_success:
            print("✅ ML Confidence Predictor training completed")
        else:
            print("❌ ML Confidence Predictor training failed")
            return False
        
        # Test 5: Train ensemble models
        print("\n📋 Test 5: Training Ensemble Models...")
        ensemble_results = await training_integrator.train_ensemble_models()
        
        if ensemble_results:
            print(f"✅ Ensemble training completed: {len(ensemble_results)} timeframes")
            for timeframe, models in ensemble_results.items():
                print(f"  {timeframe}: {len(models)} models trained")
        else:
            print("❌ Ensemble training failed")
            return False
        
        # Test 6: Load trained models
        print("\n📋 Test 6: Loading Trained Models...")
        loaded_models = await training_integrator.load_trained_models()
        
        if loaded_models:
            print(f"✅ Loaded {len(loaded_models)} model groups")
            for model_group in loaded_models.keys():
                print(f"  - {model_group}")
        else:
            print("❌ Failed to load trained models")
            return False
        
        # Test 7: Test ML Confidence Predictor with trained models
        print("\n📋 Test 7: Testing ML Confidence Predictor with Trained Models...")
        ml_predictor = await setup_ml_confidence_predictor(config)
        
        if ml_predictor:
            # Generate test market data
            test_data = generate_test_market_data(100)
            current_price = 100.0
            
            # Test predictions with trained models
            predictions = await ml_predictor.predict_confidence_table(test_data, current_price)
            
            if predictions:
                print("✅ ML Confidence Predictor predictions successful with trained models")
                print(f"Prediction keys: {list(predictions.keys())}")
            else:
                print("❌ ML Confidence Predictor predictions failed")
                return False
        else:
            print("❌ Failed to initialize ML Confidence Predictor")
            return False
        
        # Test 8: Test dual model system with trained models
        print("\n📋 Test 8: Testing Dual Model System with Trained Models...")
        dual_system = await setup_dual_model_system(config)
        
        if dual_system:
            # Test trading decision with trained models
            decision = await dual_system.make_trading_decision(test_data, current_price)
            
            if decision:
                print("✅ Dual model system decision successful with trained models")
                print(f"Decision keys: {list(decision.keys())}")
            else:
                print("❌ Dual model system decision failed")
                return False
        else:
            print("❌ Failed to initialize dual model system")
            return False
        
        # Test 9: Get training statistics
        print("\n📋 Test 9: Getting Training Statistics...")
        training_stats = training_integrator.get_training_stats()
        
        if "error" not in training_stats:
            print("✅ Training statistics retrieved successfully")
            print(f"Models trained: {training_stats['models_trained']}")
            print(f"Best model: {training_stats['best_model']}")
            print(f"Best score: {training_stats['best_score']:.4f}")
            print(f"Total training time: {training_stats['total_training_time']:.2f}s")
        else:
            print("❌ Failed to get training statistics")
            return False
        
        # Test 10: Test cleanup
        print("\n📋 Test 10: Testing Cleanup...")
        
        await training_integrator.stop()
        await ml_predictor.stop()
        await dual_system.stop()
        
        print("✅ Cleanup completed successfully")
        
        print("\n🎉 All model training integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model training integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_test_market_data(size: int) -> pd.DataFrame:
    """Generate test market data for validation."""
    np.random.seed(42)
    
    data = {
        'open': np.random.randn(size) + 100,
        'high': np.random.randn(size) + 105,
        'low': np.random.randn(size) + 95,
        'close': np.random.randn(size) + 100,
        'volume': np.random.randint(1000, 10000, size),
        'price_change': np.random.normal(0, 0.02, size),
        'volume_change': np.random.normal(0, 0.1, size),
        'volatility': np.random.exponential(0.01, size),
        'rsi': np.random.uniform(0, 100, size),
        'macd': np.random.normal(0, 0.01, size),
        'bollinger_position': np.random.uniform(0, 1, size),
        'support_distance': np.random.exponential(0.02, size),
        'resistance_distance': np.random.exponential(0.02, size),
        'trend_strength': np.random.uniform(0, 1, size),
        'momentum': np.random.normal(0, 0.01, size),
        'volume_sma_ratio': np.random.normal(1, 0.2, size),
        'price_sma_ratio': np.random.normal(1, 0.05, size),
        'atr': np.random.exponential(0.01, size),
        'stoch_k': np.random.uniform(0, 100, size),
        'stoch_d': np.random.uniform(0, 100, size),
        'williams_r': np.random.uniform(-100, 0, size),
        'cci': np.random.normal(0, 100, size),
        'adx': np.random.uniform(0, 100, size),
        'obv_change': np.random.normal(0, 1000, size),
        'vwap_deviation': np.random.normal(0, 0.01, size),
        'timestamp': pd.date_range(start='2024-01-01', periods=size, freq='1min')
    }
    
    return pd.DataFrame(data)


async def test_training_performance():
    """Test training performance and scalability."""
    print("\n🧪 Testing Training Performance...")
    
    config = {
        "model_training_integrator": {
            "models_path": "models/",
            "training_data_path": "data/training/",
            "test_size": 0.2,
            "random_state": 42
        }
    }
    
    try:
        # Initialize training integrator
        training_integrator = await setup_model_training_integrator(config)
        
        # Test different data sizes
        data_sizes = [1000, 5000, 10000]
        
        performance_results = {}
        
        for size in data_sizes:
            print(f"\n📋 Testing training performance with {size} samples...")
            
            # Generate data
            start_time = time.time()
            X, y = await training_integrator.generate_training_data(size)
            data_generation_time = time.time() - start_time
            
            if not X.empty and not y.empty:
                # Train models
                start_time = time.time()
                training_results = await training_integrator.train_models(X, y)
                training_time = time.time() - start_time
                
                if training_results:
                    performance_results[size] = {
                        "data_generation_time": data_generation_time,
                        "training_time": training_time,
                        "total_time": data_generation_time + training_time,
                        "models_trained": len(training_results['trained_models']),
                        "best_score": training_results['best_score'],
                        "best_model": training_results['best_model']
                    }
                    
                    print(f"✅ Performance test completed for {size} samples:")
                    print(f"  - Data generation: {data_generation_time:.2f}s")
                    print(f"  - Training: {training_time:.2f}s")
                    print(f"  - Total: {data_generation_time + training_time:.2f}s")
                    print(f"  - Best model: {training_results['best_model']} (F1: {training_results['best_score']:.4f})")
                else:
                    print(f"❌ Training failed for {size} samples")
            else:
                print(f"❌ Data generation failed for {size} samples")
        
        # Print performance summary
        print("\n📊 Training Performance Summary:")
        print("=" * 80)
        print(f"{'Size':<8} {'Data Gen (s)':<15} {'Training (s)':<15} {'Total (s)':<12} {'Best Score':<12}")
        print("=" * 80)
        
        for size, results in performance_results.items():
            print(f"{size:<8} {results['data_generation_time']:<15.2f} "
                  f"{results['training_time']:<15.2f} {results['total_time']:<12.2f} "
                  f"{results['best_score']:<12.4f}")
        
        await training_integrator.stop()
        
        print("\n✅ Training performance tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Training performance test failed: {e}")
        return False


async def test_model_integration():
    """Test full model integration with all components."""
    print("\n🧪 Testing Full Model Integration...")
    
    config = {
        "model_training_integrator": {
            "models_path": "models/",
            "training_data_path": "data/training/",
            "test_size": 0.2,
            "random_state": 42
        },
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
        # Initialize all components
        print("\n📋 Initializing all components...")
        
        training_integrator = await setup_model_training_integrator(config)
        ml_predictor = await setup_ml_confidence_predictor(config)
        dual_system = await setup_dual_model_system(config)
        
        if not all([training_integrator, ml_predictor, dual_system]):
            print("❌ Failed to initialize all components")
            return False
        
        print("✅ All components initialized successfully")
        
        # Train models
        print("\n📋 Training models...")
        X, y = await training_integrator.generate_training_data(3000)
        
        if not X.empty and not y.empty:
            training_results = await training_integrator.train_models(X, y)
            
            if training_results:
                print("✅ Models trained successfully")
            else:
                print("❌ Model training failed")
                return False
        else:
            print("❌ Data generation failed")
            return False
        
        # Test full integration
        print("\n📋 Testing full integration...")
        
        # Generate test data
        test_data = generate_test_market_data(50)
        current_price = 100.0
        
        # Test ML Confidence Predictor
        predictions = await ml_predictor.predict_confidence_table(test_data, current_price)
        
        if predictions:
            print("✅ ML Confidence Predictor integration working")
        else:
            print("❌ ML Confidence Predictor integration failed")
            return False
        
        # Test dual model system
        decision = await dual_system.make_trading_decision(test_data, current_price)
        
        if decision:
            print("✅ Dual model system integration working")
        else:
            print("❌ Dual model system integration failed")
            return False
        
        # Test ensemble predictions
        ensemble_predictions = await ml_predictor.predict_for_dual_model_system(
            test_data, current_price, model_type="analyst"
        )
        
        if ensemble_predictions:
            print("✅ Ensemble predictions integration working")
        else:
            print("❌ Ensemble predictions integration failed")
            return False
        
        # Cleanup
        await training_integrator.stop()
        await ml_predictor.stop()
        await dual_system.stop()
        
        print("✅ Full model integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Full model integration test failed: {e}")
        return False


async def main():
    """Run all model training integration tests."""
    print("🚀 Starting Model Training Integration Tests...")
    
    # Run tests
    test_results = []
    
    # Test 1: Basic functionality
    test_results.append(await test_model_training_integration())
    
    # Test 2: Training performance
    test_results.append(await test_training_performance())
    
    # Test 3: Full integration
    test_results.append(await test_model_integration())
    
    # Summary
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\n📊 Test Summary:")
    print(f"✅ Passed: {passed_tests}/{total_tests}")
    print(f"❌ Failed: {total_tests - passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All model training integration tests passed!")
        print("✅ Model training integration is ready for production use.")
    else:
        print("⚠️ Some model training integration tests failed.")
        print("🔧 Please check the implementation.")


if __name__ == "__main__":
    asyncio.run(main()) 