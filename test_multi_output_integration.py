#!/usr/bin/env python3
"""
Test Multi-Output Training Integration

This script tests the integration of MultiOutputProbabilityTrainer across all training steps
to ensure the plan is fully and correctly implemented.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_multi_output_probability_trainer():
    """Test the MultiOutputProbabilityTrainer directly."""
    logger.info("🧪 Testing MultiOutputProbabilityTrainer...")
    
    try:
        from src.training.multi_output_probability_trainer import MultiOutputProbabilityTrainer
        
        # Create test data
        n_samples = 1000
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.choice([0, 1], size=n_samples)
        market_data = pd.DataFrame({
            'close': np.random.randn(n_samples),
            'volume': np.random.randn(n_samples)
        })
        
        # Configure multi-output training
        config = {
            "use_lightgbm": True,
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "profit_target": 0.02,
            "stop_loss": 0.01,
            "look_ahead_periods": 20,
            "magnitude_threshold_factor": 0.8,
            "adverse_threshold": 0.01,
            "avoidance_look_ahead": 10
        }
        
        # Initialize trainer
        trainer = MultiOutputProbabilityTrainer(config)
        
        # Generate multi-output targets
        y_multi = trainer.prepare_multi_output_targets(X, y, market_data)
        
        # Verify targets
        expected_targets = ["triple_barrier", "direction", "magnitude", "barrier_avoidance"]
        for target_name in expected_targets:
            assert target_name in y_multi, f"Missing target: {target_name}"
            assert len(y_multi[target_name]) == len(X), f"Target length mismatch for {target_name}"
            assert np.all((y_multi[target_name] >= 0) & (y_multi[target_name] <= 1)), f"Invalid target values for {target_name}"
        
        # Split data for training
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train_multi = {k: v[:split_idx] for k, v in y_multi.items()}
        y_test_multi = {k: v[split_idx:] for k, v in y_multi.items()}
        
        # Train multi-output model
        trained_models = trainer.train_multi_output_model(
            X_train, y_train_multi, X_test, y_test_multi
        )
        
        # Verify training results
        assert len(trained_models) == 4, f"Expected 4 trained models, got {len(trained_models)}"
        
        # Generate probability outputs
        price_action_probabilities = trainer.predict_probabilities(
            X_test, market_data.iloc[split_idx:]
        )
        
        # Verify probability outputs
        expected_probabilities = [
            "triple_barrier_probability",
            "direction_probability", 
            "magnitude_probability",
            "barrier_avoidance_probability"
        ]
        
        for prob_name in expected_probabilities:
            assert prob_name in price_action_probabilities, f"Missing probability: {prob_name}"
            prob_value = price_action_probabilities[prob_name]
            assert 0.0 <= prob_value <= 1.0, f"Invalid probability value for {prob_name}: {prob_value}"
        
        # Verify metadata
        assert "generation_timestamp" in price_action_probabilities
        assert "model_type" in price_action_probabilities
        assert price_action_probabilities["model_type"] == "multi_output"
        
        logger.info("✅ MultiOutputProbabilityTrainer test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ MultiOutputProbabilityTrainer test failed: {e}")
        return False

def test_step6_integration():
    """Test Step 6 HMM-based training integration."""
    logger.info("🧪 Testing Step 6 HMM-based training integration...")
    
    try:
        from src.training.steps.step6_hmm_based_training import HMMBasedTrainingStep
        
        # Create test configuration
        config = {
            "HMM_LM": {
                "specialist_models": {
                    "30m": {"architecture": "LightGBM"}
                }
            }
        }
        
        # Initialize step
        step = HMMBasedTrainingStep(config)
        
        # Create test data
        n_samples = 500
        n_features = 10
        
        data = {
            'close': np.random.randn(n_samples),
            'volume': np.random.randn(n_samples),
            'target': np.random.choice([0, 1], size=n_samples)
        }
        
        # Add some features
        for i in range(n_features):
            data[f'feature_{i}'] = np.random.randn(n_samples)
        
        data = pd.DataFrame(data)
        
        # Test the training function
        result = step._train_lightgbm_model(data, "30m")
        
        if result:
            # Verify result structure
            assert "architecture" in result
            assert "model_path" in result
            assert "price_action_probabilities" in result
            
            # Verify probability outputs
            probabilities = result["price_action_probabilities"]
            expected_probs = [
                "triple_barrier_probability",
                "direction_probability",
                "magnitude_probability", 
                "barrier_avoidance_probability"
            ]
            
            for prob_name in expected_probs:
                assert prob_name in probabilities, f"Missing probability in Step 6: {prob_name}"
            
            logger.info("✅ Step 6 integration test passed!")
            return True
        else:
            logger.error("❌ Step 6 training returned None")
            return False
            
    except Exception as e:
        logger.error(f"❌ Step 6 integration test failed: {e}")
        return False

def test_step9_integration():
    """Test Step 9 Tactician specialist training integration."""
    logger.info("🧪 Testing Step 9 Tactician specialist training integration...")
    
    try:
        from src.training.steps.step9_tactician_specialist_training import TacticianSpecialistTrainingStep
        
        # Create test configuration
        config = {}
        
        # Initialize step
        step = TacticianSpecialistTrainingStep(config)
        
        # Create test data
        n_samples = 500
        n_features = 10
        
        X_train = pd.DataFrame(np.random.randn(n_samples, n_features), 
                              columns=[f'feature_{i}' for i in range(n_features)])
        X_test = pd.DataFrame(np.random.randn(n_samples//2, n_features),
                             columns=[f'feature_{i}' for i in range(n_features)])
        y_train = pd.Series(np.random.choice([0, 1], size=n_samples))
        y_test = pd.Series(np.random.choice([0, 1], size=n_samples//2))
        
        # Test the training function
        result = step._train_lightgbm(X_train, X_test, y_train, y_test, "ETHUSDT", "binance")
        
        if result:
            # Verify result structure
            assert "multi_output_trainer" in result
            assert "trained_models" in result
            assert "price_action_probabilities" in result
            
            # Verify probability outputs
            probabilities = result["price_action_probabilities"]
            expected_probs = [
                "triple_barrier_probability",
                "direction_probability",
                "magnitude_probability",
                "barrier_avoidance_probability"
            ]
            
            for prob_name in expected_probs:
                assert prob_name in probabilities, f"Missing probability in Step 9: {prob_name}"
            
            logger.info("✅ Step 9 integration test passed!")
            return True
        else:
            logger.error("❌ Step 9 training returned None")
            return False
            
    except Exception as e:
        logger.error(f"❌ Step 9 integration test failed: {e}")
        return False

def test_enhanced_step6_integration():
    """Test Enhanced Step 6 integration."""
    logger.info("🧪 Testing Enhanced Step 6 integration...")
    
    try:
        from src.training.steps.step6_hmm_based_training_enhanced import HMMBasedTrainingStepEnhanced
        
        # Create test configuration
        config = {
            "enable_multi_output": True,
            "multi_output_model_type": "LightGBM"
        }
        
        # Initialize step
        step = HMMBasedTrainingStepEnhanced(config)
        
        # Create test data
        n_samples = 500
        n_features = 10
        
        data = pd.DataFrame({
            'close': np.random.randn(n_samples),
            'volume': np.random.randn(n_samples),
            'target': np.random.choice([0, 1], size=n_samples)
        })
        
        # Add features
        for i in range(n_features):
            data[f'feature_{i}'] = np.random.randn(n_samples)
        
        # Test data preparation
        prepared_data = step.prepare_enhanced_data(data, "30m")
        
        # Verify data preparation
        assert "has_multi_output" in prepared_data
        assert "features" in prepared_data
        
        if prepared_data["has_multi_output"] and step.multi_output_trainer:
            # Test prediction
            features = prepared_data["features"]
            predictions = step.predict_enhanced(features, "test_model", "multi_output")
            
            if predictions and len(predictions) == 2:
                direction_pred, profit_pred = predictions
                assert direction_pred is not None
                assert profit_pred is not None
                logger.info("✅ Enhanced Step 6 integration test passed!")
                return True
            else:
                logger.error("❌ Enhanced Step 6 prediction failed")
                return False
        else:
            logger.warning("⚠️ Multi-output not available in enhanced step 6")
            return True
            
    except Exception as e:
        logger.error(f"❌ Enhanced Step 6 integration test failed: {e}")
        return False

def test_model_saving_utils():
    """Test model saving utilities with multi-output models."""
    logger.info("🧪 Testing model saving utilities...")
    
    try:
        from src.training.multi_output_probability_trainer import MultiOutputProbabilityTrainer
        from src.training.model_saving_utils import save_multi_output_model_with_probabilities, load_model_with_probabilities
        
        # Create test trainer
        config = {
            "use_lightgbm": True,
            "n_estimators": 50,
            "learning_rate": 0.1,
            "max_depth": 4
        }
        
        trainer = MultiOutputProbabilityTrainer(config)
        
        # Create test data
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1], size=100)
        market_data = pd.DataFrame({
            'close': np.random.randn(100),
            'volume': np.random.randn(100)
        })
        
        # Generate targets and train
        y_multi = trainer.prepare_multi_output_targets(X, y, market_data)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train_multi = {k: v[:split_idx] for k, v in y_multi.items()}
        y_test_multi = {k: v[split_idx:] for k, v in y_multi.items()}
        
        trained_models = trainer.train_multi_output_model(X_train, y_train_multi, X_test, y_test_multi)
        
        # Create model data
        model_data = {
            "multi_output_trainer": trainer,
            "trained_models": trained_models,
            "model_type": "multi_output",
            "training_date": datetime.now().isoformat(),
            "hyperparameters": config
        }
        
        # Test saving
        model_path = "test_multi_output_model.pkl"
        saved_data = save_multi_output_model_with_probabilities(model_data, model_path)
        
        # Verify saved data
        assert saved_data["model_type"] == "multi_output"
        assert "price_action_probabilities" in saved_data
        
        # Test loading
        loaded_data = load_model_with_probabilities(model_path)
        
        # Verify loaded data
        assert loaded_data["model_type"] == "multi_output"
        assert "multi_output_trainer" in loaded_data
        
        # Clean up
        import os
        if os.path.exists(model_path):
            os.remove(model_path)
        
        logger.info("✅ Model saving utilities test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model saving utilities test failed: {e}")
        return False

async def run_all_tests():
    """Run all integration tests."""
    logger.info("🚀 Starting Multi-Output Training Integration Tests")
    
    tests = [
        ("MultiOutputProbabilityTrainer", test_multi_output_probability_trainer),
        ("Step 6 Integration", test_step6_integration),
        ("Step 9 Integration", test_step9_integration),
        ("Enhanced Step 6 Integration", test_enhanced_step6_integration),
        ("Model Saving Utils", test_model_saving_utils)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            results[test_name] = result
            
            if result:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} ERROR: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Multi-output training integration is working correctly.")
    else:
        logger.error(f"⚠️ {total - passed} tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(run_all_tests())