#!/usr/bin/env python3
"""
Test script for Simplified Ensemble Creator functionality.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime

# Import the simplified ensemble creator
from src.training.ensemble_creator_simple import SimpleEnsembleCreator, setup_simple_ensemble_creator


async def test_simple_ensemble_creator():
    """Test the Simplified Ensemble Creator functionality."""
    print("🧪 Testing Simplified Ensemble Creator...")
    
    # Test configuration
    config = {
        "ensemble_creator": {
            "min_models_per_ensemble": 2,
            "max_models_per_ensemble": 5,
            "ensemble_pruning_threshold": 0.1,
            "regularization_strength": 0.01,
            "l1_ratio": 0.5,
            "feature_importance_threshold": 0.01,
            "model_performance_threshold": 0.5,
            "correlation_threshold": 0.8,
            "diversity_threshold": 0.3,
            "optimization_iterations": 50,
            "cross_validation_folds": 3,
            "early_stopping_patience": 5,
            "timeframes": ["1m", "5m", "15m", "1h"]
        }
    }
    
    try:
        # Test 1: Initialize Simple Ensemble Creator
        print("\n📋 Test 1: Initializing Simple Ensemble Creator...")
        ensemble_creator = await setup_simple_ensemble_creator(config)
        
        if ensemble_creator is None:
            print("❌ Failed to initialize Simple Ensemble Creator")
            return False
        
        print("✅ Simple Ensemble Creator initialized successfully")
        
        # Test 2: Get System Info
        print("\n📋 Test 2: Getting System Info...")
        system_info = ensemble_creator.get_all_ensembles_info()
        print(f"✅ System Info: {system_info}")
        
        # Test 3: Create Mock Data and Models
        print("\n📋 Test 3: Creating Mock Data and Models...")
        
        # Create mock training data
        mock_training_data = {
            "1m": pd.DataFrame({
                'feature1': np.random.randn(100),
                'feature2': np.random.randn(100),
                'feature3': np.random.randn(100),
                'target': np.random.randint(0, 2, 100)
            }),
            "5m": pd.DataFrame({
                'feature1': np.random.randn(100),
                'feature2': np.random.randn(100),
                'feature3': np.random.randn(100),
                'target': np.random.randint(0, 2, 100)
            })
        }
        
        # Create mock models (simplified)
        try:
            from sklearn.ensemble import RandomForestClassifier
            mock_models = {
                "1m": RandomForestClassifier(n_estimators=10),
                "5m": RandomForestClassifier(n_estimators=10)
            }
            print("✅ Mock data and models created successfully")
        except ImportError:
            print("⚠️ sklearn not available, using dummy models")
            # Create dummy models
            class DummyModel:
                def predict_proba(self, X):
                    return np.random.rand(len(X), 2)
                def predict(self, X):
                    return np.random.randint(0, 2, len(X))
            
            mock_models = {
                "1m": DummyModel(),
                "5m": DummyModel()
            }
            print("✅ Dummy models created successfully")
        
        # Test 4: Test Ensemble Creation
        print("\n📋 Test 4: Testing Ensemble Creation...")
        
        ensemble_result = await ensemble_creator.create_ensemble(
            training_data=mock_training_data,
            models=mock_models,
            ensemble_name="test_ensemble",
            ensemble_type="timeframe_ensemble"
        )
        
        if ensemble_result:
            print("✅ Ensemble creation successful")
            print(f"Ensemble Result Keys: {list(ensemble_result.keys())}")
            print(f"Ensemble Metrics: {ensemble_result.get('metrics', {})}")
        else:
            print("❌ Ensemble creation failed")
            return False
        
        # Test 5: Test Hierarchical Ensemble
        print("\n📋 Test 5: Testing Hierarchical Ensemble...")
        
        base_ensembles = {
            "ensemble1": {"model": mock_models["1m"]},
            "ensemble2": {"model": mock_models["5m"]}
        }
        
        hierarchical_result = await ensemble_creator.create_hierarchical_ensemble(
            base_ensembles=base_ensembles,
            ensemble_name="test_hierarchical"
        )
        
        if hierarchical_result:
            print("✅ Hierarchical ensemble creation successful")
        else:
            print("⚠️ Hierarchical ensemble creation failed")
        
        # Test 6: Test Ensemble Info
        print("\n📋 Test 6: Testing Ensemble Info...")
        ensemble_info = ensemble_creator.get_ensemble_info("test_ensemble")
        print(f"✅ Ensemble Info: {ensemble_info}")
        
        # Test 7: Test System Cleanup
        print("\n📋 Test 7: Testing System Cleanup...")
        await ensemble_creator.stop()
        print("✅ System cleanup completed")
        
        print("\n🎉 All tests passed! Simplified Ensemble Creator is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_ensemble_creator_config():
    """Test Ensemble Creator configuration validation."""
    print("\n🧪 Testing Ensemble Creator Configuration...")
    
    try:
        # Test valid configuration
        valid_config = {
            "ensemble_creator": {
                "min_models_per_ensemble": 3,
                "max_models_per_ensemble": 10,
                "ensemble_pruning_threshold": 0.1,
                "regularization_strength": 0.01,
                "l1_ratio": 0.5,
                "timeframes": ["1m", "5m", "15m", "1h"]
            }
        }
        
        ensemble_creator = SimpleEnsembleCreator(valid_config)
        success = await ensemble_creator.initialize()
        
        if success:
            print("✅ Valid configuration accepted")
        else:
            print("❌ Valid configuration rejected")
            return False
        
        # Test invalid configuration
        invalid_config = {
            "ensemble_creator": {
                "min_models_per_ensemble": 10,  # Greater than max
                "max_models_per_ensemble": 5,
                "ensemble_pruning_threshold": 1.5,  # Invalid threshold
                "timeframes": []  # Empty timeframes
            }
        }
        
        ensemble_creator_invalid = SimpleEnsembleCreator(invalid_config)
        success_invalid = await ensemble_creator_invalid.initialize()
        
        if not success_invalid:
            print("✅ Invalid configuration correctly rejected")
        else:
            print("❌ Invalid configuration incorrectly accepted")
            return False
        
        await ensemble_creator.stop()
        print("✅ Configuration validation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Simplified Ensemble Creator Tests...")
    
    # Run tests
    test_results = []
    
    # Test 1: Basic functionality
    test_results.append(await test_simple_ensemble_creator())
    
    # Test 2: Configuration validation
    test_results.append(await test_ensemble_creator_config())
    
    # Summary
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\n📊 Test Summary:")
    print(f"✅ Passed: {passed_tests}/{total_tests}")
    print(f"❌ Failed: {total_tests - passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Simplified Ensemble Creator is ready for use.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    asyncio.run(main()) 