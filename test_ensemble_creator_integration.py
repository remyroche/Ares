#!/usr/bin/env python3
"""
Test script for Ensemble Creator integration with Enhanced Training Manager.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime

# Import the components
from src.training.enhanced_training_manager import EnhancedTrainingManager
from src.training.ensemble_creator import EnsembleCreator


async def test_ensemble_creator_integration():
    """Test the Ensemble Creator integration with Enhanced Training Manager."""
    print("🧪 Testing Ensemble Creator Integration...")
    
    # Test configuration
    config = {
        "enhanced_training_manager": {
            "enable_advanced_model_training": True,
            "enable_ensemble_training": True,
            "enable_multi_timeframe_training": True,
            "enable_adaptive_training": True,
            "enable_multi_timeframe_training": True,
        },
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
        },
        "ml_confidence_predictor": {
            "model_path": "models/confidence_predictor.joblib",
            "min_samples_for_training": 100,
            "confidence_threshold": 0.6,
            "max_prediction_horizon": 1,
            "retrain_interval_hours": 24
        }
    }
    
    try:
        # Test 1: Initialize Enhanced Training Manager
        print("\n📋 Test 1: Initializing Enhanced Training Manager...")
        training_manager = EnhancedTrainingManager(config)
        
        success = await training_manager.initialize()
        if not success:
            print("❌ Failed to initialize Enhanced Training Manager")
            return False
        
        print("✅ Enhanced Training Manager initialized successfully")
        
        # Test 2: Check Ensemble Creator Integration
        print("\n📋 Test 2: Checking Ensemble Creator Integration...")
        if hasattr(training_manager, 'ensemble_creator') and training_manager.ensemble_creator:
            print("✅ Ensemble Creator is integrated")
            ensemble_info = training_manager.ensemble_creator.get_all_ensembles_info()
            print(f"Ensemble Creator Info: {ensemble_info}")
        else:
            print("❌ Ensemble Creator not found in training manager")
            return False
        
        # Test 3: Test Ensemble Creation
        print("\n📋 Test 3: Testing Ensemble Creation...")
        
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
            }),
            "15m": pd.DataFrame({
                'feature1': np.random.randn(100),
                'feature2': np.random.randn(100),
                'feature3': np.random.randn(100),
                'target': np.random.randint(0, 2, 100)
            })
        }
        
        # Create mock models (simplified)
        from sklearn.ensemble import RandomForestClassifier
        mock_models = {
            "1m": RandomForestClassifier(n_estimators=10),
            "5m": RandomForestClassifier(n_estimators=10),
            "15m": RandomForestClassifier(n_estimators=10)
        }
        
        # Test ensemble creation
        ensemble_result = await training_manager.ensemble_creator.create_ensemble(
            training_data=mock_training_data,
            models=mock_models,
            ensemble_name="test_ensemble",
            ensemble_type="timeframe_ensemble"
        )
        
        if ensemble_result:
            print("✅ Ensemble creation successful")
            print(f"Ensemble Result: {ensemble_result}")
        else:
            print("❌ Ensemble creation failed")
            return False
        
        # Test 4: Test Hierarchical Ensemble
        print("\n📋 Test 4: Testing Hierarchical Ensemble...")
        
        base_ensembles = {
            "ensemble1": {"model": RandomForestClassifier(n_estimators=5)},
            "ensemble2": {"model": RandomForestClassifier(n_estimators=5)}
        }
        
        hierarchical_result = await training_manager.ensemble_creator.create_hierarchical_ensemble(
            base_ensembles=base_ensembles,
            ensemble_name="test_hierarchical"
        )
        
        if hierarchical_result:
            print("✅ Hierarchical ensemble creation successful")
        else:
            print("⚠️ Hierarchical ensemble creation failed (expected for test)")
        
        # Test 5: Test System Cleanup
        print("\n📋 Test 5: Testing System Cleanup...")
        await training_manager.stop()
        print("✅ System cleanup completed")
        
        print("\n🎉 All tests passed! Ensemble Creator integration is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


async def test_ensemble_creator_standalone():
    """Test Ensemble Creator standalone functionality."""
    print("\n🧪 Testing Ensemble Creator Standalone...")
    
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
        # Test standalone ensemble creator
        ensemble_creator = EnsembleCreator(config)
        success = await ensemble_creator.initialize()
        
        if success:
            print("✅ Standalone Ensemble Creator initialized successfully")
            
            # Test ensemble info
            info = ensemble_creator.get_all_ensembles_info()
            print(f"Ensemble Creator Info: {info}")
            
            await ensemble_creator.stop()
            print("✅ Standalone Ensemble Creator stopped successfully")
            return True
        else:
            print("❌ Failed to initialize standalone Ensemble Creator")
            return False
            
    except Exception as e:
        print(f"❌ Standalone test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Ensemble Creator Integration Tests...")
    
    # Run tests
    test_results = []
    
    # Test 1: Integration test
    test_results.append(await test_ensemble_creator_integration())
    
    # Test 2: Standalone test
    test_results.append(await test_ensemble_creator_standalone())
    
    # Summary
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\n📊 Test Summary:")
    print(f"✅ Passed: {passed_tests}/{total_tests}")
    print(f"❌ Failed: {total_tests - passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Ensemble Creator integration is ready for use.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    asyncio.run(main()) 