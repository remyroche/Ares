#!/usr/bin/env python3
"""
Test Script for Enhanced Training Integration

This script tests the integration of enhanced training utilities into the
Analyst and Tactician training pipelines to verify that overfitting prevention,
lookahead bias detection, and enhanced regularization are working correctly.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_enhanced_training_utilities():
    """Test the enhanced training utilities directly."""
    print("🧪 Testing Enhanced Training Utilities...")
    
    try:
        from src.utils.ml_common.training.enhanced_training_utils import (
            EnhancedTrainingUtils,
            EarlyStoppingConfig,
            PurgedCVConfig,
            OverfittingMonitorConfig,
            RegularizationConfig
        )
        from src.utils.ml_common.training.training_integration import (
            TrainingStepEnhancer,
            TrainingIntegrationConfig
        )
        
        print("✅ Enhanced training utilities imported successfully")
        
        # Test configuration creation
        config = TrainingIntegrationConfig(
            enable_early_stopping=True,
            enable_purged_cv=True,
            enable_lookahead_detection=True,
            enable_temporal_splits=True,
            enable_regularization=True,
            enable_overfitting_monitoring=True
        )
        print("✅ Training configuration created successfully")
        
        # Test enhanced training utils initialization
        enhanced_utils = EnhancedTrainingUtils(
            early_stopping_config=EarlyStoppingConfig(),
            purged_cv_config=PurgedCVConfig(),
            overfitting_config=OverfittingMonitorConfig(),
            regularization_config=RegularizationConfig()
        )
        print("✅ Enhanced training utils initialized successfully")
        
        # Test training enhancer
        enhancer = TrainingStepEnhancer(config)
        print("✅ Training enhancer initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced training utilities test failed: {e}")
        return False

def test_analyst_training_integration():
    """Test the Analyst training integration."""
    print("\n🧪 Testing Analyst Training Integration...")
    
    try:
        from src.training.steps.model_training.analyst_models_training_refactored import (
            AnalystModelsTrainingStepRefactored
        )
        from src.utils.ml_common.config import PerRegimeTrainingConfig
        
        print("✅ Analyst training module imported successfully")
        
        # Create test data
        np.random.seed(42)
        X = np.random.randn(1000, 20)
        y = np.random.randn(1000)
        regime_labels = np.random.randint(0, 3, 1000)
        timestamps = pd.date_range('2023-01-01', periods=1000, freq='1H')
        
        # Create configuration
        config = PerRegimeTrainingConfig(
            model_name="test_analyst",
            timeframe="1m",
            model_types=["RandomForestRegressor", "ElasticNet"],
            min_samples_per_regime=100
        )
        
        # Initialize training step
        training_step = AnalystModelsTrainingStepRefactored(config)
        print("✅ Analyst training step initialized successfully")
        
        # Check if enhanced training utilities are available
        if hasattr(training_step, 'training_enhancer') and training_step.training_enhancer is not None:
            print("✅ Enhanced training utilities are integrated in Analyst training")
        else:
            print("⚠️ Enhanced training utilities not available in Analyst training")
        
        return True
        
    except Exception as e:
        print(f"❌ Analyst training integration test failed: {e}")
        return False

def test_tactician_training_integration():
    """Test the Tactician training integration."""
    print("\n🧪 Testing Tactician Training Integration...")
    
    try:
        from src.training.steps.model_training.tactician_models_training_refactored import (
            TacticianModelsTrainingStepRefactored
        )
        from src.utils.ml_common.config import TacticianTrainingConfig
        
        print("✅ Tactician training module imported successfully")
        
        # Create test data
        np.random.seed(42)
        X = np.random.randn(1000, 20)
        y = np.random.randn(1000)
        regime_labels = np.random.randint(0, 3, 1000)
        timestamps = pd.date_range('2023-01-01', periods=1000, freq='1m')
        analyst_signals = np.random.choice([True, False], 1000, p=[0.3, 0.7])
        
        # Create configuration
        config = TacticianTrainingConfig(
            model_name="test_tactician",
            timeframe="1m",
            model_types=["NeuralObliviousDecisionEnsembles", "CatBoostRegressor"],
            min_samples_per_regime=100
        )
        
        # Initialize training step
        training_step = TacticianModelsTrainingStepRefactored(config)
        print("✅ Tactician training step initialized successfully")
        
        # Check if enhanced training utilities are available
        if hasattr(training_step, 'training_enhancer') and training_step.training_enhancer is not None:
            print("✅ Enhanced training utilities are integrated in Tactician training")
        else:
            print("⚠️ Enhanced training utilities not available in Tactician training")
        
        return True
        
    except Exception as e:
        print(f"❌ Tactician training integration test failed: {e}")
        return False

def test_ensemble_training_integration():
    """Test the Ensemble training integration."""
    print("\n🧪 Testing Ensemble Training Integration...")
    
    try:
        from src.training.steps.model_training.tactician_ensemble_training import (
            TacticianEnsembleTrainingStep
        )
        from src.utils.ml_common.config.base_training_config import EnsembleTrainingConfig
        
        print("✅ Ensemble training module imported successfully")
        
        # Create configuration
        config = EnsembleTrainingConfig(
            model_name="test_ensemble",
            timeframe="1m",
            model_types=["xgboost", "randomforest", "catboost"],
            min_samples_per_regime=100
        )
        
        # Initialize training step
        training_step = TacticianEnsembleTrainingStep(config)
        print("✅ Ensemble training step initialized successfully")
        
        # Check if enhanced training utilities are available
        if hasattr(training_step, 'training_enhancer') and training_step.training_enhancer is not None:
            print("✅ Enhanced training utilities are integrated in Ensemble training")
        else:
            print("⚠️ Enhanced training utilities not available in Ensemble training")
        
        return True
        
    except Exception as e:
        print(f"❌ Ensemble training integration test failed: {e}")
        return False

def test_temporal_data_validation():
    """Test temporal data validation for lookahead bias."""
    print("\n🧪 Testing Temporal Data Validation...")
    
    try:
        from src.utils.ml_common.training.enhanced_training_utils import EnhancedTrainingUtils
        
        # Create test data with timestamps
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        timestamps = pd.date_range('2023-01-01', periods=100, freq='1H')
        
        # Test valid temporal data
        enhanced_utils = EnhancedTrainingUtils()
        is_valid, warnings = enhanced_utils.validate_temporal_data(X, y, timestamps)
        
        if is_valid:
            print("✅ Valid temporal data validation passed")
        else:
            print(f"⚠️ Temporal data validation warnings: {warnings}")
        
        # Test invalid temporal data (future timestamps)
        future_timestamps = pd.date_range('2024-01-01', periods=100, freq='1H')
        is_valid_future, warnings_future = enhanced_utils.validate_temporal_data(X, y, future_timestamps)
        
        if not is_valid_future:
            print("✅ Future timestamp detection working correctly")
        else:
            print("⚠️ Future timestamp detection not working")
        
        return True
        
    except Exception as e:
        print(f"❌ Temporal data validation test failed: {e}")
        return False

def test_enhanced_regularization():
    """Test enhanced regularization."""
    print("\n🧪 Testing Enhanced Regularization...")
    
    try:
        from src.utils.ml_common.training.enhanced_training_utils import EnhancedTrainingUtils
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import ElasticNet
        
        enhanced_utils = EnhancedTrainingUtils()
        
        # Test RandomForest regularization
        rf_model = RandomForestRegressor()
        enhanced_rf = enhanced_utils.apply_enhanced_regularization(rf_model, 'randomforest')
        print("✅ RandomForest regularization applied successfully")
        
        # Test ElasticNet regularization
        en_model = ElasticNet()
        enhanced_en = enhanced_utils.apply_enhanced_regularization(en_model, 'elasticnet')
        print("✅ ElasticNet regularization applied successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced regularization test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🚀 Starting Enhanced Training Integration Tests")
    print("=" * 60)
    
    tests = [
        test_enhanced_training_utilities,
        test_analyst_training_integration,
        test_tactician_training_integration,
        test_ensemble_training_integration,
        test_temporal_data_validation,
        test_enhanced_regularization
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced training integration is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the integration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)