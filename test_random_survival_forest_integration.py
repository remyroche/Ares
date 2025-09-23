#!/usr/bin/env python3
"""
Test script for Random Survival Forest integration with Tactician models.

This script tests the integration of Random Survival Forest into the tactician
training pipeline and verifies multi-horizon framework compatibility.
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_random_survival_forest_import():
    """Test that Random Survival Forest can be imported."""
    try:
        from src.training.steps.model_training.random_survival_forest_tactician import (
            RandomSurvivalForestTactician, 
            SurvivalAnalysisConfig,
            MultiHorizonRandomSurvivalForest
        )
        print("✅ Random Survival Forest imports successful")
        return True
    except ImportError as e:
        print(f"❌ Random Survival Forest import failed: {e}")
        return False

def test_survival_analysis_config():
    """Test SurvivalAnalysisConfig creation."""
    try:
        from src.training.steps.model_training.random_survival_forest_tactician import SurvivalAnalysisConfig
        
        config = SurvivalAnalysisConfig(
            n_estimators=100,
            max_depth=5,
            horizons=[1, 2, 5, 10],
            horizon_weights=[0.4, 0.3, 0.2, 0.1]
        )
        
        print("✅ SurvivalAnalysisConfig creation successful")
        print(f"📊 Config: {config.n_estimators} estimators, {config.max_depth} depth")
        print(f"📊 Horizons: {config.horizons} minutes")
        return True
    except Exception as e:
        print(f"❌ SurvivalAnalysisConfig creation failed: {e}")
        return False

def test_random_survival_forest_creation():
    """Test RandomSurvivalForestTactician creation."""
    try:
        from src.training.steps.model_training.random_survival_forest_tactician import (
            RandomSurvivalForestTactician, 
            SurvivalAnalysisConfig
        )
        
        config = SurvivalAnalysisConfig(
            n_estimators=50,  # Reduced for testing
            max_depth=5,
            horizons=[1, 2, 5],
            horizon_weights=[0.5, 0.3, 0.2]
        )
        
        rsf_model = RandomSurvivalForestTactician(config)
        print("✅ RandomSurvivalForestTactician creation successful")
        return True
    except Exception as e:
        print(f"❌ RandomSurvivalForestTactician creation failed: {e}")
        return False

def test_tactician_training_config():
    """Test that TacticianTrainingConfig includes RandomSurvivalForest."""
    try:
        from src.utils.ml_common.config.base_training_config import TacticianTrainingConfig
        
        config = TacticianTrainingConfig()
        
        if "RandomSurvivalForest" in config.model_types:
            print("✅ RandomSurvivalForest included in TacticianTrainingConfig")
            print(f"📊 Model types: {config.model_types}")
            return True
        else:
            print("❌ RandomSurvivalForest not found in TacticianTrainingConfig")
            return False
    except Exception as e:
        print(f"❌ TacticianTrainingConfig test failed: {e}")
        return False

def test_tactician_training_step_integration():
    """Test that TacticianModelsTrainingStepRefactored includes RandomSurvivalForest."""
    try:
        from src.training.steps.model_training.tactician_models_training_refactored import (
            TacticianModelsTrainingStepRefactored,
            TacticianTrainingConfig
        )
        
        config = TacticianTrainingConfig(
            model_types=["XGBOOST", "LIGHTGBM", "RandomSurvivalForest"]
        )
        
        training_step = TacticianModelsTrainingStepRefactored(config)
        
        # Test model instance creation
        rsf_model = training_step._create_model_instance("RandomSurvivalForest")
        
        print("✅ TacticianModelsTrainingStepRefactored integration successful")
        print(f"📊 Created model type: {type(rsf_model).__name__}")
        return True
    except Exception as e:
        print(f"❌ TacticianModelsTrainingStepRefactored integration failed: {e}")
        return False

def test_multi_horizon_integration():
    """Test multi-horizon framework integration."""
    try:
        from src.training.steps.model_training.random_survival_forest_tactician import (
            MultiHorizonRandomSurvivalForest,
            SurvivalAnalysisConfig
        )
        
        config = SurvivalAnalysisConfig(
            n_estimators=50,
            max_depth=5,
            horizons=[1, 2, 5, 10, 15, 30]
        )
        
        multi_horizon_rsf = MultiHorizonRandomSurvivalForest(config)
        
        print("✅ Multi-horizon Random Survival Forest creation successful")
        print(f"📊 Horizons: {config.horizons}")
        return True
    except Exception as e:
        print(f"❌ Multi-horizon integration failed: {e}")
        return False

def test_survival_data_preparation():
    """Test survival data preparation for training."""
    try:
        from src.training.steps.model_training.random_survival_forest_tactician import (
            RandomSurvivalForestTactician,
            SurvivalAnalysisConfig
        )
        
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        n_features = 50
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.exponential(5, n_samples)  # Exponential distribution for timing
        
        # Create analyst signals
        analyst_signals = np.random.randn(n_samples, 5)
        
        # Create HMM regime probabilities
        hmm_regime_probs = np.random.rand(n_samples, 10)
        
        # Create Random Survival Forest model
        config = SurvivalAnalysisConfig(
            n_estimators=50,
            max_depth=5,
            horizons=[1, 2, 5, 10]
        )
        
        rsf_model = RandomSurvivalForestTactician(config)
        
        # Test data preparation
        X_enhanced, feature_names = rsf_model._prepare_features(
            X, None, analyst_signals, hmm_regime_probs
        )
        
        survival_data = rsf_model._prepare_survival_data(y, None)
        
        print("✅ Survival data preparation successful")
        print(f"📊 Enhanced features shape: {X_enhanced.shape}")
        print(f"📊 Feature names count: {len(feature_names)}")
        print(f"📊 Survival data keys: {list(survival_data.keys())}")
        return True
    except Exception as e:
        print(f"❌ Survival data preparation failed: {e}")
        return False

def main():
    """Run all tests for Random Survival Forest integration."""
    print("🚀 Testing Random Survival Forest integration with Tactician models...")
    print("=" * 70)
    
    tests = [
        ("Random Survival Forest Import", test_random_survival_forest_import),
        ("SurvivalAnalysisConfig", test_survival_analysis_config),
        ("RandomSurvivalForestTactician Creation", test_random_survival_forest_creation),
        ("TacticianTrainingConfig Integration", test_tactician_training_config),
        ("TacticianModelsTrainingStepRefactored Integration", test_tactician_training_step_integration),
        ("Multi-horizon Integration", test_multi_horizon_integration),
        ("Survival Data Preparation", test_survival_data_preparation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n📈 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Random Survival Forest integration is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)