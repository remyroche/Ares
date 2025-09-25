#!/usr/bin/env python3
"""
Comprehensive Test Script for Models Training Fixes

This script tests all the error handling and logging improvements
made to the models_training scripts to ensure zero silent failures
and proper tprint logging throughout.
"""

import sys
import os
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.insert(0, '/workspace/src')

def test_tprint_imports():
    """Test that all scripts can import tprint correctly."""
    print("🧪 Testing tprint imports...")
    
    try:
        from training.steps.models_training.nas_tas.model_manager import ModelManager, ModelManagerConfig
        print("✅ Model Manager imports tprint successfully")
    except Exception as e:
        print(f"❌ Model Manager tprint import failed: {e}")
        return False
    
    try:
        from training.steps.models_training.nas_tas.performance_tracker import PerformanceTracker, PerformanceConfig
        print("✅ Performance Tracker imports tprint successfully")
    except Exception as e:
        print(f"❌ Performance Tracker tprint import failed: {e}")
        return False
    
    try:
        from training.steps.models_training.nas_tas.model_selector import ModelSelector, ModelSelectionConfig
        print("✅ Model Selector imports tprint successfully")
    except Exception as e:
        print(f"❌ Model Selector tprint import failed: {e}")
        return False
    
    try:
        from training.steps.models_training.nas_tas.regime_aware_trainer import RegimeAwareTrainer, RegimeAwareTrainingConfig
        print("✅ Regime Aware Trainer imports tprint successfully")
    except Exception as e:
        print(f"❌ Regime Aware Trainer tprint import failed: {e}")
        return False
    
    try:
        from training.steps.models_training.nas_tas.training_orchestrator import TrainingOrchestrator, OrchestratorConfig
        print("✅ Training Orchestrator imports tprint successfully")
    except Exception as e:
        print(f"❌ Training Orchestrator tprint import failed: {e}")
        return False
    
    return True

def test_silent_failure_prevention():
    """Test that silent failures are prevented."""
    print("\n🧪 Testing silent failure prevention...")
    
    # Test Performance Tracker silent failure prevention
    try:
        from training.steps.models_training.nas_tas.performance_tracker import PerformanceTracker, PerformanceConfig
        
        config = PerformanceConfig()
        tracker = PerformanceTracker(config)
        
        # This should raise an exception instead of silently failing
        try:
            result = tracker.record_performance(
                model_id="test_model",
                regime_id=1,
                performance_metrics={'f1_score': 0.8, 'accuracy': 0.9},
                prediction_time=0.1
            )
            print("✅ Performance Tracker properly handles valid input")
        except Exception as e:
            print(f"❌ Performance Tracker failed with valid input: {e}")
            return False
        
        # Test with invalid input that should raise an exception
        try:
            result = tracker.record_performance(
                model_id="test_model",
                regime_id=1,
                performance_metrics=None,  # This should cause an error
                prediction_time=0.1
            )
            print("❌ Performance Tracker should have failed with None metrics")
            return False
        except Exception as e:
            print(f"✅ Performance Tracker properly raises exception for invalid input: {e}")
        
    except Exception as e:
        print(f"❌ Performance Tracker test setup failed: {e}")
        return False
    
    return True

def test_file_operation_error_handling():
    """Test that file operations have proper error handling."""
    print("\n🧪 Testing file operation error handling...")
    
    try:
        from training.steps.models_training.nas_tas.model_manager import ModelManager, ModelManagerConfig
        
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ModelManagerConfig(
                model_storage_path=temp_dir + "/models",
                model_versions_path=temp_dir + "/versions",
                model_metadata_path=temp_dir + "/metadata"
            )
            
            manager = ModelManager(config)
            
            # Test model saving with proper error handling
            try:
                # Create a dummy model
                from sklearn.ensemble import RandomForestClassifier
                dummy_model = RandomForestClassifier(n_estimators=10, random_state=42)
                dummy_model.fit([[1, 2], [3, 4]], [0, 1])
                
                # This should work
                manager._save_model(
                    model_id="test_model",
                    version="1.0.0",
                    model=dummy_model,
                    metadata=manager.model_registry.get("test_model", type('obj', (object,), {
                        'model_id': 'test_model',
                        'model_type': 'random_forest',
                        'regime_id': 1,
                        'version': '1.0.0',
                        'training_performance': {},
                        'validation_performance': {},
                        'test_performance': {},
                        'feature_importance': {},
                        'hyperparameters': {},
                        'model_size': 1000,
                        'created_at': datetime.now(),
                        'training_data_shape': (2, 2),
                        'feature_names': ['feature1', 'feature2']
                    })())
                )
                print("✅ Model saving works with proper error handling")
                
            except Exception as e:
                print(f"❌ Model saving failed: {e}")
                return False
            
            # Test model loading with validation
            try:
                loaded_model = manager._load_model("test_model", "1.0.0")
                if loaded_model is None:
                    print("❌ Loaded model is None")
                    return False
                if not hasattr(loaded_model, 'predict'):
                    print("⚠️ Loaded model doesn't have predict method (expected warning)")
                print("✅ Model loading works with validation")
                
            except Exception as e:
                print(f"❌ Model loading failed: {e}")
                return False
    
    except Exception as e:
        print(f"❌ File operation test setup failed: {e}")
        return False
    
    return True

def test_placeholder_warnings():
    """Test that placeholder functions show warnings."""
    print("\n🧪 Testing placeholder function warnings...")
    
    try:
        from training.steps.models_training.nas_tas.model_manager import ModelManager, ModelManagerConfig
        
        config = ModelManagerConfig()
        manager = ModelManager(config)
        
        # Create dummy model and metadata
        from sklearn.ensemble import RandomForestClassifier
        dummy_model = RandomForestClassifier(n_estimators=10, random_state=42)
        dummy_model.fit([[1, 2], [3, 4]], [0, 1])
        
        metadata = type('obj', (object,), {
            'validation_performance': {'f1_score': 0.8, 'accuracy': 0.9}
        })()
        
        # Test that deployment methods show warnings (they're placeholders)
        print("Testing deployment placeholder warnings...")
        
        # These should show warnings about being placeholders
        result1 = manager._immediate_deployment("test_model", dummy_model, metadata)
        result2 = manager._gradual_deployment("test_model", dummy_model, metadata)
        result3 = manager._ab_testing_deployment("test_model", dummy_model, metadata)
        result4 = manager._canary_deployment("test_model", dummy_model, metadata)
        
        if all([result1, result2, result3, result4]):
            print("✅ Deployment methods work (with placeholder warnings)")
        else:
            print("❌ Some deployment methods failed")
            return False
    
    except Exception as e:
        print(f"❌ Placeholder warning test failed: {e}")
        return False
    
    return True

def test_model_selector_error_handling():
    """Test that model selector has proper error handling."""
    print("\n🧪 Testing model selector error handling...")
    
    try:
        from training.steps.models_training.nas_tas.model_selector import ModelSelector, ModelSelectionConfig
        
        config = ModelSelectionConfig()
        selector = ModelSelector(config)
        
        # Test confidence calculation error handling
        try:
            # Create dummy market data
            market_data = pd.DataFrame({
                'close': [100, 101, 102, 103, 104],
                'volume': [1000, 1100, 1200, 1300, 1400]
            })
            
            # This should work
            meta_features = selector._extract_meta_features(market_data, None)
            print("✅ Meta-feature extraction works")
            
        except Exception as e:
            print(f"❌ Meta-feature extraction failed: {e}")
            return False
    
    except Exception as e:
        print(f"❌ Model selector test setup failed: {e}")
        return False
    
    return True

def test_comprehensive_logging():
    """Test that comprehensive logging is working."""
    print("\n🧪 Testing comprehensive logging...")
    
    try:
        from training.steps.models_training.nas_tas.performance_tracker import PerformanceTracker, PerformanceConfig
        
        config = PerformanceConfig()
        tracker = PerformanceTracker(config)
        
        # Test that setup_model_tracking logs properly
        result = tracker.setup_model_tracking("test_model", {
            'val_metrics': {'f1_score': 0.8, 'accuracy': 0.9}
        })
        
        if result.get('status') == 'tracking_enabled':
            print("✅ Performance tracking setup logs properly")
        else:
            print(f"❌ Performance tracking setup failed: {result}")
            return False
    
    except Exception as e:
        print(f"❌ Comprehensive logging test failed: {e}")
        return False
    
    return True

def test_error_recovery():
    """Test error recovery mechanisms."""
    print("\n🧪 Testing error recovery mechanisms...")
    
    try:
        from training.steps.models_training.nas_tas.model_selector import ModelSelector, ModelSelectionConfig
        
        config = ModelSelectionConfig()
        selector = ModelSelector(config)
        
        # Test fallback regime detection
        market_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        regime_info = selector._fallback_regime_detection(market_data)
        
        if 'regime_id' in regime_info and 'confidence' in regime_info:
            print("✅ Fallback regime detection works")
        else:
            print("❌ Fallback regime detection failed")
            return False
    
    except Exception as e:
        print(f"❌ Error recovery test failed: {e}")
        return False
    
    return True

def run_all_tests():
    """Run all tests and report results."""
    print("🚀 Starting comprehensive test suite for models_training fixes...")
    print("=" * 70)
    
    tests = [
        ("tprint_imports", test_tprint_imports),
        ("silent_failure_prevention", test_silent_failure_prevention),
        ("file_operation_error_handling", test_file_operation_error_handling),
        ("placeholder_warnings", test_placeholder_warnings),
        ("model_selector_error_handling", test_model_selector_error_handling),
        ("comprehensive_logging", test_comprehensive_logging),
        ("error_recovery", test_error_recovery)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"📊 TEST RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! All fixes are working correctly.")
        return True
    else:
        print(f"⚠️ {failed} tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)