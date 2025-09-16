#!/usr/bin/env python3
"""
Test script for HMM Common Utilities Integration

This script tests the integration of common utilities into the HMM models training system.
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all common utilities can be imported."""
    print("🔄 Testing imports...")
    
    try:
        # Test common utilities imports
        from src.utils.common_operations import (
            safe_dataframe_operation,
            validate_dataframe_columns,
            calculate_data_quality_metrics,
            get_m1_gpu_manager,
            get_m1_memory_optimizer,
            get_m1_cpu_optimizer
        )
        print("✅ Common operations imported successfully")
        
        from src.utils.math_validation import (
            safe_divide,
            validate_finite,
            validate_numeric_array,
            safe_log,
            safe_sqrt
        )
        print("✅ Math validation imported successfully")
        
        from src.utils.serialization_utils import (
            JSONSerializer,
            PickleSerializer
        )
        print("✅ Serialization utilities imported successfully")
        
        from src.utils.ml_common.evaluation.evaluation_utils import EvaluationUtils
        print("✅ ML common evaluation imported successfully")
        
        # Test HMM training imports
        from src.training.steps.market_analysis.hmm_models_training.hmm_models_training_enhanced import (
            HMMModelsTrainingEnhanced,
            create_enhanced_hmm_models_training,
            HMMTrainingConfig
        )
        print("✅ HMM training enhanced imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_math_validation():
    """Test math validation functions."""
    print("\n🔄 Testing math validation...")
    
    try:
        from src.utils.math_validation import safe_divide, validate_finite, safe_sqrt
        
        # Test safe division
        result = safe_divide(10, 2, 0.0)
        assert result == 5.0, f"Expected 5.0, got {result}"
        print("✅ Safe division works correctly")
        
        # Test division by zero
        result = safe_divide(10, 0, 0.0)
        assert result == 0.0, f"Expected 0.0, got {result}"
        print("✅ Safe division by zero works correctly")
        
        # Test validate finite
        result = validate_finite(5.0, "test_value")
        assert result == 5.0, f"Expected 5.0, got {result}"
        print("✅ Validate finite works correctly")
        
        # Test safe sqrt
        result = safe_sqrt(16.0, 0.0)
        assert result == 4.0, f"Expected 4.0, got {result}"
        print("✅ Safe sqrt works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Math validation error: {e}")
        return False

def test_dataframe_operations():
    """Test DataFrame operations."""
    print("\n🔄 Testing DataFrame operations...")
    
    try:
        from src.utils.common_operations import safe_dataframe_operation, calculate_data_quality_metrics
        
        # Create test DataFrame
        df = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [2, 4, 6, 8, 10],
            'target': [0, 1, 0, 1, 0]
        })
        
        # Test safe DataFrame operation
        numeric_cols = safe_dataframe_operation(
            df, 
            lambda df: df.select_dtypes(include=[np.number]).columns
        )
        assert len(numeric_cols) == 3, f"Expected 3 numeric columns, got {len(numeric_cols)}"
        print("✅ Safe DataFrame operation works correctly")
        
        # Test data quality metrics
        quality_metrics = calculate_data_quality_metrics(df)
        assert 'null_percentage' in quality_metrics, "Missing null_percentage in quality metrics"
        print("✅ Data quality metrics work correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ DataFrame operations error: {e}")
        return False

def test_hmm_training_creation():
    """Test HMM training creation with common utilities."""
    print("\n🔄 Testing HMM training creation...")
    
    try:
        from src.training.steps.market_analysis.hmm_models_training.hmm_models_training_enhanced import (
            create_enhanced_hmm_models_training,
            HMMTrainingConfig
        )
        
        # Create configuration
        config = HMMTrainingConfig(
            model_name="test_hmm_models",
            timeframe="1h",
            n_features=10,
            sequence_length=5,
            n_regimes=2,
            model_types=["lightgbm", "elastic_net_lr"],
            hpo_trials=5,
            enable_multi_objective=True
        )
        
        # Create training step
        training_step = create_enhanced_hmm_models_training(config)
        assert training_step is not None, "Training step creation failed"
        print("✅ HMM training step created successfully")
        
        # Test hardware optimizers initialization
        assert hasattr(training_step, 'gpu_manager'), "GPU manager not initialized"
        assert hasattr(training_step, 'memory_optimizer'), "Memory optimizer not initialized"
        assert hasattr(training_step, 'cpu_optimizer'), "CPU optimizer not initialized"
        print("✅ Hardware optimizers initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ HMM training creation error: {e}")
        return False

def test_serialization():
    """Test serialization utilities."""
    print("\n🔄 Testing serialization utilities...")
    
    try:
        from src.utils.serialization_utils import JSONSerializer, PickleSerializer
        
        # Test data
        test_data = {
            'model_name': 'test_model',
            'accuracy': 0.85,
            'timestamp': '2024-01-01T00:00:00'
        }
        
        # Test JSON serialization
        json_path = "test_metadata.json"
        success = JSONSerializer.save(test_data, json_path)
        assert success, "JSON serialization failed"
        print("✅ JSON serialization works correctly")
        
        # Test JSON loading
        loaded_data = JSONSerializer.load(json_path)
        assert loaded_data is not None, "JSON loading failed"
        assert loaded_data['model_name'] == 'test_model', "JSON data mismatch"
        print("✅ JSON loading works correctly")
        
        # Cleanup
        os.remove(json_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Serialization error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting HMM Common Utilities Integration Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_math_validation,
        test_dataframe_operations,
        test_hmm_training_creation,
        test_serialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! HMM Common Utilities Integration is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the integration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)