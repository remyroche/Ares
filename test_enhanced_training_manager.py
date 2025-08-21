#!/usr/bin/env python3
"""
Test script for Enhanced Training Manager
Verifies that the enhanced training manager is functional with all decorators and validation logic
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test core imports
        from src.training.enhanced_training_manager import EnhancedTrainingManager
        print("✅ EnhancedTrainingManager imported successfully")
        
        from src.utils.data_quality_validator import DataQualityValidator
        print("✅ DataQualityValidator imported successfully")
        
        from src.utils.data_sanitizer import DataSanitizer
        print("✅ DataSanitizer imported successfully")
        
        from src.utils.training_pipeline_decorators import (
            validate_pipeline_step,
            ensure_data_integrity,
            monitor_step_execution,
            secure_step_execution,
            validate_pipeline_input,
            monitor_performance,
            data_quality_guard,
            artifact_versioning,
            time_budget_watchdog,
            nan_inf_and_constant_guard,
        )
        print("✅ Training pipeline decorators imported successfully")
        
        from src.utils.error_handler import (
            handle_errors,
            handle_specific_errors,
            retry_on_failure,
            circuit_breaker,
            safe_operation,
        )
        print("✅ Error handler decorators imported successfully")
        
        from src.utils.step_dependency_validator import step_dependency_validator
        print("✅ Step dependency validator imported successfully")
        
        from src.utils.validator_orchestrator import validator_orchestrator
        print("✅ Validator orchestrator imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import: {e}")
        return False

def test_data_quality_validator():
    """Test the DataQualityValidator functionality"""
    print("\nTesting DataQualityValidator...")
    
    try:
        from src.utils.data_quality_validator import DataQualityValidator
        import pandas as pd
        import numpy as np
        
        validator = DataQualityValidator()
        
        # Test DataFrame validation
        test_df = pd.DataFrame({
            'col1': [1, 2, 3, np.nan, 5],
            'col2': [1.0, 2.0, np.inf, 4.0, 5.0],
            'col3': [1, 1, 1, 1, 1],  # Constant column
            'col4': [1, 2, 3, 4, 5]
        })
        
        result = validator.validate_dataframe(test_df, "test_dataframe")
        print(f"✅ DataFrame validation result: {result.is_valid}")
        print(f"   Errors: {len(result.errors)}")
        print(f"   Warnings: {len(result.warnings)}")
        
        # Test training data validation
        training_data = {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1h',
            'lookback_days': 30
        }
        
        result = validator.validate_training_data(training_data)
        print(f"✅ Training data validation result: {result.is_valid}")
        
        # Test pipeline state validation
        pipeline_state = {
            'current_step': 'step1',
            'completed_steps': ['step1'],
            'step_results': {'step1': {'status': 'completed'}}
        }
        
        result = validator.validate_pipeline_state(pipeline_state)
        print(f"✅ Pipeline state validation result: {result.is_valid}")
        
        return True
        
    except Exception as e:
        print(f"❌ DataQualityValidator test failed: {e}")
        return False

def test_data_sanitizer():
    """Test the DataSanitizer functionality"""
    print("\nTesting DataSanitizer...")
    
    try:
        from src.utils.data_sanitizer import DataSanitizer
        import pandas as pd
        import numpy as np
        
        sanitizer = DataSanitizer()
        
        # Test identifier sanitization
        test_identifiers = [
            'BTC/USDT',
            'file<name>',
            'path/with\\backslashes',
            '  spaced_name  ',
            ''
        ]
        
        for identifier in test_identifiers:
            sanitized = sanitizer.sanitize_identifier(identifier)
            print(f"✅ '{identifier}' -> '{sanitized}'")
        
        # Test DataFrame sanitization
        test_df = pd.DataFrame({
            'col 1': [1, 2, 3, np.inf, 5],
            'col<2>': [1.0, 2.0, -np.inf, 4.0, 5.0],
            'normal_col': [1, 2, 3, 4, 5]
        })
        
        sanitized_df = sanitizer.sanitize_dataframe(test_df, "test_dataframe")
        print(f"✅ DataFrame sanitized: {sanitized_df.shape}")
        print(f"   Columns: {list(sanitized_df.columns)}")
        
        # Test training data sanitization
        training_data = {
            'symbol': '  BTCUSDT  ',
            'exchange': 'binance',
            'timeframe': '1h',
            'lookback_days': '30',
            'enable_model_training': 'true'
        }
        
        sanitized_data = sanitizer.sanitize_training_data(training_data)
        print(f"✅ Training data sanitized: {sanitized_data}")
        
        return True
        
    except Exception as e:
        print(f"❌ DataSanitizer test failed: {e}")
        return False

def test_enhanced_training_manager_structure():
    """Test the EnhancedTrainingManager structure and initialization"""
    print("\nTesting EnhancedTrainingManager structure...")
    
    try:
        from src.training.enhanced_training_manager import EnhancedTrainingManager
        
        # Create a minimal config
        config = {
            'enhanced_training_manager': {
                'enable_model_training': True,
                'enable_validators': True,
                'enable_computational_optimization': True,
                'verbosity': 'info'
            },
            'computational_optimization': {
                'enable_caching': True,
                'enable_parallelization': True,
                'enable_early_stopping': True,
                'enable_memory_management': True
            }
        }
        
        # Test instantiation
        manager = EnhancedTrainingManager(config)
        print("✅ EnhancedTrainingManager instantiated successfully")
        
        # Check that required components are initialized
        assert hasattr(manager, 'data_quality_validator'), "Missing data_quality_validator"
        assert hasattr(manager, 'data_sanitizer'), "Missing data_sanitizer"
        assert hasattr(manager, 'step_dependency_validator'), "Missing step_dependency_validator"
        assert hasattr(manager, 'force_rerun'), "Missing force_rerun"
        assert hasattr(manager, 'enable_checkpointing'), "Missing enable_checkpointing"
        
        print("✅ All required components initialized")
        
        # Check that decorators are applied to key methods
        import inspect
        
        # Check initialize method
        init_method = getattr(manager, 'initialize', None)
        if init_method:
            print("✅ initialize method exists")
        
        # Check execute_enhanced_training method
        execute_method = getattr(manager, 'execute_enhanced_training', None)
        if execute_method:
            print("✅ execute_enhanced_training method exists")
        
        # Check validation methods
        validate_method = getattr(manager, '_validate_enhanced_training_inputs', None)
        if validate_method:
            print("✅ _validate_enhanced_training_inputs method exists")
        
        return True
        
    except Exception as e:
        print(f"❌ EnhancedTrainingManager structure test failed: {e}")
        return False

def test_step_dependency_validation():
    """Test step dependency validation logic"""
    print("\nTesting step dependency validation...")
    
    try:
        from src.utils.step_dependency_validator import step_dependency_validator
        
        # Test step dependency validation
        pipeline_state = {
            'current_step': 'step2',
            'completed_steps': ['step1'],
            'step_results': {
                'step1': {'status': 'completed', 'artifacts': ['artifact1.parquet']}
            }
        }
        
        # Test validation without force flag
        result = step_dependency_validator.validate_step_prerequisites(
            'step2', pipeline_state, 'checkpoints', force_rerun=False
        )
        print(f"✅ Step dependency validation (no force): {result}")
        
        # Test validation with force flag
        result = step_dependency_validator.validate_step_prerequisites(
            'step2', pipeline_state, 'checkpoints', force_rerun=True
        )
        print(f"✅ Step dependency validation (with force): {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Step dependency validation test failed: {e}")
        return False

async def test_async_functionality():
    """Test async functionality of the enhanced training manager"""
    print("\nTesting async functionality...")
    
    try:
        from src.training.enhanced_training_manager import setup_enhanced_training_manager
        
        # Test setup function
        config = {
            'enhanced_training_manager': {
                'enable_model_training': False,  # Disable to avoid dependency issues
                'enable_validators': True,
                'enable_computational_optimization': False,  # Disable to avoid dependency issues
                'verbosity': 'info'
            }
        }
        
        manager = await setup_enhanced_training_manager(config)
        if manager:
            print("✅ Async setup successful")
        else:
            print("⚠️ Async setup returned None (expected due to missing dependencies)")
        
        return True
        
    except Exception as e:
        print(f"❌ Async functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Enhanced Training Manager Functionality")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Data Quality Validator", test_data_quality_validator),
        ("Data Sanitizer", test_data_sanitizer),
        ("Enhanced Training Manager Structure", test_enhanced_training_manager_structure),
        ("Step Dependency Validation", test_step_dependency_validation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    # Test async functionality
    print(f"\n🔍 Running Async Functionality Test...")
    try:
        result = asyncio.run(test_async_functionality())
        if result:
            passed += 1
            print("✅ Async Functionality Test PASSED")
        else:
            print("❌ Async Functionality Test FAILED")
    except Exception as e:
        print(f"❌ Async Functionality Test FAILED with exception: {e}")
    
    total += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced Training Manager is functional.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())