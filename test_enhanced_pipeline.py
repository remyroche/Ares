#!/usr/bin/env python3
"""
Test script for the enhanced market analysis pipeline.
This script tests the pipeline structure without requiring external dependencies.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_pipeline_imports():
    """Test that the enhanced pipeline components can be imported."""
    print("🧪 Testing enhanced pipeline imports...")
    
    try:
        # Test enhanced orchestrator import
        from src.training.steps.market_analysis.enhanced_market_analysis_orchestrator import (
            MarketAnalysisPipelineOrchestrator,
            run_enhanced_market_analysis_pipeline,
        )
        print("✅ Enhanced orchestrator imported successfully")
        
        # Test enhanced validator import
        from src.training.steps.market_analysis.enhanced_step_validator import (
            EnhancedStepValidator,
            validate_step_input,
            validate_step_output,
            validate_step_transition,
        )
        print("✅ Enhanced validator imported successfully")
        
        # Test enhanced decorators import
        from src.training.steps.market_analysis.enhanced_pipeline_decorators import (
            comprehensive_pipeline_protection,
            data_formatting,
            data_analysis_protection,
            data_access_protection,
        )
        print("✅ Enhanced decorators imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_orchestrator_initialization():
    """Test that the orchestrator can be initialized."""
    print("\n🧪 Testing orchestrator initialization...")
    
    try:
        from src.training.steps.market_analysis.enhanced_market_analysis_orchestrator import (
            MarketAnalysisPipelineOrchestrator,
        )
        
        # Test with default config
        orchestrator = MarketAnalysisPipelineOrchestrator()
        print("✅ Orchestrator initialized with default config")
        
        # Test with custom config
        config = {
            'hmm_clustering': True,
            'regime_splitting': True,
            'feature_engineering': True,
            'matrix_operations': True,
            'feature_selection': True,
        }
        orchestrator = MarketAnalysisPipelineOrchestrator(config)
        print("✅ Orchestrator initialized with custom config")
        
        # Test step configs
        step_configs = orchestrator.step_configs
        print(f"✅ Step configurations loaded: {list(step_configs.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator initialization failed: {e}")
        return False

def test_validator_initialization():
    """Test that the validator can be initialized."""
    print("\n🧪 Testing validator initialization...")
    
    try:
        from src.training.steps.market_analysis.enhanced_step_validator import (
            EnhancedStepValidator,
        )
        
        # Test with default config
        validator = EnhancedStepValidator()
        print("✅ Validator initialized with default config")
        
        # Test step schemas
        schemas = validator.get_all_step_schemas()
        print(f"✅ Step schemas loaded: {list(schemas.keys())}")
        
        # Test specific schema
        hmm_schema = validator.get_step_schema('hmm_clustering')
        if hmm_schema:
            print("✅ HMM clustering schema loaded successfully")
        else:
            print("❌ HMM clustering schema not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Validator initialization failed: {e}")
        return False

def test_decorator_functionality():
    """Test that the decorators can be applied."""
    print("\n🧪 Testing decorator functionality...")
    
    try:
        from src.training.steps.market_analysis.enhanced_pipeline_decorators import (
            data_formatting,
            data_analysis_protection,
            data_access_protection,
            comprehensive_pipeline_protection,
        )
        
        # Test data formatting decorator
        @data_formatting(
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            validation_rules={'no_nan_ratio': {'max_ratio': 0.1}}
        )
        def test_data_function(data):
            return True
        
        print("✅ Data formatting decorator applied successfully")
        
        # Test data analysis protection decorator
        @data_analysis_protection(
            max_memory_mb=1000,
            max_execution_time=300,
            allowed_operations=['test_operation']
        )
        def test_analysis_function():
            return True
        
        print("✅ Data analysis protection decorator applied successfully")
        
        # Test data access protection decorator
        @data_access_protection(
            allowed_paths=['data_cache/*'],
            audit_access=True
        )
        def test_access_function():
            return True
        
        print("✅ Data access protection decorator applied successfully")
        
        # Test comprehensive protection decorator
        @comprehensive_pipeline_protection(
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            max_memory_mb=1000,
            max_execution_time=300,
            allowed_paths=['data_cache/*'],
            audit_access=True
        )
        def test_comprehensive_function():
            return True
        
        print("✅ Comprehensive pipeline protection decorator applied successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Decorator functionality test failed: {e}")
        return False

def test_pipeline_structure():
    """Test the overall pipeline structure."""
    print("\n🧪 Testing pipeline structure...")
    
    try:
        from src.training.steps.market_analysis.enhanced_market_analysis_orchestrator import (
            MarketAnalysisPipelineOrchestrator,
        )
        
        orchestrator = MarketAnalysisPipelineOrchestrator()
        
        # Test step configurations
        step_configs = orchestrator.step_configs
        expected_steps = [
            'hmm_clustering',
            'regime_splitting', 
            'labeling',
            'feature_engineering',
            'matrix_operations',
            'feature_selection'
        ]
        
        for step in expected_steps:
            if step in step_configs:
                print(f"✅ Step '{step}' configuration found")
            else:
                print(f"❌ Step '{step}' configuration missing")
                return False
        
        # Test pipeline state tracking
        pipeline_state = orchestrator.pipeline_state
        expected_state_keys = [
            'current_step',
            'completed_steps',
            'failed_steps',
            'start_time',
            'end_time',
            'correlation_id'
        ]
        
        for key in expected_state_keys:
            if key in pipeline_state:
                print(f"✅ Pipeline state key '{key}' found")
            else:
                print(f"❌ Pipeline state key '{key}' missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline structure test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Enhanced Market Analysis Pipeline")
    print("=" * 60)
    
    tests = [
        test_pipeline_imports,
        test_orchestrator_initialization,
        test_validator_initialization,
        test_decorator_functionality,
        test_pipeline_structure,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced pipeline is ready.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)