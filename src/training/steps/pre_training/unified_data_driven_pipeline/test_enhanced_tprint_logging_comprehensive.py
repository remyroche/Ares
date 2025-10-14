#!/usr/bin/env python3
"""
Comprehensive test script for enhanced tprint logging and silent failure prevention
in the UnifiedDataDrivenPipeline.

This script tests:
1. Extensive tprint logging throughout the pipeline
2. Silent failure prevention with proper error handling
3. Input validation and error reporting
4. Comprehensive logging at all levels
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import traceback
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../..'))

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError as e:
    print(f"❌ Tprint utilities not available: {e}")
    TPRINT_AVAILABLE = False

try:
    from src.training.steps.pre_training.unified_data_driven_pipeline import (
        UnifiedDataDrivenPipeline,
        create_unified_pipeline,
        process_with_unified_pipeline,
        UnifiedPipelineConfig,
        create_default_config
    )
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"❌ Pipeline not available: {e}")
    PIPELINE_AVAILABLE = False

def create_test_data(rows: int = 1000) -> pd.DataFrame:
    """Create test data for pipeline testing."""
    np.random.seed(42)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': np.random.uniform(100, 200, rows),
        'high': np.random.uniform(100, 200, rows),
        'low': np.random.uniform(100, 200, rows),
        'close': np.random.uniform(100, 200, rows),
        'volume': np.random.uniform(1000, 10000, rows)
    })
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Add timestamp index
    data.index = pd.date_range('2023-01-01', periods=rows, freq='15T')
    
    return data

def create_test_targets(data: pd.DataFrame) -> pd.Series:
    """Create test targets for supervised learning."""
    # Simple price change targets
    targets = data['close'].pct_change().dropna()
    return targets

def test_pipeline_initialization():
    """Test pipeline initialization with comprehensive logging."""
    tprint_info("🧪 Testing pipeline initialization with comprehensive logging")
    
    try:
        # Test with default config
        tprint_info("📋 Testing with default configuration")
        config = create_default_config()
        pipeline = create_unified_pipeline(config)
        
        if pipeline is None:
            tprint_error("❌ Pipeline creation returned None")
            return False
        
        tprint_success("✅ Pipeline initialization successful with default config")
        
        # Test with custom config
        tprint_info("📋 Testing with custom configuration")
        custom_config = UnifiedPipelineConfig(
            labeling_type="analyst",
            enable_advanced_features=True,
            enable_gpu_optimizations=True
        )
        custom_pipeline = create_unified_pipeline(custom_config)
        
        if custom_pipeline is None:
            tprint_error("❌ Custom pipeline creation returned None")
            return False
        
        tprint_success("✅ Pipeline initialization successful with custom config")
        
        return True
        
    except Exception as e:
        tprint_error(f"❌ Pipeline initialization failed: {e}")
        tprint_error(f"❌ Error type: {type(e).__name__}")
        tprint_error(f"❌ Traceback: {traceback.format_exc()}")
        return False

def test_input_validation():
    """Test input validation and error handling."""
    tprint_info("🧪 Testing input validation and error handling")
    
    try:
        config = create_default_config()
        pipeline = create_unified_pipeline(config)
        
        # Test with None data
        tprint_info("📋 Testing with None data")
        try:
            result = pipeline.process(None)
            tprint_error("❌ Pipeline should have failed with None data")
            return False
        except Exception as e:
            tprint_success(f"✅ Pipeline correctly failed with None data: {type(e).__name__}")
        
        # Test with empty DataFrame
        tprint_info("📋 Testing with empty DataFrame")
        try:
            empty_data = pd.DataFrame()
            result = pipeline.process(empty_data)
            tprint_error("❌ Pipeline should have failed with empty DataFrame")
            return False
        except Exception as e:
            tprint_success(f"✅ Pipeline correctly failed with empty DataFrame: {type(e).__name__}")
        
        # Test with invalid data type
        tprint_info("📋 Testing with invalid data type")
        try:
            result = pipeline.process("invalid_data")
            tprint_error("❌ Pipeline should have failed with invalid data type")
            return False
        except Exception as e:
            tprint_success(f"✅ Pipeline correctly failed with invalid data type: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        tprint_error(f"❌ Input validation test failed: {e}")
        tprint_error(f"❌ Error type: {type(e).__name__}")
        tprint_error(f"❌ Traceback: {traceback.format_exc()}")
        return False

def test_period_optimization_logging():
    """Test period optimization with comprehensive logging."""
    tprint_info("🧪 Testing period optimization with comprehensive logging")
    
    try:
        config = create_default_config()
        pipeline = create_unified_pipeline(config)
        
        # Create test data
        data = create_test_data(500)
        tprint_info(f"📊 Created test data: {data.shape}")
        
        # Test period optimization method directly
        tprint_info("🔍 Testing _enhanced_period_optimization method")
        try:
            period_results = pipeline._enhanced_period_optimization(data, "15m")
            
            if period_results is None:
                tprint_error("❌ Period optimization returned None")
                return False
            
            if not isinstance(period_results, dict):
                tprint_error(f"❌ Period optimization returned invalid type: {type(period_results)}")
                return False
            
            if 'optimal_periods' not in period_results:
                tprint_error("❌ Period optimization missing 'optimal_periods' key")
                return False
            
            tprint_success(f"✅ Period optimization completed: {len(period_results['optimal_periods'])} optimal periods")
            tprint_info(f"📋 Optimal periods: {period_results['optimal_periods']}")
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Period optimization failed: {e}")
            tprint_error(f"❌ Error type: {type(e).__name__}")
            tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            return False
        
    except Exception as e:
        tprint_error(f"❌ Period optimization test failed: {e}")
        tprint_error(f"❌ Error type: {type(e).__name__}")
        tprint_error(f"❌ Traceback: {traceback.format_exc()}")
        return False

def test_feature_selection_logging():
    """Test feature selection with comprehensive logging."""
    tprint_info("🧪 Testing feature selection with comprehensive logging")
    
    try:
        config = create_default_config()
        pipeline = create_unified_pipeline(config)
        
        # Create test data
        data = create_test_data(500)
        targets = create_test_targets(data)
        tprint_info(f"📊 Created test data: {data.shape}, targets: {targets.shape}")
        
        # Test feature selection method directly
        tprint_info("🔍 Testing _advanced_feature_selection method")
        try:
            selection_results = pipeline._advanced_feature_selection(data, targets)
            
            if selection_results is None:
                tprint_error("❌ Feature selection returned None")
                return False
            
            if not hasattr(selection_results, 'success'):
                tprint_error("❌ Feature selection result missing 'success' attribute")
                return False
            
            if not selection_results.success:
                tprint_error(f"❌ Feature selection failed: {getattr(selection_results, 'error_message', 'Unknown error')}")
                return False
            
            if not hasattr(selection_results, 'selected_features'):
                tprint_error("❌ Feature selection result missing 'selected_features' attribute")
                return False
            
            tprint_success(f"✅ Feature selection completed: {len(selection_results.selected_features)} features selected")
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Feature selection failed: {e}")
            tprint_error(f"❌ Error type: {type(e).__name__}")
            tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            return False
        
    except Exception as e:
        tprint_error(f"❌ Feature selection test failed: {e}")
        tprint_error(f"❌ Error type: {type(e).__name__}")
        tprint_error(f"❌ Traceback: {traceback.format_exc()}")
        return False

def test_feature_generation_logging():
    """Test feature generation with comprehensive logging."""
    tprint_info("🧪 Testing feature generation with comprehensive logging")
    
    try:
        config = create_default_config()
        pipeline = create_unified_pipeline(config)
        
        # Create test data
        data = create_test_data(500)
        tprint_info(f"📊 Created test data: {data.shape}")
        
        # Create mock selection result
        class MockSelectionResult:
            def __init__(self):
                self.success = True
                self.selected_features = [
                    type('Feature', (), {'feature_name': f'feature_{i}'})() 
                    for i in range(10)
                ]
        
        selection_result = MockSelectionResult()
        
        # Test feature generation method directly
        tprint_info("🔍 Testing _generate_selected_features method")
        try:
            features_df = pipeline._generate_selected_features(data, selection_result)
            
            if features_df is None:
                tprint_error("❌ Feature generation returned None")
                return False
            
            if not isinstance(features_df, pd.DataFrame):
                tprint_error(f"❌ Feature generation returned invalid type: {type(features_df)}")
                return False
            
            if features_df.empty:
                tprint_warning("⚠️ Feature generation returned empty DataFrame")
            else:
                tprint_success(f"✅ Feature generation completed: {features_df.shape}")
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Feature generation failed: {e}")
            tprint_error(f"❌ Error type: {type(e).__name__}")
            tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            return False
        
    except Exception as e:
        tprint_error(f"❌ Feature generation test failed: {e}")
        tprint_error(f"❌ Error type: {type(e).__name__}")
        tprint_error(f"❌ Traceback: {traceback.format_exc()}")
        return False

def test_interaction_generation_logging():
    """Test interaction generation with comprehensive logging."""
    tprint_info("🧪 Testing interaction generation with comprehensive logging")
    
    try:
        config = create_default_config()
        pipeline = create_unified_pipeline(config)
        
        # Create test data
        data = create_test_data(500)
        features_df = data.copy()  # Use data as features for simplicity
        targets = create_test_targets(data)
        tprint_info(f"📊 Created test data: features={features_df.shape}, targets={targets.shape}")
        
        # Test interaction generation method directly
        tprint_info("🔍 Testing _enhanced_interaction_generation method")
        try:
            interactions = pipeline._enhanced_interaction_generation(features_df, targets)
            
            if interactions is None:
                tprint_error("❌ Interaction generation returned None")
                return False
            
            if not isinstance(interactions, list):
                tprint_error(f"❌ Interaction generation returned invalid type: {type(interactions)}")
                return False
            
            tprint_success(f"✅ Interaction generation completed: {len(interactions)} interactions")
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Interaction generation failed: {e}")
            tprint_error(f"❌ Error type: {type(e).__name__}")
            tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            return False
        
    except Exception as e:
        tprint_error(f"❌ Interaction generation test failed: {e}")
        tprint_error(f"❌ Error type: {type(e).__name__}")
        tprint_error(f"❌ Traceback: {traceback.format_exc()}")
        return False

def test_silent_failure_prevention():
    """Test that silent failures are prevented."""
    tprint_info("🧪 Testing silent failure prevention")
    
    try:
        config = create_default_config()
        pipeline = create_unified_pipeline(config)
        
        # Test with invalid statistical analysis data
        tprint_info("📋 Testing with invalid statistical analysis data")
        try:
            invalid_analysis = None
            result = pipeline._combine_period_scores_safe(invalid_analysis, None)
            tprint_error("❌ Should have failed with None statistical analysis")
            return False
        except Exception as e:
            tprint_success(f"✅ Correctly failed with None statistical analysis: {type(e).__name__}")
        
        # Test with invalid combined scores
        tprint_info("📋 Testing with invalid combined scores")
        try:
            invalid_scores = None
            result = pipeline._select_optimal_periods_safe(invalid_scores)
            tprint_error("❌ Should have failed with None combined scores")
            return False
        except Exception as e:
            tprint_success(f"✅ Correctly failed with None combined scores: {type(e).__name__}")
        
        # Test with empty combined scores
        tprint_info("📋 Testing with empty combined scores")
        try:
            empty_scores = {}
            result = pipeline._select_optimal_periods_safe(empty_scores)
            tprint_error("❌ Should have failed with empty combined scores")
            return False
        except Exception as e:
            tprint_success(f"✅ Correctly failed with empty combined scores: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        tprint_error(f"❌ Silent failure prevention test failed: {e}")
        tprint_error(f"❌ Error type: {type(e).__name__}")
        tprint_error(f"❌ Traceback: {traceback.format_exc()}")
        return False

def test_full_pipeline_logging():
    """Test full pipeline with comprehensive logging."""
    tprint_info("🧪 Testing full pipeline with comprehensive logging")
    
    try:
        config = create_default_config()
        
        # Create test data
        data = create_test_data(1000)
        targets = create_test_targets(data)
        tprint_info(f"📊 Created test data: {data.shape}, targets: {targets.shape}")
        
        # Test full pipeline processing
        tprint_info("🚀 Testing full pipeline processing")
        try:
            result = process_with_unified_pipeline(
                data=data,
                targets=targets,
                timeframe="15m",
                config=config
            )
            
            if result is None:
                tprint_error("❌ Full pipeline processing returned None")
                return False
            
            tprint_success("✅ Full pipeline processing completed successfully")
            tprint_info(f"📋 Result type: {type(result)}")
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Full pipeline processing failed: {e}")
            tprint_error(f"❌ Error type: {type(e).__name__}")
            tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            return False
        
    except Exception as e:
        tprint_error(f"❌ Full pipeline test failed: {e}")
        tprint_error(f"❌ Error type: {type(e).__name__}")
        tprint_error(f"❌ Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all comprehensive tests."""
    tprint_info("🚀 Starting comprehensive tprint logging and silent failure prevention tests")
    tprint_info(f"⏰ Test started at: {datetime.now()}")
    
    if not TPRINT_AVAILABLE:
        tprint_error("❌ Tprint utilities not available - cannot run tests")
        return False
    
    if not PIPELINE_AVAILABLE:
        tprint_error("❌ Pipeline not available - cannot run tests")
        return False
    
    # Test results
    test_results = {}
    
    # Run all tests
    tests = [
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Input Validation", test_input_validation),
        ("Period Optimization Logging", test_period_optimization_logging),
        ("Feature Selection Logging", test_feature_selection_logging),
        ("Feature Generation Logging", test_feature_generation_logging),
        ("Interaction Generation Logging", test_interaction_generation_logging),
        ("Silent Failure Prevention", test_silent_failure_prevention),
        ("Full Pipeline Logging", test_full_pipeline_logging)
    ]
    
    for test_name, test_func in tests:
        tprint_info(f"\n{'='*60}")
        tprint_info(f"🧪 Running test: {test_name}")
        tprint_info(f"{'='*60}")
        
        try:
            result = test_func()
            test_results[test_name] = result
            
            if result:
                tprint_success(f"✅ {test_name} PASSED")
            else:
                tprint_error(f"❌ {test_name} FAILED")
                
        except Exception as e:
            tprint_error(f"❌ {test_name} FAILED with exception: {e}")
            tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            test_results[test_name] = False
    
    # Summary
    tprint_info(f"\n{'='*60}")
    tprint_info("📋 TEST SUMMARY")
    tprint_info(f"{'='*60}")
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        tprint_info(f"  {test_name}: {status}")
    
    tprint_info(f"\n📊 Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        tprint_success("🎉 All tests passed! Enhanced tprint logging and silent failure prevention working correctly.")
        return True
    else:
        tprint_error(f"❌ {total_tests - passed_tests} tests failed. Please check the logs above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)