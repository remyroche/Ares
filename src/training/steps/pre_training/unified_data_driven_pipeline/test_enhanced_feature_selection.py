#!/usr/bin/env python3
"""
Test script for enhanced feature selection integration in UnifiedDataDrivenPipeline.

This script tests the multi-stage feature selection with lightweight screening
and advanced selection methods (mRMR, LASSO, RFE, etc.).
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.advanced_feature_selection import (
        AdvancedFeatureSelector, FeatureSelectionConfig
    )
    from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline import (
        create_unified_pipeline, UnifiedPipelineConfig
    )
    from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def create_test_data(n_samples=1000, n_features=200):
    """Create test data for feature selection."""
    np.random.seed(42)
    
    # Create synthetic financial data
    data = {}
    
    # Price-based features
    for i in range(50):
        data[f'price_feature_{i}'] = np.random.randn(n_samples).cumsum()
    
    # Momentum features
    for i in range(30):
        data[f'momentum_feature_{i}'] = np.random.randn(n_samples)
    
    # Volatility features
    for i in range(25):
        data[f'volatility_feature_{i}'] = np.abs(np.random.randn(n_samples))
    
    # Volume features
    for i in range(20):
        data[f'volume_feature_{i}'] = np.random.exponential(1, n_samples)
    
    # Technical indicators
    for i in range(30):
        data[f'technical_feature_{i}'] = np.random.randn(n_samples)
    
    # Cross-timeframe features
    for i in range(25):
        data[f'htf_feature_{i}'] = np.random.randn(n_samples)
    
    # Microstructure features
    for i in range(20):
        data[f'microstructure_feature_{i}'] = np.random.randn(n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create target variable (returns)
    target = np.random.randn(n_samples) * 0.01
    
    return df, target


def test_lightweight_screening():
    """Test lightweight screening methods."""
    tprint_info("🧪 Testing lightweight screening methods")
    
    try:
        # Create test data
        data, target = create_test_data(n_samples=500, n_features=100)
        
        # Configure for lightweight screening only
        config = FeatureSelectionConfig(
            enable_multi_stage_selection=True,
            enable_lightweight_screening=True,
            screening_methods=['variance', 'correlation', 'mutual_info'],
            final_selection_methods=[],  # No advanced methods
            max_screening_features=50,
            variance_threshold=0.1,
            screening_threshold=0.05,
            mutual_info_threshold=0.01
        )
        
        # Create selector
        selector = AdvancedFeatureSelector(config)
        
        # Test screening
        screened_features = selector._lightweight_screening(data, target)
        
        tprint_success(f"✅ Lightweight screening completed: {len(screened_features)} features selected")
        tprint_info(f"📊 Original features: {len(data.columns)}")
        tprint_info(f"📊 Screened features: {len(screened_features)}")
        
        # Test individual screening methods
        variance_features = selector._variance_screening(data)
        correlation_features = selector._correlation_screening(data, target)
        mi_features = selector._mutual_info_screening(data, target)
        
        tprint_info(f"📊 Variance screening: {len(variance_features)} features")
        tprint_info(f"📊 Correlation screening: {len(correlation_features)} features")
        tprint_info(f"📊 Mutual info screening: {len(mi_features)} features")
        
        return True
        
    except Exception as e:
        tprint_error(f"❌ Lightweight screening test failed: {e}")
        return False


def test_advanced_selection_methods():
    """Test advanced selection methods."""
    tprint_info("🧪 Testing advanced selection methods")
    
    try:
        # Create test data
        data, target = create_test_data(n_samples=500, n_features=50)
        
        # Configure for advanced selection only
        config = FeatureSelectionConfig(
            enable_multi_stage_selection=True,
            enable_lightweight_screening=False,  # Skip screening
            screening_methods=[],
            final_selection_methods=['mrmr', 'lasso', 'rfe'],
            final_selection_count=20
        )
        
        # Create selector
        selector = AdvancedFeatureSelector(config)
        
        # Test advanced selection
        selected_features = selector._advanced_selection_methods(data, target)
        
        tprint_success(f"✅ Advanced selection completed: {len(selected_features)} features selected")
        tprint_info(f"📊 Original features: {len(data.columns)}")
        tprint_info(f"📊 Selected features: {len(selected_features)}")
        
        return True
        
    except Exception as e:
        tprint_error(f"❌ Advanced selection test failed: {e}")
        return False


def test_multi_stage_selection():
    """Test complete multi-stage selection."""
    tprint_info("🧪 Testing complete multi-stage selection")
    
    try:
        # Create test data
        data, target = create_test_data(n_samples=1000, n_features=150)
        
        # Configure for multi-stage selection
        config = FeatureSelectionConfig(
            enable_multi_stage_selection=True,
            enable_lightweight_screening=True,
            screening_methods=['variance', 'correlation', 'mutual_info'],
            final_selection_methods=['mrmr', 'lasso', 'rfe'],
            max_screening_features=80,
            final_selection_count=30,
            variance_threshold=0.1,
            screening_threshold=0.05,
            mutual_info_threshold=0.01
        )
        
        # Create selector
        selector = AdvancedFeatureSelector(config)
        
        # Test complete multi-stage selection
        result = selector.select_features(data, target)
        
        if result.success:
            tprint_success(f"✅ Multi-stage selection completed: {len(result.selected_features)} features selected")
            tprint_info(f"📊 Original features: {len(data.columns)}")
            tprint_info(f"📊 Selected features: {len(result.selected_features)}")
            tprint_info(f"📊 Quality metrics: {result.quality_metrics}")
            tprint_info(f"📊 Diversity metrics: {result.diversity_metrics}")
            tprint_info(f"📊 Stability metrics: {result.stability_metrics}")
            return True
        else:
            tprint_error(f"❌ Multi-stage selection failed: {result.error_message}")
            return False
        
    except Exception as e:
        tprint_error(f"❌ Multi-stage selection test failed: {e}")
        return False


def test_pipeline_integration():
    """Test integration with the consolidated pipeline."""
    tprint_info("🧪 Testing pipeline integration")
    
    try:
        # Create test data
        data, target = create_test_data(n_samples=500, n_features=100)
        
        # Create pipeline config
        config = UnifiedPipelineConfig()
        
        # Create pipeline
        pipeline = create_unified_pipeline(config)
        
        # Test pipeline processing
        result = pipeline.process(data, target, list(data.columns), '1min')
        
        if result and hasattr(result, 'selected_features'):
            tprint_success(f"✅ Pipeline integration completed: {len(result.selected_features)} features selected")
            tprint_info(f"📊 Original features: {len(data.columns)}")
            tprint_info(f"📊 Selected features: {len(result.selected_features)}")
            return True
        else:
            tprint_error("❌ Pipeline integration failed: No result or selected features")
            return False
        
    except Exception as e:
        tprint_error(f"❌ Pipeline integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    tprint_info("🚀 Starting enhanced feature selection tests")
    
    tests = [
        ("Lightweight Screening", test_lightweight_screening),
        ("Advanced Selection Methods", test_advanced_selection_methods),
        ("Multi-stage Selection", test_multi_stage_selection),
        ("Pipeline Integration", test_pipeline_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        tprint_info(f"\n{'='*50}")
        tprint_info(f"Running test: {test_name}")
        tprint_info(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                tprint_success(f"✅ {test_name} passed")
            else:
                tprint_error(f"❌ {test_name} failed")
                
        except Exception as e:
            tprint_error(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    tprint_info(f"\n{'='*50}")
    tprint_info("TEST SUMMARY")
    tprint_info(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        tprint_info(f"{test_name}: {status}")
    
    tprint_info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        tprint_success("🎉 All tests passed! Enhanced feature selection is working correctly.")
        return 0
    else:
        tprint_error(f"⚠️ {total - passed} tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)