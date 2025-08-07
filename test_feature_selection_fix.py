#!/usr/bin/env python3
"""
Test script to verify the feature selection fix.
This script tests the VectorizedFeatureSelector to ensure it doesn't remove all features.
"""

import pandas as pd
import numpy as np
import asyncio
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.training.steps.vectorized_labelling_orchestrator import VectorizedFeatureSelector
from src.utils.logger import system_logger

async def test_feature_selection():
    """Test the feature selection with various scenarios."""
    
    # Test configuration
    config = {
        "feature_selection": {
            "vif_threshold": 10.0,
            "mutual_info_threshold": 0.001,
            "lightgbm_importance_threshold": 0.001,
            "min_features_to_keep": 5,
            "correlation_threshold": 0.98,
            "max_removal_percentage": 0.3,
            "enable_safety_checks": True,
            "return_original_on_failure": True
        }
    }
    
    # Create test data with various scenarios
    test_cases = [
        {
            "name": "Normal case - many features",
            "data": pd.DataFrame({
                'feature_1': np.random.randn(100),
                'feature_2': np.random.randn(100),
                'feature_3': np.random.randn(100),
                'feature_4': np.random.randn(100),
                'feature_5': np.random.randn(100),
                'feature_6': np.random.randn(100),
                'feature_7': np.random.randn(100),
                'feature_8': np.random.randn(100),
                'feature_9': np.random.randn(100),
                'feature_10': np.random.randn(100),
            }),
            "labels": np.random.choice([0, 1], size=100)
        },
        {
            "name": "Few features case",
            "data": pd.DataFrame({
                'feature_1': np.random.randn(100),
                'feature_2': np.random.randn(100),
                'feature_3': np.random.randn(100),
                'feature_4': np.random.randn(100),
                'feature_5': np.random.randn(100),
            }),
            "labels": np.random.choice([0, 1], size=100)
        },
        {
            "name": "Correlated features case",
            "data": pd.DataFrame({
                'feature_1': np.random.randn(100),
                'feature_2': np.random.randn(100) * 0.1 + np.random.randn(100) * 0.9,  # Highly correlated with feature_1
                'feature_3': np.random.randn(100),
                'feature_4': np.random.randn(100),
                'feature_5': np.random.randn(100),
                'feature_6': np.random.randn(100),
            }),
            "labels": np.random.choice([0, 1], size=100)
        },
        {
            "name": "Constant features case",
            "data": pd.DataFrame({
                'feature_1': np.random.randn(100),
                'feature_2': np.random.randn(100),
                'feature_3': np.ones(100),  # Constant feature
                'feature_4': np.random.randn(100),
                'feature_5': np.random.randn(100),
            }),
            "labels": np.random.choice([0, 1], size=100)
        }
    ]
    
    feature_selector = VectorizedFeatureSelector(config)
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*50}")
        print(f"Test Case {i+1}: {test_case['name']}")
        print(f"{'='*50}")
        
        try:
            # Test feature selection
            result = await feature_selector.select_optimal_features(
                test_case['data'], 
                test_case['labels']
            )
            
            initial_count = len(test_case['data'].columns)
            final_count = len(result.columns)
            
            print(f"Initial features: {initial_count}")
            print(f"Final features: {final_count}")
            print(f"Features removed: {initial_count - final_count}")
            
            if final_count == 0:
                print("❌ ERROR: All features were removed!")
                return False
            elif final_count < 5:
                print("⚠️  WARNING: Very few features remaining")
            else:
                print("✅ SUCCESS: Feature selection completed successfully")
                
        except Exception as e:
            print(f"❌ ERROR in test case {i+1}: {e}")
            return False
    
    print(f"\n{'='*50}")
    print("All tests completed!")
    print(f"{'='*50}")
    return True

if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_feature_selection())
    if success:
        print("✅ All feature selection tests passed!")
        sys.exit(0)
    else:
        print("❌ Some feature selection tests failed!")
        sys.exit(1) 