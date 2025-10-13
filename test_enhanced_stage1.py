#!/usr/bin/env python3
"""
Test script for the enhanced Stage 1 feature selection with HSIC and distance correlation.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.steps.pre_training.feature_selection.core.multi_stage_pipeline import (
    MultiStageFeatureSelectionPipeline, FeatureSelectionConfig
)
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error

def create_test_data(n_samples=1000, n_features=50):
    """Create test data with linear and nonlinear relationships."""
    np.random.seed(42)
    
    # Create features with different types of relationships to target
    X = pd.DataFrame()
    
    # Linear relationship features
    for i in range(10):
        X[f'linear_feature_{i}'] = np.random.randn(n_samples)
    
    # Nonlinear relationship features (quadratic, exponential, etc.)
    for i in range(10):
        base = np.random.randn(n_samples)
        X[f'quadratic_feature_{i}'] = base ** 2 + 0.1 * np.random.randn(n_samples)
        X[f'exponential_feature_{i}'] = np.exp(base * 0.5) + 0.1 * np.random.randn(n_samples)
        X[f'sine_feature_{i}'] = np.sin(base * 2) + 0.1 * np.random.randn(n_samples)
    
    # Random noise features
    for i in range(20):
        X[f'noise_feature_{i}'] = np.random.randn(n_samples)
    
    # Create target with mixed linear and nonlinear relationships
    y = (
        0.3 * X['linear_feature_0'] +  # Strong linear
        0.2 * X['linear_feature_1'] +  # Medium linear
        0.1 * X['quadratic_feature_0'] +  # Nonlinear
        0.1 * X['exponential_feature_0'] +  # Nonlinear
        0.1 * X['sine_feature_0'] +  # Nonlinear
        0.1 * np.random.randn(n_samples)  # Noise
    )
    
    return X, y

def test_enhanced_stage1():
    """Test the enhanced Stage 1 implementation."""
    tprint("🧪 Testing Enhanced Stage 1 Feature Selection")
    
    try:
        # Create test data
        tprint_info("📊 Creating test data...")
        X, y = create_test_data(n_samples=500, n_features=40)
        tprint_info(f"   📊 Data shape: {X.shape}, Target shape: {y.shape}")
        
        # Create configuration
        config = FeatureSelectionConfig()
        config.target_features = 20
        config.stage1_mrmr_weight = 0.5
        config.stage1_distance_correlation_weight = 0.3
        config.stage1_hsic_weight = 0.2
        config.hsic_enable_subsampling = True
        config.hsic_sample_size = 200
        config.distance_correlation_enable_subsampling = True
        config.distance_correlation_sample_size = 200
        
        # Initialize pipeline
        tprint_info("🚀 Initializing enhanced pipeline...")
        pipeline = MultiStageFeatureSelectionPipeline(config)
        
        # Test Stage 1 only
        tprint_info("🔍 Testing Stage 1: Enhanced Multi-Method Scoring...")
        stage1_result = pipeline._stage_1_enhanced_multi_method_scoring(X, y)
        
        # Display results
        tprint_success("✅ Stage 1 completed successfully!")
        tprint_info(f"   📊 Selected features: {len(stage1_result['selected_features'])}")
        tprint_info(f"   📊 Method: {stage1_result['method']}")
        
        # Show score distributions
        mrmr_scores = stage1_result['mrmr_scores']
        distance_corr_scores = stage1_result['distance_correlation_scores']
        hsic_scores = stage1_result['hsic_scores']
        combined_scores = stage1_result['combined_scores']
        
        tprint_info("📊 Score Statistics:")
        tprint_info(f"   mRMR scores - Mean: {np.mean(list(mrmr_scores.values())):.4f}, Max: {np.max(list(mrmr_scores.values())):.4f}")
        tprint_info(f"   Distance Correlation scores - Mean: {np.mean(list(distance_corr_scores.values())):.4f}, Max: {np.max(list(distance_corr_scores.values())):.4f}")
        tprint_info(f"   HSIC scores - Mean: {np.mean(list(hsic_scores.values())):.4f}, Max: {np.max(list(hsic_scores.values())):.4f}")
        tprint_info(f"   Combined scores - Mean: {np.mean(list(combined_scores.values())):.4f}, Max: {np.max(list(combined_scores.values())):.4f}")
        
        # Show top selected features
        tprint_info("🏆 Top 10 Selected Features:")
        sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (feature, score) in enumerate(sorted_features, 1):
            tprint_info(f"   {i:2d}. {feature}: {score:.4f}")
        
        # Test individual methods
        tprint_info("🔍 Testing individual methods...")
        
        # Test distance correlation
        try:
            distance_corr_result = pipeline._calculate_distance_correlation_scores(X, y)
            tprint_success(f"   ✅ Distance correlation: {len(distance_corr_result)} scores calculated")
        except Exception as e:
            tprint_error(f"   ❌ Distance correlation failed: {e}")
        
        # Test HSIC
        try:
            hsic_result = pipeline._calculate_hsic_scores(X, y)
            tprint_success(f"   ✅ HSIC: {len(hsic_result)} scores calculated")
        except Exception as e:
            tprint_error(f"   ❌ HSIC failed: {e}")
        
        # Test mRMR (should work as before)
        try:
            mrmr_result = pipeline._calculate_mrmr_scores(X, y)
            tprint_success(f"   ✅ mRMR: {len(mrmr_result)} scores calculated")
        except Exception as e:
            tprint_error(f"   ❌ mRMR failed: {e}")
        
        tprint_success("🎉 Enhanced Stage 1 test completed successfully!")
        return True
        
    except Exception as e:
        tprint_error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_stage1()
    if success:
        tprint_success("🎉 All tests passed!")
        sys.exit(0)
    else:
        tprint_error("❌ Tests failed!")
        sys.exit(1)