#!/usr/bin/env python3
"""
Test script to verify VectorBT integration in feature selection pipeline.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_vectorbt_integration():
    """Test VectorBT integration in feature selection pipeline."""
    print("🧪 Testing VectorBT integration in feature selection pipeline...")
    
    try:
        # Import the feature selection pipeline
        from src.training.steps.pre_training.final_feature_selection_pipeline import (
            MultiStageFeatureSelector, 
            FeatureSelectionConfig,
            VECTORBT_AVAILABLE
        )
        
        print(f"✅ VectorBT Available: {VECTORBT_AVAILABLE}")
        
        # Create test data
        np.random.seed(42)
        n_samples = 1000
        n_features = 150  # Large enough to trigger VectorBT optimizations
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randn(n_samples))
        
        print(f"📊 Test data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Create feature selection configuration
        config = FeatureSelectionConfig(
            initial_features=n_features,
            stage_1_target=120,
            stage_2_target=100,
            stage_3_target=80,
            target_features=60,
            verbose=True
        )
        
        # Initialize selector
        selector = MultiStageFeatureSelector(config)
        print("✅ MultiStageFeatureSelector initialized")
        
        # Test VectorBT status display
        print("\n🔍 VectorBT Status:")
        selector._display_vectorbt_status()
        
        # Test memory optimization
        if VECTORBT_AVAILABLE:
            print("\n🧠 Testing VectorBT memory optimization...")
            X_optimized = selector._vectorbt_memory_optimization(X, "test")
            print(f"✅ Memory optimization completed: {X.shape} → {X_optimized.shape}")
        
        # Test correlation analysis
        print("\n📊 Testing VectorBT correlation analysis...")
        corr_matrix = selector._vectorized_correlation_analysis(X)
        print(f"✅ Correlation matrix computed: {corr_matrix.shape}")
        
        # Test feature importance
        print("\n🎯 Testing VectorBT feature importance...")
        importance = selector._vectorized_feature_importance(X, y, 'rf')
        print(f"✅ Feature importance computed: {len(importance)} features")
        
        # Test ensemble scoring
        print("\n🔄 Testing VectorBT ensemble scoring...")
        ensemble_scores = selector.ensemble_scores_cv_parallel(X, y, rs=42, task='reg', n_splits=3)
        print(f"✅ Ensemble scores computed: {len(ensemble_scores)} features")
        
        # Test mRMR calculation
        print("\n🔍 Testing VectorBT mRMR calculation...")
        mrmr_scores = selector._calculate_mrmr_mid_vectorized(X, y)
        print(f"✅ mRMR scores computed: {len(mrmr_scores)} features")
        
        # Test stability scoring
        print("\n⚖️ Testing VectorBT stability scoring...")
        stability_scores = selector.stability_scores_vectorized(
            X, y, 
            lambda X_, y_: selector.ensemble_scores_cv_parallel(X_, y_, rs=42, task='reg', n_splits=3),
            rs=42, n_boot=5
        )
        print(f"✅ Stability scores computed: {len(stability_scores)} features")
        
        print("\n🎉 All VectorBT integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ VectorBT integration test failed: {e}")
        import traceback
        print(f"🔍 Error details: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_vectorbt_integration()
    sys.exit(0 if success else 1)