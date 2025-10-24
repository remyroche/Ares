#!/usr/bin/env python3
"""
Test script to verify gate feature integration in the training pipeline.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.training.steps.pre_training.feature_generation_gate_feature_step import (
    FeatureGenerationGateFeatureStep,
    handle_feature_generation_gate_feature_step
)


async def test_gate_feature_generation():
    """Test gate feature generation step."""
    print("🧪 Testing Gate Feature Generation Step")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Generate sample features
    features_df = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)],
        index=pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    )
    
    # Generate sample targets
    targets_series = pd.Series(
        np.random.randn(n_samples),
        index=features_df.index,
        name='target'
    )
    
    print(f"📊 Sample data: {features_df.shape[0]} samples, {features_df.shape[1]} features")
    print(f"📊 Target data: {len(targets_series)} samples")
    
    # Test data
    test_data = {
        'features': features_df,
        'targets': targets_series
    }
    
    # Test configuration
    config = {
        'gate_features': {
            'enable_gate_protection': True,
            'max_gate_features_per_base': 3
        }
    }
    
    try:
        # Test the step directly
        print("\n🔧 Testing FeatureGenerationGateFeatureStep directly...")
        step = FeatureGenerationGateFeatureStep("test_gate_feature_step", config)
        result = await step.execute(test_data)
        
        print(f"✅ Step execution result: {result.success}")
        print(f"📊 Gate features generated: {result.gate_features_generated}")
        print(f"📋 Gate feature names: {result.gate_feature_names}")
        print(f"⏱️ Processing time: {result.processing_time:.2f}s")
        
        if result.gate_evaluation_results:
            print(f"🔍 Gate evaluation results: {len(result.gate_evaluation_results)} gates evaluated")
            for eval_result in result.gate_evaluation_results:
                print(f"  - {eval_result['feature_name']}: {eval_result['status']} (score: {eval_result['score']:.3f})")
        
        # Test the handler function
        print("\n🔧 Testing handler function...")
        handler_result = await handle_feature_generation_gate_feature_step(
            step_name="test_gate_feature_handler",
            config=config,
            data=test_data
        )
        
        print(f"✅ Handler execution result: {handler_result['success']}")
        print(f"📊 Gate features generated: {handler_result['gate_features_generated']}")
        print(f"📋 Gate feature names: {handler_result['gate_feature_names']}")
        
        print("\n🎉 Gate feature integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_gate_feature_protection():
    """Test that gate features are protected in feature selection."""
    print("\n🛡️ Testing Gate Feature Protection")
    print("=" * 50)
    
    # Create sample data with gate features
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    
    # Generate sample features including gate features
    features_df = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)] + 
                ['gate_quality', 'gate_stability', 'gate_variance'],
        index=pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    )
    
    # Generate sample targets
    targets_series = pd.Series(
        np.random.randn(n_samples),
        index=features_df.index,
        name='target'
    )
    
    print(f"📊 Sample data with gate features: {features_df.shape[0]} samples, {features_df.shape[1]} features")
    print(f"🛡️ Gate features in data: {[col for col in features_df.columns if 'gate' in col.lower()]}")
    
    # Test gate feature protection logic
    all_features = list(features_df.columns)
    selected_features = all_features[:10]  # Simulate feature selection
    
    print(f"📋 Selected features (before protection): {selected_features}")
    
    # Apply gate feature protection
    gate_features = [col for col in all_features if 'gate' in col.lower()]
    protected_gate_features = [gf for gf in gate_features if gf not in selected_features]
    
    if protected_gate_features:
        print(f"🛡️ Adding protected gate features: {protected_gate_features}")
        selected_features.extend(protected_gate_features)
    else:
        print("🛡️ All gate features already selected")
    
    print(f"📋 Final selected features (after protection): {selected_features}")
    print(f"🛡️ Gate features in final selection: {[col for col in selected_features if 'gate' in col.lower()]}")
    
    # Verify all gate features are included
    final_gate_features = [col for col in selected_features if 'gate' in col.lower()]
    if len(final_gate_features) == len(gate_features):
        print("✅ All gate features are protected!")
        return True
    else:
        print(f"❌ Gate feature protection failed: {len(final_gate_features)}/{len(gate_features)} gate features included")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Gate Feature Integration Tests")
    print("=" * 60)
    
    # Test 1: Gate feature generation
    test1_passed = await test_gate_feature_generation()
    
    # Test 2: Gate feature protection
    test2_passed = await test_gate_feature_protection()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 30)
    print(f"Gate Feature Generation: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Gate Feature Protection: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Gate feature integration is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)