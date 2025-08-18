#!/usr/bin/env python3
"""
Test script for feature reduction and model-specific pruning implementation.
"""

import asyncio
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# Add src to path
import sys
sys.path.append('src')

from src.training.feature_selection_manager import FeatureSelectionManager
from src.training.model_specific_pruning import ModelSpecificPruning
from src.config.training import get_training_config


async def test_feature_selection_manager():
    """Test the FeatureSelectionManager."""
    print("🧪 Testing FeatureSelectionManager...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 220  # Simulate original feature count
    
    # Create sample features
    feature_names = []
    for i in range(n_features):
        if i < 50:
            feature_names.append(f"momentum_{i}")
        elif i < 100:
            feature_names.append(f"volatility_{i}")
        elif i < 150:
            feature_names.append(f"liquidity_{i}")
        elif i < 180:
            feature_names.append(f"interaction_x_{i}")
        else:
            feature_names.append(f"wavelet_{i}")
    
    # Create sample DataFrame
    features_df = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names,
        index=pd.date_range('2024-01-01', periods=n_samples, freq='1min')
    )
    
    # Create dummy target
    target = pd.Series(np.random.randint(0, 3, n_samples), index=features_df.index)
    
    # Initialize feature selection manager
    config = get_training_config()
    feature_selection_manager = FeatureSelectionManager(config)
    
    # Test feature selection
    selected_features, metadata = feature_selection_manager.select_features_step2(
        features_df, target, "ETHUSDT", "BINANCE", "data/test"
    )
    
    print(f"✅ Feature selection completed:")
    print(f"   Original features: {len(features_df.columns)}")
    print(f"   Selected features: {len(selected_features.columns)}")
    print(f"   Target features: {metadata['target_features']}")
    print(f"   Feature categories: {metadata['feature_categories']}")
    
    return selected_features, metadata


async def test_model_specific_pruning():
    """Test the ModelSpecificPruning."""
    print("\n🧪 Testing ModelSpecificPruning...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 100  # Use reduced feature count from Step 2
    
    # Create sample features
    feature_names = []
    for i in range(n_features):
        if i < 30:
            feature_names.append(f"momentum_{i}")
        elif i < 60:
            feature_names.append(f"volatility_{i}")
        elif i < 80:
            feature_names.append(f"liquidity_{i}")
        elif i < 90:
            feature_names.append(f"interaction_x_{i}")
        else:
            feature_names.append(f"wavelet_{i}")
    
    # Create sample DataFrame
    features_df = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names,
        index=pd.date_range('2024-01-01', periods=n_samples, freq='1min')
    )
    
    # Create dummy target
    target = pd.Series(np.random.randint(0, 3, n_samples), index=features_df.index)
    
    # Initialize model-specific pruning
    config = get_training_config()
    pruning_manager = ModelSpecificPruning(config)
    
    # Test different pruning strategies
    pruning_results = {}
    
    # Test neural network pruning
    nn_features, nn_metadata = pruning_manager.prune_for_neural_networks(
        features_df, target, "CNN"
    )
    pruning_results["neural_network"] = {
        "original": len(features_df.columns),
        "pruned": len(nn_features.columns),
        "metadata": nn_metadata
    }
    
    # Test linear model pruning
    linear_features, linear_metadata = pruning_manager.prune_for_linear_models(
        features_df, target, "LogisticRegression"
    )
    pruning_results["linear_model"] = {
        "original": len(features_df.columns),
        "pruned": len(linear_features.columns),
        "metadata": linear_metadata
    }
    
    # Test ensemble model pruning
    ensemble_features, ensemble_metadata = pruning_manager.prune_for_ensemble_models(
        features_df, target, "LightGBM"
    )
    pruning_results["ensemble_model"] = {
        "original": len(features_df.columns),
        "pruned": len(ensemble_features.columns),
        "metadata": ensemble_metadata
    }
    
    # Test Step 6 specific pruning
    step6_features, step6_metadata = pruning_manager.prune_for_step6_hmm_models(
        features_df, target, "1m", "CNN"
    )
    pruning_results["step6_cnn"] = {
        "original": len(features_df.columns),
        "pruned": len(step6_features.columns),
        "metadata": step6_metadata
    }
    
    # Test Step 6.5 specific pruning
    step6_5_features, step6_5_metadata = await pruning_manager.prune_for_step6_5_unified_regime(
        features_df, target
    )
    pruning_results["step6_5_unified"] = {
        "original": len(features_df.columns),
        "pruned": len(step6_5_features.columns),
        "metadata": step6_5_metadata
    }
    
    # Test Step 7 specific pruning
    step7_features, step7_metadata = await pruning_manager.prune_for_step7_ensemble(
        features_df, target
    )
    pruning_results["step7_ensemble"] = {
        "original": len(features_df.columns),
        "pruned": len(step7_features.columns),
        "metadata": step7_metadata
    }
    
    # Test Step 9 specific pruning
    step9_features, step9_metadata = await pruning_manager.prune_for_step9_tactician(
        features_df, target, "lightgbm"
    )
    pruning_results["step9_tactician"] = {
        "original": len(features_df.columns),
        "pruned": len(step9_features.columns),
        "metadata": step9_metadata
    }
    
    print("✅ Model-specific pruning completed:")
    for model_type, result in pruning_results.items():
        print(f"   {model_type}: {result['original']} -> {result['pruned']} features")
        print(f"     Strategy: {result['metadata'].get('pruning_strategy', 'unknown')}")
    
    return pruning_results


async def test_integration():
    """Test the integration of feature selection and pruning."""
    print("\n🧪 Testing Integration...")
    
    # Test feature selection first
    selected_features, selection_metadata = await test_feature_selection_manager()
    
    # Test pruning on selected features
    pruning_results = await test_model_specific_pruning()
    
    # Create integration summary
    integration_summary = {
        "feature_selection": {
            "original_features": 220,
            "selected_features": len(selected_features.columns),
            "reduction_percentage": ((220 - len(selected_features.columns)) / 220) * 100
        },
        "model_specific_pruning": pruning_results,
        "overall_reduction": {
            "step2_reduction": f"{((220 - len(selected_features.columns)) / 220) * 100:.1f}%",
            "average_model_reduction": f"{np.mean([result['pruned'] / result['original'] for result in pruning_results.values()]) * 100:.1f}%"
        }
    }
    
    print("\n📊 Integration Summary:")
    print(f"   Step 2 feature reduction: {integration_summary['overall_reduction']['step2_reduction']}")
    print(f"   Average model-specific reduction: {integration_summary['overall_reduction']['average_model_reduction']}")
    
    # Save results
    os.makedirs("test_results", exist_ok=True)
    with open("test_results/feature_reduction_test_results.json", "w") as f:
        json.dump(integration_summary, f, indent=2, default=str)
    
    print("💾 Test results saved to test_results/feature_reduction_test_results.json")
    
    return integration_summary


async def main():
    """Main test function."""
    print("🚀 Starting Feature Reduction and Model-Specific Pruning Tests")
    print("=" * 60)
    
    try:
        # Test individual components
        await test_feature_selection_manager()
        await test_model_specific_pruning()
        
        # Test integration
        integration_summary = await test_integration()
        
        print("\n✅ All tests completed successfully!")
        print("=" * 60)
        
        return integration_summary
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(main())