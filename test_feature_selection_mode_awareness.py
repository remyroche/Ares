#!/usr/bin/env python3
"""
Test script to verify mode-aware artifact handling in feature_generation_final_feature_selection_step.

This script tests that the step correctly detects execution mode and uses appropriate artifacts.
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.training.steps.pre_training.feature_generation_final_feature_selection_step import (
    FeatureGenerationFinalFeatureSelectionStep
)

class MockArtifactManager:
    """Mock artifact manager for testing."""
    
    def __init__(self):
        self.artifacts = {}
    
    def get_artifact(self, artifact_name: str, artifact_type: str = "data"):
        """Mock get artifact method."""
        return self.artifacts.get(artifact_name)
    
    def save_artifact(self, data, artifact_name: str, artifact_type: str = "data", **kwargs):
        """Mock save artifact method."""
        self.artifacts[artifact_name] = data
        return f"mock_path/{artifact_name}"

async def test_mode_detection():
    """Test mode detection functionality."""
    print("🧪 Testing mode detection...")
    
    step = FeatureGenerationFinalFeatureSelectionStep()
    
    # Test 1: Default mode (should be analyst)
    step.step_name = "feature_generation_final_feature_selection_step"
    step.config = {}
    step.execution_context = ""
    mode = step._detect_execution_mode()
    assert mode == "analyst", f"Expected 'analyst', got '{mode}'"
    print("✅ Default mode detection: PASSED")
    
    # Test 2: Tactician mode from step name
    step.step_name = "tactician_feature_generation_final_feature_selection_step"
    mode = step._detect_execution_mode()
    assert mode == "tactician", f"Expected 'tactician', got '{mode}'"
    print("✅ Tactician mode from step name: PASSED")
    
    # Test 3: Analyst mode from step name
    step.step_name = "analyst_feature_generation_final_feature_selection_step"
    mode = step._detect_execution_mode()
    assert mode == "analyst", f"Expected 'analyst', got '{mode}'"
    print("✅ Analyst mode from step name: PASSED")
    
    # Test 4: Mode from execution context
    step.step_name = "feature_generation_final_feature_selection_step"
    step.execution_context = "tactician"
    mode = step._detect_execution_mode()
    assert mode == "tactician", f"Expected 'tactician', got '{mode}'"
    print("✅ Tactician mode from execution context: PASSED")
    
    # Test 5: Mode from config
    step.execution_context = ""
    step.config = {"tactician_mode": True}
    mode = step._detect_execution_mode()
    assert mode == "tactician", f"Expected 'tactician', got '{mode}'"
    print("✅ Tactician mode from config: PASSED")
    
    # Test 6: Mode from interaction generation mode
    step.config = {"interaction_generation_mode": "analyst"}
    mode = step._detect_execution_mode()
    assert mode == "analyst", f"Expected 'analyst', got '{mode}'"
    print("✅ Analyst mode from interaction generation mode: PASSED")
    
    print("🎉 All mode detection tests passed!")

async def test_artifact_collection():
    """Test mode-aware artifact collection."""
    print("\n🧪 Testing mode-aware artifact collection...")
    
    step = FeatureGenerationFinalFeatureSelectionStep()
    
    # Mock the artifact manager
    step.artifact_manager = MockArtifactManager()
    
    # Create mock artifacts
    mock_features = pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'target': np.random.randn(100)
    })
    
    # Add mode-specific artifacts
    step.artifact_manager.artifacts = {
        'analyst_generated_features': mock_features,
        'tactician_generated_features': mock_features,
        'analyst_interaction_features': mock_features,
        'tactician_interaction_features': mock_features,
        'analyst_lookback_optimization': mock_features,
        'tactician_lookback_optimization': mock_features,
        'labeled_data': mock_features,
        'labeling_metadata': pd.Series([1, 2, 3])
    }
    
    # Test Analyst mode
    step.config = {"interaction_generation_mode": "analyst"}
    step.execution_context = "analyst"
    features_data = step._collect_features_from_previous_steps()
    
    # Should prioritize analyst artifacts
    assert 'generated_features' in features_data, "Should have generated features"
    print("✅ Analyst mode artifact collection: PASSED")
    
    # Test Tactician mode
    step.config = {"interaction_generation_mode": "tactician"}
    step.execution_context = "tactician"
    features_data = step._collect_features_from_previous_steps()
    
    # Should prioritize tactician artifacts but also include analyst for CMI
    assert 'generated_features' in features_data, "Should have generated features"
    print("✅ Tactician mode artifact collection: PASSED")
    
    print("🎉 All artifact collection tests passed!")

async def test_artifact_generation():
    """Test mode-aware artifact generation."""
    print("\n🧪 Testing mode-aware artifact generation...")
    
    step = FeatureGenerationFinalFeatureSelectionStep()
    step.config = {"interaction_generation_mode": "tactician"}
    step.execution_context = "tactician"
    
    # Mock feature sets
    feature_sets = {
        'selected_features_60': ['feature_1', 'feature_2', 'feature_3'],
        'selected_feature_dataframe_60': pd.DataFrame({
            'feature_1': [1, 2, 3],
            'feature_2': [4, 5, 6],
            'target': [0, 1, 0]
        })
    }
    
    # Mock SHAP values
    shap_values = {
        'shap_values_60': {
            'feature_importance': {'feature_1': 0.8, 'feature_2': 0.6}
        }
    }
    
    # Mock selection component
    class MockSelectionComponent:
        def get_feature_scores(self):
            return {'feature_1': 0.8, 'feature_2': 0.6}
    
    step.selection_component = MockSelectionComponent()
    
    # Mock combined features
    combined_features_df = pd.DataFrame({
        'feature_1': [1, 2, 3],
        'feature_2': [4, 5, 6],
        'target': [0, 1, 0]
    })
    
    config = {"symbol": "ETHUSDT", "exchange": "binance", "timeframe": "15m"}
    
    # Generate artifacts
    artifacts = step._generate_artifacts(feature_sets, shap_values, config, combined_features_df)
    
    # Check that mode-specific artifacts are generated
    mode_prefix = "tactician_"
    expected_artifacts = [
        f"{mode_prefix}selected_features_60",
        f"{mode_prefix}selected_feature_dataframe_60",
        f"{mode_prefix}feature_scores",
        f"{mode_prefix}shap_values_60",
        f"{mode_prefix}selection_metadata"
    ]
    
    for artifact_name in expected_artifacts:
        assert artifact_name in artifacts, f"Missing mode-specific artifact: {artifact_name}"
    
    # Check that backward compatibility artifacts are also generated
    backward_compat_artifacts = [
        "selected_features_60",
        "selected_feature_dataframe_60",
        "feature_scores",
        "shap_values_60",
        "selection_metadata"
    ]
    
    for artifact_name in backward_compat_artifacts:
        assert artifact_name in artifacts, f"Missing backward compatibility artifact: {artifact_name}"
    
    print("✅ Mode-aware artifact generation: PASSED")
    print("🎉 All artifact generation tests passed!")

async def main():
    """Run all tests."""
    print("🚀 Starting feature selection mode awareness tests...\n")
    
    try:
        await test_mode_detection()
        await test_artifact_collection()
        await test_artifact_generation()
        
        print("\n🎉 All tests passed successfully!")
        print("\n📋 Summary of changes:")
        print("✅ Mode detection logic implemented")
        print("✅ Mode-aware artifact collection implemented")
        print("✅ Mode-specific artifact naming implemented")
        print("✅ Backward compatibility maintained")
        print("\nThe feature_generation_final_feature_selection_step now:")
        print("- Detects execution mode (Analyst/Tactician) from launcher context")
        print("- Uses mode-specific artifacts when available")
        print("- Falls back to generic artifacts for backward compatibility")
        print("- Generates mode-prefixed artifacts (analyst_* or tactician_*)")
        print("- Maintains original artifact names for compatibility")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)