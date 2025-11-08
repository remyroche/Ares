#!/usr/bin/env python3
"""
Test script to verify the artifact retrieval fix is working.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.artifact_router import ArtifactRouter

def test_artifact_retrieval():
    """Test that artifact retrieval works with our fix."""
    print("🔍 Testing artifact retrieval fix...")
    
    # Initialize router
    router = ArtifactRouter(
        base_dir="artifacts",
        versioned_store_dir="versioned_artifacts",
        historical_data_dir="historical_data",
        enable_versioned_artifacts=True
    )
    
    # Test loading features that were failing before
    test_artifacts = [
        'selected_feature_dataframe_50',
        'selected_features_50',
        'final_dataset_50',
        'generated_features_15m_20251107_163517',
        'generated_features_15m_20251107_163518',
        'generated_features_15m_20251107_163519'
    ]
    
    for artifact_name in test_artifacts:
        try:
            print(f"\n📂 Trying to load: {artifact_name}")
            data = router.load(
                artifact_name=artifact_name,
                artifact_type='data',
                data_category='features'
            )
            
            if data is not None:
                if hasattr(data, 'shape'):
                    print(f"✅ SUCCESS: {artifact_name} -> shape {data.shape}")
                else:
                    print(f"✅ SUCCESS: {artifact_name} -> type {type(data)}")
            else:
                print(f"⚠️ NOT FOUND: {artifact_name}")
                
        except Exception as e:
            print(f"❌ ERROR: {artifact_name} -> {e}")
    
    print("\n🎯 Test completed!")

if __name__ == "__main__":
    test_artifact_retrieval()