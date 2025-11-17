#!/usr/bin/env python3
"""
Test script to verify the artifact retrieval fix is working.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.artifact_router import ArtifactRouter
from tests.utils.assertions import (
    assert_is_instance, assert_equals, assert_greater_than,
    assert_is_not_none, assert_is_none, assert_true
)

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
                    # Validation avec assertions standardisées
                    assert_is_instance(data.shape, tuple, "Le shape doit être un tuple", "Test validation du shape")
                    assert_equals(len(data.shape), 2, "Le shape doit avoir 2 dimensions (lignes, colonnes)", "Test validation du nombre de dimensions")
                    assert_true(all(isinstance(dim, int) for dim in data.shape), "Toutes les dimensions doivent être des entiers", "Test validation du type des dimensions")
                    assert_true(all(dim > 0 for dim in data.shape), "Toutes les dimensions doivent être positives", "Test validation des valeurs des dimensions")
                else:
                    print(f"✅ SUCCESS: {artifact_name} -> type {type(data)}")
                    # Validation avec assertions standardisées
                    assert_is_not_none(data, f"Les données pour {artifact_name} ne devraient pas être None", "Test validation des données non nulles")
            else:
                print(f"⚠️ NOT FOUND: {artifact_name}")
                # Validation avec assertions standardisées
                assert_is_none(data, f"Les données pour {artifact_name} devraient être None", "Test validation des données nulles")
                
        except Exception as e:
            print(f"❌ ERROR: {artifact_name} -> {e}")
            # Validation avec assertions standardisées
            assert_is_instance(e, Exception, f"L'erreur devrait être une Exception: {type(e)}", "Test validation du type d'exception")
    
    print("\n🎯 Test completed!")

if __name__ == "__main__":
    test_artifact_retrieval()