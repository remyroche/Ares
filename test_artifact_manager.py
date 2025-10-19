#!/usr/bin/env python3
"""
Test script to verify artifact manager fixes.
"""

import pandas as pd
import numpy as np
from src.training.steps.pre_training.utils.artifact_manager import get_pretraining_artifact_manager

def test_artifact_manager():
    """Test basic artifact manager functionality."""
    print("Testing artifact manager...")

    # Create test data
    df = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100),
        'target': np.random.randint(0, 2, 100)
    })

    series = pd.Series(np.random.randn(100), name='test_series')

    # Get artifact manager
    am = get_pretraining_artifact_manager()

    try:
        # Test saving artifacts
        print("Saving artifacts...")
        am.save(
            step_name='test_step',
            artifacts={
                'test_dataframe': df,
                'test_series': series,
                'test_object': {'key': 'value'}
            }
        )
        print("✓ Artifacts saved successfully")

        # Test retrieving artifacts
        print("Retrieving artifacts...")
        retrieved_df = am.get_artifact('test_step', 'test_dataframe')
        retrieved_series = am.get_artifact('test_step', 'test_series')
        retrieved_object = am.get_artifact('test_step', 'test_object')

        print(f"✓ Retrieved DataFrame: {type(retrieved_df)}, shape: {retrieved_df.shape}")
        print(f"✓ Retrieved Series: {type(retrieved_series)}, shape: {retrieved_series.shape}")
        print(f"✓ Retrieved Object: {type(retrieved_object)}")

        # Verify data integrity
        pd.testing.assert_frame_equal(df, retrieved_df)
        pd.testing.assert_series_equal(series, retrieved_series)
        assert retrieved_object == {'key': 'value'}

        print("✓ Data integrity verified")
        print("✓ Artifact manager test passed!")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = test_artifact_manager()
    exit(0 if success else 1)
