
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.steps.labeling.layer2_checkpoint_manager import Layer2CheckpointManager

def test_serialization():
    manager = Layer2CheckpointManager()
    
    # 1. Test Timestamp keys in dict
    ts = pd.Timestamp('2026-01-12 12:00:00')
    nested_dict = {
        ts: "value_at_ts",
        "nested": {
            pd.Timestamp('2026-01-12 13:00:00'): [ts, "plain_str"]
        }
    }
    
    # 2. Test Series with DatetimeIndex
    idx = pd.date_range('2026-01-12', periods=3, freq='15min')
    s = pd.Series([1.0, 2.0, 3.0], index=idx)
    
    data = {
        'timestamp_meta': nested_dict,
        'predictions': s,
        'array': np.array([1, 2, 3])
    }
    
    print("Serializing test data...")
    serialized = manager._serialize_for_json(data)
    
    print("\nSerialized structure:")
    print(json.dumps(serialized, indent=2))
    
    # Verify no Timestamps in keys
    json_str = json.dumps(serialized)
    print("\n✅ json.dumps succeeded")
    
    # Check specific keys
    assert ts.isoformat() in serialized['timestamp_meta']
    assert serialized['timestamp_meta'][ts.isoformat()] == "value_at_ts"
    
    # Check series conversion
    series_dict = serialized['predictions']
    assert isinstance(series_dict, dict)
    for k in series_dict.keys():
        assert isinstance(k, str)
        pd.Timestamp(k) # Verify it's a valid timestamp string
        
    print("\n✅ All assertions passed!")

if __name__ == "__main__":
    try:
        test_serialization()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
