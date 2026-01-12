
import pandas as pd
import h5py
import os
import sys

checkpoint_path = "versioned_artifacts/layer2_checkpoints/ETHUSDT/checkpoint_specialist_training.h5"

print(f"Inspecting: {checkpoint_path}")

if not os.path.exists(checkpoint_path):
    print("File not found!")
    sys.exit(1)

try:
    with pd.HDFStore(checkpoint_path, mode='r') as store:
        print("\nKeys in HDFStore:")
        print(store.keys())
        
        # Try to load specialist_predictions
        preds = {}
        for key in store.keys():
            if 'specialist_predictions' in key:
                name = key.split('/')[-1]
                preds[name] = store[key]
                print(f"\nLoaded {name}: shape={preds[name].shape}")
                print(preds[name].describe())
                
        if not preds:
            print("\nNo specialist_predictions found!")
            
except Exception as e:
    print(f"\nError using HDFStore: {e}")
    # Fallback to h5py for raw inspection
    try:
        with h5py.File(checkpoint_path, 'r') as f:
            print("\nRaw H5 Keys:")
            f.visit(lambda x: print(x))
    except Exception as e2:
         print(f"Error using h5py: {e2}")
