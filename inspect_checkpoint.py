import pandas as pd
import os
import sys

# Add src to path just in case
sys.path.append(os.getcwd())

path = "/Users/remyroche/Documents/Ares/versioned_artifacts/layer2_checkpoints/ETHUSDT/checkpoint_model_race_complete.h5"
if not os.path.exists(path):
    print(f"File not found: {path}")
    exit(1)

print(f"Inspect checkpoint: {path}")
try:
    with pd.HDFStore(path, mode='r') as store:
        keys = store.keys()
        print("Keys:", keys)
        
        # Check for model race metrics or leaderboard
        # Typical keys might be 'predictions', 'metrics', 'leaderboard'
        
        for key in keys:
            print(f"\n--- Key: {key} ---")
            try:
                obj = store.get(key)
                if isinstance(obj, pd.DataFrame):
                    print(f"DataFrame Shape: {obj.shape}")
                    print(obj.head())
                    
                    # Check for IRM XGB
                    found = False
                    if 'model' in obj.columns:
                        models = obj['model'].unique()
                        print("Models in 'model' column:", models)
                        if any('XGB_IRM' in str(m) for m in models) or any('IRM_XGB' in str(m) for m in models):
                            print("✅ Found XGB_IRM/IRM_XGB in model column!")
                            found = True
                    
                    # Check index
                    if hasattr(obj.index, 'unique'):
                        idx_vals = obj.index.unique()
                        print("Index values:", idx_vals)
                        if any('XGB_IRM' in str(m) for m in idx_vals) or any('IRM_XGB' in str(m) for m in idx_vals):
                             print("✅ Found XGB_IRM/IRM_XGB in index!")
                             found = True
                             
                elif isinstance(obj, pd.Series):
                    print(f"Series Shape: {obj.shape}")
                    print(obj.head())
                else:
                    print(f"Object type: {type(obj)}")
                    
            except Exception as e:
                print(f"Could not read key {key}: {e}")

except Exception as e:
    print(f"Failed to open store: {e}")
