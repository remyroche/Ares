
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import h5py

# Add src to path
sys.path.append(os.getcwd())

from src.utils.versioned_artifacts.store import VersionedArtifactStore

def check_valid_version():
    store_path = "/Users/remyroche/Ares/versioned_artifacts/ETHUSDT_binance_15m_long_analyst"
    h5_file = Path(store_path) / "store.h5"
    
    import h5py
    with h5py.File(h5_file, 'r') as f:
        h5_versions = list(f['versions'].keys())
    
    labeled_versions = sorted([v for v in h5_versions if 'labeled_data' in v], reverse=True)
    print(f"Found {len(labeled_versions)} labeled_data versions in HDF5")
    
    for v in labeled_versions:
        try:
            print(f"\n--- Checking version: {v} ---")
            # Load directly from H5 by reading individual datasets
            datasets = {}
            with h5py.File(h5_file, 'r') as f:
                group = f['versions'][v]
                for key in group.keys():
                    if isinstance(group[key], h5py.Dataset):
                        datasets[key] = group[key][:]
            
            if not datasets:
                print(f"  No datasets found in {v}")
                continue
                
            df = pd.DataFrame(datasets)
            if '_index' in df.columns:
                df.index = pd.to_datetime(df['_index'], unit='ns')
                df = df.drop(columns=['_index'])
            
            target_cols = [c for c in df.columns if 'label' in c or 'target' in c]
            print(f"  Rows: {len(df)}")
            if not target_cols:
                print(f"  No target columns found. Available: {df.columns.tolist()[:10]}...")
                continue
                
            for col in target_cols:
                target = df[col]
                dist = target.value_counts(dropna=False).to_dict()
                print(f"  Column: {col} | Mean: {target.mean():.4f} | Unique: {target.nunique()}")
                print(f"    Distribution: {dist}")
                
            if any(df[col].nunique() > 1 for col in target_cols if 'binary_label' in col):
                print(f"  >>> FOUND VALID VERSION: {v}")
                # Print some feature names to check pattern matching
                features = [c for c in df.columns if not any(x in c.lower() for x in ['label', 'target', 'return', 'pnl'])]
                print(f"  Total potential features: {len(features)}")
                print(f"  Feature samples: {features[:10]}")
                # break
        except Exception as e:
            print(f"  Error loading {v}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    check_valid_version()
