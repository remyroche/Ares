
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import h5py

# Add src to path
sys.path.append(os.getcwd())

from src.utils.versioned_artifacts.store import VersionedArtifactStore

def check_target_distributions():
    store_path = "/Users/remyroche/Ares/versioned_artifacts/ETHUSDT_binance_15m_long_analyst"
    h5_file = Path(store_path) / "store.h5"
    if not h5_file.exists():
        print(f"H5 file {h5_file} does not exist")
        return

    import h5py
    with h5py.File(h5_file, 'r') as f:
        h5_versions = list(f['versions'].keys())
    
    print(f"Found {len(h5_versions)} versions in H5 file")
    
    store = VersionedArtifactStore(store_path)
    labeled_versions = [v for v in h5_versions if 'labeled_data' in v]
    
    for v in labeled_versions:
        try:
            print(f"\n--- Checking version: {v} ---")
            view = store.get_view(v)
            df = view.materialize()
            
            target_cols = [c for c in df.columns if 'label' in c or 'target' in c]
            if not target_cols:
                print(f"  No target columns found. Available: {df.columns.tolist()[:10]}...")
                continue
                
            print(f"  Rows: {len(df)}")
            for col in target_cols:
                target = df[col]
                dist = target.value_counts(dropna=False).to_dict()
                print(f"  Column: {col} | Mean: {target.mean():.4f} | Unique: {target.nunique()}")
                print(f"    Distribution: {dist}")
                
        except Exception as e:
            print(f"  Error checking {v}: {e}")

if __name__ == "__main__":
    check_target_distributions()
