
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.utils.versioned_artifacts.store import VersionedArtifactStore

def check_versions():
    store_path = "/Users/remyroche/Ares/versioned_artifacts/ETHUSDT_binance_15m_long_analyst"
    if not Path(store_path).exists():
        print(f"Store path {store_path} does not exist")
        return

    store = VersionedArtifactStore(store_path)
    versions = store.list_versions()
    print(f"Total versions: {len(versions)}")
    
    for v in versions:
        if 'labeled_data' in v:
            try:
                df = store.get_view(v).materialize()
                target_cols = [c for c in df.columns if 'label' in c or 'target' in c]
                print(f"\nVersion: {v}")
                print(f"  Rows: {len(df)}")
                for col in target_cols:
                    target = df[col]
                    unique_vals = target.nunique()
                    mean_val = target.mean()
                    print(f"  Target: {col} | Unique: {unique_vals} | Mean: {mean_val:.4f}")
            except Exception as e:
                print(f"  Error loading {v}: {e}")

if __name__ == "__main__":
    check_versions()
