
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import h5py

# Add src to path
sys.path.append(os.getcwd())

def inspect_version_full(version_name):
    store_path = "/Users/remyroche/Ares/versioned_artifacts/ETHUSDT_binance_15m_long_analyst/store.h5"
    h5_file = Path(store_path) / "store.h5"
    
    with h5py.File(store_path, 'r') as f:
        group = f['versions'][version_name]
        datasets = {}
        for key in group.keys():
            if isinstance(group[key], h5py.Dataset):
                datasets[key] = group[key][:]
        
        df = pd.DataFrame(datasets)
        if '_index' in df.columns:
            df.index = pd.to_datetime(df['_index'], unit='ns')
            df = df.drop(columns=['_index'])
            
        print(f"Version: {version_name}")
        print(f"Shape: {df.shape}")
        
        target_cols = [c for c in df.columns if 'label' in c or 'target' in c]
        for col in target_cols:
            print(f"\nTarget Column: {col}")
            print(df[col].value_counts(dropna=False))
            
        # Check for specialist probability columns
        spec_probs = [c for c in df.columns if c.endswith('_specialist_probability')]
        print(f"\nEnhanced Specialist Probs found: {len(spec_probs)}")
        if spec_probs:
            print(spec_probs)

if __name__ == "__main__":
    inspect_version_full("labeled_data_ETHUSDT_15m_20251207_134646_131")
