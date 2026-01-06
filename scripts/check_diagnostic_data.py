
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.utils.versioned_artifacts import VersionedArtifactStore

def check_artifact():
    symbol = "ETHUSDT"
    exchange = "binance"
    timeframe = "15m"
    direction = "long"
    model = "analyst"
    
    store_name = f"{symbol}_{exchange}_{timeframe}_{direction}_{model}"
    store_path = Path("versioned_artifacts") / store_name
    
    if not store_path.exists():
        print(f"Store path {store_path} does not exist")
        return

    store = VersionedArtifactStore(store_path)
    try:
        # List all versions to see what's actually there
        versions = store.list_versions()
        print(f"Available versions in store: {versions}")
        
        # Try to find a labeled data version
        target_version = None
        for v in versions:
            if "labeled_data" in v:
                target_version = v
                break
        
        if not target_version:
            print("No labeled_data version found in store.")
            return
            
        print(f"Loading version: {target_version}")
        view = store.get_view(target_version)
        df = view.materialize()
        
        if df is None:
            print(f"Artifact {artifact_name} not found in {store_path}")
            return
            
        target_col = "binary_label_long"
        if target_col not in df.columns:
            print(f"Column {target_col} not in columns: {df.columns.tolist()}")
            return
            
        target = df[target_col]
        print(f"\n--- Target Distribution ({target_col}) ---")
        print(target.value_counts(dropna=False))
        print(f"Total rows: {len(df)}")
        print(f"Index range: {df.index.min()} to {df.index.max()}")
        
        # Check some specialist features
        print("\n--- Specialist Features Sample ---")
        spec_cols = [c for c in df.columns if any(s in c for s in ["risk", "smc", "liquidity", "volume", "macro", "meso", "path"])]
        if spec_cols:
            print(f"Found {len(spec_cols)} specialist columns")
            print(df[spec_cols].describe().loc[['count', 'mean', 'std']])
        else:
            print("No specialist columns found in labeled_data")
            
    except Exception as e:
        print(f"Error checking artifact: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_artifact()
