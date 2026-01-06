
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.steps.labeling.snr_diagnostics import _load_labeled_data

def probe_labels():
    symbol = "ETHUSDT"
    exchange = "binance"
    timeframe = "15m"
    direction = "long"
    model = "analyst"
    
    try:
        df = _load_labeled_data(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            direction=direction,
            model=model,
        )
        print(f"Labeled data shape: {df.shape}")
        if 'binary_label_long' in df.columns:
            print(f"binary_label_long non-null: {df['binary_label_long'].notna().sum()}")
        elif 'binary_label' in df.columns:
            print(f"binary_label non-null: {df['binary_label'].notna().sum()}")
            
        print("\nFirst 5 columns:")
        print(df.columns[:5].tolist())
        
        # Check source
        print(f"\nSource: {df.attrs.get('snr_source', 'unknown')}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    probe_labels()
