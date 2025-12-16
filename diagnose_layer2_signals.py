"""
Diagnostic script to investigate Layer 2 signal quality issues.

Checks:
1. Expert signal correlations with returns
2. Regime feature importance / usage
"""
import sys
import os
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Paths
ARTIFACT_DIR = "versioned_artifacts/ETHUSDT_binance_15m_long_analyst"
STORE_PATH = f"{ARTIFACT_DIR}/store.h5"

def load_from_store(key: str) -> pd.DataFrame:
    """Load a DataFrame from the HDF5 store."""
    try:
        with pd.HDFStore(STORE_PATH, mode='r') as store:
            if key in store:
                return store[key]
            else:
                print(f"Key '{key}' not found in store")
                return None
    except Exception as e:
        print(f"Error loading {key}: {e}")
        return None

def list_store_keys():
    """List all keys in the store."""
    try:
        with pd.HDFStore(STORE_PATH, mode='r') as store:
            return list(store.keys())
    except Exception as e:
        print(f"Error listing keys: {e}")
        return []

def analyze_expert_correlations():
    """Check if expert probabilities correlate with returns."""
    print("\n" + "="*60)
    print("EXPERT SIGNAL CORRELATION ANALYSIS")
    print("="*60)
    
    # Try to find expert signals
    keys = list_store_keys()
    print(f"\nTotal keys in store: {len(keys)}")
    
    # Look for expert-related keys
    expert_keys = [k for k in keys if any(x in k.lower() for x in ['expert', 'signal', 'momentum', 'meanrev', 'analyst', 'proba', 'pred'])]
    print(f"\nExpert-related keys found: {len(expert_keys)}")
    for k in expert_keys[:20]:
        print(f"  - {k}")
    
    # Look for market data / returns
    market_keys = [k for k in keys if any(x in k.lower() for x in ['market', 'ohlcv', 'kline', 'return'])]
    print(f"\nMarket data keys found: {len(market_keys)}")
    for k in market_keys[:10]:
        print(f"  - {k}")
    
    # Look for regime embeddings
    regime_keys = [k for k in keys if any(x in k.lower() for x in ['regime', 'leaf', 'embed'])]
    print(f"\nRegime-related keys found: {len(regime_keys)}")
    for k in regime_keys[:10]:
        print(f"  - {k}")
    
    # Look for feature importance
    importance_keys = [k for k in keys if any(x in k.lower() for x in ['importance', 'feature', 'shap', 'mdi'])]
    print(f"\nImportance-related keys found: {len(importance_keys)}")
    for k in importance_keys[:10]:
        print(f"  - {k}")

def analyze_layer2_features():
    """Check what features Layer 2 is actually using."""
    print("\n" + "="*60)
    print("LAYER 2 FEATURE ANALYSIS")
    print("="*60)
    
    keys = list_store_keys()
    
    # Look for Layer 2 specific data
    l2_keys = [k for k in keys if 'layer2' in k.lower() or 'l2' in k.lower()]
    print(f"\nLayer 2 keys found: {len(l2_keys)}")
    for k in l2_keys[:20]:
        print(f"  - {k}")
    
    # Try to load and inspect feature matrices
    feature_keys = [k for k in keys if 'feature' in k.lower() or 'x_' in k.lower()]
    print(f"\nFeature matrix keys: {len(feature_keys)}")
    for k in feature_keys[:10]:
        print(f"  - {k}")

if __name__ == "__main__":
    print("Layer 2 Signal Quality Diagnostic")
    print("="*60)
    
    # First, list all keys to understand structure
    analyze_expert_correlations()
    analyze_layer2_features()
    
    print("\n" + "="*60)
    print("DETAILED KEY LISTING")
    print("="*60)
    keys = list_store_keys()
    for k in sorted(keys):
        print(f"  {k}")
