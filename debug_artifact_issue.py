#!/usr/bin/env python3
"""
Debug script to reproduce and fix the artifact fetching issue.

The problem: generated_features_15m_* versions exist in metadata but not in HDF5 file.
This causes feature lookup to fail and fall back to using full market_data.index.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.versioned_artifacts import VersionedArtifactStore
from src.utils.artifact_router import ArtifactRouter
import pandas as pd
import numpy as np
from datetime import datetime

def main():
    print("🔍 Debugging artifact fetching issue...")
    
    # 1. Check the versioned store directly
    print("\n1️⃣ Checking versioned store directly...")
    store_path = "versioned_artifacts/UNKNOWN_binance_15m_long_analyst"
    store = VersionedArtifactStore(store_path)
    
    print(f"Store path: {store.store_path}")
    print(f"HDF5 file exists: {store.h5_file.exists()}")
    print(f"Metadata file exists: {store.metadata_file.exists()}")
    
    # List versions from metadata
    versions = store.list_versions()
    print(f"\n📋 Versions from metadata ({len(versions)} total):")
    for v in versions:
        if 'generated_features_15m' in v:
            print(f"  ✅ {v}")
        else:
            print(f"  📄 {v}")
    
    # Check HDF5 file content directly
    print(f"\n🗂️ Checking HDF5 file content directly...")
    import h5py
    with h5py.File(store.h5_file, 'r') as f:
        if 'versions' in f:
            h5_versions = list(f['versions'].keys())
            print(f"HDF5 versions ({len(h5_versions)} total):")
            for v in h5_versions:
                if 'generated_features_15m' in v:
                    print(f"  ✅ {v}")
                else:
                    print(f"  📄 {v}")
        else:
            print("❌ No 'versions' group in HDF5 file!")
    
    # 2. Try to reproduce the save process
    print(f"\n2️⃣ Attempting to reproduce save process...")
    
    # Create sample features data
    sample_data = pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'feature_3': np.random.randn(100),
    }, index=pd.date_range('2024-01-01', periods=100, freq='15T'))
    
    print(f"Sample data shape: {sample_data.shape}")
    
    # Try to save using artifact router
    router = ArtifactRouter()
    
    context = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'direction': 'long',
        'model': 'analyst'
    }
    
    try:
        print("🐛 DEBUG: Attempting to save sample features...")
        path = router.save(
            data=sample_data,
            artifact_name='generated_features_15m',
            data_category='features',
            context=context,
            metadata={'test': True}
        )
        print(f"✅ Save successful: {path}")
    except Exception as e:
        print(f"❌ Save failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Check if the issue is in the loading process
    print(f"\n3️⃣ Testing load process...")
    
    try:
        print("🐛 DEBUG: Attempting to load generated_features_15m...")
        loaded_data = router.load(
            artifact_name='generated_features_15m',
            data_category='features',
            context=context
        )
        print(f"✅ Load successful: {loaded_data.shape}")
    except Exception as e:
        print(f"❌ Load failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Check metadata vs HDF5 consistency
    print(f"\n4️⃣ Checking metadata vs HDF5 consistency...")
    
    # Reload metadata
    metadata = store._load_metadata()
    feature_versions = [v for v in metadata['versions'].keys() if 'generated_features_15m' in v]
    
    print(f"Feature versions in metadata: {len(feature_versions)}")
    for v in feature_versions:
        print(f"  📝 {v}")
        
        # Check if this version exists in HDF5
        with h5py.File(store.h5_file, 'r') as f:
            if 'versions' in f and v in f['versions']:
                print(f"    ✅ Found in HDF5")
            else:
                print(f"    ❌ MISSING from HDF5!")
    
    print("\n🔧 Analysis complete!")

if __name__ == "__main__":
    main()