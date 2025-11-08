#!/usr/bin/env python3
"""
Check what artifacts actually exist in the versioned artifacts store.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.versioned_artifacts import VersionedArtifactStore

def check_existing_artifacts():
    """Check what artifacts actually exist."""
    print("🔍 Checking existing artifacts in versioned store...")
    
    # Check the UNKNOWN_binance_15m_long_analyst store (from debug script)
    store_path = "src/utils/versioned_artifacts/UNKNOWN_binance_15m_long_analyst"
    
    if os.path.exists(store_path):
        print(f"✅ Store exists: {store_path}")
        
        try:
            store = VersionedArtifactStore(store_path=store_path, auto_version=False)
            
            # List all artifacts
            print("\n📋 Available artifacts:")
            try:
                artifacts = store.list_artifacts()
                for artifact_name in artifacts:
                    print(f"   - {artifact_name}")
            except Exception as e:
                print(f"   Error listing artifacts: {e}")
            
            # Check metadata
            print("\n📄 Metadata contents:")
            if hasattr(store, '_metadata') and store._metadata:
                for key, value in store._metadata.items():
                    if key == 'artifacts':
                        print(f"   {key}:")
                        for art_name, art_info in value.items():
                            print(f"      - {art_name}: {art_info}")
                    else:
                        print(f"   {key}: {value}")
            
            # Check HDF5 file contents
            h5_path = os.path.join(store_path, "artifacts.h5")
            if os.path.exists(h5_path):
                print(f"\n📊 HDF5 file exists: {h5_path}")
                import h5py
                with h5py.File(h5_path, 'r') as f:
                    print("   HDF5 datasets:")
                    for key in f.keys():
                        dataset = f[key]
                        shape_str = f"{dataset.shape}" if hasattr(dataset, 'shape') else 'unknown shape'
                        print(f"      - {key}: {shape_str}")
            else:
                print(f"\n❌ HDF5 file missing: {h5_path}")
                
        except Exception as e:
            print(f"❌ Error checking store: {e}")
    else:
        print(f"❌ Store not found: {store_path}")
        
        # Check what stores actually exist
        versioned_dir = "src/utils/versioned_artifacts"
        if os.path.exists(versioned_dir):
            print(f"\n📁 Available stores in {versioned_dir}:")
            for item in os.listdir(versioned_dir):
                item_path = os.path.join(versioned_dir, item)
                if os.path.isdir(item_path):
                    print(f"   - {item}")
        else:
            print(f"\n❌ Versioned artifacts directory not found: {versioned_dir}")

if __name__ == "__main__":
    check_existing_artifacts()