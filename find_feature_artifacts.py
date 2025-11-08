#!/usr/bin/env python3
"""
Find feature artifacts in all versioned artifact stores.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def find_feature_artifacts():
    """Find feature artifacts in all stores."""
    print("🔍 Searching for feature artifacts in all versioned stores...")
    
    versioned_dir = "src/utils/versioned_artifacts"
    if not os.path.exists(versioned_dir):
        print(f"❌ Versioned artifacts directory not found: {versioned_dir}")
        return
    
    # Get all stores
    stores = [d for d in os.listdir(versioned_dir) 
              if os.path.isdir(os.path.join(versioned_dir, d))]
    
    print(f"📁 Found {len(stores)} stores:")
    for store in stores:
        print(f"   - {store}")
    
    # Check each store for feature artifacts
    for store in stores:
        store_path = os.path.join(versioned_dir, store)
        print(f"\n🔍 Checking store: {store}")
        
        try:
            # Check for HDF5 file
            h5_path = os.path.join(store_path, "artifacts.h5")
            if os.path.exists(h5_path):
                import h5py
                with h5py.File(h5_path, 'r') as f:
                    datasets = list(f.keys())
                    feature_datasets = [d for d in datasets if 'feature' in d.lower()]
                    print(f"   📊 Found {len(feature_datasets)} feature datasets: {feature_datasets}")
                    
                    # Check for specific artifacts we need
                    target_artifacts = [
                        'selected_feature_dataframe_50',
                        'selected_features_50',
                        'final_dataset_50',
                        'generated_features_15m_20251107_163517',
                        'generated_features_15m_20251107_163518',
                        'generated_features_15m_20251107_163519'
                    ]
                    
                    found_artifacts = []
                    for artifact in target_artifacts:
                        if artifact in datasets:
                            found_artifacts.append(artifact)
                            print(f"   ✅ FOUND: {artifact}")
                        else:
                            print(f"   ❌ MISSING: {artifact}")
                    
                    if found_artifacts:
                        print(f"\n🎯 Store '{store}' has {len(found_artifacts)} of the required artifacts!")
                        return store, found_artifacts
            else:
                print(f"   ❌ No HDF5 file found")
                
        except Exception as e:
            print(f"   ❌ Error checking store {store}: {e}")
    
    print("\n❌ No store had the required feature artifacts")
    return None, []

if __name__ == "__main__":
    find_feature_artifacts()