#!/usr/bin/env python3
"""
Check what artifacts exist in the regular artifacts directory.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_regular_artifacts():
    """Check what artifacts exist in regular artifacts directory."""
    print("🔍 Checking existing artifacts in regular artifacts directory...")
    
    # Check regular artifacts directory
    artifacts_dir = "artifacts"
    
    if os.path.exists(artifacts_dir):
        print(f"✅ Artifacts directory exists: {artifacts_dir}")
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(artifacts_dir):
            level = root.replace(artifacts_dir, '').strip(os.sep)
            if level:
                print(f"\n📁 Level: {level}")
                
                # Check for HDF5 files
                h5_files = [f for f in files if f.endswith('.h5')]
                if h5_files:
                    for h5_file in h5_files:
                        h5_path = os.path.join(root, h5_file)
                        print(f"   📊 HDF5 file: {h5_file}")
                        
                        # Check contents
                        try:
                            import h5py
                            with h5py.File(h5_path, 'r') as f:
                                datasets = list(f.keys())
                                print(f"      Datasets: {datasets}")
                                
                                # Check for feature datasets
                                feature_datasets = [d for d in datasets if 'feature' in d.lower()]
                                if feature_datasets:
                                    print(f"      Feature datasets: {feature_datasets}")
                        except Exception as e:
                            print(f"      ❌ Error reading {h5_file}: {e}")
                
                # Check for other files
                other_files = [f for f in files if not f.endswith('.h5')]
                if other_files:
                    print(f"   📄 Other files: {other_files}")
    else:
        print(f"❌ Artifacts directory not found: {artifacts_dir}")

if __name__ == "__main__":
    check_regular_artifacts()