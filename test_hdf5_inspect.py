"""
Direct HDF5 inspection to check for data truncation.
"""
import h5py
import pandas as pd
import numpy as np
from pathlib import Path

# Path to the regime store
store_path = Path("versioned_artifacts/ETHUSDT_binance_1h_long_regime")
h5_file = store_path / "store.h5"

print(f"Inspecting HDF5 file: {h5_file}")
print(f"File exists: {h5_file.exists()}")

if h5_file.exists():
    print(f"File size: {h5_file.stat().st_size / 1024 / 1024:.2f} MB")

    with h5py.File(h5_file, 'r') as f:
        print(f"\n=== HDF5 File Structure ===")
        print(f"Root keys: {list(f.keys())}")

        if 'versions' in f:
            versions_group = f['versions']
            version_names = list(versions_group.keys())
            print(f"\n=== Versions ({len(version_names)}) ===")
            print(f"Version names: {version_names}")

            # Inspect each version
            for version_name in version_names:
                print(f"\n=== Version: {version_name} ===")
                version_group = versions_group[version_name]

                # Get attributes
                attrs = dict(version_group.attrs)
                print(f"Attributes: {attrs}")

                # Get datasets
                datasets = [k for k in version_group.keys() if not k.startswith('_')]
                print(f"Datasets (columns): {len(datasets)}")
                print(f"Column names: {datasets[:10]}...")  # First 10

                # Check index
                if '_index' in version_group:
                    index_dataset = version_group['_index']
                    index_shape = index_dataset.shape
                    index_chunks = index_dataset.chunks
                    print(f"\nIndex shape: {index_shape}")
                    print(f"Index chunks: {index_chunks}")
                    print(f"Index dtype: {index_dataset.dtype}")

                    # Load index to check actual data
                    index_data = index_dataset[:]
                    print(f"Actual index length loaded: {len(index_data)}")

                # Check first data column
                if datasets:
                    first_col = datasets[0]
                    first_col_dataset = version_group[first_col]
                    col_shape = first_col_dataset.shape
                    col_chunks = first_col_dataset.chunks
                    print(f"\nFirst column '{first_col}':")
                    print(f"  Shape: {col_shape}")
                    print(f"  Chunks: {col_chunks}")
                    print(f"  Dtype: {first_col_dataset.dtype}")

                    # Load actual data to check
                    col_data = first_col_dataset[:]
                    print(f"  Actual data length loaded: {len(col_data)}")

                # Compare metadata vs actual
                expected_rows = attrs.get('num_rows', 'N/A')
                actual_rows = len(index_data) if '_index' in version_group else 'N/A'

                print(f"\n⚠️  COMPARISON:")
                print(f"  Expected rows (metadata): {expected_rows}")
                print(f"  Actual rows (HDF5 data): {actual_rows}")

                if expected_rows != 'N/A' and actual_rows != 'N/A':
                    if expected_rows != actual_rows:
                        print(f"  ❌ MISMATCH! Data was truncated by {expected_rows - actual_rows} rows")
                    else:
                        print(f"  ✅ Match - no truncation")
else:
    print("HDF5 file not found!")
