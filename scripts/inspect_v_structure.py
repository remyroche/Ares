
import h5py
from pathlib import Path

def inspect_version_structure(version_name):
    store_path = "/Users/remyroche/Ares/versioned_artifacts/ETHUSDT_binance_15m_long_analyst/store.h5"
    if not Path(store_path).exists():
        print(f"H5 file {store_path} does not exist")
        return

    with h5py.File(store_path, 'r') as f:
        if 'versions' not in f or version_name not in f['versions']:
            print(f"Version {version_name} not found")
            return
            
        group = f['versions'][version_name]
        print(f"Structure of version: {version_name}")
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Dataset):
                print(f"  Dataset: {key} | Shape: {item.shape} | Type: {item.dtype}")
            else:
                print(f"  Group: {key}")

if __name__ == "__main__":
    # Pick one of the labeled_data versions seen earlier
    inspect_version_structure("labeled_data_ETHUSDT_15m_20251210_012459_291")
