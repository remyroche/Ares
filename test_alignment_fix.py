#!/usr/bin/env python3
"""
Quick test to verify alignment fix is working.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.versioned_artifacts.store import VersionedArtifactStore

# Check what features are available
store = VersionedArtifactStore('versioned_artifacts/UNKNOWN_binance_15m_long_analyst')
versions = store.list_versions()

print("=" * 80)
print("AVAILABLE VERSIONS IN STORE")
print("=" * 80)

feature_versions = [v for v in versions if 'feature' in v.lower() or 'generated' in v.lower()]
print(f"\nFeature-related versions ({len(feature_versions)}):")
for v in sorted(feature_versions):
    print(f"  - {v}")

label_versions = [v for v in versions if 'label' in v.lower()]
print(f"\nLabel-related versions ({len(label_versions)}):")
for v in sorted(label_versions)[-5:]:  # Last 5
    print(f"  - {v}")

# Try to load a feature version
print("\n" + "=" * 80)
print("TESTING FEATURE LOADING")
print("=" * 80)

test_artifacts = [
    'generated_features_15m',
    'generated_features',
]

for artifact_name in test_artifacts:
    try:
        print(f"\nTrying to load '{artifact_name}'...")
        data = store.get(artifact_name)
        if data is not None:
            print(f"  ✅ Found! Shape: {data.shape}")
            print(f"  Index range: {data.index.min()} to {data.index.max()}")
            print(f"  Columns: {list(data.columns[:5])}...")
            break
        else:
            print(f"  ❌ Returned None")
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "=" * 80)
