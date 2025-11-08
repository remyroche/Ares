#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

from utils.versioned_artifacts.store import VersionedArtifactStore

# Test the get_view method
store = VersionedArtifactStore('versioned_artifacts/UNKNOWN_binance_15m_long_analyst')

# List available versions
versions = store.list_versions()
print(f'Available versions: {len(versions)}')

# Find feature versions
feature_versions = [v for v in versions if 'generated_features' in v.lower()]
print(f'\nFeature versions ({len(feature_versions)}):')
for v in sorted(feature_versions):
    print(f'  - {v}')

# Try to get view of the latest generated_features
if feature_versions:
    latest = sorted(feature_versions)[-1]
    print(f'\nTesting get_view for {latest}...')
    try:
        view = store.get_view(latest)
        if view:
            print(f'✅ Got view: {view}')
            data = view.materialize()
            print(f'✅ Materialized shape: {data.shape}')
            print(f'✅ Index range: {data.index.min()} to {data.index.max()}')
        else:
            print('❌ get_view returned None')
    except Exception as e:
        print(f'❌ Error: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()
else:
    print('❌ No feature versions found!')
