#!/usr/bin/env python3
"""
Test script to verify enhanced debugging output for HDF5 versioned artifacts store.
This will help identify why the HDF5 store shows 0 versions despite being recently modified.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from utils.artifact_router import ArtifactRouter
from utils.versioned_artifacts.store import VersionedArtifactStore
from utils.versioned_artifacts.base_step_adapter import VersionedArtifactAdapter

def test_artifact_router():
    """Test the ArtifactRouter with enhanced debugging."""
    print("=" * 80)
    print("TESTING ARTIFACT ROUTER WITH ENHANCED DEBUGGING")
    print("=" * 80)
    
    # Create test data
    test_data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'target': np.random.choice([0, 1], 100)
    }, index=pd.date_range('2024-01-01', periods=100, freq='H'))
    
    # Initialize router
    router = ArtifactRouter(
        base_dir="test_artifacts",
        versioned_store_dir="test_versioned_artifacts",
        enable_versioned_artifacts=True
    )
    
    # Test context
    context = {
        'symbol': 'BTCUSDT',
        'exchange': 'binance',
        'timeframe': '1h',
        'direction': 'long',
        'model': 'analyst'
    }
    
    print("\n1. SAVING TEST DATA TO VERSIONED STORE")
    print("-" * 50)
    
    # Save test data
    save_path = router.save(
        data=test_data,
        artifact_name="test_features",
        data_category="features",
        context=context,
        metadata={'test': True, 'timestamp': datetime.now().isoformat()}
    )
    
    print(f"\nSave completed. Path: {save_path}")
    
    print("\n2. CHECKING STORE STATISTICS")
    print("-" * 50)
    
    # Get store statistics
    store = router._get_versioned_store(context)
    stats = store.get_statistics()
    print(f"Store statistics: {stats}")
    
    print("\n3. LISTING VERSIONS")
    print("-" * 50)
    
    # List versions
    versions = store.list_versions()
    print(f"Available versions: {versions}")
    
    print("\n4. LOADING TEST DATA")
    print("-" * 50)
    
    # Load test data
    try:
        loaded_data = router.load(
            artifact_name="test_features",
            data_category="features",
            context=context
        )
        print(f"Successfully loaded data with shape: {loaded_data.shape}")
        print(f"Data columns: {list(loaded_data.columns)}")
    except Exception as e:
        print(f"Error loading data: {e}")
    
    print("\n5. TESTING DIRECT STORE ACCESS")
    print("-" * 50)
    
    # Test direct store access
    direct_store = VersionedArtifactStore(
        store_path="test_versioned_artifacts/BTCUSDT_binance_1h_long_analyst",
        auto_version=True,
        enable_row_versioning=True
    )
    
    direct_stats = direct_store.get_statistics()
    print(f"Direct store statistics: {direct_stats}")
    
    direct_versions = direct_store.list_versions()
    print(f"Direct store versions: {direct_versions}")
    
    print("\n6. TESTING BASE STEP ADAPTER")
    print("-" * 50)
    
    # Test base step adapter
    adapter = VersionedArtifactAdapter(
        store_dir="test_versioned_artifacts",
        symbol='ETHUSDT',
        exchange='binance',
        timeframe='15m',
        direction='short',
        model='tactician'
    )
    
    # Save via adapter
    adapter_path = adapter.save(
        data=test_data,
        artifact_name="adapter_test_features",
        metadata={'adapter_test': True}
    )
    
    print(f"Adapter save path: {adapter_path}")
    
    # Get adapter statistics
    adapter_stats = adapter.get_statistics()
    print(f"Adapter statistics: {adapter_stats}")
    
    print("\n7. CHECKING FILE SYSTEM")
    print("-" * 50)
    
    # Check file system
    versioned_dir = Path("test_versioned_artifacts")
    if versioned_dir.exists():
        print(f"Versioned artifacts directory exists: {versioned_dir}")
        for store_dir in versioned_dir.iterdir():
            if store_dir.is_dir():
                print(f"\nStore directory: {store_dir}")
                h5_file = store_dir / "store.h5"
                metadata_file = store_dir / "metadata.json"
                
                if h5_file.exists():
                    print(f"  HDF5 file exists: {h5_file} (size: {h5_file.stat().st_size} bytes)")
                else:
                    print(f"  HDF5 file MISSING: {h5_file}")
                
                if metadata_file.exists():
                    print(f"  Metadata file exists: {metadata_file} (size: {metadata_file.stat().st_size} bytes)")
                    # Read and display metadata
                    import json
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        print(f"  Metadata versions: {list(metadata.get('versions', {}).keys())}")
                        print(f"  Current version: {metadata.get('current_version')}")
                else:
                    print(f"  Metadata file MISSING: {metadata_file}")
    else:
        print("Versioned artifacts directory does not exist!")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    test_artifact_router()