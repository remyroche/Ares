#!/usr/bin/env python3
"""
Test script to verify the HDF5 versioned artifacts store fix.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

from utils.artifact_router import ArtifactRouter
from utils.versioned_artifacts.base_step_adapter import VersionedArtifactAdapter

def test_fix():
    """Test the fix for multiple store instances."""
    print("=" * 80)
    print("TESTING HDF5 VERSIONED ARTIFACTS STORE FIX")
    print("=" * 80)
    
    # Create test data
    test_data1 = pd.DataFrame({
        'feature1': np.random.randn(50),
        'feature2': np.random.randn(50),
        'target': np.random.choice([0, 1], 50)
    }, index=pd.date_range('2024-01-01', periods=50, freq='H'))
    
    test_data2 = pd.DataFrame({
        'feature3': np.random.randn(50),
        'feature4': np.random.randn(50),
        'target': np.random.choice([0, 1], 50)
    }, index=pd.date_range('2024-01-01', periods=50, freq='H'))
    
    # Initialize router
    router = ArtifactRouter(
        base_dir="test_artifacts",
        versioned_store_dir="test_versioned_artifacts",
        enable_versioned_artifacts=True
    )
    
    # Test context 1
    context1 = {
        'symbol': 'BTCUSDT',
        'exchange': 'binance',
        'timeframe': '1h',
        'direction': 'long',
        'model': 'analyst'
    }
    
    # Test context 2
    context2 = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'direction': 'short',
        'model': 'tactician'
    }
    
    print("\n1. SAVING DATA TO TWO DIFFERENT STORES")
    print("-" * 50)
    
    # Save data to two different stores
    save_path1 = router.save(
        data=test_data1,
        artifact_name="test_features_btc",
        data_category="features",
        context=context1,
        metadata={'test': True, 'store': 'btc'}
    )
    
    save_path2 = router.save(
        data=test_data2,
        artifact_name="test_features_eth",
        data_category="features",
        context=context2,
        metadata={'test': True, 'store': 'eth'}
    )
    
    print(f"Saved BTC data to: {save_path1}")
    print(f"Saved ETH data to: {save_path2}")
    
    print("\n2. TESTING INDIVIDUAL STORE STATISTICS")
    print("-" * 50)
    
    # Get individual stores
    store1 = router._get_versioned_store(context1)
    store2 = router._get_versioned_store(context2)
    
    # Get statistics for each store
    stats1 = store1.get_statistics()
    stats2 = store2.get_statistics()
    
    print(f"Store 1 (BTC) statistics: {stats1['num_versions']} versions")
    print(f"Store 2 (ETH) statistics: {stats2['num_versions']} versions")
    
    print("\n3. TESTING NEW list_all_versions() METHOD")
    print("-" * 50)
    
    # Test the new method
    all_versions = router.list_all_versions()
    print(f"All versions across all stores: {len(all_versions)}")
    print(f"Version names: {all_versions}")
    
    print("\n4. TESTING BASE STEP ADAPTER")
    print("-" * 50)
    
    # Test base step adapter
    adapter1 = VersionedArtifactAdapter(
        store_dir="test_versioned_artifacts",
        symbol='BTCUSDT',
        exchange='binance',
        timeframe='1h',
        direction='long',
        model='analyst'
    )
    
    adapter2 = VersionedArtifactAdapter(
        store_dir="test_versioned_artifacts",
        symbol='ETHUSDT',
        exchange='binance',
        timeframe='15m',
        direction='short',
        model='tactician'
    )
    
    # Save data via adapters
    adapter1.save(
        data=test_data1,
        artifact_name="adapter_test_btc",
        metadata={'adapter_test': True}
    )
    
    adapter2.save(
        data=test_data2,
        artifact_name="adapter_test_eth",
        metadata={'adapter_test': True}
    )
    
    # Test the new method
    adapter_all_versions = adapter1.list_all_versions()
    print(f"Adapter all versions: {len(adapter_all_versions)}")
    print(f"Version names: {adapter_all_versions}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    test_fix()