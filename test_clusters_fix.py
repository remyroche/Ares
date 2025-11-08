#!/usr/bin/env python3
"""
Test script to verify that the fix for clusters/regimes works correctly.
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

def test_clusters_fix():
    """Test that clusters/regimes are handled correctly."""
    print("=" * 80)
    print("TESTING CLUSTERS/REGIMES FIX")
    print("=" * 80)
    
    # Create test data
    test_data1 = pd.DataFrame({
        'cluster1': np.random.randn(50),
        'cluster2': np.random.randn(50),
        'target': np.random.choice([0, 1], 50)
    }, index=pd.date_range('2024-01-01', periods=50, freq='H'))
    
    test_data2 = pd.DataFrame({
        'regime1': np.random.randn(50),
        'regime2': np.random.randn(50),
        'target': np.random.choice([0, 1], 50)
    }, index=pd.date_range('2024-01-01', periods=50, freq='H'))
    
    # Initialize router
    router = ArtifactRouter(
        base_dir="test_artifacts",
        versioned_store_dir="test_versioned_artifacts",
        enable_versioned_artifacts=True
    )
    
    # Test context 1 - Regular features
    context1 = {
        'symbol': 'BTCUSDT',
        'exchange': 'binance',
        'timeframe': '1h',
        'direction': 'long',
        'model': 'analyst'
    }
    
    # Test context 2 - Clusters
    context2 = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'direction': 'short',
        'model': 'tactician'
    }
    
    # Test context 3 - Regimes
    context3 = {
        'symbol': 'ADAUSDT',
        'exchange': 'binance',
        'timeframe': '4h',
        'direction': 'long',
        'model': 'analyst'
    }
    
    print("\n1. SAVING TEST DATA TO DIFFERENT STORES")
    print("-" * 50)
    
    # Save data to different stores
    save_path1 = router.save(
        data=test_data1,
        artifact_name="test_features",
        data_category="features",
        context=context1,
        metadata={'test': True, 'type': 'features'}
    )
    
    save_path2 = router.save(
        data=test_data2,
        artifact_name="test_clusters",
        data_category="clusters",
        context=context2,
        metadata={'test': True, 'type': 'clusters'}
    )
    
    save_path3 = router.save(
        data=test_data2,
        artifact_name="test_regimes",
        data_category="regimes",
        context=context3,
        metadata={'test': True, 'type': 'regimes'}
    )
    
    # Save labels to a different store
    save_path4 = router.save(
        data=test_data2,
        artifact_name="test_labels",
        data_category="labels",
        context=context2,  # Use ETHUSDT context instead of BTCUSDT
        metadata={'test': True, 'type': 'labels'}
    )
    
    print(f"Saved features to: {save_path1}")
    print(f"Saved clusters to: {save_path2}")
    print(f"Saved regimes to: {save_path3}")
    print(f"Saved labels to: {save_path4}")
    
    print("\n2. TESTING list_all_versions() WITH DIFFERENT FILTERS")
    print("-" * 50)
    
    # Test listing all versions (no filter)
    all_versions = router.list_all_versions()
    print(f"All versions (no filter): {len(all_versions)}")
    print(f"Version names: {all_versions}")
    
    # Test listing only clusters
    cluster_versions = router.list_all_versions(artifact_type="clusters")
    print(f"Cluster versions: {len(cluster_versions)}")
    print(f"Version names: {cluster_versions}")
    
    # Test listing only regimes
    regime_versions = router.list_all_versions(artifact_type="regimes")
    print(f"Regime versions: {len(regime_versions)}")
    print(f"Version names: {regime_versions}")
    
    # Test listing only features
    feature_versions = router.list_all_versions(artifact_type="features")
    print(f"Feature versions: {len(feature_versions)}")
    print(f"Version names: {feature_versions}")
    
    # Test listing only labels
    label_versions = router.list_all_versions(artifact_type="labels")
    print(f"Label versions: {len(label_versions)}")
    print(f"Version names: {label_versions}")
    
    print("\n3. TESTING BASE STEP ADAPTER")
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
    
    # Test listing all versions with adapter
    adapter_all_versions = adapter1.list_all_versions()
    print(f"Adapter all versions: {len(adapter_all_versions)}")
    print(f"Version names: {adapter_all_versions}")
    
    # Test listing only clusters with adapter
    adapter_cluster_versions = adapter1.list_all_versions(artifact_type="clusters")
    print(f"Adapter cluster versions: {len(adapter_cluster_versions)}")
    print(f"Version names: {adapter_cluster_versions}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    test_clusters_fix()