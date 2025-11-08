#!/usr/bin/env python3
"""
Simple test script to check if labeling integration step is working correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training.steps.pre_training.feature_generation_labeling_integration_step import FeatureGenerationLabelingIntegrationStep
from utils.versioned_artifacts.store import VersionedArtifactStore
import json

async def test_labeling_step():
    """Test labeling integration step."""
    print("🧪 Testing feature_generation_labeling_integration_step...")
    
    # Initialize step
    step = FeatureGenerationLabelingIntegrationStep()
    
    # Test configuration
    config = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'execution_mode': 'light'
    }
    
    # Run step
    result = await step.execute(config)
    
    if result.get('success', False):
        print("✅ Labeling integration step completed successfully")
        
        # Check if target columns were saved
        store_path = 'versioned_artifacts/UNKNOWN_binance_15m_long_analyst'
        store = VersionedArtifactStore(store_path)
        
        # Get latest version
        versions = store.list_versions()
        if versions:
            latest_version = sorted(versions)[-1]
            print(f"📋 Latest version: {latest_version}")
            
            # Get version info
            version_info = store.get_version_info(latest_version)
            columns = version_info.get('columns', [])
            
            # Check for target columns
            target_cols = [col for col in columns if 'target' in col]
            price_target_cols = [col for col in columns if 'price_target' in col]
            
            print(f"🎯 Target columns: {target_cols}")
            print(f"📊 Price target columns: {price_target_cols}")
            
            if target_cols:
                print("✅ Target columns found in versioned artifacts")
            else:
                print("❌ No target columns found in versioned artifacts")
        else:
            print("❌ No versions found in versioned artifacts")
    else:
        print(f"❌ Labeling integration step failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_labeling_step())
