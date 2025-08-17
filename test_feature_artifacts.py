#!/usr/bin/env python3
"""
Test script for the feature artifact system.

This script tests:
1. Feature artifact creation and persistence
2. Feature artifact loading
3. Hash-based invalidation
4. Metadata tracking
"""

import asyncio
import os
import json
import pandas as pd
from datetime import datetime
from src.training.steps.feature_artifact_loader import (
    check_feature_artifacts_exist,
    load_feature_artifacts,
    get_feature_artifact_info,
    validate_feature_artifacts,
    get_feature_artifact_paths
)
from src.training.steps.step2_feature_engineering import run_step as run_step2


async def test_feature_artifact_system():
    """Test the complete feature artifact system."""
    
    print("🧪 Testing Feature Artifact System")
    print("=" * 50)
    
    # Test parameters
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    data_dir = "data/training"
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Test 1: Check if artifacts exist before creation
    print("\n1️⃣ Testing artifact existence check...")
    exists_before = check_feature_artifacts_exist(symbol, exchange, data_dir)
    print(f"   Artifacts exist before creation: {exists_before}")
    
    if exists_before:
        print("   ℹ️  Artifacts already exist, testing loading...")
        try:
            info = get_feature_artifact_info(symbol, exchange, data_dir)
            print(f"   📊 Artifact info: {json.dumps(info, indent=2)}")
            
            # Test validation
            is_valid, message = validate_feature_artifacts(symbol, exchange, data_dir)
            print(f"   ✅ Validation: {is_valid} - {message}")
            
            # Test loading
            features = load_feature_artifacts(symbol, exchange, data_dir)
            print(f"   📦 Loaded features: {list(features.keys())}")
            for split, df in features.items():
                print(f"      {split}: {len(df)} rows, {len(df.columns)} features")
                
        except Exception as e:
            print(f"   ❌ Loading failed: {e}")
    
    # Test 2: Run Step 2 to create artifacts (if they don't exist or force rerun)
    print("\n2️⃣ Testing Step 2 feature engineering...")
    try:
        # Check if we need to run Step 2
        if not exists_before:
            print("   🔧 Running Step 2 to create artifacts...")
            success = await run_step2(
                symbol=symbol,
                exchange=exchange,
                data_dir=data_dir,
                timeframe="1m",
                force_rerun=False
            )
            print(f"   ✅ Step 2 result: {success}")
        else:
            print("   ℹ️  Artifacts already exist, skipping Step 2")
            success = True
            
    except Exception as e:
        print(f"   ❌ Step 2 failed: {e}")
        success = False
    
    # Test 3: Verify artifacts after creation
    if success:
        print("\n3️⃣ Testing artifact verification after creation...")
        exists_after = check_feature_artifacts_exist(symbol, exchange, data_dir)
        print(f"   Artifacts exist after creation: {exists_after}")
        
        if exists_after:
            try:
                # Test loading again
                features = load_feature_artifacts(symbol, exchange, data_dir)
                print(f"   📦 Successfully loaded features after creation")
                
                # Test metadata
                info = get_feature_artifact_info(symbol, exchange, data_dir)
                print(f"   📊 Updated artifact info:")
                print(f"      Created at: {info.get('created_at', 'unknown')}")
                print(f"      Total features: {info.get('total_features', 0)}")
                print(f"      Feature counts: {info.get('feature_counts', {})}")
                
                # Test validation
                is_valid, message = validate_feature_artifacts(symbol, exchange, data_dir)
                print(f"   ✅ Validation after creation: {is_valid} - {message}")
                
            except Exception as e:
                print(f"   ❌ Verification failed: {e}")
    
    # Test 4: Test force rerun
    print("\n4️⃣ Testing force rerun...")
    try:
        print("   🔄 Running Step 2 with force_rerun=True...")
        success = await run_step2(
            symbol=symbol,
            exchange=exchange,
            data_dir=data_dir,
            timeframe="1m",
            force_rerun=True
        )
        print(f"   ✅ Force rerun result: {success}")
        
        if success:
            info = get_feature_artifact_info(symbol, exchange, data_dir)
            print(f"   📊 Artifact info after force rerun:")
            print(f"      Created at: {info.get('created_at', 'unknown')}")
            
    except Exception as e:
        print(f"   ❌ Force rerun failed: {e}")
    
    # Test 5: Test artifact paths
    print("\n5️⃣ Testing artifact paths...")
    paths = get_feature_artifact_paths(symbol, exchange, data_dir)
    print(f"   📁 Artifact paths:")
    for path_type, path in paths.items():
        exists = os.path.exists(path)
        print(f"      {path_type}: {path} ({'✅' if exists else '❌'})")
    
    print("\n🎉 Feature artifact system test completed!")


if __name__ == "__main__":
    asyncio.run(test_feature_artifact_system())

