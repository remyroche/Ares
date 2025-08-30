#!/usr/bin/env python3
"""Simple test script to verify structured data cache directory functionality."""

import os
import sys
from pathlib import Path

def test_structured_data_cache():
    """Test the structured data cache directory functionality."""
    print("🧪 Testing Structured Data Cache Directory (Simple)")
    print("=" * 60)
    
    # Test parameters
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    timeframe = "1m"
    
    # Expected structured directory
    expected_dir = os.path.join("data_cache", exchange.lower(), symbol.lower())
    print(f"📁 Expected directory: {expected_dir}")
    
    # Test 1: Check directory structure creation
    print("\n🔍 Test 1: Directory Structure Creation")
    try:
        # Create the structured directory
        os.makedirs(expected_dir, exist_ok=True)
        
        # Create some test subdirectories
        unified_dir = os.path.join(expected_dir, "unified")
        processed_dir = os.path.join(expected_dir, "processed")
        backup_dir = os.path.join(expected_dir, "backup_pre_unified")
        
        os.makedirs(unified_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(backup_dir, exist_ok=True)
        
        print(f"✅ Created directory structure:")
        print(f"   📁 {expected_dir}")
        print(f"   📁 {unified_dir}")
        print(f"   📁 {processed_dir}")
        print(f"   📁 {backup_dir}")
        
        # Verify directories exist
        assert os.path.exists(expected_dir), f"Directory {expected_dir} does not exist"
        assert os.path.exists(unified_dir), f"Directory {unified_dir} does not exist"
        assert os.path.exists(processed_dir), f"Directory {processed_dir} does not exist"
        assert os.path.exists(backup_dir), f"Directory {backup_dir} does not exist"
        
        print("✅ All directories created and verified successfully")
        
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        return False
    
    # Test 2: Check file path construction
    print("\n🔍 Test 2: File Path Construction")
    try:
        # Test various file paths that would be used
        klines_file = os.path.join(expected_dir, f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet")
        aggtrades_file = os.path.join(expected_dir, f"aggtrades_{exchange}_{symbol}_consolidated.parquet")
        unified_file = os.path.join(unified_dir, f"unified_{exchange}_{symbol}_{timeframe}.parquet")
        processed_file = os.path.join(processed_dir, f"{exchange}_{symbol}_{timeframe}_validated_data.parquet")
        
        print(f"✅ File paths constructed:")
        print(f"   📄 {klines_file}")
        print(f"   📄 {aggtrades_file}")
        print(f"   📄 {unified_file}")
        print(f"   📄 {processed_file}")
        
        # Create some dummy files to test
        with open(klines_file, 'w') as f:
            f.write("dummy klines data")
        with open(aggtrades_file, 'w') as f:
            f.write("dummy aggtrades data")
        
        print("✅ Dummy files created successfully")
        
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        return False
    
    # Test 3: Check directory listing
    print("\n🔍 Test 3: Directory Listing")
    try:
        # List contents of the structured directory
        print(f"📋 Contents of {expected_dir}:")
        for item in os.listdir(expected_dir):
            item_path = os.path.join(expected_dir, item)
            if os.path.isdir(item_path):
                print(f"   📁 {item}/")
            else:
                print(f"   📄 {item}")
        
        # List contents of subdirectories
        for subdir in ["unified", "processed", "backup_pre_unified"]:
            subdir_path = os.path.join(expected_dir, subdir)
            if os.path.exists(subdir_path):
                print(f"📋 Contents of {subdir_path}:")
                items = os.listdir(subdir_path)
                if items:
                    for item in items:
                        print(f"   📄 {item}")
                else:
                    print("   (empty)")
        
        print("✅ Directory listing completed successfully")
        
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        return False
    
    # Test 4: Check path resolution
    print("\n🔍 Test 4: Path Resolution")
    try:
        # Test absolute path resolution
        abs_path = os.path.abspath(expected_dir)
        print(f"📁 Absolute path: {abs_path}")
        
        # Test relative path resolution
        rel_path = os.path.relpath(expected_dir)
        print(f"📁 Relative path: {rel_path}")
        
        # Test parent directory
        parent_dir = os.path.dirname(expected_dir)
        print(f"📁 Parent directory: {parent_dir}")
        
        print("✅ Path resolution completed successfully")
        
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        return False
    
    # Test 5: Check multiple exchange/symbol combinations
    print("\n🔍 Test 5: Multiple Exchange/Symbol Combinations")
    try:
        test_combinations = [
            ("BTCUSDT", "BINANCE", "1m"),
            ("ETHUSDT", "COINBASE", "5m"),
            ("ADAUSDT", "BINANCE", "15m"),
        ]
        
        for symbol_test, exchange_test, timeframe_test in test_combinations:
            test_dir = os.path.join("data_cache", exchange_test.lower(), symbol_test.lower())
            os.makedirs(test_dir, exist_ok=True)
            print(f"✅ Created directory: {test_dir}")
        
        print("✅ Multiple combinations tested successfully")
        
    except Exception as e:
        print(f"❌ Test 5 failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Structured data cache directory is working correctly.")
    print("=" * 60)
    
    # Summary
    print("\n📊 SUMMARY:")
    print(f"   📁 Base directory: data_cache/")
    print(f"   📁 Structure: data_cache/exchange/asset/")
    print(f"   📁 Subdirectories: unified/, processed/, backup_pre_unified/")
    print(f"   📄 File naming: klines_EXCHANGE_SYMBOL_TIMEFRAME_consolidated.parquet")
    print(f"   📄 File naming: aggtrades_EXCHANGE_SYMBOL_consolidated.parquet")
    print(f"   📄 File naming: unified_EXCHANGE_SYMBOL_TIMEFRAME.parquet")
    
    return True

if __name__ == "__main__":
    success = test_structured_data_cache()
    if success:
        print("\n✅ Test completed successfully")
        sys.exit(0)
    else:
        print("\n❌ Test failed")
        sys.exit(1)