#!/usr/bin/env python3
"""
Test script for LiveDataCollector

Run this to verify the live data collection system works correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.trading.data.live_data_collector import (
    LiveDataCollector,
    LiveDataConfig,
    CollectionMode,
    DataQuality,
    CollectionInterval
)

async def test_data_callback(data_point):
    """Test callback for new data."""
    print(f"📊 Received data: {data_point.timestamp} | Price: ${data_point.raw_data['close']:.2f}")

    if data_point.processed_data:
        print(f"   Processed features: {len(data_point.processed_data)} features")

    if data_point.ml_predictions:
        pred = data_point.ml_predictions['prediction']
        conf = data_point.ml_predictions['confidence']
        print(f"   ML Prediction: {pred} (confidence: {conf:.2f})")

async def test_simulated_collection():
    """Test with simulated data."""
    print("🧪 Testing simulated data collection...")

    config = LiveDataConfig(
        symbol="ETH",
        exchange="binance",
        interval=CollectionInterval.FAST,  # 15s interval (faster for testing)
        collection_mode=CollectionMode.SIMULATED,
        quality_level=DataQuality.MEDIUM,
        enable_ml_predictions=False,  # Skip ML for basic test
        buffer_size=50
    )

    collector = LiveDataCollector(config)
    collector.add_data_callback(test_data_callback)

    # Start collection
    success = await collector.start_collection()
    if not success:
        print("❌ Failed to start collection")
        return False

    # Run for 45 seconds (3 x 15s intervals)
    print("⏳ Running simulated collection for 45 seconds...")
    await asyncio.sleep(45)

    # Stop collection
    await collector.stop_collection()

    # Check results
    stats = collector.get_stats()
    print(f"📈 Final stats: {stats}")

    recent_data = collector.get_recent_data(5)
    print(f"📊 Recent data points: {len(recent_data)}")

    if len(recent_data) > 0:
        print("✅ Simulated collection test PASSED")
        return True
    else:
        print("❌ Simulated collection test FAILED")
        return False

async def test_live_collection_dry_run():
    """Test live collection initialization (dry run)."""
    print("🧪 Testing live collection initialization...")

    config = LiveDataConfig(
        symbol="ETH",
        exchange="binance",
        interval=CollectionInterval.STANDARD,  # 30 seconds
        collection_mode=CollectionMode.LIVE,
        quality_level=DataQuality.HIGH,
        enable_ml_predictions=False,  # Skip ML for test
        buffer_size=100
    )

    try:
        collector = LiveDataCollector(config)
        print("✅ LiveDataCollector initialization PASSED")

        # Test stats
        stats = collector.get_stats()
        print(f"📊 Initial stats: {stats}")

        return True

    except Exception as e:
        print(f"❌ Live collection initialization FAILED: {e}")
        return False

async def main():
    """Run all tests."""
    print("🚀 LiveDataCollector Test Suite")
    print("=" * 40)

    # Test 1: Simulated collection
    print("\n1️⃣ Testing Simulated Data Collection")
    sim_result = await test_simulated_collection()

    # Test 2: Live collection dry run
    print("\n2️⃣ Testing Live Collection Initialization")
    live_result = await test_live_collection_dry_run()

    # Summary
    print("\n" + "=" * 40)
    print("📋 Test Results:")
    print(f"   Simulated Collection: {'✅ PASS' if sim_result else '❌ FAIL'}")
    print(f"   Live Init (Dry Run): {'✅ PASS' if live_result else '❌ FAIL'}")

    if sim_result and live_result:
        print("\n🎉 All tests PASSED! LiveDataCollector is ready for use.")
        print("\n💡 Next steps:")
        print("   1. Configure your Binance API keys in config/config.yaml")
        print("   2. Train/save your ML model (see models/ directory)")
        print("   3. Run example_live_trading_analysis.py for live trading")
        return True
    else:
        print("\n❌ Some tests FAILED. Check the errors above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
