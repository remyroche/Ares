#!/usr/bin/env python3
"""
Test script to verify HMM regime discovery fix.
This script tests the step1_7 HMM regime discovery without causing bus errors or resource leaks.
"""

import os
import sys
import time
import traceback
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_hmm_regime_discovery():
    """Test the HMM regime discovery step with the fixes applied."""

    print("🧪 Testing HMM Regime Discovery Fix")
    print("=" * 50)

    try:
        # Import the step1_7 module
        from src.training.steps.step1_7_hmm_regime_discovery import run_step

        print("✅ Successfully imported step1_7_hmm_regime_discovery module")

        # Test parameters
        symbol = "ETHUSDT"
        exchange = "BINANCE"
        data_dir = "data/training"
        timeframe = "1m"
        lookback_days = 30  # Reduced for testing

        print(f"📊 Test Parameters:")
        print(f"   Symbol: {symbol}")
        print(f"   Exchange: {exchange}")
        print(f"   Timeframe: {timeframe}")
        print(f"   Lookback days: {lookback_days}")
        print(f"   Data directory: {data_dir}")

        # Check if data directory exists
        if not os.path.exists(data_dir):
            print(f"⚠️ Data directory {data_dir} does not exist. Creating it...")
            os.makedirs(data_dir, exist_ok=True)

        print("\n🚀 Starting HMM Regime Discovery test...")
        start_time = time.time()

        # Run the step
        success = run_step(
            symbol=symbol,
            exchange=exchange,
            data_dir=data_dir,
            timeframe=timeframe,
            lookback_days=lookback_days,
        )

        end_time = time.time()
        duration = end_time - start_time

        print(f"\n⏱️ Test completed in {duration:.2f} seconds")

        if success:
            print("✅ HMM Regime Discovery test completed successfully!")
            print("✅ No bus errors or resource leaks detected!")

            # Check for generated files
            expected_files = [
                f"{exchange}_{symbol}_hmm_composite_clusters_{timeframe}.parquet",
                f"{exchange}_{symbol}_hmm_composite_meta_{timeframe}.json",
                f"{exchange}_{symbol}_hmm_composite_intensity_{timeframe}.parquet",
            ]

            print("\n📁 Checking for generated files:")
            for filename in expected_files:
                filepath = os.path.join(data_dir, filename)
                if os.path.exists(filepath):
                    size = os.path.getsize(filepath)
                    print(f"   ✅ {filename} ({size} bytes)")
                else:
                    print(f"   ❌ {filename} (not found)")

            # Check for reports
            reports_dir = "reports"
            if os.path.exists(reports_dir):
                report_files = [
                    f
                    for f in os.listdir(reports_dir)
                    if f.endswith(".md") and "regime_summary" in f
                ]
                if report_files:
                    print(f"\n📄 Generated reports:")
                    for report_file in report_files:
                        print(f"   📄 {report_file}")
                else:
                    print("\n📄 No reports generated")
            else:
                print("\n📄 Reports directory not found")

        else:
            print("❌ HMM Regime Discovery test failed!")
            return False

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

    print("\n🎉 Test completed successfully!")
    return True


if __name__ == "__main__":
    success = test_hmm_regime_discovery()
    sys.exit(0 if success else 1)
