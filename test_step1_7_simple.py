#!/usr/bin/env python3
"""
Simple test script to isolate step1_7_hmm_regime_discovery execution
"""

import asyncio
import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


async def main():
    print("🚀 Starting simple step1_7_hmm_regime_discovery test...")

    try:
        # Test 1: Just import the module
        print("📦 Testing import...")
        from src.training.steps.step1_7_hmm_regime_discovery import run_step

        print("✅ Import successful")

        # Test 2: Check if the function exists
        print("🔍 Testing function existence...")
        if hasattr(run_step, "__call__"):
            print("✅ Function exists and is callable")
        else:
            print("❌ Function is not callable")
            return

        # Test 3: Try to call the function with minimal parameters
        print("🎯 Testing function call...")
        result = await run_step(symbol="ETHUSDT", exchange="BINANCE", force_rerun=True)

        print(f"✅ Function completed with result: {result}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
