#!/usr/bin/env python3
"""
Simple test script to run step1_7_hmm_regime_discovery directly
"""

import asyncio
import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


async def main():
    print("🚀 Starting step1_7_hmm_regime_discovery test...")

    try:
        from src.training.steps.step1_7_hmm_regime_discovery import run_step

        print("✅ Imported step1_7_hmm_regime_discovery successfully")

        # Run the step
        result = await run_step(
            symbol="ETHUSDT",
            exchange="BINANCE",
            data_dir="data/training",
            timeframe="1m",
            lookback_days=30,
            force_rerun=True,
        )

        print(f"✅ Step1_7 result: {result}")

        if result:
            print("✅ Step1_7 completed successfully!")
        else:
            print("❌ Step1_7 failed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
