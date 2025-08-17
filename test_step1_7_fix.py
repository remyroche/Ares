#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.training.steps.step1_7_hmm_regime_discovery import run_step


async def main():
    print("🚀 Testing Step 1.7 HMM Regime Discovery with fix...")

    # Run step1_7 with force_rerun=True to test the resampling logic
    success = await run_step(
        symbol="ETHUSDT",
        exchange="BINANCE",
        data_dir="data/training",
        timeframe="1m",
        lookback_days=180,
        force_rerun=True,  # Force rerun to test the resampling logic
        cluster_algorithm="kmeans",
        target_num_clusters=20,
        min_combination_frequency=0.003,
    )

    if success:
        print("✅ Step 1.7 completed successfully!")
    else:
        print("❌ Step 1.7 failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
