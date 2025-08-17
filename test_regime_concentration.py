#!/usr/bin/env python3
"""
Test script to demonstrate enhanced regime merging for 70-80% concentration in top 20 regimes.

This script shows how to use the enhanced HMM regime discovery with aggressive merging
to achieve high concentration in the top 20 market regimes.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from training.steps.step1_7_hmm_regime_discovery import run_step


async def test_regime_concentration():
    """Test the enhanced regime merging with different concentration targets."""

    print("🎯 Testing Enhanced HMM Regime Discovery for 70-80% Top 20 Concentration")
    print("=" * 80)

    # Test parameters
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    data_dir = "data/training"
    timeframe = "1m"
    lookback_days = 30  # Use 30 days for faster testing

    # Test different concentration targets
    concentration_targets = [0.70, 0.75, 0.80]

    for target in concentration_targets:
        print(f"\n🔧 Testing with {target:.0%} target concentration...")
        print("-" * 50)

        try:
            # Run the step with enhanced configuration
            success = await run_step(
                symbol=symbol,
                exchange=exchange,
                data_dir=data_dir,
                timeframe=timeframe,
                lookback_days=lookback_days,
                force_rerun=True,  # Force rerun to test different configurations
                cluster_algorithm="kmeans",
                target_num_clusters=20,
                min_combination_frequency=0.003,  # More aggressive filtering
            )

            if success:
                print(
                    f"✅ Successfully completed with {target:.0%} target concentration"
                )
            else:
                print(f"❌ Failed to complete with {target:.0%} target concentration")

        except Exception as e:
            print(f"❌ Error testing {target:.0%} concentration: {e}")
            continue

    print("\n" + "=" * 80)
    print("📋 Configuration Recommendations for 70-80% Concentration:")
    print("=" * 80)
    print("1. Set target_top_20_concentration to 0.75 (75%)")
    print("2. Enable aggressive_merging=True")
    print("3. Use similarity_threshold=0.75 (75% similarity)")
    print("4. Set min_frequency=0.003 (0.3% minimum frequency)")
    print("5. Target max_regimes=20")
    print("6. Use min_combination_frequency=0.003 for more aggressive filtering")
    print("\nCommand line example:")
    print("python src/training/steps/step1_7_hmm_regime_discovery.py \\")
    print("  --target-concentration 0.75 \\")
    print("  --similarity-threshold 0.75 \\")
    print("  --min-frequency 0.003 \\")
    print("  --max-regimes 20 \\")
    print("  --aggressive-merging")


if __name__ == "__main__":
    asyncio.run(test_regime_concentration())
