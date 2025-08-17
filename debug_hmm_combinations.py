#!/usr/bin/env python3
"""
Debug HMM Combinations Script

This script analyzes the HMM regime discovery process to understand why
0 distinct market archetypes are being found.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def analyze_hmm_combinations(symbol="ETHUSDT", exchange="BINANCE", timeframe="1m"):
    """Analyze HMM combinations to understand the clustering issue."""

    data_dir = "data/training"

    # Check if HMM block states exist
    block_states_path = os.path.join(
        data_dir, f"{exchange}_{symbol}_hmm_block_states_{timeframe}.parquet"
    )
    if not os.path.exists(block_states_path):
        print(f"❌ HMM block states file not found: {block_states_path}")
        return

    print(f"📊 Analyzing HMM combinations for {exchange}_{symbol}_{timeframe}")
    print("=" * 80)

    # Load block states
    block_df = pd.read_parquet(block_states_path)
    print(f"📈 Loaded {len(block_df)} rows of HMM block states")

    # Extract state columns
    state_cols = [col for col in block_df.columns if col.endswith("_state_id")]
    print(f"🔍 Found {len(state_cols)} state columns: {state_cols}")

    # Analyze each block's states
    for col in state_cols:
        block_name = col.replace("_state_id", "")
        states = block_df[col].dropna()
        unique_states = states.unique()
        state_counts = states.value_counts()

        print(f"\n📊 Block: {block_name}")
        print(f"   Unique states: {len(unique_states)} - {sorted(unique_states)}")
        print(f"   State distribution:")
        for state, count in state_counts.items():
            percentage = (count / len(states)) * 100
            print(f"     State {state}: {count} ({percentage:.2f}%)")

    # Create combinations
    print(f"\n🔗 Creating combinations...")
    combination_keys = []
    for idx in range(len(block_df)):
        key_parts = []
        for col in state_cols:
            state_val = block_df.iloc[idx][col]
            if pd.notna(state_val):
                block_name = col.replace("_state_id", "")
                key_parts.append(f"{block_name}:{int(state_val)}")
        if key_parts:
            combination_keys.append("|".join(key_parts))
        else:
            combination_keys.append("")

    combo_series = pd.Series(combination_keys)
    combo_counts = combo_series.value_counts()

    print(f"📊 Combination Analysis:")
    print(f"   Total combinations: {len(combo_series)}")
    print(f"   Unique combinations: {len(combo_counts)}")
    print(f"   Non-empty combinations: {len(combo_series[combo_series != ''])}")

    # Show top combinations
    print(f"\n🏆 Top 10 combinations:")
    for i, (combo, count) in enumerate(combo_counts.head(10).items()):
        percentage = (count / len(combo_series)) * 100
        print(f"   {i+1}. {combo}: {count} ({percentage:.3f}%)")

    # Analyze filtering threshold
    min_count_threshold = max(5, int(0.005 * len(combo_series)))
    print(f"\n🎯 Filtering Analysis:")
    print(f"   Minimum count threshold: {min_count_threshold}")
    print(
        f"   Combinations above threshold: {len(combo_counts[combo_counts >= min_count_threshold])}"
    )
    print(
        f"   Combinations below threshold: {len(combo_counts[combo_counts < min_count_threshold])}"
    )

    # Show combinations that would be kept
    kept_combos = combo_counts[combo_counts >= min_count_threshold]
    print(f"\n✅ Combinations that would be kept:")
    for combo, count in kept_combos.items():
        percentage = (count / len(combo_series)) * 100
        print(f"   {combo}: {count} ({percentage:.3f}%)")

    # Check if we have enough data for clustering
    if len(kept_combos) == 0:
        print(f"\n❌ PROBLEM: No combinations meet the minimum threshold!")
        print(f"   This is why 0 distinct market archetypes are found.")
        print(f"   The threshold of {min_count_threshold} is too high for the data.")

        # Suggest solutions
        print(f"\n💡 Suggested solutions:")
        print(f"   1. Lower the minimum count threshold")
        print(f"   2. Increase the data size")
        print(f"   3. Reduce the number of HMM states per block")
        print(f"   4. Use a different clustering approach")

        # Calculate what threshold would work
        if len(combo_counts) > 0:
            max_count = combo_counts.max()
            suggested_threshold = max(1, max_count // 10)  # Use 10% of max count
            print(f"\n   Suggested threshold: {suggested_threshold}")
            print(
                f"   This would keep {len(combo_counts[combo_counts >= suggested_threshold])} combinations"
            )

    elif len(kept_combos) < 2:
        print(
            f"\n⚠️ WARNING: Only {len(kept_combos)} combination(s) meet the threshold!"
        )
        print(f"   This may not be enough for meaningful clustering.")

    else:
        print(f"\n✅ SUCCESS: {len(kept_combos)} combinations meet the threshold")
        print(f"   This should be sufficient for clustering.")


def main():
    """Main function to run the analysis."""
    print("🔍 HMM Combinations Debug Analysis")
    print("=" * 80)

    # Analyze for different timeframes
    timeframes = ["1m", "5m", "15m", "30m"]

    for tf in timeframes:
        print(f"\n{'='*20} TIMEFRAME: {tf} {'='*20}")
        analyze_hmm_combinations(timeframe=tf)
        print()


if __name__ == "__main__":
    main()
