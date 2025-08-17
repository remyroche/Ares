#!/usr/bin/env python3
"""
Test script to verify HMM state naming fixes work correctly.
"""

import json
import sys
import os

sys.path.append("src")

from src.training.steps.step1_7_hmm_regime_discovery import (
    _name_states,
    _name_momentum_states,
    _name_volatility_states,
    _name_liquidity_states,
    _name_microstructure_states,
    _generate_archetype_descriptions,
)


def test_state_naming_fixes():
    """Test the state naming fixes with actual data."""

    # Load the actual HMM meta data
    meta_file = "data/training/BINANCE_ETHUSDT_hmm_composite_meta_1m.json"

    if not os.path.exists(meta_file):
        print(f"❌ Meta file not found: {meta_file}")
        return False

    with open(meta_file, "r") as f:
        meta_data = json.load(f)

    print("🔍 Testing HMM State Naming Fixes")
    print("=" * 60)

    # Test state naming for each block
    state_feature_medians = meta_data.get("state_feature_medians", {})

    for block_name in ["momentum", "volatility", "liquidity", "microstructure"]:
        print(f"\n📊 Testing {block_name.upper()} state naming:")
        print("-" * 40)

        if block_name not in state_feature_medians:
            print(f"❌ No data for {block_name}")
            continue

        # Convert string keys to int keys for the function
        medians = {int(k): v for k, v in state_feature_medians[block_name].items()}

        # Test the naming function
        try:
            new_names = _name_states(block_name, medians)

            print(f"✅ Generated {len(new_names)} state names:")
            for state_id, name in sorted(new_names.items()):
                print(f"   State {state_id}: {name}")

                # Show the actual feature values for context
                if state_id in medians:
                    features = medians[state_id]
                    feature_str = ", ".join(
                        [f"{k}: {v:.3f}" for k, v in features.items()]
                    )
                    print(f"      Features: {feature_str}")

        except Exception as e:
            print(f"❌ Error naming {block_name} states: {e}")
            return False

    # Test archetype description generation
    print(f"\n📝 Testing Archetype Description Generation:")
    print("-" * 40)

    try:
        # Get the required data for archetype descriptions
        cluster_centroids = meta_data.get("cluster_centroids", {})
        state_names = meta_data.get("state_names", {})
        cluster_counts = meta_data.get("combination_counts", {})

        # Convert cluster_counts to the expected format
        cluster_counts_int = {int(k): v for k, v in cluster_counts.items()}

        # Create dummy block_posteriors (not used in the current implementation)
        block_posteriors = {}

        new_descriptions = _generate_archetype_descriptions(
            cluster_centroids, state_names, block_posteriors, cluster_counts_int
        )

        print(f"✅ Generated {len(new_descriptions)} archetype descriptions:")
        for cluster_id in sorted(new_descriptions.keys())[:10]:  # Show first 10
            description = new_descriptions[cluster_id]
            print(f"   Archetype {cluster_id}: {description}")

        if len(new_descriptions) > 10:
            print(f"   ... and {len(new_descriptions) - 10} more descriptions")

    except Exception as e:
        print(f"❌ Error generating archetype descriptions: {e}")
        return False

    print(f"\n✅ All tests completed successfully!")
    return True


def compare_old_vs_new():
    """Compare old vs new state names."""

    # Load the actual HMM meta data
    meta_file = "data/training/BINANCE_ETHUSDT_hmm_composite_meta_1m.json"

    with open(meta_file, "r") as f:
        meta_data = json.load(f)

    print("\n🔄 Comparing Old vs New State Names:")
    print("=" * 60)

    state_names = meta_data.get("state_names", {})

    for block_name in ["momentum", "volatility", "liquidity", "microstructure"]:
        print(f"\n📊 {block_name.upper()} Block:")
        print("-" * 30)

        if block_name not in state_names:
            continue

        old_names = state_names[block_name]

        # Get new names using our fixed function
        state_feature_medians = meta_data.get("state_feature_medians", {})
        if block_name in state_feature_medians:
            medians = {int(k): v for k, v in state_feature_medians[block_name].items()}
            new_names = _name_states(block_name, medians)

            print(f"State | Old Name | New Name")
            print(f"------|----------|----------")
            for state_id in sorted(old_names.keys()):
                old_name = old_names[state_id]
                new_name = new_names.get(int(state_id), "N/A")
                print(f"{state_id:5} | {old_name:30} | {new_name}")

    # Compare archetype descriptions
    print(f"\n📝 Archetype Descriptions Comparison:")
    print("-" * 40)

    old_descriptions = meta_data.get("archetype_descriptions", {})

    # Get new descriptions
    cluster_centroids = meta_data.get("cluster_centroids", {})
    cluster_counts = meta_data.get("combination_counts", {})
    cluster_counts_int = {int(k): v for k, v in cluster_counts.items()}
    block_posteriors = {}

    new_descriptions = _generate_archetype_descriptions(
        cluster_centroids, state_names, block_posteriors, cluster_counts_int
    )

    print(f"Archetype | Old Description | New Description")
    print(f"----------|-----------------|-----------------")
    for cluster_id in sorted(old_descriptions.keys())[:5]:  # Show first 5
        old_desc = old_descriptions[cluster_id]
        new_desc = new_descriptions.get(cluster_id, "N/A")

        # Truncate for display
        old_short = old_desc[:50] + "..." if len(old_desc) > 50 else old_desc
        new_short = new_desc[:50] + "..." if len(new_desc) > 50 else new_desc

        print(f"{cluster_id:9} | {old_short:15} | {new_short}")


if __name__ == "__main__":
    print("🧪 HMM State Naming Fix Test")
    print("=" * 60)

    success = test_state_naming_fixes()

    if success:
        compare_old_vs_new()
        print(f"\n🎉 All tests passed! The fixes should work correctly.")
    else:
        print(f"\n❌ Tests failed. Please check the implementation.")
        sys.exit(1)
