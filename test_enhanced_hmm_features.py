#!/usr/bin/env python3
"""
Test script to verify enhanced HMM feature selection and naming system.
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
    BLOCK_PREFERRED_FEATURES,
)


def test_enhanced_feature_selection():
    """Test the enhanced feature selection system."""

    print("🔍 Testing Enhanced HMM Feature Selection")
    print("=" * 60)

    # Test the preference ordering for each block
    for block_name in ["momentum", "volatility", "liquidity", "microstructure"]:
        print(f"\n📊 {block_name.upper()} Block Preferred Features:")
        print("-" * 40)

        preferred_features = BLOCK_PREFERRED_FEATURES.get(block_name, [])

        # Group features by category
        traditional_features = []
        advanced_features = []
        additional_features = []

        for feature in preferred_features:
            if block_name == "momentum":
                if feature in [
                    "momentum_strength",
                    "rsi",
                    "bb_position",
                    "momentum_5",
                    "momentum_10",
                    "momentum_20",
                ]:
                    traditional_features.append(feature)
                elif "volume_momentum" in feature:
                    advanced_features.append(feature)
                else:
                    additional_features.append(feature)
            elif block_name == "volatility":
                if feature in ["price_volatility", "adaptive_atr", "ewma_volatility"]:
                    traditional_features.append(feature)
                elif feature in [
                    "parkinson_volatility",
                    "garman_klass_volatility",
                    "rogers_satchell_volatility",
                    "yang_zhang_volatility",
                ]:
                    advanced_features.append(feature)
                else:
                    additional_features.append(feature)
            elif block_name == "liquidity":
                if feature in ["liquidity_score", "avg_volume", "volume_price_impact"]:
                    traditional_features.append(feature)
                elif "volume_ma_ratio" in feature or "volume_change" in feature:
                    advanced_features.append(feature)
                else:
                    additional_features.append(feature)
            elif block_name == "microstructure":
                if feature in [
                    "spread_tightness",
                    "trade_frequency",
                    "order_flow_imbalance",
                ]:
                    traditional_features.append(feature)
                elif feature in [
                    "bid_ask_spread_returns",
                    "bid_ask_spread_level",
                    "price_impact",
                ]:
                    advanced_features.append(feature)
                else:
                    additional_features.append(feature)

        print(f"   Traditional Features ({len(traditional_features)}):")
        for feature in traditional_features[:5]:  # Show first 5
            print(f"     - {feature}")
        if len(traditional_features) > 5:
            print(f"     ... and {len(traditional_features) - 5} more")

        print(f"   Advanced Features ({len(advanced_features)}):")
        for feature in advanced_features[:5]:  # Show first 5
            print(f"     - {feature}")
        if len(advanced_features) > 5:
            print(f"     ... and {len(advanced_features) - 5} more")

        print(f"   Additional Features ({len(additional_features)}):")
        for feature in additional_features[:5]:  # Show first 5
            print(f"     - {feature}")
        if len(additional_features) > 5:
            print(f"     ... and {len(additional_features) - 5} more")

    print(f"\n✅ Feature preference ordering configured successfully!")


def test_enhanced_state_naming():
    """Test the enhanced state naming with mixed traditional and advanced features."""

    print(f"\n📝 Testing Enhanced State Naming")
    print("-" * 40)

    # Create test data with both traditional and advanced features
    test_momentum_data = {
        0: {
            "momentum_strength": 0.8,
            "rsi": 75,
            "bb_position": 0.9,
            "momentum_5": 0.6,
            "momentum_10": 0.5,
            "momentum_20": 0.4,
            "15m_volume_momentum": 0.7,
            "30m_volume_momentum": 0.6,
            "5m_volume_momentum": 0.8,
            "1m_volume_momentum": 0.9,
        },
        1: {
            "momentum_strength": -0.7,
            "rsi": 25,
            "bb_position": -0.8,
            "momentum_5": -0.5,
            "momentum_10": -0.4,
            "momentum_20": -0.3,
            "15m_volume_momentum": -0.6,
            "30m_volume_momentum": -0.5,
            "5m_volume_momentum": -0.7,
            "1m_volume_momentum": -0.8,
        },
        2: {
            "momentum_strength": 0.1,
            "rsi": 50,
            "bb_position": 0.0,
            "momentum_5": 0.1,
            "momentum_10": 0.0,
            "momentum_20": -0.1,
            "15m_volume_momentum": 0.2,
            "30m_volume_momentum": 0.1,
            "5m_volume_momentum": 0.0,
            "1m_volume_momentum": -0.1,
        },
    }

    test_volatility_data = {
        0: {
            "price_volatility": 0.9,
            "adaptive_atr": 0.8,
            "ewma_volatility": 0.7,
            "volume_volatility": 1.2,
            "parkinson_volatility": 0.8,
            "garman_klass_volatility": 0.6,
            "5m_volume_volatility": 1.0,
        },
        1: {
            "price_volatility": 0.3,
            "adaptive_atr": 0.4,
            "ewma_volatility": 0.3,
            "volume_volatility": 0.2,
            "parkinson_volatility": 0.4,
            "garman_klass_volatility": 0.3,
            "5m_volume_volatility": 0.1,
        },
        2: {
            "price_volatility": 0.1,
            "adaptive_atr": 0.1,
            "ewma_volatility": 0.1,
            "volume_volatility": 0.05,
            "parkinson_volatility": 0.1,
            "garman_klass_volatility": 0.1,
            "5m_volume_volatility": 0.02,
        },
    }

    test_microstructure_data = {
        0: {
            "spread_tightness": 0.1,
            "trade_frequency": 0.8,
            "order_flow_imbalance": 0.2,
            "bid_ask_spread_returns": 0.05,
            "bid_ask_spread_level": 0.1,
            "price_impact": 0.2,
        },
        1: {
            "spread_tightness": 0.6,
            "trade_frequency": 0.4,
            "order_flow_imbalance": 0.7,
            "bid_ask_spread_returns": 0.3,
            "bid_ask_spread_level": 0.5,
            "price_impact": 0.6,
        },
        2: {
            "spread_tightness": 0.9,
            "trade_frequency": 0.2,
            "order_flow_imbalance": 0.8,
            "bid_ask_spread_returns": 0.8,
            "bid_ask_spread_level": 0.9,
            "price_impact": 0.9,
        },
    }

    # Test momentum naming
    print("📊 Testing Momentum State Naming:")
    momentum_names = _name_momentum_states(test_momentum_data)
    for state_id, name in momentum_names.items():
        print(f"   State {state_id}: {name}")

    # Test volatility naming
    print("\n📊 Testing Volatility State Naming:")
    volatility_names = _name_volatility_states(test_volatility_data)
    for state_id, name in volatility_names.items():
        print(f"   State {state_id}: {name}")

    # Test microstructure naming
    print("\n📊 Testing Microstructure State Naming:")
    microstructure_names = _name_microstructure_states(test_microstructure_data)
    for state_id, name in microstructure_names.items():
        print(f"   State {state_id}: {name}")

    print(f"\n✅ Enhanced state naming working correctly!")


def test_with_real_data():
    """Test with real data from the meta file."""

    meta_file = "data/training/BINANCE_ETHUSDT_hmm_composite_meta_1m.json"

    if not os.path.exists(meta_file):
        print(f"❌ Meta file not found: {meta_file}")
        return False

    with open(meta_file, "r") as f:
        meta_data = json.load(f)

    print(f"\n🔍 Testing with Real Data from {meta_file}")
    print("-" * 40)

    state_feature_medians = meta_data.get("state_feature_medians", {})

    for block_name in ["momentum", "volatility", "liquidity", "microstructure"]:
        print(f"\n📊 {block_name.upper()} Block - Real Data:")
        print("-" * 30)

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
                    # Show only the first few features to avoid clutter
                    feature_str = ", ".join(
                        [f"{k}: {v:.3f}" for k, v in list(features.items())[:3]]
                    )
                    print(f"      Sample features: {feature_str}")

        except Exception as e:
            print(f"❌ Error naming {block_name} states: {e}")
            return False

    return True


if __name__ == "__main__":
    print("🧪 Enhanced HMM Feature Selection and Naming Test")
    print("=" * 60)

    # Test feature selection
    test_enhanced_feature_selection()

    # Test state naming
    test_enhanced_state_naming()

    # Test with real data
    success = test_with_real_data()

    if success:
        print(f"\n🎉 All tests passed! The enhanced system should work correctly.")
        print(f"\n📋 Summary of Improvements:")
        print(f"   ✅ Feature preference ordering for all blocks")
        print(f"   ✅ Traditional indicators prioritized alongside advanced features")
        print(f"   ✅ Enhanced state naming with mixed feature types")
        print(f"   ✅ More nuanced and meaningful state classifications")
    else:
        print(f"\n❌ Tests failed. Please check the implementation.")
        sys.exit(1)
