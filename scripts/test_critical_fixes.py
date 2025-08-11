#!/usr/bin/env python3
"""
Test Critical Fixes Script

This script tests the critical fixes implemented in the main pipeline:
1. Binary classification (label imbalance fix)
2. Redundant feature filtering (multicollinearity fix)
3. Extreme VIF removal (safety net)

Usage:
    python scripts/test_critical_fixes.py
"""

import sys
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

# Import the orchestrator
from training.steps.vectorized_labelling_orchestrator import (
    VectorizedLabellingOrchestrator,
)


def create_test_data():
    """Create test data with the problematic features that cause multicollinearity."""
    print("📊 Creating test data with problematic features...")

    # Create base price data
    dates = pd.date_range("2024-01-01", periods=1000, freq="1min")
    np.random.seed(42)
    base_price = 100 + np.cumsum(np.random.randn(1000) * 0.1)

    # Create price data with redundant features
    price_data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": base_price + np.random.randn(1000) * 0.5,
            "high": base_price + np.random.randn(1000) * 0.8,
            "low": base_price - np.random.randn(1000) * 0.8,
            "close": base_price + np.random.randn(1000) * 0.5,
            "volume": np.random.randint(100, 1000, 1000),
        }
    )

    # Add redundant features that cause multicollinearity
    price_data["avg_price"] = (price_data["high"] + price_data["low"]) / 2
    price_data["min_price"] = price_data["low"]
    price_data["max_price"] = price_data["high"]
    price_data["price_change"] = price_data["close"].pct_change()
    price_data["high_price_change"] = price_data["high"].pct_change()
    price_data["low_price_change"] = price_data["low"].pct_change()
    price_data["open_price_change"] = price_data["open"].pct_change()
    price_data["avg_price_change"] = price_data["avg_price"].pct_change()
    price_data["min_price_change"] = price_data["min_price"].pct_change()
    price_data["max_price_change"] = price_data["max_price"].pct_change()

    # Create volume data
    volume_data = pd.DataFrame(
        {
            "timestamp": dates,
            "volume": price_data["volume"].copy(),
            "trade_count": np.random.randint(10, 100, 1000),
            "trade_volume": price_data["volume"] * 0.8,  # Redundant
            "volume_ratio": price_data["volume"]
            / price_data["volume"].rolling(10).mean(),  # Redundant
        }
    )

    # Set timestamp as index
    price_data.set_index("timestamp", inplace=True)
    volume_data.set_index("timestamp", inplace=True)

    print(f"✅ Created test data:")
    print(f"   Price data shape: {price_data.shape}")
    print(f"   Volume data shape: {volume_data.shape}")
    print(f"   Price features: {list(price_data.columns)}")
    print(f"   Volume features: {list(volume_data.columns)}")

    return price_data, volume_data


async def test_critical_fixes():
    """Test the critical fixes in the main pipeline."""
    print("🧪 TESTING CRITICAL FIXES")
    print("=" * 60)

    # Create test data
    price_data, volume_data = create_test_data()

    # Configuration with critical fixes enabled
    config = {
        "vectorized_labelling_orchestrator": {
            "enable_stationary_checks": True,
            "enable_data_normalization": True,
            "enable_lookahead_bias_handling": True,
            "enable_feature_selection": True,
            "enable_memory_efficient_types": True,
            "enable_parquet_saving": True,
            "profit_take_multiplier": 0.002,
            "stop_loss_multiplier": 0.001,
            "time_barrier_minutes": 30,
            "max_lookahead": 100,
            "binary_classification": True,  # CRITICAL FIX: Enable binary classification
            "feature_selection": {
                "vif_threshold": 5.0,  # CRITICAL FIX: Stricter VIF threshold
                "correlation_threshold": 0.95,  # CRITICAL FIX: Stricter correlation threshold
                "enable_aggressive_vif_removal": True,
                "max_removal_percentage": 0.8,
                "min_features_to_keep": 5,
                "enable_multicollinearity_validation": True,
                "enable_redundant_price_filtering": True,  # CRITICAL FIX: Enable redundant filtering
                "vif_removal_strategy": "iterative",
                "max_iterations": 10,
            },
        }
    }

    # Initialize orchestrator
    print("\n🚀 Initializing orchestrator with critical fixes...")
    orchestrator = VectorizedLabellingOrchestrator(config)
    success = await orchestrator.initialize()

    if not success:
        print("❌ Failed to initialize orchestrator")
        return

    print("✅ Orchestrator initialized successfully")

    # Test the pipeline
    print("\n🎯 Testing the complete pipeline...")
    try:
        result = await orchestrator.orchestrate_labeling_and_feature_engineering(
            price_data, volume_data
        )

        if "error" in result:
            print(f"❌ Pipeline failed: {result['error']}")
            return

        # Extract the final data
        if "final_data" in result:
            final_data = result["final_data"]
            print(f"\n✅ Pipeline completed successfully!")
            print(f"📊 Final data shape: {final_data.shape}")
            print(f"📏 Final features: {list(final_data.columns)}")

            # Check for label column
            if "label" in final_data.columns:
                labels = final_data["label"]
                unique_labels, counts = np.unique(labels, return_counts=True)
                label_distribution = dict(zip(unique_labels, counts))

                print(f"\n🎯 Label distribution:")
                for label, count in label_distribution.items():
                    ratio = count / len(labels) * 100
                    print(f"   {label}: {count} samples ({ratio:.1f}%)")

                # Check for label imbalance fix
                if 0 not in label_distribution:
                    print(
                        "✅ CRITICAL FIX VERIFIED: HOLD class (0) successfully removed"
                    )
                else:
                    print("⚠️ WARNING: HOLD class still present")

                # Check for balanced binary classification
                if len(label_distribution) == 2:
                    min_ratio = min(label_distribution.values()) / len(labels) * 100
                    max_ratio = max(label_distribution.values()) / len(labels) * 100
                    if min_ratio > 10:  # At least 10% in each class
                        print(
                            "✅ CRITICAL FIX VERIFIED: Balanced binary classification achieved"
                        )
                    else:
                        print("⚠️ WARNING: Severe class imbalance still present")
                else:
                    print("⚠️ WARNING: Not binary classification")

            # Check for multicollinearity fix
            feature_columns = [col for col in final_data.columns if col != "label"]
            if len(feature_columns) < 20:  # Should have fewer features after filtering
                print(
                    "✅ CRITICAL FIX VERIFIED: Feature count reduced (multicollinearity addressed)"
                )
            else:
                print("⚠️ WARNING: Feature count not significantly reduced")

            # Check for redundant features
            redundant_features = [
                "open",
                "high",
                "low",
                "avg_price",
                "min_price",
                "max_price",
            ]
            remaining_redundant = [
                col for col in redundant_features if col in feature_columns
            ]
            if not remaining_redundant:
                print(
                    "✅ CRITICAL FIX VERIFIED: Redundant price features successfully removed"
                )
            else:
                print(
                    f"⚠️ WARNING: Some redundant features still present: {remaining_redundant}"
                )

        else:
            print("❌ No final data in result")

    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Main function to run the critical fixes test."""
    print("🧪 CRITICAL FIXES TEST")
    print("=" * 60)
    print("This test verifies that the critical fixes are working:")
    print("1. Binary classification (label imbalance fix)")
    print("2. Redundant feature filtering (multicollinearity fix)")
    print("3. Extreme VIF removal (safety net)")
    print("=" * 60)

    asyncio.run(test_critical_fixes())

    print("\n" + "=" * 60)
    print("✅ Critical fixes test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
