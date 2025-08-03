#!/usr/bin/env python3
# scripts/show_timeframe_config.py

"""
Show Timeframe Configuration

This script displays the current timeframe configuration in a user-friendly format.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import CONFIG


def show_timeframe_config():
    """Display the current timeframe configuration."""
    print("🎯 Ares Multi-Timeframe Configuration")
    print("=" * 60)

    # Get configuration
    timeframes = CONFIG.get("TIMEFRAMES", {})
    timeframe_sets = CONFIG.get("TIMEFRAME_SETS", {})
    default_set = CONFIG.get("DEFAULT_TIMEFRAME_SET", "swing")
    mtf_config = CONFIG.get("MULTI_TIMEFRAME_TRAINING", {})

    print("\n📊 Multi-Timeframe Training Settings:")
    print("-" * 40)
    print(f"Parallel Training: {mtf_config.get('enable_parallel_training', True)}")
    print(f"Ensemble Creation: {mtf_config.get('enable_ensemble', True)}")
    print(f"Cross Validation: {mtf_config.get('enable_cross_validation', True)}")
    print(f"Ensemble Method: {mtf_config.get('ensemble_method', 'weighted_average')}")
    print(f"Max Parallel Workers: {mtf_config.get('max_parallel_workers', 3)}")

    print(f"\n🎯 Individual Timeframes ({len(timeframes)} total):")
    print("-" * 40)

    # Group timeframes by trading style
    trading_styles = {}
    for tf, info in timeframes.items():
        style = info.get("trading_style", "unknown")
        if style not in trading_styles:
            trading_styles[style] = []
        trading_styles[style].append((tf, info))

    for style, tf_list in trading_styles.items():
        print(f"\n📈 {style.upper().replace('_', ' ')}:")
        for tf, info in tf_list:
            print(
                f"  {tf:>4} | {info.get('purpose', 'Unknown'):<50} | Weight: {info.get('ensemble_weight', 0):.2f}",
            )

    print(f"\n📋 Predefined Timeframe Sets ({len(timeframe_sets)} total):")
    print("-" * 40)

    for set_name, set_info in timeframe_sets.items():
        is_default = " ⭐" if set_name == default_set else ""
        print(f"\n{set_name}{is_default}:")
        print(f"  Timeframes: {', '.join(set_info.get('timeframes', []))}")
        print(f"  Description: {set_info.get('description', 'No description')}")
        print(f"  Use Case: {set_info.get('use_case', 'No use case specified')}")

    print("\n🔧 Current Configuration:")
    print("-" * 40)
    print(f"Default timeframe set: {default_set}")
    print(
        f"Default timeframes: {', '.join(timeframe_sets.get(default_set, {}).get('timeframes', []))}",
    )

    # Show ensemble weights for default set
    default_timeframes = timeframe_sets.get(default_set, {}).get("timeframes", [])
    if default_timeframes:
        print("\n⚖️  Ensemble Weights for Default Set:")
        total_weight = 0
        for tf in default_timeframes:
            weight = timeframes.get(tf, {}).get("ensemble_weight", 0)
            total_weight += weight
            print(f"  {tf}: {weight:.2f}")
        print(f"  Total: {total_weight:.2f}")

    print("\n💡 Usage Examples:")
    print("-" * 40)
    print("# Use default swing trading timeframes")
    print(
        "python ares_launcher.py multi-timeframe --symbol ETHUSDT --exchange BINANCE",
    )
    print()
    print("# Use specific timeframes")
    print(
        "python scripts/run_multi_timeframe_training.py --symbol ETHUSDT --timeframes 1h,4h,1d",
    )
    print()
    print("# Use predefined scalping set")
    print(
        "python scripts/run_multi_timeframe_training.py --symbol ETHUSDT --timeframes 1m,5m,15m",
    )
    print()
    print("# List all available timeframes")
    print("python scripts/run_multi_timeframe_training.py --list-timeframes")


def show_timeframe_details(timeframe: str):
    """Show detailed information about a specific timeframe."""
    timeframes = CONFIG.get("TIMEFRAMES", {})

    if timeframe not in timeframes:
        print(f"❌ Timeframe '{timeframe}' not found in configuration")
        return

    info = timeframes[timeframe]

    print(f"📊 Detailed Information for {timeframe}")
    print("=" * 50)
    print(f"Purpose: {info.get('purpose', 'Unknown')}")
    print(f"Trading Style: {info.get('trading_style', 'Unknown')}")
    print(f"Lookback Days: {info.get('lookback_days', 'Unknown')}")
    print(f"Feature Set: {info.get('feature_set', 'Unknown')}")
    print(f"Optimization Trials: {info.get('optimization_trials', 'Unknown')}")
    print(f"Ensemble Weight: {info.get('ensemble_weight', 'Unknown')}")
    print(f"Description: {info.get('description', 'No description')}")


def main():
    """Main function."""
    if len(sys.argv) > 1:
        timeframe = sys.argv[1]
        show_timeframe_details(timeframe)
    else:
        show_timeframe_config()


if __name__ == "__main__":
    main()
