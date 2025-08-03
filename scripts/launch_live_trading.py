#!/usr/bin/env python3
"""
Launch live trading with GUI automatically.
Usage: python scripts/launch_live_trading.py <symbol> <exchange>
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.gui_launcher import GUILauncher


def main():
    """Launch live trading with GUI."""
    parser = argparse.ArgumentParser(description="Launch live trading with GUI")
    parser.add_argument("symbol", help="Trading symbol (e.g., BTCUSDT)")
    parser.add_argument("exchange", help="Exchange name (e.g., BINANCE)")
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip live trading confirmation",
    )

    args = parser.parse_args()

    # Safety check for live trading
    if not args.confirm:
        print("⚠️  WARNING: You are about to start LIVE TRADING!")
        print(f"   Symbol: {args.symbol}")
        print(f"   Exchange: {args.exchange}")
        print("   This will use real money!")
        print()
        response = input("Are you sure you want to continue? (type 'yes' to confirm): ")
        if response.lower() != "yes":
            print("❌ Live trading cancelled.")
            sys.exit(0)

    # Construct the bot command
    bot_command = ["main_launcher.py", "trade", args.symbol, args.exchange, "live"]

    print(f"🚀 Launching LIVE trading for {args.symbol} on {args.exchange} with GUI...")
    print("⚠️  Using real money - trade carefully!")

    # Create GUI launcher and run
    launcher = GUILauncher()
    launcher.run_with_bot(bot_command, auto_gui=not args.no_gui)


if __name__ == "__main__":
    main()
