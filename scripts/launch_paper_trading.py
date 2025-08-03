#!/usr/bin/env python3
"""
Launch paper trading with GUI automatically.
Usage: python scripts/launch_paper_trading.py <symbol> <exchange>
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.gui_launcher import GUILauncher


def main():
    """Launch paper trading with GUI."""
    parser = argparse.ArgumentParser(description="Launch paper trading with GUI")
    parser.add_argument("symbol", help="Trading symbol (e.g., BTCUSDT)")
    parser.add_argument("exchange", help="Exchange name (e.g., BINANCE)")
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI")

    args = parser.parse_args()

    # Construct the bot command
    bot_command = ["main_launcher.py", "trade", args.symbol, args.exchange, "paper"]

    print(
        f"🚀 Launching paper trading for {args.symbol} on {args.exchange} with GUI...",
    )

    # Create GUI launcher and run
    launcher = GUILauncher()
    launcher.run_with_bot(bot_command, auto_gui=not args.no_gui)


if __name__ == "__main__":
    main()
