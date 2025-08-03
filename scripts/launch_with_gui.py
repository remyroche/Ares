#!/usr/bin/env python3
"""
Simple wrapper to launch the bot with GUI automatically.
Usage: python scripts/launch_with_gui.py <original_bot_command>
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.gui_launcher import GUILauncher


def main():
    """Launch the bot with GUI using the original command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/launch_with_gui.py <original_bot_command>")
        print("Examples:")
        print(
            "  python scripts/launch_with_gui.py main_launcher.py trade ETHUSDT BINANCE paper",
        )
        print("  python scripts/launch_with_gui.py main_launcher.py manager")
        print(
            "  python scripts/launch_with_gui.py scripts/training_cli.py full-test-run ETHUSDT",
        )
        sys.exit(1)

    # Get the bot command (everything after the script name)
    bot_command = sys.argv[1:]

    # Create GUI launcher and run
    launcher = GUILauncher()
    launcher.run_with_bot(bot_command, auto_gui=True)


if __name__ == "__main__":
    main()
