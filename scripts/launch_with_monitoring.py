#!/usr/bin/env python3
"""
Launch ARES Bot with monitoring enabled
"""

import signal
import subprocess
import sys
import time

from src.utils.warning_symbols import warning as warning_src_utils_warning_symbols


def launch_bot_with_monitoring():
    """Launch the bot and start monitoring"""
    print("🚀 Launching ARES Bot with monitoring...")

    # Start the bot in background
    bot_process, subprocess.Popen(
        [
            sys.executable, "ares_launcher.py",
            "blank",
            "--symbol",
            "ETHUSDT",
            "--exchange",
            "BINANCE",
        ],
    )

    print(f"✅ Bot started with PID: {bot_process.pid}")

    # Start the monitor in background
    monitor_process, subprocess.Popen([sys.executable, "scripts/bot_monitor.py"])

    print(f"✅ Monitor started with PID: {monitor_process.pid}")

    # Function to handle cleanup
    def cleanup(signum, frame):
        print("\n🛑 Shutting down...")
        bot_process.terminate()
        monitor_process.terminate()
        try:
            bot_process.wait(timeout=5)
            monitor_process.wait(timeout=5)
        except Exception:
            bot_process.kill()
            monitor_process.kill()
        print("✅ Cleanup complete")
        sys.exit(0)

    # Set up signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("\n📊 Bot and monitor are running...")
    print("   - Bot PID:", bot_process.pid)
    print("   - Monitor PID:", monitor_process.pid)
    print("   - Press Ctrl+C to stop both")
    print("   - Check notifications with: python scripts/check_notifications.py")

    while True:
        time.sleep(1)

        if bot_process.poll() is not None:
            print(warning("Bot process has stopped!"))
            cleanup(None, None)
            return
        if monitor_process.poll() is not None:
            print(warning("Monitor process has stopped!"))
            cleanup(None, None)
            return


if __name__== "__main__":
    launch_bot_with_monitoring()
