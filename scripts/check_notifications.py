#!/usr/bin/env python3
"""
Check for ARES Bot notifications and alert AI Assistant
"""

import json
from src.utils.warning_symbols import (
    error,
    warning,
    critical,
    problem,
    failed,
    invalid,
    missing,
    timeout,
    connection_error,
    validation_error,
    initialization_error,
    execution_error,
)
from pathlib import Path


def check_notifications():
    """Check for bot notifications and display them"""
    project_root = Path(__file__).parent.parent
    notification_file = project_root / "state/ai_notification.json"
    status_file = project_root / "state/bot_status.json"

    print("🔍 Checking for ARES Bot notifications...")

    # Check if notification file exists
    if notification_file.exists():
        try:
            with open(notification_file) as f:
                notification = json.load(f)

            print("\n🚨 NOTIFICATION RECEIVED:")
            print(f"   Time: {notification.get('timestamp', 'Unknown')}")
            print(f"   Message: {notification.get('message', 'Unknown issue')}")

            if notification.get("issues"):
                print("\n📋 Issues detected:")
                for issue in notification["issues"]:
                    print(
                        f"   - {issue.get('file', 'Unknown')}: {issue.get('line', 'Unknown error')}",
                    )

            print("\n🤖 AI Assistant: Please investigate and fix the issues!")
            print(f"   Full details: {notification_file}")

            # Mark as read by renaming the file
            read_file = notification_file.parent / f"{notification_file.stem}.json.read"
            notification_file.rename(read_file)
            print(f"   ✅ Notification marked as read: {read_file}")

        except Exception as e:
            print(warning("Error reading notification file: {e}")))
    else:
        print("✅ No new notifications")

    # Check current bot status
    if status_file.exists():
        try:
            with open(status_file) as f:
                status = json.load(f)

            print("\n📊 Current Bot Status:")
            print(f"   Running: {'✅ Yes' if status.get('running') else '❌ No'}")
            print(f"   Last Check: {status.get('last_check', 'Unknown')}")

            if status.get("issues"):
                print(f"   Recent Issues: {len(status['issues'])}")

        except Exception as e:
            print(warning("Error reading status file: {e}")))


def check_logs_for_errors():
    """Check recent log files for errors"""
    project_root = Path(__file__).parent.parent
    log_dir = project_root / "logs"

    if not log_dir.exists():
        print(missing("Logs directory not found")))
        return

    print("\n📋 Checking recent logs for errors...")

    for log_file in log_dir.glob("*.log"):
        try:
            with open(log_file) as f:
                lines = f.readlines()
                # Check last 20 lines for errors
                error_lines = []
                for line in lines[-20:]:
                    if any(
                        error_keyword in line.lower()
                        for error_keyword in [
                            "error",
                            "exception",
                            "traceback",
                            "failed",
                            "❌",
                            "💥",
                        ]
                    ):
                        error_lines.append(line.strip())

                if error_lines:
                    print(f"\n⚠️ Errors in {log_file.name}:")
                    for line in error_lines:
                        print(f"   {line}")

        except Exception as e:
            print(warning("Error reading {log_file}: {e}")))


if __name__ == "__main__":
    check_notifications()
    check_logs_for_errors()
