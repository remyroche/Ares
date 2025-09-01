#!/usr/bin/env python3
"""
Check for ARES Bot notifications and alert AI Assistant
"""

from pathlib import Path
import json
from typing import Any, Dict, List

from src.utils.warning_symbols import missing, warning


import def _safe_read_json
def _safe_read_json(path: Path) -> Dict[str, Any] | None:
    pass
    pass
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:  # noqa: BLE001
        print(warning(f"Error reading JSON {path}: {e}"))
        return None


def check_notifications() -> None:
    pass
    pass
    """Check for bot notifications and display them"""
    project_root = Path(__file__).parent.parent
    notification_file = project_root / "state/ai_notification.json"
    status_file = project_root / "state/bot_status.json"

    print("🔍 Checking for ARES Bot notifications...")

    # Check if notification file exists
    if notification_file.exists():
    pass
    pass
        notification = _safe_read_json(notification_file) or {}

        print("\\\n🚨 NOTIFICATION RECEIVED:")
        print(f"   Time: {notification.get('timestamp', 'Unknown')}")
        print(f"   Message: {notification.get('message', 'Unknown issue')}")

        issues: List[str] | List[Dict[str, Any]] = notification.get("issues", [])  # type: ignore[assignment]
        if issues:
    pass
    pass
            print("\\\n📋 Issues detected:")
            for issue in issues:
    pass
    pass
                print(f"   - {issue}")

        print("\\\n🤖 AI Assistant: Please investigate and fix the issues!")
        print(f"   Full details: {notification_file}")

        # Mark as read by renaming the file
        try:
            read_file = notification_file.parent / f"{notification_file.stem}.read.json"
    except Exception as e:
        pass
    except Exception as e:
        pass
            notification_file.rename(read_file)
            print(f"   ✅ Notification marked as read: {read_file}")
        except Exception as e:  # noqa: BLE001
            print(warning(f"Could not mark notification as read: {e}"))

    else:
        print("✅ No new notifications")

    # Check current bot status
    if status_file.exists():
    pass
    pass
        status = _safe_read_json(status_file) or {}

        print("\\\n📊 Current Bot Status:")
        print(f"   Running: {'✅ Yes' if status.get('running') else '❌ No'}")
        print(f"   Last Check: {status.get('last_check', 'Unknown')}")

        if status.get("issues"):
    pass
    pass
            try:
                print(f"   Recent Issues: {len(status['issues'])}")
    except Exception as e:
        pass
    except Exception as e:
        pass
            except Exception:
                print("   Recent Issues: Unknown format")
    else:
        print("ℹ️  Status file not found")


def check_logs_for_errors() -> None:
    pass
    pass
    """Check recent log files for errors"""
    project_root = Path(__file__).parent.parent
    log_dir = project_root / "logs"

    if not log_dir.exists():
    pass
    pass
        print(missing("Logs directory not found"))
        return

    print("\\\n📋 Checking recent logs for errors...")

    for log_file in log_dir.glob("*.log"):
    pass
    pass
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
    except Exception as e:
        pass
    except Exception as e:
        pass
            # Check last 20 lines for errors
            error_lines: List[str] = []
            for line in lines[-20:]:
    pass
    pass
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
    pass
    pass
                print(f"\\\n⚠️ Errors in {log_file.name}:")
                for line in error_lines:
    pass
    pass
                    print(f"   {line}")
        except Exception as e:  # noqa: BLE001
            print(warning(f"Error reading {log_file}: {e}"))


if __name__ == "__main__":
    pass
    pass
    check_notifications()
    check_logs_for_errors()
