#!/usr/bin/env python3
"""
ARES Bot Monitor - Notifies AI Assistant when bot stops or encounters issues
"""

from datetime import datetime
from pathlib import Path
from src.utils.logger import system_logger
import json
import sys
import time
import psutil
from typing import Any, Dict, List

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.warning_symbols import error, warning


class BotMonitor:
    def __init__(self) -> None:
        self.logger = system_logger.getChild("BotMonitor")
        self.project_root = project_root
        self.monitor_interval = 30  # Check every 30 seconds
        self.status_file = project_root / "state/bot_status.json"
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self.last_status = self._load_status()

    def _load_status(self) -> Dict[str, Any]:
        """Load the last known status of the bot"""
        try:
            if self.status_file.exists():
                with open(self.status_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:  # noqa: BLE001
            print(error(f"Error loading status file: {e}"))
        return {"running": False, "last_check": None, "issues": []}

    def _save_status(self, status: Dict[str, Any]) -> None:
        """Save the current status of the bot"""
        try:
            with open(self.status_file, "w", encoding="utf-8") as f:
                json.dump(status, f, indent=2, default=str)
        except Exception as e:  # noqa: BLE001
            print(error(f"Error saving status file: {e}"))

    def _check_python_processes(self) -> List[Dict[str, Any]]:
        """Check if any ARES-related Python processes are running"""
        ares_processes: List[Dict[str, Any]] = []

        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = " ".join(proc.info.get("cmdline") or [])
                if (
                    "ares_launcher.py" in cmdline
                    or ("python" in (proc.info.get("name") or ""))
                    and "ares" in cmdline.lower()
                ):
                    ares_processes.append(
                        {
                            "pid": proc.info.get("pid"),
                            "cmdline": cmdline,
                            "status": proc.status(),
                        },
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return ares_processes

    def _check_log_files(self) -> List[Dict[str, Any]]:
        """Check recent log files for errors"""
        log_dir = project_root / "logs"
        if not log_dir.exists():
            print(warning("Logs directory not found"))
            return []

        issues: List[Dict[str, Any]] = []
        current_time = time.time()

        for log_file in log_dir.glob("*.log"):
            try:
                # Check if log file was modified in the last 5 minutes
                if current_time - log_file.stat().st_mtime < 300:
                    with open(log_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    # Check last 50 lines for errors
                    for line in lines[-50:]:
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
                            issues.append(
                                {
                                    "file": log_file.name,
                                    "line": line.strip(),
                                    "timestamp": datetime.fromtimestamp(
                                        log_file.stat().st_mtime,
                                    ).isoformat(),
                                },
                            )
            except Exception as e:  # noqa: BLE001
                print(error(f"Error reading log file {log_file}: {e}"))

        return issues

    def _notify_ai_assistant(self, message: str, issues: List[Dict[str, Any]] | None = None) -> None:
        """Notify the AI assistant about issues"""
        notification = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "issues": issues or [],
            "bot_status": self._load_status(),
            "action_required": True,
        }

        # Save notification to a file that the AI assistant can read
        notification_file = project_root / "state/ai_notification.json"
        try:
            notification_file.parent.mkdir(parents=True, exist_ok=True)
            with open(notification_file, "w", encoding="utf-8") as f:
                json.dump(notification, f, indent=2, default=str)

            # Also print to console for immediate visibility
            print(f"\n🚨 ARES BOT ALERT: {message}")
            if issues:
                print("📋 Issues detected:")
                for issue in issues:
                    print(
                        f"   - {issue.get('file', 'Unknown')}: {issue.get('line', 'Unknown error')}",
                    )
            print(f"📝 Full details saved to: {notification_file}")
            print(
                "🤖 AI Assistant: Please check the notification file and fix the issues.\n",
            )
        except Exception as e:  # noqa: BLE001
            print(error(f"Error saving notification: {e}"))

    def monitor(self) -> None:
        """Main monitoring loop"""
        self.logger.info("🤖 Starting ARES Bot Monitor...")

        while True:
            try:
                current_time = datetime.now()

                # Check for running processes
                ares_processes = self._check_python_processes()
                is_running = len(ares_processes) > 0

                # Check for recent issues in logs
                recent_issues = self._check_log_files()

                # Update status
                current_status: Dict[str, Any] = {
                    "running": is_running,
                    "last_check": current_time.isoformat(),
                    "processes": ares_processes,
                    "issues": recent_issues,
                }

                # Check if status changed
                status_changed = self.last_status.get("running") != is_running or len(
                    recent_issues
                ) > len(self.last_status.get("issues", []))

                if status_changed:
                    if not is_running and self.last_status.get("running"):
                        message = "🚨 ARES Bot has stopped running!"
                        self._notify_ai_assistant(message, recent_issues)
                    elif recent_issues and len(recent_issues) > len(
                        self.last_status.get("issues", [])
                    ):
                        message = "⚠️ New issues detected in ARES Bot logs!"
                        self._notify_ai_assistant(message, recent_issues)
                    elif is_running and not self.last_status.get("running"):
                        message = "✅ ARES Bot is now running again"
                        self._notify_ai_assistant(message)

                # Save current status
                self._save_status(current_status)
                self.last_status = current_status

                # Log status
                if is_running:
                    self.logger.info(
                        f"✅ Bot is running ({len(ares_processes)} processes)",
                    )
                else:
                    print(warning("❌ Bot is not running"))

                if recent_issues:
                    self.logger.warning(
                        f"⚠️ {len(recent_issues)} recent issues detected",
                    )

                # Wait before next check
                time.sleep(self.monitor_interval)

            except KeyboardInterrupt:
                self.logger.info("🛑 Bot monitor stopped by user")
                break
            except Exception as e:  # noqa: BLE001
                print(error(f"Error in monitoring loop: {e}"))
                time.sleep(self.monitor_interval)


def main() -> None:
    """Main entry point"""
    monitor = BotMonitor()
    monitor.monitor()


if __name__ == "__main__":
    main()
