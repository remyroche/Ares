"""Launcher for the enhanced monitoring dashboard GUI."""

import asyncio
import importlib
import sys
from pathlib import Path
from typing import Any, Dict

import tkinter as tk
from tkinter import messagebox

# Ensure the GUI package is importable when running directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .monitoring.gui.enhanced_dashboard import create_enhanced_monitoring_dashboard
from ...utils.logger import system_logger


REQUIRED_MODULES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
}


def _show_error_dialog(title: str, message: str) -> None:
    """Display an error dialog without showing the root window."""

    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(title, message)
    root.destroy()


def check_dependencies() -> bool:
    """Check if required dependencies are available."""

    missing_deps = []
    for display_name, module_name in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing_deps.append(display_name)

    if not missing_deps:
        return True

    error_msg = (
        f"Missing required dependencies: {', '.join(missing_deps)}\n\n"
        "Please install them using:\n"
        f"pip install {' '.join(missing_deps)}"
    )
    _show_error_dialog("Missing Dependencies", error_msg)
    return False


def create_default_config() -> Dict[str, Any]:
    """Create default configuration for the dashboard."""

    return {
        "monitoring_gui": {
            "window_width": 1600,
            "window_height": 1000,
            "refresh_interval_ms": 5000,
        },
        "enhanced_monitoring": {
            "enable_monitoring": True,
            "enable_explanations": True,
            "enable_ensemble_monitoring": True,
            "enable_csv_export": True,
            "export_interval_days": 30,
            "max_memory_decisions": 10000,
            "export_directory": "monitoring_exports",
        },
        "daily_summary_tracker": {
            "enable_real_time_updates": True,
            "summary_retention_days": 365,
            "export_directory": "daily_summaries",
        },
        "csv_export": {
            "export_directory": "monitoring_exports",
            "include_raw_data": True,
            "include_summary_stats": True,
            "decimal_precision": 6,
        },
    }


def main() -> int:
    """Main launcher function."""

    try:
        if not check_dependencies():
            return 1
        config = create_default_config()
        system_logger.info("Launching Enhanced Monitoring Dashboard...")
        dashboard = create_enhanced_monitoring_dashboard(config)
        dashboard.run()
        return 0
    except Exception as exc:  # pragma: no cover - defensive logging
        system_logger.exception("Error launching dashboard: %s", exc)
        _show_error_dialog("Launch Error", f"Failed to launch dashboard:\n{exc}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
