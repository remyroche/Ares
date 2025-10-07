"""
from ...utils.logger import system_logger
Launch Enhanced Monitoring Dashboard

A launcher script for the enhanced monitoring dashboard GUI.
"""
import asyncio
import sys
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from typing import Dict, Any
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from .monitoring.gui.enhanced_dashboard import create_enhanced_monitoring_dashboard
from ...utils.logger import system_logger
import logging

def check_dependencies() -> bool:
    """Check if required dependencies are available."""
    missing_deps = []
    try:
        pass
    except ImportError:
        missing_deps.append('pandas')
    try:
        pass
    except ImportError:
        missing_deps.append('numpy')
    try:
        pass
    except ImportError:
        missing_deps.append('matplotlib')
    try:
        pass
    except ImportError:
        missing_deps.append('seaborn')
    if missing_deps:
        error_msg = f"Missing required dependencies: {', '.join(missing_deps)}\n\n"
        error_msg += 'Please install them using:\n'
        error_msg += f"pip install {' '.join(missing_deps)}"
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror('Missing Dependencies', error_msg)
        return False
    return True

def create_default_config() -> Dict[str, Any]:
    """Create default configuration for the dashboard."""
    return {'monitoring_gui': {'window_width': 1600, 'window_height': 1000, 'refresh_interval_ms': 5000}, 'enhanced_monitoring': {'enable_monitoring': True, 'enable_explanations': True, 'enable_ensemble_monitoring': True, 'enable_csv_export': True, 'export_interval_days': 30, 'max_memory_decisions': 10000, 'export_directory': 'monitoring_exports'}, 'daily_summary_tracker': {'enable_real_time_updates': True, 'summary_retention_days': 365, 'export_directory': 'daily_summaries'}, 'csv_export': {'export_directory': 'monitoring_exports', 'include_raw_data': True, 'include_summary_stats': True, 'decimal_precision': 6}}

def main() -> int:
    """Main launcher function."""
    try:
        if not check_dependencies():
            return 1
        config = create_default_config()
        system_logger.info('Launching Enhanced Monitoring Dashboard...')
        dashboard = create_enhanced_monitoring_dashboard(config)
        dashboard.run()
        return 0
    except Exception as e:
        system_logger.error(f'Error launching dashboard: {e}')
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror('Launch Error', f'Failed to launch dashboard:\n{e}')
        return 1
if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)