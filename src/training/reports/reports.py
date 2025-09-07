"""
Training Reports Module

This module provides centralized reporting utilities for training steps,
including file saving, report generation, and data export capabilities.
"""

import json
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Union
from src.utils.logger import system_logger

logger = system_logger.getChild('TrainingReports')


class CentralizedReportManager:
    """Centralized manager for all training reports and data exports."""

    def __init__(self, base_path: Optional[str] = None):
        """Initialize the report manager."""
        if base_path is None:
            # Use the project's src/training/reports directory
            # Get the absolute path to the src/training/reports directory
            current_file_path = Path(__file__).resolve()
            self.base_path = current_file_path.parent  # src/training/reports
        else:
            self.base_path = Path(base_path)

        # Ensure the directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Report manager initialized with base path: {self.base_path}")

    def save_report(self,
                   data: Any,
                   filename: str,
                   file_format: str = 'json',
                   subdirectory: Optional[str] = None) -> Optional[str]:
        """Save data to a file in the specified format."""
        try:
            # Create subdirectory if specified
            save_path = self.base_path
            if subdirectory:
                save_path = save_path / subdirectory
                save_path.mkdir(parents=True, exist_ok=True)

            # Generate full file path
            if not filename.endswith(f'.{file_format}'):
                filename = f"{filename}.{file_format}"

            full_path = save_path / filename

            # Save based on format
            if file_format.lower() == 'json':
                with open(full_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str)
            elif file_format.lower() == 'csv':
                if isinstance(data, pd.DataFrame):
                    data.to_csv(full_path, index=False)
                elif isinstance(data, dict):
                    # Convert dict to DataFrame if possible
                    try:
                        df = pd.DataFrame.from_dict(data, orient='index').T
                        df.to_csv(full_path, index=False)
                    except:
                        # Fallback: save as JSON
                        with open(full_path.with_suffix('.json'), 'w') as f:
                            json.dump(data, f, indent=2, default=str)
                        full_path = full_path.with_suffix('.json')
                else:
                    # Convert to string and save as text
                    with open(full_path.with_suffix('.txt'), 'w') as f:
                        f.write(str(data))
                    full_path = full_path.with_suffix('.txt')
            elif file_format.lower() in ['md', 'txt']:
                with open(full_path, 'w', encoding='utf-8') as f:
                    if isinstance(data, dict):
                        # Convert dict to readable format
                        f.write("# Training Report\n\n")
                        for key, value in data.items():
                            f.write(f"## {key}\n\n")
                            if isinstance(value, (dict, list)):
                                f.write(f"```json\n{json.dumps(value, indent=2, default=str)}\n```\n\n")
                            else:
                                f.write(f"{value}\n\n")
                    else:
                        f.write(str(data))
            else:
                # Default to JSON for unknown formats
                with open(full_path.with_suffix('.json'), 'w') as f:
                    json.dump(data if isinstance(data, dict) else {'data': str(data)},
                             f, indent=2, default=str)
                full_path = full_path.with_suffix('.json')

            logger.info(f"💾 Report saved: {full_path}")
            return str(full_path)

        except Exception as e:
            logger.error(f"❌ Failed to save report: {e}")
            return None


# Global report manager instance
_report_manager = CentralizedReportManager()


def save_training_report(data: Any,
                        step_name: str,
                        report_type: str,
                        symbol: Optional[str] = None,
                        timeframe: Optional[str] = None,
                        file_format: str = 'json',
                        subdirectory: Optional[str] = None) -> Optional[str]:
    """
    Save training report data to file with standardized naming.

    Args:
        data: The data to save (dict, DataFrame, or other serializable object)
        step_name: Name of the training step (e.g., 'step02_5_sr_optimization')
        report_type: Type of report (e.g., 'comprehensive_analysis', 'basic_sr_analysis')
        symbol: Trading symbol (optional)
        timeframe: Timeframe (optional)
        file_format: File format ('json', 'csv', 'md', 'txt')
        subdirectory: Optional subdirectory under reports/

    Returns:
        Path to the saved file, or None if saving failed
    """
    try:
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build filename components
        filename_parts = [step_name, report_type, timestamp]

        if symbol:
            filename_parts.insert(1, symbol)
        if timeframe:
            filename_parts.insert(2, timeframe) if symbol else filename_parts.insert(1, timeframe)

        # Create filename
        filename = "_".join(filename_parts)

        # Use step name as subdirectory if none specified
        if subdirectory is None:
            subdirectory = step_name

        # Save the report
        return _report_manager.save_report(
            data=data,
            filename=filename,
            file_format=file_format,
            subdirectory=subdirectory
        )

    except Exception as e:
        logger.error(f"❌ Failed to save training report: {e}")
        return None


def get_report_path(step_name: str,
                   report_type: str,
                   symbol: Optional[str] = None,
                   timeframe: Optional[str] = None) -> Path:
    """Get the expected path for a report without creating it."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_parts = [step_name, report_type, timestamp]

    if symbol:
        filename_parts.insert(1, symbol)
    if timeframe:
        filename_parts.insert(2, timeframe) if symbol else filename_parts.insert(1, timeframe)

    filename = "_".join(filename_parts)
    subdirectory = step_name

    return _report_manager.base_path / subdirectory / f"{filename}.json"


def list_reports(step_name: Optional[str] = None) -> Dict[str, list]:
    """List all available reports, optionally filtered by step."""
    try:
        reports = {}

        if step_name:
            step_dir = _report_manager.base_path / step_name
            if step_dir.exists():
                reports[step_name] = [f.name for f in step_dir.glob("*") if f.is_file()]
        else:
            for step_dir in _report_manager.base_path.glob("*"):
                if step_dir.is_dir():
                    reports[step_dir.name] = [f.name for f in step_dir.glob("*") if f.is_file()]

        return reports

    except Exception as e:
        logger.error(f"❌ Failed to list reports: {e}")
        return {}

