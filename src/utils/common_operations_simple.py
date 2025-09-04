from __future__ import annotations
from typing import Dict, List, Optional, Union, Any, Tuple
"""
Simplified Common Operations Utility Module

This module provides commonly used operations without external dependencies.
"""

import datetime
import json
import logging
import time
from pathlib import Path


def get_current_datetime() -> datetime.datetime:
    """Get current datetime."""
    return datetime.datetime.now()


def get_today() -> datetime.date:
    """Get today's date."""
    return datetime.date.today()


def format_datetime(dt: datetime.datetime, fmt: str='%Y-%m-%d %H:%M:%S') -> str:
    """Format datetime to string."""
    return dt.strftime(fmt)


def safe_file_exists(file_path: str | Path) -> bool:
    """Safely check if file exists."""
    try:
        return Path(file_path).exists()
    except Exception:
        return False


def safe_json_dump(data: Any, file_path: str | Path, indent: int = 2) -> bool:
    """Safely dump data to JSON file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        return True
    except Exception as e:
        logging.error(f"Failed to save JSON to {file_path}: {e}")
        return False


def safe_json_load(file_path: str | Path) -> Optional[Dict[str, Any]]:
    """Safely load data from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON from {file_path}: {e}")
        return None


def ensure_directory(dir_path: str | Path) -> bool:
    """Ensure directory exists, create if it doesn't."""
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"Failed to create directory {dir_path}: {e}")
        return False


def get_file_size(file_path: str | Path) -> int:
    """Get file size in bytes."""
    try:
        return Path(file_path).stat().st_size
    except Exception:
        return 0


def is_file_empty(file_path: str | Path) -> bool:
    """Check if file is empty."""
    return get_file_size(file_path) == 0


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> Tuple[bool, List[str]]:
    """Validate configuration dictionary."""
    missing_keys = []
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)
    
    return len(missing_keys) == 0, missing_keys


def create_pipeline_id(symbol: str, exchange: str, timeframe: str) -> str:
    """Create a unique pipeline ID."""
    timestamp = int(time.time())
    return f"pipeline_{symbol}_{exchange}_{timeframe}_{timestamp}"


def log_pipeline_start(logger: logging.Logger, pipeline_name: str, config: Dict[str, Any]) -> None:
    """Log pipeline start with configuration."""
    logger.info(f"🚀 Starting {pipeline_name}")
    logger.info(f"📊 Configuration: {config}")


def log_pipeline_success(logger: logging.Logger, pipeline_name: str, execution_time: float) -> None:
    """Log pipeline success."""
    logger.info(f"✅ {pipeline_name} completed successfully in {execution_time:.2f} seconds")


def log_pipeline_failure(logger: logging.Logger, pipeline_name: str, error: str, execution_time: float) -> None:
    """Log pipeline failure."""
    logger.error(f"❌ {pipeline_name} failed: {error}")
    logger.error(f"⏱️ Execution time: {execution_time:.2f} seconds")


def format_execution_time(seconds: float) -> str:
    """Format execution time in a human-readable format."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"