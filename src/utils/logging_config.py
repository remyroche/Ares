"""Logging configuration utilities."""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

def get_logger(name: str = None) -> Any:
    """Get a logger instance."""
    return logging.getLogger(name or __name__)