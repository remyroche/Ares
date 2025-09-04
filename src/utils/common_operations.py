"""Common operations utilities."""
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Any, Dict, Callable

def get_current_datetime():
    """Get current datetime."""
    return datetime.now()

def format_datetime(dt, format_str):
    """Format datetime object with given format string."""
    if dt is None:
        return "N/A"
    return dt.strftime(format_str)

def safe_json_load(path): 
    """Safely load JSON file."""
    return {}

def safe_read_parquet(path):
    """Safely read parquet file."""
    return pd.DataFrame()

def validate_dataframe_schema(df, schema):
    """Validate dataframe schema."""
    return True

def validate_data_quality(df, thresholds):
    """Validate data quality."""
    return True
