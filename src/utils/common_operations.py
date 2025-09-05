"""Common operations utilities."""
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import json
import os
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
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def safe_json_dump(data, path, indent=2):
    """Safely dump JSON data."""
    try:
        ensure_directory(os.path.dirname(path))
        with open(path, 'w') as f:
            json.dump(data, f, indent=indent)
        return True
    except Exception as e:
        print(f"Error saving JSON to {path}: {e}")
        return False

def safe_read_parquet(path):
    """Safely read parquet file."""
    try:
        if os.path.exists(path):
            return pd.read_parquet(path)
        else:
            print(f"Parquet file not found: {path}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error reading parquet file {path}: {e}")
        return pd.DataFrame()

def ensure_directory(path):
    """Ensure directory exists."""
    try:
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)
            return True
    except Exception as e:
        print(f"Error creating directory {path}: {e}")
    return False

def validate_dataframe_schema(df, schema):
    """Validate dataframe schema."""
    try:
        if df is None or df.empty:
            return False
        # Basic schema validation
        for col in schema.get('required_columns', []):
            if col not in df.columns:
                return False
        return True
    except Exception:
        return False

def validate_data_quality(df, thresholds):
    """Validate data quality."""
    try:
        if df is None or df.empty:
            return False
        
        # Check for minimum rows
        min_rows = thresholds.get('min_rows', 0)
        if len(df) < min_rows:
            return False
            
        # Check for null ratio
        max_null_ratio = thresholds.get('max_null_ratio', 0.5)
        null_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if null_ratio > max_null_ratio:
            return False
            
        return True
    except Exception:
        return False

def standardize_price_action_probabilities(probabilities):
    """Standardize price action probabilities to ensure they sum to 1 and are non-negative."""
    try:
        if not isinstance(probabilities, dict):
            return probabilities
            
        # Ensure all values are non-negative
        standardized = {k: max(0.0, float(v)) for k, v in probabilities.items()}
        
        # Normalize to sum to 1
        total = sum(standardized.values())
        if total > 0:
            standardized = {k: v / total for k, v in standardized.items()}
        else:
            # If all probabilities are 0, set equal probabilities
            n = len(standardized)
            standardized = {k: 1.0 / n for k in standardized.keys()}
            
        return standardized
    except Exception as e:
        print(f"Error standardizing probabilities: {e}")
        # Return default probabilities
        return {
            "triple_barrier_probability": 0.25,
            "direction_probability": 0.25,
            "magnitude_probability": 0.25,
            "barrier_avoidance_probability": 0.25
        }

def safe_to_parquet(df, path):
    """Safely save DataFrame to parquet file."""
    try:
        ensure_directory(os.path.dirname(path))
        df.to_parquet(path)
        return True
    except Exception as e:
        print(f"Error saving parquet file {path}: {e}")
        return False

def safe_copy(src, dst):
    """Safely copy file."""
    try:
        import shutil
        ensure_directory(os.path.dirname(dst))
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        print(f"Error copying file from {src} to {dst}: {e}")
        return False
