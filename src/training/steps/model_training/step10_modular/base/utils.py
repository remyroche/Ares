from ..standardized_parquet_handler import standardized_parquet_handler
"""Step 10 Base Utilities.

This module contains common utility functions used throughout the
unified regime intelligence system.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.utils.logger import system_logger
import logging
import numpy as np

logger = system_logger.getChild('Step10Utils')

def ensure_directory(path: str) -> str:
    """Ensure directory exists, create if necessary.

    Args:
        path: Directory path to ensure

    Returns:
        Absolute path to the directory
    """
    try:
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        abs_path = str(path_obj.absolute())
        logger.debug(f"✅ Directory ensured: {abs_path}")
        return abs_path
    except Exception as e:
        logger.error(f"❌ Failed to create directory {path}: {e}")
        raise

def safe_json_dump(data: Any, file_path: str, indent: int = 2) -> bool:
    """Safely dump data to JSON file with error handling.

    Args:
        data: Data to dump
        file_path: Output file path
        indent: JSON indentation level

    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_directory(str(Path(file_path).parent))

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, default=str)

        logger.debug(f"✅ JSON saved: {file_path}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to save JSON to {file_path}: {e}")
        return False

def standardize_price_action_probabilities(probabilities: Dict[str, float]) -> Dict[str, float]:
    """Standardize price action probabilities to ensure they sum to 1.

    Args:
        probabilities: Raw probability dictionary

    Returns:
        Standardized probability dictionary
    """
    try:
        if not probabilities:
            return {"neutral": 1.0}

        # Filter out invalid probabilities
        valid_probs = {k: v for k, v in probabilities.items()
                      if isinstance(v, (int, float)) and v >= 0}

        if not valid_probs:
            return {"neutral": 1.0}

        # Normalize to sum to 1
        total = sum(valid_probs.values())
        if total == 0:
            return {"neutral": 1.0}

        standardized = {k: v / total for k, v in valid_probs.items()}

        # Ensure we have reasonable probabilities
        if len(standardized) == 1:
            key = list(standardized.keys())[0]
            standardized[key] = 1.0

        return standardized

    except Exception as e:
        logger.error(f"❌ Failed to standardize probabilities: {e}")
        return {"neutral": 1.0}

def validate_data_quality(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate data quality for Step 10 processing.

    Args:
        data: Data dictionary to validate

    Returns:
        Validation results dictionary
    """
    results = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }

    try:
        # Check for required keys
        required_keys = ["hmm_states", "features"]
        for key in required_keys:
            if key not in data:
                results["errors"].append(f"Missing required key: {key}")
                results["is_valid"] = False

        # Validate data types and shapes
        if "features" in data:
            features = data["features"]
            if hasattr(features, 'shape'):
                results["stats"]["feature_shape"] = features.shape
                if len(features.shape) < 2:
                    results["errors"].append("Features must be at least 2-dimensional")
                    results["is_valid"] = False
            else:
                results["warnings"].append("Features object doesn't have shape attribute")

        # Check HMM states
        if "hmm_states" in data:
            hmm_states = data["hmm_states"]
            if isinstance(hmm_states, dict):
                results["stats"]["timeframes"] = list(hmm_states.keys())
                results["stats"]["hmm_states_count"] = len(hmm_states)
            else:
                results["errors"].append("HMM states must be a dictionary")
                results["is_valid"] = False

    except Exception as e:
        results["errors"].append(f"Validation failed: {str(e)}")
        results["is_valid"] = False

    return results
