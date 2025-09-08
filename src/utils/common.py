"""
Common utilities with passthrough functions for common operations.
"""

from typing import Any, Optional, Union, Dict, List

def safe_get(dictionary: Dict, key: str, default: Any = None) -> Any:
    """Safely get value from dictionary."""
    return dictionary.get(key, default)

def safe_set(dictionary: Dict, key: str, value: Any) -> None:
    """Safely set value in dictionary."""
    dictionary[key] = value

def safe_list_get(lst: List, index: int, default: Any = None) -> Any:
    """Safely get item from list by index."""
    try:
        return lst[index]
    except (IndexError, TypeError):
        return default

def safe_list_append(lst: List, item: Any) -> None:
    """Safely append item to list."""
    if isinstance(lst, list):
        lst.append(item)

def validate_type(value: Any, expected_type: type) -> bool:
    """Validate if value is of expected type."""
    return isinstance(value, expected_type)

def safe_convert(value: Any, target_type: type, default: Any = None) -> Any:
    """Safely convert value to target type."""
    try:
        return target_type(value)
    except (ValueError, TypeError):
        return default

def merge_dicts(*dicts: Dict) -> Dict:
    """Merge multiple dictionaries."""
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result

def flatten_list(nested_list: List) -> List:
    """Flatten nested list."""
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result

# Export all functions
__all__ = [
    'safe_get',
    'safe_set',
    'safe_list_get',
    'safe_list_append',
    'validate_type',
    'safe_convert',
    'merge_dicts',
    'flatten_list'
]
