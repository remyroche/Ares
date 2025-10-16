"""
Shared configuration load/save utilities with light-weight dataclass support.

These helpers consolidate repeated YAML/JSON I/O patterns across ML, validation,
and code quality configuration modules while keeping backward compatibility.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass, fields
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Type, TypeVar, Union

import yaml

logger = logging.getLogger(__name__)

T = TypeVar("T")

def to_serializable_dict(obj: Any) -> Dict[str, Any]:
    """Convert a config-like object to a serializable dictionary.

    Order of preference:
    1) obj.to_dict()
    2) dataclasses.asdict(obj)
    3) obj if already a dict
    4) obj.__dict__ filtered of private attributes
    """
    if obj is None:
        return {}

    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return dict(getattr(obj, "to_dict")())
        except Exception as exc:
            logger.debug("to_dict() failed, falling back to dataclass/attrs conversion: %s", exc)

    if is_dataclass(obj):
        return asdict(obj)

    if isinstance(obj, Mapping):
        return dict(obj)

    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}

    raise TypeError(f"Unsupported object type for serialization: {type(obj)!r}")

def instantiate_from_dict(target_cls: Type[T], data: Mapping[str, Any]) -> T:
    """Instantiate target class from a dictionary with graceful fallbacks.

    Order of preference:
    1) target_cls.from_dict(data)
    2) dataclass field-filtered kwargs
    3) direct **data construction
    """
    if hasattr(target_cls, "from_dict") and callable(getattr(target_cls, "from_dict")):
        return getattr(target_cls, "from_dict")(dict(data))  # type: ignore[no-any-return]

    try:
        if is_dataclass(target_cls):
            field_names = {f.name for f in fields(target_cls)}
            filtered = {k: v for k, v in data.items() if k in field_names}
            return target_cls(**filtered)  # type: ignore[misc]
    except Exception:
        # Fall through to generic construction
        pass

    return target_cls(**dict(data))  # type: ignore[misc]

def _read_file(filepath: Union[str, Path]) -> Dict[str, Any]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    if suffix in (".yml", ".yaml"):
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    raise ValueError(f"Unsupported configuration file format: {suffix}")

def _write_file(data: Mapping[str, Any], filepath: Union[str, Path]) -> None:
    path = Path(filepath)
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(dict(data), f, indent=2)
        return

    if suffix in (".yml", ".yaml"):
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(dict(data), f, default_flow_style=False, indent=2)
        return

    raise ValueError(f"Unsupported configuration file format: {suffix}")

def save_to_file(obj: Any, filepath: Union[str, Path]) -> None:
    """Serialize a config-like object and write it to YAML or JSON file."""
    data = to_serializable_dict(obj)
    _write_file(data, filepath)

def load_from_file(filepath: Union[str, Path], target_cls: Optional[Type[T]] = None) -> Union[T, Dict[str, Any]]:
    """Load config data from YAML/JSON and optionally instantiate a target class.

    If target_cls is None, returns a plain dict.
    """
    data = _read_file(filepath)
    if target_cls is None:
        return data
    return instantiate_from_dict(target_cls, data)

def merge_dicts(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dicts and return a new merged dict."""
    result: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
