"""Shared helpers for enforcing standardized DataFrame column namespaces.

This module centralizes the naming conventions that the pre-training steps must
follow when producing pandas ``DataFrame`` objects. Downstream components rely
on the prefix-based namespaces defined here to distinguish between feature,
label, target, and metadata columns. Helper utilities are provided to simplify
renaming, validation, and discovery of namespaced columns.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Set, Tuple, Union

import pandas as pd

class ColumnNamespace(str, Enum):
    """Logical namespaces supported by the pre-training pipeline."""

    FEATURE = "feature"
    LABEL = "label"
    TARGET = "target"
    META = "meta"

# Canonical prefixes that the pipeline should emit.
_NAMESPACE_TO_PREFIX = {
    ColumnNamespace.FEATURE: "X_",
    ColumnNamespace.LABEL: "y_",
    ColumnNamespace.TARGET: "y_",
    ColumnNamespace.META: "meta_",
}

# Historical/legacy prefixes that may appear at step boundaries.  These values
# are accepted for ingestion but will be translated into the canonical forms.
_LEGACY_NAMESPACE_PREFIXES: Mapping[ColumnNamespace, Tuple[str, ...]] = {
    ColumnNamespace.FEATURE: ("feat__", "feature__", "feature_", "feat_", "x__", "x_"),
    ColumnNamespace.LABEL: ("label__", "labels__", "label_", "labels_", "y__"),
    ColumnNamespace.TARGET: (
        "target__",
        "targets__",
        "target_",
        "targets_",
        "label__",
        "labels__",
        "label_",
        "labels_",
        "y__",
    ),
    ColumnNamespace.META: ("meta__",),
}

# Column aliases that we want to interpret as belonging to a namespace even
# without a prefix.  These are typically found when interacting with external
# utilities (e.g. scikit-learn) that expect ``X``/``y`` naming conventions.
_NAMESPACE_ALIASES: Mapping[ColumnNamespace, Tuple[str, ...]] = {
    ColumnNamespace.FEATURE: ("x", "features", "feature", "feat"),
    ColumnNamespace.LABEL: ("y", "label", "labels", "target", "targets", "return"),
    ColumnNamespace.TARGET: ("y", "label", "labels", "target", "targets", "return"),
    ColumnNamespace.META: ("meta",),
}

_ALIAS_DEFAULT_BASE: Mapping[ColumnNamespace, Mapping[str, str]] = {
    ColumnNamespace.FEATURE: {"x": "feature", "features": "feature", "feat": "feature"},
    ColumnNamespace.LABEL: {"y": "target", "label": "label", "labels": "label", "target": "target", "targets": "target", "return": "return"},
    ColumnNamespace.TARGET: {"y": "target", "label": "label", "labels": "label", "target": "target", "targets": "target", "return": "return"},
    ColumnNamespace.META: {"meta": "meta"},
}

def _all_prefixes() -> Tuple[str, ...]:
    prefixes = set(_NAMESPACE_TO_PREFIX.values())
    for legacy_values in _LEGACY_NAMESPACE_PREFIXES.values():
        prefixes.update(legacy_values)
    return tuple(sorted(prefixes))

ALLOWED_PREFIXES: Tuple[str, ...] = _all_prefixes()

# Columns that may legitimately appear without a namespace prefix.
DEFAULT_ALLOWED_UNPREFIXED: Set[str] = {
    "timestamp",
    "symbol",
    "exchange",
    "regime_state",
    "sample_weight",
    "weight",
    "event_start",
    "event_end",
    "window_start",
    "window_end",
}

def get_namespace_prefix(namespace: Union[ColumnNamespace, str]) -> str:
    """Return the canonical prefix for a namespace."""

    ns = ColumnNamespace(namespace) if not isinstance(namespace, ColumnNamespace) else namespace
    return _NAMESPACE_TO_PREFIX[ns]

def _match_prefix(column: str, namespace: ColumnNamespace) -> Optional[str]:
    """Return ``column`` without a recognized prefix for ``namespace`` if present."""

    prefix = _NAMESPACE_TO_PREFIX[namespace]
    if column.startswith(prefix):
        return column[len(prefix) :]

    for legacy_prefix in _LEGACY_NAMESPACE_PREFIXES.get(namespace, ()):  # pragma: no branch - small tuple
        if column.startswith(legacy_prefix):
            return column[len(legacy_prefix) :]
    return None

def _match_alias(column: str, namespace: ColumnNamespace) -> Optional[str]:
    """Return a base name when ``column`` matches a namespace alias."""

    lower_column = column.lower()
    for alias in _NAMESPACE_ALIASES.get(namespace, ()):  # pragma: no branch - small tuple
        alias_lower = alias.lower()
        if lower_column == alias_lower:
            default_base = _ALIAS_DEFAULT_BASE.get(namespace, {}).get(alias_lower, alias_lower)
            return default_base
        underscored = f"{alias_lower}_"
        if lower_column.startswith(underscored):
            remainder = column[len(underscored) :]
            return remainder or _ALIAS_DEFAULT_BASE.get(namespace, {}).get(alias_lower, alias_lower)
        double_underscored = f"{alias_lower}__"
        if lower_column.startswith(double_underscored):
            remainder = column[len(double_underscored) :]
            return remainder or _ALIAS_DEFAULT_BASE.get(namespace, {}).get(alias_lower, alias_lower)
    return None

def strip_namespace(column: str) -> Tuple[str, Optional[ColumnNamespace]]:
    """Remove a known namespace prefix from ``column`` if present."""

    for namespace in ColumnNamespace:
        matched = _match_prefix(column, namespace)
        if matched is not None:
            return matched, namespace
    return column, None

def ensure_namespace(column: str, namespace: Union[ColumnNamespace, str]) -> str:
    """Ensure ``column`` is namespaced with the prefix associated to ``namespace``."""

    base_name, current_namespace = strip_namespace(column)
    target_namespace = ColumnNamespace(namespace)

    if current_namespace == target_namespace:
        # Column already conforms to the namespace; ensure canonical prefix usage.
        return f"{get_namespace_prefix(target_namespace)}{base_name}"

    if current_namespace is not None and current_namespace != target_namespace:
        return f"{get_namespace_prefix(target_namespace)}{base_name}"

    alias_base = _match_alias(column, target_namespace)
    if alias_base is not None:
        base_name = alias_base

    return f"{get_namespace_prefix(target_namespace)}{base_name}"

def ensure_dataframe_namespace(
    df: pd.DataFrame,
    namespace: Union[ColumnNamespace, str],
    columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``columns`` renamed into ``namespace``."""

    if df is None or df.empty:
        return df

    ns = ColumnNamespace(namespace)
    target_columns = list(columns) if columns is not None else list(df.columns)
    rename_map = {col: ensure_namespace(col, ns) for col in target_columns}
    if not rename_map:
        return df
    return df.rename(columns=rename_map)

def filter_namespace_columns(
    columns: Iterable[str],
    namespace: Union[ColumnNamespace, str],
) -> List[str]:
    """Filter ``columns`` by namespace."""

    target_namespace = ColumnNamespace(namespace)
    namespaced: List[str] = []
    for column in columns:
        base, detected_namespace = strip_namespace(column)
        if detected_namespace == target_namespace:
            namespaced.append(f"{get_namespace_prefix(target_namespace)}{base}")
            continue
        if _match_alias(column, target_namespace) is not None:
            namespaced.append(ensure_namespace(column, target_namespace))
    return namespaced

def standardize_namespace_frame(
    data: pd.DataFrame,
    namespace: Union[ColumnNamespace, str],
    *,
    allowed_unprefixed: Optional[Iterable[str]] = None,
    preserve_meta: bool = True,
) -> pd.DataFrame:
    """Return a copy of ``data`` with columns coerced into a namespace."""

    if data is None or len(data) == 0:
        return data

    allowed = set(DEFAULT_ALLOWED_UNPREFIXED)
    if allowed_unprefixed is not None:
        allowed.update(allowed_unprefixed)

    rename_map: Dict[str, str] = {}
    target_namespace = ColumnNamespace(namespace)

    for column in data.columns:
        if column in allowed:
            continue

        if preserve_meta and filter_namespace_columns([column], ColumnNamespace.META):
            rename_map[column] = ensure_namespace(column, ColumnNamespace.META)
            continue

        rename_map[column] = ensure_namespace(column, target_namespace)

    if rename_map:
        data = data.rename(columns=rename_map)
    return data

def find_nonconforming_columns(
    columns: Iterable[str],
    allowed_unprefixed: Optional[Iterable[str]] = None,
) -> List[str]:
    """Return columns that do not use an allowed namespace or whitelist."""

    allowed_set = set(DEFAULT_ALLOWED_UNPREFIXED)
    if allowed_unprefixed is not None:
        allowed_set.update(allowed_unprefixed)

    invalid: List[str] = []
    for column in columns:
        if column in allowed_set:
            continue
        if any(column.startswith(prefix) for prefix in ALLOWED_PREFIXES):
            continue
        invalid.append(column)
    return invalid

def validate_column_names(
    columns: Iterable[str],
    *,
    allowed_unprefixed: Optional[Iterable[str]] = None,
    frame_name: str = "DataFrame",
) -> None:
    """Raise ``ValueError`` if ``columns`` contains non-namespaced values."""

    invalid = find_nonconforming_columns(columns, allowed_unprefixed)
    if invalid:
        raise ValueError(
            f"{frame_name} contains columns without an approved namespace prefix: {sorted(invalid)}. "
            f"Allowed prefixes: {ALLOWED_PREFIXES}"
        )

def validate_dataframe_names(
    df: pd.DataFrame,
    *,
    allowed_unprefixed: Optional[Iterable[str]] = None,
    frame_name: str = "DataFrame",
) -> None:
    """Validate namespace prefixes for DataFrame columns."""

    if df is None or df.empty:
        return
    validate_column_names(df.columns, allowed_unprefixed=allowed_unprefixed, frame_name=frame_name)

def map_by_namespace(columns: Iterable[str]) -> Mapping[ColumnNamespace, List[str]]:
    """Group columns by namespace."""

    grouped: MutableMapping[ColumnNamespace, List[str]] = {ns: [] for ns in ColumnNamespace}
    for column in columns:
        base, ns = strip_namespace(column)
        if ns is not None:
            grouped[ns].append(column)
    return grouped

def ensure_namespace_for_mapping(mapping: Mapping[str, str], namespace: Union[ColumnNamespace, str]) -> Mapping[str, str]:
    """Return a copy of mapping with namespaced values."""

    ns = ColumnNamespace(namespace)
    return {ensure_namespace(key, ns): ensure_namespace(value, ns) for key, value in mapping.items()}
