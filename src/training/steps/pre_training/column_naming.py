"""Shared helpers for enforcing standardized DataFrame column namespaces.

This module centralizes the naming conventions that the pre-training steps must
follow when producing pandas ``DataFrame`` objects. Downstream components rely
on the prefix-based namespaces defined here to distinguish between feature,
label, target, and metadata columns. Helper utilities are provided to simplify
renaming, validation, and discovery of namespaced columns.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable, List, Mapping, MutableMapping, Optional, Set, Tuple, Union

import pandas as pd


class ColumnNamespace(str, Enum):
    """Logical namespaces supported by the pre-training pipeline."""

    FEATURE = "feature"
    LABEL = "label"
    TARGET = "target"
    META = "meta"


_NAMESPACE_TO_PREFIX = {
    ColumnNamespace.FEATURE: "feat__",
    ColumnNamespace.LABEL: "label__",
    ColumnNamespace.TARGET: "target__",
    ColumnNamespace.META: "meta__",
}

ALLOWED_PREFIXES: Tuple[str, ...] = tuple(_NAMESPACE_TO_PREFIX.values())

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


def strip_namespace(column: str) -> Tuple[str, Optional[ColumnNamespace]]:
    """Remove a known namespace prefix from ``column`` if present."""

    for namespace, prefix in _NAMESPACE_TO_PREFIX.items():
        if column.startswith(prefix):
            return column[len(prefix) :], namespace
    return column, None


def ensure_namespace(column: str, namespace: Union[ColumnNamespace, str]) -> str:
    """Ensure ``column`` is namespaced with the prefix associated to ``namespace``."""

    base_name, current_namespace = strip_namespace(column)
    if current_namespace == ColumnNamespace(namespace):
        return column
    return f"{get_namespace_prefix(namespace)}{base_name}"


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

    prefix = get_namespace_prefix(namespace)
    return [col for col in columns if col.startswith(prefix)]


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
