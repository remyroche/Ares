"""Minimal Pandera stub used when the real library is unavailable."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional

import pandas as pd
import re

# Import core utilities for enhanced error handling and DataFrame operations
try:
    from ...utils.tprint import tprint, tprint_debug, tprint_error, tprint_warning
    from ...utils.common_operations import (
        validate_dataframe, safe_divide, timed_operation, format_bytes
    )
except ImportError:
    # Fallback imports if utils are not available
    def tprint(*args, **kwargs): pass
    def tprint_debug(*args, **kwargs): pass
    def tprint_error(*args, **kwargs): pass
    def tprint_warning(*args, **kwargs): pass
    def validate_dataframe(df): return isinstance(df, pd.DataFrame) and not df.empty
    def safe_divide(a, b, default=0.0): return a / b if b != 0 else default
    def timed_operation(func): return func
    def format_bytes(bytes_value): return f"{bytes_value}B"


class SchemaError(Exception):
    """Replacement for pandera.errors.SchemaError."""

    def __init__(self, message: str, failure_cases: Optional[pd.DataFrame] = None):
        super().__init__(message)
        self.failure_cases = failure_cases if failure_cases is not None else pd.DataFrame()


class Column:
    """Lightweight stand-in for :class:`pandera.Column` accepting common kwargs."""

    def __init__(
        self,
        dtype: str,
        *,
        checks: Optional[Any] = None,
        nullable: bool = False,
        unique: bool = False,
        allow_duplicates: bool = True,
        coerce: bool = False,
        required: bool = True,
        regex: bool = False,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> None:
        self.dtype = dtype
        self.checks = checks
        self.nullable = nullable
        self.unique = unique
        self.allow_duplicates = allow_duplicates
        self.coerce = coerce
        self.required = required
        self.regex = regex
        self.name = name
        self.metadata = metadata or {}
        self.extra = extra


class Index:
    """Lightweight stand-in for :class:`pandera.Index`."""

    def __init__(
        self,
        dtype: str,
        *,
        coerce: bool = False,
        nullable: bool = False,
        name: Optional[str] = None,
        **extra: Any,
    ) -> None:
        self.dtype = dtype
        self.coerce = coerce
        self.nullable = nullable
        self.name = name
        self.extra = extra


class DataFrameSchema:
    def __init__(
        self,
        columns: Optional[Dict[str, Column]] = None,
        dtype: Optional[str] = None,
        index: Optional[Index] = None,
        strict: bool = False,
        coerce: bool = False,
        **extra: Any,
    ) -> None:
        self.columns = columns or {}
        self.dtype = dtype
        self.index = index
        self.strict = strict
        self.coerce = coerce
        self.extra = extra

    @timed_operation
    def validate(self, df: pd.DataFrame, lazy: bool = False) -> pd.DataFrame:
        """Enhanced validation with core utilities integration."""

        tprint_debug("Starting schema validation with stub")
        failures = []

        if not validate_dataframe(df):
            tprint_error("Input DataFrame is invalid")
            raise SchemaError("Input must be a valid pandas DataFrame")

        validated = df.copy()
        original_memory = df.memory_usage(deep=True).sum()

        if self.dtype in {"float", "float64"}:
            for column in validated.columns:
                validated[column] = pd.to_numeric(validated[column], errors="coerce")
            if validated.isnull().any().any():
                failures.append({"check": "dtype", "failure_case": "non-numeric value"})

        for name, column in self.columns.items():
            if getattr(column, "regex", False):
                pattern = re.compile(name)
                matching_columns = [col for col in validated.columns if pattern.fullmatch(str(col))]
            else:
                matching_columns = [name]

            if not matching_columns:
                if getattr(column, "required", True):
                    failures.append({"check": "required_column", "failure_case": name})
                continue

            for col_name in matching_columns:
                if col_name not in validated.columns:
                    if getattr(column, "required", True):
                        failures.append({"check": "required_column", "failure_case": col_name})
                    continue

                if column.coerce:
                    if column.dtype in {"float", "float64"}:
                        validated[col_name] = pd.to_numeric(validated[col_name], errors="coerce")
                    elif column.dtype in {"int", "int64"}:
                        validated[col_name] = pd.to_numeric(validated[col_name], errors="coerce")

                if not column.nullable and validated[col_name].isnull().any():
                    failures.append({"check": "nullable", "failure_case": col_name})

        if self.index:
            if self.index.coerce and self.index.dtype.startswith("datetime"):
                validated.index = pd.to_datetime(validated.index, errors="coerce")
            if validated.index.dtype.kind != "M":
                failures.append({"check": "index", "failure_case": "expected datetime index"})
            if not self.index.nullable and validated.index.isnull().any():
                failures.append({"check": "index_nullable", "failure_case": "null index entries"})

        if failures:
            tprint_error(f"Schema validation failed with {len(failures)} errors")
            raise SchemaError("Schema validation failed", pd.DataFrame(failures))

        # Log memory usage improvement
        final_memory = validated.memory_usage(deep=True).sum()
        memory_reduction = safe_divide(original_memory - final_memory, original_memory, 0.0) * 100
        if memory_reduction > 0:
            tprint_debug(f"Memory usage reduced by {memory_reduction:.1f}% during validation")

        tprint_debug("Schema validation completed successfully")
        return validated


class _PanderaStub:
    DataFrameSchema = DataFrameSchema
    Column = Column
    Index = Index
    Float = "float"
    Float64 = "float64"
    Int64 = "int64"
    DateTime = "datetime64[ns]"


pandera_stub = _PanderaStub()
errors = SimpleNamespace(SchemaError=SchemaError)
