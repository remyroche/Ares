"""Minimal Pandera stub used when the real library is unavailable."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Optional

import pandas as pd


class SchemaError(Exception):
    """Replacement for pandera.errors.SchemaError."""

    def __init__(self, message: str, failure_cases: Optional[pd.DataFrame] = None):
        super().__init__(message)
        self.failure_cases = failure_cases if failure_cases is not None else pd.DataFrame()


@dataclass
class Column:
    dtype: str
    coerce: bool = False
    nullable: bool = False


@dataclass
class Index:
    dtype: str
    coerce: bool = False
    nullable: bool = False


class DataFrameSchema:
    def __init__(
        self,
        columns: Optional[Dict[str, Column]] = None,
        dtype: Optional[str] = None,
        index: Optional[Index] = None,
        strict: bool = False,
        coerce: bool = False,
    ) -> None:
        self.columns = columns or {}
        self.dtype = dtype
        self.index = index
        self.strict = strict
        self.coerce = coerce

    def validate(self, df: pd.DataFrame, lazy: bool = False) -> pd.DataFrame:
        failures = []

        validated = df.copy()

        if self.dtype in {"float", "float64"}:
            for column in validated.columns:
                validated[column] = pd.to_numeric(validated[column], errors="coerce")
            if validated.isnull().any().any():
                failures.append({"check": "dtype", "failure_case": "non-numeric value"})

        for name, column in self.columns.items():
            if name not in validated.columns:
                failures.append({"check": "required_column", "failure_case": name})
                continue

            if column.coerce:
                if column.dtype in {"float", "float64"}:
                    validated[name] = pd.to_numeric(validated[name], errors="coerce")
                elif column.dtype in {"int", "int64"}:
                    validated[name] = pd.to_numeric(validated[name], errors="coerce")

            if not column.nullable and validated[name].isnull().any():
                failures.append({"check": "nullable", "failure_case": name})

        if self.index:
            if self.index.coerce and self.index.dtype.startswith("datetime"):
                validated.index = pd.to_datetime(validated.index, errors="coerce")
            if validated.index.dtype.kind != "M":
                failures.append({"check": "index", "failure_case": "expected datetime index"})
            if not self.index.nullable and validated.index.isnull().any():
                failures.append({"check": "index_nullable", "failure_case": "null index entries"})

        if failures:
            raise SchemaError("Schema validation failed", pd.DataFrame(failures))

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
