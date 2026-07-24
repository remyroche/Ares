"""Strict support validation for deterministic path-archetype taxonomies.

This module deliberately separates the one approved taxonomy merge from support
validation.  Validation never changes labels: a geometry with insufficient
class support is rejected rather than repaired by reassigning rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


LEGACY_FAST_CLASSES: tuple[str, str] = (
    "fast_clean_winner",
    "fast_winner_early_drawdown",
)
FAST_REALIZATION_WINNER = "fast_realization_winner"

# The downstream CatBoost class order is a contract.  Do not infer it from a
# particular sample: a missing class must make the geometry ineligible.
MERGED_PATH_ARCHETYPE_CLASSES: tuple[str, ...] = (
    "immediate_adverse_path",
    "early_mfe_full_reversal",
    FAST_REALIZATION_WINNER,
    "late_breakout",
    "slow_grinder",
    "noisy_timeout_usable_mfe",
    "dead_timeout",
)


@dataclass(frozen=True)
class PathArchetypeSupportConfig:
    """Support floors for a hard-label path taxonomy.

    ``min_month_side_denominator=None`` means every non-empty calendar
    month x side cell is checked.  Passing a positive integer is an explicit
    opt-in exemption for cells with fewer rows, and those exemptions are
    reported rather than silently discarded.
    """

    label_column: str = "path_geometry_label"
    timestamp_column: str = "__ts__"
    side_column: str = "side"
    classes: tuple[str, ...] = MERGED_PATH_ARCHETYPE_CLASSES
    min_global_class_share: float = 0.01
    min_month_side_class_share: float = 0.005
    min_month_side_denominator: int | None = None

    def validate(self) -> None:
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError("classes must be a non-empty explicit unique taxonomy")
        if set(LEGACY_FAST_CLASSES).intersection(self.classes):
            raise ValueError("classes must use the merged fast_realization_winner taxonomy")
        for name, value in (
            ("min_global_class_share", self.min_global_class_share),
            ("min_month_side_class_share", self.min_month_side_class_share),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.min_month_side_denominator is not None and self.min_month_side_denominator < 1:
            raise ValueError("min_month_side_denominator must be positive or None")


@dataclass(frozen=True)
class PathArchetypeSupportResult:
    """All support tables plus a deterministic geometry action."""

    global_support: pd.DataFrame
    month_support: pd.DataFrame
    side_support: pd.DataFrame
    month_side_support: pd.DataFrame
    violations: pd.DataFrame
    exemptions: pd.DataFrame
    recommended_action: str
    rows: int
    config: PathArchetypeSupportConfig

    @property
    def accepted(self) -> bool:
        return self.recommended_action == "accept_geometry"


def merge_fast_realization_winner(
    values: pd.Series | pd.DataFrame,
    *,
    label_column: str = "path_geometry_label",
) -> pd.Series | pd.DataFrame:
    """Return a copy with only the approved two-class fast merge applied.

    Every other label, order, index, dtype where feasible, and DataFrame
    column is preserved.  This is intentionally the sole relabelling helper
    used by the support contract.
    """

    if isinstance(values, pd.DataFrame):
        if label_column not in values:
            raise KeyError(f"missing label column: {label_column}")
        result = values.copy()
        result[label_column] = merge_fast_realization_winner(result[label_column])
        return result
    if not isinstance(values, pd.Series):
        raise TypeError("values must be a pandas Series or DataFrame")
    mapped = values.mask(values.isin(LEGACY_FAST_CLASSES), FAST_REALIZATION_WINNER)
    # Preserve nullable-string semantics instead of coercing unrelated labels.
    return mapped.rename(values.name)


def _empty_support(index_columns: Sequence[str], classes: Sequence[str]) -> pd.DataFrame:
    columns = [*index_columns, "path_geometry_label", "rows", "denominator_rows", "share"]
    return pd.DataFrame(columns=columns)


def _support_table(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    label_column: str,
    classes: Sequence[str],
) -> pd.DataFrame:
    """Cross-join explicit groups/classes and calculate zero-inclusive shares."""

    original_group_columns = tuple(group_columns)
    if frame.empty:
        return _empty_support(original_group_columns, classes)
    groups = frame.loc[:, list(original_group_columns)].drop_duplicates().copy() if original_group_columns else pd.DataFrame({"__all__": ["all"]})
    work = frame.copy()
    if not group_columns:
        work["__all__"] = "all"
        group_columns = ("__all__",)
    denominator = (
        work.groupby(list(group_columns), dropna=False, observed=False)
        .size()
        .rename("denominator_rows")
        .reset_index()
    )
    counts = (
        work.groupby([*group_columns, label_column], dropna=False, observed=False)
        .size()
        .rename("rows")
        .reset_index()
        .rename(columns={label_column: "path_geometry_label"})
    )
    grid = groups.merge(pd.DataFrame({"path_geometry_label": list(classes)}), how="cross")
    result = grid.merge(counts, on=[*group_columns, "path_geometry_label"], how="left")
    result = result.merge(denominator, on=list(group_columns), how="left")
    result["rows"] = result["rows"].fillna(0).astype(np.int64)
    result["denominator_rows"] = result["denominator_rows"].fillna(0).astype(np.int64)
    result["share"] = np.divide(
        result["rows"].to_numpy(dtype=float),
        result["denominator_rows"].to_numpy(dtype=float),
        out=np.zeros(len(result), dtype=float),
        where=result["denominator_rows"].to_numpy(dtype=float) > 0.0,
    )
    if "__all__" in result:
        result = result.drop(columns="__all__")
    return result.sort_values([*original_group_columns, "path_geometry_label"], kind="stable").reset_index(drop=True)


def _violation_rows(table: pd.DataFrame, *, scope: str, threshold: float, denominator_floor: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if table.empty:
        return pd.DataFrame(), pd.DataFrame()
    eligible = pd.Series(True, index=table.index)
    if denominator_floor is not None:
        eligible = table["denominator_rows"].ge(denominator_floor)
    deficient = eligible & table["share"].lt(threshold)
    violations = table.loc[deficient].copy()
    exemptions = table.loc[~eligible].copy()
    for result, status in ((violations, "violation"), (exemptions, "explicit_tiny_denominator_exemption")):
        result["scope"] = scope
        result["threshold"] = threshold
        result["status"] = status
    return violations, exemptions


def validate_path_archetype_support(
    frame: pd.DataFrame,
    config: PathArchetypeSupportConfig = PathArchetypeSupportConfig(),
) -> PathArchetypeSupportResult:
    """Validate a *previously merged* hard-label taxonomy without relabelling.

    Naive timestamps follow the repository's UTC contract.  Invalid timestamps,
    missing labels, unexpected labels, legacy fast labels, and unsupported
    expected classes all lead to ``reject_geometry``.
    """

    config.validate()
    required = (config.label_column, config.timestamp_column, config.side_column)
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"missing required support columns: {missing}")
    work = frame.loc[:, list(required)].copy()
    timestamps = pd.to_datetime(work[config.timestamp_column], utc=True, errors="coerce")
    work["__month__"] = timestamps.dt.strftime("%Y-%m").astype("string")
    labels = work[config.label_column].astype("string")
    work[config.label_column] = labels
    work[config.side_column] = work[config.side_column].astype("string")

    structural: list[dict[str, object]] = []
    invalid_timestamp_rows = int(timestamps.isna().sum())
    if invalid_timestamp_rows:
        structural.append({"scope": "input", "status": "invalid_timestamp", "rows": invalid_timestamp_rows})
    missing_side_rows = int(work[config.side_column].isna().sum())
    if missing_side_rows:
        structural.append({"scope": "input", "status": "missing_side", "rows": missing_side_rows})
    missing_label_rows = int(labels.isna().sum())
    if missing_label_rows:
        structural.append({"scope": "input", "status": "missing_label", "rows": missing_label_rows})
    legacy_rows = int(labels.isin(LEGACY_FAST_CLASSES).sum())
    if legacy_rows:
        structural.append({"scope": "taxonomy", "status": "legacy_fast_labels_not_merged", "rows": legacy_rows})
    unexpected = sorted(set(labels.dropna().unique()).difference(config.classes))
    for label in unexpected:
        structural.append({"scope": "taxonomy", "status": "unexpected_class", "path_geometry_label": label, "rows": int(labels.eq(label).sum())})

    valid = work.loc[timestamps.notna() & work[config.side_column].notna() & labels.notna()].copy()
    global_support = _support_table(valid, group_columns=(), label_column=config.label_column, classes=config.classes)
    month_support = _support_table(valid, group_columns=("__month__",), label_column=config.label_column, classes=config.classes)
    side_support = _support_table(valid, group_columns=(config.side_column,), label_column=config.label_column, classes=config.classes)
    month_side_support = _support_table(valid, group_columns=("__month__", config.side_column), label_column=config.label_column, classes=config.classes)

    global_violations, _ = _violation_rows(global_support, scope="global", threshold=config.min_global_class_share)
    month_side_violations, exemptions = _violation_rows(
        month_side_support,
        scope="month_side",
        threshold=config.min_month_side_class_share,
        denominator_floor=config.min_month_side_denominator,
    )
    violations = pd.concat(
        [pd.DataFrame(structural), global_violations, month_side_violations],
        ignore_index=True,
        sort=False,
    )
    action = "accept_geometry" if violations.empty else "reject_geometry"
    return PathArchetypeSupportResult(
        global_support=global_support,
        month_support=month_support,
        side_support=side_support,
        month_side_support=month_side_support,
        violations=violations,
        exemptions=exemptions,
        recommended_action=action,
        rows=int(len(frame)),
        config=config,
    )
