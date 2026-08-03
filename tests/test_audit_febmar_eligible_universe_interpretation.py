from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "audit_febmar_eligible_universe_interpretation.py"
SPEC = importlib.util.spec_from_file_location("eligible_universe_interpretation", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_month": ["2025-02"] * 4,
        "candidate_id": ["a", "b", "c", "d"],
        "side_name": ["long", "short", "long", "short"],
        "__symbol__": ["BTC", "BTC", "ETH", "ETH"],
        "__ts__": [pd.Timestamp("2025-02-01T00:00:00Z")] * 4,
        "base_group_rows_timestamp_global": [4] * 4,
        "base_group_rows_timestamp_side": [2] * 4,
    })


def test_hourly_universe_proves_group_size_is_exact_cardinality() -> None:
    result = MODULE.hourly_universe(_frame())
    row = result.iloc[0]
    assert row.canonical_rows == 4 and row.assets == 2 and row.sides == 2
    assert row.derived_assets_times_sides == 4
    assert row.exact_global_cardinality_match


def test_density_audit_does_not_misclassify_group_size_as_density() -> None:
    result = MODULE.density_field_audit(["base_input__median_volume_z"])
    group = result.loc[result.field.str.contains("candidate_group_rows")].iloc[0]
    raw = result.loc[result.field.str.contains("raw/pre-filter")].iloc[0]
    assert group.status == "NOT_TRUE_DENSITY"
    assert raw.status == "MISSING"
