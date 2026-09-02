from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_strict_r3_p8u_precision_preservation_feature_prescreen_v1.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("p8u_feature_prescreen", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_development_months_require_cross_year_span() -> None:
    months = MODULE._months("2025-11,2026-03,2026-07")
    assert [f"{month:%Y-%m}" for month in months] == ["2025-11", "2026-03", "2026-07"]
    with pytest.raises(ValueError):
        MODULE._months("2026-01,2026-03,2026-05")


def test_subspace_draws_are_deterministic_and_bounded() -> None:
    fields = tuple("abcdefghij")
    first = MODULE._subspace_fields(fields, count=5, size=4)
    second = MODULE._subspace_fields(fields, count=5, size=4)
    assert first == second
    assert len(first) == 5
    assert all(len(item) == 4 and len(set(item)) == 4 for item in first)


def test_feature_family_routing_is_stable() -> None:
    assert MODULE._family("funding_rate_change_1h") == "funding"
    assert MODULE._family("oi_zscore_24h") == "oi_leverage"
    assert MODULE._family("book_impact_residual") == "liquidity"
    assert MODULE._family("transition_entropy") == "state"
