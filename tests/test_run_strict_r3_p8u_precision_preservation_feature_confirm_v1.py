from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_strict_r3_p8u_precision_preservation_feature_confirm_v1.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("p8u_feature_confirm", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_confirmation_months_are_fixed_cross_year_panel() -> None:
    assert [f"{item:%Y-%m}" for item in MODULE._months("2025-11,2026-01,2026-03,2026-05,2026-07")] == list(MODULE.MONTHS)
    with pytest.raises(ValueError):
        MODULE._months("2025-11,2026-03,2026-05,2026-07,2026-08")


def test_frozen_field_contract_requires_bounded_unique_fields(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"selected_features": [f"f{i}" for i in range(25)]}))
    assert len(MODULE._fields(good)) == 25
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"selected_features": ["f"] * 25}))
    with pytest.raises(ValueError):
        MODULE._fields(bad)


def test_confirmation_uses_timestamp_level_not_candidate_level_components() -> None:
    source = SCRIPT.read_text()
    assert 'candidate_parts[0]["__decision_ts__"].duplicated()' in source
    assert 'candidate_parts[0]["candidate_id"]' not in source
