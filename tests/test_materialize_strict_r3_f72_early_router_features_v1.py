from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "materialize_strict_r3_f72_early_router_features_v1.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("f72_early_router_features", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_early_materializer_defaults_to_retained_f72_selection() -> None:
    assert MODULE.DEFAULT_B_WINNER.name == "selection.json"
    assert "strict_r3_b0_family_addback_20260826_v1_policy_ordinal_base_g3" in str(MODULE.DEFAULT_B_WINNER)


def test_early_materializer_uses_the_frozen_180_day_context() -> None:
    source = SCRIPT.read_text()
    assert 'parser.add_argument("--warmup-days", type=int, default=180)' in source


def test_minimal_contract_declares_rv48_causal_parent_without_exposing_it() -> None:
    source = SCRIPT.read_text()
    assert 'GENERATION_ONLY_DEPENDENCIES = ("rv_120h",)' in source
    assert 'generation_fields = list(dict.fromkeys([*fields, *GENERATION_ONLY_DEPENDENCIES]))' in source
    assert '"generation_only_dependencies": list(GENERATION_ONLY_DEPENDENCIES)' in source


def test_missing_predecessor_identity_partition_is_explicitly_opt_in_and_coverage_guarded() -> None:
    source = SCRIPT.read_text()
    assert "--allow-missing-predecessor-identity-partition" in source
    assert "if position == 0 and allow_missing_predecessor_partition" in source
    assert "missing predecessor partition also removed the first decision-time coverage" in source


def test_early_materializer_reads_selected_features_and_records_key(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"selected_features": ["a", "b"]}))
    fields, key = MODULE._read_fields(selection, ("selected_features", "features"))
    assert fields == ["a", "b"]
    assert key == "selected_features"


def test_early_materializer_rejects_an_undeclared_contract_key(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"other": ["a"]}))
    with pytest.raises(ValueError, match="one of"):
        MODULE._read_fields(selection, ("selected_features", "features"))
