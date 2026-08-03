import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("common_transition", ROOT / "scripts/run_common_semantic_transition_cost_clearing_ablation.py")
MOD = importlib.util.module_from_spec(SPEC); assert SPEC and SPEC.loader; SPEC.loader.exec_module(MOD)


def _context(timestamps):
    data = {"signal_context_utc": pd.to_datetime(timestamps), "common_transition_context_available": True}
    data.update({column: 1.0 for column in MOD.CANONICAL_FEATURES})
    return pd.DataFrame(data)


def _exact(timestamps):
    return pd.DataFrame({"candidate_id": [f"x{i}" for i in range(len(timestamps))], "side_name": "long", "__symbol__": "X", "__ts__": pd.to_datetime(timestamps)})


def test_common_geometry_is_exact_timestamp_join_with_no_fill():
    exact = _exact(["2026-01-01T00:00Z", "2026-01-01T01:00Z"])
    joined, audit = MOD.exact_timestamp_join(exact, _context(["2026-01-01T00:00Z"]), lineage="test")
    assert joined.candidate_id.tolist() == ["x0"]
    assert audit.loc[0, "missing_timestamp_candidate_rows"] == 1
    assert audit.loc[0, "fill"] == "none"


def test_common_geometry_incomplete_features_are_excluded_not_filled():
    exact = _exact(["2026-01-01T00:00Z"])
    context = _context(["2026-01-01T00:00Z"])
    context.loc[0, MOD.TRANSITION[0]] = float("nan")
    try:
        MOD.exact_timestamp_join(exact, context, lineage="test")
    except ValueError as error:
        assert "no exact complete" in str(error)
    else:
        raise AssertionError("incomplete context must not be filled")


def test_feature_family_contract_is_complete_and_disjoint():
    assert len(MOD.CANONICAL_FEATURES) == 90
    assert len(MOD.STATE) == 36 and len(MOD.TRANSITION) == 54
    assert set(MOD.STATE).isdisjoint(MOD.TRANSITION)
    assert set(MOD.STATE).union(MOD.TRANSITION) == set(MOD.CANONICAL_FEATURES)
