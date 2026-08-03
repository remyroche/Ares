from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "materialize_july20_23_transition_extension.py"
SPEC = importlib.util.spec_from_file_location("july20_23_transition_extension", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frozen_source_supports_causal_mapping_and_exact_h12_join() -> None:
    source = MODULE.DEFAULT_SOURCE
    scored_labels, candidates, support, _ = MODULE.load_frozen_source(source)
    mapping, coordinates, coordinate_audit = MODULE.build_causal_mapping(scored_labels, support)
    assert len(mapping) == len(scored_labels) == 5760
    assert len(coordinates) == len(mapping)
    assert support["history_resolution_max_utc"].lt(support["execution_decision_utc"]).all()
    assert coordinates["mapping_available_at"].le(coordinates["execution_decision_utc"]).all()
    assert coordinate_audit["reference_window_end_utc"].le(coordinate_audit["snapshot_utc"]).all()
    assert {"__ts__", "side_name", "candidate_id"}.issubset(candidates.columns)


def test_extension_geometry_is_exact_90_fields_without_fill() -> None:
    _, candidates, _, _ = MODULE.load_frozen_source(MODULE.DEFAULT_SOURCE)
    geometry = MODULE.build_geometry(candidates)
    assert list(geometry.columns[1:91]) == list(MODULE.CANONICAL_FEATURES)
    assert len(MODULE.CANONICAL_FEATURES) == 90
    # The initial 12h has no exact 12h prior state and is deliberately missing.
    initial = geometry.iloc[:12]
    lagged = [column for column in MODULE.CANONICAL_FEATURES if "past_delta_12h" in column]
    assert initial.loc[:, lagged].isna().all().all()
    assert geometry.loc[geometry["common_transition_context_available"], MODULE.CANONICAL_FEATURES].notna().all().all()


def test_materialized_extension_is_manifest_bound_and_nonpromotable() -> None:
    root = MODULE.DEFAULT_OUTPUT
    manifest, _ = MODULE._read_manifest(root, label="materialized extension")
    panel_path = root / "transition_panel.parquet"
    assert manifest["promotion_eligible"] is False
    assert manifest["mapping_provenance_role"] == MODULE.PROVENANCE
    assert manifest["outputs_sha256"][panel_path.name] == MODULE.sha256(panel_path)
    panel = pd.read_parquet(panel_path)
    assert panel["context_available"].all()
    assert panel["mapping_provenance_role"].eq(MODULE.PROVENANCE).all()
