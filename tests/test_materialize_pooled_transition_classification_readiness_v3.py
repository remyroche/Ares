from __future__ import annotations

import hashlib
import json

import pandas as pd

from scripts.materialize_historical_current_common_transition_geometry import CANONICAL_FEATURES, RAW_FIELDS
from scripts.materialize_pooled_transition_classification_readiness_v3 import common_geometry_requirement


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_common_root(tmp_path, *, nofill: str):
    root = tmp_path / "common"
    root.mkdir()
    identity = pd.DataFrame({"__ts__": [pd.Timestamp("2023-01-01T00:00:00Z"), pd.Timestamp("2023-01-01T01:00:00Z")], "__symbol__": ["A", "B"], "side_name": ["long", "short"], "candidate_id": ["one", "two"]})
    historical = identity.copy()
    historical["__decision_ts__"] = historical["__ts__"] + pd.Timedelta(hours=1)
    historical["common_transition_context_available"] = True
    for field in CANONICAL_FEATURES:
        historical[field] = 1.0
    hourly = pd.DataFrame({"signal_context_utc": [pd.Timestamp("2023-01-01T00:00:00Z")]})
    current = pd.DataFrame({"signal_context_utc": [pd.Timestamp("2026-01-01T00:00:00Z")], "common_transition_context_available": [True]})
    for field in CANONICAL_FEATURES:
        hourly[field] = 1.0
        current[field] = 1.0
    files = {"historical_candidate_context": ("historical_candidate_context.parquet", historical), "historical_hourly_state_geometry": ("historical_hourly_state_geometry.parquet", hourly), "current_v4_semantic_context": ("current_v4_semantic_context.parquet", current)}
    outputs = {}
    for name, (filename, frame) in files.items():
        path = root / filename
        frame.to_parquet(path, index=False)
        outputs[name] = {"path": str(path), "sha256": _sha(path)}
    mapping = {"raw_field_overlap": list(RAW_FIELDS), "canonical_feature_columns": list(CANONICAL_FEATURES)}
    audit = {"no_fill": nofill, "canonical_parity": {"feature_count": 90, "all_common_features_declared_by_current_v4": True, "historical_columns_equal_contract": True, "current_columns_equal_contract": True}, "raw_name_overlap": {"count": 9, "fields": list(RAW_FIELDS), "exact_expected_nine": True}}
    for name, payload in (("semantic_mapping", mapping), ("audit", audit)):
        path = root / f"{name}.json"
        path.write_text(json.dumps(payload))
    current_panel = tmp_path / "current.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(current_panel, index=False)
    manifest = {"schema": "historical_current_common_transition_geometry_v1", "status": "MATERIALIZED_STRICT_SEMANTIC_COMMON_GEOMETRY", "outputs": outputs, "semantic_mapping": {"path": "semantic_mapping.json", "sha256": _sha(root / "semantic_mapping.json")}, "audit": {"path": "audit.json", "sha256": _sha(root / "audit.json")}, "sources": {"current_v4_panel": {"sha256": _sha(current_panel)}}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "manifest.sha256").write_text(f"{_sha(root / 'manifest.json')}  manifest.json\n")
    return root, identity, current_panel


def test_common_geometry_requirement_accepts_full_hash_bound_contract(tmp_path) -> None:
    root, labels, current = _write_common_root(tmp_path, nofill="historical lags use exact timestamp reindex; no asof/resample/interpolation/ffill/bfill")
    result = common_geometry_requirement(root, labels, list(CANONICAL_FEATURES), current)
    assert result["ready"] is True
    assert "90 canonical" in result["reason"]


def test_common_geometry_requirement_accepts_equivalent_label_packet_order(tmp_path) -> None:
    root, labels, current = _write_common_root(tmp_path, nofill="historical lags use exact timestamp reindex; no asof/resample/interpolation/ffill/bfill")
    result = common_geometry_requirement(root, labels.iloc[::-1].reset_index(drop=True), list(CANONICAL_FEATURES), current)
    assert result["ready"] is True


def test_common_geometry_requirement_fails_closed_without_no_fill_audit(tmp_path) -> None:
    root, labels, current = _write_common_root(tmp_path, nofill="not recorded")
    result = common_geometry_requirement(root, labels, list(CANONICAL_FEATURES), current)
    assert result["ready"] is False
    assert "no-fill" in result["reason"]
