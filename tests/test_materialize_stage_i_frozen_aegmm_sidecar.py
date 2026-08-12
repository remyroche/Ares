from __future__ import annotations

import json
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

import scripts.materialize_stage_i_frozen_aegmm_sidecar as bridge


def _selector(tmp_path):
    root = tmp_path / "selector"
    root.mkdir()
    features = pd.DataFrame({
        "candidate_id": ["a", "b", "c"],
        "__ts__": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
        "__symbol__": ["BTC", "ETH", "SOL"],
        "x": [0.1, 0.2, 0.3],
    })
    ledger = features.loc[:, ["candidate_id", "__ts__", "__symbol__"]].copy()
    ledger["side_name"] = ["long", "short", "long"]
    features_path = root / "selector_features.parquet"
    features.to_parquet(features_path, index=False)
    ledger.to_parquet(root / "selector_ledger.parquet", index=False)
    root.joinpath("manifest.json").write_text(json.dumps({
        "artifact_integrity": {
            "selector_features_sha256": bridge.file_sha256(features_path),
        },
    }))
    return root


def test_frozen_aegmm_sidecar_is_identity_bound_and_forces_row_independent_projection(tmp_path, monkeypatch) -> None:
    selector = _selector(tmp_path)
    state_path = tmp_path / "state.pkl"
    pd.to_pickle({"enabled": True, "feature_columns": ["side", "x"]}, state_path)
    observed = {}

    def fake_transform(frame, state, *, index, prefix):
        observed["contract"] = state["temporal_feature_contract"]
        assert list(frame.columns) == ["side", "x"]
        return pd.DataFrame({f"{prefix}gmm_ood_score": np.asarray(frame["x"], dtype=np.float32)}, index=index)

    monkeypatch.setattr(bridge, "transform_ae_gmm_features", fake_transform)
    output = tmp_path / "sidecar.parquet"
    manifest = bridge.materialize(
        selector_dir=selector, state_path=state_path, output_path=output, min_source_overlap=1.0,
    )
    assert observed["contract"] == "row_independent_v1"
    assert manifest["state_input_overlap"] == 1.0
    assert manifest["side_reconstructed_from_selector_ledger"] is True
    assert manifest["frozen_state_sha256"] == sha256(state_path.read_bytes()).hexdigest()
    produced = pd.read_parquet(output)
    assert produced.columns.tolist() == [*bridge.IDENTITY, "meta_lgbm_gmm_ood_score"]


def test_frozen_aegmm_sidecar_rejects_insufficient_input_overlap(tmp_path) -> None:
    selector = _selector(tmp_path)
    state_path = tmp_path / "state.pkl"
    pd.to_pickle({"enabled": True, "feature_columns": ["missing_1", "missing_2", "x"]}, state_path)
    with pytest.raises(ValueError, match="source overlap"):
        bridge.materialize(
            selector_dir=selector, state_path=state_path, output_path=tmp_path / "out.parquet",
            min_source_overlap=0.50,
        )
