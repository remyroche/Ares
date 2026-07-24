from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.feature_transform_contract import (
    file_sha256,
    ordered_names_hash,
)
from extreme_price_movements.features_gmm_ae import ae_gmm_learned_transform_hash
from extreme_price_movements.inference import live_meta_feature_overlays as overlays


def _state() -> dict:
    return {
        "schema_version": "ae_gmm_v1",
        "feature_columns": ["feature_a", "feature_b"],
        "center": [0.0, 0.0],
        "scale": [1.0, 1.0],
        "gmm_n_components": 2,
        "gmm_covariance_type": "diag",
        "temporal_feature_contract": "row_independent_v1",
    }


def test_live_loader_verifies_single_cycle_state_contract(tmp_path, monkeypatch) -> None:
    run_id = "contract_run"
    state_dir = tmp_path / "artifacts" / run_id / "ae_gmm_state"
    state_dir.mkdir(parents=True)
    state_path = state_dir / "ae_gmm_state.pkl"
    state_path.write_bytes(b"serialized-state")
    state = _state()
    transform_hash = ae_gmm_learned_transform_hash(state)
    manifest = {
        "contract": "single_cycle_frozen_ae_gmm_bundle_v2",
        "state_sha256": file_sha256(state_path),
        "input_feature_order_hash": ordered_names_hash(state["feature_columns"]),
        "learned_transform_hash": transform_hash,
        "cycle_state_hash": transform_hash,
        "materialized_transform_rules": {"clip": [-8.0, 8.0]},
    }
    (state_dir / "ae_gmm_state_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(overlays, "load_ae_gmm_state_artifact", lambda _: state)
    overlays.load_live_ae_gmm_state_payload.cache_clear()

    payload = overlays.load_live_ae_gmm_state_payload(str(tmp_path), run_id)

    assert payload["input_feature_columns"] == ["feature_a", "feature_b"]
    assert payload["manifest"]["cycle_state_hash"] == transform_hash


def test_live_loader_accepts_bare_hex_input_order_hash(tmp_path, monkeypatch) -> None:
    run_id = "bare_hex_contract_run"
    state_dir = tmp_path / "artifacts" / run_id / "ae_gmm_state"
    state_dir.mkdir(parents=True)
    state_path = state_dir / "ae_gmm_state.pkl"
    state_path.write_bytes(b"serialized-state")
    state = _state()
    transform_hash = ae_gmm_learned_transform_hash(state)
    manifest = {
        "contract": "single_cycle_frozen_ae_gmm_bundle_v2",
        "state_sha256": file_sha256(state_path).removeprefix("sha256:"),
        "input_feature_order_hash": ordered_names_hash(
            state["feature_columns"]
        ).removeprefix("sha256:"),
        "learned_transform_hash": transform_hash,
        "cycle_state_hash": transform_hash,
        "materialized_transform_rules": {"clip": [-8.0, 8.0]},
    }
    (state_dir / "ae_gmm_state_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(overlays, "load_ae_gmm_state_artifact", lambda _: state)
    overlays.load_live_ae_gmm_state_payload.cache_clear()

    payload = overlays.load_live_ae_gmm_state_payload(str(tmp_path), run_id)

    assert payload["input_feature_columns"] == ["feature_a", "feature_b"]


def test_live_loader_rejects_state_hash_mismatch(tmp_path, monkeypatch) -> None:
    run_id = "bad_contract_run"
    state_dir = tmp_path / "artifacts" / run_id / "ae_gmm_state"
    state_dir.mkdir(parents=True)
    state_path = state_dir / "ae_gmm_state.pkl"
    state_path.write_bytes(b"serialized-state")
    (state_dir / "ae_gmm_state_manifest.json").write_text(
        json.dumps({"state_sha256": "not-the-state-hash"})
    )
    monkeypatch.setattr(overlays, "load_ae_gmm_state_artifact", lambda _: _state())
    overlays.load_live_ae_gmm_state_payload.cache_clear()

    assert overlays.load_live_ae_gmm_state_payload(str(tmp_path), run_id) == {}


def test_live_loader_routes_distinct_ae_gmm_states_by_side(tmp_path, monkeypatch) -> None:
    run_id = "side_routed_contract_run"
    artifact_root = tmp_path / "artifacts" / run_id
    artifact_root.mkdir(parents=True)
    long_path = tmp_path / "long_state.pkl"
    short_path = tmp_path / "short_state.pkl"
    long_path.write_bytes(b"long-state")
    short_path.write_bytes(b"short-state")
    states = {
        str(long_path.resolve()): {**_state(), "feature_columns": ["long_input"]},
        str(short_path.resolve()): {**_state(), "feature_columns": ["short_input"]},
    }
    (artifact_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "side_routed_model_handoff_v1",
                "routes": {
                    "long": {
                        "ae_gmm": {
                            "path": str(long_path),
                            "sha256": file_sha256(long_path),
                            "input_feature_order_hash": ordered_names_hash(["long_input"]),
                        }
                    },
                    "short": {
                        "ae_gmm": {
                            "path": str(short_path),
                            "sha256": file_sha256(short_path),
                            "input_feature_order_hash": ordered_names_hash(["short_input"]),
                        }
                    },
                },
                "routing_contract": {"key": "side_name"},
            }
        )
    )
    monkeypatch.setattr(
        overlays,
        "load_ae_gmm_state_artifact",
        lambda path: states[str(path.resolve())],
    )
    overlays.load_live_ae_gmm_state_payload.cache_clear()

    payload = overlays.load_live_ae_gmm_state_payload(str(tmp_path), run_id)

    assert overlays.live_ae_gmm_input_feature_columns(payload, "long") == ["long_input"]
    assert overlays.live_ae_gmm_input_feature_columns(payload, "short") == ["short_input"]
    assert overlays.live_ae_gmm_input_feature_columns(payload) == [
        "long_input",
        "short_input",
    ]


def test_live_ae_gmm_overwrite_recomputes_upstream_regime_inputs(monkeypatch) -> None:
    source_calls: list[bool] = []

    def _materialize_source(
        features,
        *,
        side,
        signal_bar_ts,
        required_columns,
        overwrite_existing=False,
    ):
        source_calls.append(bool(overwrite_existing))
        out = features.copy()
        if overwrite_existing:
            out["regime_input"] = np.float32(2.0)
        return out

    def _transform(features, state, *, index=None):
        values = pd.to_numeric(features["regime_input"], errors="coerce").to_numpy(
            dtype=np.float32,
            copy=False,
        )
        return pd.DataFrame({"gmm_ood_score": values}, index=index)

    monkeypatch.setattr(
        overlays,
        "materialize_live_source_regime_features",
        _materialize_source,
    )
    monkeypatch.setattr(overlays, "transform_ae_gmm_features", _transform)

    features = pd.DataFrame(
        {"regime_input": np.asarray([1.0, 1.0], dtype=np.float32)},
        index=["AAA", "BBB"],
    )
    payload = {
        "state": {
            "enabled": True,
            "feature_columns": ["regime_input"],
        },
        "state_path": "test-state.pkl",
    }
    result = overlays.materialize_live_ae_gmm_features(
        features,
        side="long",
        signal_bar_ts="2026-07-17T09:00:00Z",
        required_columns=["gmm_ood_score"],
        state_payload=payload,
        overwrite_existing=True,
    )

    assert source_calls == [True]
    np.testing.assert_array_equal(
        result["gmm_ood_score"].to_numpy(),
        np.asarray([2.0, 2.0], dtype=np.float32),
    )


def test_live_ae_gmm_transform_selects_complete_route_before_transform(monkeypatch) -> None:
    calls: list[str] = []

    def _transform(features, state, *, index=None):
        calls.append(str(state["route_marker"]))
        return pd.DataFrame(
            {
                "gmm_ood_score": np.full(
                    len(features), float(state["output_value"]), dtype=np.float32
                )
            },
            index=index,
        )

    monkeypatch.setattr(overlays, "transform_ae_gmm_features", _transform)
    payload = {
        "schema": "side_routed_ae_gmm_payload_v1",
        "states_by_side": {
            "long": {
                "state": {
                    "enabled": True,
                    "feature_columns": ["long_input"],
                    "route_marker": "long",
                    "output_value": 1.0,
                }
            },
            "short": {
                "state": {
                    "enabled": True,
                    "feature_columns": ["short_input"],
                    "route_marker": "short",
                    "output_value": -1.0,
                }
            },
        },
    }
    features = pd.DataFrame(
        {
            "long_input": np.asarray([2.0], dtype=np.float32),
            "short_input": np.asarray([3.0], dtype=np.float32),
        },
        index=["BTC/USD:USD"],
    )

    long_result = overlays.materialize_live_ae_gmm_features(
        features,
        side="long",
        signal_bar_ts="2026-07-17T09:00:00Z",
        required_columns=["gmm_ood_score"],
        state_payload=payload,
    )
    short_result = overlays.materialize_live_ae_gmm_features(
        features,
        side="short",
        signal_bar_ts="2026-07-17T09:00:00Z",
        required_columns=["gmm_ood_score"],
        state_payload=payload,
    )

    assert calls == ["long", "short"]
    assert long_result.loc["BTC/USD:USD", "gmm_ood_score"] == pytest.approx(1.0)
    assert short_result.loc["BTC/USD:USD", "gmm_ood_score"] == pytest.approx(-1.0)
