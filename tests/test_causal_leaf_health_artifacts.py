from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.causal_leaf_health import CausalLeafHealthConfig
from extreme_price_movements.causal_leaf_health_artifacts import (
    build_strict_oof_causal_leaf_health,
    spool_completed_strict_oof_family_inputs,
)
from extreme_price_movements.causal_leaf_health_streaming import (
    materialize_strict_oof_causal_leaf_health_streaming,
)


def _empty_shard() -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": pd.Series(dtype="string"),
        "decision_ts": pd.Series(dtype="datetime64[ns, UTC]"),
        "label_available_ts": pd.Series(dtype="datetime64[ns, UTC]"),
        "side_name": pd.Series(dtype="string"), "fold_id": pd.Series(dtype="string"),
        "feature_generation_ts": pd.Series(dtype="datetime64[ns, UTC]"),
        "feature_contract_sha256": pd.Series(dtype="string"), "base_expected_bps": pd.Series(dtype="float64"),
        "asset": pd.Series(dtype="string"), "r3_class": pd.Series(dtype="int8"),
    })


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "strict"
    root.mkdir()
    transport = "transport_a"
    (root / "strict_oof_reasoning_manifest.json").write_text(json.dumps({
        "status": "STRICT_OOF_BASE_REASONING_MATERIALIZED",
        "transports": [transport],
    }))
    timestamp = pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T02:00:00Z"], utc=True)
    shard = pd.DataFrame({
        "candidate_id": ["a", "b"], "decision_ts": timestamp,
        "label_available_ts": timestamp + pd.Timedelta(hours=1),
        "side_name": ["long", "long"], "fold_id": ["fold-1", "fold-1"],
        "feature_generation_ts": timestamp,
        "feature_contract_sha256": ["f" * 64] * 2,
        "base_expected_bps": [5.0, 5.0], "asset": ["A", "B"], "r3_class": [2, 1],
    })
    for side, content in (("long", shard), ("short", _empty_shard())):
        directory = root / "base_prediction_shards" / transport / side
        directory.mkdir(parents=True)
        content.to_parquet(directory / "strict_oof_predictions.parquet", index=False)
        _empty_shard().to_parquet(directory / "outer_predictions.parquet", index=False)

    artifact = root / "strict_oof_base_reasoning" / transport / "folds" / "long" / "fold-1" / "p_clear"
    artifact.mkdir(parents=True)
    assignments = pd.DataFrame({
        "candidate_id": ["a", "b"], "__ts__": timestamp, "side_name": ["long", "long"],
        "head_name": ["p_clear", "p_clear"], "fold_id": ["fold-1", "fold-1"],
        "leaf_assignment__model_00_head_tree_000": np.array([101, 102], dtype=np.uint64),
    })
    assignments.to_parquet(artifact / "leaf_assignments.parquet", index=False)
    pd.DataFrame({
        "head_name": ["p_clear", "p_clear"], "side_name": ["long", "long"], "fold_id": ["fold-1", "fold-1"],
        "model_slot": [0, 0], "head_tree_slot": [0, 0], "leaf_token": np.array([101, 102], dtype=np.uint64),
        "rule_signature": ["r0", "r1"], "tree_leaf_value": [1.0, -1.0],
        "ensemble_tree_contribution": [1.0, -1.0],
    }).to_parquet(artifact / "leaf_rule_catalog.parquet", index=False)
    pd.DataFrame({
        "candidate_id": ["a", "b"], "__ts__": timestamp, "side_name": ["long", "long"],
        "head_name": ["p_clear", "p_clear"], "fold_id": ["fold-1", "fold-1"],
        "base_prediction": [.7, .6],
    }).to_parquet(artifact / "base_reasoning_predictions.parquet", index=False)
    pd.DataFrame({
        "candidate_id": ["a", "b"], "__ts__": timestamp, "side_name": ["long", "long"],
        "head_name": ["p_clear", "p_clear"], "fold_id": ["fold-1", "fold-1"],
        "label__r3_class": [2, 1], "label__net_bps": [50.0, -50.0],
        "label__label_available_ts": timestamp + pd.Timedelta(hours=1),
    }).to_parquet(artifact / "base_reasoning_labels.parquet", index=False)
    (artifact / "base_reasoning_manifest.json").write_text(json.dumps({
        "status": "MATERIALIZED_STRICT_OOF", "head_name": "p_clear", "side_name": "long", "fold_id": "fold-1",
        "provenance": {"model_hashes": ["m"], "feature_contract_sha256": "f" * 64, "class_index": 2},
    }))
    return root


def test_collects_only_matching_completed_strict_head_artifacts(tmp_path: Path) -> None:
    root = _root(tmp_path)
    context = pd.DataFrame({
        "regime_available_utc": pd.date_range("2023-12-31", periods=48, freq="h", tz="UTC"),
        "ctx": np.arange(48, dtype=float),
    })
    result = build_strict_oof_causal_leaf_health(
        [root], causal_context=context, context_feature_columns=("ctx",),
        config=CausalLeafHealthConfig(min_timestamp_support=1, min_day_support=1, min_symbol_support=1),
    )
    assert set(result.health_features["candidate_id"]) == {"a", "b"}
    assert result.family_candidate_states["head_name"].eq("p_clear").all()
    assert not any("leaf_assignment" in column for column in result.family_candidate_states)


def test_spool_keeps_candidate_and_token_free_contribution_parts_separate(tmp_path: Path) -> None:
    root = _root(tmp_path)
    spool = spool_completed_strict_oof_family_inputs([root], tmp_path / "spool")
    assert spool.manifest_path.is_file()
    assert len(spool.candidate_parts) == len(spool.contribution_parts) == 1
    candidates = pd.read_parquet(spool.candidate_parts[0])
    contributions = pd.read_parquet(spool.contribution_parts[0])
    assert set(candidates["candidate_id"]) == {"a", "b"}
    assert not any("leaf_" in str(column) for column in contributions.columns)


def test_streaming_materialiser_consumes_spool_with_strict_health_contract(tmp_path: Path) -> None:
    root = _root(tmp_path)
    spool = spool_completed_strict_oof_family_inputs([root], tmp_path / "spool")
    context = pd.DataFrame({
        "regime_available_utc": pd.date_range("2023-12-31", periods=48, freq="h", tz="UTC"),
        "ctx": np.arange(48, dtype=float),
    })
    output = materialize_strict_oof_causal_leaf_health_streaming(
        spool.root, tmp_path / "health_stream", causal_context=context,
        context_feature_columns=("ctx",),
        config=CausalLeafHealthConfig(min_timestamp_support=1, min_day_support=1, min_symbol_support=1),
        batch_rows=1,
    )
    health = pd.read_parquet(output / "base_leaf_health_features_oof.parquet")
    assert set(health["candidate_id"]) == {"a", "b"}
    assert not any("leaf_" in str(column) for column in health.columns)
    assert json.loads((output / "health_materialization_manifest.json").read_text())["status"] == "CAUSAL_LEAF_HEALTH_MATERIALIZED"
    reference = build_strict_oof_causal_leaf_health(
        [root], causal_context=context, context_feature_columns=("ctx",),
        config=CausalLeafHealthConfig(min_timestamp_support=1, min_day_support=1, min_symbol_support=1),
    ).health_features.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    streamed = health.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    shared = [
        name for name in reference.columns if name.startswith(("base_health__h1__", "base_health__h2__", "base_health__h3__"))
    ]
    assert np.allclose(streamed[shared].to_numpy(float), reference[shared].to_numpy(float), equal_nan=True)


def test_streaming_materialiser_uses_frozen_h5_selection(tmp_path: Path) -> None:
    root = _root(tmp_path)
    spool = spool_completed_strict_oof_family_inputs([root], tmp_path / "spool")
    relationship = "continuous_regime__relationship_break__market__residual_abs_30d"
    context = pd.DataFrame({
        "regime_available_utc": pd.date_range("2023-12-31", periods=48, freq="h", tz="UTC"),
        "ctx": np.arange(48, dtype=float), relationship: np.full(48, 0.3),
    })
    key = ("f" * 64, "long", "p_clear", "r0", "positive")
    output = materialize_strict_oof_causal_leaf_health_streaming(
        spool.root, tmp_path / "health_stream", causal_context=context,
        context_feature_columns=("ctx", relationship),
        config=CausalLeafHealthConfig(
            min_timestamp_support=1, min_day_support=1, min_symbol_support=1,
            selected_relationship_families=frozenset({key}),
        ), batch_rows=1,
    )
    health = pd.read_parquet(output / "base_leaf_health_features_oof.parquet").set_index("candidate_id")
    field = "base_health__h5__p_clear__positive__availability"
    assert health.loc["a", field] == pytest.approx(1.0)
    assert health.loc["b", field] == pytest.approx(0.0)
    assert (output / "leaf_relationship_breaks.parquet").is_file()
