from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.base_reasoning_representation import (
    BaseReasoningRepresentationConfig,
    BaseReasoningRepresentationError,
    build_base_reasoning_representation,
)
import extreme_price_movements.base_reasoning_representation as representation_module


HEADS = (("p_adverse", 0), ("p_weak", 1), ("p_clear", 2))


def _write_transport(root: Path, *, second_label_at_feature_time: bool = False) -> Path:
    transport = "transport_one"
    index_rows: list[dict[str, object]] = []
    for side in ("long", "short"):
        shard_rows: list[dict[str, object]] = []
        outer_rows: list[dict[str, object]] = []
        for fold_num, fold in enumerate(("inner_00", "inner_01", "outer")):
            fold_id = f"{transport}_{fold}"
            timestamp = pd.Timestamp("2024-01-02", tz="UTC") + pd.Timedelta(days=2 * fold_num)
            candidate = f"{side}-{fold}"
            label_available = timestamp + pd.Timedelta(hours=1)
            if second_label_at_feature_time and fold_num == 0:
                label_available = timestamp + pd.Timedelta(days=2)
            prediction_row = {
                "candidate_id": candidate, "decision_ts": timestamp,
                "feature_generation_ts": timestamp, "label_available_ts": label_available,
                "side_name": side, "fold_id": fold_id, "net_bps": 20.0,
            }
            (outer_rows if fold == "outer" else shard_rows).append(prediction_row)
            for head, class_index in HEADS:
                relative = Path("strict_oof_base_reasoning") / transport / "folds" / side / fold / head
                artifact = root / relative
                artifact.mkdir(parents=True)
                identity = pd.DataFrame({"candidate_id": [candidate], "__ts__": [timestamp], "side_name": [side]})
                pd.DataFrame({
                    **identity, "head_name": [head], "fold_id": [fold_id],
                    "leaf_assignment__model_00_head_tree_000": np.array([101], dtype=np.uint64),
                }).to_parquet(artifact / "leaf_assignments.parquet", index=False)
                pd.DataFrame({
                    **identity, "head_name": [head], "fold_id": [fold_id],
                    "base_reasoning__g3_contribution_svd_00": np.array([2.0], dtype=np.float32),
                    "base_reasoning__g3_contribution_svd_01": np.array([-1.0], dtype=np.float32),
                }).to_parquet(artifact / "contribution_bundle.parquet", index=False)
                pd.DataFrame({
                    **identity, "head_name": [head], "fold_id": [fold_id],
                    "base_reasoning__g3_balance": np.array([0.5], dtype=np.float32),
                    "base_reasoning__g1_leaf_train_frequency_mean": np.array([0.25], dtype=np.float32),
                    "base_reasoning__g1_leaf_surprisal_mean": np.array([1.2], dtype=np.float32),
                }).to_parquet(artifact / "base_reasoning_features.parquet", index=False)
                pd.DataFrame({
                    "head_name": [head], "side_name": [side], "fold_id": [fold_id],
                    "model_slot": [0], "head_tree_slot": [0], "leaf_token": np.array([101], dtype=np.uint64),
                    "rule_signature": ["same-structural-rule"], "train_leaf_frequency": np.array([0.25], dtype=np.float32),
                    "path_depth": [3], "unique_feature_count": [2],
                }).to_parquet(artifact / "leaf_rule_catalog.parquet", index=False)
                (artifact / "base_reasoning_manifest.json").write_text(json.dumps({
                    "status": "MATERIALIZED_STRICT_OOF", "head_name": head,
                    "side_name": side, "fold_id": fold_id,
                }))
                index_rows.append({
                    "transport": transport, "side_model": side, "head_name": head,
                    "class_index": class_index, "fold_name": fold, "fold_id": fold_id,
                    "artifact_dir": str(relative), "strict_status": "MATERIALIZED_STRICT_OOF",
                    "eval_start_utc": timestamp.isoformat(),
                })
        shard = root / "base_prediction_shards" / transport / side
        shard.mkdir(parents=True)
        pd.DataFrame(shard_rows).to_parquet(shard / "strict_oof_predictions.parquet", index=False)
        pd.DataFrame(outer_rows).to_parquet(shard / "outer_predictions.parquet", index=False)
    index_path = root / "strict_oof_reasoning_artifact_index.parquet"
    pd.DataFrame(index_rows).to_parquet(index_path, index=False)
    return index_path


def test_builds_compact_g2_g3_features_with_causal_prior_outcomes(tmp_path: Path) -> None:
    index = _write_transport(tmp_path / "source")
    result = build_base_reasoning_representation(
        index, tmp_path / "compact",
        config=BaseReasoningRepresentationConfig(batch_rows=1, max_bundle_components=2),
    )
    assert result.row_count == 18
    destination = tmp_path / "compact"
    assert {"contribution_bundle_features_oof.parquet", "base_reasoning_features_oof.parquet", "leaf_rule_signatures.parquet", "base_reasoning_representation_manifest.json"} == {item.name for item in destination.iterdir()}
    features = pd.read_parquet(destination / "base_reasoning_features_oof.parquet")
    assert features["transport"].eq("transport_one").all()
    assert features.contribution_direction.eq("positive").all()
    assert features.filter(like="base_reasoning__").dtypes.eq(np.dtype("float32")).all()
    assert not any(token in name.lower() for name in features for token in ("leaf_token", "leaf_id", "leaf_assignment", "raw_leaf"))
    assert "base_reasoning__g1_leaf_train_frequency_mean" in features
    later = features.loc[features.fold_id.str.endswith("inner_01")]
    assert later["base_reasoning__g2_recurrent_family_prior_outcome_mean"].eq(20.0).all()
    outer = features.loc[features.meta_partition.eq("outer_test")]
    assert len(outer) == 6
    assert outer["base_reasoning__g2_recurrent_family_prior_outcome_mean"].eq(20.0).all()
    bundle = pd.read_parquet(destination / "contribution_bundle_features_oof.parquet")
    assert "base_reasoning__g3_contribution_bundle_weighted_svd_00" in bundle
    signatures = pd.read_parquet(destination / "leaf_rule_signatures.parquet")
    assert "leaf_token" not in signatures
    manifest = json.loads((destination / "base_reasoning_representation_manifest.json").read_text())
    assert manifest["contract"]["no_training_or_evaluation"] is True
    assert manifest["counts"]["rows_by_meta_partition"] == {"inner_oof": 12, "outer_test": 6}


def test_requires_strictly_prior_label_availability(tmp_path: Path) -> None:
    index = _write_transport(tmp_path / "source", second_label_at_feature_time=True)
    build_base_reasoning_representation(index, tmp_path / "compact")
    features = pd.read_parquet(tmp_path / "compact" / "base_reasoning_features_oof.parquet")
    later = features.loc[features.fold_id.str.endswith("inner_01")]
    assert later["base_reasoning__g2_recurrent_family_prior_outcome_mean"].eq(0.0).all()


def test_vectorized_batches_preserve_scalar_g2_g3_contract(tmp_path: Path) -> None:
    """A one-tree fixture has closed-form scalar expectations for every fold."""
    index = _write_transport(tmp_path / "source")
    build_base_reasoning_representation(
        index,
        tmp_path / "compact",
        # One element forces the memory cap to split every row while retaining
        # the exact old causal streaming semantics.
        config=BaseReasoningRepresentationConfig(
            batch_rows=10_000, max_bundle_components=2, max_vectorized_elements=1,
        ),
    )
    features = pd.read_parquet(tmp_path / "compact" / "base_reasoning_features_oof.parquet")
    bundle = pd.read_parquet(tmp_path / "compact" / "contribution_bundle_features_oof.parquet")
    expected = {
        "inner_00": np.float32(0.25 * np.log1p(1)),
        "inner_01": np.float32(0.25 * np.log1p(2)),
        "outer": np.float32(0.25 * np.log1p(3)),
    }
    for suffix, value in expected.items():
        mask = features.fold_id.str.endswith(suffix)
        np.testing.assert_allclose(
            features.loc[mask, "base_reasoning__g2_recurrent_family_weight"].to_numpy(),
            value,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            bundle.loc[mask, "base_reasoning__g3_contribution_bundle_weighted_svd_00"].to_numpy(),
            np.float32(2.0) * value,
            rtol=1e-6,
        )


def test_vectorized_path_has_no_row_tree_dataframe_scalar_lookup() -> None:
    """Guard the optimization structure without a noisy wall-clock assertion."""
    source = inspect.getsource(representation_module.build_base_reasoning_representation)
    assert "assignment.iloc[row][column]" not in source
    assert "_map_assignment_positions(" in source
    assert "_effective_batch_rows(" in source


def test_fails_closed_for_missing_semantic_head(tmp_path: Path) -> None:
    index = _write_transport(tmp_path / "source")
    table = pd.read_parquet(index)
    table = table.loc[~((table.side_model == "short") & (table.fold_name == "inner_01") & (table.head_name == "p_weak"))]
    table.to_parquet(index, index=False)
    with pytest.raises(BaseReasoningRepresentationError, match="partial transport"):
        build_base_reasoning_representation(index, tmp_path / "compact")
