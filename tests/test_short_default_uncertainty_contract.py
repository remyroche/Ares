from __future__ import annotations

import json

import numpy as np

from extreme_price_movements.short_default_uncertainty_contract import (
    apply_or_revert_to_v11,
    feature_schema_hash,
    sha256_file,
    validate_frozen_challenger,
)


def _artifact(tmp_path):
    feature_order = ["a", "b"]
    parent = tmp_path / "parent.txt"
    diagnostic = tmp_path / "diagnostic.parquet"
    neighbor_index = tmp_path / "neighbors.parquet"
    normalization = tmp_path / "normalization_references.npz"
    for path, payload in (
        (parent, b"parent"),
        (diagnostic, b"diagnostic"),
        (neighbor_index, b"neighbors"),
        (normalization, b"normalization"),
    ):
        path.write_bytes(payload)
    manifest = {
        "status": "frozen_research_challenger_not_live",
        "candidate_id": "candidate",
        "candidate": {"threshold": 0.85, "alpha": 0.04},
        "feature_schema": {
            "hash": feature_schema_hash(feature_order),
            "transform_schema": "train_iqr_scale_clip5_v1",
        },
        "provenance_hashes": {
            "feature_schema_hash": feature_schema_hash(feature_order),
            "normalization_array_hash": sha256_file(normalization),
            "neighbor_training_index_hash": sha256_file(neighbor_index),
            "parent_model_hash": sha256_file(parent),
            "diagnostic_source_hash": sha256_file(diagnostic),
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return feature_order, parent, diagnostic, neighbor_index


def test_contract_validation_passes_exact_runtime_inputs(tmp_path) -> None:
    features, parent, diagnostic, neighbor_index = _artifact(tmp_path)
    result = validate_frozen_challenger(
        tmp_path,
        runtime_feature_order=features,
        runtime_transform_schema="train_iqr_scale_clip5_v1",
        parent_model_path=parent,
        diagnostic_source_path=diagnostic,
        neighbor_training_index_path=neighbor_index,
    )
    assert result["valid"] is True
    assert result["action"] == "apply_challenger"


def test_contract_mismatch_reverts_to_parent_v11_rank(tmp_path) -> None:
    features, parent, diagnostic, neighbor_index = _artifact(tmp_path)
    result = validate_frozen_challenger(
        tmp_path,
        runtime_feature_order=["b", "a"],
        runtime_transform_schema="train_iqr_scale_clip5_v1",
        parent_model_path=parent,
        diagnostic_source_path=diagnostic,
        neighbor_training_index_path=neighbor_index,
    )
    rank = np.array([0.95], dtype=np.float32)
    applied, audit = apply_or_revert_to_v11(
        rank, np.array([1.0], dtype=np.float32), np.array([True]), result
    )
    np.testing.assert_allclose(applied, rank)
    assert audit["applied"] is False
    assert "feature_schema_hash_mismatch" in result["failures"]
