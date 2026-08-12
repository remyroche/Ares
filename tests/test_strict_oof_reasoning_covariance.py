from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.strict_oof_reasoning_covariance import (
    StrictOOFReasoningCovarianceConfig,
    StrictOOFReasoningCovarianceError,
    analyze_strict_oof_reasoning_covariance,
    discover_strict_oof_reasoning_artifacts,
    write_strict_oof_reasoning_covariance,
)


def _artifact(root: Path, *, head: str, class_index: int, fold: str = "fold_a", side: str = "long") -> Path:
    path = root / side / fold / head
    path.mkdir(parents=True)
    identity = pd.DataFrame(
        {
            "candidate_id": [f"c{index}" for index in range(8)],
            "__ts__": pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC"),
            "side_name": side,
        }
    )
    features = identity.assign(
        head_name=head,
        fold_id=fold,
        base_reasoning__g1_leaf_surprisal_mean=np.linspace(0.1, 0.8, 8),
        base_reasoning__g2_path_depth_mean=np.array([1, 2, 1, 3, 2, 4, 3, 4], dtype=float),
        base_reasoning__g3_balance=np.linspace(-0.3, 0.4, 8),
        # Fold-local coordinate: explicitly excluded, even though it varies.
        base_reasoning__g3_contribution_svd_00=np.arange(8, dtype=float),
    )
    predictions = identity.assign(
        base_prediction=np.linspace(0.2, 0.9, 8),
        head_name=head,
        class_index=class_index,
        fold_id=fold,
    )
    labels = identity.assign(
        label__r3_class=[0, 1, 2, 0, 1, 2, 1, 2],
        label__net_bps=np.linspace(-80.0, 80.0, 8),
        head_name=head,
        fold_id=fold,
    )
    features.to_parquet(path / "base_reasoning_features.parquet", index=False)
    predictions.to_parquet(path / "base_reasoning_predictions.parquet", index=False)
    labels.to_parquet(path / "base_reasoning_labels.parquet", index=False)
    (path / "base_reasoning_manifest.json").write_text(
        json.dumps(
            {
                "status": "MATERIALIZED_STRICT_OOF",
                "side_name": side,
                "fold_id": fold,
                "head_name": head,
                "provenance": {"feature_contract_sha256": "contract-a", "class_index": class_index},
            }
        )
    )
    return path


def test_combines_all_scalar_bundles_with_head_prefixes_and_excludes_svd(tmp_path: Path) -> None:
    root = tmp_path / "strict_oof_base_reasoning"
    _artifact(root, head="p_adverse", class_index=0)
    _artifact(root, head="p_clear", class_index=2)

    result = analyze_strict_oof_reasoning_covariance(
        [root], config=StrictOOFReasoningCovarianceConfig(max_pairwise_features=6, min_pairwise_rows=4)
    )

    assert result.artifact_count == 2
    assert result.fold_count == 1
    assert set(result.feature_summary.feature_bundle) == {"g1", "g2", "g3"}
    assert {"p_adverse", "p_clear"} == set(result.feature_summary.feature_head)
    assert result.feature_summary.feature_name.str.startswith("base_reasoning__p_").all()
    assert not result.feature_summary.feature_name.str.contains("svd").any()
    assert len(result.pairwise_feature_selection.loc[result.pairwise_feature_selection.selected_for_pairwise]) == 6
    assert not any("leaf_token" in column or "raw_leaf" in column for column in result.pairwise_correlation.columns)
    assert result.association_summary.diagnostic_only.all()
    own_targets = set(result.association_summary.diagnostic_target)
    assert {"base_prediction__p_adverse", "base_prediction__p_clear", "net_bps", "semantic_label__p_adverse", "semantic_label__p_clear"}.issubset(own_targets)
    assert result.bundle_summary.groupby(["feature_head", "feature_bundle"], observed=True).size().eq(1).all()

    destination = write_strict_oof_reasoning_covariance(result, tmp_path / "output")
    assert (destination / "pairwise_correlation.parquet").exists()
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["contracts"]["g3"].startswith("fold-local")
    assert manifest["contracts"]["raw_leaf_tokens"].startswith("never")


def test_rejects_head_identity_mismatch_and_manifest_status(tmp_path: Path) -> None:
    root = tmp_path / "strict_oof_base_reasoning"
    path = _artifact(root, head="p_clear", class_index=2)
    features = pd.read_parquet(path / "base_reasoning_features.parquet")
    features.loc[0, "head_name"] = "wrong_head"
    features.to_parquet(path / "base_reasoning_features.parquet", index=False)
    with pytest.raises(StrictOOFReasoningCovarianceError, match="head mismatches manifest"):
        analyze_strict_oof_reasoning_covariance([path], config=StrictOOFReasoningCovarianceConfig(min_pairwise_rows=4))

    _artifact(root, head="p_adverse", class_index=0)
    manifest_path = root / "long" / "fold_a" / "p_adverse" / "base_reasoning_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "PARTIAL"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(StrictOOFReasoningCovarianceError, match="not materialised strict OOF"):
        analyze_strict_oof_reasoning_covariance([root], config=StrictOOFReasoningCovarianceConfig(min_pairwise_rows=4))


def test_discovery_accepts_an_artifact_manifest(tmp_path: Path) -> None:
    path = _artifact(tmp_path, head="p_clear", class_index=2)
    assert discover_strict_oof_reasoning_artifacts([path / "base_reasoning_manifest.json"]) == [path.resolve()]
