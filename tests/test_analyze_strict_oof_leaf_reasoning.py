from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_strict_oof_leaf_reasoning import (
    TOP_RULES_PER_CELL,
    analyze_strict_oof_leaf_reasoning,
    discover_reasoning_artifacts,
    write_analysis,
)


def _artifact(root: Path, *, fold: str, tokens: tuple[int, int, int, int], signature: str) -> Path:
    path = root / fold / "p_clear"
    path.mkdir(parents=True)
    timestamp = pd.date_range("2024-01-01", periods=6, freq="2D", tz="UTC")
    identity = pd.DataFrame({"candidate_id": [f"{fold}-{index}" for index in range(6)], "__ts__": timestamp, "side_name": "long"})
    assignments = identity.assign(
        leaf_assignment__model_00_head_tree_000=np.array([tokens[0], tokens[0], tokens[0], tokens[1], tokens[1], tokens[1]], dtype="uint64"),
        leaf_assignment__model_00_head_tree_001=np.array([tokens[2], tokens[2], tokens[3], tokens[3], tokens[3], tokens[2]], dtype="uint64"),
    )
    predictions = identity.assign(base_model_prediction=[.1, .2, .8, .9, .8, .7], head_name="p_clear", class_index=2, fold_id=fold)
    labels = identity.assign(label__r3_class=[0, 1, 2, 2, 2, 1], label__net_bps=[-30., -10., 20., 30., 40., 5.], head_name="p_clear", fold_id=fold)
    features = identity.assign(
        head_name="p_clear", fold_id=fold,
        base_reasoning__g3_balance=[.2, .2, .2, .3, .3, .3],
        base_reasoning__g3_contrib_entropy=[.4] * 6,
        base_reasoning__g3_top1_abs_share=[.6] * 6,
        base_reasoning__g3_top3_abs_share=[.9] * 6,
        # Must never be consumed by the cross-fold scalar summary.
        base_reasoning__g3_contribution_svd_00=np.arange(6, dtype="float32"),
    )
    catalog = pd.DataFrame(
        {
            "head_name": ["p_clear"] * 4,
            "side_name": ["long"] * 4,
            "fold_id": [fold] * 4,
            "model_slot": [0, 0, 0, 0],
            "tree_index": [0, 0, 1, 1],
            "leaf_token": np.array(tokens, dtype="uint64"),
            "rule_signature": [signature, f"{signature}-other", signature, f"{signature}-other-2"],
            "rule_structural_path_json": [json.dumps([{"feature": "trend", "decision_type": "<=", "threshold_band_index": 2}]), json.dumps([{"feature": "flow", "decision_type": "<=", "threshold_band_index": 3}]), json.dumps([{"feature": "trend", "decision_type": "<=", "threshold_band_index": 2}]), json.dumps([{"feature": "risk", "decision_type": "<=", "threshold_band_index": 4}])],
        }
    )
    assignments.to_parquet(path / "leaf_assignments.parquet", index=False)
    predictions.to_parquet(path / "base_reasoning_predictions.parquet", index=False)
    labels.to_parquet(path / "base_reasoning_labels.parquet", index=False)
    features.to_parquet(path / "base_reasoning_features.parquet", index=False)
    catalog.to_parquet(path / "leaf_rule_catalog.parquet", index=False)
    (path / "base_reasoning_manifest.json").write_text(json.dumps({"status": "MATERIALIZED_STRICT_OOF", "provenance": {"feature_contract_sha256": "same-contract"}}))
    return path


def test_analyzer_streams_tree_columns_and_drops_raw_tokens_before_cross_fold_work(tmp_path: Path) -> None:
    root = tmp_path / "strict_oof_base_reasoning"
    _artifact(root, fold="fold_a", tokens=(11, 12, 13, 14), signature="recurrent")
    _artifact(root, fold="fold_b", tokens=(21, 22, 23, 24), signature="recurrent")
    discovered = discover_reasoning_artifacts([root])
    assert len(discovered) == 2

    result = analyze_strict_oof_leaf_reasoning([root])
    assert result.artifact_count == 2
    assert "leaf_token" in result.leaf_health
    assert not any("leaf_token" in name or "svd" in name for name in result.rule_instance_summary.columns)
    assert result.leaf_health["active_label_mean"].between(0.0, 1.0).all()
    # Same G2 signature appears in both fold-local catalogs, producing an
    # explicit cross-fold recurrence measurement rather than token matching.
    recurrent = result.signature_recurrence.loc[result.signature_recurrence.rule_signature.eq("recurrent")].iloc[0]
    assert recurrent["recurring_fold_count"] == 2
    assert recurrent["fold_recurrence_fraction"] == 1.0
    assert result.rule_prefilter_audit.groupby(["side_name", "head_name", "contribution_direction"], observed=True).size().le(TOP_RULES_PER_CELL).all()
    assert set(result.cluster_overview[["threshold", "linkage"]].itertuples(index=False, name=None)) == {
        (threshold, linkage) for threshold in (.60, .70, .80, .90) for linkage in ("average", "complete")
    }

    destination = write_analysis(result, tmp_path / "output")
    assert (destination / "leaf_health.parquet").exists()
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["contracts"]["g3"].startswith("balance")
    assert "axes" in manifest["contracts"]["g3"]
