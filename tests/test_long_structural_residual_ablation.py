from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_long_structural_residual_ablation import (
    _COMPATIBILITY_HEALTH,
    _HISTORICAL_HEALTH,
    _PORTABILITY_HEALTH,
    _purged_meta_train,
    _target_arms,
    _validate_fold_partitions,
    feature_arms,
)


def _feature_frame() -> pd.DataFrame:
    values = {
        "candidate_id": ["a", "b"], "__ts__": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
        "label_available_ts": pd.to_datetime(["2024-01-01 12:00", "2024-01-02 12:00"], utc=True),
        "side_name": ["long", "long"], "gross_bps": [120.0, -20.0], "net_bps": [20.0, -120.0],
        "label_valid": [True, True], "barrier_relevance_0_5": [4, 1], "mfe_mae_label_valid": [True, True],
        "atr_bps": [100.0, 100.0], "query_id": ["a", "b"], "month": ["2024-01", "2024-01"],
        "meta_partition": ["meta_train", "meta_train"], "fold": ["x", "x"],
        "feature_contract_sha256": ["x", "x"], "base_raw_score": [0.1, 0.2], "base_expected_bps": [0.0, 0.0],
        "raw_feature": [0.3, 0.4], "aegmm_dae_b16_00": [0.2, 0.5],
        "base_reasoning__g1_leaf_surprisal_mean": [0.2, 0.3],
        "base_structural_family__f": [0.7, 0.5],
    }
    for name in (*_HISTORICAL_HEALTH, *_PORTABILITY_HEALTH, *_COMPATIBILITY_HEALTH):
        values[name] = [0.1, 0.2]
    for number in range(30):
        values[f"raw_feature_{number:02d}"] = [float(number), float(number + 1)]
    return pd.DataFrame(values)


def test_feature_arms_are_nested_and_keep_only_token_free_tree_representations():
    arms = feature_arms(_feature_frame())
    assert "base_reasoning__g1_leaf_surprisal_mean" not in arms["R0_raw_aegmm_base"]
    assert "base_reasoning__g1_leaf_surprisal_mean" in arms["R1_reasoning_memberships"]
    assert "base_structural_family__f" in arms["R1_reasoning_memberships"]
    assert all("support_h12" not in field for fields in arms.values() for field in fields)
    assert set(arms["R0_raw_aegmm_base"]).issubset(arms["R4_compatibility_health"])


def test_target_arms_are_direct_residual_grades_without_support_filtering():
    targets = _target_arms(_feature_frame())
    assert set(targets) == {"hybrid_50_150", "bps_50_150"}
    assert all(values.tolist() == [2, 1] for values in targets.values())


def test_partition_validation_and_horizon_purge_are_separate():
    frame = pd.concat([
        _feature_frame().assign(meta_partition="meta_train", __ts__=pd.Timestamp("2024-01-01", tz="UTC"), label_available_ts=pd.Timestamp("2024-01-01 12:00", tz="UTC")),
        _feature_frame().assign(meta_partition="meta_calibration", __ts__=pd.Timestamp("2024-01-02", tz="UTC"), label_available_ts=pd.Timestamp("2024-01-02 12:00", tz="UTC")),
        _feature_frame().assign(meta_partition="test", __ts__=pd.Timestamp("2024-01-03", tz="UTC"), label_available_ts=pd.Timestamp("2024-01-03 12:00", tz="UTC")),
    ], ignore_index=True)
    _validate_fold_partitions(frame, "x")
    assert len(_purged_meta_train(
        frame.loc[frame.meta_partition.eq("meta_train")],
        frame.loc[frame.meta_partition.eq("meta_calibration")], "x",
    )) == 2
    contaminated = frame.copy()
    contaminated.loc[contaminated.meta_partition.eq("meta_train"), "label_available_ts"] = pd.Timestamp("2024-01-02", tz="UTC")
    _validate_fold_partitions(contaminated, "x")
    with pytest.raises(ValueError, match="purge"):
        _purged_meta_train(
            contaminated.loc[contaminated.meta_partition.eq("meta_train")],
            contaminated.loc[contaminated.meta_partition.eq("meta_calibration")], "x",
        )
