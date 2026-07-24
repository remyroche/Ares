from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.run_materialized_trailing_label_topk_lgbm_hpo as base_runner

from extreme_price_movements.features_gmm_ae import (
    AE_GMM_FEATURE_COLUMNS,
    ae_gmm_input_feature_order_hash,
    ae_gmm_learned_transform_hash,
    save_ae_gmm_state_artifact,
)
from extreme_price_movements.lgbm_pipeline import (
    canonical_base_feature_selection_recipe,
    cumulative_positive_mda_keep_count,
)
from scripts.run_materialized_trailing_label_topk_lgbm_hpo import (
    _materialize_frozen_ae_gmm_output_sidecar,
    _rank_top_indices_by_side,
    _resolve_base_model_features,
)


SELECTED_OUTPUTS = [
    "dae_b16_00",
    "gmm_prob_0",
    "gmm_entropy",
    "gmm_cluster_id",
]


def _frozen_state(*, temporal_contract: str = "row_independent_v1") -> dict:
    latent_dim = 16
    state = {
        "enabled": True,
        "schema_version": "ae_gmm_v1",
        "feature_columns": ["f0", "f1"],
        "center": [0.0, 0.0],
        "scale": [1.0, 1.0],
        "cycle_input_fill_values": {"f0": 0.0, "f1": 0.0},
        "ae_state": {"models": {}},
        "gmm_n_components": 2,
        "gmm_covariance_type": "diag",
        "gmm_numeric_contract": "fixed_order_rowwise_v1",
        "gmm_weights": [0.4, 0.6],
        "gmm_means": [
            [0.0 for _ in range(latent_dim)],
            [1.0 for _ in range(latent_dim)],
        ],
        "gmm_covariances": [
            [1.0 for _ in range(latent_dim)],
            [2.0 for _ in range(latent_dim)],
        ],
        "temporal_feature_contract": temporal_contract,
        "smooth_lambda": 0.0,
    }
    state["input_feature_order_hash"] = ae_gmm_input_feature_order_hash(
        state["feature_columns"]
    )
    state["cycle_state_hash"] = ae_gmm_learned_transform_hash(state)
    return state


def _write_labels(labels_path: Path, *, rows: int = 2_307) -> None:
    labels_path.mkdir()
    positions = np.arange(rows, dtype=np.float32)
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC"),
            "__symbol__": np.where(positions.astype(np.int64) % 2, "BBB", "AAA"),
            "side": np.where(positions.astype(np.int64) % 3, 1, -1).astype(np.int8),
            "f0": np.sin(positions / 9.0).astype(np.float32),
            "f1": np.cos(positions / 13.0).astype(np.float32),
        }
    )
    frame.iloc[:1_103].to_parquet(labels_path / "part_000.parquet", index=False)
    frame.iloc[1_103:].to_parquet(labels_path / "part_001.parquet", index=False)


def _persist_state(path: Path, *, temporal_contract: str = "row_independent_v1") -> Path:
    save_ae_gmm_state_artifact(_frozen_state(temporal_contract=temporal_contract), path)
    return path


def _mock_static_store_loader(
    frame: pd.DataFrame,
    *,
    feature_dir: Path,
    selected_features: list[str],
    min_feature_finite_frac: float = 0.50,
) -> tuple[pd.DataFrame, dict]:
    del feature_dir, min_feature_finite_frac
    timestamps = pd.to_datetime(frame["__ts__"], utc=True)
    positions = (
        (timestamps - pd.Timestamp("2025-01-01", tz="UTC"))
        / pd.Timedelta(hours=1)
    ).to_numpy(dtype=np.float32)
    values = {
        "f0": np.sin(positions / 9.0).astype(np.float32),
        "f1": np.cos(positions / 13.0).astype(np.float32),
    }
    result = pd.DataFrame(index=frame.index)
    for feature in selected_features:
        if feature in values:
            result[feature] = values[feature]
    return result, {"reader": "mock_static_feature_store"}


def test_frozen_selected_output_sidecar_is_chunk_invariant_and_bit_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels_path = tmp_path / "labels"
    _write_labels(labels_path)
    state_path = _persist_state(tmp_path / "state.pkl")
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    monkeypatch.setattr(
        base_runner, "_load_feature_store_columns", _mock_static_store_loader
    )

    first_path, first_contract = _materialize_frozen_ae_gmm_output_sidecar(
        labels_path=labels_path,
        feature_dir=feature_dir,
        state_path=state_path,
        output_path=tmp_path / "outputs" / "chunk_1000.parquet",
        output_features=SELECTED_OUTPUTS,
        chunk_rows=1_000,
    )
    second_path, second_contract = _materialize_frozen_ae_gmm_output_sidecar(
        labels_path=labels_path,
        feature_dir=feature_dir,
        state_path=state_path,
        output_path=tmp_path / "outputs" / "chunk_1151.parquet",
        output_features=SELECTED_OUTPUTS,
        chunk_rows=1_151,
    )

    first = pd.read_parquet(first_path)
    second = pd.read_parquet(second_path)
    pd.testing.assert_frame_equal(first, second, check_dtype=True, check_exact=True)
    assert list(first.columns) == ["__ts__", "__symbol__", "side", *SELECTED_OUTPUTS]
    assert first_contract["rows"] == len(first) == 2_307
    assert second_contract["rows"] == len(second) == 2_307
    assert first_contract["temporal_feature_contract"] == "row_independent_v1"
    assert second_contract["chunk_rows"] == 1_151


def test_frozen_selected_output_sidecar_rejects_non_row_independent_state(
    tmp_path: Path,
) -> None:
    state_path = _persist_state(
        tmp_path / "ordered_state.pkl",
        temporal_contract="ordered_batch_sequence_v1",
    )

    with pytest.raises(RuntimeError, match="requires row_independent_v1"):
        _materialize_frozen_ae_gmm_output_sidecar(
            labels_path=tmp_path / "labels",
            feature_dir=tmp_path / "features",
            state_path=state_path,
            output_path=tmp_path / "outputs" / "forbidden.parquet",
            output_features=SELECTED_OUTPUTS,
            chunk_rows=1_000,
        )

    assert not (tmp_path / "outputs" / "forbidden.parquet").exists()


def test_frozen_sidecar_materializes_complete_meta_context_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels_path = tmp_path / "labels"
    _write_labels(labels_path, rows=32)
    state_path = _persist_state(tmp_path / "state.pkl")
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    monkeypatch.setattr(
        base_runner, "_load_feature_store_columns", _mock_static_store_loader
    )

    output_path, contract = _materialize_frozen_ae_gmm_output_sidecar(
        labels_path=labels_path,
        feature_dir=feature_dir,
        state_path=state_path,
        output_path=tmp_path / "outputs" / "complete.parquet",
        output_features=AE_GMM_FEATURE_COLUMNS,
        chunk_rows=1_000,
    )

    assert set(AE_GMM_FEATURE_COLUMNS).issubset(pd.read_parquet(output_path).columns)
    assert contract["output_feature_count"] == len(AE_GMM_FEATURE_COLUMNS)


def test_canonical_mda_selection_ceiling_caps_positive_prefix_at_150() -> None:
    recipe = canonical_base_feature_selection_recipe()
    ceiling = int(recipe["maximum_feature_count"])

    keep_n, status, _floor = cumulative_positive_mda_keep_count(
        np.linspace(1.0, 0.01, 300),
        cumulative_fraction=float(recipe["cumulative_positive_importance_fraction"]),
        maximum_feature_count=ceiling,
    )

    assert ceiling == 150
    assert keep_n == ceiling
    assert status.endswith("_capped_150")


def test_base_to_meta_top30_is_ranked_independently_per_side() -> None:
    scores = np.concatenate(
        [np.linspace(0.0, 0.4, 10), np.linspace(0.6, 1.0, 10)]
    )
    sides = np.concatenate([np.full(10, -1), np.full(10, 1)])

    selected = _rank_top_indices_by_side(scores, sides, 0.30)
    selected_sides = sides[selected]

    assert len(selected) == 6
    assert int(np.sum(selected_sides == -1)) == 3
    assert int(np.sum(selected_sides == 1)) == 3
    assert set(selected.tolist()) == {7, 8, 9, 17, 18, 19}


def test_complete_ae_gmm_context_does_not_widen_fixed_base_contract() -> None:
    frame = pd.DataFrame(
        {
            "raw_selected": [1.0],
            "gmm_prob_0": [0.4],
            "gmm_prob_1": [0.6],
            "gmm_entropy": [0.7],
        }
    )

    features = _resolve_base_model_features(
        frame,
        ["raw_selected", "gmm_prob_0"],
    )

    assert features == ["raw_selected", "gmm_prob_0"]
