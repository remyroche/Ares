from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import extreme_price_movements.lgbm_pipeline as lgbm_pipeline
from extreme_price_movements.features_denoising_ae import _forward

from extreme_price_movements.features_gmm_ae import (
    AE_GMM_CLUSTER_CANDIDATES,
    AE_GMM_CYCLE_CONTRACT_VERSION,
    AE_GMM_REG_COVAR_CANDIDATES,
    AE_GMM_TRANSFORM_HASH_V2,
    _diag_gmm_stats,
    _time_spread_sample_indices,
    ae_gmm_cycle_reference_indices,
    ae_gmm_cycle_sample_identity_hash,
    ae_gmm_input_feature_order_hash,
    ae_gmm_learned_transform_hash,
    ae_gmm_state_manifest,
    load_ae_gmm_state_artifact,
    save_ae_gmm_state_artifact,
    transform_ae_gmm_features,
)
from extreme_price_movements.lgbm_pipeline import (
    _ae_gmm_cycle_state_path,
    _cross_val_oof_lgbm_with_meta_features,
    _fit_or_load_ae_gmm_cycle_state_for_selection,
    _load_ae_gmm_cycle_state,
)


def _minimal_enabled_state() -> dict:
    latent_dim = 16
    state = {
        "enabled": True,
        "schema_version": "ae_gmm_v1",
        "feature_columns": ["f0", "f1"],
        "center": [0.0, 1.0],
        "scale": [1.0, 2.0],
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
        "cycle_contract_version": "single_fit_begin_middle_end_v1",
    }
    state["input_feature_order_hash"] = ae_gmm_input_feature_order_hash(
        state["feature_columns"]
    )
    return state


def test_default_component_grid_covers_three_through_eight() -> None:
    assert AE_GMM_CLUSTER_CANDIDATES == (3, 4, 5, 6, 7, 8)
    assert AE_GMM_REG_COVAR_CANDIDATES == (5e-4, 1e-3, 2e-3, 3e-3)


def test_time_spread_sampler_covers_beginning_middle_and_end() -> None:
    idx = _time_spread_sample_indices(300, 30)
    assert len(idx) == 30
    assert int(np.sum(idx < 100)) == 10
    assert int(np.sum((idx >= 100) & (idx < 200))) == 10
    assert int(np.sum(idx >= 200)) == 10
    assert idx[0] == 0
    assert idx[-1] == 299


def test_cycle_state_path_is_shared_across_model_layers(tmp_path: Path) -> None:
    cfg = {"data_root": str(tmp_path), "run_id": "cycle_001"}
    expected = tmp_path / "artifacts" / "cycle_001" / "ae_gmm_cycle" / "state.pkl"
    assert _ae_gmm_cycle_state_path(cfg) == expected

    state = _minimal_enabled_state()
    state["cycle_state_hash"] = ae_gmm_learned_transform_hash(state)
    save_ae_gmm_state_artifact(state, expected)

    base_loaded = _load_ae_gmm_cycle_state(cfg)
    meta_loaded = _load_ae_gmm_cycle_state(cfg)
    assert base_loaded["cycle_state_hash"] == meta_loaded["cycle_state_hash"]
    assert base_loaded["feature_columns"] == meta_loaded["feature_columns"]
    assert ae_gmm_learned_transform_hash(base_loaded) == ae_gmm_learned_transform_hash(
        meta_loaded
    )


def test_explicit_cycle_state_path_overrides_run_default(tmp_path: Path) -> None:
    explicit = tmp_path / "frozen" / "state.pkl"
    cfg = {
        "data_root": str(tmp_path),
        "run_id": "cycle_001",
        "lgbm_ae_gmm_cycle_state_path": str(explicit),
    }
    assert _ae_gmm_cycle_state_path(cfg) == explicit


def test_cycle_state_path_prefers_output_run_over_source_run(tmp_path: Path) -> None:
    cfg = {
        "data_root": str(tmp_path),
        "output_run_id": "new_training_cycle",
        "run_id": "feature_source",
        "_label_artifact_run_id": "label_source",
    }
    assert _ae_gmm_cycle_state_path(cfg) == (
        tmp_path
        / "artifacts"
        / "new_training_cycle"
        / "ae_gmm_cycle"
        / "state.pkl"
    )


def test_cycle_state_rejects_reordered_input_contract(tmp_path: Path) -> None:
    path = tmp_path / "state.pkl"
    state = _minimal_enabled_state()
    state["cycle_state_hash"] = ae_gmm_learned_transform_hash(state)
    state["feature_columns"] = list(reversed(state["feature_columns"]))
    with pytest.raises(ValueError, match="stale input order hash"):
        save_ae_gmm_state_artifact(state, path)


def test_cycle_transform_rejects_missing_input_column() -> None:
    state = _minimal_enabled_state()
    state["cycle_state_hash"] = ae_gmm_learned_transform_hash(state)
    try:
        transform_ae_gmm_features(
            {"f0": np.zeros(3, dtype=np.float32)},
            state,
        )
    except ValueError as exc:
        assert "missing 1/2 columns" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing AE/GMM cycle input was accepted")


def test_row_independent_cycle_transform_is_batch_order_invariant() -> None:
    state = _minimal_enabled_state()
    state["temporal_feature_contract"] = "row_independent_v1"
    state["smooth_lambda"] = 0.0
    state["cycle_state_hash"] = ae_gmm_learned_transform_hash(state)
    x = pd.DataFrame(
        {"f0": [0.0, 0.5, 1.0], "f1": [1.0, 1.5, 2.0]}, dtype=np.float32
    )
    direct = transform_ae_gmm_features(x, state).reset_index(drop=True)
    order = np.asarray([2, 0, 1], dtype=np.int64)
    inverse = np.argsort(order)
    permuted = transform_ae_gmm_features(
        x.iloc[order].reset_index(drop=True), state
    ).iloc[inverse].reset_index(drop=True)
    np.testing.assert_array_equal(direct.to_numpy(), permuted.to_numpy())


def test_cycle_reference_sampling_is_source_order_invariant() -> None:
    identities = pd.DataFrame(
        [
            (timestamp, symbol, side)
            for timestamp in pd.date_range(
                "2025-01-01", periods=30, freq="h", tz="UTC"
            )
            for symbol in ("AAA", "BBB", "CCC")
            for side in ("long", "short")
        ],
        columns=["timestamp", "symbol", "side"],
    )
    first_idx = ae_gmm_cycle_reference_indices(
        identities["timestamp"],
        symbols=identities["symbol"],
        sides=identities["side"],
        max_rows=60,
    )
    first_rows = identities.iloc[first_idx].reset_index(drop=True)
    first_hash = ae_gmm_cycle_sample_identity_hash(
        first_rows["timestamp"],
        symbols=first_rows["symbol"],
        sides=first_rows["side"],
    )

    shuffled = identities.sample(frac=1.0, random_state=91).reset_index(drop=True)
    second_idx = ae_gmm_cycle_reference_indices(
        shuffled["timestamp"],
        symbols=shuffled["symbol"],
        sides=shuffled["side"],
        max_rows=60,
    )
    second_rows = shuffled.iloc[second_idx].reset_index(drop=True)
    second_hash = ae_gmm_cycle_sample_identity_hash(
        second_rows["timestamp"],
        symbols=second_rows["symbol"],
        sides=second_rows["side"],
    )

    pd.testing.assert_frame_equal(first_rows, second_rows)
    assert first_hash == second_hash
    assert set(first_rows["side"]) == {"long", "short"}
    assert first_rows["symbol"].nunique() == 3


def test_v2_transform_hash_covers_fill_clip_and_smoothing() -> None:
    state = _minimal_enabled_state()
    state.update(
        {
            "learned_transform_hash_version": AE_GMM_TRANSFORM_HASH_V2,
            "cycle_input_fill_values": {"f0": 0.0, "f1": 1.0},
            "clip": [-8.0, 8.0],
            "smooth_lambda": 0.0,
            "temporal_feature_contract": "row_independent_v1",
        }
    )
    baseline = ae_gmm_learned_transform_hash(state)
    state["cycle_input_fill_values"]["f0"] = 0.25
    assert ae_gmm_learned_transform_hash(state) != baseline
    state["cycle_input_fill_values"]["f0"] = 0.0
    state["clip"] = [-6.0, 6.0]
    assert ae_gmm_learned_transform_hash(state) != baseline


def test_cycle_v2_persistence_rejects_outcome_assisted_or_incomplete_state(
    tmp_path: Path,
) -> None:
    state = _minimal_enabled_state()
    state.update(
        {
            "cycle_contract_version": AE_GMM_CYCLE_CONTRACT_VERSION,
            "learned_transform_hash_version": AE_GMM_TRANSFORM_HASH_V2,
            "cycle_input_fill_values": {"f0": 0.0, "f1": 1.0},
            "temporal_feature_contract": "row_independent_v1",
            "smooth_lambda": 0.0,
            "representation_selection_outcome_free": False,
            "representation_selection_outcome_keys": ["returns"],
            "cycle_reference_ordering": "timestamp_utc,symbol,side",
            "cycle_reference_sample_identity_hash": "abc",
            "cycle_reference_rows_available": 300,
            "cycle_reference_rows_sampled": 200,
            "cycle_reference_symbol_count": 3,
            "cycle_reference_sampled_symbol_count": 3,
            "cycle_reference_side_counts": {"long": 100, "short": 100},
        }
    )
    with pytest.raises(ValueError, match="outcome_free must be true"):
        save_ae_gmm_state_artifact(state, tmp_path / "state.pkl")


def test_cycle_manifest_persists_reference_and_transform_contract() -> None:
    state = _minimal_enabled_state()
    state.update(
        {
            "temporal_feature_contract": "row_independent_v1",
            "cycle_reference_fold": "largest_train_fold",
            "cycle_reference_start": "2025-01-01T00:00:00+00:00",
            "cycle_reference_end": "2026-06-30T23:00:00+00:00",
            "cycle_reference_rows_available": 1_000_000,
            "cycle_reference_rows_sampled": 100_000,
            "cycle_reference_sample_policy": "beginning_middle_end_time_spread",
            "cycle_input_fill_values": {"f0": 0.25, "f1": 1.5},
        }
    )
    state["cycle_state_hash"] = ae_gmm_learned_transform_hash(state)

    manifest = ae_gmm_state_manifest(state)

    assert manifest["cycle_contract_version"] == "single_fit_begin_middle_end_v1"
    assert manifest["cycle_state_hash"] == state["cycle_state_hash"]
    assert manifest["cycle_reference_rows_available"] == 1_000_000
    assert manifest["cycle_reference_rows_sampled"] == 100_000
    assert manifest["cycle_reference_sample_policy"] == (
        "beginning_middle_end_time_spread"
    )
    assert manifest["temporal_feature_contract"] == "row_independent_v1"
    assert manifest["cycle_input_fill_values"] == {"f0": 0.25, "f1": 1.5}
    assert "persisted cycle_input_fill_values" in manifest["transform_rules"][
        "missing_input_policy"
    ]
    assert manifest["transform_rules"]["row_independent_temporal_outputs"] == (
        "zero-filled until a causal per-symbol history state is supplied"
    )


def test_final_oof_rejects_missing_cycle_state_before_fold_fit() -> None:
    n = 800
    x = pd.DataFrame(
        {
            "f0": np.linspace(-1.0, 1.0, n, dtype=np.float32),
            "f1": np.linspace(1.0, -1.0, n, dtype=np.float32),
        }
    )
    y = (np.arange(n) % 2).astype(np.float32)
    with pytest.raises(RuntimeError, match="fold-local AE/GMM fitting is disabled"):
        _cross_val_oof_lgbm_with_meta_features(
            x,
            y,
            np.ones(n, dtype=np.float32),
            ["f0", "f1", "gmm_entropy"],
            classifier=True,
            params={},
            random_state=7,
            n_splits=2,
            timestamps=pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
            ae_gmm_input_features=["f0", "f1"],
            ae_gmm_feature_names=["gmm_entropy"],
            ae_gmm_enabled=True,
            fixed_ae_gmm_state=None,
        )


def test_feature_selection_fits_cycle_state_once_then_reuses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit_calls = 0

    def _fake_fit(*args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        assert kwargs.get("outcome_free") is True
        assert kwargs.get("y_metric") is None
        assert kwargs.get("returns") is None
        assert kwargs.get("label_context") is None
        state = _minimal_enabled_state()
        state["temporal_feature_contract"] = "row_independent_v1"
        state["smooth_lambda"] = 0.0
        state["cycle_input_fill_values"] = {"f0": 0.0, "f1": 1.0}
        return state

    monkeypatch.setattr(lgbm_pipeline, "_fit_ae_gmm_post_selection_state", _fake_fit)
    x = pd.DataFrame(
        {
            "f0": np.linspace(-1.0, 1.0, 300, dtype=np.float32),
            "f1": np.linspace(1.0, -1.0, 300, dtype=np.float32),
        }
    )
    y = (np.arange(len(x)) % 2).astype(np.float32)
    assets = np.asarray([f"SYM_{i % 5}" for i in range(len(x))], dtype=object)
    cfg = {
        "data_root": str(tmp_path),
        "output_run_id": "cycle_001",
        "lgbm_ae_gmm_features_enabled": True,
    }
    first, first_state, first_diag = _fit_or_load_ae_gmm_cycle_state_for_selection(
        x,
        y_metric=y,
        returns=y,
        label_context=None,
        timestamps=pd.date_range("2025-01-01", periods=len(x), freq="h", tz="UTC"),
        assets=assets,
        random_state=11,
        cfg=cfg,
    )
    second, second_state, second_diag = _fit_or_load_ae_gmm_cycle_state_for_selection(
        x,
        y_metric=y,
        returns=y,
        label_context=None,
        timestamps=pd.date_range("2025-01-01", periods=len(x), freq="h", tz="UTC"),
        assets=assets,
        random_state=99,
        cfg=cfg,
    )

    assert fit_calls == 1
    assert first_diag["state_source"] == "fit_once_for_cycle"
    assert second_diag["state_source"] == "loaded_cycle_state"
    assert first_diag["cycle_state_hash"] == second_diag["cycle_state_hash"]
    assert first_state["cycle_state_hash"] == second_state["cycle_state_hash"]
    assert _ae_gmm_cycle_state_path(cfg).is_file()
    np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())


def test_row_independent_contract_neutralizes_all_sequence_outputs() -> None:
    state = _minimal_enabled_state()
    state["temporal_feature_contract"] = "row_independent_v1"
    state["smooth_lambda"] = 0.0
    x = pd.DataFrame(
        {"f0": [0.0, 0.5, 1.0], "f1": [1.0, 1.5, 2.0]}, dtype=np.float32
    )

    transformed = transform_ae_gmm_features(x, state)
    temporal_columns = [
        "gmm_posterior_delta_1",
        "gmm_posterior_accel_1",
        "cluster_entropy_delta_1",
        "cluster_entropy_accel_1",
        "min_mahalanobis_delta_1",
        "expected_mahalanobis_delta_1",
        "expected_mahalanobis_accel_1",
        "cluster_speed",
        "cluster_acceleration",
        "time_since_cluster_change",
        "rolling_cluster_stability",
        "cluster_flip_count_20",
        "dae_reconstruction_error_delta_1",
        "dae_reconstruction_error_accel_1",
        "latent_speed",
        "latent_acceleration",
    ]

    np.testing.assert_array_equal(
        transformed[temporal_columns].to_numpy(),
        np.zeros((len(transformed), len(temporal_columns)), dtype=np.float32),
    )


def test_cycle_transform_reindexes_present_inputs_to_frozen_order() -> None:
    state = _minimal_enabled_state()
    state["temporal_feature_contract"] = "row_independent_v1"
    state["smooth_lambda"] = 0.0
    x = pd.DataFrame(
        {"f0": [0.0, 0.5, 1.0], "f1": [1.0, 1.5, 2.0]}, dtype=np.float32
    )

    expected = transform_ae_gmm_features(x, state)
    reordered = transform_ae_gmm_features(x[["f1", "f0"]], state)

    np.testing.assert_array_equal(expected.to_numpy(), reordered.to_numpy())


@pytest.mark.parametrize("identity_name", ["symbols", "sides"])
def test_cycle_identity_contract_rejects_misaligned_vectors(
    identity_name: str,
) -> None:
    timestamps = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
    kwargs = {identity_name: ["x", "y"]}

    with pytest.raises(ValueError, match="identities must match timestamp rows"):
        ae_gmm_cycle_reference_indices(timestamps, max_rows=4, **kwargs)
    with pytest.raises(ValueError, match="identities must match timestamp rows"):
        ae_gmm_cycle_sample_identity_hash(timestamps, **kwargs)


def test_cycle_v2_rejects_forbidden_context_and_incomplete_fill(
    tmp_path: Path,
) -> None:
    state = _minimal_enabled_state()
    state.update(
        {
            "cycle_contract_version": AE_GMM_CYCLE_CONTRACT_VERSION,
            "learned_transform_hash_version": AE_GMM_TRANSFORM_HASH_V2,
            "cycle_input_fill_values": {"f0": 0.0},
            "temporal_feature_contract": "row_independent_v1",
            "smooth_lambda": 0.0,
            "representation_selection_outcome_free": True,
            "representation_selection_context_keys": ["side", "returns"],
            "representation_selection_outcome_keys": [],
            "cycle_reference_ordering": "timestamp_utc,symbol,side",
            "cycle_reference_sample_identity_hash": "abc",
            "cycle_reference_rows_available": 300,
            "cycle_reference_rows_sampled": 200,
            "cycle_reference_symbol_count": 3,
            "cycle_reference_sampled_symbol_count": 3,
            "cycle_reference_side_counts": {"long": 100, "short": 100},
        }
    )

    with pytest.raises(ValueError, match="forbidden keys.*returns"):
        save_ae_gmm_state_artifact(state, tmp_path / "state.pkl")

    state["representation_selection_context_keys"] = ["side"]
    with pytest.raises(ValueError, match="missing columns.*f1"):
        save_ae_gmm_state_artifact(state, tmp_path / "state.pkl")


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_cycle_transform_rejects_nonfinite_required_candidate_rows(
    bad_value: float,
) -> None:
    state = _minimal_enabled_state()
    x = pd.DataFrame(
        {"f0": [0.0, bad_value], "f1": [1.0, 2.0]}, dtype=np.float32
    )

    with pytest.raises(
        ValueError,
        match="nonfinite_candidate_rows=1/2.*nonfinite_values=1",
    ):
        transform_ae_gmm_features(x, state)


def test_diag_gmm_stats_are_exact_across_batch_partitions() -> None:
    rng = np.random.default_rng(519)
    state = {
        "gmm_means": rng.normal(size=(5, 16)).astype(np.float32),
        "gmm_covariances": rng.uniform(0.1, 3.0, size=(5, 16)).astype(np.float32),
        "gmm_weights": np.asarray([0.1, 0.2, 0.25, 0.15, 0.3], dtype=np.float32),
    }
    latent = rng.normal(size=(64, 16)).astype(np.float32)

    batch = _diag_gmm_stats(latent, state)
    singleton = tuple(
        np.concatenate(
            [_diag_gmm_stats(latent[pos : pos + 1], state)[part] for pos in range(len(latent))],
            axis=0,
        )
        for part in range(3)
    )

    for batch_values, singleton_values in zip(batch, singleton):
        np.testing.assert_array_equal(batch_values, singleton_values)


def test_denoising_ae_forward_is_batch_partition_invariant() -> None:
    rng = np.random.default_rng(812)
    widths = [4, 7, 3, 7, 4]
    spec = {
        "activation": "relu",
        "bottleneck": 3,
        "coefs": [
            rng.normal(size=(left, right)).astype(np.float32)
            for left, right in zip(widths[:-1], widths[1:])
        ],
        "intercepts": [
            rng.normal(size=right).astype(np.float32)
            for right in widths[1:]
        ],
    }
    values = rng.normal(size=(19, 4)).astype(np.float32)

    latent_batch, reconstruction_batch = _forward(spec, values)
    singleton_outputs = [_forward(spec, values[pos : pos + 1]) for pos in range(len(values))]
    latent_singletons = np.concatenate([value[0] for value in singleton_outputs], axis=0)
    reconstruction_singletons = np.concatenate(
        [value[1] for value in singleton_outputs],
        axis=0,
    )

    np.testing.assert_array_equal(latent_batch, latent_singletons)
    np.testing.assert_array_equal(reconstruction_batch, reconstruction_singletons)
