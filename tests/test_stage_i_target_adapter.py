from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_i_target_adapter import (
    CUMULATIVE_ORDINAL5_O,
    FOLD_QUANTILE_RESIDUAL3,
    SOFT_SCALAR_S,
    StageITargetAdapterError,
    canonical_sha256,
    bind_target_contract,
    fit_fold_quantile_residual3,
    file_sha256,
    load_base_target_winner_bundle,
    recover_base_score,
    reconstruct_fold_quantile_residual3,
    normalized_selector_sample_weight_context,
    training_objectives,
    verify_target_contract,
)
from extreme_price_movements.stage_i_model_hpo import run_stage_i_model_hpo


def _contract_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": ["a", "b", "c"],
        "__ts__": pd.date_range("2023-01-01", periods=3, freq="h", tz="UTC"),
        "__symbol__": ["BTC", "ETH", "SOL"],
        "side_name": ["long"] * 3,
        "target_value": [0.0, 0.5, 1.0],
        "gross_bps": [-200.0, 100.0, 400.0],
        "net_bps": [-300.0, 0.0, 300.0],
        "target_valid": [True, True, True],
        "sample_weight": [1.0, 0.5, 1.0],
    })


def test_contract_binds_identity_target_winner_economics_validity_and_weight() -> None:
    frame = _contract_frame()
    contract = bind_target_contract(
        frame, family=SOFT_SCALAR_S, layer="base", target_name="S__sl4_tp6",
        geometry="sl4_tp6", target_columns=("target_value",),
    )
    verify_target_contract(frame, contract)
    assert contract.geometry == "sl4_tp6"
    assert len({
        contract.identity_sha256, contract.target_sha256, contract.economics_sha256,
        contract.validity_sha256, contract.weight_sha256,
    }) == 5

    for column in ("candidate_id", "target_value", "net_bps", "target_valid", "sample_weight"):
        changed = frame.copy()
        if column == "candidate_id":
            changed.loc[0, column] = "changed"
        elif column == "target_valid":
            changed.loc[0, column] = False
        else:
            changed.loc[0, column] = float(changed.loc[0, column]) + 0.25
        with pytest.raises(StageITargetAdapterError, match="contract drift"):
            verify_target_contract(changed, contract)


def test_contract_hashes_invalid_path_missing_values_without_treating_them_as_labels() -> None:
    frame = _contract_frame()
    frame.loc[0, "target_valid"] = False
    frame.loc[0, ["target_value", "gross_bps", "net_bps"]] = np.nan
    contract = bind_target_contract(
        frame, family=SOFT_SCALAR_S, layer="base", target_name="S__sl4_tp6",
        geometry="sl4_tp6", target_columns=("target_value",),
    )
    verify_target_contract(frame, contract)
    changed = frame.copy()
    changed.loc[0, "net_bps"] = -999.0
    with pytest.raises(StageITargetAdapterError, match="contract drift"):
        verify_target_contract(changed, contract)


def test_selector_weight_context_normalizes_each_internal_fit(monkeypatch) -> None:
    from extreme_price_movements import lgbm_pipeline

    observed = []

    def fake_fit(_frame, _target, sample_weight, **_kwargs):
        observed.append(np.asarray(sample_weight, dtype=float))
        return object()

    monkeypatch.setattr(lgbm_pipeline, "_fit_lgbm_model", fake_fit)
    with normalized_selector_sample_weight_context():
        lgbm_pipeline._fit_lgbm_model(
            pd.DataFrame({"x": [1.0, 2.0, 3.0]}), np.asarray([0.0, 0.5, 1.0]),
            np.asarray([0.5, 1.0, 2.0]), classifier=False, params={},
        )
    assert len(observed) == 1
    assert observed[0].mean() == pytest.approx(1.0)


def test_hybrid_winner_uses_certainty_for_mda_but_retains_hybrid_hpo(tmp_path) -> None:
    target = pd.DataFrame({
        "candidate_id": ["a", "b", "c"],
        "__ts__": pd.date_range("2023-01-01", periods=3, freq="h", tz="UTC"),
        "__symbol__": ["BTC", "ETH", "SOL"],
        "side_name": ["long"] * 3,
        "decision_ts": pd.date_range("2023-01-01 01:00", periods=3, freq="h", tz="UTC"),
        "label_available_ts": pd.date_range("2023-01-01 13:00", periods=3, freq="h", tz="UTC"),
        "target_valid": [True] * 3,
        "gross_bps": [200.0, -100.0, 50.0],
        "net_bps": [100.0, -200.0, -50.0],
        "target_value": [1, 0, 2],
        "event": [2, 0, 1],
        "contract_certainty": [0.0, 0.5, 1.0],
        "sample_weight_base_component": [0.5, 0.75, 1.0],
        "sample_weight_requires_fold_local_fit": [True] * 3,
        "target_family": ["ordinal_O"] * 3,
        "target_name": ["O_a0p25__sl2_tp7"] * 3,
        "geometry": ["sl2_tp7"] * 3,
        "weight_mode": ["hybrid"] * 3,
    })
    target_path = tmp_path / "winner_target_handoff.parquet"
    target.to_parquet(target_path, index=False)
    weight = {
        "schema": "stage_i_base_target_training_weight_contract_v1",
        "mode": "hybrid",
        "fit_scope": "recomputed from each permitted training fold only",
    }
    weight["contract_sha256"] = canonical_sha256(weight)
    weight_path = tmp_path / "training_weight_contract.json"
    weight_path.write_text(json.dumps(weight))
    manifest = {
        "schema": "stage_i_base_target_winner_bundle_v1",
        "status": "complete",
        "artifact_sha256": {
            target_path.name: file_sha256(target_path),
            weight_path.name: file_sha256(weight_path),
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    loaded, contract, _ = load_base_target_winner_bundle(tmp_path, side="long")
    assert loaded.weight_mode.unique().tolist() == ["hybrid"]
    assert contract.metadata["mda_selection_weight_mode"] == "contract_certainty"
    assert contract.metadata["hpo_training_weight_mode"] == "hybrid"


def test_base_reconstruction_is_explicit_per_family() -> None:
    scalar, scalar_simplex = recover_base_score(SOFT_SCALAR_S, np.array([-1.0, 0.4, 2.0]))
    assert scalar_simplex is None
    np.testing.assert_allclose(scalar, [0.0, 0.4, 1.0])

    cumulative = np.array([[0.9, 0.7, 0.6, 0.2], [0.2, 0.8, 0.1, 0.9]])
    ordinal, simplex = recover_base_score(CUMULATIVE_ORDINAL5_O, cumulative)
    assert simplex is not None and simplex.shape == (2, 5)
    np.testing.assert_allclose(simplex.sum(axis=1), 1.0)
    assert np.all((ordinal >= 0.0) & (ordinal <= 1.0))
    assert [item["objective"] for item in training_objectives(CUMULATIVE_ORDINAL5_O)] == ["binary"] * 4


def test_fold_quantile_residual_uses_fold_local_terciles_winsor_and_centered_correction() -> None:
    residual = np.array([-300, -200, -100, -50, -10, 0, 20, 50, 100, 200, 300, 500], dtype=float)
    mapped = np.full(len(residual), 25.0)
    labels, state = fit_fold_quantile_residual3(residual + mapped, mapped)
    assert set(labels) == {0, 1, 2}
    assert state.thresholds_bps[0] < 0 <= state.thresholds_bps[1]
    assert state.shrinkage_support == 50.0
    assert state.correction_clip_bps == 200.0
    prior_rows = np.tile(np.asarray(state.class_prior), (3, 1))
    correction, reconstructed = reconstruct_fold_quantile_residual3(
        prior_rows, np.array([10.0, 20.0, 30.0]), state,
    )
    np.testing.assert_allclose(correction, 0.0, atol=1e-6)
    np.testing.assert_allclose(reconstructed, [10.0, 20.0, 30.0], atol=1e-6)


def test_fold_quantile_residual_semantic_gate_fails_closed() -> None:
    # Every residual is positive, so the middle tercile cannot straddle zero.
    with pytest.raises(StageITargetAdapterError, match="semantic gate"):
        fit_fold_quantile_residual3(
            np.arange(1.0, 31.0), np.zeros(30, dtype=float),
        )


def test_meta_family_cannot_be_silently_bound_as_base() -> None:
    frame = _contract_frame()
    with pytest.raises(StageITargetAdapterError, match="meta-only"):
        bind_target_contract(
            frame, family=FOLD_QUANTILE_RESIDUAL3, layer="base",
            target_name="bad", geometry="sl4_tp6", target_columns=("target_value",),
        )


class _RegressionModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return 0.5 + 0.4 * np.tanh(frame.iloc[:, 0].to_numpy(float))


class _ThreeClassModel:
    classes_ = np.array([0, 1, 2], dtype=np.int8)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        value = np.tanh(frame.iloc[:, 0].to_numpy(float))
        return np.column_stack([0.3 - 0.1 * value, np.full(len(frame), 0.4), 0.3 + 0.1 * value])


def _adapter_fit(_x, _y, _weight, *, classifier, params, **_kwargs):
    if classifier:
        assert params["objective"] in {"multiclass", "binary"}
        return _ThreeClassModel()
    assert params["objective"] in {"regression_l1", "huber"}
    return _RegressionModel()


def _runtime_contract(family: str, layer: str, rows: int):
    from extreme_price_movements.stage_i_target_adapter import StageITargetContract

    return StageITargetContract(
        family=family, layer=layer, target_name=family, geometry="sl4_tp6",
        identity_sha256="1" * 64, target_sha256="2" * 64,
        economics_sha256="3" * 64, validity_sha256="4" * 64,
        weight_sha256="5" * 64, rows=rows, target_columns=("target",),
    )


def test_hpo_soft_scalar_uses_explicit_regression_adapter() -> None:
    n = 900
    timestamps = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    signal = np.sin(np.arange(n) / 17.0)
    result = run_stage_i_model_hpo(
        pd.DataFrame({"signal": signal}), (signal + 1.0) / 2.0,
        selected_feature_names=("signal",), candidate_ids=[f"c{i}" for i in range(n)],
        exact_net_bps=signal * 200.0,
        decision_timestamps=timestamps,
        label_available_timestamps=timestamps + pd.Timedelta(hours=12),
        side="long", layer="base", target_contract=_runtime_contract(SOFT_SCALAR_S, "base", n),
        hpo_trials=1, hpo_patience=1, n_validation_folds=3, min_train_rows=20,
        fit_model=_adapter_fit,
    )
    assert result.target_family == SOFT_SCALAR_S
    assert result.oof_probabilities is None
    assert result.best_params["objective"] == "regression_l1"


def test_hpo_recomputes_hybrid_weights_inside_each_strict_training_fold() -> None:
    n = 900
    timestamps = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    signal = np.sin(np.arange(n) / 17.0)
    seen_weights: list[np.ndarray] = []

    def fit_with_weight_audit(_x, _y, sample_weight, *, classifier, params, **_kwargs):
        assert not classifier and params["objective"] == "regression_l1"
        seen_weights.append(np.asarray(sample_weight, dtype=float).copy())
        return _RegressionModel()

    fold_weight_frame = pd.DataFrame({
        "decision_ts": timestamps,
        "contract_certainty": np.resize(np.asarray([0.2, 0.5, 0.8, 1.0]), n),
        "causal_regime": np.resize(np.asarray(["low", "low", "mid", "high"]), n),
    })
    result = run_stage_i_model_hpo(
        pd.DataFrame({"signal": signal}), (signal + 1.0) / 2.0,
        selected_feature_names=("signal",), candidate_ids=[f"h{i}" for i in range(n)],
        exact_net_bps=signal * 200.0,
        decision_timestamps=timestamps,
        label_available_timestamps=timestamps + pd.Timedelta(hours=12),
        side="long", layer="base", target_contract=_runtime_contract(SOFT_SCALAR_S, "base", n),
        sample_weight=np.ones(n), fold_local_weight_frame=fold_weight_frame,
        fold_local_weight_mode="hybrid", fold_local_regime_column="causal_regime",
        hpo_trials=1, hpo_patience=1, n_validation_folds=3, min_train_rows=20,
        fit_model=fit_with_weight_audit,
    )
    assert seen_weights
    assert all(np.mean(weight) == pytest.approx(1.0, abs=1e-6) for weight in seen_weights)
    assert all(np.unique(np.round(weight, 6)).size > 2 for weight in seen_weights)
    assert all(row["training_weight_fit_scope"] == "strict_fold_train_only" for row in result.fold_audit)
    assert all(row["training_weight_mode"] == "hybrid" for row in result.oof_fold_audit)


def test_hpo_fold_quantile_meta_reconstructs_around_fixed_base_ev() -> None:
    n = 900
    timestamps = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    signal = np.resize(np.asarray([-1.0, -0.4, 0.2, 0.8]), n)
    mapped = np.full(n, 10.0)
    net = mapped + signal * 250.0
    result = run_stage_i_model_hpo(
        pd.DataFrame({"signal": signal}), np.zeros(n),
        selected_feature_names=("signal",), candidate_ids=[f"c{i}" for i in range(n)],
        exact_net_bps=net, decision_timestamps=timestamps,
        label_available_timestamps=timestamps + pd.Timedelta(hours=12),
        side="long", layer="meta",
        target_contract=_runtime_contract(FOLD_QUANTILE_RESIDUAL3, "meta", n),
        prediction_offset_bps=mapped, hpo_trials=1, hpo_patience=1,
        n_validation_folds=3, min_train_rows=20, fit_model=_adapter_fit,
    )
    assert result.target_family == FOLD_QUANTILE_RESIDUAL3
    assert result.oof_probabilities is not None
    assert result.oof_probabilities.shape == (n, 3)
    assert result.best_params["objective"] == "multiclass"


def test_hpo_direct_fq3_uses_native_score_offset_and_rejects_bps_map() -> None:
    from dataclasses import replace

    n = 900
    timestamps = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(20260803)
    native = rng.uniform(-0.9, 0.9, n)
    economic_signal = np.sin(np.arange(n) / 17.0)
    net = economic_signal * 300.0 + np.resize(np.asarray([-25.0, 0.0, 25.0]), n)
    contract = replace(
        _runtime_contract(FOLD_QUANTILE_RESIDUAL3, "meta", n),
        metadata={
            "meta_target_semantics": "same_side_direct_base_output_correctness_q33_v1",
            "base_input_semantics": "same_side_direct_base_output_without_bps_conversion_v1",
            "native_score_domain": [-1.0, 1.0],
        },
    )
    frame = pd.DataFrame({"base_raw_score": native, "context": np.sin(np.arange(n) / 13.0)})
    result = run_stage_i_model_hpo(
        frame, net, selected_feature_names=("base_raw_score", "context"),
        candidate_ids=[f"r3-{i}" for i in range(n)], exact_net_bps=net,
        decision_timestamps=timestamps,
        label_available_timestamps=timestamps + pd.Timedelta(hours=12),
        side="long", layer="meta", target_contract=contract,
        prediction_offset_native_score=native, hpo_trials=1, hpo_patience=1,
        n_validation_folds=3, min_train_rows=20, fit_model=_adapter_fit,
    )
    assert result.oof_probabilities is not None and result.oof_probabilities.shape == (n, 3)
    finite = np.isfinite(result.oof_score)
    reconstructed = native[finite] + result.oof_score[finite]
    assert (reconstructed >= -1.000001).all() and (reconstructed <= 1.000001).all()
    with pytest.raises(Exception, match="forbids a bps offset"):
        run_stage_i_model_hpo(
            frame, net, selected_feature_names=("base_raw_score", "context"),
            candidate_ids=[f"r3-{i}" for i in range(n)], exact_net_bps=net,
            decision_timestamps=timestamps,
            label_available_timestamps=timestamps + pd.Timedelta(hours=12),
            side="long", layer="meta", target_contract=contract,
            prediction_offset_bps=native, hpo_trials=1, hpo_patience=1,
            n_validation_folds=3, min_train_rows=20, fit_model=_adapter_fit,
        )
