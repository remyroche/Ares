from __future__ import annotations

from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_i_adapter_winner_bundle import (
    StageIAdapterWinnerBundle,
    StageIAdapterWinnerCell,
)
from extreme_price_movements.stage_i_causal_admission import Causal21dAdmissionSpec
from extreme_price_movements.stage_i_target_adapter import (
    FOLD_QUANTILE_RESIDUAL3,
    LEGACY_R3_MULTICLASS3,
    SOFT_SCALAR_S,
    bind_target_contract,
    canonical_sha256,
    file_sha256,
)
from extreme_price_movements.stage_i_target_specific_oos import (
    DIRECT_BASE_INPUT_SEMANTICS,
    DIRECT_FQ3_SEMANTICS,
    DirectCorrectnessState,
    StageITargetSpecificInput,
    StageITargetSpecificFinalist,
    FrozenR3FinalistInput,
    TargetSpecificOOSError,
    compare_target_specific_finalists,
    direct_fq3_selector_fit_context,
    fit_direct_fq3_estimator,
    _is_reserved_source_feature,
    _validate_frozen_r3_handoff,
    load_target_specific_finalist_artifact,
    preflight_strict_meta_availability,
    validate_preflight_strict_meta_availability_equality,
    write_frozen_r3_finalist_normalizer,
    run_stage_i_target_specific_oos,
)


def test_causal_path_features_are_allowed_without_opening_path_label_leakage() -> None:
    """Only the explicit entry-time path summaries may cross the source gate."""
    assert not _is_reserved_source_feature("path_entropy_24")
    assert not _is_reserved_source_feature("path_efficiency_12")
    assert _is_reserved_source_feature("path_future_gross")
    assert _is_reserved_source_feature("future_slope_atr_per_hour")


def test_frozen_r3_handoff_projects_a_signed_common_subset_in_contract_order() -> None:
    source, _cell = _frozen_r3_direct_source("long", n=48)
    selected = np.arange(1, len(source.contract_frame), 2)
    contract = source.contract_frame.iloc[selected].reset_index(drop=True)
    score, states, folds = _validate_frozen_r3_handoff(
        source, side="long", contract=contract,
    )
    expected = source.frozen_base_oof.iloc[selected].reset_index(drop=True)
    assert np.allclose(score, expected.base_raw_score.to_numpy(np.float32))
    assert np.allclose(states[:, 2] - states[:, 0], score)
    assert np.array_equal(folds, expected.base_oof_fold_id.to_numpy(np.int16))


class _Regressor:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.clip(0.5 + 0.4 * frame["base_feature"].to_numpy(float), 0.0, 1.0)


class _Classifier:
    classes_ = np.asarray([0, 1, 2], dtype=np.int8)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        signal = np.tanh(frame["context"].to_numpy(float))
        return np.column_stack([0.25 - 0.1 * signal, np.full(len(frame), 0.5), 0.25 + 0.1 * signal])


def _fit(_x, _y, _w, *, classifier: bool, **_kwargs):
    return _Classifier() if classifier else _Regressor()


def _source(side: str, *, n: int = 192, include_bad_converted_feature: bool = False) -> tuple[StageITargetSpecificInput, StageIAdapterWinnerCell]:
    signal = np.resize(np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0]), n)
    timestamp = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    net = signal * 250.0 + np.resize(np.asarray([-20.0, 0.0, 20.0]), n)
    contract = pd.DataFrame({
        "candidate_id": [f"{side}-{i}" for i in range(n)], "__ts__": timestamp,
        "__symbol__": np.resize(np.asarray(["BTC", "ETH"]), n), "side_name": side,
        "base_target": np.clip((signal + 1.0) / 2.0, 0.0, 1.0), "meta_target": net,
        "gross_bps": net + 100.0, "net_bps": net, "target_valid": True, "sample_weight": 1.0,
        "decision_ts": timestamp + pd.Timedelta(hours=1), "label_available_ts": timestamp + pd.Timedelta(hours=13),
    })
    meta_metadata = {
        "meta_target_semantics": DIRECT_FQ3_SEMANTICS,
        "base_input_semantics": DIRECT_BASE_INPUT_SEMANTICS,
        "required_regime_features": ["regime"], "required_context_features": ["context"],
        "required_trust_features": ["base_output_entropy"],
    }
    base_contract = bind_target_contract(
        contract, family=SOFT_SCALAR_S, layer="base", target_name="S", geometry="TP6_SL4_H12",
        target_columns=("base_target",), metadata={"training_weight_contract": {"mode": "uniform"}},
    )
    meta_contract = bind_target_contract(
        contract, family=FOLD_QUANTILE_RESIDUAL3, layer="meta", target_name="FQ3", geometry="TP6_SL4_H12",
        target_columns=("meta_target",), metadata=meta_metadata,
    )
    meta_features = (
        "context", "regime", "base_raw_score", "base_state_p0", "base_state_p1",
        "base_output_entropy", "base_output_top2_margin", "base_output_max_probability",
    )
    if include_bad_converted_feature:
        meta_features = (*meta_features, "prequential_base_expected_net_bps")
    base_manifest = {
        "status": "complete", "side": side, "target_contract_sha256": base_contract.sha256,
        "selected_feature_contract": ["base_feature"], "correlation_policy": "grouped-preserve",
    }
    meta_manifest = {
        "status": "complete", "side": side, "target_contract_sha256": meta_contract.sha256,
        "selected_feature_contract": list(meta_features), "correlation_policy": "grouped-preserve",
    }
    cell = StageIAdapterWinnerCell(
        side=side, base_features=("base_feature",), meta_features=meta_features,
        base_params={"objective": "regression_l1"}, meta_params={"objective": "multiclass", "num_class": 3},
        base_target_contract=base_contract, meta_target_contract=meta_contract,
        base_selector_manifest_sha256="a" * 64, meta_selector_manifest_sha256="b" * 64,
        required_same_side_base_handoff_features=("base_raw_score", "base_state_p0", "base_state_p1"),
    )
    role_contract = {
        "schema": "stage_i_target_specific_causal_feature_roles_v1",
        "base_source_features": ["base_feature"],
        "meta_source_features": ["context", "regime"],
    }
    availability = {
        month: (
            {"source_available": True}
            if month == "2024-01"
            else {"source_available": False, "source_gap_reason": "not materialised in compact test fixture"}
        )
        for month in pd.period_range("2024-01", "2026-12", freq="M").astype(str)
    }
    month_contract = {
        "schema": "stage_i_target_specific_2024_2026_month_coverage_v1",
        "expected_months": list(availability), "source_availability": availability,
    }
    evaluation_contracts = {
        "schema": "stage_i_target_specific_evaluation_target_contracts_v1",
        "side": side,
        "training_base_target_contract_sha256": base_contract.sha256,
        "training_meta_target_contract_sha256": meta_contract.sha256,
        "base": base_contract.to_dict(), "meta": meta_contract.to_dict(),
    }
    source_manifest = {
        "artifact_sha256": {"features.parquet": "c" * 64, "contract.parquet": "d" * 64},
        "causal_feature_role_contract": role_contract,
        "causal_feature_role_contract_sha256": canonical_sha256(role_contract),
        "evaluation_month_contract": month_contract,
        "evaluation_month_contract_sha256": canonical_sha256(month_contract),
        "evaluation_target_contracts": evaluation_contracts,
        "evaluation_target_contracts_sha256": canonical_sha256(evaluation_contracts),
    }
    source = StageITargetSpecificInput(
        side=side,
        frame=pd.DataFrame({
            "candidate_id": contract.candidate_id, "__ts__": timestamp, "__symbol__": contract.__symbol__,
            "base_feature": signal, "context": signal, "regime": np.resize([0.0, 1.0], n),
        }),
        contract_frame=contract, source_manifest=source_manifest, source_manifest_sha256=canonical_sha256(source_manifest),
        source_file_sha256=source_manifest["artifact_sha256"], base_selector_manifest=base_manifest,
        meta_selector_manifest=meta_manifest, base_selector_manifest_sha256="a" * 64,
        meta_selector_manifest_sha256="b" * 64, base_target_column="base_target", meta_target_column="meta_target",
        n_validation_folds=3, min_train_rows=24,
    )
    return source, cell


def _frozen_r3_direct_source(side: str, *, n: int = 192) -> tuple[StageITargetSpecificInput, StageIAdapterWinnerCell]:
    source, old_cell = _source(side, n=n)
    contract = source.contract_frame.copy()
    signal = np.resize(np.asarray([-0.9, -0.45, 0.0, 0.45, 0.9]), n).astype(np.float32)
    contract["r3_class"] = np.resize(np.asarray([0, 1, 2], dtype=np.int8), n)
    base_contract = bind_target_contract(
        contract, family=LEGACY_R3_MULTICLASS3, layer="base", target_name="R3_frozen_control",
        geometry="TP6_SL4_H12", target_columns=("r3_class",),
        metadata={"training_weight_contract": {"mode": "uniform"}},
    )
    meta_contract = bind_target_contract(
        contract, family=FOLD_QUANTILE_RESIDUAL3, layer="meta", target_name="FQ3",
        geometry="TP6_SL4_H12", target_columns=("meta_target",),
        metadata={
            "meta_target_semantics": DIRECT_FQ3_SEMANTICS,
            "base_input_semantics": DIRECT_BASE_INPUT_SEMANTICS,
            "required_regime_features": ["regime"], "required_context_features": ["context"],
            "required_trust_features": ["base_output_entropy"],
        },
    )
    p_clear = (signal + 1.0) / 2.0 * 0.8 + 0.1
    p_adverse = (1.0 - signal) / 2.0 * 0.8 + 0.1
    p_weak = np.full(n, 0.2, dtype=np.float32)
    scale = p_clear + p_adverse + p_weak
    p_clear, p_adverse, p_weak = p_clear / scale, p_adverse / scale, p_weak / scale
    native = p_clear - p_adverse
    frozen = contract.loc[:, ["candidate_id", "__ts__", "__symbol__", "side_name", "decision_ts", "label_available_ts"]].copy()
    frozen["exact_net_bps"], frozen["exact_gross_bps"] = contract.net_bps, contract.gross_bps
    frozen["r3_p_adverse"], frozen["r3_p_weak"], frozen["r3_p_clear"] = p_adverse, p_weak, p_clear
    frozen["r3_opportunity_score"], frozen["base_raw_score"] = native, native
    frozen["base_oof_fold_id"] = np.resize(np.asarray([0, 1, 2], dtype=np.int16), n)
    base_manifest = {
        "status": "complete", "side": side, "target_contract_sha256": base_contract.sha256,
        "target_contract": base_contract.to_dict(), "selected_feature_contract": ["base_feature"],
        "correlation_policy": "grouped-preserve", "selector_base_oof_sha256": "e" * 64,
        "hpo_oof_score_semantics": "P(clear)-P(adverse)",
        "hpo_oof_regeneration_fold_audit": [{
            "fold_id": fold, "train_rows": 24, "validation_rows": 64,
            "validation_start_utc": "2024-01-02T00:00:00Z",
            "train_max_label_available_utc": "2024-01-01T23:00:00Z",
            "strict_prior_resolved": True,
        } for fold in range(3)],
    }
    meta_features = (*old_cell.meta_features, "base_state_p2")
    meta_manifest = {
        "status": "complete", "side": side, "target_contract_sha256": meta_contract.sha256,
        "selected_feature_contract": list(meta_features), "correlation_policy": "grouped-preserve",
    }
    cell = StageIAdapterWinnerCell(
        side=side, base_features=old_cell.base_features, meta_features=meta_features,
        base_params={"objective": "multiclass", "num_class": 3},
        meta_params={"objective": "multiclass", "num_class": 3},
        base_target_contract=base_contract, meta_target_contract=meta_contract,
        base_selector_manifest_sha256="a" * 64, meta_selector_manifest_sha256="b" * 64,
        required_same_side_base_handoff_features=("base_raw_score", "base_state_p0", "base_state_p1", "base_state_p2"),
    )
    evaluation_contracts = {
        "schema": "stage_i_target_specific_evaluation_target_contracts_v1",
        "side": side,
        "training_base_target_contract_sha256": base_contract.sha256,
        "training_meta_target_contract_sha256": meta_contract.sha256,
        "base": base_contract.to_dict(), "meta": meta_contract.to_dict(),
    }
    source_manifest = dict(source.source_manifest)
    source_manifest["evaluation_target_contracts"] = evaluation_contracts
    source_manifest["evaluation_target_contracts_sha256"] = canonical_sha256(evaluation_contracts)
    return replace(
        source, contract_frame=contract, source_manifest=source_manifest,
        source_manifest_sha256=canonical_sha256(source_manifest), base_selector_manifest=base_manifest,
        meta_selector_manifest=meta_manifest, base_target_column="r3_class",
        frozen_base_oof=frozen, frozen_base_oof_manifest=base_manifest,
        frozen_base_oof_file_sha256="e" * 64, frozen_base_oof_manifest_sha256="a" * 64,
    ), cell


def test_frozen_r3_runs_only_direct_fq3_meta_and_preserves_native_score(tmp_path) -> None:
    long, long_cell = _frozen_r3_direct_source("long")
    short, short_cell = _frozen_r3_direct_source("short")
    fit_calls: list[bool] = []

    def classifier_only_fit(_x, _y, _w, *, classifier: bool, **_kwargs):
        fit_calls.append(bool(classifier))
        if not classifier:
            raise AssertionError("frozen R3 base must never be refit")
        return _Classifier()

    bundle = StageIAdapterWinnerBundle(cells=(long_cell, short_cell), code_revision="immutable-r3-test")
    manifest = run_stage_i_target_specific_oos(
        bundle=bundle, inputs=(long, short), output_dir=tmp_path / "r3_direct_fq3",
        fit_model=classifier_only_fit,
        admission_spec=Causal21dAdmissionSpec(min_reference_rows=20, min_side_reference_rows=4, bins=4),
    )
    assert fit_calls and all(fit_calls)
    assert manifest["promotion_rule"].startswith("joint reconstructed")
    prediction = pd.read_parquet(tmp_path / "r3_direct_fq3" / "full_history_strict_oof_predictions.parquet")
    expected = pd.concat([long.frozen_base_oof, short.frozen_base_oof], ignore_index=True)
    expected = expected.set_index(["side_name", "candidate_id"])["base_raw_score"]
    observed = prediction.set_index(["side_name", "candidate_id"])["base_direct_score"]
    assert np.allclose(observed.loc[expected.index], expected)
    assert {
        "meta_p_error_tercile_0", "meta_p_error_tercile_1", "meta_p_error_tercile_2"
    }.issubset(prediction)
    provenance = pd.read_parquet(tmp_path / "r3_direct_fq3" / "fold_provenance.parquet")
    frozen_base = provenance.loc[provenance.layer.eq("base")]
    assert frozen_base.frozen_completed_base.fillna(False).all()
    joint = pd.read_parquet(tmp_path / "r3_direct_fq3" / "joint_stack_promotion_score.parquet")
    assert joint.joint_stack_only.all() and "base_direct_score" not in joint.columns


def test_frozen_r3_rejects_non_strict_or_converted_handoff(tmp_path) -> None:
    source, cell = _frozen_r3_direct_source("long")
    short, short_cell = _frozen_r3_direct_source("short")
    drift = source.frozen_base_oof.copy()
    finite = drift.base_raw_score.notna()
    drift.loc[finite, "base_raw_score"] += 0.01
    bad = replace(source, frozen_base_oof=drift)
    bundle = StageIAdapterWinnerBundle(cells=(cell, short_cell), code_revision="immutable-r3-test")
    with pytest.raises(TargetSpecificOOSError, match="native direct simplex/contrast"):
        run_stage_i_target_specific_oos(
            bundle=bundle, inputs=(bad, short), output_dir=tmp_path / "bad_r3", fit_model=_fit,
            admission_spec=Causal21dAdmissionSpec(min_reference_rows=20, min_side_reference_rows=4, bins=4),
        )


def test_direct_fq3_reusable_estimator_fits_fold_local_labels_without_bps_map() -> None:
    frame = pd.DataFrame({
        "base_raw_score": np.linspace(-0.9, 0.9,  ninety := 90),
        "context": np.resize(np.asarray([-1.0, 0.0, 1.0]), ninety),
    })
    net = np.linspace(-300.0, 300.0, ninety) + np.resize(np.asarray([-30.0, 0.0, 30.0]), ninety)
    captured: dict[str, np.ndarray] = {}

    def fit_classifier(x, y, w, *, classifier, **_kwargs):
        assert classifier
        captured["labels"] = np.asarray(y)
        captured["weights"] = np.asarray(w)
        return _Classifier()

    estimator = fit_direct_fq3_estimator(
        frame, net, np.ones(ninety), params={"objective": "multiclass", "num_class": 3},
        score_domain=(-1.0, 1.0), fit_model=fit_classifier,
    )
    assert set(captured["labels"]) == {0, 1, 2}
    assert estimator.state.score_lower == -1.0 and estimator.state.score_upper == 1.0
    assert np.isfinite(estimator.predict(frame)).all()
    with pytest.raises(TargetSpecificOOSError, match="pre-mapped"):
        fit_direct_fq3_estimator(
            frame.assign(prequential_base_expected_net_bps=0.0), net, np.ones(ninety),
            params={"objective": "multiclass", "num_class": 3},
            score_domain=(-1.0, 1.0), fit_model=fit_classifier,
        )


def test_direct_fq3_labels_are_invariant_to_context_permutation_and_require_raw_base() -> None:
    n =  ninety = 90
    frame = pd.DataFrame({
        "base_raw_score": np.resize(np.asarray([-0.8, -0.2, 0.25, 0.75]), n),
        "context": np.arange(n, dtype=float),
    })
    net = np.sin(np.arange(n) / 7.0) * 250.0
    labels: list[np.ndarray] = []

    def capture_fit(_x, y, _w, *, classifier, **_kwargs):
        assert classifier
        labels.append(np.asarray(y).copy())
        return _Classifier()

    fit_direct_fq3_estimator(
        frame, net, None, params={"objective": "multiclass", "num_class": 3},
        score_domain=(-1.0, 1.0), fit_model=capture_fit,
    )
    permuted = frame.copy()
    permuted["context"] = permuted.context.sample(frac=1.0, random_state=7).to_numpy()
    fit_direct_fq3_estimator(
        permuted, net, None, params={"objective": "multiclass", "num_class": 3},
        score_domain=(-1.0, 1.0), fit_model=capture_fit,
    )
    assert np.array_equal(labels[0], labels[1])
    with pytest.raises(TargetSpecificOOSError, match="requires base_raw_score"):
        fit_direct_fq3_estimator(
            frame.drop(columns="base_raw_score"), net, None,
            params={"objective": "multiclass", "num_class": 3},
            score_domain=(-1.0, 1.0), fit_model=capture_fit,
        )


def test_direct_fq3_selector_uses_parent_labels_and_neutralizes_unsupported_children(monkeypatch) -> None:
    from extreme_price_movements import lgbm_pipeline

    parent = DirectCorrectnessState(
        thresholds=(-0.2, 0.2), class_prior=(1 / 3, 1 / 3, 1 / 3),
        class_locations=(-0.5, 0.0, 0.5), class_support=(30, 30, 30),
        score_lower=-1.0, score_upper=1.0,
    )
    captured: list[np.ndarray] = []

    def fake_fit(_frame, labels, _weight, **_kwargs):
        captured.append(np.asarray(labels).copy())
        return _Classifier()

    monkeypatch.setattr(lgbm_pipeline, "_fit_lgbm_model", fake_fit)
    supported_labels = np.resize(np.asarray([0, 1, 2], dtype=np.int8), 12)
    frame = pd.DataFrame({
        "base_raw_score": np.linspace(-0.9, 0.9, 12),
        "context": np.arange(12, dtype=float),
    })
    with direct_fq3_selector_fit_context(parent_state=parent):
        first = lgbm_pipeline._fit_lgbm_model(
            frame, supported_labels, np.ones(12), params={}, classifier=False,
        )
        perturbed = frame.copy()
        perturbed["base_raw_score"] = -perturbed["base_raw_score"]
        second = lgbm_pipeline._fit_lgbm_model(
            perturbed, supported_labels, np.ones(12), params={}, classifier=False,
        )
        neutral = lgbm_pipeline._fit_lgbm_model(
            frame.iloc[:4], np.asarray([2, 2, 2, 2]), np.ones(4),
            params={}, classifier=False,
        )
        with pytest.raises(TargetSpecificOOSError, match="protected base_raw_score"):
            lgbm_pipeline._fit_lgbm_model(
                frame.drop(columns="base_raw_score"), supported_labels, np.ones(12),
                params={}, classifier=False,
            )

    assert np.array_equal(captured[0], supported_labels)
    assert np.array_equal(captured[1], supported_labels)
    assert first.nested_support_audit["status"] == "supported_parent_defined_three_class_fit"
    assert second.nested_support_audit["status"] == "supported_parent_defined_three_class_fit"
    probability = neutral.predict_proba(frame.iloc[:4])
    np.testing.assert_allclose(probability, np.full((4, 3), 1 / 3), atol=1e-7)
    assert neutral.nested_support_audit["status"] == "unsupported_child_class_support_neutral_prior"


def test_direct_fq3_keeps_systematic_calibration_offset_as_a_learnable_target() -> None:
    from extreme_price_movements.stage_i_target_specific_oos import _fit_direct_correctness

    # A high native base score makes even q67 negative.  This is systematic
    # overestimation for the meta layer to learn, not invalid supervision.
    net = np.linspace(-300.0, 300.0,  ninety := 90)
    base = np.full(ninety, 0.8)
    labels, state = _fit_direct_correctness(net, base, score_domain=(-1.0, 1.0))

    assert state.thresholds[1] < 0.0
    assert tuple(np.bincount(labels, minlength=3)) == state.class_support
    assert all(value > 0 for value in state.class_support)


def test_direct_fq3_oos_is_strict_maps_only_after_meta_and_reports_side_month(tmp_path) -> None:
    long, long_cell = _source("long")
    short, short_cell = _source("short")
    bundle = StageIAdapterWinnerBundle(cells=(long_cell, short_cell), code_revision="immutable-test")
    manifest = run_stage_i_target_specific_oos(
        bundle=bundle, inputs=(long, short), output_dir=tmp_path / "out", fit_model=_fit,
        admission_spec=Causal21dAdmissionSpec(min_reference_rows=20, min_side_reference_rows=4, bins=4),
    )
    assert manifest["meta_input"] == DIRECT_BASE_INPUT_SEMANTICS
    assert "only pooled global after" in manifest["ranking"]
    prediction = pd.read_parquet(tmp_path / "out" / "strict_oof_predictions.parquet")
    assert {
        "base_state_p0", "base_state_p1", "meta_direct_score",
        "meta_p_error_tercile_0", "meta_p_error_tercile_1", "meta_p_error_tercile_2",
        "base_causal_21d_expected_net_bps", "meta_causal_21d_expected_net_bps",
    }.issubset(prediction)
    # bps maps are output-only: no converted base bps is present in the FQ3 source ledger.
    assert "prequential_base_expected_net_bps" not in prediction
    provenance = pd.read_parquet(tmp_path / "out" / "fold_provenance.parquet")
    meta = provenance.loc[provenance.layer.eq("meta") & ~provenance.skipped.fillna(False)]
    assert not meta.empty and meta.strict_prior_resolved.all()
    metrics = pd.read_parquet(tmp_path / "out" / "per_side_month_base_meta_metrics.parquet")
    assert {"base", "meta"}.issubset(metrics.layer.dropna())
    assert ((metrics.side_name == "pooled_global") & (metrics.month == "all")).any()
    coverage = pd.read_parquet(tmp_path / "out" / "2024_2026_side_month_coverage_audit.parquet")
    assert len(coverage) == 2 * 36 * 2
    assert coverage.promotion_coverage_status.eq("pass").all()
    joint = pd.read_parquet(tmp_path / "out" / "joint_stack_promotion_score.parquet")
    assert joint.joint_stack_only.all() and "base_direct_score" not in joint.columns
    assert (tmp_path / "out" / "worst_period_diagnostics.parquet").is_file()


def test_direct_fq3_rejects_preconverted_base_feature(tmp_path) -> None:
    long, long_cell = _source("long", include_bad_converted_feature=True)
    short, short_cell = _source("short", include_bad_converted_feature=True)
    bundle = StageIAdapterWinnerBundle(cells=(long_cell, short_cell), code_revision="immutable-test")
    with pytest.raises(TargetSpecificOOSError, match="converted base bps"):
        run_stage_i_target_specific_oos(
            bundle=bundle, inputs=(long, short), output_dir=tmp_path / "out", fit_model=_fit,
            admission_spec=Causal21dAdmissionSpec(min_reference_rows=20, min_side_reference_rows=4, bins=4),
        )


def test_direct_fq3_rejects_source_hash_and_allows_independent_layer_policy(tmp_path) -> None:
    long, long_cell = _source("long")
    short, short_cell = _source("short")
    bundle = StageIAdapterWinnerBundle(cells=(long_cell, short_cell), code_revision="immutable-test")
    bad_hash = replace(long, source_file_sha256={"features.parquet": "x" * 64, "contract.parquet": "d" * 64})
    with pytest.raises(TargetSpecificOOSError, match="artifact hash drift"):
        run_stage_i_target_specific_oos(
            bundle=bundle, inputs=(bad_hash, short), output_dir=tmp_path / "hash", fit_model=_fit,
            admission_spec=Causal21dAdmissionSpec(min_reference_rows=20, min_side_reference_rows=4, bins=4),
        )
    drift_meta = dict(short.meta_selector_manifest)
    drift_meta["correlation_policy"] = "pre-mda-spearman-representative"
    independent_policy = replace(short, meta_selector_manifest=drift_meta)
    result = run_stage_i_target_specific_oos(
        bundle=bundle, inputs=(long, independent_policy), output_dir=tmp_path / "policy", fit_model=_fit,
        admission_spec=Causal21dAdmissionSpec(min_reference_rows=20, min_side_reference_rows=4, bins=4),
    )
    assert result["source_lineage"]["short"]["meta_correlation_policy"] == "pre-mda-spearman-representative"


@pytest.mark.parametrize("feature", ["god_feature", "net_bps", "path_future_leak"])
def test_direct_fq3_rejects_unapproved_or_reserved_base_features(feature: str, tmp_path) -> None:
    long, long_cell = _source("long")
    short, short_cell = _source("short")
    frame = long.frame.assign(**{feature: np.linspace(0.0, 1.0, len(long.frame))})
    base_manifest = dict(long.base_selector_manifest)
    base_manifest["selected_feature_contract"] = [feature]
    changed_source = replace(long, frame=frame, base_selector_manifest=base_manifest)
    changed_cell = replace(long_cell, base_features=(feature,))
    bundle = StageIAdapterWinnerBundle(cells=(changed_cell, short_cell), code_revision="immutable-test")
    with pytest.raises(TargetSpecificOOSError, match="reserved|approved causal inventory"):
        run_stage_i_target_specific_oos(
            bundle=bundle, inputs=(changed_source, short), output_dir=tmp_path / feature, fit_model=_fit,
            admission_spec=Causal21dAdmissionSpec(min_reference_rows=20, min_side_reference_rows=4, bins=4),
        )


def test_direct_fq3_rejects_unknown_or_source_supplied_base_state(tmp_path) -> None:
    long, long_cell = _source("long")
    short, short_cell = _source("short")
    unknown_meta = (*long_cell.meta_features, "base_state_p9")
    meta_manifest = dict(long.meta_selector_manifest)
    meta_manifest["selected_feature_contract"] = list(unknown_meta)
    changed_source = replace(long, meta_selector_manifest=meta_manifest)
    changed_cell = replace(long_cell, meta_features=unknown_meta)
    bundle = StageIAdapterWinnerBundle(cells=(changed_cell, short_cell), code_revision="immutable-test")
    with pytest.raises(TargetSpecificOOSError, match="reserved"):
        run_stage_i_target_specific_oos(
            bundle=bundle, inputs=(changed_source, short), output_dir=tmp_path / "unknown_state", fit_model=_fit,
            admission_spec=Causal21dAdmissionSpec(min_reference_rows=20, min_side_reference_rows=4, bins=4),
        )
    supplied_state = replace(long, frame=long.frame.assign(base_state_p0=0.5))
    with pytest.raises(TargetSpecificOOSError, match="illegally supplies generated handoffs"):
        run_stage_i_target_specific_oos(
            bundle=StageIAdapterWinnerBundle(cells=(long_cell, short_cell), code_revision="immutable-test"),
            inputs=(supplied_state, short), output_dir=tmp_path / "supplied_state", fit_model=_fit,
            admission_spec=Causal21dAdmissionSpec(min_reference_rows=20, min_side_reference_rows=4, bins=4),
        )


def test_finalist_promotion_uses_reconstructed_meta_only_on_identical_rows(tmp_path) -> None:
    del tmp_path
    prediction = pd.DataFrame({
        "candidate_id": ["l1", "s1", "l2", "s2"], "side_name": ["long", "short", "long", "short"],
        "decision_ts": pd.date_range("2024-02-01", periods=4, freq="D", tz="UTC"),
        "target_valid": True, "strict_oof_available": True,
        "exact_gross_bps": [250.0, 230.0, 180.0, 170.0], "exact_net_bps": [150.0, 130.0, 80.0, 70.0],
        "meta_causal_21d_expected_net_bps": [180.0, 160.0, 100.0, 90.0],
        "meta_causal_21d_admitted": True, "base_direct_score": [0.1, 0.9, 0.2, 0.8],
    })
    # Alter a base-only diagnostic that is not part of the comparison identity
    # or reconstructed meta score.  The result must remain a joint-stack row.
    alternate = prediction.copy()
    alternate["base_direct_score"] = 0.0
    summary, attribution = compare_target_specific_finalists((
        StageITargetSpecificFinalist("S", prediction, {"base_family": "S", "shared_population_contract_sha256": "a" * 64}),
        StageITargetSpecificFinalist("R3", alternate, {"base_family": "R3", "shared_population_contract_sha256": "a" * 64}),
    ))
    assert set(summary.finalist) == {"S", "R3"}
    assert summary.joint_stack_only.all()
    assert "base_direct_score" not in summary.columns
    assert set(attribution.finalist).issubset({"S", "R3"})


def test_undeclared_zero_strict_month_coverage_fails_promotion(tmp_path) -> None:
    long, long_cell = _source("long")
    short, short_cell = _source("short")
    def _missing_month(source: StageITargetSpecificInput) -> StageITargetSpecificInput:
        manifest = dict(source.source_manifest)
        contract = dict(manifest["evaluation_month_contract"])
        availability = dict(contract["source_availability"])
        availability["2024-02"] = {"source_available": True}
        contract["source_availability"] = availability
        manifest["evaluation_month_contract"] = contract
        manifest["evaluation_month_contract_sha256"] = canonical_sha256(contract)
        return replace(source, source_manifest=manifest, source_manifest_sha256=canonical_sha256(manifest))
    bundle = StageIAdapterWinnerBundle(cells=(long_cell, short_cell), code_revision="immutable-test")
    with pytest.raises(TargetSpecificOOSError, match="promotion coverage gate failed"):
        run_stage_i_target_specific_oos(
            bundle=bundle, inputs=(_missing_month(long), _missing_month(short)), output_dir=tmp_path / "coverage",
            fit_model=_fit, admission_spec=Causal21dAdmissionSpec(min_reference_rows=20, min_side_reference_rows=4, bins=4),
        )


def _frozen_coverage() -> pd.DataFrame:
    rows = []
    for side in ("long", "short"):
        for month in pd.period_range("2024-01", "2026-12", freq="M").astype(str):
            available = month == "2024-02"
            rows.append({
                "side_name": side, "month": month, "layer": "meta",
                "source_available": available,
                "source_gap_reason": "compact frozen fixture gap" if not available else "",
                "promotion_coverage_status": "pass",
            })
    return pd.DataFrame(rows)


def _frozen_r3_source() -> FrozenR3FinalistInput:
    strict = pd.DataFrame({
        "candidate_id": ["l1", "s1", "l2", "s2"],
        "side_name": ["long", "short", "long", "short"],
        "decision_ts": pd.date_range("2024-02-01", periods=4, freq="D", tz="UTC"),
        "target_valid": True, "strict_oof_available": True,
        "exact_gross_bps": [220.0, 260.0, 180.0, 210.0],
        "exact_net_bps": [120.0, 160.0, 80.0, 110.0],
    })
    admission = strict.loc[:, ["candidate_id", "side_name"]].copy()
    admission["causal_expected_net_bps"] = [130.0, 170.0, 90.0, 120.0]
    admission["admitted"] = True
    return FrozenR3FinalistInput(
        strict_oof_predictions=strict, admission_predictions=admission,
        coverage_audit=_frozen_coverage(),
        strict_oof_manifest={
            "status": "complete", "contract": {"geometry": "TP6_SL4_H12"},
            "files": {"strict_oof_predictions.parquet": "1" * 64},
        },
        admission_manifest={
            "status": "complete", "mapping": "side-local 21-day trailing causal mapping",
            "threshold_bps": 50.0,
            "artifacts": {"legacy/causal_admission_predictions.parquet": {"sha256": "3" * 64}},
        },
        strict_oof_file_sha256="1" * 64, strict_oof_manifest_sha256="2" * 64,
        admission_file_sha256="3" * 64, admission_manifest_sha256="4" * 64,
        coverage_audit_sha256="5" * 64,
    )


def test_frozen_r3_normalizer_is_schema_only_and_comparator_ready(tmp_path) -> None:
    source = _frozen_r3_source()
    manifest = write_frozen_r3_finalist_normalizer(
        source=source, output_dir=tmp_path / "r3", finalist_name="R3",
    )
    assert manifest["normalizer"].startswith("schema-only")
    assert manifest["source_lineage"]["admission_file_sha256"] == "3" * 64
    finalist = load_target_specific_finalist_artifact(tmp_path / "r3", name="R3")
    assert {
        "meta_causal_21d_expected_net_bps", "meta_causal_21d_admitted",
    }.issubset(finalist.predictions)
    assert finalist.predictions.meta_causal_21d_admitted.all()


def test_frozen_r3_normalizer_requires_exact_manifest_declared_source_hashes(tmp_path) -> None:
    source = _frozen_r3_source()
    unrelated = replace(
        source,
        strict_oof_manifest={
            "status": "complete", "contract": {"geometry": "TP6_SL4_H12"},
            "files": {"unrelated_predictions.parquet": "1" * 64},
        },
    )
    with pytest.raises(TargetSpecificOOSError, match="no declared hash"):
        write_frozen_r3_finalist_normalizer(source=unrelated, output_dir=tmp_path / "unrelated")


def test_frozen_r3_normalizer_rejects_mismatched_or_ambiguous_source_manifest_hashes(tmp_path) -> None:
    source = _frozen_r3_source()
    mismatch = replace(
        source,
        admission_manifest={
            **source.admission_manifest,
            "artifacts": {"causal_admission_predictions.parquet": {"sha256": "f" * 64}},
        },
    )
    with pytest.raises(TargetSpecificOOSError, match="does not match supplied ledger"):
        write_frozen_r3_finalist_normalizer(source=mismatch, output_dir=tmp_path / "mismatch")
    ambiguous = replace(
        source,
        strict_oof_manifest={
            **source.strict_oof_manifest,
            "artifacts": {"legacy/strict_oof_predictions.parquet": {"sha256": "1" * 64}},
        },
    )
    with pytest.raises(TargetSpecificOOSError, match="ambiguous declarations"):
        write_frozen_r3_finalist_normalizer(source=ambiguous, output_dir=tmp_path / "ambiguous")


def test_finalist_comparator_cli_requires_manifest_hashes_and_coverage(tmp_path) -> None:
    from scripts.compare_stage_i_target_specific_finalists import main

    source = _frozen_r3_source()
    left, right = tmp_path / "left", tmp_path / "right"
    write_frozen_r3_finalist_normalizer(source=source, output_dir=left, finalist_name="R3")
    write_frozen_r3_finalist_normalizer(source=source, output_dir=right, finalist_name="S")
    for root in (left, right):
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["shared_population_contract_sha256"] = "a" * 64
        manifest_path.write_text(json.dumps(manifest))
    assert main([
        "--finalist", f"R3={left}", "--finalist", f"S={right}",
        "--output-dir", str(tmp_path / "comparison"),
    ]) == 0
    # Byte drift is caught from the manifest before the parquet is trusted.
    with (right / "strict_oof_predictions.parquet").open("ab") as handle:
        handle.write(b"drift")
    with pytest.raises(TargetSpecificOOSError, match="checksum drift"):
        main([
            "--finalist", f"R3={left}", "--finalist", f"S={right}",
            "--output-dir", str(tmp_path / "comparison_drift"),
        ])


def test_finalist_comparison_allows_geometry_specific_economics() -> None:
    common = pd.DataFrame({
        "candidate_id": ["l1", "s1"], "side_name": ["long", "short"],
        "decision_ts": pd.date_range("2024-02-01", periods=2, freq="D", tz="UTC"),
        "target_valid": True, "strict_oof_available": True,
        "exact_gross_bps": [220.0, 230.0], "exact_net_bps": [120.0, 130.0],
        "meta_causal_21d_expected_net_bps": [140.0, 150.0], "meta_causal_21d_admitted": True,
    })
    other_geometry = common.copy()
    other_geometry["exact_gross_bps"] = [300.0, 190.0]
    other_geometry["exact_net_bps"] = [200.0, 90.0]
    score, _ = compare_target_specific_finalists((
        StageITargetSpecificFinalist("S", common, {"declared_target_geometry": "TP6_SL4_H12", "shared_population_contract_sha256": "a" * 64}),
        StageITargetSpecificFinalist("R3", other_geometry, {"declared_target_geometry": "TP3_SL2_H12", "shared_population_contract_sha256": "a" * 64}),
    ))
    assert set(score.finalist) == {"S", "R3"}


def test_target_specific_cli_preflight_reports_independent_layer_policies(tmp_path, capsys) -> None:
    from scripts.run_stage_i_target_specific_oos import main

    input_root, base_root, meta_root = tmp_path / "input", tmp_path / "base", tmp_path / "meta"
    cells = []
    for side in ("long", "short"):
        source, cell = _source(side)
        source_dir = input_root / side
        source_dir.mkdir(parents=True)
        feature_path, contract_path = source_dir / "features.parquet", source_dir / "contract.parquet"
        source.frame.to_parquet(feature_path, index=False)
        source.contract_frame.to_parquet(contract_path, index=False)
        base_dir, meta_dir = base_root / side, meta_root / side
        base_dir.mkdir(parents=True)
        meta_dir.mkdir(parents=True)
        base_manifest = dict(source.base_selector_manifest)
        meta_manifest = dict(source.meta_selector_manifest)
        # Prove the preflight no longer assumes the two layer policies are one
        # value while retaining side-local selector contracts.
        meta_manifest["correlation_policy"] = "pre-mda-spearman-representative"
        (base_dir / "manifest.json").write_text(json.dumps(base_manifest))
        (meta_dir / "manifest.json").write_text(json.dumps(meta_manifest))
        cells.append(replace(
            cell,
            base_selector_manifest_sha256=file_sha256(base_dir / "manifest.json"),
            meta_selector_manifest_sha256=file_sha256(meta_dir / "manifest.json"),
        ))
        source_manifest = dict(source.source_manifest)
        source_manifest.update({
            "base_target_column": source.base_target_column,
            "meta_target_column": source.meta_target_column,
            "n_validation_folds": source.n_validation_folds,
            "min_train_rows": source.min_train_rows,
            "artifact_sha256": {
                "features.parquet": file_sha256(feature_path),
                "contract.parquet": file_sha256(contract_path),
            },
        })
        (source_dir / "manifest.json").write_text(json.dumps(source_manifest))
    bundle = StageIAdapterWinnerBundle(cells=tuple(cells), code_revision="preflight-test")
    winner = tmp_path / "winner.json"
    winner.write_text(json.dumps(bundle.to_dict()))
    assert main([
        "--winner-bundle", str(winner), "--input-root", str(input_root),
        "--base-selector-dir", str(base_root), "--meta-selector-dir", str(meta_root), "--preflight",
    ]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert {row["base_correlation_policy"] for row in receipt["sides"]} == {"grouped-preserve"}
    assert {row["meta_correlation_policy"] for row in receipt["sides"]} == {"pre-mda-spearman-representative"}
    assert receipt["strict_meta_availability"]["model_fit_performed"] is False
    assert receipt["strict_meta_availability"]["per_side"]["long"]["strict_meta_rows"] > 0


def test_preflight_requires_identical_projected_strict_meta_availability() -> None:
    long, long_cell = _source("long")
    short, short_cell = _source("short")
    bundle = StageIAdapterWinnerBundle(cells=(long_cell, short_cell), code_revision="preflight")
    baseline = preflight_strict_meta_availability(bundle, (long, short))
    equal = preflight_strict_meta_availability(bundle, (long, short))
    receipt = validate_preflight_strict_meta_availability_equality({
        "scalar_S": baseline, "ordinal_O": equal,
    })
    assert receipt["status"] == "complete"
    broken_frame = equal[0].copy()
    broken_frame.loc[0, "strict_oof_available"] = not bool(broken_frame.loc[0, "strict_oof_available"])
    required = sorted((
        "candidate_id", "side_name", "decision_ts", "target_valid",
        "base_strict_oof_available", "strict_oof_available",
    ))
    broken_receipt = dict(equal[1])
    broken_receipt["availability_sha256"] = canonical_sha256(
        broken_frame.loc[:, required].sort_values(
            ["side_name", "decision_ts", "candidate_id"], kind="stable",
        ).astype(str).to_dict(orient="records")
    )
    with pytest.raises(TargetSpecificOOSError, match="identical rows/base/meta strict availability"):
        validate_preflight_strict_meta_availability_equality({
            "scalar_S": baseline, "ordinal_O": (broken_frame, broken_receipt),
        })
