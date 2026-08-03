from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_i_causal_admission import Causal21dAdmissionSpec
from extreme_price_movements.stage_i_feature_selection import (
    STAGE_I_ACTIVE_CONTRACTS,
    STAGE_I_META_BASE_OOF_HANDOFF_FEATURES,
)
from extreme_price_movements.stage_i_production_oos import (
    StageIFeatureSelectionReuseException,
    StageIOOSCalendar,
    StageIProductionOOSError,
    StageIProductionWinnerBundle,
    StageISideProductionInput,
    StageIWinnerCell,
    build_stage_i_production_plans,
    run_stage_i_production_oos,
    validate_stage_i_strict_prediction_flags,
)
from extreme_price_movements.stage_i_strict_oof import StageIStrictOOFResult


def _digest(value: dict) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _bundle(*, revision: str = "a" * 40) -> StageIProductionWinnerBundle:
    cells = []
    for contract in STAGE_I_ACTIVE_CONTRACTS:
        source = {"panel": "frozen", "side": contract.side, "version": 1}
        fields = (
            ("base_signal", "base_context")
            if contract.layer == "base"
            else ("meta_context", *STAGE_I_META_BASE_OOF_HANDOFF_FEATURES)
        )
        params = {"n_estimators": 8, "objective": "multiclass", "num_class": 3} if contract.layer == "base" else {"n_estimators": 8, "objective": "huber"}
        selector = {"selector": "frozen", "version": 1, "selected_feature_contract": list(fields), "best_params": params}
        cells.append(StageIWinnerCell(
            contract=contract, selected_feature_names=fields,
            lgbm_params=params,
            selector_manifest=selector, selector_manifest_sha256=_digest(selector),
            source_manifest=source, source_manifest_sha256=_digest(source),
        ))
    return StageIProductionWinnerBundle(
        cells=tuple(cells), code_revision=revision,
        calendar=StageIOOSCalendar("2024-01-01T00:00:00Z", "2026-12-31T23:00:00Z"),
        feature_selection_exception=StageIFeatureSelectionReuseException(
            approved=True, selection_reference_start_utc="2026-05-31T00:00:00Z",
            selection_reference_end_utc="2026-06-30T00:00:00Z",
            rationale="User-approved full-period feature selection reused backward for future runs.",
        ),
    )


def _side_input(side: str, *, start: str = "2024-02-01", n: int = 48) -> StageISideProductionInput:
    signal = pd.date_range(start, periods=n, freq="D", tz="UTC")
    r3 = np.resize(np.array([0, 1, 2], dtype=np.int8), n)
    net = np.where(r3 == 2, 160.0, np.where(r3 == 1, 15.0, -190.0)).astype(np.float32)
    panel_manifest = {"panel": "frozen", "side": side, "version": 1}
    return StageISideProductionInput(
        side=side, candidate_ids=[f"{side}-{i}" for i in range(n)],
        symbols=np.where(np.arange(n) % 2, "ETH/USD:USD", "BTC/USD:USD"),
        signal_close_timestamps=signal, decision_timestamps=signal + pd.Timedelta(hours=1),
        label_available_timestamps=signal + pd.Timedelta(hours=13),
        frame=pd.DataFrame({"base_signal": r3.astype(np.float32), "base_context": np.arange(n, dtype=np.float32), "meta_context": np.arange(n, dtype=np.float32) / n}),
        r3_target=r3, exact_net_bps=net, exact_gross_bps=net + 100.0,
        panel_manifest=panel_manifest, panel_manifest_sha256=_digest(panel_manifest),
        n_validation_folds=2, min_train_rows=4,
    )


def _fake_generate(plan):
    n = len(plan.frame)
    r3 = np.asarray(plan.r3_target)
    clear = np.where(r3 == 2, .75, np.where(r3 == 1, .35, .08)).astype(np.float32)
    adverse = np.where(r3 == 0, .75, np.where(r3 == 1, .25, .08)).astype(np.float32)
    weak = 1.0 - clear - adverse
    base = (clear - adverse) * 200.0
    residual = np.linspace(-10.0, 10.0, n, dtype=np.float32)
    available = np.ones(n, dtype=bool)
    available[:4] = False
    residual[~available] = np.nan
    predictions = pd.DataFrame({
        "candidate_id": list(plan.candidate_ids), "candidate_key": [f"{plan.side}::{x}" for x in plan.candidate_ids],
        "side_name": plan.side, "decision_ts": pd.to_datetime(plan.decision_timestamps, utc=True),
        "label_available_ts": pd.to_datetime(plan.label_available_timestamps, utc=True),
        "exact_gross_bps": np.asarray(plan.exact_gross_bps), "exact_net_bps": np.asarray(plan.exact_net_bps),
        "base_strict_oof_available": np.ones(n, dtype=bool),
        "strict_oof_available": available, "r3_p_adverse": adverse, "r3_p_weak": weak,
        "r3_p_clear": clear, "r3_opportunity_score": clear - adverse,
        "prequential_base_expected_net_bps": base, "residual_oof_bps": residual,
        "reconstructed_expected_net_bps": base + residual,
    })
    provenance = pd.DataFrame({"strict_prior_resolved": [True], "side": [plan.side], "layer": ["base_r3"]})
    return StageIStrictOOFResult(plan.side, predictions, provenance, {"side": plan.side}, {"side": plan.side})


def _fake_writer(results, output, *, admission_spec, admission_reference_results=None):
    output.mkdir(parents=True)
    prediction = pd.concat([result.predictions for result in results], ignore_index=True)
    prediction.to_parquet(output / "raw_oof_predictions.parquet", index=False)
    return {"schema": "fake", "status": "complete", "rows": {"input": len(prediction)}}


def test_bundle_builds_exact_side_plans_with_explicit_reused_backward_lineage() -> None:
    bundle = _bundle()
    plans, identity = build_stage_i_production_plans(bundle, [_side_input("long"), _side_input("short")])
    assert [plan.side for plan in plans] == ["long", "short"]
    assert all(tuple(plan.meta_feature_names[-5:]) == STAGE_I_META_BASE_OOF_HANDOFF_FEATURES for plan in plans)
    assert {"candidate_id", "symbol", "signal_close_ts", "decision_ts", "side_name", "source_label_available_ts", "source_exact_gross_bps", "source_exact_net_bps"}.issubset(identity["long"].columns)
    assert bundle.feature_selection_exception.disposition.endswith("exception")


def test_plan_rejects_noncanonical_signal_to_decision_timing() -> None:
    invalid = _side_input("long")
    invalid = replace(invalid, decision_timestamps=pd.to_datetime(invalid.signal_close_timestamps, utc=True))
    with pytest.raises(StageIProductionOOSError, match=r"signal-close \+1h"):
        build_stage_i_production_plans(_bundle(), [invalid, _side_input("short")])


def test_atomic_output_is_restart_safe_and_has_global_attribution(tmp_path) -> None:
    bundle = _bundle()
    output = tmp_path / "stage_i"
    manifest = run_stage_i_production_oos(
        bundle, [_side_input("long"), _side_input("short")], output,
        admission_spec=Causal21dAdmissionSpec(min_reference_rows=4, bins=4),
        generate=_fake_generate, write_strict=_fake_writer,
    )
    assert manifest["winner_bundle_sha256"] == bundle.sha256
    assert set(manifest["selected_input_content_sha256"]) == {"long", "short"}
    assert all(len(value) == 64 for value in manifest["selected_input_content_sha256"].values())
    report = pd.read_parquet(output / "detailed_base_meta_21d_pooled_global_metrics.parquet")
    assert {"base", "meta_residual"}.issubset(report.layer)
    assert {"month", "week", "side", "month_side", "week_side", "worst_month", "worst_week", "worst_month_side", "worst_week_side"}.issubset(set(report.scope))
    pooled = report[report.row_type.eq("pooled_global")]
    assert pooled.selection.eq("pooled_global_once_no_timestamp_or_side_rerank").all()
    raw = pooled[pooled.admission_mode.eq("without_21d_admission")]
    admitted = pooled[pooled.admission_mode.eq("with_side_local_causal_21d_admission")]
    comparable = raw.merge(
        admitted[["layer", "top_fraction", "requested_selected_rows"]],
        on=["layer", "top_fraction"], suffixes=("_raw", "_admitted"),
    )
    assert not comparable.empty
    assert comparable.requested_selected_rows_raw.to_numpy().tolist() == comparable.requested_selected_rows_admitted.to_numpy().tolist()
    assert {"rank_ic_net", "calibration_slope", "calibration_intercept_bps"}.issubset(report.columns)
    reused = run_stage_i_production_oos(
        bundle, [_side_input("long"), _side_input("short")], output,
        admission_spec=Causal21dAdmissionSpec(min_reference_rows=4, bins=4),
        generate=lambda _: (_ for _ in ()).throw(AssertionError("must not regenerate")), write_strict=_fake_writer,
    )
    assert reused["restart_status"] == "reused_verified_immutable_artifact"
    changed = _side_input("long")
    changed_frame = changed.frame.copy()
    changed_frame.loc[0, "base_signal"] += 1.0
    with pytest.raises(StageIProductionOOSError, match="different selected input content"):
        run_stage_i_production_oos(
            bundle, [replace(changed, frame=changed_frame), _side_input("short")], output,
            admission_spec=Causal21dAdmissionSpec(min_reference_rows=4, bins=4),
            generate=lambda _: (_ for _ in ()).throw(AssertionError("must not regenerate")),
            write_strict=_fake_writer,
        )
    with pytest.raises(FileExistsError, match="different frozen winner bundle"):
        run_stage_i_production_oos(
            _bundle(revision="b" * 40), [_side_input("long"), _side_input("short")], output,
            admission_spec=Causal21dAdmissionSpec(min_reference_rows=4, bins=4),
            generate=_fake_generate, write_strict=_fake_writer,
        )


def test_zero_admission_is_an_explicit_comparable_view(tmp_path) -> None:
    output = tmp_path / "zero_admission"
    run_stage_i_production_oos(
        _bundle(), [_side_input("long"), _side_input("short")], output,
        # More support than this deliberately small OOS fixture can provide.
        admission_spec=Causal21dAdmissionSpec(min_reference_rows=1_000, bins=4),
        generate=_fake_generate, write_strict=_fake_writer,
    )
    report = pd.read_parquet(output / "detailed_base_meta_21d_pooled_global_metrics.parquet")
    zero = report[
        report.admission_mode.eq("with_side_local_causal_21d_admission")
        & report.row_type.eq("pooled_global")
    ]
    assert len(zero) == 8  # 2 layers x four predeclared tails.
    assert zero.eligible_rows.eq(0).all()
    assert zero.selected_rows.eq(0).all()
    assert zero.requested_selected_rows.notna().all()


def test_precalendar_history_trains_but_is_not_persisted_or_evaluated(tmp_path) -> None:
    output = tmp_path / "with_history"
    inputs = [_side_input("long", start="2023-12-01", n=80), _side_input("short", start="2023-12-01", n=80)]
    run_stage_i_production_oos(
        _bundle(), inputs, output, admission_spec=Causal21dAdmissionSpec(min_reference_rows=4, bins=4),
        generate=_fake_generate, write_strict=_fake_writer,
    )
    full = pd.read_parquet(output / "full_history_raw_oof_predictions.parquet")
    evaluation = pd.read_parquet(output / "evaluation_window_raw_oof_predictions.parquet")
    base_strict = pd.read_parquet(output / "evaluation_window_base_strict_oof_predictions.parquet")
    meta_strict = pd.read_parquet(output / "evaluation_window_meta_strict_oof_predictions.parquet")
    assert pd.to_datetime(full.signal_close_ts, utc=True).min() < pd.Timestamp("2024-01-01", tz="UTC")
    assert pd.to_datetime(evaluation.signal_close_ts, utc=True).min() >= pd.Timestamp("2024-01-01", tz="UTC")
    assert pd.to_datetime(base_strict.signal_close_ts, utc=True).min() >= pd.Timestamp("2024-01-01", tz="UTC")
    assert base_strict.base_strict_oof_available.all()
    assert meta_strict.strict_oof_available.all()
    assert len(full) > len(evaluation) > 0
    admission = pd.read_parquet(output / "base_causal_21d_admission_audit.parquet")
    january = admission[pd.to_datetime(admission.snapshot_utc, utc=True).ge(pd.Timestamp("2024-01-01", tz="UTC"))]
    assert not january.empty
    assert january.reference_rows.max() > 0
    assert january.used_pre_evaluation_reference_history.all()


def test_strict_flags_are_explicit_and_non_oof_scores_fail_closed() -> None:
    source = _fake_generate(build_stage_i_production_plans(_bundle(), [_side_input("long"), _side_input("short")])[0][0]).predictions
    source.loc[0, "base_strict_oof_available"] = False
    with pytest.raises(StageIProductionOOSError, match="base non-OOF"):
        validate_stage_i_strict_prediction_flags(source)


def test_generator_must_preserve_full_population_and_canonical_keys(tmp_path) -> None:
    def omitted(plan):
        result = _fake_generate(plan)
        return replace(result, predictions=result.predictions.iloc[1:].reset_index(drop=True))

    with pytest.raises(StageIProductionOOSError, match="complete frozen candidate population"):
        run_stage_i_production_oos(
            _bundle(), [_side_input("long"), _side_input("short")], tmp_path / "omitted",
            generate=omitted, write_strict=_fake_writer,
        )

    def changed_key(plan):
        result = _fake_generate(plan)
        result.predictions.loc[0, "candidate_key"] = "spoofed"
        return result

    with pytest.raises(StageIProductionOOSError, match="side::candidate_id"):
        run_stage_i_production_oos(
            _bundle(), [_side_input("long"), _side_input("short")], tmp_path / "changed_key",
            generate=changed_key, write_strict=_fake_writer,
        )


def test_calendar_cannot_be_a_narrow_2024_to_2026_sliver() -> None:
    with pytest.raises(StageIProductionOOSError, match="cover the 2024--2026"):
        StageIOOSCalendar("2024-12-01T00:00:00Z", "2026-01-02T00:00:00Z")


def test_exact_source_labels_panel_hash_and_selected_feature_audit_are_enforced(tmp_path) -> None:
    output = tmp_path / "audit"
    run_stage_i_production_oos(
        _bundle(), [_side_input("long"), _side_input("short")], output,
        admission_spec=Causal21dAdmissionSpec(min_reference_rows=4, bins=4),
        generate=_fake_generate, write_strict=_fake_writer,
    )
    audit = pd.read_parquet(output / "selected_feature_coverage_audit.parquet")
    assert audit.status.eq("pass").all()
    assert audit.generated_same_side_r3_handoff.any()
    bad_input = replace(_side_input("long"), panel_manifest_sha256="0" * 64)
    with pytest.raises(StageIProductionOOSError, match="panel/input manifest"):
        build_stage_i_production_plans(_bundle(), [bad_input, _side_input("short")])
    different_panel = {"panel": "different", "side": "long", "version": 1}
    mismatched_input = replace(
        _side_input("long"), panel_manifest=different_panel,
        panel_manifest_sha256=_digest(different_panel),
    )
    with pytest.raises(StageIProductionOOSError, match="winner source manifests"):
        build_stage_i_production_plans(_bundle(), [mismatched_input, _side_input("short")])

    def wrong_labels(plan):
        result = _fake_generate(plan)
        result.predictions.loc[0, "exact_net_bps"] += 1.0
        return result

    with pytest.raises(StageIProductionOOSError, match="exact_net_bps differs"):
        run_stage_i_production_oos(
            _bundle(), [_side_input("long"), _side_input("short")], tmp_path / "bad_labels",
            admission_spec=Causal21dAdmissionSpec(min_reference_rows=4, bins=4),
            generate=wrong_labels, write_strict=_fake_writer,
        )


def test_winner_requires_selector_list_params_and_layer_runtime_semantics() -> None:
    contract = STAGE_I_ACTIVE_CONTRACTS[0]
    fields = ("base_signal", "base_context")
    params = {"objective": "huber", "n_estimators": 8}
    selector = {"selected_feature_contract": list(fields), "best_params": params}
    with pytest.raises(StageIProductionOOSError, match="objective=multiclass"):
        StageIWinnerCell(
            contract=contract, selected_feature_names=fields, lgbm_params=params,
            selector_manifest=selector, selector_manifest_sha256=_digest(selector),
            source_manifest={"source": 1}, source_manifest_sha256=_digest({"source": 1}),
        )

    hpo_only = {"n_estimators": 8}
    selector = {"selected_feature_contract": list(fields), "best_params": hpo_only}
    accepted = StageIWinnerCell(
        contract=contract, selected_feature_names=fields,
        lgbm_params={**hpo_only, "objective": "multiclass", "num_class": 3},
        selector_manifest=selector, selector_manifest_sha256=_digest(selector),
        source_manifest={"source": 1}, source_manifest_sha256=_digest({"source": 1}),
    )
    assert accepted.lgbm_params["objective"] == "multiclass"
