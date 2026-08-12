from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_i_causal_admission import Causal21dAdmissionSpec
from extreme_price_movements.stage_iv_native_artifact_runner import (
    FROZEN_MODEL_SERIALIZATION_FORMAT,
    _metrics,
    _globally_bounded_joint_book,
    NativeBasePrediction,
    StageIVNativeCell,
    StageIVNativeFrozenArtifact,
    StageIVNativeFrozenOOSPlan,
    StageIVNativePlan,
    StageIVNativeRunnerError,
    StageIVNativeRunnerSpec,
    generate_stage_iv_native_side_oof,
    run_stage_iv_native_frozen_oos,
    run_stage_iv_native_artifact_sweep,
)
from extreme_price_movements.stage_i_target_specific_oos import _fit_direct_correctness
from extreme_price_movements.stage_i_target_adapter import canonical_sha256
from extreme_price_movements.stage_iv_native_materializer import (
    FROZEN_OOS_MATERIALIZER_SCHEMA,
    StageIVNativeMaterializationError,
    load_stage_iv_native_frozen_oos_launch,
)


class _NativeModel:
    def __init__(self, column: str, shift: float) -> None:
        self.column = column
        self.shift = shift

    def predict_native(self, frame: pd.DataFrame) -> NativeBasePrediction:
        score = np.clip(frame[self.column].to_numpy(float) + self.shift, -0.95, 0.95)
        adverse = np.clip((1.0 - score) / 3.0, 0.02, 0.90)
        clear = np.clip((1.0 + score) / 3.0, 0.02, 0.90)
        weak = np.maximum(1.0 - adverse - clear, 0.02)
        states = np.column_stack([adverse, weak, clear])
        states /= states.sum(axis=1, keepdims=True)
        return NativeBasePrediction(score, states)


class _MetaModel:
    classes_ = np.asarray([0, 1, 2], dtype=np.int8)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        score = frame.base_raw_score.to_numpy(float)
        logits = np.column_stack([-score, np.ones(len(score)) * 0.2, score])
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits)
        return probability / probability.sum(axis=1, keepdims=True)


def _fitters(records: list[tuple[str, tuple[str, ...]]]):
    def base(frame, target, _weight, layer, _params):
        records.append((layer, tuple(frame.columns)))
        raw = [name for name in frame if not name.startswith("__stage_iv_")]
        return _NativeModel(raw[0], float(np.mean(target)) * 0.00001)

    def meta(frame, _labels, _weight, layer, _params):
        records.append((layer, tuple(frame.columns)))
        return _MetaModel()

    return base, meta


def _plan(
    side: str, fraction: float, *, route: str = "both",
    burns: tuple[int, int, int] = (24, 12, 10),
) -> StageIVNativePlan:
    timestamps, per_timestamp = 300, 2
    n = timestamps * per_timestamp
    rng = np.random.default_rng(12 if side == "long" else 37)
    decision = pd.date_range("2024-01-01", periods=timestamps, freq="2h", tz="UTC").repeat(2)
    rank = np.tile(np.asarray([-0.65, 0.65]), timestamps)
    broad = np.clip(rank + rng.normal(0.0, 0.08, n), -0.9, 0.9)
    tail = np.clip(rank + rng.normal(0.0, 0.05, n), -0.9, 0.9)
    context = rng.normal(size=n)
    net = 170.0 * tail + 18.0 * context + rng.normal(0.0, 15.0, n)
    broad_burn, tail_burn, meta_burn = burns
    return StageIVNativePlan(
        side=side,
        candidate_ids=[f"{side}-{index}" for index in range(n)],
        symbols=np.where(np.arange(n) % 3, "BTC", "ETH"),
        frame=pd.DataFrame({"broad_signal": broad, "tail_signal": tail, "context": context}),
        base_target=tail,
        exact_net_bps=net,
        decision_timestamps=decision,
        label_available_timestamps=decision + pd.Timedelta(hours=13),
        broad_feature_names=("broad_signal",),
        tail_feature_names=("tail_signal",),
        meta_feature_names=("context",),
        tail_fraction=fraction,
        broad_min_train_rows=broad_burn,
        tail_min_train_rows=tail_burn,
        meta_min_train_rows=meta_burn,
        min_handoff_history_rows=10,
        n_validation_folds=3,
        broad_output_route=route,
    )


def _cell(cell_id: str, fraction: float, route: str, burns: tuple[int, int, int]):
    return StageIVNativeCell(
        cell_id=cell_id,
        plans=(
            _plan("long", fraction, route=route, burns=burns),
            _plan("short", fraction, route=route, burns=burns),
        ),
        source_lineage={"ledger": sha256(b"frozen-ledger").hexdigest()},
    )


def _spec() -> StageIVNativeRunnerSpec:
    return StageIVNativeRunnerSpec(
        control_cell_id="x20_burnA_both",
        admission_spec=Causal21dAdmissionSpec(
            min_reference_rows=20, min_side_reference_rows=8, bins=4,
        ),
        top_fractions=(0.10, 0.20),
        selection_top_fraction=0.10,
    )


def test_native_chain_uses_tail_direct_fq3_and_never_bps_residual() -> None:
    records: list[tuple[str, tuple[str, ...]]] = []
    base, meta = _fitters(records)
    prediction, folds, summary = generate_stage_iv_native_side_oof(
        _plan("long", 0.30), base_fitter=base, meta_fitter=meta,
    )
    assert prediction.joint_meta_strict_oof_available.any()
    assert "meta_same_side_residual_oof_score" not in prediction
    assert "meta_reconstructed_expected_net_bps" not in prediction
    meta_columns = next(columns for layer, columns in records if layer == "meta")
    assert "base_raw_score" in meta_columns
    assert "base_state_p0" in meta_columns
    assert "base_output_entropy" in meta_columns
    assert "__stage_iv_broad_native_score" in meta_columns
    scored = prediction.loc[prediction.joint_meta_strict_oof_available]
    assert np.allclose(
        scored[[f"meta_p_error_tercile_{index}" for index in range(3)]].sum(axis=1),
        1.0,
    )
    assert folds.loc[folds.layer.eq("meta"), "target_semantics"].eq(
        "same_side_direct_base_output_correctness_q33_v1"
    ).all()
    assert summary["legacy_mapped_bps_residual"] is False


def test_native_plan_rejects_mapped_bps_meta_input() -> None:
    plan = _plan("short", 0.30)
    plan.frame["prequential_expected_net_bps"] = 10.0
    plan = StageIVNativePlan(
        **{**plan.__dict__, "meta_feature_names": ("prequential_expected_net_bps",)}
    )
    base, meta = _fitters([])
    with pytest.raises(StageIVNativeRunnerError, match="mapped, outcome, or future-path"):
        generate_stage_iv_native_side_oof(plan, base_fitter=base, meta_fitter=meta)


def test_native_plan_rejects_path_outcomes_and_noncanonical_cost() -> None:
    plan = _plan("short", 0.30)
    plan.frame["future_path_return"] = 1.0
    unsafe = StageIVNativePlan(
        **{**plan.__dict__, "broad_feature_names": ("future_path_return",)}
    )
    base, meta = _fitters([])
    with pytest.raises(StageIVNativeRunnerError, match="mapped, outcome, or future-path"):
        generate_stage_iv_native_side_oof(unsafe, base_fitter=base, meta_fitter=meta)
    wrong_cost = StageIVNativePlan(**{**plan.__dict__, "cost_bps": 99.0})
    with pytest.raises(StageIVNativeRunnerError, match="fixed 100-bps cost"):
        generate_stage_iv_native_side_oof(wrong_cost, base_fitter=base, meta_fitter=meta)


def test_artifact_sweep_is_atomic_sequential_and_joint_meta_only(tmp_path: Path) -> None:
    records: list[tuple[str, tuple[str, ...]]] = []
    base, meta = _fitters(records)
    cells = (
        _cell("x20_burnA_both", 0.20, "both", (24, 12, 10)),
        _cell("x30_burnB_tail", 0.30, "tail", (28, 14, 11)),
        _cell("x40_burnC_meta", 0.40, "meta", (32, 16, 12)),
        _cell("x50_burnA_neither", 0.50, "neither", (24, 12, 10)),
    )
    output = tmp_path / "stage_iv_native"
    result = run_stage_iv_native_artifact_sweep(
        cells, output_directory=output, base_fitter=base, meta_fitter=meta,
        spec=_spec(),
    )
    assert result.winner["selection_layer"] == "joint_meta_only"
    assert result.winner["cell_id"] == "x20_burnA_both"
    assert result.winner["control_cell_id"] == "x20_burnA_both"
    assert result.winner["decision"] == "NO_STAGE_IV_ADVANCE"
    pooled = result.metrics.loc[result.metrics.scope.eq("pooled_global")]
    assert set(pooled.layer) == {"broad_base", "tail_base", "joint_meta"}
    assert pooled.loc[pooled.layer.ne("joint_meta"), "diagnostic_only"].all()
    assert pooled.loc[pooled.layer.eq("joint_meta"), "promotable"].all()
    assert set(pooled.admission_scope) == {"without_admission", "with_admission"}
    gate = pd.read_csv(output / "joint_meta_winner_comparison.csv")
    assert gate.global_top_k_rows.nunique() == 1
    assert gate.global_top_k_rows.iloc[0] == int(np.ceil(
        0.10 * gate.common_globally_scored_rows.iloc[0]
    ))
    assert (output / "checksums.json").is_file()
    assert len(list((output / "checkpoints").iterdir())) == 4
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["legacy_mapped_bps_residual"] is False
    assert manifest["execution"] == "explicit_cells_sequential_with_atomic_cell_checkpoints"
    assert manifest["top_k_denominator"] == (
        "common_globally_scored_population_before_admission"
    )
    assert manifest["reported_global_top_fractions"] == [.01, .05, .1, .2]
    assert {
        "per_cell_side_month_zero_complete_metrics.parquet",
        "fold_windows_and_label_cutoffs.parquet",
        "map_admission_support.parquet",
        "feature_contracts_and_source_hashes.parquet",
    }.issubset({path.name for path in output.iterdir()})
    zero_complete = pd.read_parquet(output / "per_cell_side_month_zero_complete_metrics.parquet")
    assert {0.01, 0.05, 0.10, 0.20}.issubset(set(zero_complete.top_fraction))
    assert "zero_selected" in zero_complete
    with pytest.raises(StageIVNativeRunnerError, match="already exists"):
        run_stage_iv_native_artifact_sweep(
            cells, output_directory=output, base_fitter=base, meta_fitter=meta,
            spec=_spec(),
        )


def test_admission_never_changes_the_common_global_top_k_denominator() -> None:
    rows = 100
    frame = pd.DataFrame({
        "candidate_key": [f"long::{index}" for index in range(rows)],
        "side_name": "long",
        "decision_ts": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
        "exact_net_bps": np.linspace(-20.0, 80.0, rows),
        "exact_gross_bps": np.linspace(80.0, 180.0, rows),
        "joint_meta_causal_21d_common_bps": np.arange(rows, dtype=float),
        # Only 20 rows are admitted. A favorable admitted-subset denominator
        # would select 2; the common 100-row denominator must still select 10.
        "joint_meta_causal_21d_admitted": np.arange(rows) >= 80,
    })
    metrics = _metrics(
        frame, cell_id="candidate", layer="joint_meta", top_fractions=(0.10,),
        diagnostic_only=False, common_population_rows=rows,
    )
    larger_support = frame.copy()
    larger_support["joint_meta_causal_21d_admitted"] = np.arange(rows) >= 50
    control_metrics = _metrics(
        larger_support, cell_id="control", layer="joint_meta",
        top_fractions=(0.10,), diagnostic_only=False,
        common_population_rows=rows,
    )
    admitted = metrics.loc[
        metrics.scope.eq("pooled_global")
        & metrics.admission_scope.eq("with_admission")
    ].iloc[0]
    assert admitted.common_globally_scored_rows == 100
    assert admitted.global_top_k_rows == 10
    assert admitted.eligible_rows == 20
    assert admitted.selected_global_rows == 10
    control_admitted = control_metrics.loc[
        control_metrics.scope.eq("pooled_global")
        & control_metrics.admission_scope.eq("with_admission")
    ].iloc[0]
    assert control_admitted.eligible_rows == 50
    assert control_admitted.global_top_k_rows == 10
    assert control_admitted.selected_global_rows == 10
    common = set(frame.candidate_key)
    candidate_book, candidate_support = _globally_bounded_joint_book(
        frame, common_keys=common, global_top_k_rows=10, require_admission=True,
    )
    control_book, control_support = _globally_bounded_joint_book(
        larger_support, common_keys=common, global_top_k_rows=10,
        require_admission=True,
    )
    assert (candidate_support, control_support) == (20, 50)
    assert len(candidate_book) == len(control_book) == 10


def test_sweep_fails_closed_when_declared_control_is_absent(tmp_path: Path) -> None:
    base, meta = _fitters([])
    cells = (
        _cell("x20", 0.20, "both", (24, 12, 10)),
        _cell("x30", 0.30, "tail", (28, 14, 11)),
        _cell("x40", 0.40, "meta", (32, 16, 12)),
        _cell("x50", 0.50, "neither", (24, 12, 10)),
    )
    spec = StageIVNativeRunnerSpec(
        control_cell_id="missing_control",
        admission_spec=Causal21dAdmissionSpec(
            min_reference_rows=20, min_side_reference_rows=8, bins=4,
        ),
        top_fractions=(0.10,), selection_top_fraction=0.10,
    )
    with pytest.raises(StageIVNativeRunnerError, match="declared control cell"):
        run_stage_iv_native_artifact_sweep(
            cells, output_directory=tmp_path / "absent_control",
            base_fitter=base, meta_fitter=meta, spec=spec,
        )


def test_sweep_rejects_cells_with_different_frozen_source_lineage(tmp_path: Path) -> None:
    base, meta = _fitters([])
    cells = list((
        _cell("x20", 0.20, "both", (24, 12, 10)),
        _cell("x30", 0.30, "tail", (28, 14, 11)),
        _cell("x40", 0.40, "meta", (32, 16, 12)),
        _cell("x50", 0.50, "neither", (24, 12, 10)),
    ))
    cells[1] = StageIVNativeCell(
        cell_id=cells[1].cell_id, plans=cells[1].plans,
        source_lineage={"ledger": sha256(b"different-ledger").hexdigest()},
    )
    with pytest.raises(StageIVNativeRunnerError, match="identical frozen source lineage"):
        run_stage_iv_native_artifact_sweep(
            cells, output_directory=tmp_path / "lineage_drift", base_fitter=base,
            meta_fitter=meta, spec=StageIVNativeRunnerSpec(
                control_cell_id="x20",
                admission_spec=Causal21dAdmissionSpec(
                    min_reference_rows=20, min_side_reference_rows=8, bins=4,
                ),
                top_fractions=(0.10,), selection_top_fraction=0.10,
            ),
        )


def _frozen_oos_plan(tmp_path: Path, side: str) -> StageIVNativeFrozenOOSPlan:
    import joblib

    source = _plan(side, .30)
    oos_rows = 80
    frame = source.frame.iloc[:oos_rows].copy()
    oos_start = pd.Timestamp("2024-06-01T00:00:00Z")
    decision = pd.date_range(oos_start, periods=oos_rows, freq="2h", tz="UTC")
    history = pd.DataFrame({
        "decision_ts": pd.date_range(oos_start - pd.Timedelta(days=4), periods=24, freq="2h", tz="UTC"),
        "broad_native_score": np.linspace(-.8, .8, 24),
    })
    reference_decision = pd.date_range(oos_start - pd.Timedelta(days=8), periods=40, freq="2h", tz="UTC")
    reference = pd.DataFrame({
        "candidate_key": [f"{side}::ref-{i}" for i in range(40)], "side_name": side,
        "decision_ts": reference_decision,
        "label_available_ts": reference_decision + pd.Timedelta(hours=13),
        "exact_net_bps": np.linspace(-80., 140., 40),
        "joint_meta_native_score": np.linspace(-.7, .7, 40),
    })
    _, state = _fit_direct_correctness(
        reference.exact_net_bps.to_numpy(), reference.joint_meta_native_score.to_numpy(),
        score_domain=(-1., 1.),
    )
    model_root = tmp_path / f"models_{side}"
    model_root.mkdir()
    model_values = {
        "broad_model": _NativeModel("broad_signal", 0.0),
        "tail_model": _NativeModel("tail_signal", 0.0),
        "meta_model": _MetaModel(),
    }
    model_artifacts = {}
    for role, model in model_values.items():
        model_path = model_root / f"{role}.joblib"
        joblib.dump(model, model_path)
        model_artifacts[role] = {
            "path": str(model_path), "sha256": sha256(model_path.read_bytes()).hexdigest(),
            "format": FROZEN_MODEL_SERIALIZATION_FORMAT,
        }
    model_manifest = sha256(json.dumps(model_artifacts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    artifact = StageIVNativeFrozenArtifact(
        artifact_id=f"frozen-{side}", artifact_sha256=sha256(side.encode()).hexdigest(),
        freeze_cutoff_timestamp=oos_start - pd.Timedelta(hours=1), side=side,
        broad_model=model_values["broad_model"], tail_model=model_values["tail_model"],
        meta_model=model_values["meta_model"], direct_fq3_state=state,
        broad_feature_names=("broad_signal",), tail_feature_names=("tail_signal",),
        meta_feature_names=("context",), broad_output_route="both", tail_fraction=.30,
        min_handoff_history_rows=10, score_domain=(-1., 1.),
        pre_oos_handoff_history=history, pre_oos_mapping_reference=reference,
        model_artifacts=model_artifacts, model_artifact_manifest_sha256=model_manifest,
    )
    return StageIVNativeFrozenOOSPlan(
        artifact=artifact, candidate_ids=[f"oos-{i}" for i in range(oos_rows)],
        symbols=np.where(np.arange(oos_rows) % 2, "BTC", "ETH"), frame=frame,
        exact_net_bps=np.linspace(-30., 90., oos_rows), decision_timestamps=decision,
        label_available_timestamps=decision + pd.Timedelta(hours=13),
    )


def test_frozen_oos_replay_uses_only_frozen_models_and_publishes_full_reports(tmp_path: Path) -> None:
    plans = (_frozen_oos_plan(tmp_path, "long"), _frozen_oos_plan(tmp_path, "short"))
    output = tmp_path / "frozen_oos"
    result = run_stage_iv_native_frozen_oos(
        plans, output_directory=output,
        admission_spec=Causal21dAdmissionSpec(min_reference_rows=20, min_side_reference_rows=8, bins=4),
    )
    assert result.manifest["untouched_oos"] is True
    assert result.manifest["refit_forbidden"] is True
    assert set(result.metrics.top_fraction) == {0.01, .05, .10, .20}
    assert {
        "frozen_oos_predictions.parquet", "frozen_oos_global_top_metrics.parquet",
        "frozen_oos_side_month_zero_complete_metrics.parquet",
        "frozen_oos_map_admission_support.parquet", "frozen_oos_windows_and_label_cutoffs.parquet",
        "frozen_oos_feature_contracts_and_source_hashes.json",
    }.issubset({path.name for path in output.iterdir()})
    prediction = pd.read_parquet(output / "frozen_oos_predictions.parquet")
    assert prediction.joint_meta_frozen_oos_available.any()
    assert prediction.direct_fq3_semantics.eq("same_side_direct_base_output_correctness_q33_v1").all()
    assert prediction.frozen_model_artifact_manifest_sha256.notna().all()
    assert result.manifest["serialized_model_loading"]["format"] == FROZEN_MODEL_SERIALIZATION_FORMAT


def test_frozen_oos_rejects_an_artifact_not_frozen_before_the_test_period(tmp_path: Path) -> None:
    plan = _frozen_oos_plan(tmp_path, "long")
    late = StageIVNativeFrozenArtifact(
        **{**plan.artifact.__dict__, "freeze_cutoff_timestamp": plan.decision_timestamps[0]}
    )
    bad = StageIVNativeFrozenOOSPlan(**{**plan.__dict__, "artifact": late})
    with pytest.raises(StageIVNativeRunnerError, match="cutoff must precede"):
        run_stage_iv_native_frozen_oos(
            (bad,), output_directory=tmp_path / "late_artifact",
            admission_spec=Causal21dAdmissionSpec(min_reference_rows=20, min_side_reference_rows=8, bins=4),
        )


def test_frozen_oos_rejects_serialized_model_checksum_drift(tmp_path: Path) -> None:
    plan = _frozen_oos_plan(tmp_path, "long")
    model_path = Path(plan.artifact.model_artifacts["meta_model"]["path"])
    model_path.write_bytes(model_path.read_bytes() + b"drift")
    with pytest.raises(StageIVNativeRunnerError, match="model file SHA-256 drift"):
        run_stage_iv_native_frozen_oos(
            (plan,), output_directory=tmp_path / "drift",
            admission_spec=Causal21dAdmissionSpec(min_reference_rows=20, min_side_reference_rows=8, bins=4),
        )


def test_frozen_oos_rejects_missing_serialized_model_file(tmp_path: Path) -> None:
    plan = _frozen_oos_plan(tmp_path, "long")
    Path(plan.artifact.model_artifacts["tail_model"]["path"]).unlink()
    with pytest.raises(StageIVNativeRunnerError, match="model file is absent"):
        run_stage_iv_native_frozen_oos(
            (plan,), output_directory=tmp_path / "missing",
            admission_spec=Causal21dAdmissionSpec(min_reference_rows=20, min_side_reference_rows=8, bins=4),
        )


def test_frozen_oos_materializer_loads_only_declared_checked_models(tmp_path: Path) -> None:
    plans = {side: _frozen_oos_plan(tmp_path, side) for side in ("long", "short")}
    sides = {}
    for side, plan in plans.items():
        artifact = plan.artifact
        history_path, reference_path, panel_path = (
            tmp_path / f"{side}_history.parquet", tmp_path / f"{side}_reference.parquet", tmp_path / f"{side}_panel.parquet",
        )
        artifact.pre_oos_handoff_history.to_parquet(history_path, index=False)
        artifact.pre_oos_mapping_reference.to_parquet(reference_path, index=False)
        panel = plan.frame.copy()
        panel["candidate_id"], panel["symbol"] = plan.candidate_ids, plan.symbols
        panel["decision_ts"], panel["label_available_ts"], panel["exact_net_bps"] = plan.decision_timestamps, plan.label_available_timestamps, plan.exact_net_bps
        panel.to_parquet(panel_path, index=False)
        feature_sets = {
            "broad_feature_names": list(artifact.broad_feature_names),
            "tail_feature_names": list(artifact.tail_feature_names),
            "meta_feature_names": list(artifact.meta_feature_names),
        }
        payload = {
            "artifact_id": artifact.artifact_id, "side": side,
            "freeze_cutoff_timestamp": str(artifact.freeze_cutoff_timestamp),
            "model_artifact_manifest_sha256": artifact.model_artifact_manifest_sha256,
            "handoff_history_sha256": sha256(history_path.read_bytes()).hexdigest(),
            "mapping_reference_sha256": sha256(reference_path.read_bytes()).hexdigest(),
            "direct_fq3_state": artifact.direct_fq3_state.to_dict(), "feature_sets": feature_sets,
            "broad_output_route": artifact.broad_output_route, "tail_fraction": artifact.tail_fraction,
            "min_handoff_history_rows": artifact.min_handoff_history_rows, "score_domain": list(artifact.score_domain),
        }
        sides[side] = {
            "artifact": {
                "artifact_id": artifact.artifact_id, "artifact_sha256": canonical_sha256(payload),
                "freeze_cutoff_timestamp": str(artifact.freeze_cutoff_timestamp), "side": side,
                "model_artifacts": artifact.model_artifacts,
                "model_artifact_manifest_sha256": artifact.model_artifact_manifest_sha256,
                "pre_oos_handoff_history": {"path": str(history_path), "sha256": payload["handoff_history_sha256"]},
                "pre_oos_mapping_reference": {"path": str(reference_path), "sha256": payload["mapping_reference_sha256"]},
                "direct_fq3_state": artifact.direct_fq3_state.to_dict(), **feature_sets,
                "broad_output_route": artifact.broad_output_route, "tail_fraction": artifact.tail_fraction,
                "min_handoff_history_rows": artifact.min_handoff_history_rows, "score_domain": list(artifact.score_domain),
            },
            "oos_panel": {"path": str(panel_path), "sha256": sha256(panel_path.read_bytes()).hexdigest()},
            "columns": {"candidate_id": "candidate_id", "symbol": "symbol", "decision_ts": "decision_ts", "label_available_ts": "label_available_ts", "exact_net_bps": "exact_net_bps"},
        }
    spec_path = tmp_path / "frozen_spec.json"
    spec_path.write_text(json.dumps({
        "schema": FROZEN_OOS_MATERIALIZER_SCHEMA,
        "admission_spec": {"min_reference_rows": 20, "min_side_reference_rows": 8, "bins": 4},
        "sides": sides,
    }))
    launch = load_stage_iv_native_frozen_oos_launch(spec_path)
    assert launch.launch_manifest["frozen_only"] is True
    assert len(launch.plans) == 2
    assert launch.plans[0].artifact.model_artifacts["meta_model"]["format"] == FROZEN_MODEL_SERIALIZATION_FORMAT
    Path(sides["long"]["artifact"]["model_artifacts"]["meta_model"]["path"]).write_bytes(b"drift")
    with pytest.raises(StageIVNativeMaterializationError, match="frozen model SHA-256 drift"):
        load_stage_iv_native_frozen_oos_launch(spec_path)
