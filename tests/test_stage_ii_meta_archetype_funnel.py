from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_i_causal_admission import Causal21dAdmissionSpec
from extreme_price_movements.stage_ii_meta_archetype_funnel import (
    StageIIDiscoveryCandidate,
    StageIIFunnelError,
    StageIIFunnelSpec,
    StageIIMetaPredictionRequest,
    StageIIMetaPredictionResult,
    _control_selection_summary,
    run_stage_ii_meta_archetype_funnel,
)
from extreme_price_movements.stage_ii_meta_archetypes import StageIIMetaArchetypeConfig


def _frame(rows: int = 1800) -> pd.DataFrame:
    rng = np.random.default_rng(616)
    decision = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    side = np.where(np.arange(rows) % 2, "long", "short")
    regime = rng.normal(size=rows).astype("float32")
    trust = rng.normal(size=rows).astype("float32")
    base = (20 * regime + 4 * trust).astype("float32")
    mode = np.where(regime > .65, 1.0, np.where(regime < -.65, -1.0, 0.0))
    net = (base + mode * 120 + np.where(side == "long", 10, -8) + rng.normal(0, 4, rows)).astype("float32")
    clear = np.clip(.45 + .20 * regime, .05, .85)
    adverse = np.clip(.18 - .08 * regime, .03, .40)
    weak = 1.0 - clear - adverse
    return pd.DataFrame({
        "candidate_id": [f"c-{i}" for i in range(rows)],
        "symbol": np.where(np.arange(rows) % 3, "A", "B"),
        "decision_ts": decision,
        "label_available_ts": decision + pd.Timedelta(hours=13),
        "side_name": side,
        "exact_net_bps": net,
        "exact_gross_bps": net + 100.0,
        "prequential_base_expected_net_bps": base,
        "r3_p_adverse": adverse.astype("float32"),
        "r3_p_weak": weak.astype("float32"),
        "r3_p_clear": clear.astype("float32"),
        "r3_is_strict_oof": True,
        "r3_source_side": side,
        "r3_fit_end_ts": pd.Timestamp("2024-12-31T23:00:00Z"),
        "r3_score_semantics": "same_side_direct_strict_oof_probabilities_without_conversion",
        "r3_oof_fold_id": 0,
        "base_map_is_prequential": True,
        "base_map_source_side": side,
        "base_map_max_label_available_ts": decision - pd.Timedelta(hours=1),
        "realised_mfe_atr": (mode + rng.normal(0, .15, rows)).astype("float32"),
        "regime": regime,
        "trust": trust,
        "meta_signal": (regime + .1 * trust).astype("float32"),
    })


def _candidate() -> StageIIDiscoveryCandidate:
    return StageIIDiscoveryCandidate(
        candidate_id="path_mfe_k3",
        config=StageIIMetaArchetypeConfig(
            path_descriptor_cols=("realised_mfe_atr",), components=3,
            min_side_rows=120, min_component_rows=25, min_train_rows=300,
            oof_folds=3, random_state=17,
        ),
        causal_feature_cols=("regime", "trust", "prequential_base_expected_net_bps"),
    )


def _spec(**changes: object) -> StageIIFunnelSpec:
    values: dict[str, object] = dict(
        meta_feature_cols=("meta_signal",), min_oof_rows=200,
        min_economic_separation_bps=5.0, min_stable_fold_fraction=0.2,
        max_symbol_share=0.90,
        admission_spec=Causal21dAdmissionSpec(min_reference_rows=4, bins=4, net_floor_bps=-10_000),
        base_r3_oof_fold_catalog=({
            "fold_id": 0,
            "train_max_label_available_ts": "2024-12-31T00:00:00Z",
            "validation_start_ts": "2025-01-01T00:00:00Z",
        },),
    )
    values.update(changes)
    return StageIIFunnelSpec(**values)


def _predictor(request: StageIIMetaPredictionRequest) -> StageIIMetaPredictionResult:
    # Test the production contract rather than modelling economics: the output
    # is explicitly a raw residual, and input path columns must be absent.
    assert "realised_mfe_atr" not in request.frame
    assert request.base_r3_probability_columns == ("r3_p_adverse", "r3_p_weak", "r3_p_clear")
    np.testing.assert_allclose(request.base_r3_probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert request.base_handoff_provenance["r3_strict_oof"] is True
    assert {"r3_p_adverse", "r3_p_weak", "r3_p_clear"}.issubset(request.frame.columns)
    score = request.frame.iloc[:, 0].to_numpy(np.float32) * .25
    return StageIIMetaPredictionResult(
        candidate_ids=request.candidate_ids,
        predicted_residual_bps=score,
        oof_fold_ids=np.zeros(len(request.candidate_ids), dtype=np.int32),
        provenance={
            "strict_oof": True, "layer": "meta_residual",
            "score_semantics": "raw_predicted_residual_bps",
            "base_model_changed": False,
            "base_handoff": request.base_handoff_provenance,
            "feature_columns": request.feature_columns,
            "folds": [{
                "fold_id": 0,
                "train_max_label_available_ts": "2025-01-01T13:00:00Z",
                "validation_start_ts": "2025-01-02T00:00:00Z",
            }],
        },
    )


def test_stage_ii_runs_bounded_discovery_then_identical_meta_only_controls() -> None:
    result = run_stage_ii_meta_archetype_funnel(
        _frame(), spec=_spec(), candidates=[_candidate()], meta_oof_predictor=_predictor,
    )
    assert result.selected_candidate_id == "path_mfe_k3"
    assert result.selected_control_arm in {"none", "soft_memberships", "prior", "both"}
    assert result.candidate_audit.disposition.eq("retained").sum() == 1
    controls = result.control_metrics.loc[result.control_metrics.arm.isin(["none", "soft_memberships", "prior", "both"])]
    assert set(controls.arm) == {"none", "soft_memberships", "prior", "both"}
    assert controls.candidate_rows.nunique() == 1
    tails = result.control_metrics.loc[result.control_metrics.admission_scope.notna()]
    assert set(tails.admission_scope) == {"without_21d_admission", "with_21d_side_local_admission"}
    assert set(tails.top_fraction) == {.01, .05, .10, .20}
    assert (tails.ranking_basis == "pooled_global_after_common_bps_mapping_no_side_or_month_rerank").all()
    assert {"side", "month", "month_side"}.issubset(set(result.selected_contributions.scope))
    assert result.manifest["hard_routing"] is False
    assert result.manifest["local_experts"] is False
    assert result.manifest["base_model_changed"] is False
    assert result.manifest["base_handoff"]["r3_semantics"] == "same_side_direct_strict_oof_probabilities_without_conversion"
    tails = result.control_metrics.loc[result.control_metrics.record_type.eq("pooled_tail_metric")]
    assert not tails.duplicated(["arm", "admission_scope", "top_fraction"]).any()


def test_stage_ii_rejects_unbounded_component_candidate_before_any_model_is_called() -> None:
    invalid = StageIIDiscoveryCandidate(
        candidate_id="k8", config=StageIIMetaArchetypeConfig(path_descriptor_cols=("realised_mfe_atr",), components=8),
        causal_feature_cols=("regime",),
    )
    with pytest.raises(StageIIFunnelError, match="3, 4, 5 or 6"):
        run_stage_ii_meta_archetype_funnel(_frame(), spec=_spec(), candidates=[invalid], meta_oof_predictor=_predictor)


def test_realised_path_coordinates_cannot_leak_into_the_meta_control() -> None:
    with pytest.raises(StageIIFunnelError, match="realised path descriptors"):
        run_stage_ii_meta_archetype_funnel(
            _frame(), spec=_spec(meta_feature_cols=("meta_signal", "realised_mfe_atr")),
            candidates=[_candidate()], meta_oof_predictor=_predictor,
        )


def test_missing_or_non_simplex_direct_r3_base_handoff_fails_closed() -> None:
    broken = _frame()
    broken.loc[0, "r3_p_clear"] = np.float32(.10)
    with pytest.raises(StageIIFunnelError, match="probability simplex"):
        run_stage_ii_meta_archetype_funnel(
            broken, spec=_spec(), candidates=[_candidate()], meta_oof_predictor=_predictor,
        )


def test_stage_ii_rejected_candidate_does_not_silently_fall_back_to_base_or_meta_run() -> None:
    result = run_stage_ii_meta_archetype_funnel(
        _frame(), spec=_spec(min_economic_separation_bps=99_999.0), candidates=[_candidate()], meta_oof_predictor=_predictor,
    )
    assert result.selected_candidate_id is None
    assert result.selected_control_arm is None
    assert result.manifest["decision"] == "NO_STAGE_II_ARCHETYPE_ADVANCES"
    assert result.candidate_audit.disposition.eq("diagnostic").all()


def test_meta_predictor_cannot_relabel_a_mapped_or_base_score_as_residual() -> None:
    def invalid(request: StageIIMetaPredictionRequest) -> StageIIMetaPredictionResult:
        result = _predictor(request)
        return StageIIMetaPredictionResult(
            result.candidate_ids, result.predicted_residual_bps, result.oof_fold_ids,
            {**result.provenance, "score_semantics": "causal_21d_mapped_expected_net_bps"},
        )
    with pytest.raises(StageIIFunnelError, match="unconverted predicted residual"):
        run_stage_ii_meta_archetype_funnel(_frame(), spec=_spec(), candidates=[_candidate()], meta_oof_predictor=invalid)


def test_meta_oof_row_assignments_must_fall_in_their_declared_validation_fold() -> None:
    def invalid(request: StageIIMetaPredictionRequest) -> StageIIMetaPredictionResult:
        result = _predictor(request)
        return StageIIMetaPredictionResult(
            result.candidate_ids, result.predicted_residual_bps, result.oof_fold_ids,
            {**result.provenance, "folds": [{
                "fold_id": 0,
                "train_max_label_available_ts": "2025-01-01T13:00:00Z",
                "validation_start_ts": "2030-01-01T00:00:00Z",
            }]},
        )
    with pytest.raises(StageIIFunnelError, match="falls outside"):
        run_stage_ii_meta_archetype_funnel(_frame(), spec=_spec(), candidates=[_candidate()], meta_oof_predictor=invalid)


def test_control_selection_prioritises_worst_month_and_side_over_aggregate_top_tail() -> None:
    def metric(arm: str, net: float) -> dict[str, object]:
        return {
            "arm": arm, "admission_scope": "without_21d_admission", "top_fraction": .10,
            "original_population_rows": 100, "net_bps_per_trade": net,
        }
    metrics = pd.DataFrame([
        metric("aggregate", 120.0), metric("aggregate", 120.0) | {"admission_scope": "with_21d_side_local_admission"},
        metric("robust", 100.0), metric("robust", 100.0) | {"admission_scope": "with_21d_side_local_admission"},
    ])
    contributions = pd.DataFrame([
        # Aggregate winner has a very poor selected month/side contribution.
        {"arm": "aggregate", "admission_scope": scope, "top_fraction": .10, "scope": kind, "selected_rows": 10, "net_bps_per_trade": value}
        for scope in ("without_21d_admission", "with_21d_side_local_admission")
        for kind, value in (("month", -80.0), ("side", -30.0))
    ] + [
        {"arm": "robust", "admission_scope": scope, "top_fraction": .10, "scope": kind, "selected_rows": 10, "net_bps_per_trade": value}
        for scope in ("without_21d_admission", "with_21d_side_local_admission")
        for kind, value in (("month", 10.0), ("side", 5.0))
    ])
    aggregate = _control_selection_summary(metrics.loc[metrics.arm.eq("aggregate")], contributions.loc[contributions.arm.eq("aggregate")], arm="aggregate", spec=_spec())
    robust = _control_selection_summary(metrics.loc[metrics.arm.eq("robust")], contributions.loc[contributions.arm.eq("robust")], arm="robust", spec=_spec())
    winner = pd.DataFrame([aggregate, robust]).sort_values(
        ["selection_worst_month_net_bps_per_trade", "selection_worst_side_net_bps_per_trade", "selection_mean_top_tail_net_bps_per_trade", "selection_max_side_share", "arm"],
        ascending=[False, False, False, True, True], kind="stable",
    ).iloc[0]
    assert bool(aggregate["selection_eligible"])
    assert winner.arm == "robust"
