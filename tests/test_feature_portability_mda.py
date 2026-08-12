from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.feature_portability_mda import (
    ChronologicalTransport,
    FeaturePortabilityMDAError,
    FrozenR3ModelContract,
    R3CostContract,
    _class_to_common_net_bps_map,
    _fit_predict_by_side,
    _pooled_top10_net_bps,
    materialize_feature_portability_f4_evidence,
)
from extreme_price_movements.feature_portability_selection import REQUIRED_EVIDENCE_COLUMNS


def _panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(41)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    for day in range(220):
        for side_index, side in enumerate(("long", "short")):
            signal = np.sin(day / 8.0) + 0.25 * side_index + rng.normal(scale=0.06)
            rows.append({
                "candidate_id": f"{day}-{side}", "decision_ts": start + pd.Timedelta(days=day),
                "label_available_ts": start + pd.Timedelta(days=day + 1), "side_name": side,
                "net_bps": 30.0 * signal + rng.normal(scale=1.0), "base_x": rng.normal(), "raw": signal,
                "raw__causal_rank_w90": signal, "raw__causal_rank_w180": signal * 0.9,
                "raw__causal_robust_z_w90": signal * 1.05, "raw__causal_robust_z_w180": signal * 0.95,
                "raw__causal_delta_p4": signal * 1.1, "raw__causal_delta_p24": signal * 0.8,
            })
    panel = pd.DataFrame(rows)
    panel["gross_bps"] = panel["net_bps"] + 100.0
    panel["r3_class"] = np.select(
        [panel["net_bps"].lt(-5.0), panel["net_bps"].lt(12.0)], [0, 1], default=2
    ).astype(np.int8)
    panel["frozen_r3_sample_weight"] = 1.0
    return panel


def _contracts() -> dict[str, dict[str, list[str]]]:
    transforms = [
        "raw__causal_rank_w90", "raw__causal_rank_w180",
        "raw__causal_robust_z_w90", "raw__causal_robust_z_w180",
        "raw__causal_delta_p4", "raw__causal_delta_p24",
    ]
    return {
        "F0_control": {"long": ["base_x"], "short": ["base_x"]},
        "F3_plus_relative": {"long": ["raw", *transforms], "short": ["raw", *transforms]},
    }


def _contracts_with_safe_second_source() -> dict[str, dict[str, list[str]]]:
    suffixes = (
        "__causal_rank_w90", "__causal_rank_w180", "__causal_robust_z_w90",
        "__causal_robust_z_w180", "__causal_delta_p4", "__causal_delta_p24",
    )
    f3 = ["raw", "safe_raw", *(f"{source}{suffix}" for suffix in suffixes for source in ("raw", "safe_raw"))]
    return {
        "F0_control": {"long": ["base_x"], "short": ["base_x"]},
        "F3_plus_relative": {"long": f3, "short": f3},
    }


def _transport() -> ChronologicalTransport:
    return ChronologicalTransport(
        "development_a", "2024-01-01", "2024-06-01", "2024-08-01"
    )


def _transports() -> tuple[ChronologicalTransport, ChronologicalTransport]:
    # F4 is a cross-transport selection.  These overlapping test windows are
    # deliberately synthetic but each has predecessor chronology and no final
    # November surface.
    return (
        _transport(),
        ChronologicalTransport("development_b", "2024-01-01", "2024-04-01", "2024-08-01"),
    )


def _r3_cost() -> R3CostContract:
    return R3CostContract(
        class_column="r3_class", gross_bps_column="gross_bps", net_bps_column="net_bps",
        expected_cost_bps=100.0, sample_weight_column="frozen_r3_sample_weight",
    )


def _r3_model() -> FrozenR3ModelContract:
    return FrozenR3ModelContract(
        model_id="frozen_actual_r3_test", params={"objective": "multiclass", "num_class": 3}, random_seed=7,
    )


def _frozen_r3_callback(
    train_features: np.ndarray,
    train_classes: np.ndarray,
    train_sample_weight: np.ndarray,
    eval_features: np.ndarray,
    *,
    seed: int,
    model_contract: FrozenR3ModelContract,
) -> np.ndarray:
    """A deterministic stand-in for a supplied, already-frozen R3 classifier.

    It receives only R3 labels/weights and feature matrices: realised net bps
    cannot enter this callback as a regression target.
    """
    assert set(train_classes).issubset({0, 1, 2})
    assert np.isfinite(train_sample_weight).all() and (train_sample_weight > 0).all()
    assert model_contract.model_id == "frozen_actual_r3_test"
    signal = eval_features.mean(axis=1)
    logits = np.column_stack((-signal, np.zeros(len(signal)), signal))
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    return probability / probability.sum(axis=1, keepdims=True)


def _result():
    return materialize_feature_portability_f4_evidence(
        _panel(), representation_features=_contracts(), control_representation="F0_control",
        f3_representation="F3_plus_relative", transports=_transports(), inner_folds=2,
        r3_cost=_r3_cost(), r3_model=_r3_model(), r3_fit_predict=_frozen_r3_callback,
    )


def test_audits_actual_transformed_fields_and_emits_exact_selector_schema() -> None:
    result = _result()
    assert list(result.evidence.columns) == [
        "representation", "transport", "feature_count", "coverage", "incremental_top10_net_bps",
        "transport_mda_bps", "development_stage", "chronological_verified", "global_ranking_verified",
        "ranking_scope", "model_hpo_performed", "incremental_vs_f3_top10_net_bps",
        "full_f3_control_eligible",
    ]
    assert REQUIRED_EVIDENCE_COLUMNS.issubset(result.evidence.columns)
    coverage = result.transformed_coverage
    assert set(coverage["feature"]) == {
        "raw__causal_rank_w90", "raw__causal_rank_w180",
        "raw__causal_robust_z_w90", "raw__causal_robust_z_w180",
        "raw__causal_delta_p4", "raw__causal_delta_p24",
    }
    assert coverage["is_actual_f3_transform"].all()
    assert coverage["passes_99pct_coverage"].all()
    assert result.evidence["ranking_scope"].eq("pooled_global").all()
    assert not result.evidence["model_hpo_performed"].any()
    assert {"F3_plus_relative", "F4_compact_top01", "F4_compact_top02", "F4_compact_top03"} == set(result.evidence["representation"])
    assert result.compact_contracts["final_november_oos_consumed"] is False
    assert set(result.compact_contracts["representations"]) == {"F4_compact_top01", "F4_compact_top02", "F4_compact_top03"}
    assert result.source_intersection_coverage["selected_cross_transport_source_intersection"].all()
    assert result.manifest["r3_class_to_common_bps_map"].startswith("fold_train_only")


def test_grouped_mda_is_fold_local_and_label_resolved_before_boundary() -> None:
    result = _result()
    folds = result.fold_mda
    assert folds["labels_resolved_before_fold_evaluation"].all()
    assert (pd.to_datetime(folds["train_max_label_available_ts"], utc=True) < pd.to_datetime(folds["fold_evaluation_start"], utc=True)).all()
    assert folds["ranking_scope"].eq("pooled_global").all()
    assert folds["permutation_style"].eq(
        "joint_row_shuffle_by_side_of_actual_f3_transforms_on_prefiltered_complete_candidates"
    ).all()
    assert (folds["evaluation_rows_after_contract_completeness"] <= folds["evaluation_rows_before_contract_completeness"]).all()
    assert folds["evaluation_rows"].eq(folds["evaluation_rows_after_contract_completeness"]).all()
    assert folds.filter(like="train_common_net_bps").notna().all().all()
    assert result.manifest["final_november_oos_consumed"] is False


def test_global_top10_is_one_pooled_book_not_side_local() -> None:
    # A per-side top-10 would take one long and one short in this tiny example;
    # the actual pooled top-10 takes only the highest common-bps candidate.
    surface = pd.DataFrame({
        "candidate_id": ["l1", "l2", "s1", "s2"], "side_name": ["long", "long", "short", "short"],
        "score_common_net_bps": [100.0, 1.0, 99.0, 0.0], "net_bps": [10.0, -20.0, 30.0, -30.0],
    })
    net, trades = _pooled_top10_net_bps(surface, target_column="net_bps")
    assert trades == 1
    assert net == pytest.approx(10.0)


def test_r3_scores_use_train_only_class_map_and_not_direct_net_regression() -> None:
    panel = _panel()
    cost = _r3_cost()
    train = panel.loc[panel.decision_ts.lt(pd.Timestamp("2024-06-01", tz="UTC")) & panel.label_available_ts.lt(pd.Timestamp("2024-06-01", tz="UTC"))]
    evaluate = panel.loc[(panel.decision_ts.ge(pd.Timestamp("2024-06-01", tz="UTC"))) & (panel.decision_ts.lt(pd.Timestamp("2024-06-10", tz="UTC")))].copy()
    # Deliberately absurd evaluation outcomes must not alter prediction scores:
    # they are used only after ranking to measure the economic tail.
    evaluate["net_bps"] = 999_999.0
    evaluate["gross_bps"] = 1_000_099.0
    scores = _fit_predict_by_side(
        train, evaluate, contract=_contracts()["F3_plus_relative"], r3_cost=cost,
        r3_fit_predict=_frozen_r3_callback, r3_model=_r3_model(), seed=7,
    )
    long_train = train.loc[train.side_name.eq("long")]
    expected_map = _class_to_common_net_bps_map(long_train, r3_cost=cost)
    long_eval = evaluate.loc[evaluate.side_name.eq("long")]
    probability = _frozen_r3_callback(
        long_train.loc[:, _contracts()["F3_plus_relative"]["long"]].to_numpy(float),
        long_train["r3_class"].to_numpy(np.int8), long_train["frozen_r3_sample_weight"].to_numpy(float),
        long_eval.loc[:, _contracts()["F3_plus_relative"]["long"]].to_numpy(float), seed=7, model_contract=_r3_model(),
    )
    actual = scores.loc[scores.side_name.eq("long")].sort_values("candidate_id")
    expected = probability @ expected_map
    assert np.allclose(actual["score_common_net_bps"].to_numpy(), expected[np.argsort(long_eval.candidate_id.to_numpy())])
    assert expected_map[2] == pytest.approx(long_train.loc[long_train.r3_class.eq(2), "gross_bps"].mean() - 100.0)


def test_no_direct_net_fallback_is_available() -> None:
    with pytest.raises(FeaturePortabilityMDAError, match="explicit frozen R3"):
        materialize_feature_portability_f4_evidence(
            _panel(), representation_features=_contracts(), control_representation="F0_control",
            f3_representation="F3_plus_relative", transports=_transports(),
            r3_cost=_r3_cost(), r3_model=_r3_model(), r3_fit_predict=None,  # type: ignore[arg-type]
        )


def test_final_november_oos_is_rejected() -> None:
    with pytest.raises(FeaturePortabilityMDAError, match="final November OOS"):
        ChronologicalTransport("final_oos", "2024-01-01", "2024-11-01", "2024-12-01")


def test_grouped_mda_prefilters_sparse_transforms_before_permutation() -> None:
    panel = _panel()
    # Sparse values remain inside the 99%-coverage contract, but moving them
    # by permutation must not change the control/permuted candidate universe.
    panel.loc[[200, 201], "raw__causal_delta_p24"] = np.nan
    result = materialize_feature_portability_f4_evidence(
        panel, representation_features=_contracts(), control_representation="F0_control",
        f3_representation="F3_plus_relative", transports=_transports(), inner_folds=2,
        r3_cost=_r3_cost(), r3_model=_r3_model(), r3_fit_predict=_frozen_r3_callback,
    )
    folds = result.fold_mda
    assert (folds["evaluation_rows_after_contract_completeness"] < folds["evaluation_rows_before_contract_completeness"]).any()
    assert folds["evaluation_rows"].eq(folds["evaluation_rows_after_contract_completeness"]).all()


def test_ineligible_full_f3_is_diagnostic_only_and_f4_uses_safe_source_intersection() -> None:
    panel = _panel()
    suffixes = (
        "__causal_rank_w90", "__causal_rank_w180", "__causal_robust_z_w90",
        "__causal_robust_z_w180", "__causal_delta_p4", "__causal_delta_p24",
    )
    for suffix in suffixes:
        panel[f"safe_raw{suffix}"] = panel[f"raw{suffix}"] * 0.7
    panel["safe_raw"] = panel["raw"] * 0.7
    # Make one full-F3 source unusable in both transport evaluation windows.
    # The safe source still gives each side an exact, cross-transport F4 list.
    panel.loc[panel.decision_ts.ge(pd.Timestamp("2024-05-01", tz="UTC")), "raw__causal_delta_p24"] = np.nan
    result = materialize_feature_portability_f4_evidence(
        panel, representation_features=_contracts_with_safe_second_source(), control_representation="F0_control",
        f3_representation="F3_plus_relative", transports=_transports(), inner_folds=2,
        r3_cost=_r3_cost(), r3_model=_r3_model(), r3_fit_predict=_frozen_r3_callback,
    )
    assert result.manifest["full_f3_diagnostic"]["eligible"] is False
    assert "F3_plus_relative" not in set(result.evidence["representation"])
    assert not result.evidence["full_f3_control_eligible"].any()
    assert result.evidence["incremental_vs_f3_top10_net_bps"].isna().all()
    selected = result.source_intersection_coverage.loc[
        result.source_intersection_coverage["selected_cross_transport_source_intersection"]
    ]
    assert set(selected["source_field"]) == {"safe_raw"}
    assert result.compact_contracts["coverage_safe_source_intersection"] == {
        "long": ["safe_raw"], "short": ["safe_raw"],
    }
