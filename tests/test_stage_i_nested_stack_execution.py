from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.stage_i_nested_feature_challenger import NestedFeatureChallengePlan, NestedFeatureSet
from extreme_price_movements.stage_i_nested_lgbm_hooks import FixedLGBMContract, fold_local_meta_feature_selector
from extreme_price_movements.stage_i_nested_stack_execution import GuardedMetaArmSpec, NestedStackConfig, NestedStackInput, execute_matched_nested_stack


def _plan() -> NestedFeatureChallengePlan:
    sets = []
    for name, features in (("automatic_sparse", ("x1",)), ("top20", ("x1", "x2"))):
        sets.append(NestedFeatureSet("long", name, None, features, (), {f: i + 1 for i, f in enumerate(features)}, {f: f for f in features}, {f: 1 for f in features}, {"selected_automatic_sparse": len(features)}, name))
    return NestedFeatureChallengePlan("long", "a" * 64, "b" * 64, "audit.csv", (), (), {}, tuple(sets))


def _data() -> NestedStackInput:
    n = 72
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    frame = pd.DataFrame({"candidate_id": np.arange(n), "__ts__": ts, "__symbol__": ["BTC"] * n, "side_name": ["long"] * n, "decision_ts": ts + pd.Timedelta(hours=1), "label_available_ts": ts + pd.Timedelta(hours=13), "r3_class": np.arange(n) % 3, "exact_net_bps": np.linspace(-120, 160, n), "x1": np.arange(n), "x2": np.arange(n) * 2, "regime_context": np.arange(n) % 2})
    return NestedStackInput("long", frame, ("x1", "x2"), ("regime_context",))


def _base(train, y, valid, feature_set):
    raw = valid.iloc[:, 0].to_numpy(float)
    clear = 0.2 + 0.5 * ((raw - raw.min()) / max(1.0, raw.max() - raw.min()))
    adverse = 0.9 - clear
    return np.column_stack([adverse, np.full(len(valid), 0.1), clear])


def _meta(train, y, weight, valid, spec):
    assert not any("prequential" in column or "expected_net" in column for column in train.columns)
    if spec.family == "ordinal": return np.tile([0.1, 0.2, 0.3, 0.4], (len(valid), 1))
    if spec.family == "quantile_ordinal_residual": return np.tile([0.2, 0.6, 0.2], (len(valid), 1))
    return np.full(len(valid), 0.4)


def test_adapter_refits_matched_chronological_stack_with_direct_probability_handoff() -> None:
    result = execute_matched_nested_stack(_data(), _plan(), base_predictor=_base, meta_predictor=_meta, meta_arms=(GuardedMetaArmSpec("reliable", "reliability"), GuardedMetaArmSpec("veto", "overestimate_veto"), GuardedMetaArmSpec("ordinal", "ordinal"), GuardedMetaArmSpec("tercile", "quantile_ordinal_residual", residual_clip_bps=200.0), GuardedMetaArmSpec("clipped", "clipped_residual")), meta_feature_selector=fold_local_meta_feature_selector(FixedLGBMContract(base_params={}, meta_params={}, meta_feature_cap=8)), config=NestedStackConfig(n_validation_folds=4, min_base_train_rows=12, min_meta_train_rows=4))
    assert set(result.base_outputs) == {"automatic_sparse", "top20"}
    base = result.base_outputs["automatic_sparse"]
    assert {"r3_p_adverse", "r3_p_weak", "r3_p_clear", "r3_opportunity_score", "base_r3_entropy"}.issubset(base.columns)
    assert not any("prequential" in column or "expected_net" in column for column in base.columns)
    assert result.metrics.layer.eq("base").any() and result.metrics.layer.eq("meta").any()
    assert {0.01, 0.05, 0.10} == set(result.metrics.top_fraction)
    assert {"worst_month_net_bps_per_trade", "worst_fold_net_bps_per_trade", "side_attribution"}.issubset(result.metrics.columns)
    meta = result.meta_outputs[("top20", "veto")]
    assert 0 < len(meta.frame) < len(base)
    assert meta.fold_provenance.base_candidate_fraction.eq(0.30).all()
    assert meta.fold_provenance.train_rows.lt(
        meta.fold_provenance.train_pool_rows
    ).all()
    assert meta.fold_provenance.validation_rows.lt(
        meta.fold_provenance.validation_pool_rows
    ).all()
    assert meta.fold_provenance.base_candidate_ranking_scope.str.contains(
        "never_per_timestamp"
    ).all()
    assert meta.fold_provenance.meta_feature_count.le(8).all()
    assert meta.fold_provenance.selected_meta_features.map(lambda value: "r3_p_clear" in value).all()
    tercile = result.meta_outputs[("automatic_sparse", "tercile")]
    assert {
        "meta_p_lower_residual_tercile", "meta_p_middle_residual_tercile",
        "meta_p_upper_residual_tercile", "meta_prior_p_lower_residual_tercile",
        "meta_prior_p_middle_residual_tercile",
        "meta_prior_p_upper_residual_tercile"
    }.issubset(tercile.frame.columns)
    assert tercile.fold_provenance.target_residual_q33_bps.notna().all()
    assert tercile.fold_provenance.target_residual_q67_bps.notna().all()
    assert tercile.fold_provenance.target_class_location_method.eq(
        "training_class_winsorized_mean_q05_q95_shrunk_to_global_winsorized_mean"
    ).all()
    assert {
        "target_prior_accuracy", "target_prior_log_loss",
        "target_prior_multiclass_brier", "target_prior_rps",
        "target_log_loss_skill", "target_brier_skill", "target_rps_skill",
        "target_balanced_accuracy", "target_ordinal_expected_class_spearman",
    }.issubset(tercile.fold_provenance.columns)
    assert "target_brier" in result.metrics.columns
