from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import extreme_price_movements.stage_i_model_hpo as stage_i_hpo

from extreme_price_movements.lgbm_pipeline import (
    _aggregate_mda_feature_audit,
    _stage_i_adaptive_depth_summary,
    _stage_i_depth_dependent_survivors,
    _stage_i_vectorized_coarse_gate,
    _stage_i_second_pass_feature_names,
    _stage_i_targeted_zero_use_prescreen_evidence,
)
from extreme_price_movements.stage_i_model_hpo import (
    StageILightGBMDatasetCache,
    _fit_stage_i_model,
    run_stage_i_model_hpo,
)


class _SignalModel:
    classes_ = np.asarray([0, 1, 2], dtype=np.int8)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        signal = np.tanh(frame.iloc[:, 0].to_numpy(float))
        return np.column_stack([
            0.3 - 0.1 * signal,
            np.full(len(frame), 0.4),
            0.3 + 0.1 * signal,
        ])


def test_hpo_successive_halving_is_deterministic_and_full_evidence_for_winner() -> None:
    fit_calls: list[int] = []

    def fake_fit(x, _y, _weight, *, classifier, params, **_kwargs):
        assert classifier and params["objective"] == "multiclass"
        fit_calls.append(len(x))
        return _SignalModel()

    timestamps = pd.date_range("2022-01-01", periods=900, freq="D", tz="UTC")
    signal = np.sin(np.arange(len(timestamps)) / 13.0)
    target = np.resize(np.asarray([0, 1, 2], dtype=np.int8), len(timestamps))
    result = run_stage_i_model_hpo(
        pd.DataFrame({"signal": signal}),
        target,
        selected_feature_names=["signal"],
        candidate_ids=[f"c-{i:04d}" for i in range(len(timestamps))],
        exact_net_bps=np.where(target == 2, 175.0, np.where(target == 0, -180.0, 5.0)),
        decision_timestamps=timestamps,
        label_available_timestamps=timestamps + pd.Timedelta(hours=12),
        side="long",
        layer="base",
        hpo_trials=9,
        hpo_patience=9,
        n_validation_folds=3,
        min_train_rows=20,
        fit_model=fake_fit,
    )
    assert result.hpo_schedule is not None
    assert result.hpo_schedule["fold_budgets"] == [3, 3, 3]
    assert result.hpo_schedule["training_row_fractions"] == [0.25, 0.5, 1.0]
    assert result.hpo_schedule["tree_fractions"] == [0.35, 0.65, 1.0]
    assert result.actual_trials == 9
    assert result.completed_trials == 1
    assert result.stop_reason == "deterministic_successive_halving_completed"
    assert len(result.fold_audit) == 3
    assert len(result.hpo_schedule_sha256) == 64
    assert len(result.hpo_request_sha256) == 64
    # Every rung covers all three eras. Earlier rungs are cheaper through
    # time-spread rows and fewer trees rather than biased earliest-fold-only
    # evidence; the winner then receives a full three-fold OOF regeneration.
    assert len(fit_calls) == 42
    finalist = [row for row in result.trial_audit if row["status"] == "complete_full_budget"]
    assert len(finalist) == 1
    assert [rung["fold_budget"] for rung in finalist[0]["rung_audit"]] == [3, 3, 3]
    assert all(rung["fold_ids"] == [0, 1, 2] for rung in finalist[0]["rung_audit"])
    assert [rung["training_row_fraction"] for rung in finalist[0]["rung_audit"]] == [0.25, 0.5, 1.0]
    assert min(fit_calls[:27]) < max(fit_calls[-6:])


def test_halving_checkpoint_reuses_only_exact_completed_native_rungs(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restart avoids fits already completed, but always regenerates OOF."""
    calls: list[int] = []

    def fake_evaluate_params(*, frame, available, params, validation_blocks, **_kwargs):
        calls.append(len(frame))
        n = len(frame)
        raw = np.linspace(-0.2, 0.2, n, dtype=np.float32)
        probability = np.column_stack([
            np.full(n, 0.3, dtype=np.float32),
            np.full(n, 0.4, dtype=np.float32),
            np.full(n, 0.3, dtype=np.float32),
        ])
        rows = [{
            "validation_max_label_available_utc": pd.Timestamp(available.max()).isoformat(),
        } for _ in validation_blocks]
        # Different proposals are still ranked deterministically.
        return raw, probability, float(params["n_estimators"]), rows, {"score": float(params["n_estimators"])}

    monkeypatch.setattr(stage_i_hpo, "_evaluate_params", fake_evaluate_params)
    timestamps = pd.date_range("2022-01-01", periods=900, freq="D", tz="UTC")
    target = np.resize(np.asarray([0, 1, 2], dtype=np.int8), len(timestamps))
    kwargs = dict(
        selected_feature_names=["signal"],
        candidate_ids=[f"c-{i:04d}" for i in range(len(timestamps))],
        exact_net_bps=np.where(target == 2, 175.0, -180.0),
        decision_timestamps=timestamps,
        label_available_timestamps=timestamps + pd.Timedelta(hours=12),
        side="long", layer="base", hpo_trials=9, hpo_patience=9,
        n_validation_folds=3, min_train_rows=20,
        successive_halving_checkpoint_dir=tmp_path / "halving",
    )
    frame = pd.DataFrame({"signal": np.sin(np.arange(len(timestamps)) / 13.0)})
    first = run_stage_i_model_hpo(frame, target, **kwargs)
    first_calls = len(calls)
    second = run_stage_i_model_hpo(frame, target, **kwargs)
    assert first.best_params == second.best_params
    assert len(calls) == first_calls + 1  # frozen winner OOF regeneration
    checkpoint = second.feasibility_contract["successive_halving_checkpoint"]
    assert checkpoint["hits"] == 13
    assert checkpoint["misses"] == 0
    assert checkpoint["writes"] == 0
    assert (tmp_path / "halving" / "halving_rungs.json").is_file()

    drifted = frame.copy()
    drifted.loc[0, "signal"] += 0.01
    with pytest.raises(stage_i_hpo.StageIModelHPOError, match="checkpoint lineage drift"):
        run_stage_i_model_hpo(drifted, target, **kwargs)


def test_cached_lightgbm_dataset_is_prediction_identical_to_sklearn_fit() -> None:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        rng.normal(size=(600, 5)).astype(np.float32),
        columns=[f"f{i}" for i in range(5)],
    )
    target = np.resize(np.asarray([0, 1, 2], dtype=np.int8), len(frame))
    weight = np.linspace(0.75, 1.25, len(frame), dtype=np.float32)
    params = {
        "objective": "multiclass", "num_class": 3,
        "n_estimators": 80, "learning_rate": 0.03,
        "max_depth": 4, "num_leaves": 16, "min_child_samples": 20,
        "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
        "reg_alpha": 1.0, "reg_lambda": 5.0, "max_bin": 63,
        "random_state": 19, "n_jobs": 2, "verbosity": -1,
    }
    reference = _fit_stage_i_model(
        frame, target, weight, classifier=True, params=params,
    )
    cache = StageILightGBMDatasetCache(max_entries=4)
    first = cache.fit(frame, target, weight, classifier=True, params=params)
    second = cache.fit(frame, target, weight, classifier=True, params=params)
    np.testing.assert_array_equal(reference.predict_proba(frame), first.predict_proba(frame))
    np.testing.assert_array_equal(first.predict_proba(frame), second.predict_proba(frame))
    assert cache.audit()["misses"] == 1
    assert cache.audit()["hits"] == 1


def test_cross_cohort_adaptive_depth_clear_harmful_useful_and_borderline() -> None:
    harmful = _stage_i_adaptive_depth_summary(
        {"c0": -0.3, "c1": -0.2, "c2": -0.1},
        lower=-0.3, upper=-0.05, phantom_threshold=0.02,
    )
    useful = _stage_i_adaptive_depth_summary(
        {"c0": 0.2, "c1": 0.1, "c2": 0.3},
        lower=0.08, upper=0.32, phantom_threshold=0.02,
    )
    borderline = _stage_i_adaptive_depth_summary(
        {"c0": 0.2, "c1": -0.1, "c2": 0.05},
        lower=-0.04, upper=0.16, phantom_threshold=0.02,
    )
    warmup = _stage_i_adaptive_depth_summary(
        {"c2": -0.2}, lower=-0.3, upper=-0.1, phantom_threshold=0.02,
    )
    assert harmful["stop_reason"] == "conclusive_harmful_all_cohorts"
    assert useful["stop_reason"] == "conclusive_useful_positive_every_cohort"
    assert borderline["stop_reason"] == "borderline_or_regime_unstable_full_repeat_budget"
    assert warmup["stop_reason"] == "insufficient_evaluable_cohorts_after_warmup"
    assert warmup["cohort_signs"] == {"c2": "negative"}


def test_depth_dependent_pruning_uses_tiered_round_gate_and_protects_context() -> None:
    names = [f"plain_{i:03d}" for i in range(236)] + [
        "causal_regime_transition", "archetype_support", "base_trust_score",
        "required_base_handoff",
    ]
    stats = pd.DataFrame({
        "feature": names,
        "hard_drop": False,
        "confidence_label": "borderline",
        "mda_evaluated": True,
        "mda_n_repeats": 12,
        "unavailable_warmup_cohort_count": 0,
        "adaptive_depth_stop_reason": "borderline_or_regime_unstable_full_repeat_budget",
        "mda_lower_95": np.linspace(-0.2, 0.2, len(names)),
        "feature_score": np.linspace(0.0, 1.0, len(names)),
    })
    stats.loc[stats.feature.eq("plain_000"), "unavailable_warmup_cohort_count"] = 2
    stats.loc[stats.feature.eq("causal_regime_transition"), "confidence_label"] = "harmful"
    selected, audit = _stage_i_depth_dependent_survivors(
        stats, names, protected={"required_base_handoff"},
        context_features={"causal_regime_transition", "archetype_support", "base_trust_score"},
        context_contract_source="fixture_context_contract_v1",
        context_reasons={
            "causal_regime_transition": "regime",
            "archetype_support": "archetype",
            "base_trust_score": "trust",
        },
    )
    assert audit["mode"] == "coarse_70pct_above_200"
    assert len(selected) == 168
    assert "plain_000" in selected  # causal warm-up is not negative evidence
    assert "archetype_support" in selected
    assert "base_trust_score" in selected
    assert "required_base_handoff" in selected
    assert "causal_regime_transition" not in selected  # context can leave only on adverse evidence
    assert audit["context_contract_source"] == "fixture_context_contract_v1"
    assert audit["context_reasons"]["base_trust_score"] == "trust"

    conservative, conservative_audit = _stage_i_depth_dependent_survivors(
        stats.iloc[:200], names[:200], protected=set(),
    )
    assert len(conservative) == 160
    assert conservative_audit["mode"] == "coarse_80pct_to_120"

    floor_selected, floor_audit = _stage_i_depth_dependent_survivors(
        stats.iloc[:120], names[:120], protected=set(),
    )
    assert floor_selected == names[:120]
    assert floor_audit["mode"] == "conservative_at_or_below_floor"


def test_cross_cohort_second_pass_remeasures_every_early_stopped_era() -> None:
    first = pd.DataFrame({
        "feature": ["mixed"] * 3 + ["stable"] * 3,
        "mda_cohort_id": ["early", "middle", "late"] * 2,
        "mda_feature_evaluable": True,
        "mda_mean": [0.30, -0.20, 0.02, 0.20, 0.25, 0.30],
        "n_repeats": [3, 3, 12, 3, 3, 3],
        "shadow_null_threshold": 0.01,
        "split_count": 1,
        "confidence_label": "borderline",
        "reason": "first_pass",
        "method": "fixture",
    })
    cfg = {
        "confidence_level": 0.95,
        "min_effect_size": 0.0,
        "max_repeats": 12,
        "stage_i_adaptive_min_evaluable_cohorts": 3,
        "stage_i_adaptive_two_pass_enforce": True,
        "decision_default_for_borderline": "keep",
    }
    assert _stage_i_second_pass_feature_names(
        ["mixed", "stable"], first, cfg=cfg
    ) == ("mixed",)
    with pytest.raises(RuntimeError, match="lacks full second-pass repeats"):
        _aggregate_mda_feature_audit(["mixed", "stable"], first, cfg=cfg)
    repaired = first.copy()
    repaired.loc[repaired.feature.eq("mixed"), "n_repeats"] = 12
    result = _aggregate_mda_feature_audit(["mixed", "stable"], repaired, cfg=cfg)
    row = result.set_index("feature").loc["mixed"]
    assert row["adaptive_depth_stop_reason"] == "borderline_or_regime_unstable_full_repeat_budget"
    assert set(__import__("json").loads(row["adaptive_depth_actual_repeats_by_cohort"]).values()) == {12}


def test_cross_cohort_second_pass_is_capped_without_turning_unmeasured_rows_negative() -> None:
    rows = []
    for index in range(6):
        for cohort in ("early", "middle", "late"):
            rows.append({
                "feature": f"f{index}", "mda_cohort_id": cohort,
                "mda_feature_evaluable": True,
                "mda_mean": (-0.1 if cohort == "middle" else 0.1),
                "n_repeats": 3, "shadow_null_threshold": 0.01,
                "split_count": 1, "confidence_label": "borderline",
                "reason": "first_pass", "method": "fixture",
            })
    frame = pd.DataFrame(rows)
    # The aggregate helper derives the deterministic first-pass score.
    selected = _stage_i_second_pass_feature_names(
        [f"f{i}" for i in range(6)], frame,
        cfg={"confidence_level": 0.95, "max_repeats": 12,
             "stage_i_adaptive_min_evaluable_cohorts": 3,
             "stage_i_adaptive_two_pass_enforce": True,
             "decision_default_for_borderline": "review",
             "stage_i_second_pass_max_features": 2},
    )
    assert len(selected) == 2
    assert set(selected).issubset({f"f{i}" for i in range(6)})


def test_vectorized_coarse_gate_is_time_spread_and_preserves_protected_fields() -> None:
    rng = np.random.default_rng(71)
    y = rng.normal(size=500).astype(np.float32)
    frame = pd.DataFrame(rng.normal(size=(500, 40)).astype(np.float32), columns=[f"f{i}" for i in range(40)])
    frame["f0"] = y + rng.normal(scale=0.01, size=len(y))
    selected, audit = _stage_i_vectorized_coarse_gate(
        frame, y, list(frame.columns), protected={"f39"},
        cfg={"stage_i_coarse_gate_enabled": True, "stage_i_coarse_gate_trigger": 20,
             "stage_i_coarse_gate_max_features": 20, "stage_i_coarse_gate_row_fraction": 0.60,
             "stage_i_coarse_gate_max_rows": 300},
    )
    assert audit["applied"] is True
    assert audit["row_count"] == 300
    assert len(selected) == 21
    assert {"f0", "f39"}.issubset(selected)


def test_aggressive_pruning_never_ranks_away_untested_features() -> None:
    names = [f"tested_{i:03d}" for i in range(210)] + [f"unused_{i:03d}" for i in range(30)]
    stats = pd.DataFrame({
        "feature": names,
        "hard_drop": False,
        "confidence_label": ["borderline"] * 210 + ["unused_exact_zero"] * 30,
        "mda_evaluated": [True] * 210 + [False] * 30,
        "mda_n_repeats": [12] * 210 + [0] * 30,
        "mda_lower_95": np.linspace(-1.0, 1.0, len(names)),
        "feature_score": np.linspace(0.0, 1.0, len(names)),
    })
    # Even a stale/contradictory hard-drop bit cannot substitute for evidence.
    stats.loc[stats.feature.eq("unused_000"), "hard_drop"] = True
    selected, audit = _stage_i_depth_dependent_survivors(
        stats, names, protected=set(), context_features=set(), keep_fraction=0.70,
    )
    assert set(names[-30:]).issubset(selected)
    assert audit["untested_without_member_or_group_evidence_retained"] == 30


def test_context_protection_is_exact_contract_not_name_regex() -> None:
    names = [f"plain_{i:03d}" for i in range(237)] + [
        "chop_score", "volatility_zscore", "looks_like_regime_but_not_declared",
    ]
    stats = pd.DataFrame({
        "feature": names, "hard_drop": False, "confidence_label": "borderline",
        "mda_evaluated": True, "mda_n_repeats": 12,
        "mda_lower_95": np.arange(len(names), dtype=float),
        "feature_score": np.arange(len(names), dtype=float),
    })
    # Make the three named fields lowest-ranked; only exact declared fields survive.
    stats.loc[stats.feature.isin(names[-3:]), ["mda_lower_95", "feature_score"]] = -1000.0
    selected, audit = _stage_i_depth_dependent_survivors(
        stats, names, protected=set(),
        context_features={"chop_score", "volatility_zscore"},
        context_contract_source="stage_i_causal_context_protection_v1",
    )
    assert "chop_score" in selected and "volatility_zscore" in selected
    assert "looks_like_regime_but_not_declared" not in selected
    assert audit["context_features"] == ["chop_score", "volatility_zscore"]


def test_three_cohort_zero_use_below_phantom_can_prune_by_evidence_not_order() -> None:
    names = [f"zero_{i:03d}" for i in range(240)]
    margin = np.linspace(-1.0, -0.001, len(names))
    stats = pd.DataFrame({
        "feature": names,
        "hard_drop": False,
        "confidence_label": "unused_exact_zero",
        "mda_evaluated": False,
        "mda_n_repeats": 0,
        "baseline_cohort_model_count": 3,
        "zero_use_cohort_count": 3,
        "targeted_prescreen_below_phantom_all_eras": True,
        "targeted_prescreen_phantom_margin_max": margin,
        "mda_lower_95": 0.0,
        "feature_score": 0.0,
        "unavailable_warmup_cohort_count": 0,
    })
    selected, audit = _stage_i_depth_dependent_survivors(
        stats.sample(frac=1.0, random_state=17),
        names,
        protected=set(),
        context_features=set(),
    )
    assert len(selected) == 168
    expected = set(
        stats.nlargest(168, "targeted_prescreen_phantom_margin_max")["feature"]
    )
    assert set(selected) == expected
    assert audit["zero_use_three_cohort_below_phantom_evidence_count"] == 240
    assert audit["zero_use_fallback_removed"] == 72


def test_zero_use_fallback_refuses_to_break_equal_evidence_ties() -> None:
    names = [f"equal_{i:03d}" for i in range(240)]
    stats = pd.DataFrame({
        "feature": names, "hard_drop": False,
        "confidence_label": "unused_exact_zero", "mda_evaluated": False,
        "mda_n_repeats": 0, "baseline_cohort_model_count": 3,
        "zero_use_cohort_count": 3,
        "targeted_prescreen_below_phantom_all_eras": True,
        "targeted_prescreen_phantom_margin_max": -0.5,
        "mda_lower_95": 0.0, "feature_score": 0.0,
        "unavailable_warmup_cohort_count": 0,
    })
    selected, audit = _stage_i_depth_dependent_survivors(
        stats, names, protected=set(), context_features=set(),
    )
    assert selected == names
    assert audit["target_count_shortfall_due_to_untested_evidence_gate"] == 72
    assert audit["quota_status"] == "shortfall_fail_closed"


def test_targeted_zero_use_evidence_is_feature_order_invariant() -> None:
    rng = np.random.default_rng(123)
    rows = 360
    frame = pd.DataFrame({
        "a": rng.normal(size=rows),
        "b": rng.normal(size=rows),
    })
    target = (rng.normal(size=rows) > 0.0).astype(np.float32)
    timestamps = pd.date_range("2023-01-01", periods=rows, freq="h", tz="UTC")
    left = _stage_i_targeted_zero_use_prescreen_evidence(
        frame, target, returns=np.where(target > 0, 100.0, -100.0),
        timestamps=timestamps, features=["a", "b"], classifier=True,
        random_state=19,
    ).sort_values("feature").reset_index(drop=True)
    right = _stage_i_targeted_zero_use_prescreen_evidence(
        frame, target, returns=np.where(target > 0, 100.0, -100.0),
        timestamps=timestamps, features=["b", "a"], classifier=True,
        random_state=19,
    ).sort_values("feature").reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)
    assert left["targeted_prescreen_evaluable_era_count"].eq(3).all()


def _tier_fixture(count: int) -> tuple[list[str], pd.DataFrame]:
    names = [f"tier_{index:03d}" for index in range(count)]
    return names, pd.DataFrame({
        "feature": names,
        "hard_drop": False,
        "confidence_label": "borderline",
        "mda_evaluated": True,
        "mda_n_repeats": 12,
        "unavailable_warmup_cohort_count": 0,
        "mda_lower_95": np.linspace(-1.0, 1.0, count),
        "feature_score": np.linspace(-0.5, 0.5, count),
    })


@pytest.mark.parametrize(
    ("start_count", "expected_count", "expected_mode"),
    [
        (250, 175, "coarse_70pct_above_200"),
        (190, 152, "coarse_80pct_to_120"),
        (123, 120, "coarse_80pct_to_120"),
    ],
)
def test_tiered_coarse_gate_counts(
    start_count: int, expected_count: int, expected_mode: str,
) -> None:
    names, stats = _tier_fixture(start_count)
    selected, audit = _stage_i_depth_dependent_survivors(
        stats, names, protected=set(), context_features=set(),
    )
    assert len(selected) == expected_count
    assert audit["mode"] == expected_mode
    assert audit["actual_removal_count"] == start_count - expected_count
    assert audit["minimum_removal_quota"] == min(5, start_count - 120)
    assert audit["minimum_evidenced_removals_per_selection_checkpoint"] == min(
        5, start_count - 120
    )
    assert audit["quota_unit"] == "aggregated_pruning_round_not_cv_fold"
    assert audit["fold_specific_feature_contracts"] is False


def test_tiered_round_removes_minimum_five_when_only_five_are_evidenced() -> None:
    names, stats = _tier_fixture(130)
    stats.loc[5:, "confidence_label"] = "unused_exact_zero"
    stats.loc[5:, "mda_evaluated"] = False
    stats.loc[5:, "mda_n_repeats"] = 0
    selected, audit = _stage_i_depth_dependent_survivors(
        stats, names, protected=set(), context_features=set(),
    )
    assert len(selected) == 125
    assert audit["actual_removal_count"] == 5
    assert audit["quota_status"] == "met"
    assert audit["target_removal_shortfall"] == 5


def test_tiered_round_persists_shortfall_and_fails_closed_without_five() -> None:
    names, stats = _tier_fixture(130)
    stats.loc[4:, "confidence_label"] = "unused_exact_zero"
    stats.loc[4:, "mda_evaluated"] = False
    stats.loc[4:, "mda_n_repeats"] = 0
    selected, audit = _stage_i_depth_dependent_survivors(
        stats, names, protected=set(), context_features=set(),
    )
    assert selected == names
    assert audit["quota_status"] == "shortfall_fail_closed"
    assert audit["minimum_removal_shortfall"] == 1
    assert audit["stop_requested"] is True


def test_tiered_round_protection_and_equal_evidence_ties_fail_closed() -> None:
    names, stats = _tier_fixture(130)
    stats.loc[:, ["mda_lower_95", "feature_score"]] = 0.0
    protected = set(names[-5:])
    stats.loc[11:124, "confidence_label"] = "unused_exact_zero"
    stats.loc[11:124, "mda_evaluated"] = False
    stats.loc[11:124, "mda_n_repeats"] = 0
    selected, audit = _stage_i_depth_dependent_survivors(
        stats, names, protected=protected, context_features=set(),
    )
    assert selected == names
    assert protected.issubset(selected)
    assert audit["quota_status"] == "shortfall_fail_closed"
    assert audit["equal_evidence_boundary_tie_retained"] == 11
    assert audit["reason"] == "equal_evidence_boundary_prevents_minimum_round_quota"


def test_coarse_gate_never_removes_below_120() -> None:
    names, stats = _tier_fixture(120)
    stats["hard_drop"] = True
    stats["confidence_label"] = "harmful"
    selected, audit = _stage_i_depth_dependent_survivors(
        stats, names, protected=set(), context_features=set(),
    )
    assert selected == names
    assert audit["mode"] == "conservative_at_or_below_floor"
    assert audit["minimum_removal_quota"] == 0


def test_tiered_contract_persists_the_checkpoint_minimum_removal_rule() -> None:
    from extreme_price_movements.stage_i_feature_selection import (
        stage_i_tiered_pruning_contract,
    )

    contract = stage_i_tiered_pruning_contract()
    assert contract["transition_feature_floor"] == 120
    assert contract["transition_keep_fraction"] == 0.80
    assert contract["minimum_evidenced_removals_per_selection_checkpoint"] == 5
    assert contract["minimum_evidenced_removals_per_round"] == 5
