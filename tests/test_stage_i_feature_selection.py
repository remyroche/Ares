import numpy as np
import pandas as pd
import pytest

from extreme_price_movements import lgbm_pipeline as lp
from extreme_price_movements.stage_i_feature_selection import (
    STAGE_I_ACTIVE_CONTRACTS,
    StageIHeadContract,
    STAGE_I_META_BASE_OOF_HANDOFF_FEATURES,
    run_stage_i_head_selection,
    run_stage_i_active_matrix,
    resolve_stage_i_feature_universe,
    stage_i_prefix_confirmation,
    stage_i_mda_config,
    validate_stage_i_label_availability,
)


def test_stage_i_is_limited_to_the_four_inventoried_cells() -> None:
    assert len(STAGE_I_ACTIVE_CONTRACTS) == 4
    assert {(c.layer, c.side) for c in STAGE_I_ACTIVE_CONTRACTS} == {
        ("base", "long"),
        ("base", "short"),
        ("meta", "long"),
        ("meta", "short"),
    }
    cfg = stage_i_mda_config(STAGE_I_ACTIVE_CONTRACTS[0], report_root="/tmp/stage_i")
    assert cfg["correlation_threshold"] == pytest.approx(0.95)
    assert cfg["topk_fracs"] == [0.10]
    assert cfg["objective"] == "signed_top10_trade_economics"
    assert cfg["shadow_null_enabled"]
    assert cfg["stage_i_phantom_in_fit_threshold"]
    assert cfg["dedicated_chronological_cohorts_enabled"] is True
    assert cfg["cohort_count"] == 3
    assert cfg["max_train_rows_per_mda_model"] == 20_000
    assert cfg["mda_train_rows"] == 60_000
    assert cfg["mda_eval_rows"] == 20_000


def test_stage_i_rejects_an_mda_model_budget_above_20k() -> None:
    with pytest.raises(ValueError, match="max_train_rows_per_mda_model"):
        stage_i_mda_config(
            STAGE_I_ACTIVE_CONTRACTS[0],
            report_root="/tmp/stage_i",
            overrides={"max_train_rows_per_mda_model": 20_001},
        )


def test_mda_cohort_plan_caps_each_model_but_aggregates_disjoint_eras() -> None:
    row_count = 120_000
    timestamps = pd.date_range(
        "2024-01-01", periods=row_count, freq="min", tz="UTC",
    )
    cfg = stage_i_mda_config(
        STAGE_I_ACTIVE_CONTRACTS[0], report_root="/tmp/stage_i",
    )

    cohorts, metadata = lp._plan_disjoint_chronological_mda_cohorts(
        timestamps,
        row_count=row_count,
        cfg=cfg,
        purge_hours=13.0,
    )

    assert len(cohorts) == 3
    assert [cohort["era_label"] for cohort in cohorts] == [
        "early", "middle", "late",
    ]
    assert max(cohort["train_rows"] for cohort in cohorts) <= 20_000
    train_parts = [cohort["train_indices"] for cohort in cohorts]
    eval_parts = [cohort["evaluation_indices"] for cohort in cohorts]
    train_union = np.unique(np.concatenate(train_parts))
    eval_union = np.unique(np.concatenate(eval_parts))
    assert len(train_union) == sum(len(part) for part in train_parts)
    assert len(eval_union) == sum(len(part) for part in eval_parts)
    assert len(np.intersect1d(train_union, eval_union)) == 0
    assert len(train_union) > 20_000
    assert metadata["aggregate_unique_train_rows"] == len(train_union)
    assert metadata["aggregate_unique_evaluation_rows"] == len(eval_union)
    assert metadata["pairwise_train_overlap_rows"] == 0
    assert metadata["pairwise_evaluation_overlap_rows"] == 0
    assert metadata["train_evaluation_cross_role_overlap_rows"] == 0
    timestamp_ns = timestamps.asi8
    purge_ns = pd.Timedelta(hours=13).value
    for cohort in cohorts:
        train_idx = cohort["train_indices"]
        eval_idx = cohort["evaluation_indices"]
        assert timestamp_ns[train_idx].max() + purge_ns < timestamp_ns[eval_idx].min()


def test_mda_feature_aggregation_persists_worst_and_latest_cohort_support() -> None:
    audit = pd.DataFrame(
        {
            "feature": ["signal"] * 3,
            "mda_cohort_id": ["mda_cohort_01", "mda_cohort_02", "mda_cohort_03"],
            "fold_i": [1, 2, 3],
            "mda_mean": [3.0, -1.0, 2.0],
            "shadow_null_threshold": [0.0, 0.0, 0.0],
            "split_count": [4.0, 4.0, 4.0],
            "n_repeats": [3, 3, 3],
            "confidence_label": ["strong_keep", "harmful", "weak_keep"],
            "method": ["full_permutation"] * 3,
            "reason": ["complete"] * 3,
        }
    )

    aggregated = lp._aggregate_mda_feature_audit(
        ["signal"], audit, cfg={"min_effect_size": 0.0},
    ).iloc[0]

    assert aggregated.cohort_count == 3
    assert aggregated.mda_median == pytest.approx(2.0)
    assert aggregated.mda_mad == pytest.approx(1.0)
    assert aggregated.positive_cohort_rate == pytest.approx(2.0 / 3.0)
    assert aggregated.worst_cohort_mda == pytest.approx(-1.0)
    assert aggregated.latest_cohort_mda == pytest.approx(2.0)
    assert aggregated.latest_cohort_label == "weak_keep"


def test_dedicated_mda_branch_refits_one_capped_model_per_cohort(monkeypatch) -> None:
    rng = np.random.default_rng(83)
    row_count = 1_200
    timestamps = pd.date_range(
        "2024-01-01", periods=row_count, freq="h", tz="UTC",
    )
    X = pd.DataFrame(
        {
            "signal": rng.normal(size=row_count).astype(np.float32),
            "context": rng.normal(size=row_count).astype(np.float32),
        }
    )
    y = (0.8 * X.signal + 0.2 * X.context).to_numpy(dtype=np.float32)
    cfg = stage_i_mda_config(
        STAGE_I_ACTIVE_CONTRACTS[0],
        report_root="/tmp/stage_i",
        overrides={
            "max_train_rows_per_mda_model": 100,
            "mda_train_rows": 300,
            "mda_eval_rows": 90,
            "min_repeats": 1,
            "max_repeats": 1,
            "shadow_null_enabled": False,
            "group_mda_enabled": False,
            "permutation_mode": "full",
            "write_mda_report": False,
            "require_economic_prefix_score": False,
        },
    )
    monkeypatch.setattr(lp, "LGBM_STABILITY_CONFIGS", 1)
    monkeypatch.setattr(lp, "LGBM_PERMUTATION_TOP_CONFIGS", 1)
    monkeypatch.setattr(lp, "LGBM_FEATURE_SELECTION_N_ESTIMATORS", 8)
    monkeypatch.setattr(lp, "LGBM_CV_MODE", "interleaved")
    monkeypatch.setattr(lp, "LGBM_CV_SPLITS", 3)
    original_fit = lp._fit_lgbm_model
    dedicated_fit_rows: list[int] = []

    def recording_fit(X_train, y_train, sample_weight, **kwargs):
        if int((kwargs.get("params") or {}).get("random_state", 0)) >= 50_000:
            dedicated_fit_rows.append(len(X_train))
        return original_fit(X_train, y_train, sample_weight, **kwargs)

    monkeypatch.setattr(lp, "_fit_lgbm_model", recording_fit)

    _stats, _oof, metrics = lp._lgbm_stability_selection_pass(
        X,
        y,
        np.ones(row_count, dtype=np.float32),
        list(X.columns),
        classifier=False,
        timestamps=timestamps,
        returns=(100.0 * y).astype(np.float32),
        metric_y=y,
        random_state=11,
        mda_config=cfg,
    )

    assert dedicated_fit_rows == [100, 100, 100]
    assert metrics["mda_cohort_count"] == 3
    assert metrics["mda_max_observed_train_rows_per_model"] == 100
    assert metrics["mda_aggregate_unique_train_rows"] == 300
    assert metrics["mda_pairwise_train_overlap_rows"] == 0
    assert metrics["mda_pairwise_evaluation_overlap_rows"] == 0


def test_signed_top10_trade_score_penalizes_a_wrong_tail() -> None:
    # Positive trade economics are deliberately separated from non-negative
    # estimator weights.  Selecting the two losing rows must score worse.
    y = np.array([1, 1, 0, 0], dtype=np.float32)
    economics = np.array([80.0, 60.0, -100.0, -90.0], dtype=np.float32)
    cfg = stage_i_mda_config(STAGE_I_ACTIVE_CONTRACTS[0], report_root="/tmp/stage_i")
    right = lp._topk_mda_score(
        y,
        np.array([0.99, 0.98, 0.20, 0.10], dtype=np.float32),
        sample_weight=np.ones(4, dtype=np.float32),
        cfg=cfg,
        economic_outcomes=economics,
    )
    wrong = lp._topk_mda_score(
        y,
        np.array([0.20, 0.10, 0.99, 0.98], dtype=np.float32),
        sample_weight=np.ones(4, dtype=np.float32),
        cfg=cfg,
        economic_outcomes=economics,
    )
    assert right["signed_trade_economics_at_10"] > 0.0
    assert wrong["signed_trade_economics_at_10"] < 0.0
    assert right["score"] > wrong["score"]
    with pytest.raises(ValueError, match="explicit exact-net"):
        lp._topk_mda_score(
            y,
            np.array([0.99, 0.98, 0.20, 0.10], dtype=np.float32),
            sample_weight=np.ones(4, dtype=np.float32),
            cfg=cfg,
        )


def test_residual_mda_ranks_common_bps_score_not_raw_residual() -> None:
    # Raw residual says row 0 is best, while the frozen base expected-net
    # score makes row 1 the actual best common-bps decision. The latter is
    # the only ranking a residual feature permutation is allowed to affect.
    y = np.ones(10, dtype=np.float32)
    economics = np.array([-120.0, 140.0] + [-10.0] * 8, dtype=np.float32)
    residual_baseline = np.array([9.0, 0.0] + [0.0] * 8, dtype=np.float32)
    residual_permuted = np.array([0.0, 9.0] + [0.0] * 8, dtype=np.float32)
    frozen_base = np.array([0.0, 20.0] + [0.0] * 8, dtype=np.float32)
    cfg = stage_i_mda_config(STAGE_I_ACTIVE_CONTRACTS[2], report_root="/tmp/stage_i")
    raw = lp._topk_mda_score(
        y, residual_baseline, sample_weight=None, cfg=cfg, economic_outcomes=economics
    )
    baseline = lp._topk_mda_score(
        y,
        residual_baseline,
        sample_weight=None,
        cfg=cfg,
        economic_outcomes=economics,
        prediction_offset=frozen_base,
    )
    permuted = lp._topk_mda_score(
        y,
        residual_permuted,
        sample_weight=None,
        cfg=cfg,
        economic_outcomes=economics,
        prediction_offset=frozen_base,
    )
    assert raw["score"] == pytest.approx(-120.0)
    assert baseline["score"] == pytest.approx(140.0)
    assert permuted["score"] == pytest.approx(140.0)
    assert baseline["ranking_score_space"] == "prediction_plus_frozen_offset"
    with pytest.raises(ValueError, match="exactly aligned"):
        lp._topk_mda_score(
            y,
            residual_baseline,
            sample_weight=None,
            cfg=cfg,
            economic_outcomes=economics,
            prediction_offset=frozen_base[:-1],
        )


def test_multiclass3_fit_preserves_three_classes_and_scores_pclear_minus_padverse() -> None:
    rng = np.random.default_rng(17)
    X = pd.DataFrame({"f1": rng.normal(size=90), "f2": rng.normal(size=90)})
    y = np.repeat(np.array([0, 1, 2], dtype=np.int8), 30)
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "n_estimators": 8,
        "learning_rate": 0.1,
        "max_depth": 3,
        "random_state": 7,
        "verbose": -1,
    }
    model = lp._fit_lgbm_model(
        X, y, np.ones(len(y), dtype=np.float32), classifier=True, params=params
    )
    assert set(model.classes_) == {0, 1, 2}
    probabilities = model.predict_proba(X)
    score = lp._predict_lgbm_raw(model, X, "multiclass3")
    np.testing.assert_allclose(score, probabilities[:, 2] - probabilities[:, 0])
    # Existing binary semantics stay P(class1), rather than using the new
    # clear-minus-adverse scalar route.
    binary = lp._fit_lgbm_model(
        X, (y == 2).astype(np.int8), np.ones(len(y), dtype=np.float32),
        classifier=True,
        params={k: v for k, v in {**params, "objective": "binary"}.items() if k != "num_class"},
    )
    np.testing.assert_allclose(
        lp._predict_lgbm_raw(binary, X, "classifier"), binary.predict_proba(X)[:, 1]
    )


def test_correlation_groups_are_fit_from_training_rows_only() -> None:
    train = pd.DataFrame(
        {
            "x": np.arange(80, dtype=np.float32),
            "x_copy": np.arange(80, dtype=np.float32),
            "noise": np.tile(np.array([0.0, 1.0], dtype=np.float32), 40),
        }
    )
    # The held-out distribution deliberately breaks this relationship.  It is
    # not passed into the grouping function at all.
    valid = pd.DataFrame(
        {
            "x": np.arange(80, dtype=np.float32),
            "x_copy": np.random.default_rng(91).normal(size=80).astype(np.float32),
            "noise": np.tile(np.array([1.0, 0.0], dtype=np.float32), 40),
        }
    )
    groups = lp._correlation_groups_for_mda(
        train, list(train.columns), threshold=0.95, random_state=7
    )
    valid_groups = lp._correlation_groups_for_mda(
        valid, list(valid.columns), threshold=0.95, random_state=7
    )
    assert any({0, 1}.issubset(set(group)) for group in groups)
    assert not any({0, 1}.issubset(set(group)) for group in valid_groups)


def test_label_availability_gate_requires_exact_signal_plus_13h() -> None:
    ts = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    diag = validate_stage_i_label_availability(ts, ts + pd.Timedelta(hours=13))
    assert diag["label_availability_gate_hours"] == pytest.approx(13.0)
    assert diag["label_availability_max_hours"] == pytest.approx(13.0)
    with pytest.raises(ValueError, match="exact signal-close-to-H12"):
        validate_stage_i_label_availability(ts, ts + pd.Timedelta(hours=12))
    with pytest.raises(ValueError, match="exact signal-close-to-H12"):
        validate_stage_i_label_availability(ts, ts + pd.Timedelta(hours=14))
    with pytest.raises(ValueError, match="missing"):
        validate_stage_i_label_availability(ts, [ts[0], None, ts[2]])


def test_stage_i_runner_forces_strict_chronological_cv_and_restores_it(tmp_path) -> None:
    contract = StageIHeadContract("base", "long", "R3_economic_simplex_b25")
    frame = pd.DataFrame({"f1": [0.1, 0.2], "f2": [0.3, 0.4]})
    cfg = {
        "base_shared_feature_keys": ["BASE"],
        "base_product_feature_keys": [],
        "BASE": ["f1", "f2"],
    }
    ts = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    before = (
        lp.LGBM_CV_MODE,
        lp.LGBM_PURGE_HOURS,
        lp.LGBM_FORWARD_BURNIN_STRICT,
        lp.LGBM_FORWARD_ALLOW_SHORT_HISTORY_FALLBACK,
        lp.LGBM_SELECTION_SE_MULT,
    )

    def fake_train(*_args, **_kwargs):
        assert lp.LGBM_CV_MODE == "forward_burnin"
        assert lp.LGBM_PURGE_HOURS >= 13.0
        assert lp.LGBM_FORWARD_BURNIN_STRICT is True
        assert lp.LGBM_FORWARD_ALLOW_SHORT_HISTORY_FALLBACK is False
        assert lp.LGBM_SELECTION_SE_MULT == pytest.approx(1.0)
        np.testing.assert_allclose(_kwargs["returns"], [50.0, -100.0])
        return {"selected_feature_names": ["f1"]}

    result = run_stage_i_head_selection(
        frame,
        np.array([0.0, 1.0], dtype=np.float32),
        contract=contract,
        cfg=cfg,
        report_root=tmp_path,
        train_candidate=fake_train,
        candidate_kwargs={
            "timestamps": ts,
            "label_available_timestamps": ts + pd.Timedelta(hours=13),
            "exact_net_bps": np.array([50.0, -100.0], dtype=np.float32),
            "exact_net_units": "bps",
            "r3_metric_target": np.array([-1.0, 0.0], dtype=np.float32),
        },
    )
    assert result is not None
    assert result["stage_i_input_features"] == ["f1", "f2"]
    assert (
        lp.LGBM_CV_MODE,
        lp.LGBM_PURGE_HOURS,
        lp.LGBM_FORWARD_BURNIN_STRICT,
        lp.LGBM_FORWARD_ALLOW_SHORT_HISTORY_FALLBACK,
        lp.LGBM_SELECTION_SE_MULT,
    ) == before


def test_stage_i_forbids_preset_feature_bypass(tmp_path) -> None:
    contract = StageIHeadContract("base", "long", "R3_economic_simplex_b25")
    frame = pd.DataFrame({"f1": [0.1, 0.2], "f2": [0.3, 0.4]})
    ts = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="forbids preset_feature_names"):
        run_stage_i_head_selection(
            frame,
            np.array([0, 2], dtype=np.int8),
            contract=contract,
            cfg={
                "base_shared_feature_keys": ["BASE"],
                "base_long_feature_keys": [],
                "BASE": ["f1", "f2"],
            },
            report_root=tmp_path,
            train_candidate=lambda *_args, **_kwargs: {},
            candidate_kwargs={
                "timestamps": ts,
                "label_available_timestamps": ts + pd.Timedelta(hours=13),
                "exact_net_bps": np.array([-100.0, 100.0]),
                "exact_net_units": "bps",
                "r3_metric_target": np.array([-1.0, 1.0]),
                "preset_feature_names": ["f1", "f2"],
            },
        )


def test_r3_simplex_contract_requires_and_forwards_multiclass3(tmp_path) -> None:
    contract = StageIHeadContract("base", "long", "R3_economic_simplex_b25")
    frame = pd.DataFrame({"f1": [0.1, 0.2, 0.3], "f2": [0.3, 0.4, 0.5]})
    cfg = {
        "base_shared_feature_keys": ["BASE"],
        "base_long_feature_keys": [],
        "BASE": ["f1", "f2"],
    }
    ts = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    captured = {}
    with pytest.raises(ValueError, match="r3_metric_target"):
        run_stage_i_head_selection(
            frame,
            np.array([0, 1, 2], dtype=np.int8),
            contract=contract,
            cfg=cfg,
            report_root=tmp_path,
            train_candidate=lambda *_args, **_kwargs: None,
            candidate_kwargs={
                "timestamps": ts,
                "label_available_timestamps": ts + pd.Timedelta(hours=13),
                "exact_net_bps": np.array([10.0, 20.0, -50.0]),
                "exact_net_units": "bps",
            },
        )
    result = run_stage_i_head_selection(
        frame,
        np.array([0, 1, 2], dtype=np.int8),
        contract=contract,
        cfg=cfg,
        report_root=tmp_path,
        train_candidate=lambda *_args, **kwargs: captured.update(kwargs) or {},
        candidate_kwargs={
            "timestamps": ts,
            "label_available_timestamps": ts + pd.Timedelta(hours=13),
            "exact_net_bps": np.array([10.0, 20.0, -50.0]),
            "exact_net_units": "bps",
            "r3_metric_target": np.array([-1.0, 0.0, 1.0]),
        },
    )
    assert result is not None
    assert captured["mode"] == "multiclass3"
    np.testing.assert_allclose(captured["hard_labels"], [-1.0, 0.0, 1.0])


def test_feature_universe_is_side_and_active_head_specific() -> None:
    cfg = {
        "base_shared_feature_keys": ["BASE_SHARED"],
        "base_long_feature_keys": ["BASE_LONG"],
        "base_short_feature_keys": ["BASE_SHORT"],
        "meta_shared_feature_keys": ["META_SHARED"],
        "meta_product_feature_keys": ["META_PRODUCT"],
        "STAGE_I_M6_SHARED_UNION_META_FEATURE_KEYS": ["M6_POOL"],
        "STAGE_I_REQUIRED_SAME_SIDE_BASE_OOF_HANDOFF_FEATURE_KEYS": list(STAGE_I_META_BASE_OOF_HANDOFF_FEATURES),
        "BASE_SHARED": ["base_shared"],
        "BASE_LONG": ["base_long"],
        "BASE_SHORT": ["base_short"],
        "META_SHARED": ["meta_shared"],
        "META_PRODUCT": ["meta_product"],
        "M6_POOL": ["m6_value"],
        "INACTIVE_AUX_POOL": ["must_not_enter"],
    }
    long = resolve_stage_i_feature_universe(
        cfg, layer="base", side="long", available_columns=list(cfg.keys()) + ["base_shared", "base_long", "base_short"]
    )
    short = resolve_stage_i_feature_universe(
        cfg, layer="base", side="short", available_columns=list(cfg.keys()) + ["base_shared", "base_long", "base_short"]
    )
    meta = resolve_stage_i_feature_universe(
        cfg,
        layer="meta",
        side="long",
        head="shared_exact_net_residual",
        available_columns=["meta_shared", "meta_product", "m6_value", "must_not_enter", *STAGE_I_META_BASE_OOF_HANDOFF_FEATURES],
    )
    assert long == ["base_shared", "base_long"]
    assert short == ["base_shared", "base_short"]
    assert meta == ["meta_shared", "meta_product", "m6_value", *STAGE_I_META_BASE_OOF_HANDOFF_FEATURES]


def test_stage_i_runner_requires_explicit_bps_economics(tmp_path) -> None:
    contract = StageIHeadContract("base", "long", "R3_economic_simplex_b25")
    frame = pd.DataFrame({
        "f1": [0.1, 0.2], "f2": [0.3, 0.4],
        "r3_p_adverse": [.2, .3], "r3_p_weak": [.3, .3], "r3_p_clear": [.5, .4],
        "r3_opportunity_score": [.3, .1], "prequential_base_expected_net_bps": [45., -16.],
    })
    cfg = {"base_shared_feature_keys": ["BASE"], "base_long_feature_keys": [], "BASE": ["f1", "f2"]}
    ts = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="exact_net_bps"):
        run_stage_i_head_selection(
            frame,
            np.array([0.0, 1.0], dtype=np.float32),
            contract=contract,
            cfg=cfg,
            report_root=tmp_path,
            train_candidate=lambda *_args, **_kwargs: None,
            candidate_kwargs={
                "timestamps": ts,
                "label_available_timestamps": ts + pd.Timedelta(hours=13),
            },
        )


def test_meta_stage_i_rejects_non_oof_or_cross_side_base_provenance(tmp_path) -> None:
    contract = StageIHeadContract("meta", "short", "shared_exact_net_residual")
    frame = pd.DataFrame({
        "f1": [0.1, 0.2], "f2": [0.3, 0.4],
        "r3_p_adverse": [.2, .3], "r3_p_weak": [.3, .3], "r3_p_clear": [.5, .4],
        "r3_opportunity_score": [.3, .1], "prequential_base_expected_net_bps": [45., -16.],
    })
    cfg = {
        "meta_shared_feature_keys": ["META"],
        "meta_product_feature_keys": [],
        "STAGE_I_M6_SHARED_UNION_META_FEATURE_KEYS": [],
        "STAGE_I_REQUIRED_SAME_SIDE_BASE_OOF_HANDOFF_FEATURE_KEYS": list(STAGE_I_META_BASE_OOF_HANDOFF_FEATURES),
        "META": ["f1", "f2"],
    }
    ts = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="strict same-side OOF"):
        run_stage_i_head_selection(
            frame,
            np.array([0.0, 1.0], dtype=np.float32),
            contract=contract,
            cfg=cfg,
            report_root=tmp_path,
            train_candidate=lambda *_args, **_kwargs: None,
            candidate_kwargs={
                "timestamps": ts,
                "label_available_timestamps": ts + pd.Timedelta(hours=13),
                "exact_net_bps": np.array([5.0, -5.0], dtype=np.float32),
                "exact_net_units": "bps",
                "base_oof_provenance": {"side": "long", "strict_oof": True},
            },
        )


def test_meta_stage_i_rejects_missing_direct_same_side_handoff(tmp_path) -> None:
    contract = StageIHeadContract("meta", "long", "shared_exact_net_residual")
    frame = pd.DataFrame({"f1": [.1, .2], "f2": [.3, .4]})
    cfg = {
        "meta_shared_feature_keys": ["META"],
        "meta_product_feature_keys": [],
        "STAGE_I_M6_SHARED_UNION_META_FEATURE_KEYS": [],
        "STAGE_I_REQUIRED_SAME_SIDE_BASE_OOF_HANDOFF_FEATURE_KEYS": list(STAGE_I_META_BASE_OOF_HANDOFF_FEATURES),
        "META": ["f1", "f2"],
    }
    ts = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="exact same-side base OOF handoff"):
        run_stage_i_head_selection(
            frame,
            np.array([1., -1.], dtype=np.float32),
            contract=contract,
            cfg=cfg,
            report_root=tmp_path,
            train_candidate=lambda *_args, **_kwargs: None,
            candidate_kwargs={
                "timestamps": ts,
                "label_available_timestamps": ts + pd.Timedelta(hours=13),
                "exact_net_bps": np.array([1., -1.], dtype=np.float32),
                "exact_net_units": "bps",
                "base_oof_provenance": {"side": "long", "strict_oof": "true"},
                "frozen_base_expected_net_bps": np.array([1., -1.], dtype=np.float32),
                "frozen_base_expected_net_units": "bps",
            },
        )


def test_meta_stage_i_requires_and_passes_frozen_base_common_bps_offset(tmp_path) -> None:
    contract = StageIHeadContract("meta", "short", "shared_exact_net_residual")
    frame = pd.DataFrame({
        "f1": [0.1, 0.2], "f2": [0.3, 0.4],
        "r3_p_adverse": [.2, .3], "r3_p_weak": [.3, .3], "r3_p_clear": [.5, .4],
        "r3_opportunity_score": [.3, .1], "prequential_base_expected_net_bps": [45., -16.],
    })
    cfg = {
        "meta_shared_feature_keys": ["META"],
        "meta_product_feature_keys": [],
        "STAGE_I_M6_SHARED_UNION_META_FEATURE_KEYS": [],
        "STAGE_I_REQUIRED_SAME_SIDE_BASE_OOF_HANDOFF_FEATURE_KEYS": list(STAGE_I_META_BASE_OOF_HANDOFF_FEATURES),
        "META": ["f1", "f2"],
    }
    ts = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    common = {
        "timestamps": ts,
        "label_available_timestamps": ts + pd.Timedelta(hours=13),
        "exact_net_bps": np.array([50.0, -20.0], dtype=np.float32),
        "exact_net_units": "bps",
        "base_oof_provenance": {"side": "short", "strict_oof": True},
    }
    with pytest.raises(ValueError, match="frozen_base_expected_net_bps"):
        run_stage_i_head_selection(
            frame,
            np.array([5.0, -4.0], dtype=np.float32),
            contract=contract,
            cfg=cfg,
            report_root=tmp_path,
            train_candidate=lambda *_args, **_kwargs: None,
            candidate_kwargs=common,
        )

    captured = {}

    def fake_train(*_args, **kwargs):
        captured.update(kwargs)
        return {"selected_feature_names": ["f1", *STAGE_I_META_BASE_OOF_HANDOFF_FEATURES]}

    result = run_stage_i_head_selection(
        frame,
        np.array([5.0, -4.0], dtype=np.float32),
        contract=contract,
        cfg=cfg,
        report_root=tmp_path,
        train_candidate=fake_train,
        candidate_kwargs={
            **common,
            "frozen_base_expected_net_bps": np.array([45.0, -16.0], dtype=np.float32),
            "frozen_base_expected_net_units": "bps",
        },
    )
    np.testing.assert_allclose(captured["prediction_offset"], [45.0, -16.0])
    assert captured["cfg"]["mda_config"]["require_prediction_offset"] is True
    assert captured["cfg"]["mda_config"]["pre_mda_bypass_features"] == list(STAGE_I_META_BASE_OOF_HANDOFF_FEATURES)
    assert captured["cfg"]["mda_config"]["force_include_features"] == list(STAGE_I_META_BASE_OOF_HANDOFF_FEATURES)
    assert result is not None
    assert result["stage_i_mda_ranking_score"] == (
        "frozen_base_expected_net_bps_plus_predicted_residual_bps"
    )
    assert result["stage_i_meta_target"] == (
        "exact_net_bps_minus_frozen_causal_base_expected_net_bps"
    )
    assert result["stage_i_selected_feature_contract"] == [
        "f1", *STAGE_I_META_BASE_OOF_HANDOFF_FEATURES
    ]


def test_candidate_selector_fails_closed_when_residual_mda_offset_is_missing() -> None:
    # This is intentionally a tiny frame: the offset guard must fire before
    # any resource-intensive selection work or small-sample fallback.
    with pytest.raises(ValueError, match="requires a frozen, aligned prediction_offset"):
        lp.train_lgbm_stability_candidate(
            pd.DataFrame({"f1": [0.1, 0.2], "f2": [0.3, 0.4]}),
            np.array([1.0, -1.0], dtype=np.float32),
            mode="regressor",
            hpo_objective_mode="train_meta",
            cfg={"mda_config": {"require_prediction_offset": True}},
        )


def test_active_matrix_rejects_partial_or_uninventoried_jobs(tmp_path) -> None:
    with pytest.raises(ValueError, match="exactly four"):
        run_stage_i_active_matrix(
            {},
            cfg={},
            report_root=tmp_path,
            train_candidate=lambda *_args, **_kwargs: None,
        )


def test_prefix_confirmation_uses_smallest_one_se_prefix() -> None:
    result = stage_i_prefix_confirmation(
        {
            "pruning_history": [
                {
                    "J_final": 1.00,
                    "J_se": 0.10,
                    "mda_economic_baseline_score_mean": 1.00,
                    "mda_economic_baseline_score_se": 0.10,
                    "n_features_end": 30,
                },
                {
                    "J_final": 0.95,
                    "J_se": 0.05,
                    "mda_economic_baseline_score_mean": 0.95,
                    "mda_economic_baseline_score_se": 0.05,
                    "n_features_end": 12,
                },
                {
                    "J_final": 0.80,
                    "J_se": 0.02,
                    "mda_economic_baseline_score_mean": 0.80,
                    "mda_economic_baseline_score_se": 0.02,
                    "n_features_end": 6,
                },
            ]
        }
    )
    assert result["available"]
    assert result["confirmed_prefix_feature_count"] == 12


def test_prefix_confirmation_uses_economic_mda_not_raw_model_j() -> None:
    # Legacy J would choose the 30-field raw model. The signed-economic MDA
    # score makes 12 fields the clear decision-time winner instead.
    result = stage_i_prefix_confirmation(
        {
            "pruning_history": [
                {
                    "J_final": 0.95,
                    "J_se": 0.02,
                    "mda_economic_baseline_score_mean": 10.0,
                    "mda_economic_baseline_score_se": 2.0,
                    "n_features_end": 30,
                },
                {
                    "J_final": 0.60,
                    "J_se": 0.02,
                    "mda_economic_baseline_score_mean": 30.0,
                    "mda_economic_baseline_score_se": 5.0,
                    "n_features_end": 12,
                },
            ]
        }
    )
    assert result["best_score"] == pytest.approx(30.0)
    assert result["confirmed_prefix_feature_count"] == 12
    assert result["score_key"] == "mda_economic_baseline_score_mean"
    with pytest.raises(ValueError, match="signed-economic MDA"):
        stage_i_prefix_confirmation(
            {"pruning_history": [{"J_final": 1.0, "J_se": 0.1, "n_features_end": 5}]}
        )
