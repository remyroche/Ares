import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from extreme_price_movements import lgbm_pipeline as lp


def _fixture_frame(n: int = 260, seed: int = 7) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = ((x1 - 0.45 * x2 + rng.normal(scale=0.35, size=n)) > 0).astype(int)
    return (
        pd.DataFrame(
            {
                "x1": x1,
                "x1_dup": x1,
                "x2": x2,
                "x3_noise": x3,
                "x_unused_constant": np.ones(n),
            }
        ),
        y,
    )


def _fit_fixture(seed: int = 11):
    X, y = _fixture_frame(seed=seed)
    tr = np.arange(0, 180)
    va = np.arange(180, len(X))
    params = lp._base_lgbm_params(
        seed,
        classifier=True,
        overrides={
            "n_estimators": 50,
            "learning_rate": 0.06,
            "max_depth": 3,
            "num_leaves": 8,
            "min_child_samples": 18,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "deterministic": True,
            "force_col_wise": True,
        },
    )
    model = lp._fit_lgbm_model(
        X.iloc[tr].reset_index(drop=True),
        y[tr],
        np.ones(len(tr), dtype=np.float32),
        classifier=True,
        params=params,
    )
    pred = lp._predict_lgbm_raw(model, X.iloc[va].reset_index(drop=True), "classifier")
    gain, split = lp._feature_importances(model, X.shape[1])
    return X, y, tr, va, params, model, pred, gain, split


def _mda_cfg(**overrides):
    base = {
        "enabled": True,
        "shadow_null_enabled": False,
        "group_mda_enabled": False,
        "min_repeats": 3,
        "max_repeats": 3,
        "early_stop_strong_keep": False,
        "early_stop_null_drop": False,
        "permutation_mode": "path_gated_lgbm",
        "confidence_level": 0.95,
    }
    base.update(overrides)
    return lp._resolve_lgbm_mda_config(base)


def test_topk_opportunity_score_weighted_precision_recall_f1():
    y = np.array([1, 0, 1, 0, 1])
    score = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    out = lp.topk_opportunity_score(
        y,
        score,
        topk_fracs=(0.20, 0.40),
        topk_frac_weights=(0.75, 0.25),
    )
    assert out["precision_at_20"] == pytest.approx(1.0)
    assert out["precision_at_40"] == pytest.approx(0.5)
    assert out["recall_at_20"] == pytest.approx(1.0 / 3.0)
    assert out["f1_at_20"] == pytest.approx(0.5)
    assert out["score"] == pytest.approx(0.875)


def test_mda_score_is_archetype_conditioned_by_default() -> None:
    n = 240
    labels = np.asarray(["long_mixed"] * 120 + ["short_breakout"] * 120)
    y = np.tile(np.asarray([1, 0], dtype=np.float32), n // 2)
    score = np.full(n, 0.5, dtype=np.float32)
    score[:120] = np.where(y[:120] > 0.5, 0.9, 0.1)
    result = lp._topk_mda_score(
        y,
        score,
        sample_weight=np.ones(n, dtype=np.float32),
        cfg=lp._resolve_lgbm_mda_config(
            {
                "shadow_null_enabled": False,
                "archetype_min_rows": 64,
            }
        ),
        archetype_labels=labels,
    )
    assert result["archetype_score_count"] == 2
    assert "archetype_macro_score" in result
    assert "archetype_worst_score" in result
    assert result["score"] != pytest.approx(result["global_score"])
    permuted_score = score.copy()
    permuted_score[:120] = permuted_score[:120][::-1]
    permuted = lp._topk_mda_score(
        y,
        permuted_score,
        sample_weight=np.ones(n, dtype=np.float32),
        cfg=lp._resolve_lgbm_mda_config(
            {
                "shadow_null_enabled": False,
                "archetype_min_rows": 64,
            }
        ),
        archetype_labels=labels,
    )
    assert result["score"] > permuted["score"]


def test_exact_unused_lightgbm_feature_is_not_permuted():
    X, y, tr, va, params, model, pred, gain, split = _fit_fixture()
    cfg = _mda_cfg(permutation_mode="path_gated_lgbm")
    feature_audit = lp._compute_topk_mda_audit(
        model,
        X.iloc[tr].reset_index(drop=True),
        y[tr],
        np.ones(len(tr), dtype=np.float32),
        X.iloc[va].reset_index(drop=True),
        y[va],
        base_pred=pred,
        classifier=True,
        sample_weight_valid=np.ones(len(va), dtype=np.float32),
        rng=np.random.default_rng(123),
        cfg=cfg,
        feature_names=list(X.columns),
        split_counts=split,
        gain_importance=gain,
        model_params=params,
        random_state=123,
    )[2]
    row = feature_audit.set_index("feature").loc["x_unused_constant"]
    assert row["method"] == "exact_unused_skip"
    assert row["mda_mean"] == pytest.approx(0.0)
    assert row["mda_std"] == pytest.approx(0.0)
    assert row["mda_std_err"] == pytest.approx(0.0)
    assert int(row["n_repeats"]) == 0
    assert row["confidence_label"] == "unused_exact_zero"


def test_topk_mda_supports_fractional_soft_binary_regressor() -> None:
    X, hard = _fixture_frame(seed=31)
    soft = (0.10 + 0.80 * hard).astype(np.float32)
    tr = np.arange(0, 180)
    va = np.arange(180, len(X))
    params = lp._base_lgbm_params(
        31,
        classifier=False,
        overrides={
            "n_estimators": 50,
            "learning_rate": 0.06,
            "max_depth": 3,
            "num_leaves": 8,
            "min_child_samples": 18,
            "deterministic": True,
            "force_col_wise": True,
        },
    )
    model = lp._fit_lgbm_model(
        X.iloc[tr].reset_index(drop=True),
        soft[tr],
        np.ones(len(tr), dtype=np.float32),
        classifier=False,
        params=params,
    )
    pred = lp._predict_lgbm_raw(model, X.iloc[va].reset_index(drop=True), "regressor")
    gain, split = lp._feature_importances(model, X.shape[1])
    audit = lp._compute_topk_mda_audit(
        model,
        X.iloc[tr].reset_index(drop=True),
        soft[tr],
        np.ones(len(tr), dtype=np.float32),
        X.iloc[va].reset_index(drop=True),
        soft[va],
        base_pred=pred,
        classifier=False,
        sample_weight_valid=np.ones(len(va), dtype=np.float32),
        rng=np.random.default_rng(31),
        cfg=_mda_cfg(permutation_mode="path_gated_lgbm"),
        feature_names=list(X.columns),
        split_counts=split,
        gain_importance=gain,
        model_params=params,
        random_state=31,
    )[2]
    assert not audit.empty
    assert set(audit["feature"]) == set(X.columns)
    assert np.isfinite(pd.to_numeric(audit["mda_mean"], errors="coerce")).all()


def test_path_gated_mda_matches_full_permutation_same_seed():
    X, y, tr, va, params, model, pred, gain, split = _fit_fixture(seed=17)
    common = {
        "shadow_null_enabled": False,
        "group_mda_enabled": False,
        "min_repeats": 4,
        "max_repeats": 4,
        "early_stop_strong_keep": False,
        "early_stop_null_drop": False,
    }
    kwargs = dict(
        model=model,
        X_train=X.iloc[tr].reset_index(drop=True),
        y_train=y[tr],
        sample_weight_train=np.ones(len(tr), dtype=np.float32),
        X_valid=X.iloc[va].reset_index(drop=True),
        y_valid=y[va],
        base_pred=pred,
        classifier=True,
        sample_weight_valid=np.ones(len(va), dtype=np.float32),
        feature_names=list(X.columns),
        split_counts=split,
        gain_importance=gain,
        model_params=params,
        random_state=99,
    )
    full = lp._compute_topk_mda_audit(
        rng=np.random.default_rng(999),
        cfg=_mda_cfg(**common, permutation_mode="full"),
        **kwargs,
    )[2]
    gated = lp._compute_topk_mda_audit(
        rng=np.random.default_rng(999),
        cfg=_mda_cfg(**common, permutation_mode="path_gated_lgbm"),
        **kwargs,
    )[2]
    merged = full[["feature", "mda_mean"]].merge(
        gated[["feature", "mda_mean"]],
        on="feature",
        suffixes=("_full", "_gated"),
    )
    assert np.max(np.abs(merged["mda_mean_full"] - merged["mda_mean_gated"])) <= 1e-10


def test_adaptive_confidence_labels_and_null_early_stop():
    assert lp._mda_label_from_bounds(0.10, 0.02, 0.18, 0.0) == "strong_keep"
    assert lp._mda_label_from_bounds(0.10, -0.01, 0.18, 0.0) == "weak_keep"
    assert lp._mda_label_from_bounds(-0.04, -0.08, -0.01, 0.0) == "harmful"
    assert lp._mda_label_from_bounds(0.01, -0.02, 0.02, 0.05) == "null_or_weak"
    assert lp._mda_final_action("redundant_group_member") == "drop_candidate"

    X, y, tr, va, params, model, pred, gain, split = _fit_fixture(seed=21)
    cfg = _mda_cfg(
        min_effect_size=999.0,
        min_repeats=2,
        max_repeats=8,
        early_stop_null_drop=True,
        early_stop_strong_keep=False,
    )
    feature_audit = lp._compute_topk_mda_audit(
        model,
        X.iloc[tr].reset_index(drop=True),
        y[tr],
        np.ones(len(tr), dtype=np.float32),
        X.iloc[va].reset_index(drop=True),
        y[va],
        base_pred=pred,
        classifier=True,
        sample_weight_valid=np.ones(len(va), dtype=np.float32),
        rng=np.random.default_rng(55),
        cfg=cfg,
        feature_names=list(X.columns),
        split_counts=split,
        gain_importance=gain,
        model_params=params,
        random_state=55,
    )[2]
    used = feature_audit[feature_audit["split_count"].astype(float) > 0].iloc[0]
    assert used["confidence_label"] == "null_or_weak"
    assert int(used["n_repeats"]) == 2


def test_redundant_group_member_drops_after_aggregation():
    cfg = _mda_cfg()
    audit = pd.DataFrame(
        {
            "feature": ["representative", "redundant"],
            "mda_mean": [0.10, 0.0],
            "split_count": [3, 3],
            "confidence_label": ["strong_keep", "redundant_group_member"],
            "n_repeats": [3, 3],
            "shadow_null_threshold": [0.0, 0.0],
            "selected": [True, False],
            "group_id": ["group_0000", "group_0000"],
            "group_mda_mean": [0.12, 0.12],
            "group_mda_lower_95": [0.08, 0.08],
            "group_mda_upper_95": [0.16, 0.16],
        }
    )
    agg = lp._aggregate_mda_feature_audit(
        ["representative", "redundant"],
        audit,
        cfg=cfg,
    ).set_index("feature")
    assert agg.loc["representative", "final_action"] == "keep"
    assert bool(agg.loc["representative", "selected"])
    assert agg.loc["redundant", "confidence_label"] == "redundant_group_member"
    assert agg.loc["redundant", "final_action"] == "drop_candidate"
    assert not bool(agg.loc["redundant", "selected"])


def test_time_features_bypass_univariate_but_remain_mda_audited():
    columns = ["x1", "x2", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    cfg = {
        "lgbm_time_feature_selector_bypass_enabled": True,
        "lgbm_time_feature_selector_bypass_features": [
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
        ],
    }
    forced = lp._resolve_lgbm_time_feature_selector_bypass_features(
        columns,
        cfg,
        objective_mode="train_meta",
    )
    assert forced == ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]

    selected = lp._append_lgbm_forced_selector_features(["x1"], columns, forced)
    assert selected == ["x1", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]

    uni_stats = pd.DataFrame(
        {
            "feature": columns,
            "passed": [True, False, False, False, False, False],
            "direction_stability": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "univariate_j": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    marked = lp._mark_lgbm_forced_selector_bypass(uni_stats, forced)
    assert marked.set_index("feature").loc["hour_sin", "passed"]
    assert marked.set_index("feature").loc["dow_cos", "selector_bypass"]

    mda_cfg = lp._extend_lgbm_mda_force_include(_mda_cfg(), forced)
    audit = pd.DataFrame(
        {
            "feature": selected,
            "mda_mean": [0.10, 0.0, 0.0, 0.0, 0.0],
            "split_count": [3, 0, 0, 0, 0],
            "confidence_label": [
                "strong_keep",
                "unused_exact_zero",
                "unused_exact_zero",
                "unused_exact_zero",
                "unused_exact_zero",
            ],
            "n_repeats": [3, 0, 0, 0, 0],
            "shadow_null_threshold": [0.0, 0.0, 0.0, 0.0, 0.0],
            "selected": [True, False, False, False, False],
        }
    )
    agg = lp._aggregate_mda_feature_audit(selected, audit, cfg=mda_cfg).set_index(
        "feature"
    )
    for feature in forced:
        assert agg.loc[feature, "confidence_label"] == "forced_keep"
        assert agg.loc[feature, "final_action"] == "keep"
        assert bool(agg.loc[feature, "selected"])


def test_grouped_mda_keeps_representative_for_correlated_group():
    X, y, tr, va, params, model, pred, gain, split = _fit_fixture(seed=31)
    cfg = _mda_cfg(
        group_mda_enabled=True,
        correlation_threshold=0.99,
        confidence_level=0.50,
        min_repeats=3,
        max_repeats=3,
    )
    feature_audit, group_audit = lp._compute_topk_mda_audit(
        model,
        X.iloc[tr].reset_index(drop=True),
        y[tr],
        np.ones(len(tr), dtype=np.float32),
        X.iloc[va].reset_index(drop=True),
        y[va],
        base_pred=pred,
        classifier=True,
        sample_weight_valid=np.ones(len(va), dtype=np.float32),
        rng=np.random.default_rng(77),
        cfg=cfg,
        feature_names=list(X.columns),
        split_counts=split,
        gain_importance=gain,
        model_params=params,
        random_state=77,
    )[2:4]
    assert not group_audit.empty
    dup_group = group_audit[group_audit["features"].astype(str).str.contains("x1_dup")]
    assert not dup_group.empty
    reps = set(str(dup_group.iloc[0]["selected_representatives"]).split("|"))
    assert reps
    members = set(str(dup_group.iloc[0]["features"]).split("|"))
    member_rows = feature_audit[feature_audit["feature"].isin(members)]
    assert member_rows["selected"].astype(bool).any()


def test_group_first_family_screen_skips_null_individual_permutations():
    X, y, tr, va, params, model, pred, gain, split = _fit_fixture(seed=37)
    result = lp._compute_topk_mda_audit(
        model,
        X.iloc[tr].reset_index(drop=True),
        y[tr],
        np.ones(len(tr), dtype=np.float32),
        X.iloc[va].reset_index(drop=True),
        y[va],
        base_pred=pred,
        classifier=True,
        sample_weight_valid=np.ones(len(va), dtype=np.float32),
        rng=np.random.default_rng(91),
        cfg=_mda_cfg(
            group_first_screen_enabled=True,
            group_first_screen_kind="feature_family",
            group_first_min_repeats=2,
            group_first_max_repeats=2,
            group_first_drop_null=True,
            min_effect_size=999.0,
            group_mda_enabled=False,
        ),
        feature_names=list(X.columns),
        split_counts=split,
        gain_importance=gain,
        model_params=params,
        random_state=91,
    )
    feature_audit, group_audit, _shadow, repeats, diag = result[2:]
    assert diag["group_first_screen_enabled"]
    assert diag["group_first_screened_feature_count"] > 0
    assert group_audit["group_kind"].astype(str).str.startswith("screen_").any()
    skipped = feature_audit[feature_audit["method"].eq("group_first_null_skip")]
    assert not skipped.empty
    assert skipped["n_repeats"].eq(0).all()
    assert repeats["entity_type"].eq("screen_group").any()


def test_shadow_null_calibration_and_report_artifacts(tmp_path):
    X, y, tr, va, params, model, pred, gain, split = _fit_fixture(seed=41)
    del model, pred, gain, split
    cfg = _mda_cfg(
        shadow_null_enabled=True,
        shadow_max_features=3,
        shadow_n_repeats=2,
        min_repeats=2,
        max_repeats=2,
    )
    threshold, shadow_df = lp._shadow_null_mda_calibration(
        params,
        X.iloc[tr].reset_index(drop=True),
        y[tr],
        np.ones(len(tr), dtype=np.float32),
        X.iloc[va].reset_index(drop=True),
        y[va],
        classifier=True,
        sample_weight_valid=np.ones(len(va), dtype=np.float32),
        feature_names=list(X.columns),
        cfg=cfg,
        random_state=42,
    )
    assert np.isfinite(threshold)
    assert len(shadow_df) == 3
    assert {"shadow_feature", "template_feature", "shadow_mda_mean"}.issubset(
        shadow_df.columns
    )

    feature_audit = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "confidence_label": ["strong_keep", "unused_exact_zero"],
            "final_action": ["keep", "drop_candidate"],
            "selected": [True, False],
            "mda_shadow_null_threshold": [0.12, 0.12],
        }
    )
    report_cfg = dict(cfg)
    report_cfg["report_dir"] = str(tmp_path)
    paths = lp._write_lgbm_mda_report(
        report_cfg,
        feature_audit,
        pd.DataFrame({"group_id": ["g0"], "features": ["a|b"]}),
        shadow_df,
        pd.DataFrame({"entity_type": ["feature"], "importance": [0.1]}),
        selected_features=["a"],
        baseline_metrics={"score": 0.7, "precision_at_10": 0.8},
    )
    assert (tmp_path / "mda_feature_selection_report.json").exists()
    assert (tmp_path / "mda_feature_audit.csv").exists()
    loaded = json.loads((tmp_path / "mda_feature_selection_report.json").read_text())
    assert loaded["summary"]["n_features_total"] == 2
    assert loaded["shadow_null"]["threshold"] == pytest.approx(0.12)
    assert "shadow_distribution_quantile_threshold" in loaded["shadow_null"]
    assert loaded["selected_features"] == ["a"]
    assert paths["report_json_path"].endswith("mda_feature_selection_report.json")
