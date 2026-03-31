import numpy as np
import pandas as pd

from extreme_price_movements.config import RIDGE_FEATURE_COLS
from extreme_price_movements.data_store import LazyFeatureDict
from extreme_price_movements.lgbm_based_mask_generation import (
    CanonicalRuleMaskResolver,
    DictionaryMaskResolver,
    ExtractedRule,
    FeatureMetadata,
    FeatureProcessor,
    IndependentRulePruner,
    MaskAssessor,
    RuleCondition,
    RuleExtractor,
    RuleScorer,
    atomic_to_csv,
    build_rule_model_importance_scores,
    build_label_step_sliceplanner_keep_idx,
    build_stage_a_rejection_map,
    build_walk_forward_folds,
    create_pre_global_registry,
    filter_complete_feature_rows,
    list_preload_training_symbols,
    make_support_preference_weights,
    make_regime_weights,
    make_surprisal_sample_weights,
    select_stage_a_contexts,
    apply_test_mode,
)


def test_prepare_features_preserves_raw_binary_regime_features():
    fp = FeatureProcessor()
    feature_dict = {
        "ema20_gt_ema50": np.array([0.0, 1.0, np.nan, 1.0], dtype=np.float32),
    }
    timestamps = np.arange(4, dtype=np.int64)
    symbol_codes = np.array(["A", "A", "B", "B"], dtype=object)

    X, metadata, audits = fp.prepare_features(
        feature_dict=feature_dict,
        timestamps=timestamps,
        symbol_codes=symbol_codes,
        cfg={"boolean_only": True, "min_feature_support": 1},
        active_groups=("regime",),
    )

    assert X.shape == (4, 1)
    assert metadata[0].feature_name == "reg_ema20_gt_ema50_raw"
    assert metadata[0].group == "regime"
    assert metadata[0].booleanization_method == "raw_binary"
    assert metadata[0].threshold_type == "binary"
    assert np.allclose(X[[0, 1, 3], 0], np.array([0.0, 1.0, 1.0], dtype=np.float32))
    assert np.isnan(X[2, 0])
    audit = audits["booleanization_support_audit"]
    assert audit.loc[0, "generated_boolean"] == "reg_ema20_gt_ema50_raw"


def test_prepare_features_interleaves_groups_after_quality_checks():
    fp = FeatureProcessor()
    feature_dict = {
        "dist_ema20_atr": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "ema20_gt_ema50": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
    }
    timestamps = np.arange(4, dtype=np.int64)
    symbol_codes = np.array(["A", "A", "B", "B"], dtype=object)

    _, metadata, _ = fp.prepare_features(
        feature_dict=feature_dict,
        timestamps=timestamps,
        symbol_codes=symbol_codes,
        cfg={"boolean_only": True, "min_feature_support": 1},
        active_groups=("regime", "location"),
    )

    groups = [m.group for m in metadata[:2]]
    assert groups[:2] == ["regime", "location"]


def test_prepare_features_uses_rank_norm_and_persists_band_thresholds():
    fp = FeatureProcessor()
    feature_dict = {
        "dist_ema20_atr": np.array(
            [1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5], dtype=np.float32
        )
    }
    timestamps = np.arange(8, dtype=np.int64)
    symbol_codes = np.array(["A", "A", "A", "A", "B", "B", "B", "B"], dtype=object)

    _, metadata, _ = fp.prepare_features(
        feature_dict=feature_dict,
        timestamps=timestamps,
        symbol_codes=symbol_codes,
        cfg={"boolean_only": True, "min_feature_support": 1},
        active_groups=("location",),
    )

    assert any(m.booleanization_method == "rank_norm" for m in metadata)
    band_meta = next(m for m in metadata if m.threshold_type == "band_quantile")
    assert band_meta.threshold_value is not None
    assert band_meta.threshold_upper_value is not None
    assert 0.0 <= band_meta.threshold_value < band_meta.threshold_upper_value <= 1.0


def test_prepare_features_skips_reserved_target_side_features():
    fp = FeatureProcessor()
    feature_dict = {
        "dist_ema20_atr": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "target_eff": np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        "target_eff_surprisal": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
    }
    timestamps = np.arange(4, dtype=np.int64)
    symbol_codes = np.array(["A", "A", "B", "B"], dtype=object)

    X, metadata, _ = fp.prepare_features(
        feature_dict=feature_dict,
        timestamps=timestamps,
        symbol_codes=symbol_codes,
        cfg={"boolean_only": True, "min_feature_support": 1},
        active_groups=("regime", "location"),
    )

    feature_names = [m.feature_name for m in metadata]
    assert X.shape[1] > 0
    assert any("dist_ema20_atr" in name for name in feature_names)
    assert all("target_eff" not in name for name in feature_names)
    assert all("surprisal" not in name for name in feature_names)


def test_independent_rule_pruner_rejects_support_below_five_percent():
    pruner = IndependentRulePruner(
        {
            "support_min_pct": 0.05,
            "max_support_pct": 0.20,
            "prune_base_hurdle": 0.0,
            "prune_target_support_pct": 0.10,
            "min_tree_discoveries": 1,
            "min_sign_consistency": 0.0,
        }
    )
    df = pd.DataFrame(
        [
            {
                "canonical_key": "narrow",
                "mean_support_pct": 0.049,
                "display_arity": 2,
                "mean_net_ret": 0.1,
                "sign_consistency": 1.0,
                "discovery_count": 10,
            },
            {
                "canonical_key": "ok",
                "mean_support_pct": 0.10,
                "display_arity": 2,
                "mean_net_ret": 0.1,
                "sign_consistency": 1.0,
                "discovery_count": 10,
            },
        ]
    )

    out = pruner.prune(df)
    assert out["canonical_key"].tolist() == ["ok"]


def test_independent_rule_pruner_default_hurdle_keeps_borderline_rule() -> None:
    df = pd.DataFrame(
        [
            {
                "canonical_key": "rule",
                "mean_support_pct": 0.10,
                "display_arity": 2,
                "mean_net_ret": 0.00009,
                "sign_consistency": 1.0,
                "discovery_count": 10,
            }
        ]
    )

    out = IndependentRulePruner(
        {
            "support_min_pct": 0.05,
            "max_support_pct": 0.20,
            "prune_base_hurdle": 0.00010,
            "prune_target_support_pct": 0.10,
            "min_sign_consistency": 0.0,
        }
    ).prune(df)

    assert out["canonical_key"].tolist() == ["rule"]




def test_make_support_preference_weights_prefers_target_band_rows():
    x = np.array(
        [
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
        ],
        dtype=np.float32,
    )
    weights = make_support_preference_weights(
        x,
        target_pct=0.33,
        preferred_low_pct=0.25,
        preferred_high_pct=0.40,
        strength=0.5,
    )

    assert weights[0] > weights[2]


def test_make_regime_weights_accepts_string_symbol_labels():
    returns = np.array([0.1, -0.2, 0.15, -0.05], dtype=np.float32)
    symbols = np.array(["A", "A", "B", "B"], dtype=object)

    weights = make_regime_weights(returns, symbols, horizon=3)

    assert weights.shape == returns.shape
    assert np.all(np.isfinite(weights))
    assert weights.dtype == np.float32


def test_compute_oof_learnability_score_uses_auc_for_binary_targets():
    preds = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32)
    y = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    score, coverage = MaskAssessor._compute_oof_learnability_score(
        preds, y, np.ones_like(y, dtype=bool), min_predicted_points=1
    )

    assert np.isfinite(score)
    assert score > 0.9
    assert coverage == 1.0


def test_compute_oof_learnability_score_uses_correlation_for_continuous_targets():
    preds = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    y = np.array([0.15, 0.25, 0.35, 0.45], dtype=np.float32)
    score, coverage = MaskAssessor._compute_oof_learnability_score(
        preds, y, np.ones_like(y, dtype=bool), min_predicted_points=1
    )

    assert np.isfinite(score)
    assert score > 0.9
    assert coverage == 1.0


def test_build_walk_forward_folds_is_forward_only():
    folds = build_walk_forward_folds(20, 4, min_train_frac=0.5, embargo=1)
    assert folds
    for tr_idx, va_idx in folds:
        assert np.issubdtype(tr_idx.dtype, np.integer)
        assert np.issubdtype(va_idx.dtype, np.integer)
        assert tr_idx.max() < va_idx.min()


def test_apply_test_mode_sets_smaller_pipeline_profile():
    cfg = apply_test_mode({"preset": "production"})

    assert cfg["test_mode"] is True
    assert cfg["n_folds"] == 3
    assert cfg["sliceplanner_outer_n_folds"] == 3
    assert cfg["mask_opt_max_symbols"] == 100
    assert cfg["mask_opt_lookback_years"] == 3.0


def test_stage_b_resolver_maps_canonical_key_to_saved_context():
    x_stage_b = np.array(
        [
            [1, 1],
            [1, 0],
            [0, 1],
            [0, 0],
        ],
        dtype=np.float32,
    )
    metadata = [
        FeatureMetadata(
            "trigger_long", 0, "trigger", "trigger_long", "trigger", "boolean"
        ),
        FeatureMetadata("ctx__abcd", 1, "context", "ctx__abcd", "context", "boolean"),
    ]
    resolver = CanonicalRuleMaskResolver(
        x_stage_b,
        metadata,
        context_lookup={"ctx__abcd": np.array([True, False, True, False])},
        context_key_map={"ctx__abcd": "(*)|(loc_a==1)|(reg_a==1)"},
        slot_order=("trigger", "location", "regime"),
    )
    key = "(trigger_long==1)|(loc_a==1)|(reg_a==1)"
    mask = resolver.get_mask(key)
    assert mask.tolist() == [True, False, False, False]
    assert resolver.get_parent_context_key(key) == "(*)|(loc_a==1)|(reg_a==1)"


def test_dictionary_resolver_supports_composites():
    resolver = DictionaryMaskResolver(
        {
            "(a==1)|(*)|(*)": np.array([True, False, False]),
            "(b==1)|(*)|(*)": np.array([False, True, False]),
        }
    )
    mask = resolver.get_mask("Composite((a==1)|(*)|(*))_OR_((b==1)|(*)|(*))")
    assert mask.tolist() == [True, True, False]


def test_stage_b_uplift_uses_parent_context_oos():
    resolver = DictionaryMaskResolver(
        {
            "(trigger==1)|(loc==1)|(reg==1)": np.array(
                [True, False, True, False, True, False]
            ),
            "(*)|(loc==1)|(reg==1)": np.array([True, True, True, True, True, True]),
        },
        parent_context_map={"(trigger==1)|(loc==1)|(reg==1)": "(*)|(loc==1)|(reg==1)"},
    )
    scorer = RuleScorer(
        [],
        {
            "min_support_count_validation": 1,
            "min_presence_freq": 0.0,
            "min_sign_consistency": 0.0,
            "prune_base_hurdle": 0.0,
            "prune_support_exp": 0.5,
        },
        mask_resolver=resolver,
    )
    folds = [
        (np.array([0, 1], dtype=np.int32), np.array([2, 3], dtype=np.int32)),
        (np.array([0, 1, 2, 3], dtype=np.int32), np.array([4, 5], dtype=np.int32)),
    ]
    summary, _ = scorer.score_key_oos(
        "(trigger==1)|(loc==1)|(reg==1)",
        fwd_ret=np.array([0.0, 0.0, 0.03, 0.01, 0.04, 0.01]),
        folds=folds,
        resolver=resolver,
        require_uplift=True,
        parent_context_key="(*)|(loc==1)|(reg==1)",
    )
    assert summary["mean_baseline_ret"] > 0
    assert summary["mean_uplift"] > 0


def test_support_objective_scores_preferred_band_and_excludes_outside_bounds():
    scorer = RuleScorer([], {})

    assert scorer._compute_support_objective_score(0.10) == 1.0
    assert scorer._compute_support_objective_score(0.075) == 1.0
    assert scorer._compute_support_objective_score(0.125) == 1.0

    edge_low = scorer._compute_support_objective_score(0.05)
    edge_high = scorer._compute_support_objective_score(0.15)
    assert 0.0 < edge_low < 1.0
    assert 0.0 < edge_high < 1.0

    assert scorer._compute_support_objective_score(0.049) == -np.inf
    assert scorer._compute_support_objective_score(0.151) == -np.inf


def test_list_preload_training_symbols_uses_training_universe(monkeypatch):
    class DummyStore:
        pass

    captured = {}

    def fake_get_training_universe(margin_symbols, cfg, store, ts_sig=None):
        captured["margin_symbols"] = margin_symbols
        captured["cfg"] = cfg
        captured["store"] = store
        captured["ts_sig"] = ts_sig
        return ["ETH/USDT", "BTC/USDT", "SOL/USDT"]

    monkeypatch.setattr(
        "extreme_price_movements.lgbm_based_mask_generation.get_training_universe",
        fake_get_training_universe,
    )

    cfg = {"offline_backtest_skip_universe_refresh": True}
    store = DummyStore()
    symbols = list_preload_training_symbols(store, cfg, max_symbols=2)

    assert symbols == ["ETH/USDT", "BTC/USDT"]
    assert captured["margin_symbols"] is None
    assert captured["cfg"] is cfg
    assert captured["store"] is store
    assert captured["ts_sig"] is None


def test_lazy_feature_dict_skips_mixed_timestamp_integer_indices():
    lazy = LazyFeatureDict(
        {
            "upper_wick_ratio": {
                "GOOD": (
                    np.array(
                        [
                            pd.Timestamp("2026-01-01 00:00:00"),
                            pd.Timestamp("2026-01-01 01:00:00"),
                        ],
                        dtype=object,
                    ),
                    np.array([1.0, 2.0], dtype=np.float32),
                ),
                "BAD": (
                    np.array([pd.Timestamp("2026-01-01 00:00:00"), 1], dtype=object),
                    np.array([3.0, 4.0], dtype=np.float32),
                ),
            }
        }
    )

    df = lazy["upper_wick_ratio"]

    assert list(df.columns) == ["GOOD", "BAD"]
    assert len(df) == 2
    assert int(df["BAD"].notna().sum()) == 1


def test_filter_complete_feature_rows_drops_any_row_with_missing_feature():
    data = pd.DataFrame({"timestamp": [1, 2, 3], "symbol": ["A", "A", "A"]})
    feature_dict = {
        "reg_a": np.array([1.0, np.nan, 1.0], dtype=np.float32),
        "loc_a": np.array([0.0, 1.0, 1.0], dtype=np.float32),
        "trig_a": np.array([1.0, 1.0, np.nan], dtype=np.float32),
    }
    fwd = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    fwd_norm = np.array([0.5, 1.0, 1.5], dtype=np.float32)

    data_f, feat_f, fwd_f, fwd_norm_f, meta = filter_complete_feature_rows(
        data, feature_dict, fwd, fwd_norm
    )

    assert len(data_f) == 1
    assert np.isclose(float(fwd_f[0]), 0.1)
    assert np.isclose(float(fwd_norm_f[0]), 0.5)
    assert meta["dropped_rows"] == 2
    assert feat_f["reg_a"].tolist() == [1.0]


def test_rule_extractor_rejects_negative_stage_a_context_slots():
    metadata = [
        FeatureMetadata("loc_a", 0, "location", "loc_a", "location", "boolean"),
        FeatureMetadata("reg_a", 1, "regime", "reg_a", "regime", "boolean"),
    ]
    extractor = RuleExtractor(
        metadata,
        {},
        positive_only_groups=("location", "regime"),
    )
    valid, reason = extractor._is_path_valid(
        [
            RuleCondition("loc_a", 0, "location", 0, "<=", 0.5),
            RuleCondition("reg_a", 1, "regime", 1, ">", 0.5),
        ]
    )
    assert not valid
    assert reason == "negative_not_allowed_location"


def test_rule_extractor_allows_negative_slots_when_not_restricted():
    metadata = [
        FeatureMetadata("loc_a", 0, "location", "loc_a", "location", "boolean"),
        FeatureMetadata("reg_a", 1, "regime", "reg_a", "regime", "boolean"),
    ]
    extractor = RuleExtractor(metadata, {})
    valid, reason = extractor._is_path_valid(
        [
            RuleCondition("loc_a", 0, "location", 0, "<=", 0.5),
            RuleCondition("reg_a", 1, "regime", 1, ">", 0.5),
        ]
    )
    assert valid
    assert reason == "valid"


def test_rule_extractor_stage_a_preserves_multiple_positive_conditions_within_group():
    metadata = [
        FeatureMetadata("loc_a", 0, "location", "loc_a", "location", "boolean"),
        FeatureMetadata("loc_b", 1, "location", "loc_b", "location", "boolean"),
        FeatureMetadata("reg_a", 2, "regime", "reg_a", "regime", "boolean"),
        FeatureMetadata("reg_b", 3, "regime", "reg_b", "regime", "boolean"),
    ]
    extractor = RuleExtractor(
        metadata,
        {},
        positive_only_groups=("location", "regime"),
        required_positive_groups=("location", "regime"),
        collapse_duplicate_groups=("location", "regime"),
    )
    reduced, reason = extractor._reduce_conditions(
        [
            RuleCondition("loc_a", 0, "location", 1, ">", 0.5),
            RuleCondition("loc_b", 1, "location", 1, ">", 0.5),
            RuleCondition("reg_a", 2, "regime", 1, ">", 0.5),
            RuleCondition("reg_b", 3, "regime", 1, ">", 0.5),
        ]
    )
    assert reason is None
    assert reduced is not None
    assert [(c.feature_name, c.normalized_value) for c in reduced] == [
        ("loc_a", 1),
        ("loc_b", 1),
        ("reg_a", 1),
        ("reg_b", 1),
    ]
    valid, valid_reason = extractor._is_path_valid(reduced)
    assert valid
    assert valid_reason == "valid"


def test_rule_extractor_stage_a_rejects_same_feature_contradictions():
    metadata = [
        FeatureMetadata("loc_a", 0, "location", "loc_a", "location", "boolean"),
        FeatureMetadata("reg_a", 1, "regime", "reg_a", "regime", "boolean"),
    ]
    extractor = RuleExtractor(
        metadata,
        {},
        positive_only_groups=("location", "regime"),
        required_positive_groups=("location", "regime"),
        collapse_duplicate_groups=("location", "regime"),
    )
    reduced, reason = extractor._reduce_conditions(
        [
            RuleCondition("loc_a", 0, "location", 0, "<=", 0.5),
            RuleCondition("loc_a", 0, "location", 1, ">", 0.5),
            RuleCondition("reg_a", 1, "regime", 0, "<=", 0.5),
            RuleCondition("reg_a", 1, "regime", 1, ">", 0.5),
        ]
    )
    assert reduced is None
    assert reason in {"contradiction_loc_a", "contradiction_in_collapsed_group_loc_a"}


def test_rule_extractor_stage_a_requires_both_location_and_regime():
    metadata = [
        FeatureMetadata("loc_a", 0, "location", "loc_a", "location", "boolean"),
        FeatureMetadata("reg_a", 1, "regime", "reg_a", "regime", "boolean"),
    ]
    extractor = RuleExtractor(
        metadata,
        {},
        positive_only_groups=("location", "regime"),
        required_positive_groups=("location", "regime"),
        collapse_duplicate_groups=("location", "regime"),
    )
    valid, reason = extractor._is_path_valid(
        [RuleCondition("loc_a", 0, "location", 1, ">", 0.5)]
    )
    assert not valid
    assert reason == "missing_required_group_regime"


def test_build_label_step_sliceplanner_keep_idx_maps_clean_event_ids(monkeypatch):
    class DummyPlan:
        def __init__(self, fit_idx, tag="fit_inner"):
            self.fit_idx = np.asarray(fit_idx, dtype=np.int64)
            self.tag = tag

    class DummyPlanner:
        def __init__(self, cfg):
            self.cfg = cfg

        def build(self, events):
            # Simulate planner-side dedup dropping rows 1 and 4 from the original grid.
            clean = events.iloc[[0, 2, 3, 5]].copy()
            return {
                "events": clean,
                "consumer_plans": {
                    "regime_search": [
                        DummyPlan([1, 3], tag="fit_inner"),
                    ]
                },
            }

    monkeypatch.setattr(
        "extreme_price_movements.lgbm_based_mask_generation.SlicePlanner",
        DummyPlanner,
    )

    timestamps = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    symbols = pd.Index(["A/USDT", "A/USDC", "B/USDT"])
    keep_idx, meta = build_label_step_sliceplanner_keep_idx(
        timestamps=timestamps,
        symbols=symbols,
        cfg={},
    )

    # fit_idx=[1,3] in clean space should map back to original event_id [2,5]
    assert keep_idx.tolist() == [2, 5]
    assert meta["sliceplanner_applied"] is True


def test_feature_processor_adds_dense_regime_quantiles_and_median_band():
    processor = FeatureProcessor()
    src = RIDGE_FEATURE_COLS[0]
    feature_dict = {src: np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)}
    timestamps = np.array(
        [
            pd.Timestamp("2025-01-01 00:00:00", tz="UTC"),
            pd.Timestamp("2025-01-01 00:00:00", tz="UTC"),
            pd.Timestamp("2025-01-01 01:00:00", tz="UTC"),
            pd.Timestamp("2025-01-01 01:00:00", tz="UTC"),
        ],
        dtype=object,
    )
    symbol_codes = np.array(["A", "B", "A", "B"], dtype=object)
    processor._run_feature_quality_checks = lambda x_raw, raw_names, cfg: (
        x_raw,
        raw_names,
        pd.DataFrame(
            columns=[
                "feature_name",
                "status",
                "reason",
                "support",
                "group",
                "regime_family",
            ]
        ),
    )
    x, metadata, _ = processor.prepare_features(
        feature_dict,
        timestamps,
        symbol_codes,
        cfg={},
        active_groups=("regime",),
    )
    names = [m.feature_name for m in metadata]
    assert names == [f"reg_{src}_raw"]
    assert x.shape[1] == 1


def test_feature_quality_dedup_preserves_cross_source_duplicates():
    processor = FeatureProcessor()
    processor.metadata = {
        "loc_src_a_ts_top20": FeatureMetadata(
            "loc_src_a_ts_top20",
            0,
            "location",
            "src_a",
            "context",
            "boolean",
        ),
        "loc_src_b_ts_top20": FeatureMetadata(
            "loc_src_b_ts_top20",
            1,
            "location",
            "src_b",
            "context",
            "boolean",
        ),
        "loc_src_a_ts_bot80": FeatureMetadata(
            "loc_src_a_ts_bot80",
            2,
            "location",
            "src_a",
            "context",
            "boolean",
        ),
    }

    x_raw = np.array(
        [
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    names = ["loc_src_a_ts_top20", "loc_src_b_ts_top20", "loc_src_a_ts_bot80"]

    x_clean, retained_names, audit = processor._run_feature_quality_checks(
        x_raw, names, cfg={"min_feature_support": 1}
    )

    assert retained_names == ["loc_src_a_ts_top20", "loc_src_b_ts_top20"]
    assert x_clean.shape == (4, 2)

    dropped = audit[audit["status"] == "dropped"]
    assert len(dropped) == 1
    assert dropped.iloc[0]["feature_name"] == "loc_src_a_ts_bot80"
    assert dropped.iloc[0]["reason"] == "duplicate_of_loc_src_a_ts_top20"


def test_select_stage_a_contexts_returns_empty_summary_headers():
    selected, summary = select_stage_a_contexts(
        {"accepted_registry": pd.DataFrame()}, {}
    )
    assert selected.empty
    assert list(summary.columns) == ["reason", "count"]


def test_create_pre_global_registry_uses_selected_stage_a_contexts():
    side_results = {
        "stage_a": pd.DataFrame(
            [
                {
                    "canonical_key": "(*)|(loc_selected==1)|(reg_selected==1)",
                    "side": "long",
                }
            ]
        ),
        "stage_a_result": {
            "accepted_registry": pd.DataFrame(
                [
                    {
                        "canonical_key": "(*)|(loc_rejected==1)|(reg_rejected==1)",
                        "side": "long",
                    }
                ]
            )
        },
        "stage_b": pd.DataFrame(),
    }

    pre_global = create_pre_global_registry(side_results)

    assert pre_global["canonical_key"].tolist() == [
        "(*)|(loc_selected==1)|(reg_selected==1)"
    ]
    assert pre_global["origin_stage"].tolist() == ["stage_a"]


def test_atomic_to_csv_preserves_headers_for_empty_dataframes(tmp_path):
    output_path = tmp_path / "candidate_rule_registry.csv"

    atomic_to_csv(
        pd.DataFrame(),
        output_path,
        expected_columns=["canonical_key", "accepted", "preset"],
    )

    written = pd.read_csv(output_path)
    assert list(written.columns) == ["canonical_key", "accepted", "preset"]
    assert written.empty


def test_mask_assessor_avg_trades_per_day_uses_unique_days():
    data = pd.DataFrame(
        {
            "symbol": ["A", "A", "B", "B"],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 00:00:00",
                    "2025-01-01 01:00:00",
                    "2025-01-02 00:00:00",
                    "2025-01-02 01:00:00",
                ]
            ),
        }
    )
    mask = np.array([True, False, True, True], dtype=bool)

    total_symbol_days = MaskAssessor._compute_total_symbol_days(data)
    avg_trades = MaskAssessor._compute_avg_trades_per_day(mask, total_symbol_days)

    assert avg_trades == 15.0


def test_mask_assessor_subset_auc_ignores_unscored_rows():
    from extreme_price_movements.config import TEST_FEATURE_KEYS
    rng = np.random.RandomState(0)
    n = 10000
    signal = rng.normal(size=n).astype(np.float32)
    x = signal.reshape(-1, 1)
    fwd_ret = np.where(signal > 0, 1.0, -1.0).astype(np.float32)
    mask = np.zeros(n, dtype=bool)
    mask[:4000] = True
    mask[4000:7000] = True
    mask[7000:10000] = True
    folds = [
        (np.arange(0, 4000, dtype=np.int32), np.arange(4000, 7000, dtype=np.int32)),
        (np.arange(0, 7000, dtype=np.int32), np.arange(7000, 10000, dtype=np.int32)),
    ]

    # Needs to match one of TEST_FEATURE_KEYS to not be filtered out
    source_name = TEST_FEATURE_KEYS[0] if TEST_FEATURE_KEYS else "regime_signal"

    assessor = MaskAssessor(
        metadata=[
            FeatureMetadata(
                "regime_signal",
                0,
                "regime",
                source_name,
                "regime",
                "continuous",
            )
        ],
        cfg={},
    )

    auc, coverage = assessor._compute_subset_auc(x, fwd_ret, mask, folds)

    assert np.isfinite(auc)
    assert auc > 0.7
    assert coverage > 0.5


def test_mask_assessor_subset_auc_reports_missing_oof_coverage():
    n = 200
    signal = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    x = signal.reshape(-1, 1)
    fwd_ret = np.where(signal > 0, 1.0, -1.0).astype(np.float32)
    mask = np.zeros(n, dtype=bool)
    mask[:100] = True
    folds = [
        (np.arange(0, 100, dtype=np.int32), np.arange(100, 150, dtype=np.int32)),
        (np.arange(0, 150, dtype=np.int32), np.arange(150, 200, dtype=np.int32)),
    ]

    assessor = MaskAssessor(
        metadata=[
            FeatureMetadata(
                "regime_signal",
                0,
                "regime",
                "regime_signal",
                "regime",
                "continuous",
            )
        ],
        cfg={},
    )

    auc, coverage = assessor._compute_subset_auc(x, fwd_ret, mask, folds)

    assert np.isnan(auc)
    assert coverage == 0.0


def test_mask_assessor_baseline_auc_supports_continuous_targets():
    from extreme_price_movements.config import TEST_FEATURE_KEYS

    rng = np.random.RandomState(1)
    n = 1500
    signal = rng.normal(size=n).astype(np.float32)
    x = signal.reshape(-1, 1)
    fwd_ret = (0.3 * signal + 0.05 * rng.normal(size=n)).astype(np.float32)
    folds = [
        (np.arange(0, 900, dtype=np.int32), np.arange(900, 1200, dtype=np.int32)),
        (np.arange(0, 1200, dtype=np.int32), np.arange(1200, 1500, dtype=np.int32)),
    ]

    source_name = TEST_FEATURE_KEYS[0] if TEST_FEATURE_KEYS else "regime_signal"
    assessor = MaskAssessor(
        metadata=[
            FeatureMetadata(
                "regime_signal",
                0,
                "regime",
                source_name,
                "regime",
                "continuous",
            )
        ],
        cfg={
            "learnability_min_train_samples_continuous": 200,
            "learnability_min_val_samples_continuous": 100,
            "learnability_min_predicted_points_continuous": 100,
        },
    )

    score, coverage = assessor._compute_baseline_auc(x, fwd_ret, folds)

    assert np.isfinite(score)
    assert score > 0.2
    assert coverage > 0.3


def test_build_rule_model_importance_scores_aggregates_feature_gain_per_rule():
    rules = [
        ExtractedRule(
            rule_id="r1",
            canonical_key="rule_a",
            conditions=[
                RuleCondition("f1", 0, "regime", 1, ">", 0.5),
                RuleCondition("f2", 1, "location", 1, ">", 0.5),
            ],
            model_id="m",
            fold_id=0,
            seed=1,
            tree_index=0,
            leaf_index=0,
            leaf_value=0.1,
            support_train=10,
        ),
        ExtractedRule(
            rule_id="r2",
            canonical_key="rule_b",
            conditions=[
                RuleCondition("f3", 2, "regime", 1, ">", 0.5),
            ],
            model_id="m",
            fold_id=0,
            seed=1,
            tree_index=0,
            leaf_index=1,
            leaf_value=0.1,
            support_train=10,
        ),
    ]
    feature_importance_records = [
        {"fold_id": 0, "seed": 1, "feature_name": "f1", "group": "regime", "regime_family": "reg", "gain": 9.0, "split": 3.0},
        {"fold_id": 0, "seed": 1, "feature_name": "f2", "group": "location", "regime_family": "", "gain": 3.0, "split": 1.0},
        {"fold_id": 0, "seed": 1, "feature_name": "f3", "group": "regime", "regime_family": "reg", "gain": 1.0, "split": 1.0},
    ]

    out = build_rule_model_importance_scores(rules, feature_importance_records)
    out = out.set_index("canonical_key")

    assert "rule_a" in out.index
    assert "rule_b" in out.index
    assert out.loc["rule_a", "rule_gain_score"] > out.loc["rule_b", "rule_gain_score"]
    assert (
        out.loc["rule_a", "rule_model_importance_score"]
        > out.loc["rule_b", "rule_model_importance_score"]
    )


def test_build_stage_a_rejection_map_captures_stage_funnel():
    stage_a_result = {
        "scored_registry": pd.DataFrame(
            [
                {
                    "canonical_key": "rule_pass",
                    "accepted": True,
                    "rejection_reason": "",
                },
                {
                    "canonical_key": "rule_low_support",
                    "accepted": False,
                    "rejection_reason": "low_support|below_hurdle",
                },
                {
                    "canonical_key": "rule_low_presence",
                    "accepted": False,
                    "rejection_reason": "low_presence",
                },
            ]
        ),
        "scorer_accepted": pd.DataFrame(
            [
                {
                    "canonical_key": "rule_pass",
                    "accepted": True,
                    "mean_support_pct": 0.05,
                    "hurdle_excess": 0.01,
                    "sign_consistency": 0.9,
                    "discovery_count": 3,
                }
            ]
        ),
        "consolidated_registry": pd.DataFrame(
            [
                {
                    "canonical_key": "rule_pass",
                    "mean_support_pct": 0.05,
                    "hurdle_excess": 0.01,
                    "sign_consistency": 0.9,
                    "discovery_count": 3,
                    "dominated_by_parent": False,
                },
                {
                    "canonical_key": "rule_broad",
                    "mean_support_pct": 0.4,
                    "hurdle_excess": -0.01,
                    "sign_consistency": 0.7,
                    "discovery_count": 1,
                    "dominated_by_parent": True,
                },
            ]
        ),
        "candidate_registry": pd.DataFrame(
            [
                {
                    "canonical_key": "rule_pass",
                    "mean_support_pct": 0.05,
                    "directional_mean_ret": 0.02,
                    "presence_freq": 0.8,
                    "sign_consistency": 0.9,
                    "display_arity": 2,
                    "dominated_by_parent": False,
                    "is_structurally_sound": True,
                },
                {
                    "canonical_key": "rule_unassessed",
                    "mean_support_pct": 0.03,
                    "directional_mean_ret": 0.01,
                    "presence_freq": 0.7,
                    "sign_consistency": 0.85,
                    "display_arity": 2,
                    "dominated_by_parent": False,
                    "is_structurally_sound": False,
                },
            ]
        ),
        "assessment_df": pd.DataFrame(
            [
                {
                    "canonical_key": "rule_pass",
                    "is_structurally_sound": True,
                    "rejection_reason": "",
                }
            ]
        ),
        "accepted_registry": pd.DataFrame(
            [
                {
                    "canonical_key": "rule_pass",
                    "mean_support_pct": 0.05,
                    "directional_mean_ret": 0.02,
                    "presence_freq": 0.8,
                    "sign_consistency": 0.9,
                    "display_arity": 2,
                    "dominated_by_parent": False,
                    "is_structurally_sound": True,
                },
                {
                    "canonical_key": "rule_reject_structural",
                    "mean_support_pct": 0.05,
                    "directional_mean_ret": 0.02,
                    "presence_freq": 0.8,
                    "sign_consistency": 0.9,
                    "display_arity": 2,
                    "dominated_by_parent": False,
                    "is_structurally_sound": False,
                },
            ]
        ),
    }
    winning_contexts = pd.DataFrame(
        [
            {
                "canonical_key": "rule_pass",
                "mean_support_pct": 0.05,
                "directional_mean_ret": 0.02,
                "presence_freq": 0.8,
                "sign_consistency": 0.9,
                "display_arity": 2,
                "dominated_by_parent": False,
                "is_structurally_sound": True,
            }
        ]
    )
    cfg = {
        "min_support_count_validation": 2,
        "min_presence_freq": 0.4,
        "min_sign_consistency": 0.75,
        "max_support_pct": 0.25,
        "min_tree_discoveries": 2,
        "learnability_min_oof_coverage": 0.25,
        "min_avg_trades_per_day_10_symbols": 0.1,
        "min_context_support_pct": 0.01,
        "min_context_presence_freq": 0.5,
        "min_context_sign_consistency": 0.8,
        "min_context_display_arity": 2,
    }

    rejection_map = build_stage_a_rejection_map(stage_a_result, winning_contexts, cfg)

    stage_counts = rejection_map["stage_name"].value_counts().to_dict()
    assert stage_counts["scorer"] > 0
    assert stage_counts["pruner"] > 0
    assert stage_counts["mask_assessor"] > 0
    assert stage_counts["context_selector"] > 0

    scorer_low_support = rejection_map[
        (rejection_map["stage_name"] == "scorer")
        & (rejection_map["gate_name"] == "low_support")
    ].iloc[0]
    assert int(scorer_low_support["rejected_count"]) == 1
    assert int(scorer_low_support["passed_count"]) == 1

    assessor_unassessed = rejection_map[
        (rejection_map["stage_name"] == "mask_assessor")
        & (rejection_map["gate_name"] == "not_assessed_min_mask_support")
    ].iloc[0]
    assert int(assessor_unassessed["rejected_count"]) == 1

    selector_structural = rejection_map[
        (rejection_map["stage_name"] == "context_selector")
        & (rejection_map["gate_name"] == "reject_structural")
    ].iloc[0]
    assert int(selector_structural["rejected_count"]) == 1
    assert int(selector_structural["passed_count"]) == 1


def test_make_surprisal_sample_weights_is_bounded_and_monotonic():
    surprisal_bits = np.array([np.nan, 0.0, 1.5, 3.0, 9.0], dtype=np.float32)
    weights = make_surprisal_sample_weights(
        surprisal_bits,
        alpha=0.2,
        reference_bits=3.0,
        w_min=1.0,
        w_max=1.2,
    )

    assert weights.dtype == np.float32
    np.testing.assert_allclose(weights[0], 1.0)
    np.testing.assert_allclose(weights[1], 1.0)
    assert weights[2] > weights[1]
    np.testing.assert_allclose(weights[3], 1.2)
    np.testing.assert_allclose(weights[4], 1.2)
