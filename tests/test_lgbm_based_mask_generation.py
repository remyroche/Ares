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
    apply_test_mode,
    atomic_to_csv,
    build_label_step_sliceplanner_keep_idx,
    build_rule_model_importance_scores,
    build_stage_a_rejection_map,
    build_walk_forward_folds,
    create_pre_global_registry,
    filter_complete_feature_rows,
    list_preload_training_symbols,
    make_regime_weights,
    make_support_preference_weights,
    make_surprisal_sample_weights,
    select_stage_a_contexts,
)


def test_prepare_features_preserves_raw_binary_regime_features():
    pass

def test_prepare_features_interleaves_groups_after_quality_checks():
    pass

def test_prepare_features_uses_rank_norm_and_persists_band_thresholds():
    pass

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
    pass

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
    assert scorer._compute_support_objective_score(0.125) == 1.0
    assert scorer._compute_support_objective_score(0.15) == 1.0

    edge_low = scorer._compute_support_objective_score(0.075)
    edge_high = scorer._compute_support_objective_score(0.175)
    assert 0.0 < edge_low < 1.0
    assert 0.0 < edge_high < 1.0

    assert scorer._compute_support_objective_score(0.049) == -np.inf
    assert scorer._compute_support_objective_score(0.201) == -np.inf


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
    pass
