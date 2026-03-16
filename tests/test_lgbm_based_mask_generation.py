import numpy as np
import pandas as pd

from extreme_price_movements.config import RIDGE_FEATURE_COLS
from extreme_price_movements.data_store import LazyFeatureDict
from extreme_price_movements.lgbm_based_mask_generation import (
    CanonicalRuleMaskResolver,
    DictionaryMaskResolver,
    FeatureProcessor,
    FeatureMetadata,
    RuleCondition,
    RuleExtractor,
    RuleConsolidator,
    RuleScorer,
    build_walk_forward_folds,
    build_label_step_sliceplanner_keep_idx,
    filter_complete_feature_rows,
    list_preload_training_symbols,
)


def test_build_walk_forward_folds_is_forward_only():
    folds = build_walk_forward_folds(20, 4, min_train_frac=0.5, embargo=1)
    assert folds
    for tr_idx, va_idx in folds:
        assert tr_idx.dtype == np.int32
        assert va_idx.dtype == np.int32
        assert tr_idx.max() < va_idx.min()


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
        FeatureMetadata("trigger_long", 0, "trigger", "trigger_long", "trigger", "boolean"),
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
            "(trigger==1)|(loc==1)|(reg==1)": np.array([True, False, True, False, True, False]),
            "(*)|(loc==1)|(reg==1)": np.array([True, True, True, True, True, True]),
        },
        parent_context_map={"(trigger==1)|(loc==1)|(reg==1)": "(*)|(loc==1)|(reg==1)"},
    )
    scorer = RuleScorer([], {"min_support_count_validation": 1, "min_presence_freq": 0.0, "min_sign_consistency": 0.0, "prune_base_hurdle": 0.0, "prune_support_exp": 0.5}, mask_resolver=resolver)
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


def test_dilate_mask_by_symbol_is_symbol_safe():
    consolidator = RuleConsolidator([], {}, mask_resolver=None)
    data = np.array(["A", "B", "A", "B", "A", "B"], dtype=object)
    df = pd.DataFrame({"symbol": data})
    mask = np.array([True, False, False, False, False, False])
    dilated = consolidator._dilate_mask_by_symbol(mask, df, bars=1)
    assert dilated.tolist() == [True, False, True, False, False, False]


def test_semantic_relation_detects_same_regime_location():
    consolidator = RuleConsolidator([], {}, mask_resolver=None)
    row_a = pd.Series(
        {"canonical_key": "(t1==1)|(loc==1)|(reg==1)", "parent_context_key": "(*)|(loc==1)|(reg==1)"}
    )
    row_b = pd.Series(
        {"canonical_key": "(t2==1)|(loc==1)|(reg==1)", "parent_context_key": "(*)|(loc==1)|(reg==1)"}
    )
    assert consolidator._semantic_relation(row_a, row_b) == "same_regime_location"


def test_semantic_relation_uses_row_name_when_canonical_key_column_missing():
    consolidator = RuleConsolidator([], {}, mask_resolver=None)
    row_a = pd.Series({"parent_context_key": None}, name="(t1==1)|(loc==1)|(reg==1)")
    row_b = pd.Series({"parent_context_key": None}, name="(t2==1)|(loc==1)|(reg==1)")
    assert consolidator._semantic_relation(row_a, row_b) == "same_regime_location"


def test_consolidator_row_key_uses_series_name_when_column_missing():
    consolidator = RuleConsolidator([], {}, mask_resolver=None)
    row = pd.Series({"hurdle_excess": 0.1}, name="(t1==1)|(loc==1)|(reg==1)")
    assert consolidator._row_key(row) == "(t1==1)|(loc==1)|(reg==1)"


def test_ridge_pair_diagnostic_prefers_complementary_pair():
    resolver = DictionaryMaskResolver(
        {
            "A": np.array([1, 0, 1, 0, 1, 0], dtype=bool),
            "B": np.array([0, 1, 0, 1, 0, 1], dtype=bool),
        }
    )
    consolidator = RuleConsolidator(
        [],
        {"merge_ridge_alpha": 1.0, "merge_ridge_min_train": 4, "merge_ridge_min_valid": 2},
        mask_resolver=resolver,
    )
    folds = [
        (np.array([0, 1, 2, 3], dtype=np.int32), np.array([4, 5], dtype=np.int32)),
    ]
    diag = consolidator._evaluate_ridge_pair(
        "A",
        "B",
        resolver,
        fwd_ret=np.array([0.03, -0.01, 0.02, -0.02, 0.04, -0.03], dtype=np.float32),
        folds=folds,
    )
    assert diag["ridge_mean_net_ret"] > 0


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

    data_f, feat_f, fwd_f, fwd_norm_f, meta = filter_complete_feature_rows(data, feature_dict, fwd, fwd_norm)

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


def test_rule_extractor_stage_a_collapses_duplicate_groups_to_single_positive_slot():
    metadata = [
        FeatureMetadata("loc_a", 0, "location", "loc_a", "location", "boolean"),
        FeatureMetadata("loc_b", 1, "location", "loc_b", "location", "boolean"),
        FeatureMetadata("reg_a", 2, "regime", "reg_a", "regime", "boolean"),
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
            RuleCondition("loc_b", 1, "location", 1, ">", 0.5),
            RuleCondition("reg_a", 2, "regime", 1, ">", 0.5),
        ]
    )
    assert reason is None
    assert reduced is not None
    assert [(c.feature_name, c.normalized_value) for c in reduced] == [
        ("loc_b", 1),
        ("reg_a", 1),
    ]
    valid, valid_reason = extractor._is_path_valid(reduced)
    assert valid
    assert valid_reason == "valid"


def test_rule_extractor_stage_a_allows_same_group_negative_refinements():
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
    assert reason is None
    assert reduced is not None
    assert [(c.feature_name, c.normalized_value) for c in reduced] == [
        ("loc_a", 1),
        ("reg_a", 1),
    ]
    valid, valid_reason = extractor._is_path_valid(reduced)
    assert valid
    assert valid_reason == "valid"


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
        pd.DataFrame(),
    )
    x, metadata, _ = processor.prepare_features(
        feature_dict,
        timestamps,
        symbol_codes,
        cfg={},
        active_groups=("regime",),
    )
    names = [m.feature_name for m in metadata]
    assert f"reg_{src}_hybrid_top20" in names
    assert f"reg_{src}_hybrid_bot20" in names
    assert f"reg_{src}_hybrid_top40" in names
    assert f"reg_{src}_hybrid_bot40" in names
    assert f"reg_{src}_hybrid_top60" in names
    assert f"reg_{src}_hybrid_bot60" in names
    assert f"reg_{src}_hybrid_top80" in names
    assert f"reg_{src}_hybrid_bot80" in names
    assert f"reg_{src}_hybrid_band30_70" in names
    assert x.shape[1] == 9
