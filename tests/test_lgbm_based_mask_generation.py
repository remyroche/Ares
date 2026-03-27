import numpy as np
import pandas as pd

from extreme_price_movements.config import RIDGE_FEATURE_COLS
from extreme_price_movements.data_store import LazyFeatureDict
from extreme_price_movements.lgbm_based_mask_generation import (
    CanonicalRuleMaskResolver,
    DictionaryMaskResolver,
    FeatureMetadata,
    FeatureProcessor,
    MaskAssessor,
    RuleCondition,
    RuleExtractor,
    RuleScorer,
    atomic_to_csv,
    build_label_step_sliceplanner_keep_idx,
    build_stage_a_rejection_map,
    build_walk_forward_folds,
    create_pre_global_registry,
    filter_complete_feature_rows,
    list_preload_training_symbols,
    select_stage_a_contexts,
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
    assert f"reg_{src}_ts_top20" in names
    assert f"reg_{src}_ts_bot20" in names
    assert f"reg_{src}_ts_top40" in names
    assert f"reg_{src}_ts_bot40" in names
    assert f"reg_{src}_ts_band30_70" in names
    assert f"reg_{src}_ts_bot60" in names
    assert f"reg_{src}_ts_top80" in names
    assert f"reg_{src}_ts_bot80" in names
    assert x.shape[1] == 18


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

    avg_trades = MaskAssessor._compute_avg_trades_per_day(mask, data)

    assert avg_trades == 15.0


def test_mask_assessor_subset_auc_ignores_unscored_rows():
    rng = np.random.RandomState(0)
    n = 300
    signal = rng.normal(size=n).astype(np.float32)
    x = signal.reshape(-1, 1)
    fwd_ret = np.where(signal > 0, 1.0, -1.0).astype(np.float32)
    mask = np.zeros(n, dtype=bool)
    mask[:100] = True
    mask[100:160] = True
    mask[200:260] = True
    folds = [
        (np.arange(0, 100, dtype=np.int32), np.arange(100, 200, dtype=np.int32)),
        (np.arange(0, 200, dtype=np.int32), np.arange(200, 300, dtype=np.int32)),
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

    assert auc > 0.9
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
            [{"canonical_key": "rule_pass", "accepted": True}]
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

    assert rejection_map["stage_name"].tolist().count("scorer") == 5
    assert rejection_map["stage_name"].tolist().count("pruner") == 4
    assert rejection_map["stage_name"].tolist().count("mask_assessor") == 7
    assert rejection_map["stage_name"].tolist().count("context_selector") == 6

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
