import numpy as np
import pandas as pd

from extreme_price_movements.mask_optimiser import (
    _phase3_parent_mode,
    _phase3_parent_relation_type,
    _phase3_parent_seed_key,
)
from extreme_price_movements.trigger_discovery import (
    TriggerDiscoveryConfig,
    TriggerTemplate,
    _apply_template,
    _ridge_rank_trigger_prescreen_rows,
    build_trigger_feature_frame,
    compute_horizon_edge_metrics,
    compute_timing_metrics,
    compute_trigger_score,
    evaluate_trigger_for_regime,
    generate_trigger_templates,
    prune_non_dominated_triggers,
)


def _make_shared(n: int = 16) -> dict:
    idx = np.arange(n, dtype=np.float32)
    return {
        "open": 100.0 + idx,
        "high": 100.5 + idx,
        "low": 99.5 + idx,
        "close": 100.2 + idx,
        "volume": np.full(n, 10.0, dtype=np.float32),
        "day_ids": np.arange(n, dtype=np.int32),
        "symbol_codes": np.zeros(n, dtype=np.int32),
        "timestamps": np.arange(n, dtype=np.int64),
    }


def test_trigger_template_id_is_deterministic():
    config = TriggerDiscoveryConfig()
    t1 = generate_trigger_templates(config)[0]
    t2 = generate_trigger_templates(config)[0]
    assert t1.trigger_id == t2.trigger_id
    assert t1.trigger_params_json == t2.trigger_params_json


def test_build_trigger_feature_frame_uses_shifted_rolling_extreme():
    shared = _make_shared(10)
    shared["high"][5] = 999.0
    asset_groups = {0: np.arange(10, dtype=np.int32)}
    feature_frame = build_trigger_feature_frame(shared, asset_groups)

    # The anchor at t=5 must exclude the current bar's spike.
    assert feature_frame["rolling_high_5"][5] < 200.0
    assert feature_frame["rolling_high_5"][6] == np.float32(999.0)


def test_build_trigger_feature_frame_populates_added_primitives():
    shared = _make_shared(80)
    asset_groups = {0: np.arange(80, dtype=np.int32)}
    feature_frame = build_trigger_feature_frame(shared, asset_groups)

    for feature_name in (
        "range",
        "true_range",
        "atr_14",
        "atr_100",
        "body",
        "open_location_in_bar",
        "signed_body_ratio",
        "ema_50",
        "ema_slope_ema20_3",
        "distance_to_ema50_atr",
        "returns_10",
        "acceleration_close_atr",
        "volume_ma_20",
        "inside_bar",
        "outside_bar",
    ):
        assert feature_name in feature_frame


def test_compute_timing_metrics_known_path():
    shared = _make_shared(6)
    shared["high"] = np.array([10, 12, 15, 11, 10, 9], dtype=np.float32)
    shared["low"] = np.array([10, 9, 8, 7, 6, 5], dtype=np.float32)
    shared["close"] = np.array([10, 11, 12, 9, 8, 7], dtype=np.float32)
    feature_frame = build_trigger_feature_frame(shared, {0: np.arange(6, dtype=np.int32)})
    event_mask = np.array([True, False, False, False, False, False])

    metrics = compute_timing_metrics(
        event_mask=event_mask,
        feature_frame=feature_frame,
        asset_groups={0: np.arange(6, dtype=np.int32)},
        horizon_bars=4,
        is_long=True,
    )

    assert metrics["bars_to_mfe_mean"] == 2.0
    assert metrics["bars_to_mae_mean"] == 4.0
    assert 0.0 <= metrics["timing_precision_score"] <= 1.0


def test_compute_horizon_edge_metrics_captures_1h_and_3h_edge():
    shared = _make_shared(8)
    shared["close"] = np.array([100, 101, 102, 103, 104, 105, 106, 107], dtype=np.float32)
    shared["timestamps"] = np.arange(8, dtype=np.int64)
    feature_frame = build_trigger_feature_frame(shared, {0: np.arange(8, dtype=np.int32)})
    event_mask = np.array([True, True, False, False, False, False, False, False])
    cv_splits = [
        (np.arange(0, 4, dtype=np.int32), np.arange(4, 6, dtype=np.int32)),
        (np.arange(0, 6, dtype=np.int32), np.arange(6, 8, dtype=np.int32)),
    ]

    metrics_1h = compute_horizon_edge_metrics(
        event_mask=event_mask,
        feature_frame=feature_frame,
        asset_groups={0: np.arange(8, dtype=np.int32)},
        cv_splits=cv_splits,
        horizon_bars=1,
        is_long=True,
    )
    metrics_3h = compute_horizon_edge_metrics(
        event_mask=event_mask,
        feature_frame=feature_frame,
        asset_groups={0: np.arange(8, dtype=np.int32)},
        cv_splits=cv_splits,
        horizon_bars=3,
        is_long=True,
    )

    assert metrics_1h["mean_forward_return"] > 0.0
    assert metrics_1h["delta"] > 0.0
    assert metrics_1h["shrunk_delta"] > 0.0
    assert metrics_3h["mean_forward_return"] > metrics_1h["mean_forward_return"]
    assert metrics_3h["delta"] > 0.0


def test_compute_trigger_score_includes_horizon_edge_terms():
    config = TriggerDiscoveryConfig()
    low_row = pd.Series(
        {
            "delta_r_shrunk": 0.0,
            "trigger_edge_shrunk_1h": 0.001,
            "trigger_edge_shrunk_3h": 0.001,
            "S_r": 0.8,
            "primary_predictability_gain": 0.0,
            "timing_precision_score": 0.7,
            "dispersion_to_edge_ratio": 0.5,
            "trigger_gain_vs_parent": 0.0,
            "trigger_delta_support_vs_parent": 1.0,
            "support_multiplier": 1.0,
            "positive_fold_fraction_r": 1.0,
        }
    )
    high_row = low_row.copy()
    high_row["trigger_edge_shrunk_1h"] = 0.01
    high_row["trigger_edge_shrunk_3h"] = 0.02

    low_raw, low_final = compute_trigger_score(low_row, config)
    high_raw, high_final = compute_trigger_score(high_row, config)

    assert high_raw > low_raw
    assert high_final > low_final


def test_support_filter_rejects_tiny_trigger_subset():
    config = TriggerDiscoveryConfig(min_trigger_events=5, min_fold_events=2, min_trigger_support_ratio=0.2)
    shared = _make_shared(12)
    shared["runtime_cfg"] = {"phase2_metric_max_samples_per_class": 10}
    feature_frame = build_trigger_feature_frame(shared, {0: np.arange(12, dtype=np.int32)})
    parent_mask = np.ones(12, dtype=bool)
    signed_returns = np.linspace(0.0, 0.1, 12, dtype=np.float32)
    cv_splits = [
        (np.arange(0, 6, dtype=np.int32), np.arange(6, 9, dtype=np.int32)),
        (np.arange(0, 9, dtype=np.int32), np.arange(9, 12, dtype=np.int32)),
    ]

    def stub_metric_fn(*args, **kwargs):
        return {
            "delta_r": 0.01,
            "delta_r_fold_mean": 0.01,
            "delta_r_fold_std": 0.001,
            "positive_fold_fraction_r": 1.0,
            "primary_predictability_gain": 0.0,
            "continuation_predictability_gain": 0.0,
            "reversal_predictability_gain": 0.0,
            "bucket_primary_delta_fold_mean": 0.0,
            "bucket_primary_delta_fold_std": 0.0,
            "bucket_primary_delta_fold_count": 2.0,
        }

    template = TriggerTemplate(
        trigger_family="pullback_recovery",
        trigger_template_name="close_crosses_above_ema",
        params={"ema_len": 10},
        trigger_direction="conditional",
        trigger_anchor_feature="ema_10",
        definition="cross",
    )

    row = evaluate_trigger_for_regime(
        parent_regime_row=pd.Series({"name": "base", "regime_id": "base", "delta_r_shrunk": 0.0, "D_r": 0.0, "S_r": 0.0}),
        parent_mask=parent_mask,
        trigger_template=template,
        feature_frame=feature_frame,
        cv_splits=cv_splits,
        signed_returns=signed_returns,
        config=config,
        compute_full_metrics_fn=stub_metric_fn,
        mode="long",
        shared=shared,
        feature_dict={},
        asset_groups={0: np.arange(12, dtype=np.int32)},
        parent_timing_metrics={"timing_precision_score": 0.0},
    )

    assert row is None


def test_prune_non_dominated_triggers_removes_dominated_rows():
    df = pd.DataFrame(
        [
            {
                "parent_regime_id": "base",
                "trigger_id": "a",
                "delta_r_shrunk": 0.02,
                "S_r": 0.8,
                "D_r": 0.2,
                "timing_precision_score": 0.7,
                "total_events": 200,
            },
            {
                "parent_regime_id": "base",
                "trigger_id": "b",
                "delta_r_shrunk": 0.01,
                "S_r": 0.7,
                "D_r": 0.3,
                "timing_precision_score": 0.6,
                "total_events": 180,
            },
        ]
    )
    pruned = prune_non_dominated_triggers(df, TriggerDiscoveryConfig())
    kept = pruned[pruned["non_dominated_flag"]]
    assert list(kept["trigger_id"]) == ["a"]


def test_ridge_rank_trigger_prescreen_rows_keeps_top_quartile_capped_at_four():
    signed_returns = np.array(
        [0.8, 0.7, 0.9, -0.8, -0.7, -0.9, 0.6, -0.6, 0.5, -0.5],
        dtype=np.float32,
    )
    parent_mask = np.ones_like(signed_returns, dtype=bool)
    masks = [
        np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0], dtype=bool),
        np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1], dtype=bool),
        np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=bool),
        np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=bool),
        np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=bool),
        np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=bool),
        np.array([1, 0, 0, 1, 1, 0, 0, 1, 1, 0], dtype=bool),
        np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1], dtype=bool),
    ]
    prescreen_rows = [
        {
            "entry_mask": mask,
            "cheap_prescore": float(idx),
            "trigger_template": generate_trigger_templates(TriggerDiscoveryConfig())[0],
        }
        for idx, mask in enumerate(masks)
    ]

    kept = _ridge_rank_trigger_prescreen_rows(
        prescreen_rows=prescreen_rows,
        parent_mask=parent_mask,
        signed_returns=signed_returns,
        keep_fraction=0.25,
        max_keep=4,
        alpha=1.0,
    )

    assert len(kept) == 2
    assert all("ridge_prescreen_abs_coef" in row for row in kept)
    assert kept[0]["ridge_prescreen_abs_coef"] >= kept[1]["ridge_prescreen_abs_coef"]


def test_phase3_parent_helpers_use_trigger_parent_when_enabled():
    row = pd.Series({"name": "base::trigger", "parent_regime_id": "base"})
    assert _phase3_parent_mode({"enable_trigger_discovery_stage": True}) == "regime_trigger"
    assert _phase3_parent_mode({"enable_trigger_discovery_stage": False}) == "base_regime"
    assert _phase3_parent_seed_key(row) == "base"
    assert _phase3_parent_relation_type(1, True) == "regime_trigger_conditioner"


def test_all_configured_templates_instantiate_and_apply():
    config = TriggerDiscoveryConfig(enable_compression_release_triggers=True)
    shared = _make_shared(120)
    asset_groups = {0: np.arange(120, dtype=np.int32)}
    feature_frame = build_trigger_feature_frame(shared, asset_groups)
    templates = generate_trigger_templates(config)

    names = {template.trigger_template_name for template in templates}
    assert "ema_reclaim_touch" in names
    assert "simple_close_breakout" in names
    assert "expansion_bar" in names
    assert "impulse_bar" in names
    assert "relaxed_sweep" in names
    assert "compression_release" in names
    assert "compressed_breakout_up_down" in names

    for template in templates:
        mask = _apply_template(template, feature_frame, is_long=True)
        assert mask.shape[0] == shared["close"].shape[0]
        assert mask.dtype == bool


def test_expansion_body_breakout_applies_on_short_side():
    shared = _make_shared(40)
    asset_groups = {0: np.arange(40, dtype=np.int32)}
    feature_frame = build_trigger_feature_frame(shared, asset_groups)
    template = TriggerTemplate(
        trigger_family="breakout",
        trigger_template_name="expansion_body_breakout",
        params={"lookback": 5, "body_ratio_min": 0.6, "range_atr_min": 1.2},
        trigger_direction="conditional",
        trigger_anchor_feature="rolling_high_5",
        definition="breakout + expansion + strong body",
    )

    mask = _apply_template(template, feature_frame, is_long=False)

    assert mask.shape[0] == shared["close"].shape[0]
    assert mask.dtype == bool
