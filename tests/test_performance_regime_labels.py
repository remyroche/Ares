import numpy as np
import pandas as pd

from extreme_price_movements.performance_regimes.feature_matrix import (
    apply_frozen_feature_pipeline,
    build_market_state_feature_matrix,
    fit_market_state_feature_pipeline,
)
from extreme_price_movements.performance_regimes.labels import (
    build_strategy_performance_labels,
)


def test_strategy_performance_label_anchors_and_weights_are_fold_local():
    ts = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "strategy": ["s1"] * 4,
            "performance": [-1.0, 0.0, 1.0, 100.0],
        }
    )

    train = frame.iloc[:3]
    bundle = build_strategy_performance_labels(
        train,
        strategy_col="strategy",
        timestamp_col="timestamp",
        performance_col="performance",
        strategies=["s1"],
        ewma_halflife=1,
        anchor_mode="minmax",
    )
    labels = bundle.by_strategy["s1"]
    worst_idx = labels.ewma_performance.idxmin()
    best_idx = labels.ewma_performance.idxmax()
    median_value = labels.anchors["median"]
    median_idx = (labels.ewma_performance - median_value).abs().idxmin()

    assert labels.bad_label.loc[median_idx] == 0.5
    assert labels.good_label.loc[median_idx] == 0.5
    assert labels.bad_label.loc[worst_idx] == 1.0
    assert labels.good_label.loc[worst_idx] == 0.0
    assert labels.bad_label.loc[best_idx] == 0.0
    assert labels.good_label.loc[best_idx] == 1.0
    assert labels.bad_sample_weight.loc[median_idx] == 1.0
    assert labels.bad_sample_weight.loc[worst_idx] == 4.0
    assert labels.good_sample_weight.loc[best_idx] == 4.0

    full = build_strategy_performance_labels(
        frame,
        strategy_col="strategy",
        timestamp_col="timestamp",
        performance_col="performance",
        strategies=["s1"],
        ewma_halflife=1,
        anchor_mode="minmax",
    )
    assert full.by_strategy["s1"].anchors["best"] != labels.anchors["best"]


def test_strategy_performance_label_default_ewma_halflife_is_3d():
    ts = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "strategy": ["s1"] * 4,
            "performance": [-1.0, 0.0, 1.0, 0.5],
        }
    )

    bundle = build_strategy_performance_labels(
        frame,
        strategy_col="strategy",
        timestamp_col="timestamp",
        performance_col="performance",
        strategies=["s1"],
    )

    assert bundle.diagnostics["ewma_halflife"].iloc[0] == "3D"


def test_strategy_performance_labels_can_target_multi_day_loss_streaks():
    ts = pd.date_range("2026-01-01", periods=36, freq="6h", tz="UTC")
    performance = [0.01] * 6 + [-0.01] * 18 + [0.01] * 12
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "strategy": ["s1"] * len(ts),
            "performance": performance,
        }
    )

    bundle = build_strategy_performance_labels(
        frame,
        strategy_col="strategy",
        timestamp_col="timestamp",
        performance_col="performance",
        strategies=["s1"],
        ewma_halflife=4,
        anchor_mode="minmax",
        loss_streak_target_min_hours=72.0,
        loss_streak_target_full_hours=168.0,
        loss_streak_label_weight=1.0,
        loss_streak_sample_weight_multiplier=2.0,
    )
    labels = bundle.by_strategy["s1"]

    assert labels.loss_streak_hours.max() == 108.0
    assert labels.loss_streak_bad_pressure.max() > 0.0
    assert labels.loss_streak_bad_pressure.loc[ts[16]] == 0.0
    assert labels.loss_streak_bad_pressure.loc[ts[23]] > 0.0
    assert labels.bad_sample_weight.loc[ts[23]] > labels.bad_sample_weight.loc[ts[16]]
    assert bundle.diagnostics["loss_streak_target_min_hours"].iloc[0] == 72.0


def test_strategy_performance_loss_streaks_carry_across_sparse_active_losses():
    ts = pd.date_range("2026-01-01", periods=9, freq="24h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": [ts[0], ts[3], ts[6], ts[8]] + list(ts),
            "strategy": ["s1", "s1", "s1", "s1"] + ["s2"] * len(ts),
            "performance": [-0.01, -0.02, -0.03, 0.01] + [0.0] * len(ts),
        }
    )

    bundle = build_strategy_performance_labels(
        frame,
        strategy_col="strategy",
        timestamp_col="timestamp",
        performance_col="performance",
        strategies=["s1"],
        ewma_halflife=4,
        anchor_mode="minmax",
        loss_streak_target_min_hours=72.0,
        loss_streak_target_full_hours=168.0,
    )
    labels = bundle.by_strategy["s1"]

    assert labels.loss_streak_hours.loc[ts[3]] >= 96.0
    assert labels.loss_streak_hours.loc[ts[6]] >= 168.0
    assert labels.loss_streak_bad_pressure.loc[ts[6]] == 1.0
    assert labels.loss_streak_hours.loc[ts[8]] == 0.0


def test_strategy_performance_labels_support_density_drawdown_utility_and_cooldown_pressures():
    ts = pd.date_range("2026-01-01", periods=16, freq="h", tz="UTC")
    performance = [0.02, 0.01, 0.0, -0.01, -0.02, 0.0, -0.03, -0.02] + [0.01] * 8
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "strategy": ["s1"] * len(ts),
            "performance": performance,
        }
    )

    bundle = build_strategy_performance_labels(
        frame,
        strategy_col="strategy",
        timestamp_col="timestamp",
        performance_col="performance",
        strategies=["s1"],
        ewma_halflife=2,
        anchor_mode="minmax",
        loss_streak_label_weight=0.0,
        risk_label_modes=["density", "drawdown", "utility", "cooldown"],
        rolling_bad_regime_windows_hours=(4.0, 8.0),
        loss_density_label_weight=1.0,
        loss_density_min_negative_share=0.40,
        loss_density_full_negative_share=0.75,
        drawdown_label_weight=0.75,
        utility_label_weight=0.75,
        cooldown_label_weight=1.0,
        cooldown_hours=3.0,
        cooldown_trigger=0.50,
    )
    labels = bundle.by_strategy["s1"]

    assert labels.loss_density_bad_pressure.max() > 0.0
    assert labels.drawdown_bad_pressure.max() > 0.0
    assert labels.utility_bad_pressure.max() > 0.0
    assert labels.cooldown_bad_pressure.max() > 0.0
    assert labels.composite_bad_pressure.max() > 0.0
    assert labels.bad_label.loc[labels.composite_bad_pressure.idxmax()] >= 0.5
    assert bundle.diagnostics["composite_bad_pressure_share"].iloc[0] > 0.0


def test_strategy_performance_labels_can_use_forward_bad_target_without_feature_pressure():
    ts = pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC")
    performance = [0.01, 0.01, 0.0, 0.0, -0.03, -0.02, -0.01, -0.02, 0.01, 0.01]
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "strategy": ["s1"] * len(ts),
            "performance": performance,
        }
    )

    bundle = build_strategy_performance_labels(
        frame,
        strategy_col="strategy",
        timestamp_col="timestamp",
        performance_col="performance",
        strategies=["s1"],
        ewma_halflife=2,
        anchor_mode="minmax",
        loss_streak_label_weight=0.0,
        risk_label_modes=["forward"],
        forward_bad_label_weight=1.0,
        forward_bad_window_hours=4.0,
        loss_density_min_negative_share=0.50,
        loss_density_full_negative_share=0.75,
    )
    labels = bundle.by_strategy["s1"]

    assert labels.forward_bad_pressure.loc[ts[3]] > 0.0
    assert labels.bad_label.loc[ts[3]] >= 0.5
    assert bundle.diagnostics["forward_bad_pressure_share"].iloc[0] > 0.0


def test_market_state_feature_matrix_reports_missing_families_and_aggregates_timestamps():
    ts = pd.to_datetime(["2026-01-01 00:00Z", "2026-01-01 00:00Z", "2026-01-01 01:00Z"])
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "x": [1.0, 3.0, 5.0],
            "y": [0.0, np.nan, -2.0],
        }
    )
    matrix = build_market_state_feature_matrix(
        frame,
        timestamp_col="timestamp",
        feature_families={"meta": ["x", "missing"], "drift": ["y"]},
        aggregation_config={"*": ["mean", "fraction_missing", "breadth_above_threshold"]},
    )

    assert "missing" in matrix.missing_families["meta"]
    assert "meta__x__mean" in matrix.X.columns
    assert matrix.X.loc[ts[0], "meta__x__mean"] == 2.0
    assert matrix.X.loc[ts[0], "drift__y__fraction_missing"] == 0.5
    assert set(matrix.diagnostics["family"]) == {"meta", "drift"}


def test_frozen_feature_pipeline_preserves_train_columns_on_validation():
    train = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
        }
    )
    valid = train.drop(columns=["y"]).copy()
    matrix, artifact = fit_market_state_feature_pipeline(
        train,
        timestamp_col="timestamp",
        feature_families={"meta": ["x", "y"]},
        aggregation_config={"*": ["mean"]},
    )

    frozen = apply_frozen_feature_pipeline(valid, artifact)

    assert list(frozen.X.columns) == list(matrix.X.columns)
    assert "meta__y__mean" in frozen.X.columns
    assert frozen.X["meta__y__mean"].isna().all()
