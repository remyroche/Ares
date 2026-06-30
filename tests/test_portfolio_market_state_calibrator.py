import numpy as np
import pandas as pd

from extreme_price_movements.performance_regimes.archetypes import (
    build_cross_strategy_archetype_features,
)
from extreme_price_movements.performance_regimes.portfolio_calibration import (
    PortfolioCalibratorConfig,
    _max_consecutive_loss_hours,
    apply_portfolio_actions,
    build_portfolio_action_targets_from_labels,
    score_frozen_portfolio_calibrator,
    threshold_archetype_scores_for_modulation,
    train_portfolio_calibrator,
)
from extreme_price_movements.performance_regimes.labels import build_strategy_performance_labels


def test_portfolio_actions_allow_full_deactivation_without_final_clamp():
    idx = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    scores = pd.DataFrame({"s1": [0.9, 0.9, 0.1], "s2": [0.9, 0.1, 0.1]}, index=idx)
    actions = pd.DataFrame(
        {
            "s1__threshold_delta": [0.0, 0.0, 1.0],
            "s1__weight_log_delta": [0.0, 0.0, 0.0],
            "s1__activation_gate": [1.0, -1.0, -1.0],
            "s2__threshold_delta": [0.0, 1.0, 1.0],
            "s2__weight_log_delta": [0.0, 0.0, 0.0],
            "s2__activation_gate": [1.0, -1.0, -1.0],
        },
        index=idx,
    )

    weights = apply_portfolio_actions(
        scores,
        actions,
        base_thresholds={"s1": 0.5, "s2": 0.5},
        base_weights={"s1": 1.0, "s2": 1.0},
        activation_cutoffs={"s1": 0.0, "s2": 0.0},
        allow_cash=True,
        renormalize=True,
    )

    assert weights.iloc[0].sum() == 1.0
    assert weights.iloc[1].sum() == 0.0
    assert weights.iloc[2].sum() == 0.0


def test_rank_delta_participates_in_activation_before_weight_renormalization():
    idx = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    scores = pd.DataFrame({"s1": [0.9, 0.9]}, index=idx)
    ranks = pd.DataFrame({"s1": [0.8, 0.8]}, index=idx)
    actions = pd.DataFrame(
        {
            "s1__threshold_delta": [0.0, 0.0],
            "s1__rank_delta": [0.1, -0.2],
            "s1__weight_log_delta": [0.0, 0.0],
            "s1__activation_gate": [1.0, 1.0],
        },
        index=idx,
    )

    weights = apply_portfolio_actions(
        scores,
        actions,
        base_thresholds={"s1": 0.5},
        base_rank_thresholds={"s1": 0.85},
        strategy_ranks=ranks,
        base_weights={"s1": 1.0},
        allow_cash=True,
        renormalize=True,
    )

    assert weights.iloc[0]["s1"] == 1.0
    assert weights.iloc[1]["s1"] == 0.0


def test_portfolio_calibrator_scores_all_action_channels_and_cross_strategy_features():
    idx = pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC")
    intensities = pd.DataFrame(
        {
            "strategy_s1_bad_archetype_1": [0.0, 0.2, 0.8, 1.0, 0.3, 0.0],
            "strategy_s2_good_archetype_1": [0.5, 0.7, 0.9, 0.1, 0.0, 0.2],
        },
        index=idx,
    )
    metadata = pd.DataFrame(
        {
            "archetype_id": list(intensities.columns),
            "strategy": ["s1", "s2"],
            "direction": ["bad", "good"],
        }
    )
    cross = build_cross_strategy_archetype_features(intensities, metadata)
    calibrator = train_portfolio_calibrator(
        cross.X,
        strategies=["s1", "s2"],
        config=PortfolioCalibratorConfig(backend="linear"),
    )
    actions = score_frozen_portfolio_calibrator(cross.X, calibrator)

    assert {"bad_breadth", "good_breadth"}.issubset(cross.X.columns)
    for strategy in ["s1", "s2"]:
        for action in ["threshold_delta", "rank_delta", "weight_log_delta", "activation_gate"]:
            assert f"{strategy}__{action}" in actions.columns

    ebm_fallback = train_portfolio_calibrator(
        cross.X,
        strategies=["s1"],
        config=PortfolioCalibratorConfig(backend="ebm_gam"),
    )
    assert "effective_backend" in ebm_fallback.diagnostics.columns


def test_archetype_modulation_threshold_neutralizes_weak_activity_scores():
    idx = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    p_active = pd.DataFrame(
        {
            "strategy_s1_bad_archetype_1": [0.40, 0.54, 0.55, 0.90],
            "strategy_s1_good_archetype_1": [0.10, 0.70, 0.20, 0.80],
        },
        index=idx,
    )

    thresholded = threshold_archetype_scores_for_modulation(p_active, min_p_active=0.55)

    assert np.allclose(thresholded.p_active["strategy_s1_bad_archetype_1"], [0.0, 0.0, 0.55, 0.90])
    assert np.allclose(thresholded.activity_scores["strategy_s1_bad_archetype_1"], [0.0, 0.0, 0.10, 0.80])
    assert np.allclose(
        thresholded.modulation_scores["strategy_s1_bad_archetype_1"],
        [0.0, 0.0, 0.0, (0.90 - 0.55) / (1.0 - 0.55)],
    )
    assert thresholded.diagnostics["suppressed_share"].between(0.0, 1.0).all()


def test_archetype_modulation_threshold_keeps_minimum_active_share_when_floor_is_too_high():
    idx = pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC")
    p_active = pd.DataFrame(
        {
            "strategy_s1_bad_archetype_1": np.linspace(0.01, 0.30, len(idx)),
        },
        index=idx,
    )

    thresholded = threshold_archetype_scores_for_modulation(
        p_active,
        min_p_active=0.55,
        min_active_share=0.20,
    )

    diag = thresholded.diagnostics.iloc[0]
    assert diag["effective_min_p_active"] < 0.55
    assert diag["active_share_after_threshold"] >= 0.20
    assert thresholded.modulation_scores["strategy_s1_bad_archetype_1"].max() > 0.0


def test_archetype_modulation_threshold_can_require_strict_p_active_certainty():
    idx = pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC")
    p_active = pd.DataFrame(
        {
            "strategy_s1_bad_archetype_1": np.linspace(0.01, 0.30, len(idx)),
        },
        index=idx,
    )

    thresholded = threshold_archetype_scores_for_modulation(
        p_active,
        min_p_active=0.55,
        min_active_share=0.20,
        relax_floor_to_min_active_share=False,
    )

    diag = thresholded.diagnostics.iloc[0]
    assert diag["effective_min_p_active"] == 0.55
    assert not bool(diag["floor_relaxed_to_min_active_share"])
    assert diag["active_share_after_threshold"] == 0.0
    assert thresholded.modulation_scores["strategy_s1_bad_archetype_1"].max() == 0.0


def test_portfolio_calibrator_applies_archetype_threshold_during_train_and_score():
    idx = pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC")
    X = pd.DataFrame(
        {
            "strategy_s1_bad_archetype_1": [-0.5, 0.0, 0.05, 0.2, 0.7, 1.0],
            "plain_market_state": [0.0] * 6,
        },
        index=idx,
    )
    targets = {
        "s1": pd.DataFrame(
            {
                "threshold_delta": [0.0, 0.0, 0.0, 0.1, 0.3, 0.4],
                "rank_delta": [0.0] * 6,
                "weight_log_delta": [0.0] * 6,
                "activation_gate": [1.0] * 6,
            },
            index=idx,
        )
    }

    calibrator = train_portfolio_calibrator(
        X,
        strategies=["s1"],
        action_targets=targets,
        config=PortfolioCalibratorConfig(
            backend="linear",
            archetype_score_threshold=0.10,
            archetype_score_ramp_power=2.0,
            archetype_score_ramp_gain=3.0,
            archetype_base_p_active_floor=0.55,
        ),
    )
    actions = score_frozen_portfolio_calibrator(X, calibrator)

    assert "archetype_score_threshold" in calibrator.diagnostics.columns
    assert "archetype_score_ramp_power" in calibrator.diagnostics.columns
    assert "archetype_score_ramp_gain" in calibrator.diagnostics.columns
    assert "archetype_effective_p_active_threshold" in calibrator.diagnostics.columns
    assert np.allclose(calibrator.diagnostics["archetype_score_ramp_power"], 2.0)
    assert np.allclose(calibrator.diagnostics["archetype_score_ramp_gain"], 3.0)
    assert np.allclose(calibrator.diagnostics["archetype_effective_p_active_threshold"], 0.595)
    assert calibrator.diagnostics["archetype_feature_nonzero_share"].max() < 1.0
    assert actions["s1__threshold_delta"].iloc[2] <= actions["s1__threshold_delta"].iloc[3]


def test_consecutive_loss_streak_hours_uses_timestamp_bar_length():
    idx = pd.date_range("2026-01-01", periods=8, freq="6h", tz="UTC")
    returns = pd.Series([0.1, -0.1, -0.2, -0.1, 0.0, -0.1, -0.1, 0.2], index=idx)

    assert _max_consecutive_loss_hours(returns) == 18.0


def test_optuna_portfolio_objective_reports_hr_and_loss_streak_diagnostics():
    idx = pd.date_range("2026-01-01", periods=12, freq="6h", tz="UTC")
    X = pd.DataFrame(
        {
            "strategy_s1_good_archetype_1": np.linspace(0.0, 1.0, len(idx)),
            "strategy_s2_bad_archetype_1": np.linspace(1.0, 0.0, len(idx)),
        },
        index=idx,
    )
    targets = {
        "s1": pd.DataFrame(
            {
                "threshold_delta": 0.0,
                "rank_delta": 0.0,
                "weight_log_delta": np.linspace(-0.5, 0.8, len(idx)),
                "activation_gate": np.linspace(-1.0, 1.0, len(idx)),
            },
            index=idx,
        ),
        "s2": pd.DataFrame(
            {
                "threshold_delta": 0.0,
                "rank_delta": 0.0,
                "weight_log_delta": np.linspace(0.8, -0.5, len(idx)),
                "activation_gate": np.linspace(1.0, -1.0, len(idx)),
            },
            index=idx,
        ),
    }
    strategy_returns = pd.DataFrame(
        {
            "s1": [-0.05, -0.04, -0.03, -0.02, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.05, 0.04],
            "s2": [0.04, 0.03, 0.02, 0.01, -0.02, -0.03, -0.04, -0.05, -0.04, -0.03, -0.02, -0.01],
        },
        index=idx,
    )

    calibrator = train_portfolio_calibrator(
        X,
        strategies=["s1", "s2"],
        action_targets=targets,
        strategy_returns=strategy_returns,
        config=PortfolioCalibratorConfig(
            backend="optuna",
            optuna_trials=4,
            optuna_objective="hybrid",
            optuna_mse_weight=0.05,
            optuna_hit_rate_weight=2.0,
            optuna_loss_streak_weight=4.0,
            optuna_loss_streak_hours=24.0,
            optuna_cash_share_target=0.25,
            optuna_cash_share_weight=3.0,
            optuna_cash_share_excess_power=2.0,
            optuna_unjustified_deactivation_weight=5.0,
            optuna_unjustified_deactivation_gate_margin=0.10,
            optuna_tune_archetype_score_threshold=False,
            optuna_tune_archetype_score_ramp=False,
        ),
    )

    action_diag = calibrator.diagnostics[
        calibrator.diagnostics["action"].isin(["activation_gate", "weight_log_delta"])
    ]
    assert action_diag["effective_backend"].str.contains("optuna").any()
    assert action_diag["optuna_portfolio_hit_rate"].notna().any()
    assert action_diag["optuna_portfolio_max_loss_streak_hours"].notna().any()
    assert action_diag["optuna_portfolio_cash_share"].notna().any()
    assert action_diag["optuna_portfolio_cash_share_excess"].notna().any()
    assert action_diag["optuna_portfolio_cash_share_penalty"].notna().any()
    assert action_diag["optuna_portfolio_unjustified_deactivation_share"].notna().any()
    assert np.allclose(action_diag["optuna_cash_share_target"], 0.25)
    assert np.allclose(action_diag["optuna_cash_share_weight"], 3.0)
    assert np.allclose(action_diag["optuna_cash_share_excess_power"], 2.0)
    assert np.allclose(action_diag["optuna_unjustified_deactivation_weight"], 5.0)
    assert np.allclose(action_diag["optuna_unjustified_deactivation_gate_margin"], 0.10)
    assert "optuna_portfolio_active_utility_lcb" in action_diag.columns
    assert "optuna_portfolio_loss_density_excess_mean" in action_diag.columns
    assert action_diag["optuna_portfolio_active_utility_lcb"].notna().any()
    assert action_diag["optuna_portfolio_loss_density_excess_mean"].notna().any()


def test_portfolio_action_targets_from_labels_teach_deactivation_and_weight_changes():
    ts = pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC")
    trades = pd.DataFrame(
        {
            "timestamp": list(ts) * 2,
            "strategy": ["s1"] * len(ts) + ["s2"] * len(ts),
            "performance": [-2, -1, -0.5, 0, 0.5, 1, 1.5, 2] + [2, 1, 0.5, 0, -0.5, -1, -1.5, -2],
        }
    )
    labels = build_strategy_performance_labels(
        trades,
        strategy_col="strategy",
        timestamp_col="timestamp",
        performance_col="performance",
        strategies=["s1", "s2"],
        ewma_halflife=2,
        anchor_mode="minmax",
    )

    targets = build_portfolio_action_targets_from_labels(labels, ts, strategies=["s1", "s2"])

    assert set(targets.by_strategy) == {"s1", "s2"}
    assert (targets.by_strategy["s1"]["activation_gate"] < 0.0).any()
    assert targets.by_strategy["s1"]["weight_log_delta"].std() > 0.0
    assert targets.diagnostics["activation_target_deactivation_share"].between(0.0, 1.0).all()


def test_action_target_activation_quality_threshold_controls_deactivation_share():
    ts = pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC")
    trades = pd.DataFrame(
        {
            "timestamp": ts,
            "strategy": ["s1"] * len(ts),
            "performance": [-2, -1, -0.25, 0, 0.25, 1, 1.5, 2],
        }
    )
    labels = build_strategy_performance_labels(
        trades,
        strategy_col="strategy",
        timestamp_col="timestamp",
        performance_col="performance",
        strategies=["s1"],
        ewma_halflife=2,
        anchor_mode="minmax",
    )

    default_targets = build_portfolio_action_targets_from_labels(labels, ts, strategies=["s1"])
    sparse_deactivation_targets = build_portfolio_action_targets_from_labels(
        labels,
        ts,
        strategies=["s1"],
        activation_gate_quality_threshold=-0.25,
    )

    default_share = default_targets.diagnostics.loc[0, "activation_target_deactivation_share"]
    sparse_share = sparse_deactivation_targets.diagnostics.loc[0, "activation_target_deactivation_share"]
    assert sparse_share < default_share
    assert sparse_deactivation_targets.diagnostics.loc[0, "activation_gate_quality_threshold"] == -0.25


def test_action_targets_can_apply_bad_regime_pressure_penalties():
    ts = pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC")
    trades = pd.DataFrame(
        {
            "timestamp": ts,
            "strategy": ["s1"] * len(ts),
            "performance": [0.02, 0.01, -0.02, -0.03, -0.01, -0.02, 0.01, 0.01, -0.03, -0.02, 0.0, 0.01],
        }
    )
    labels = build_strategy_performance_labels(
        trades,
        strategy_col="strategy",
        timestamp_col="timestamp",
        performance_col="performance",
        strategies=["s1"],
        ewma_halflife=2,
        anchor_mode="minmax",
        loss_streak_label_weight=0.0,
        risk_label_modes=["density"],
        loss_density_label_weight=1.0,
        rolling_bad_regime_windows_hours=(4.0,),
        loss_density_min_negative_share=0.40,
        loss_density_full_negative_share=0.75,
    )

    neutral = build_portfolio_action_targets_from_labels(labels, ts, strategies=["s1"])
    penalized = build_portfolio_action_targets_from_labels(
        labels,
        ts,
        strategies=["s1"],
        bad_regime_threshold_penalty_scale=0.25,
        bad_regime_rank_penalty_scale=0.25,
        bad_regime_weight_penalty_scale=1.0,
        bad_regime_activation_penalty_scale=1.0,
    )
    pressure_idx = labels.by_strategy["s1"].composite_bad_pressure.idxmax()

    assert penalized.by_strategy["s1"].loc[pressure_idx, "threshold_delta"] >= neutral.by_strategy["s1"].loc[
        pressure_idx, "threshold_delta"
    ]
    assert penalized.by_strategy["s1"].loc[pressure_idx, "weight_log_delta"] <= neutral.by_strategy["s1"].loc[
        pressure_idx, "weight_log_delta"
    ]
    assert penalized.by_strategy["s1"].loc[pressure_idx, "activation_gate"] <= neutral.by_strategy["s1"].loc[
        pressure_idx, "activation_gate"
    ]
    assert penalized.diagnostics.loc[0, "bad_regime_pressure_active_share"] > 0.0
