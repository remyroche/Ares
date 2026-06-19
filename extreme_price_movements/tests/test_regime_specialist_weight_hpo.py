from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.regime_specialist_weight_hpo import (
    RegimeSpecialistWeightHPOConfig,
    RegimeSpecialistWeightHPOSpace,
    hpo_config_from_mapping,
    hpo_space_from_mapping,
    optimize_regime_specialist_weight_hpo,
    precision_score_top_fracs,
    score_regime_specialist_weight_trial,
    suggest_weight_config,
)


class _FakeTrial:
    def __init__(self) -> None:
        self.calls: dict[str, tuple[float, float, bool]] = {}

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        self.calls[name] = (float(low), float(high), bool(log))
        if log:
            return float(np.sqrt(float(low) * float(high)))
        return float((float(low) + float(high)) / 2.0)


def test_suggest_weight_config_uses_continuous_tpe_ranges():
    trial = _FakeTrial()
    cfg = suggest_weight_config(trial, space=RegimeSpecialistWeightHPOSpace())

    assert trial.calls["analogue_gamma"] == (1.5, 3.0, False)
    assert trial.calls["replay_gamma"] == (1.5, 3.0, False)
    assert "recency_power" not in trial.calls
    assert trial.calls["tau_adaptive"] == (10000.0, 40000.0, True)
    assert trial.calls["tau_replay"] == (25000.0, 100000.0, True)
    assert trial.calls["min_current_plus_analogue_mass"] == (0.50, 0.60, False)
    assert trial.calls["less_interesting_max_mass"] == (0.30, 0.50, False)
    assert cfg.tau_current == pytest.approx(cfg.tau_analogue)
    assert cfg.tau_normal == pytest.approx(cfg.tau_irrelevant)
    assert 1.5 <= cfg.analogue_gamma <= 3.0
    assert 1.5 <= cfg.replay_gamma <= 3.0
    assert cfg.recency_power == pytest.approx(0.5)


def test_hpo_config_and_space_can_be_loaded_from_project_mapping():
    cfg = {
        "lgbm_regime_specialist_weight_hpo_trials": 17,
        "lgbm_regime_specialist_weight_hpo_early_stop_patience": 6,
        "lgbm_regime_specialist_weight_hpo_random_state": 123,
        "lgbm_regime_specialist_weight_hpo_return_scale": 10000.0,
        "lgbm_regime_specialist_weight_hpo_analogue_gamma_low": 1.7,
        "lgbm_regime_specialist_weight_hpo_analogue_gamma_high": 2.7,
        "lgbm_regime_specialist_weight_hpo_tau_adaptive_low": 9000.0,
        "lgbm_regime_specialist_weight_hpo_tau_adaptive_high": 30000.0,
        "lgbm_regime_specialist_weight_hpo_min_total_n_eff_reliability": 0.61,
        "lgbm_regime_specialist_recency_power": 0.31,
    }

    hpo_cfg = hpo_config_from_mapping(cfg)
    space = hpo_space_from_mapping(cfg)

    assert hpo_cfg.n_trials == 17
    assert hpo_cfg.early_stop_patience == 6
    assert hpo_cfg.random_state == 123
    assert hpo_cfg.return_scale == pytest.approx(10000.0)
    assert space.analogue_gamma_low == pytest.approx(1.7)
    assert space.analogue_gamma_high == pytest.approx(2.7)
    assert space.tau_adaptive_low == pytest.approx(9000.0)
    assert space.tau_adaptive_high == pytest.approx(30000.0)
    assert hpo_cfg.min_total_n_eff_reliability == pytest.approx(0.61)
    assert space.recency_power == pytest.approx(0.31)


def test_precision_score_top_fracs_blends_top_10_20_30_percent():
    y = np.asarray([1, 1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    score = np.asarray([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.float32)

    out = precision_score_top_fracs(y, score)

    assert out["p_at_10"] == pytest.approx(1.0)
    assert out["p_at_20"] == pytest.approx(1.0)
    assert out["p_at_30"] == pytest.approx(2.0 / 3.0)
    assert out["precision_score"] == pytest.approx(0.25 * 1.0 + 0.50 * 1.0 + 1.00 * (2.0 / 3.0))


def test_weight_hpo_objective_uses_requested_recent_window_terms():
    n = 40
    ts = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
    y = np.zeros(n, dtype=np.float32)
    score = np.arange(n, dtype=np.float32)
    returns = np.zeros(n, dtype=np.float32)
    # Last 28 days have high-score positives and positive top-30 returns.
    y[-8:] = 1.0
    returns[-8:] = 0.02

    out = score_regime_specialist_weight_trial(
        y,
        score,
        returns,
        ts,
        config=RegimeSpecialistWeightHPOConfig(
            return_scale=1.0,
            concentration_penalty_weight=0.0,
            low_ess_penalty_weight=0.0,
            adaptive_floor_penalty_weight=0.0,
            replay_cap_penalty_weight=0.0,
            n_eff_reliability_penalty_weight=0.0,
            adaptive_n_eff_penalty_weight=0.0,
            current_focus_penalty_weight=0.0,
            recent_focus_penalty_weight=0.0,
        ),
    )

    assert out["rows_last_2w"] == 15
    assert out["rows_last_4w"] == 29
    assert out["precision_score_last_2w"]["precision_score"] > 0.0
    assert out["precision_score_last_4w"]["precision_score"] > 0.0
    assert out["mean_return_top30_last_4w"] > 0.0
    assert out["auc_last_4w"] > 0.5
    expected = (
        1.0 * out["precision_score_last_2w"]["precision_score"]
        + 0.5 * out["precision_score_last_4w"]["precision_score"]
        + 0.5 * out["mean_return_top30_last_4w"]
        + 0.25 * out["auc_last_4w"]
    )
    assert out["objective_value"] == pytest.approx(expected)


def test_weight_hpo_penalizes_low_n_eff_and_weak_recent_focus():
    n = 40
    ts = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
    y = np.ones(n, dtype=np.float32)
    score = np.arange(n, dtype=np.float32)
    returns = np.ones(n, dtype=np.float32) * 0.01
    weights = np.ones(n, dtype=np.float32)
    weights[-28:] = 0.01

    weak = score_regime_specialist_weight_trial(
        y,
        score,
        returns,
        ts,
        sample_weight=weights,
        weight_diagnostics={
            "adaptive_n_eff_reliability": 0.05,
            "replay_n_eff_reliability": 0.10,
            "actual_current_weight_mass": 0.02,
            "actual_analogue_weight_mass": 0.10,
            "actual_less_interesting_weight_mass": 0.40,
            "less_interesting_mass_cap": 0.50,
            "min_current_plus_analogue_mass": 0.50,
        },
        config=RegimeSpecialistWeightHPOConfig(
            return_scale=0.0,
            concentration_penalty_weight=0.0,
            low_ess_penalty_weight=0.0,
            adaptive_floor_penalty_weight=0.0,
            replay_cap_penalty_weight=0.0,
            min_total_n_eff_reliability=0.50,
            min_adaptive_n_eff_reliability=0.25,
            min_current_weight_mass=0.08,
            min_recent_4w_weight_mass=0.20,
        ),
    )

    assert weak["recent_4w_weight_mass"] < 0.20
    assert weak["penalties"]["total_n_eff_reliability_penalty"] > 0.0
    assert weak["penalties"]["adaptive_n_eff_reliability_penalty"] > 0.0
    assert weak["penalties"]["current_focus_penalty"] > 0.0
    assert weak["penalties"]["recent_4w_focus_penalty"] > 0.0


def test_optuna_weight_hpo_wrapper_runs_with_tpe_when_available():
    pytest.importorskip("optuna")

    def evaluator(weight_config, trial):
        value = -abs(float(weight_config.analogue_gamma) - 2.1)
        return {
            "objective_value": value,
            "dummy_metric": value,
        }

    result = optimize_regime_specialist_weight_hpo(
        evaluator,
        hpo_config=RegimeSpecialistWeightHPOConfig(
            n_trials=5,
            early_stop_patience=3,
            random_state=7,
        ),
    )

    assert result["sampler"] == "TPESampler"
    assert result["n_trials_requested"] == 5
    assert 1 <= result["completed_trials"] <= 5
    assert "analogue_gamma" in result["best_params"]
    assert "tau_adaptive" in result["best_params"]
