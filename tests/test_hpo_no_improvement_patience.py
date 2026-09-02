"""Regression tests for the shared bounded-HPO stop contract."""

from __future__ import annotations

import optuna

from extreme_price_movements.residual_lambdarank_hpo import stop_after_no_improvement


def test_stops_after_twenty_non_improving_completed_trials():
    study = optuna.create_study(direction="maximize")

    def objective(trial: optuna.Trial) -> float:
        return 1.0 if trial.number == 0 else 0.0

    study.optimize(objective, n_trials=100, callbacks=[stop_after_no_improvement(patience=20)])

    assert len(study.trials) == 21  # one strict improvement, then twenty misses
    assert study.user_attrs["stop_reason"] == "no_improvement_patience"
    assert study.user_attrs["trials_since_improvement"] == 20


def test_pruned_trial_consumes_no_improvement_budget():
    study = optuna.create_study(direction="maximize")

    def objective(trial: optuna.Trial) -> float:
        if trial.number == 0:
            return 1.0
        raise optuna.TrialPruned()

    study.optimize(objective, n_trials=100, callbacks=[stop_after_no_improvement(patience=20)])

    assert len(study.trials) == 21
    assert study.user_attrs["stop_reason"] == "no_improvement_patience"
