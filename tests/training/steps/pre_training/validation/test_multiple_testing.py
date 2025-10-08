from types import SimpleNamespace

import pytest
from src.training.steps.pre_training.validation.schemas import (
    apply_multiple_testing_correction,
    report_hypothesis_count,
    track_and_control_hypotheses,
)


def test_apply_multiple_testing_correction_enriches_metrics():
    metrics = {
        "short": {"p_value": 0.01},
        "medium": {"p_value": 0.04},
        "long": {"p_value": 0.25},
        "diagnostic": {"statistic": 1.0},
    }

    corrected = apply_multiple_testing_correction(metrics, alpha=0.05)

    assert set(corrected.keys()) == {"short", "medium", "long", "diagnostic"}
    assert corrected["short"]["adjusted_p_value"] == pytest.approx(0.03, rel=1e-9)
    assert corrected["short"]["reject_null_corrected"] is True
    assert corrected["medium"]["adjusted_p_value"] == pytest.approx(0.06, rel=1e-9)
    assert corrected["medium"]["reject_null_corrected"] is False

    metadata = corrected["diagnostic"]["multiple_testing_correction"]
    assert metadata["adjustment_applied"] is False
    assert metadata["hypothesis_count"] == 3


def test_track_and_control_hypotheses_includes_corrected_metrics():
    horizon_metrics = {
        "short": {"p_value": 0.01},
        "medium": {"p_value": 0.02},
    }

    report = track_and_control_hypotheses(
        horizon_results=horizon_metrics,
        feature_results={"feature_alpha": 0.03},
        lookback_results={"lookback_beta": 0.5},
    )

    corrected = report.get("corrected_horizon_metrics")
    assert corrected is not None
    assert corrected["short"]["reject_null_corrected"] is True
    assert corrected["short"]["multiple_testing_correction"]["hypothesis_count"] == 2


def test_report_hypothesis_count_matches_product():
    horizon_weights = SimpleNamespace(micro=0.0, small=0.4, medium=0.35, high=0.25)
    transaction_costs = SimpleNamespace(scenarios=[{"maker_fee": 0.0002}, {"maker_fee": 0.0004}])
    regime_config = SimpleNamespace(regimes=["bull", "bear", "crab"])
    config = SimpleNamespace(
        horizon_weights=horizon_weights,
        transaction_costs=transaction_costs,
        enable_regime_aware_labeling=True,
        regime_config=regime_config,
    )

    statistics = report_hypothesis_count(config)

    assert statistics["horizon_count"] == 3
    assert statistics["transaction_cost_scenarios"] == 2
    assert statistics["regime_configurations"] == 3
    assert statistics["total_hypotheses"] == 18
    assert statistics["bonferroni_threshold"] == pytest.approx(0.05 / 18)
