from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.audit_cross_era_tail_payoff_challenger_heads import (
    attach_exact_1m_event_targets,
    pinball_loss,
    quantile_metrics,
    _finite_binary_metrics,
)


def _identity() -> dict[str, list[object]]:
    return {
        "candidate_id": ["a", "b", "c", "d"],
        "__ts__": pd.date_range("2026-07-20", periods=4, freq="h", tz="UTC"),
        "__symbol__": ["A/USD:USD"] * 4,
        "side_name": ["long"] * 4,
    }


def test_exact_1m_join_derives_economic_competing_risk_classes() -> None:
    predictions = pd.DataFrame({**_identity(), "positive_net": [1, 0, 0, 0]})
    labels = pd.DataFrame({
        **_identity(),
        "__soft_tb_first_event__": ["timeout", "adverse_first_or_conflict", "timeout", "favorable_first"],
    })
    actual = attach_exact_1m_event_targets(predictions, labels)
    assert actual["event_code"].tolist() == [0, 1, 2, 3]
    assert set(actual["event_target_origin"]) == {"harmonized_exact_1m"}


def test_exact_1m_join_requires_complete_identity_coverage() -> None:
    predictions = pd.DataFrame({**_identity(), "positive_net": [0, 0, 0, 0]})
    labels = pd.DataFrame({
        **{key: value[:3] for key, value in _identity().items()},
        "__soft_tb_first_event__": ["timeout"] * 3,
    })
    with pytest.raises(ValueError, match="coverage"):
        attach_exact_1m_event_targets(predictions, labels)


def test_probability_metrics_report_perfect_ranking_and_calibration() -> None:
    actual = _finite_binary_metrics(np.array([0, 0, 1, 1]), np.array([0.0, 0.0, 1.0, 1.0]))
    assert actual["auc"] == pytest.approx(1.0)
    assert actual["pr_auc"] == pytest.approx(1.0)
    assert actual["brier"] == pytest.approx(0.0)
    assert actual["ece_10"] == pytest.approx(0.0)


def test_quantile_metrics_measure_skill_and_signed_decile_payoff() -> None:
    target = np.array([1.0, 2.0, 3.0, 4.0])
    prediction = target.copy()
    signed_net = target.copy()
    actual = quantile_metrics(target, prediction, signed_net, 0.5)
    assert pinball_loss(target, prediction, 0.5) == pytest.approx(0.0)
    assert actual["baseline_pinball_skill"] == pytest.approx(1.0)
    assert actual["conditional_spearman"] == pytest.approx(1.0)
    assert actual["top_decile_realized_payoff_bps"] == pytest.approx(4.0)
    assert actual["bottom_decile_realized_payoff_bps"] == pytest.approx(1.0)
