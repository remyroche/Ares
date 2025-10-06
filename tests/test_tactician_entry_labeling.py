import numpy as np
import pandas as pd
import pytest

from src.training.steps.models_training.tactician_pre_ml_orchestration import (
    TacticianDifferentiatedLabeler,
    TacticianLabelingConfig,
    TacticianPreMLConfig,
    TacticianPreMLOrchestrator,
)

def _sample_market_data(periods: int = 16) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="15min")
    base = 100 + np.linspace(0, 1.5, periods)
    oscillation = 0.4 * np.sin(np.linspace(0, 3 * np.pi, periods))
    close = base + oscillation
    high = close + 0.3
    low = close - 0.3
    open_price = close + 0.05

    data = pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.linspace(1200, 2200, periods)
    }, index=index)
    return data


def test_differentiated_labeler_creates_entry_labels_for_green_periods():
    data = _sample_market_data(20)
    signals = pd.Series([0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0], index=data.index)

    config = TacticianLabelingConfig(
        min_entry_window_minutes=1,
        entry_quality_threshold=0.05,
        max_adverse_movement_pct=1.0,
        min_favorable_movement_pct=0.05,
    )
    labeler = TacticianDifferentiatedLabeler(config)

    labels, metrics = labeler.create_entry_timing_labels(data, signals)

    assert (labels > 0).any(), "Expected at least one positive entry label within green periods"
    assert metrics["overall_quality"] > 0, "Quality metric should reflect generated labels"


def test_differentiated_labeler_respects_regime_thresholds():
    data = _sample_market_data(12)
    signals = pd.Series(1, index=data.index)
    regime_series = pd.Series([0] * 6 + [1] * 6, index=data.index)

    config = TacticianLabelingConfig(
        min_entry_window_minutes=1,
        entry_quality_threshold=0.05,
        max_adverse_movement_pct=1.0,
        min_favorable_movement_pct=0.05,
        regime_specific_thresholds={
            "regime_1": {
                "max_adverse_movement_pct": 0.01,
                "min_favorable_movement_pct": 0.05,
            }
        }
    )

    labeler = TacticianDifferentiatedLabeler(config)
    labels, _ = labeler.create_entry_timing_labels(data, signals, regime_series)

    positive_indices = labels[labels > 0].index
    assert all(idx < data.index[6] for idx in positive_indices), "Regime-specific thresholds should suppress later entries"


def test_orchestrator_builds_entry_label_artifacts():
    data = _sample_market_data(18)
    analyst_predictions = pd.DataFrame({
        "confidence": np.linspace(0.3, 0.7, len(data)),
        "green_light": ([0, 0] + [1] * (len(data) - 4) + [0, 0])[:len(data)]
    }, index=data.index)

    config = TacticianPreMLConfig(
        analyst_confidence_threshold=0.2,
        labeling_config=TacticianLabelingConfig(min_entry_window_minutes=1, entry_quality_threshold=0.05)
    )
    orchestrator = TacticianPreMLOrchestrator(config)

    bundle = orchestrator._create_entry_label_artifacts(data, analyst_predictions, None)
    assert bundle is not None, "Expected entry labeling artifacts to be created"

    artifacts = bundle["artifacts"]["multi_horizon_labeling_result"]
    labels = artifacts["labels"]
    assert "tactician_entry_target" in labels.columns
    assert artifacts["method"] == "tactician_entry_labeling"
    assert artifacts["quality_scores"]["tactician_entry_target"]["overall_quality"] == pytest.approx(
        bundle["quality_metrics"]["overall_quality"], rel=1e-6
    )


