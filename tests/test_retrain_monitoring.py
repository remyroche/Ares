import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import importlib.util

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "monitoring" / "retrain_monitoring.py"
spec = importlib.util.spec_from_file_location("retrain_monitoring", MODULE_PATH)
retrain_monitoring = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(retrain_monitoring)

MonitoringConfig = retrain_monitoring.MonitoringConfig
MonitoringStatus = retrain_monitoring.MonitoringStatus
MonitoringMetrics = retrain_monitoring.MonitoringMetrics
RetrainDecisionTree = retrain_monitoring.RetrainDecisionTree
RetrainTrigger = retrain_monitoring.RetrainTrigger


def _make_metrics(calibration_loss: float, psi_scores=None) -> MonitoringMetrics:
    return MonitoringMetrics(
        timestamp=datetime.now(),
        calibration_loss=calibration_loss,
        psi_scores=psi_scores or {},
        correlation_drift=0.0,
        latency_p95=0.0,
        latency_p99=0.0,
        missing_data_pct=0.0,
        status=MonitoringStatus.HEALTHY,
        alerts=[],
    )


def test_calibration_retrain_requires_three_consecutive_breaches():
    config = MonitoringConfig(calibration_loss_threshold=2.0, scheduled_retrain_time="")
    decision_tree = RetrainDecisionTree(config)
    calibration_monitor = decision_tree.calibration_monitor

    metrics = _make_metrics(calibration_loss=config.calibration_loss_threshold + 0.5)

    calibration_monitor.consecutive_breaches = 0
    assert not decision_tree.should_retrain(metrics).should_retrain

    calibration_monitor.consecutive_breaches = 2
    assert not decision_tree.should_retrain(metrics).should_retrain

    calibration_monitor.consecutive_breaches = 3
    decision = decision_tree.should_retrain(metrics)

    assert decision.should_retrain
    assert decision.trigger == RetrainTrigger.CALIBRATION_LOSS


def test_scheduled_retrain_respects_configured_time():
    config = MonitoringConfig(
        scheduled_retrain_time="02:00",
        scheduled_retrain_timezone="UTC",
    )
    decision_tree = RetrainDecisionTree(config)
    tz = ZoneInfo("UTC")

    decision_tree.last_retrain = datetime(2023, 12, 31, 2, 5, tzinfo=tz)

    before_window = datetime(2024, 1, 1, 1, 59, tzinfo=tz)
    assert not decision_tree._is_scheduled_retrain_time(before_window)

    scheduled_window = datetime(2024, 1, 1, 2, 5, tzinfo=tz)
    assert decision_tree._is_scheduled_retrain_time(scheduled_window)

    decision_tree.last_retrain = scheduled_window
    later_same_day = datetime(2024, 1, 1, 3, 0, tzinfo=tz)
    assert not decision_tree._is_scheduled_retrain_time(later_same_day)


def test_psi_triggers_only_for_monitored_columns():
    config = MonitoringConfig(psi_threshold=0.1, scheduled_retrain_time="")
    decision_tree = RetrainDecisionTree(config)
    psi_monitor = decision_tree.psi_monitor

    base = pd.DataFrame(
        {
            "σ_EW": np.linspace(0, 1, 1000),
            "vwap_dist": np.linspace(0, 1, 1000),
            "other": np.linspace(0, 1, 1000),
        }
    )
    psi_monitor.update_reference(base)

    # Drift isolated to an unmonitored column should not trigger PSI retrain.
    other_drift = base.copy()
    other_drift["other"] = np.linspace(5, 6, 1000)
    psi_scores_other = psi_monitor.calculate_psi(other_drift)
    assert set(psi_scores_other.keys()) <= {"σ_EW", "vwap_dist"}

    metrics = _make_metrics(calibration_loss=0.0, psi_scores=psi_scores_other)
    decision = decision_tree.should_retrain(metrics)
    assert not decision.should_retrain

    # Drift on a monitored column should trigger the PSI retrain decision.
    sigma_drift = base.copy()
    sigma_drift["σ_EW"] = np.linspace(5, 6, 1000)
    psi_scores_sigma = psi_monitor.calculate_psi(sigma_drift)
    assert psi_scores_sigma["σ_EW"] > config.psi_threshold

    metrics = _make_metrics(calibration_loss=0.0, psi_scores=psi_scores_sigma)
    decision = decision_tree.should_retrain(metrics)

    assert decision.should_retrain
    assert decision.trigger == RetrainTrigger.PSI_DRIFT
