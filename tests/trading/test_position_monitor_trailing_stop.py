import numpy as np
import pytest

from src.tactician.position_monitor import PositionAction, PositionMonitor


@pytest.fixture
def monitor_config():
    return {
        "position_monitor": {
            "profit_taking": {
                "confidence_scaling": True,
                "min_confidence_for_profit": 0.5,
                "confidence_profit_multiplier": 0.4,
                "tiered_profit_taking": True,
                "trailing_stop_enabled": True,
            },
            "trailing_stop": {
                "enabled": True,
                "atr_multiplier": 1.0,
                "min_distance": 0.5,
                "confidence_activation": 0.6,
            },
        }
    }


def test_trailing_stop_long_tightens_threshold(monitor_config):
    monitor = PositionMonitor(monitor_config)

    position_data = {
        "position_id": "long-1",
        "symbol": "ETHUSDT",
        "side": "LONG",
        "entry_price": 100.0,
        "current_price": 115.0,
        "quantity": 1.0,
        "atr_series": np.array([0.8, 0.9, 1.0]),
        "sigma_series": np.array([0.4, 0.5, 0.45]),
        "trailing_stop_state": {
            "level": 108.0,
            "peak_price": 110.0,
        },
    }

    action, reason = monitor._evaluate_trailing_stop(position_data, combined_confidence=0.85)

    assert action is PositionAction.TRAILING_STOP
    assert "Long trailing stop updated" in reason
    assert position_data["trailing_stop_state"]["level"] > 108.0
    assert position_data["trailing_stop_state"]["peak_price"] == pytest.approx(115.0)


def test_trailing_stop_short_tightens_threshold(monitor_config):
    monitor = PositionMonitor(monitor_config)

    position_data = {
        "position_id": "short-1",
        "symbol": "BTCUSDT",
        "side": "SHORT",
        "entry_price": 100.0,
        "current_price": 88.0,
        "quantity": 1.0,
        "atr_series": np.array([1.1, 1.0, 0.9]),
        "sigma_series": np.array([0.6, 0.55, 0.5]),
        "trailing_stop_state": {
            "level": 105.0,
            "trough_price": 92.0,
        },
    }

    action, reason = monitor._evaluate_trailing_stop(position_data, combined_confidence=0.9)

    assert action is PositionAction.TRAILING_STOP
    assert "Short trailing stop updated" in reason
    assert position_data["trailing_stop_state"]["level"] < 105.0
    assert position_data["trailing_stop_state"]["trough_price"] == pytest.approx(88.0)


def test_trailing_stop_take_profit_activation(monitor_config):
    monitor = PositionMonitor(monitor_config)

    position_data = {
        "position_id": "long-2",
        "symbol": "ETHUSDT",
        "side": "LONG",
        "entry_price": 100.0,
        "current_price": 116.0,
        "quantity": 1.0,
        "atr_series": np.array([0.7, 0.8, 0.75]),
        "sigma_series": np.array([0.3, 0.35, 0.32]),
    }

    action, _ = monitor._evaluate_trailing_stop(position_data, combined_confidence=0.82)
    assert action is PositionAction.TRAILING_STOP

    trailing_level = position_data["trailing_stop_state"]["level"]
    position_data["current_price"] = trailing_level - 0.05

    action, reason = monitor._evaluate_trailing_stop(position_data, combined_confidence=0.82)

    assert action is PositionAction.TAKE_PROFIT
    assert "trailing stop triggered" in reason.lower()
    assert position_data["trailing_stop_state"].get("triggered") is True
