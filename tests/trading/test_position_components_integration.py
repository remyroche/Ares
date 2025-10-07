import pytest

from src.tactician.position_division_strategy import PositionDivisionStrategy
from src.tactician.position_monitor import PositionMonitor


@pytest.fixture
def extended_config(tmp_path):
    return {
        "position_monitor": {
            "profit_taking": {
                "confidence_scaling": True,
                "min_confidence_for_profit": 0.55,
                "confidence_profit_multiplier": 0.45,
                "tiered_profit_taking": True,
                "trailing_stop_enabled": True,
            },
            "trailing_stop": {
                "enabled": True,
                "atr_multiplier": 1.6,
                "min_distance": 0.01,
                "confidence_activation": 0.65,
            },
        },
        "position_division_strategy": {
            "max_positions": 4,
            "position_size_limit": 0.22,
            "take_profit_pct": 0.025,
            "stop_loss_pct": 0.012,
            "directional_allocations": {"long": 0.6, "short": 0.4},
            "volatility_overrides": {"high": 0.18, "low": 0.1},
        },
    }


def test_position_monitor_ingests_optimizer_schema(extended_config):
    monitor = PositionMonitor(extended_config)

    optimizer_payload = {
        "confidence_very_low": 0.18,
        "confidence_low": 0.32,
        "confidence_medium": 0.55,
        "confidence_high": 0.82,
        "base_profit_target": 0.045,
        "min_confidence_for_profit": 0.6,
        "confidence_profit_multiplier": 0.35,
        "profit_tier_1": 0.3,
        "profit_tier_2": 0.55,
        "profit_tier_3": 0.8,
        "base_stop_loss": -0.045,
        "atr_multiplier": 1.8,
        "volatility_adjustment_factor": 1.2,
        "max_hold_time": 5400,
        "min_hold_time": 240,
        "confidence_time_scaling_factor": 1.1,
        "trailing_atr_multiplier": 2.1,
        "trailing_min_distance": 0.012,
        "trailing_confidence_activation": 0.68,
        "regime_transition_penalty": 0.12,
        "regime_specific_scaling": 1.05,
    }

    converted = monitor._convert_optimization_results(optimizer_payload)
    monitor.optimized_parameters = converted
    monitor.confidence_thresholds = monitor._get_optimized_confidence_thresholds()
    monitor.pnl_thresholds = monitor._get_optimized_pnl_thresholds()
    monitor.profit_taking_config = monitor._get_optimized_profit_taking_config()
    monitor.trailing_stop_config = monitor._get_optimized_trailing_stop_config()
    monitor.regime_aware_config = monitor._get_optimized_regime_aware_config()

    assert monitor.confidence_thresholds["high"] == pytest.approx(0.82)
    assert monitor.pnl_thresholds["profit_target"] == pytest.approx(0.045)
    assert monitor.trailing_stop_config["atr_multiplier"] == pytest.approx(2.1)
    assert monitor.trailing_stop_config["min_distance"] == pytest.approx(0.012)
    assert monitor.regime_aware_config["transition_penalty"] == pytest.approx(0.12)


def test_position_division_strategy_schema_ingestion(extended_config):
    strategy = PositionDivisionStrategy(extended_config)

    assert strategy.max_positions == 4
    assert strategy.position_size_limit == pytest.approx(0.22)
    assert "directional_allocations" in strategy.strategy_config
    assert strategy.strategy_config["volatility_overrides"]["high"] == pytest.approx(0.18)

    assert strategy._validate_configuration() is True

    sizes = strategy._calculate_position_sizes(total_capital=100000, num_positions=3, confidence_score=0.7)
    assert len(sizes) == 3
    assert sizes[0] > sizes[1]  # confidence tilt applied

    tp_sl_levels = strategy._calculate_tp_sl_levels({"volatility": 0.03, "regime": "trend"})
    assert len(tp_sl_levels["take_profit"]) == strategy.max_positions
    assert len(tp_sl_levels["stop_loss"]) == strategy.max_positions
