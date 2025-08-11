# src/config/trading.py

from typing import Any

from src.config.environment import get_environment_settings


def get_trading_config() -> dict[str, Any]:
    """
    Get the complete trading configuration.

    Returns:
        dict: Complete trading configuration
    """
    settings = get_environment_settings()

    return {
        # --- Basic Trading Parameters ---
        "trading_symbol": settings.trade_symbol,
        "exchange_name": settings.exchange_name,
        "trading_interval": settings.timeframe,
        "initial_equity": settings.initial_equity,
        "taker_fee": 0.0004,
        "maker_fee": 0.0002,
        "state_file": "ares_state.json",
        "lookback_years": 2,  # 2 years of historical data
        # --- Exchange Configurations ---
        "exchanges": {
            "binance": {
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "api_key": None,  # Will be set dynamically
                "api_secret": None,  # Will be set dynamically
            },
            "gateio": {
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "api_key": settings.gateio_api_key,
                "api_secret": settings.gateio_api_secret,
            },
            "mexc": {
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "api_key": settings.mexc_api_key,
                "api_secret": settings.mexc_api_secret,
            },
            "okx": {
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "api_key": settings.okx_api_key,
                "api_secret": settings.okx_api_secret,
                "password": settings.okx_password,
            },
        },
        # --- Risk Management Configuration ---
        "risk_management": {
            "max_position_size": 0.3,  # Maximum position size as fraction of portfolio (30%)
            "max_daily_loss": 0.1,  # Maximum daily loss as fraction of portfolio (10%)
            "max_drawdown": 0.50,  # Maximum drawdown before stopping (50%)
            "kill_switch_threshold": 0.50,  # Loss threshold for kill switch (50%)
            "position_sizing": {
                "confidence_based_scaling": True,  # Enable confidence-based position sizing
                "base_position_size": 0.05,  # Base position size (5% of portfolio)
                "max_positions_per_signal": 5,  # Maximum number of positions for same signal
                "confidence_thresholds": {
                    "low_confidence": 0.6,  # Confidence threshold for low confidence
                    "medium_confidence": 0.75,  # Confidence threshold for medium confidence
                    "high_confidence": 0.85,  # Confidence threshold for high confidence
                    "very_high_confidence": 0.95,  # Confidence threshold for very high confidence
                },
                "position_size_multipliers": {
                    "low_confidence": 0.5,  # 50% of base size for low confidence
                    "medium_confidence": 1.0,  # 100% of base size for medium confidence
                    "high_confidence": 1.5,  # 150% of base size for high confidence
                    "very_high_confidence": 2.0,  # 200% of base size for very high confidence
                },
                "successive_position_rules": {
                    "enable_successive_positions": True,  # Enable multiple positions for high confidence
                    "min_confidence_for_successive": 0.85,  # Minimum confidence for successive positions
                    "max_successive_positions": 3,  # Maximum successive positions
                    "position_spacing_minutes": 15,  # Minutes between successive positions
                    "size_reduction_factor": 0.8,  # Each successive position is 80% of previous
                    "max_total_exposure": 0.3,  # Maximum total exposure across all positions (30%)
                },
                "volatility_adjustment": {
                    "enable_volatility_scaling": True,
                    "atr_multiplier": 1.0,
                    "volatility_thresholds": {
                        "low_volatility": 0.02,  # 2% ATR for low volatility
                        "medium_volatility": 0.05,  # 5% ATR for medium volatility
                        "high_volatility": 0.10,  # 10% ATR for high volatility
                    },
                    "volatility_multipliers": {
                        "low_volatility": 1.2,  # Increase size by 20% in low volatility
                        "medium_volatility": 1.0,  # Normal size in medium volatility
                        "high_volatility": 0.7,  # Reduce size by 30% in high volatility
                    },
                },
                "regime_based_adjustment": {
                    "enable_regime_adjustment": True,
                    "regime_multipliers": {
                        "BULL_TREND": 1.2,  # Increase size by 20% in bull trend
                        "BEAR_TREND": 0.8,  # Reduce size by 20% in bear trend
                        "SIDEWAYS_RANGE": 0.9,  # Reduce size by 10% in sideways
                    },
                },
                "risk_limits": {
                    "max_single_position": 0.15,  # Maximum single position (15%)
                    "max_total_exposure": 0.3,  # Maximum total exposure (30%)
                    "max_correlation_exposure": 0.2,  # Maximum exposure to correlated assets
                    "min_position_size": 0.01,  # Minimum position size (1%)
                    "max_leverage": 10.0,  # Maximum leverage allowed
                },
            },
            "dynamic_risk_management": {
                "enable_dynamic_risk": True,
                "drawdown_adjustment": {
                    "enable_drawdown_scaling": True,
                    "drawdown_thresholds": {
                        "warning": 0.1,  # 10% drawdown - warning
                        "reduction": 0.2,  # 20% drawdown - reduce position sizes
                        "aggressive": 0.3,  # 30% drawdown - aggressive reduction
                        "emergency": 0.4,  # 40% drawdown - emergency mode
                    },
                    "size_reduction_factors": {
                        "warning": 0.9,  # Reduce to 90% of normal size
                        "reduction": 0.7,  # Reduce to 70% of normal size
                        "aggressive": 0.5,  # Reduce to 50% of normal size
                        "emergency": 0.2,  # Reduce to 20% of normal size
                    },
                },
                "performance_adjustment": {
                    "enable_performance_scaling": True,
                    "performance_thresholds": {
                        "excellent": 0.2,  # 20% profit - excellent performance
                        "good": 0.1,  # 10% profit - good performance
                        "neutral": 0.0,  # 0% profit - neutral performance
                        "poor": -0.1,  # -10% profit - poor performance
                    },
                    "size_adjustment_factors": {
                        "excellent": 1.2,  # Increase to 120% of normal size
                        "good": 1.1,  # Increase to 110% of normal size
                        "neutral": 1.0,  # Normal size
                        "poor": 0.8,  # Reduce to 80% of normal size
                    },
                },
            },
        },
        # --- Position Management Configuration ---
        "position_management": {
            "position_closing": {
                "enable_dynamic_closing": True,
                "confidence_based_closing": True,
                "time_based_closing": True,
                "profit_taking": {
                    "enable_profit_taking": True,
                    "profit_targets": {
                        "conservative": 0.02,  # 2% profit target
                        "moderate": 0.05,  # 5% profit target
                        "aggressive": 0.10,  # 10% profit target
                    },
                    "partial_profit_taking": {
                        "enable_partial_taking": True,
                        "partial_targets": [0.02, 0.05, 0.08],  # Multiple targets
                        "partial_sizes": [0.3, 0.3, 0.4],  # Size to take at each target
                    },
                },
                "stop_loss": {
                    "enable_stop_loss": True,
                    "stop_loss_types": {
                        "fixed": 0.02,  # 2% fixed stop loss
                        "trailing": 0.015,  # 1.5% trailing stop loss
                        "atr_based": 2.0,  # 2x ATR stop loss
                    },
                    "dynamic_stop_loss": {
                        "enable_dynamic_stop": True,
                        "confidence_based_stop": True,
                        "regime_based_stop": True,
                    },
                },
            },
            "position_monitoring": {
                "enable_real_time_monitoring": True,
                "monitoring_interval_seconds": 10,
                "alert_thresholds": {
                    "drawdown_warning": 0.05,  # 5% drawdown warning
                    "drawdown_critical": 0.15,  # 15% drawdown critical
                    "profit_warning": 0.20,  # 20% profit warning
                    "profit_critical": 0.50,  # 50% profit critical
                },
                "auto_rebalancing": {
                    "enable_auto_rebalancing": True,
                    "rebalancing_threshold": 0.1,  # 10% deviation triggers rebalancing
                    "rebalancing_interval_hours": 24,  # Rebalance every 24 hours
                },
            },
        },
        # --- Pipeline Configuration ---
        "pipeline": {
            "loop_interval_seconds": 10,  # Main loop interval for live trading
            "max_retries": 3,  # Maximum retries for failed operations
            "timeout_seconds": 30,  # Timeout for operations
        },
        # --- Analyst Configuration ---
        "analyst": {
            "unified_regime_classifier": {
                "min_data_points": 500,  # Reduced from 1000 to allow smaller datasets
                "n_states": 4,  # BULL, BEAR, SIDEWAYS, VOLATILE
                "n_iter": 100,
                "random_state": 42,
                "target_timeframe": "1h",
                "volatility_period": 10,
                "enable_sr_integration": True,
                # Regime thresholds (tunable)
                "adx_sideways_threshold": 18,  # Lowered for better regime balance (was 20)
                "volatility_threshold": 0.020,  # Slightly lower so VOLATILE can appear when returns std rises
                "atr_normalized_threshold": 0.028,  # ATR/close threshold to mark VOLATILE
                "volatility_percentile_threshold": 0.75,  # Top 25% vol considered high
            },
            "analysis_interval": 3600,
            "max_analysis_history": 100,
            "enable_technical_analysis": True,
            "enable_dual_model_system": True,
            "enable_market_health_analysis": True,
            "enable_liquidation_risk_analysis": True,
        },
    }


def get_risk_management_config() -> dict[str, Any]:
    """
    Get risk management configuration.

    Returns:
        dict: Risk management configuration
    """
    trading_config = get_trading_config()
    return trading_config.get("risk_management", {})


def get_position_management_config() -> dict[str, Any]:
    """
    Get position management configuration.

    Returns:
        dict: Position management configuration
    """
    trading_config = get_trading_config()
    return trading_config.get("position_management", {})


def get_exchange_config() -> dict[str, Any]:
    """
    Get exchange configuration.

    Returns:
        dict: Exchange configuration
    """
    trading_config = get_trading_config()
    return trading_config.get("exchanges", {})


def get_pipeline_config() -> dict[str, Any]:
    """
    Get pipeline configuration.

    Returns:
        dict: Pipeline configuration
    """
    trading_config = get_trading_config()
    return trading_config.get("pipeline", {})


def get_position_sizing_config() -> dict[str, Any]:
    """
    Get position sizing configuration.

    Returns:
        dict: Position sizing configuration
    """
    risk_config = get_risk_management_config()
    return risk_config.get("position_sizing", {})


def get_dynamic_risk_config() -> dict[str, Any]:
    """
    Get dynamic risk management configuration.

    Returns:
        dict: Dynamic risk management configuration
    """
    risk_config = get_risk_management_config()
    return risk_config.get("dynamic_risk_management", {})


def get_position_closing_config() -> dict[str, Any]:
    """
    Get position closing configuration.

    Returns:
        dict: Position closing configuration
    """
    position_config = get_position_management_config()
    return position_config.get("position_closing", {})


def get_position_monitoring_config() -> dict[str, Any]:
    """
    Get position monitoring configuration.

    Returns:
        dict: Position monitoring configuration
    """
    position_config = get_position_management_config()
    return position_config.get("position_monitoring", {})
