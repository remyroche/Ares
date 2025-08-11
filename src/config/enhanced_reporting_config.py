#!/usr/bin/env python3
"""
Enhanced Reporting Configuration

This module provides configuration for the enhanced reporting system
that integrates paper trading, live trading, and backtesting with
consistent detailed metrics across all trading modes.
"""

from typing import Any


def get_enhanced_reporting_config() -> dict[str, Any]:
    """
    Get comprehensive configuration for enhanced reporting system.

    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    return {
        # Enhanced Trading Launcher Configuration
        "enhanced_trading_launcher": {
            "enable_paper_trading": True,
            "enable_live_trading": False,  # Set to True when live trading is ready
            "enable_backtesting": True,
            "enable_detailed_reporting": True,
            "report_interval": 3600,  # 1 hour
            "auto_generate_reports": True,
        },
        # Paper Trading Integration Configuration
        "paper_trading_integration": {
            "enable_detailed_reporting": True,
            "enable_real_time_reporting": True,
            "report_interval": 3600,  # 1 hour
            "auto_export_reports": True,
            "export_formats": ["json", "csv", "html"],
        },
        # Enhanced Paper Trader Configuration
        "paper_trader": {
            "initial_balance": 10000.0,
            "max_position_size": 0.1,
            "commission_rate": 0.001,
            "slippage_rate": 0.0005,
            "enable_risk_management": True,
            "max_drawdown": 0.2,
            "enable_detailed_reporting": True,
        },
        # Enhanced Backtester Configuration
        "enhanced_backtester": {
            "initial_balance": 10000.0,
            "commission_rate": 0.001,
            "slippage_rate": 0.0005,
            "max_position_size": 0.1,
            "enable_detailed_reporting": True,
            "auto_generate_reports": True,
            "export_formats": ["json", "csv", "html"],
        },
        # Paper Trading Reporter Configuration
        "paper_trading_reporter": {
            "enable_detailed_reporting": True,
            "report_directory": "reports/paper_trading",
            "export_formats": ["json", "csv", "html"],
            "auto_cleanup_old_reports": True,
            "max_report_age_days": 30,
            "enable_real_time_tracking": True,
            "track_market_indicators": True,
            "track_market_health": True,
            "track_ml_confidence": True,
        },
        # Market Health Analyzer Configuration
        "market_health_analyzer": {
            "analysis_interval": 3600,
            "enable_volatility_analysis": True,
            "enable_market_health_metrics": True,
            "enable_liquidity_analysis": True,
            "enable_stress_analysis": True,
            "volatility_regime_thresholds": {
                "low": 0.1,
                "medium": 0.2,
                "high": 0.3,
            },
            "liquidity_thresholds": {
                "low": 0.3,
                "medium": 0.6,
                "high": 0.8,
            },
            "stress_thresholds": {
                "low": 0.2,
                "medium": 0.5,
                "high": 0.8,
            },
        },
        # ML Confidence Predictor Configuration
        "ml_confidence_predictor": {
            "enable_confidence_tracking": True,
            "confidence_thresholds": {
                "low": 0.3,
                "medium": 0.6,
                "high": 0.8,
            },
            "track_individual_models": True,
            "track_ensemble_agreement": True,
            "track_model_diversity": True,
            "track_prediction_consistency": True,
        },
        # Performance Reporter Configuration
        "performance_reporter": {
            "enable_real_time_reporting": True,
            "report_interval": 3600,
            "export_directory": "reports/performance",
            "export_formats": ["json", "csv", "html"],
            "enable_advanced_analytics": True,
            "enable_risk_analysis": True,
            "enable_attribution_analysis": True,
            "enable_forecasting": True,
        },
        # Trade Tracker Configuration
        "trade_tracker": {
            "enable_detailed_tracking": True,
            "track_ensemble_decisions": True,
            "track_regime_analysis": True,
            "track_decision_paths": True,
            "track_model_behaviors": True,
            "auto_export_trades": True,
            "export_formats": ["json", "csv"],
        },
        # Reporting Directories
        "reporting_directories": {
            "paper_trading": "reports/paper_trading",
            "backtesting": "reports/backtesting",
            "live_trading": "reports/live_trading",
            "performance": "reports/performance",
            "launcher": "reports/launcher",
        },
        # Export Configuration
        "export_config": {
            "default_formats": ["json", "csv", "html"],
            "auto_export": True,
            "timestamp_format": "%Y%m%d_%H%M%S",
            "compression": False,
            "max_file_size_mb": 100,
        },
        # Metrics Configuration
        "metrics_config": {
            "track_pnl_metrics": True,
            "track_risk_metrics": True,
            "track_performance_metrics": True,
            "track_portfolio_metrics": True,
            "track_trade_metrics": True,
            "track_market_metrics": True,
            "track_ml_metrics": True,
            # PnL Metrics
            "pnl_metrics": {
                "absolute_pnl": True,
                "percentage_pnl": True,
                "unrealized_pnl": True,
                "realized_pnl": True,
                "total_cost": True,
                "total_proceeds": True,
                "commission_paid": True,
                "slippage_paid": True,
                "net_pnl": True,
            },
            # Risk Metrics
            "risk_metrics": {
                "max_drawdown": True,
                "sharpe_ratio": True,
                "sortino_ratio": True,
                "calmar_ratio": True,
                "var_95": True,
                "expected_shortfall": True,
                "downside_deviation": True,
                "tail_risk": True,
            },
            # Performance Metrics
            "performance_metrics": {
                "total_trades": True,
                "win_rate": True,
                "profit_factor": True,
                "average_win": True,
                "average_loss": True,
                "largest_win": True,
                "largest_loss": True,
                "consecutive_wins": True,
                "consecutive_losses": True,
            },
            # Portfolio Metrics
            "portfolio_metrics": {
                "total_value": True,
                "cash_balance": True,
                "positions_count": True,
                "symbol_positions": True,
                "portfolio_diversification": True,
                "sector_allocation": True,
                "geographic_allocation": True,
            },
            # Trade Metrics
            "trade_metrics": {
                "trade_type": True,
                "leverage": True,
                "duration": True,
                "strategy": True,
                "order_type": True,
                "position_sizing": True,
                "execution_quality": True,
                "risk_metrics": True,
            },
            # Market Metrics
            "market_metrics": {
                "market_indicators": True,
                "market_health": True,
                "volatility_regime": True,
                "liquidity_score": True,
                "stress_score": True,
                "market_strength": True,
                "volume_health": True,
                "price_trend": True,
                "market_regime": True,
            },
            # ML Metrics
            "ml_metrics": {
                "analyst_confidence": True,
                "tactician_confidence": True,
                "ensemble_confidence": True,
                "meta_learner_confidence": True,
                "individual_model_confidences": True,
                "ensemble_agreement": True,
                "model_diversity": True,
                "prediction_consistency": True,
            },
        },
        # Market Indicators Configuration
        "market_indicators": {
            "rsi": True,
            "macd": True,
            "bollinger_bands": True,
            "atr": True,
            "volume_sma": True,
            "price_sma": True,
            "volatility": True,
            "momentum": True,
            "support_resistance": True,
        },
        # Trade Type Classification
        "trade_types": {
            "sides": ["long", "short"],
            "durations": ["scalping", "day_trading", "swing", "position"],
            "strategies": [
                "breakout",
                "mean_reversion",
                "momentum",
                "arbitrage",
                "hedging",
            ],
            "order_types": ["market", "limit", "stop", "stop_limit"],
        },
        # Position Sizing Configuration
        "position_sizing": {
            "track_absolute_size": True,
            "track_portfolio_percentage": True,
            "track_risk_percentage": True,
            "track_max_position_size": True,
            "track_position_ranking": True,
        },
        # Real-time Reporting Configuration
        "real_time_reporting": {
            "enable": True,
            "interval_seconds": 3600,  # 1 hour
            "auto_export": True,
            "export_formats": ["json"],
            "max_reports_in_memory": 100,
        },
        # Data Retention Configuration
        "data_retention": {
            "max_trade_history_days": 365,
            "max_report_history_days": 90,
            "max_performance_data_days": 180,
            "auto_cleanup": True,
            "backup_before_cleanup": True,
        },
        # Error Handling Configuration
        "error_handling": {
            "continue_on_reporting_error": True,
            "log_reporting_errors": True,
            "retry_failed_reports": True,
            "max_retry_attempts": 3,
            "retry_delay_seconds": 60,
        },
        # Validation Configuration
        "validation": {
            "validate_trade_data": True,
            "validate_market_data": True,
            "validate_ml_data": True,
            "validate_performance_data": True,
            "strict_validation": False,
        },
    }


def get_paper_trading_config() -> dict[str, Any]:
    """
    Get configuration specifically for paper trading with enhanced reporting.

    Returns:
        Dict[str, Any]: Paper trading configuration
    """
    base_config = get_enhanced_reporting_config()

    # Override for paper trading specific settings
    return {
        **base_config,
        "enhanced_trading_launcher": {
            **base_config["enhanced_trading_launcher"],
            "enable_paper_trading": True,
            "enable_live_trading": False,
            "enable_backtesting": False,
        },
        "paper_trading_integration": {
            **base_config["paper_trading_integration"],
            "enable_real_time_reporting": True,
            "report_interval": 1800,  # 30 minutes for paper trading
        },
    }


def get_backtesting_config() -> dict[str, Any]:
    """
    Get configuration specifically for backtesting with enhanced reporting.

    Returns:
        Dict[str, Any]: Backtesting configuration
    """
    base_config = get_enhanced_reporting_config()

    # Override for backtesting specific settings
    return {
        **base_config,
        "enhanced_trading_launcher": {
            **base_config["enhanced_trading_launcher"],
            "enable_paper_trading": False,
            "enable_live_trading": False,
            "enable_backtesting": True,
        },
        "enhanced_backtester": {
            **base_config["enhanced_backtester"],
            "auto_generate_reports": True,
            "export_formats": ["json", "csv", "html"],
        },
    }


def get_live_trading_config() -> dict[str, Any]:
    """
    Get configuration specifically for live trading with enhanced reporting.

    Returns:
        Dict[str, Any]: Live trading configuration
    """
    base_config = get_enhanced_reporting_config()

    # Override for live trading specific settings
    return {
        **base_config,
        "enhanced_trading_launcher": {
            **base_config["enhanced_trading_launcher"],
            "enable_paper_trading": False,
            "enable_live_trading": True,
            "enable_backtesting": False,
        },
        "paper_trading_integration": {
            **base_config["paper_trading_integration"],
            "enable_real_time_reporting": True,
            "report_interval": 900,  # 15 minutes for live trading
        },
    }


def validate_enhanced_reporting_config(config: dict[str, Any]) -> bool:
    """
    Validate enhanced reporting configuration.

    Args:
        config: Configuration dictionary to validate

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        required_sections = [
            "enhanced_trading_launcher",
            "paper_trading_integration",
            "paper_trader",
            "enhanced_backtester",
            "paper_trading_reporter",
            "metrics_config",
        ]

        for section in required_sections:
            if section not in config:
                print(f"Missing required configuration section: {section}")
                return False

        # Validate specific settings
        launcher_config = config["enhanced_trading_launcher"]
        if not any(
            [
                launcher_config.get("enable_paper_trading", False),
                launcher_config.get("enable_live_trading", False),
                launcher_config.get("enable_backtesting", False),
            ],
        ):
            print("At least one trading mode must be enabled")
            return False

        return True

    except Exception as e:
        print(f"Error validating configuration: {e}")
        return False


def get_minimal_config() -> dict[str, Any]:
    """
    Get minimal configuration for basic enhanced reporting.

    Returns:
        Dict[str, Any]: Minimal configuration
    """
    return {
        "enhanced_trading_launcher": {
            "enable_paper_trading": True,
            "enable_backtesting": True,
            "enable_detailed_reporting": True,
        },
        "paper_trading_reporter": {
            "enable_detailed_reporting": True,
            "report_directory": "reports/paper_trading",
            "export_formats": ["json"],
        },
        "metrics_config": {
            "track_pnl_metrics": True,
            "track_risk_metrics": True,
            "track_performance_metrics": True,
        },
    }
