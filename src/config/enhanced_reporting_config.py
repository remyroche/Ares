#!/usr/bin/env python3
"""
Enhanced Reporting Configuration

This module provides configuration for the enhanced reporting system
that integrates paper trading, live trading, and backtesting with
consistent detailed metrics across all trading modes.
"""

from typing import Any




def get_backtesting_config() -> dict[str, Any]:
    """
    Get configuration specifically for backtesting with enhanced reporting.

    Returns:
        Dict[str, Any]: Backtesting configuration
    """
    base_config , get_enhanced_reporting_config()

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



def validate_enhanced_reporting_config(config: dict[str, Any]) -> bool:
    """
    Validate enhanced reporting configuration.

    Args:
        config: Configuration dictionary to validate

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        required_sections , [
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

