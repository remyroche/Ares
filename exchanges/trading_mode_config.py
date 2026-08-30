"""
Trading Mode Configuration

Configuration management for mode-aware exchange interface.
"""

import os
from typing import Optional
from enum import Enum
from dataclasses import dataclass
from .mode_aware_exchange_interface import TradingMode, ModeAwareConfig


class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class TradingModeSettings:
    """Trading mode settings"""
    mode: TradingMode
    environment: Environment
    initial_balance: float
    enable_order_book_simulation: bool
    simulation_commission_rate: float
    log_trades: bool
    enable_risk_management: bool
    max_position_size: float
    max_daily_loss: float


class TradingModeConfigManager:
    """Manages trading mode configuration"""

    def __init__(self):
        self.settings: Optional[TradingModeSettings] = None

    def load_from_environment(self) -> TradingModeSettings:
        """Load configuration from environment variables"""
        # Get mode from environment
        mode_str = os.getenv("TRADING_MODE", "PAPER").upper()
        try:
            mode = TradingMode(mode_str.lower())
        except ValueError:
            mode = TradingMode.PAPER  # Default to paper trading

        # Get environment
        env_str = os.getenv("ENVIRONMENT", "DEVELOPMENT").upper()
        try:
            environment = Environment(env_str.lower())
        except ValueError:
            environment = Environment.DEVELOPMENT

        # Get other settings
        initial_balance = float(os.getenv("INITIAL_BALANCE", "100000.0"))
        enable_order_book_simulation = os.getenv("ENABLE_ORDER_BOOK_SIMULATION", "true").lower() == "true"
        simulation_commission_rate = float(os.getenv("SIMULATION_COMMISSION_RATE", "0.001"))
        log_trades = os.getenv("LOG_TRADES", "true").lower() == "true"
        enable_risk_management = os.getenv("ENABLE_RISK_MANAGEMENT", "true").lower() == "true"
        max_position_size = float(os.getenv("MAX_POSITION_SIZE", "10000.0"))
        max_daily_loss = float(os.getenv("MAX_DAILY_LOSS", "5000.0"))

        self.settings = TradingModeSettings(
            mode=mode,
            environment=environment,
            initial_balance=initial_balance,
            enable_order_book_simulation=enable_order_book_simulation,
            simulation_commission_rate=simulation_commission_rate,
            log_trades=log_trades,
            enable_risk_management=enable_risk_management,
            max_position_size=max_position_size,
            max_daily_loss=max_daily_loss
        )

        return self.settings

    def get_config(self) -> ModeAwareConfig:
        """Get ModeAwareConfig from current settings"""
        if not self.settings:
            self.load_from_environment()

        return ModeAwareConfig(
            mode=self.settings.mode,
            initial_balance=self.settings.initial_balance,
            enable_order_book_simulation=self.settings.enable_order_book_simulation,
            simulation_commission_rate=self.settings.simulation_commission_rate,
            log_trades=self.settings.log_trades
        )

    def is_paper_mode(self) -> bool:
        """Check if currently in paper trading mode"""
        if not self.settings:
            self.load_from_environment()
        return self.settings.mode == TradingMode.PAPER

    def is_trade_mode(self) -> bool:
        """Check if currently in live trading mode"""
        if not self.settings:
            self.load_from_environment()
        return self.settings.mode == TradingMode.TRADE

    def get_environment(self) -> Environment:
        """Get current environment"""
        if not self.settings:
            self.load_from_environment()
        return self.settings.environment

    def should_enable_risk_management(self) -> bool:
        """Check if risk management should be enabled"""
        if not self.settings:
            self.load_from_environment()
        return self.settings.enable_risk_management

    def get_risk_limits(self) -> dict:
        """Get risk management limits"""
        if not self.settings:
            self.load_from_environment()
        return {
            "max_position_size": self.settings.max_position_size,
            "max_daily_loss": self.settings.max_daily_loss
        }


# Global configuration manager instance
config_manager = TradingModeConfigManager()


def get_trading_mode_config() -> ModeAwareConfig:
    """Get the current trading mode configuration"""
    return config_manager.get_config()


def is_paper_trading() -> bool:
    """Check if currently in paper trading mode"""
    return config_manager.is_paper_mode()


def is_live_trading() -> bool:
    """Check if currently in live trading mode"""
    return config_manager.is_trade_mode()


def get_environment() -> Environment:
    """Get current environment"""
    return config_manager.get_environment()