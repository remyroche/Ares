# src/config/trading.py

from typing import Any

from src.config.environment import get_environment_settings


def get_trading_config() -> dict[str, Any]:
    """Get the complete trading configuration.

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
"light": 0.05,  # 5% drawdown
"moderate": 0.15,  # 15% drawdown
"severe": 0.25,  # 25% drawdown
},
"position_size_reductions": {
"light": 0.8,  # Reduce to 80% of normal size
"moderate": 0.5,  # Reduce to 50% of normal size
"severe": 0.2,  # Reduce to 20% of normal size
},
},
},
},
# --- Stop Loss and Take Profit Configuration ---
"stop_loss": {
"enable_stop_loss": True,
"stop_loss_type": "trailing",  # 'fixed' or 'trailing'
"fixed_stop_loss_pct": 0.02,  # 2% fixed stop loss
"trailing_stop_config": {
"activation_threshold": 0.01,  # Activate trailing stop at 1% profit
"trailing_distance": 0.005,  # 0.5% trailing distance
"lock_profit_threshold": 0.03,  # Lock profit at 3% gain
},
},
"take_profit": {
"enable_take_profit": True,
"take_profit_type": "dynamic",  # 'fixed' or 'dynamic'
"fixed_take_profit_pct": 0.05,  # 5% fixed take profit
"dynamic_take_profit_config": {
"base_take_profit": 0.03,  # 3% base take profit
"volatility_multiplier": 1.5,  # Multiply by volatility
"max_take_profit": 0.15,  # Maximum 15% take profit
},
},
# --- Time-based Exit Configuration ---
"time_based_exit": {
"enable_time_exit": True,
"max_holding_time_hours": 24,  # Maximum holding time
"profit_lock_time_hours": 4,  # Lock profit after 4 hours
"loss_cut_time_hours": 2,  # Cut loss after 2 hours
},
}


def get_exchange_config(exchange_name: str) -> dict[str, Any]:
    """Get configuration for a specific exchange.

Args:
        exchange_name: Name of the exchange

Returns:
        dict: Exchange configuration

"""
trading_config = get_trading_config()
exchanges = trading_config.get("exchanges", {})
return exchanges.get(exchange_name.lower(), {})


def get_risk_management_config() -> dict[str, Any]:
    """Get risk management configuration.

Returns:
        dict: Risk management configuration

"""
trading_config = get_trading_config()
return trading_config.get("risk_management", {})


def get_position_sizing_config() -> dict[str, Any]:
    """Get position sizing configuration.

Returns:
        dict: Position sizing configuration

"""
risk_config = get_risk_management_config()
return risk_config.get("position_sizing", {})


def get_stop_loss_config() -> dict[str, Any]:
    """Get stop loss configuration.

Returns:
        dict: Stop loss configuration

"""
trading_config = get_trading_config()
return trading_config.get("stop_loss", {})


def get_take_profit_config() -> dict[str, Any]:
    """Get take profit configuration.

Returns:
        dict: Take profit configuration

"""
trading_config = get_trading_config()
return trading_config.get("take_profit", {})


def get_time_based_exit_config() -> dict[str, Any]:
    """Get time-based exit configuration.

Returns:
        dict: Time-based exit configuration

"""
trading_config = get_trading_config()
return trading_config.get("time_based_exit", {})
