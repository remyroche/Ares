"""Re-export signal generation components for shared TAS utilities."""

from training.steps.market_analysis.tas_regime.trading.signal_generator import (  # noqa: F401
    SignalConfig,
    TradingSignalGenerator,
)

__all__ = ["SignalConfig", "TradingSignalGenerator"]
