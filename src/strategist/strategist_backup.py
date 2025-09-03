"""
Strategist module for trading strategy generation.

This module provides the Strategist class which is responsible for:
- Strategy Generation: Create trading strategies based on market analysis
- Market Analysis Integration: Combine analyst and tactician inputs
- Strategy History Management: Track and store strategy performance
"""
from src.core.decorators import handles_errors, retry, timeout

from src.core.domain import handle_specific_errors

# src/strategist/strategist.py

from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

    handle_specific_errors,
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    failed,
    invalid,
    missing,

if TYPE_CHECKING:
    from src.analyst.analyst import Analyst
    from src.tactician.tactician import Tactician


class Strategist:
    # TODO: Consider extracting common error logging patterns into helper methods
    """
    Strategy-Level Strategist component responsible for:
    - Strategy Generation: Create trading strategies based on market analysis
    - Market Analysis Integration: Combine analyst and tactician inputs
    - Strategy History Management: Track and store strategy performance

    Note: Position sizing is handled by the Tactician component
    """
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize strategist with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("Strategist")

        # Strategist state
        self.is_running: bool = False
        self.strategy_results: dict[str, Any] = {}
        self.strategy_history: list[dict[str, Any]] = []
        self.current_strategy: dict[str, Any] = {}

        # Configuration
        self.strategist_config: dict[str, Any] = self.config.get("strategist", {})
        self.strategy_interval: int = (
            self.strategist_config.get("strategy_interval", 1800)
        self.max_strategy_history: int = (
            self.strategist_config.get("max_strategy_history", 50)
        # Risk management (excluding position sizing which is handled by Tactician)
        self.enable_risk_management: bool = (
            self.strategist_config.get("enable_risk_management", True)

        # Strategy parameters (position sizing handled by Tactician)
        self.min_confidence_threshold: float = (
            self.strategist_config.get("min_confidence_threshold", 0.6)

        # Technical indicator thresholds and strategy type (for profile/reference only)
        tech_cfg = self.strategist_config.get("technical_indicator_thresholds", {})
        self.rsi_oversold: float = tech_cfg.get("rsi_oversold", 30.0)
        self.rsi_overbought: float = tech_cfg.get("rsi_overbought", 70.0)
        self.sma_fast_window: int = tech_cfg.get("sma_fast_window", 20)
        self.sma_slow_window: int = tech_cfg.get("sma_slow_window", 50)
        self.volume_ratio_high: float = tech_cfg.get("volume_ratio_high", 1.5)
        self.volume_ratio_low: float = tech_cfg.get("volume_ratio_low", 0.5)
        self.price_volatility_window: int = tech_cfg.get("price_volatility_window", 20)

        self.strategy_type: str = (
            self.strategist_config.get("strategy_type", "technical_analysis")

        # Component references (will be set during initialization)
        self.analyst: Analyst | None = None
        self.tactician: Tactician | None = None

    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid strategist configuration"),
            AttributeError: (False, "Missing required strategist parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="strategist initialization",
    async def initialize(self) -> bool:
        """
        Initialize strategist with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Strategist...")

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error(invalid("Invalid configuration for strategist"))
                return False

            # Initialize strategy components
            await self._initialize_strategy_components()

            self.logger.info("✅ Strategist initialized successfully")
            return True

        except (ValueError, TypeError, KeyError) as e:
            self.logger.exception(failed(f"❌ Strategist initialization failed: {e}"))
            return False

    @handles_errors(ValueError, AttributeError, fallback=None,
        context="strategy components initialization",
    async def _initialize_strategy_components(self) -> None:
        """Initialize strategy components."""
        try:
            # Initialize risk management
            if self.enable_risk_management:
                self.logger.info("Initializing risk management components...")

            # Position sizing is handled by the Tactician component
            # No position sizing initialization in Strategist

            self.logger.info("✅ Strategy components initialized successfully")

        except Exception as e:  # TODO: Consider more specific exception types
            self.logger.exception(f"Error initializing strategy components: {e}")
            raise

    @handles_errors(ValueError, TypeError, fallback=False,
        context="configuration validation",
    )
    # TODO: Refactor to reduce complexity (current: 6)

    def _validate_configuration(self) -> bool:
        """Validate strategist configuration."""
        try:
            required_keys = ["strategy_interval", "max_strategy_history"]
            for key in required_keys:
                if key not in self.strategist_config:
                    self.logger.error(missing(f"Missing required configuration key: {key}"))
                    return False

            # Position sizing parameters are handled by the Tactician component
            # No position sizing validation in Strategist

            if self.min_confidence_threshold < 0 or self.min_confidence_threshold > 1:
                self.logger.error(invalid("Invalid min_confidence_threshold value"))
                return False

            return True

        except Exception as e:  # TODO: Consider more specific exception types
            self.logger.exception(f"Error validating configuration: {e}")
            return False

    @handles_errors(
        error_handlers={
            ValueError: (None, "Invalid market data for strategy generation"),
            AttributeError: (None, "Missing required market data fields"),
            KeyError: (None, "Missing required market data keys"),
        },
        default_return=None,
        context="strategy generation",
    async def generate_strategy(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        analysis_results: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Generate trading strategy based on market data and analysis results.

        Args:
            market_data: Market data for analysis
            current_price: Current asset price
            analysis_results: Results from market analysis (Step 1)

        Returns:
            dict[str, Any] | None: Generated strategy or None if failed
        """
        try:
            if not self._validate_market_data(market_data):
                self.logger.error("Invalid market data for strategy generation")
                return None

            self.logger.info("🎯 Generating trading strategy...")

            # Extract key market indicators
            market_indicators = self._extract_market_indicators(market_data, current_price)

            # Generate base strategy
            base_strategy = await self._generate_base_strategy(market_indicators, current_price)

            # Integrate analysis results if available
            if analysis_results:
                base_strategy = await self._integrate_analysis_results(base_strategy, analysis_results)

            # Apply risk management
            if self.enable_risk_management:
                base_strategy = await self._apply_risk_management(base_strategy, current_price)

            # Position sizing is handled by the Tactician component
            # No position sizing applied in Strategist

            # Store strategy results
            await self._store_strategy_results(base_strategy)

            self.logger.info("✅ Strategy generation completed successfully")
            return base_strategy

        except Exception as e:  # TODO: Consider more specific exception types
            self.logger.exception(f"Error generating strategy: {e}")
            return None

    @handles_errors(ValueError, TypeError, fallback=False,
        context="market data validation",
    )
    # TODO: Refactor to reduce complexity (current: 6)

    def _validate_market_data(self, market_data: pd.DataFrame) -> bool:
        """Validate market data for strategy generation."""
        try:
            if market_data is None or market_data.empty:
                self.logger.error("Market data is None or empty")
                return False

            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in market_data.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False

            # Check for sufficient data points
            if len(market_data) < 20:
                self.logger.error("Insufficient market data points for strategy generation")
                return False

            return True

        except Exception as e:  # TODO: Consider more specific exception types
            self.logger.exception(f"Error validating market data: {e}")
            return False

    @handles_errors(ValueError, TypeError, fallback={},
        context="market indicators extraction",
    def _extract_market_indicators(self, market_data: pd.DataFrame, current_price: float) -> dict[str, Any]:
        """Extract key market indicators from market data."""
        try:
            indicators = {}

            # Price indicators
            indicators["current_price"] = current_price
            indicators["price_change"] = (current_price - market_data["close"].iloc[-2]) / market_data["close"].iloc[-2]

            # Price volatility with configurable window
            volatility_window = max(2, int(self.price_volatility_window))
            indicators["price_volatility"] = (
                market_data["close"].pct_change().rolling(window=volatility_window).std().iloc[-1]

            # Volume indicators
            indicators["volume_ma"] = market_data["volume"].rolling(window=20).mean().iloc[-1]
            indicators["volume_ratio"] = market_data["volume"].iloc[-1] / indicators["volume_ma"]

            # Technical indicators
            sma_fast = max(2, int(self.sma_fast_window))
            sma_slow = max(sma_fast + 1, int(self.sma_slow_window))
            indicators["sma_20"] = market_data["close"].rolling(window=sma_fast).mean().iloc[-1]
            indicators["sma_50"] = market_data["close"].rolling(window=sma_slow).mean().iloc[-1]
            indicators["rsi"] = self._calculate_rsi(market_data["close"])

            # Trend indicators
            indicators["trend"] = "BULLISH" if indicators["sma_20"] > indicators["sma_50"] else "BEARISH"
            indicators["momentum"] = "POSITIVE" if indicators["price_change"] > 0 else "NEGATIVE"

            return indicators

        except Exception as e:  # TODO: Consider more specific exception types
            self.logger.exception(f"Error extracting market indicators: {e}")
            return {}

    @handles_errors(ValueError, TypeError, fallback=0.0,
        context="RSI calculation",
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

        except Exception as e:  # TODO: Consider more specific exception types
            self.logger.exception(f"Error calculating RSI: {e}")
            return 50.0

    @handles_errors(ValueError, TypeError, fallback={},
        context="base strategy generation",
    async def _generate_base_strategy(self, indicators: dict[str, Any], current_price: float) -> dict[str, Any]:
        """Generate base trading strategy from market indicators."""
        try:
            return {
                "timestamp": datetime.now().isoformat(),
                "strategy_type": self.strategy_type,
                "confidence": 0.0,  # To be set by ML/HMM via analysis integration
                "direction": "HOLD",  # To be set by ML/HMM via analysis integration
                "entry_price": current_price,
                "stop_loss": None,
                "take_profit": None,
                "position_size": 0.0,
                "risk_level": "MEDIUM",
                "indicators": indicators,
                "reasoning": [
                    "Base strategy initialized; awaiting ML/HMM decision from DualModelSystem",
                ],
                "strategy_profile": {
                    "rsi_oversold": self.rsi_oversold,
                    "rsi_overbought": self.rsi_overbought,
                    "sma_fast_window": self.sma_fast_window,
                    "sma_slow_window": self.sma_slow_window,
                    "volume_ratio_high": self.volume_ratio_high,
                    "volume_ratio_low": self.volume_ratio_low,
                    "price_volatility_window": self.price_volatility_window,
                },
            }
            # Do not use handcrafted feature weights for direction/confidence
            # Direction and confidence will be set by ML/HMM via _integrate_analysis_results


        except Exception as e:  # TODO: Consider more specific exception types
            self.logger.exception(f"Error generating base strategy: {e}")
            return {}

    @handles_errors(ValueError, TypeError, fallback={},
        context="analysis results integration",
    )
    # TODO: Refactor to reduce complexity (current: 7)

    async def _integrate_analysis_results(self, strategy: dict[str, Any], analysis_results: dict[str, Any]) -> dict[str, Any]:
        """Integrate analysis results from Step 1 into strategy."""
        try:
            if not analysis_results:
                return strategy

            # Integrate market health analysis
            market_health = analysis_results.get("market_health", {})
            if market_health:
                health_score = market_health.get("health_score", 0.5)
                strategy["market_health_score"] = health_score
                strategy["confidence"] = (strategy["confidence"] + health_score) / 2
                strategy["reasoning"].append(f"Market health score: {health_score:.3f}")

            # Integrate liquidation risk analysis
            liquidation_risk = analysis_results.get("liquidation_risk", {})
            if liquidation_risk:
                risk_level = liquidation_risk.get("risk_level", "MEDIUM")
                strategy["liquidation_risk"] = risk_level
                if risk_level == "HIGH":
                    strategy["confidence"] *= 0.8  # Reduce confidence for high risk
                    strategy["reasoning"].append("High liquidation risk - reduced confidence")

            # Integrate trading decision from dual model system (ML/HMM-driven)
            trading_decision = analysis_results.get("trading_decision", {})
            if trading_decision:
                decision_confidence = trading_decision.get("final_confidence", 0.0)
                decision_direction = trading_decision.get("direction", "HOLD")

                # Set strategy solely from ML/HMM decision
                strategy["dual_model_direction"] = decision_direction
                strategy["dual_model_confidence"] = decision_confidence
                strategy["direction"] = decision_direction
                strategy["confidence"] = decision_confidence
                strategy["reasoning"].append("Direction and confidence set by DualModelSystem")

            return strategy

        except Exception as e:  # TODO: Consider more specific exception types
            self.logger.exception(f"Error integrating analysis results: {e}")
            return strategy

    @handles_errors(ValueError, TypeError, fallback={},
        context="risk management application",
    async def _apply_risk_management(self, strategy: dict[str, Any], current_price: float) -> dict[str, Any]:
        """Apply risk management to strategy."""
        try:
            if strategy["direction"] == "HOLD":
                return strategy

            # Calculate stop loss and take profit levels
            volatility = strategy["indicators"]["price_volatility"]

            if strategy["direction"] == "LONG":
                # Stop loss: 2x volatility below current price
                stop_loss_pct = volatility * 2
                strategy["stop_loss"] = current_price * (1 - stop_loss_pct)
                strategy["take_profit"] = current_price * (1 + stop_loss_pct * 2)  # 2:1 risk-reward
            else:  # SHORT
                # Stop loss: 2x volatility above current price
                stop_loss_pct = volatility * 2
                strategy["stop_loss"] = current_price * (1 + stop_loss_pct)
                strategy["take_profit"] = current_price * (1 - stop_loss_pct * 2)  # 2:1 risk-reward

            # Adjust confidence based on risk-reward ratio
            risk_reward_ratio = 2.0  # 2:1 risk-reward ratio
            if risk_reward_ratio >= 2.0:
                strategy["confidence"] *= 1.1  # Boost confidence for good risk-reward
                strategy["reasoning"].append(f"Good risk-reward ratio: {risk_reward_ratio:.1f}")

            return strategy

        except Exception as e:  # TODO: Consider more specific exception types
            self.logger.exception(f"Error applying risk management: {e}")
            return strategy

    # Position sizing is handled by the Tactician component
    # This method has been removed to avoid overlap with Tactician responsibilities

    @handles_errors(ValueError, TypeError, fallback=None,
        context="strategy results storage",
    async def _store_strategy_results(self, strategy: dict[str, Any]) -> None:
        """Store strategy results in history."""
        try:
            # Store current strategy
            self.current_strategy = strategy.copy()

            # Add to history
            self.strategy_history.append(strategy.copy())

            # Limit history size
            if len(self.strategy_history) > self.max_strategy_history:
                self.strategy_history = self.strategy_history[-self.max_strategy_history:]

            # Update strategy results
            self.strategy_results = {
                "current_strategy": self.current_strategy,
                "history_count": len(self.strategy_history),
                "last_updated": datetime.now().isoformat(),
            }

        except Exception as e:  # TODO: Consider more specific exception types
            self.logger.exception(f"Error storing strategy results: {e}")

    def get_strategy_results(self) -> dict[str, Any]:
        """
        Get current strategy results.

        Returns:
            dict[str, Any]: Current strategy results
        """
        return self.strategy_results.copy()

    def get_current_strategy(self) -> dict[str, Any]:
        """
        Get current strategy.

        Returns:
            dict[str, Any]: Current strategy
        """
        return self.current_strategy.copy()

    def get_strategy_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get strategy history.

        Args:
            limit: Maximum number of history entries to return

        Returns:
            list[dict[str, Any]]: Strategy history
        """
        history = self.strategy_history.copy()
        if limit:
            history = history[-limit:]
        return history

    @handles_errors(Exception,, fallback=None,
        context="strategist stop",
    async def stop(self) -> None:
        """Stop the strategist and cleanup resources."""
        try:
            self.logger.info("🛑 Stopping Strategist...")
            self.is_running = False
            self.logger.info("✅ Strategist stopped successfully")

        except Exception as e:  # TODO: Consider more specific exception types
            self.logger.exception(failed(f"❌ Failed to stop Strategist: {e}"))
