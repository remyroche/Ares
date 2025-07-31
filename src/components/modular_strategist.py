# src/components/modular_strategist.py

from typing import Dict, Any, Optional
from datetime import datetime

from src.interfaces import IStrategist, AnalysisResult, StrategyResult, EventType
from src.interfaces.base_interfaces import IExchangeClient, IStateManager, IEventBus
from src.utils.logger import system_logger
from src.config import settings
from src.utils.error_handler import (
    handle_errors,
    handle_data_processing_errors,
)


class ModularStrategist(IStrategist):
    """
    Modular implementation of the Strategist that implements the IStrategist interface.
    Uses dependency injection and event-driven communication.
    """

    def __init__(
        self,
        exchange_client: IExchangeClient,
        state_manager: IStateManager,
        event_bus: Optional[IEventBus] = None,
    ):
        """
        Initialize the modular strategist.

        Args:
            exchange_client: Exchange client for data access
            state_manager: State manager for persistence
            event_bus: Optional event bus for communication
        """
        self.exchange = exchange_client
        self.state_manager = state_manager
        self.event_bus = event_bus
        self.logger = system_logger.getChild("ModularStrategist")
        self.config = settings.get("strategist", {})
        self.last_analyst_timestamp = None
        self.running = False
        self.strategy_performance = {}

        self.logger.info("ModularStrategist initialized")

    @handle_errors(
        exceptions=(Exception,), default_return=None, context="modular_strategist_start"
    )
    async def start(self) -> None:
        """Start the modular strategist"""
        self.logger.info("Starting ModularStrategist")
        self.running = True

        # Subscribe to analysis events if event bus is available
        if self.event_bus:
            await self.event_bus.subscribe(
                EventType.ANALYSIS_COMPLETED, self._handle_analysis_result
            )

        self.logger.info("ModularStrategist started")

    @handle_errors(
        exceptions=(Exception,), default_return=None, context="modular_strategist_stop"
    )
    async def stop(self) -> None:
        """Stop the modular strategist"""
        self.logger.info("Stopping ModularStrategist")
        self.running = False

        # Unsubscribe from events if event bus is available
        if self.event_bus:
            await self.event_bus.unsubscribe(
                EventType.ANALYSIS_COMPLETED, self._handle_analysis_result
            )

        self.logger.info("ModularStrategist stopped")

    @handle_errors(
        exceptions=(Exception,), default_return=None, context="formulate_strategy"
    )
    async def formulate_strategy(
        self, analysis_result: AnalysisResult
    ) -> StrategyResult:
        """
        Formulate trading strategy based on analysis result.

        Args:
            analysis_result: Analysis result from the analyst

        Returns:
            Strategy result
        """
        if not self.running:
            self.logger.warning("Strategist not running, skipping strategy formulation")
            return None

        self.logger.debug(f"Formulating strategy for {analysis_result.symbol}")

        try:
            # Determine positional bias
            position_bias = await self._determine_positional_bias(analysis_result)

            # Determine leverage cap
            leverage_cap = await self._determine_leverage_cap(analysis_result)

            # Determine max notional size
            max_notional_size = await self._determine_max_notional_size(analysis_result)

            # Calculate risk parameters
            risk_parameters = await self._calculate_risk_parameters(analysis_result)

            # Assess market conditions
            market_conditions = await self._assess_market_conditions(analysis_result)

            # Build strategy result
            strategy_result = StrategyResult(
                timestamp=datetime.now(),
                symbol=analysis_result.symbol,
                position_bias=position_bias,
                leverage_cap=leverage_cap,
                max_notional_size=max_notional_size,
                risk_parameters=risk_parameters,
                market_conditions=market_conditions,
            )

            # Publish strategy formulated event
            if self.event_bus:
                await self.event_bus.publish(
                    EventType.STRATEGY_FORMULATED, strategy_result, "ModularStrategist"
                )

            # Update performance tracking
            await self._update_strategy_performance(strategy_result)

            return strategy_result

        except Exception as e:
            self.logger.error(f"Strategy formulation failed: {e}", exc_info=True)
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="update_strategy_parameters",
    )
    async def update_strategy_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Update strategy parameters.

        Args:
            parameters: New strategy parameters
        """
        self.logger.info("Updating strategy parameters")

        try:
            # Update configuration
            for key, value in parameters.items():
                if key in self.config:
                    self.config[key] = value

            # Update state manager
            self.state_manager.set_state("strategy_parameters", parameters)

            self.logger.info("Strategy parameters updated successfully")

        except Exception as e:
            self.logger.error(
                f"Failed to update strategy parameters: {e}", exc_info=True
            )

    @handle_errors(
        exceptions=(Exception,), default_return={}, context="get_strategy_performance"
    )
    async def get_strategy_performance(self) -> Dict[str, Any]:
        """
        Get strategy performance metrics.

        Returns:
            Strategy performance metrics
        """
        return self.strategy_performance.copy()

    async def _handle_analysis_result(self, event) -> None:
        """Handle analysis result events"""
        analysis_result = event.data
        await self.formulate_strategy(analysis_result)

    @handle_data_processing_errors(
        exceptions=(ValueError, TypeError, ZeroDivisionError),
        default_return="NEUTRAL",
        context="determine_positional_bias",
    )
    async def _determine_positional_bias(self, analysis_result: AnalysisResult) -> str:
        """Determine positional bias based on analysis"""
        try:
            confidence = analysis_result.confidence
            signal = analysis_result.signal
            market_regime = analysis_result.market_regime

            # High confidence signals
            if confidence > 0.7:
                if signal == "BUY":
                    return "LONG"
                elif signal == "SELL":
                    return "SHORT"

            # Medium confidence with strong market regime
            elif confidence > 0.5:
                if market_regime == "BULL_TREND" and signal == "BUY":
                    return "LONG"
                elif market_regime == "BEAR_TREND" and signal == "SELL":
                    return "SHORT"

            # Low confidence or neutral signals
            return "NEUTRAL"

        except Exception as e:
            self.logger.error(f"Error determining positional bias: {e}")
            return "NEUTRAL"

    @handle_data_processing_errors(
        exceptions=(ValueError, TypeError, ZeroDivisionError),
        default_return=1.0,
        context="determine_leverage_cap",
    )
    async def _determine_leverage_cap(self, analysis_result: AnalysisResult) -> float:
        """Determine leverage cap based on analysis"""
        try:
            confidence = analysis_result.confidence
            market_regime = analysis_result.market_regime
            risk_metrics = analysis_result.risk_metrics

            # Base leverage from config
            base_leverage = self.config.get("base_leverage", 1.0)

            # Adjust based on confidence
            if confidence > 0.8:
                leverage_multiplier = 1.5
            elif confidence > 0.6:
                leverage_multiplier = 1.2
            else:
                leverage_multiplier = 1.0

            # Adjust based on market regime
            if market_regime == "BULL_TREND":
                regime_multiplier = 1.1
            elif market_regime == "BEAR_TREND":
                regime_multiplier = 1.1
            else:
                regime_multiplier = 1.0

            # Adjust based on risk metrics
            risk_score = risk_metrics.get("liquidation_risk", 0.5)
            if risk_score < 0.3:
                risk_multiplier = 1.2
            elif risk_score > 0.7:
                risk_multiplier = 0.8
            else:
                risk_multiplier = 1.0

            leverage_cap = (
                base_leverage
                * leverage_multiplier
                * regime_multiplier
                * risk_multiplier
            )

            # Apply maximum leverage limit
            max_leverage = self.config.get("max_leverage", 10.0)
            leverage_cap = min(leverage_cap, max_leverage)

            return leverage_cap

        except Exception as e:
            self.logger.error(f"Error determining leverage cap: {e}")
            return 1.0

    @handle_data_processing_errors(
        exceptions=(ValueError, TypeError, ZeroDivisionError),
        default_return=0.1,
        context="determine_max_notional_size",
    )
    async def _determine_max_notional_size(
        self, analysis_result: AnalysisResult
    ) -> float:
        """Determine maximum notional size based on analysis"""
        try:
            confidence = analysis_result.confidence
            market_health = analysis_result.features.get("market_health", 0.5)

            # Base position size from config
            base_position_size = self.config.get("base_position_size", 0.1)

            # Adjust based on confidence
            if confidence > 0.8:
                size_multiplier = 1.5
            elif confidence > 0.6:
                size_multiplier = 1.2
            else:
                size_multiplier = 1.0

            # Adjust based on market health
            if market_health > 0.7:
                health_multiplier = 1.2
            elif market_health < 0.3:
                health_multiplier = 0.8
            else:
                health_multiplier = 1.0

            max_notional_size = base_position_size * size_multiplier * health_multiplier

            # Apply maximum position size limit
            max_position_size = self.config.get("max_position_size", 0.5)
            max_notional_size = min(max_notional_size, max_position_size)

            return max_notional_size

        except Exception as e:
            self.logger.error(f"Error determining max notional size: {e}")
            return 0.1

    @handle_data_processing_errors(
        exceptions=(ValueError, TypeError, ZeroDivisionError),
        default_return={},
        context="calculate_risk_parameters",
    )
    async def _calculate_risk_parameters(
        self, analysis_result: AnalysisResult
    ) -> Dict[str, float]:
        """Calculate risk parameters based on analysis"""
        try:
            risk_metrics = analysis_result.risk_metrics
            technical_indicators = analysis_result.technical_indicators

            risk_parameters = {
                "stop_loss_pct": self.config.get("default_stop_loss", 0.05),
                "take_profit_pct": self.config.get("default_take_profit", 0.10),
                "max_drawdown": self.config.get("max_drawdown", 0.20),
                "risk_per_trade": self.config.get("risk_per_trade", 0.02),
            }

            # Adjust based on volatility
            if "atr" in technical_indicators:
                atr = technical_indicators["atr"]
                current_price = analysis_result.features.get("close", 100)
                if current_price > 0:
                    atr_pct = atr / current_price
                    risk_parameters["stop_loss_pct"] = max(atr_pct * 2, 0.02)
                    risk_parameters["take_profit_pct"] = max(atr_pct * 4, 0.05)

            # Adjust based on liquidation risk
            liquidation_risk = risk_metrics.get("liquidation_risk", 0.5)
            if liquidation_risk > 0.7:
                risk_parameters["risk_per_trade"] *= 0.5
            elif liquidation_risk < 0.3:
                risk_parameters["risk_per_trade"] *= 1.2

            return risk_parameters

        except Exception as e:
            self.logger.error(f"Error calculating risk parameters: {e}")
            return {
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.10,
                "max_drawdown": 0.20,
                "risk_per_trade": 0.02,
            }

    @handle_data_processing_errors(
        exceptions=(ValueError, TypeError, ZeroDivisionError),
        default_return={},
        context="assess_market_conditions",
    )
    async def _assess_market_conditions(
        self, analysis_result: AnalysisResult
    ) -> Dict[str, Any]:
        """Assess market conditions based on analysis"""
        try:
            market_conditions = {
                "volatility": "MEDIUM",
                "trend_strength": "MEDIUM",
                "liquidity": "MEDIUM",
                "market_sentiment": "NEUTRAL",
            }

            # Assess volatility
            technical_indicators = analysis_result.technical_indicators
            if "atr" in technical_indicators:
                atr = technical_indicators["atr"]
                current_price = analysis_result.features.get("close", 100)
                if current_price > 0:
                    atr_pct = atr / current_price
                    if atr_pct > 0.05:
                        market_conditions["volatility"] = "HIGH"
                    elif atr_pct < 0.02:
                        market_conditions["volatility"] = "LOW"

            # Assess trend strength
            market_regime = analysis_result.market_regime
            if market_regime in ["BULL_TREND", "BEAR_TREND"]:
                market_conditions["trend_strength"] = "STRONG"
            elif market_regime == "SIDEWAYS":
                market_conditions["trend_strength"] = "WEAK"

            # Assess market sentiment
            confidence = analysis_result.confidence
            if confidence > 0.7:
                market_conditions["market_sentiment"] = "BULLISH"
            elif confidence < 0.3:
                market_conditions["market_sentiment"] = "BEARISH"

            return market_conditions

        except Exception as e:
            self.logger.error(f"Error assessing market conditions: {e}")
            return {
                "volatility": "MEDIUM",
                "trend_strength": "MEDIUM",
                "liquidity": "MEDIUM",
                "market_sentiment": "NEUTRAL",
            }

    async def _update_strategy_performance(
        self, strategy_result: StrategyResult
    ) -> None:
        """Update strategy performance tracking"""
        try:
            timestamp = strategy_result.timestamp
            symbol = strategy_result.symbol

            if symbol not in self.strategy_performance:
                self.strategy_performance[symbol] = {
                    "total_decisions": 0,
                    "long_decisions": 0,
                    "short_decisions": 0,
                    "neutral_decisions": 0,
                    "avg_leverage": 0.0,
                    "avg_position_size": 0.0,
                    "last_updated": timestamp,
                }

            performance = self.strategy_performance[symbol]
            performance["total_decisions"] += 1

            if strategy_result.position_bias == "LONG":
                performance["long_decisions"] += 1
            elif strategy_result.position_bias == "SHORT":
                performance["short_decisions"] += 1
            else:
                performance["neutral_decisions"] += 1

            # Update averages
            current_avg_leverage = performance["avg_leverage"]
            current_avg_position_size = performance["avg_position_size"]
            total_decisions = performance["total_decisions"]

            performance["avg_leverage"] = (
                current_avg_leverage * (total_decisions - 1)
                + strategy_result.leverage_cap
            ) / total_decisions
            performance["avg_position_size"] = (
                current_avg_position_size * (total_decisions - 1)
                + strategy_result.max_notional_size
            ) / total_decisions

            performance["last_updated"] = timestamp

        except Exception as e:
            self.logger.error(f"Error updating strategy performance: {e}")
