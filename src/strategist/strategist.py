import asyncio
import time
from typing import Dict, Any, Optional, Union # Ensure Union is imported

from src.utils.logger import system_logger
from src.config import settings
from src.utils.state_manager import StateManager
from src.exchange.binance import BinanceExchange # Import if needed for type hinting
from src.utils.error_handler import (
    handle_errors,
    handle_data_processing_errors,
    handle_type_conversions,
    error_context,
    ErrorRecoveryStrategies
)

class Strategist:
    """
    The Strategist translates the Analyst's rich intelligence into a high-level trading plan.
    It now uses detailed technical analysis (VWAP, MAs, etc.) to formulate its strategy.
    """

    def __init__(self, exchange_client: Optional[BinanceExchange] = None, state_manager: Optional[StateManager] = None):
        # Allow exchange_client and state_manager to be optional for ModelManager instantiation
        self.exchange = exchange_client
        self.state_manager = state_manager
        self.logger = system_logger.getChild('Strategist')
        self.config = settings.get("strategist", {})
        self.last_analyst_timestamp = None
        self.logger.info("Strategist initialized.")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="strategist_start"
    )
    async def start(self):
        """
        Starts the main strategist loop, waiting for new intelligence from the Analyst.
        """
        if self.state_manager is None:
            self.logger.error("Strategist cannot start: StateManager not provided.")
            return

        self.logger.info("Strategist started. Waiting for new analyst intelligence...")
        while True:
            try:
                analyst_intelligence = self.state_manager.get_state("analyst_intelligence")
                
                if analyst_intelligence and analyst_intelligence.get("timestamp") != self.last_analyst_timestamp:
                    self.last_analyst_timestamp = analyst_intelligence.get("timestamp")
                    self.logger.info("New analyst intelligence detected. Running strategy formulation.")
                    await self.formulate_strategy(analyst_intelligence)
                
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                self.logger.info("Strategist task cancelled.")
                break
            except Exception as e:
                self.logger.error(f"An error occurred in the Strategist loop: {e}", exc_info=True)
                await asyncio.sleep(10)

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="formulate_strategy"
    )
    async def formulate_strategy(self, analyst_intelligence: Dict[str, Any]):
        """
        Formulates the high-level strategy based on the Analyst's detailed findings.
        """
        self.logger.info("--- Starting Strategy Formulation ---")
        try:
            market_regime = analyst_intelligence.get("market_regime", "UNCERTAIN")
            liquidation_risk = analyst_intelligence.get("liquidation_risk_score", 100)
            confidence = analyst_intelligence.get("directional_confidence_score", 0.5)
            tech_analysis = analyst_intelligence.get("technical_signals", {}) # Corrected key

            # Determine positional bias using new detailed data
            positional_bias = self._determine_positional_bias(market_regime, tech_analysis)

            # Determine leverage cap based on risk and confidence
            leverage_cap = self._determine_leverage_cap(liquidation_risk, confidence)
            
            # Determine max notional trade size
            max_notional_size = self._determine_max_notional_size(market_regime)

            strategist_params = {
                "timestamp": int(time.time() * 1000),
                "positional_bias": positional_bias,
                "max_allowable_leverage": leverage_cap,
                "max_notional_trade_size": max_notional_size,
                "source_analyst_timestamp": self.last_analyst_timestamp
            }

            if self.state_manager: # Only set state if state_manager is available
                self.state_manager.set_state("strategist_params", strategist_params)
            self.logger.info(f"Strategy formulated. Bias: {positional_bias}, Leverage Cap: {leverage_cap}x")
            self.logger.debug(f"Strategist Params: {strategist_params}")

        except Exception as e:
            self.logger.error(f"Error during strategy formulation: {e}", exc_info=True)

    @handle_errors(
        exceptions=(KeyError, TypeError, AttributeError),
        default_return="NEUTRAL",
        context="determine_positional_bias"
    )
    def _determine_positional_bias(self, market_regime: str, tech_analysis: Dict) -> str:
        """Determines the trading bias using a combination of regime and technicals."""
        
        regime_bias_map = self.config.get("regime_to_bias_map", {"BULL_TREND": "LONG", "BEAR_TREND": "SHORT"})
        regime_bias = regime_bias_map.get(market_regime, "NEUTRAL")

        ma_bias = "NEUTRAL"
        vwap_bias = "NEUTRAL"
        
        mas = tech_analysis.get('moving_averages', {})
        if mas.get('sma_9', 0) > mas.get('sma_50', 0): ma_bias = "LONG"
        elif mas.get('sma_9', 0) < mas.get('sma_50', 0): ma_bias = "SHORT"

        # VWAP analysis
        vwap_data = tech_analysis.get('vwap', {})
        if vwap_data.get('price_vs_vwap', 0) > 0: vwap_bias = "LONG"
        elif vwap_data.get('price_vs_vwap', 0) < 0: vwap_bias = "SHORT"

        # Combine biases (simple majority)
        biases = [regime_bias, ma_bias, vwap_bias]
        long_count = biases.count("LONG")
        short_count = biases.count("SHORT")
        
        if long_count > short_count:
            return "LONG"
        elif short_count > long_count:
            return "SHORT"
        else:
            return "NEUTRAL"

    @handle_errors(
        exceptions=(ValueError, TypeError, ZeroDivisionError),
        default_return=1,
        context="determine_leverage_cap"
    )
    def _determine_leverage_cap(self, liquidation_risk: float, confidence: float) -> int:
        """Determines the maximum allowable leverage based on risk and confidence."""
        # Base leverage cap from config
        base_leverage = self.config.get("base_leverage_cap", 10)
        
        # Adjust based on liquidation risk (higher risk = lower leverage)
        risk_factor = max(0.1, 1 - (liquidation_risk / 100))
        
        # Adjust based on confidence (higher confidence = higher leverage)
        confidence_factor = max(0.1, min(2.0, confidence * 2))
        
        leverage_cap = int(base_leverage * risk_factor * confidence_factor)
        return max(1, min(leverage_cap, 20))  # Clamp between 1 and 20

    @handle_errors(
        exceptions=(ValueError, TypeError),
        default_return=1000.0,
        context="determine_max_notional_size"
    )
    def _determine_max_notional_size(self, market_regime: str) -> float:
        """Determines the maximum notional trade size based on market regime."""
        base_size = self.config.get("base_notional_size", 1000.0)
        
        # Adjust based on market regime
        regime_multipliers = {
            "BULL_TREND": 1.5,
            "BEAR_TREND": 0.8,
            "SIDEWAYS": 1.0,
            "HIGH_IMPACT": 0.5
        }
        
        multiplier = regime_multipliers.get(market_regime, 1.0)
        return base_size * multiplier

