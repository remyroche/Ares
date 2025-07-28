import asyncio
import time
from typing import Dict, Any

from src.utils.logger import logger
from src.config import settings
from src.utils.state_manager import StateManager

class Strategist:
    """
    The Strategist translates the Analyst's rich intelligence into a high-level trading plan.
    It now uses detailed technical analysis (VWAP, MAs, etc.) to formulate its strategy.
    """

    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.logger = logger.getChild('Strategist')
        self.config = settings.get("strategist", {})
        self.last_analyst_timestamp = None
        self.logger.info("Strategist initialized.")

    async def start(self):
        """
        Starts the main strategist loop, waiting for new intelligence from the Analyst.
        """
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

    async def formulate_strategy(self, analyst_intelligence: Dict[str, Any]):
        """
        Formulates the high-level strategy based on the Analyst's detailed findings.
        """
        self.logger.info("--- Starting Strategy Formulation ---")
        try:
            # Extract rich data from the analyst's packet
            market_regime = analyst_intelligence.get("market_regime", "UNCERTAIN")
            liquidation_risk = analyst_intelligence.get("liquidation_risk_score", 100)
            confidence = analyst_intelligence.get("directional_confidence_score", 0.5)
            tech_analysis = analyst_intelligence.get("technical_analysis", {})

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

            self.state_manager.set_state("strategist_params", strategist_params)
            self.logger.info(f"Strategy formulated. Bias: {positional_bias}, Leverage Cap: {leverage_cap}x")
            self.logger.debug(f"Strategist Params: {strategist_params}")

        except Exception as e:
            self.logger.error(f"Error during strategy formulation: {e}", exc_info=True)

    def _determine_positional_bias(self, market_regime: str, tech_analysis: Dict) -> str:
        """Determines the trading bias using a combination of regime and technicals."""
        
        # Bias from regime
        regime_bias_map = self.config.get("regime_to_bias_map", {"BULL_TREND": "LONG", "BEAR_TREND": "SHORT"})
        regime_bias = regime_bias_map.get(market_regime, "NEUTRAL")

        # Bias from technicals
        ma_bias = "NEUTRAL"
        vwap_bias = "NEUTRAL"
        
        mas = tech_analysis.get('moving_averages', {})
        if mas.get('sma_9', 0) > mas.get('sma_50', 0): ma_bias = "LONG"
        elif mas.get('sma_9', 0) < mas.get('sma_50', 0): ma_bias = "SHORT"

        price_to_vwap = tech_analysis.get('price_to_vwap_ratio', 1.0)
        if price_to_vwap > 1.005: # Price >0.5% above VWAP
            vwap_bias = "LONG"
        elif price_to_vwap < 0.995: # Price >0.5% below VWAP
            vwap_bias = "SHORT"

        # Combine biases (simple voting)
        biases = [regime_bias, ma_bias, vwap_bias]
        long_votes = biases.count("LONG")
        short_votes = biases.count("SHORT")

        if long_votes > short_votes:
            return "LONG"
        if short_votes > long_votes:
            return "SHORT"
        
        return "NEUTRAL"

    def _determine_leverage_cap(self, liquidation_risk: float, confidence: float) -> int:
        """Calculates an appropriate leverage cap."""
        base_leverage = self.config.get("base_leverage", 10)
        risk_factor = 1 - (liquidation_risk / 100)
        confidence_factor = 0.8 + (confidence * 0.4)
        leverage = base_leverage * risk_factor * confidence_factor
        return int(max(1, min(self.config.get("max_leverage_limit", 25), leverage)))

    def _determine_max_notional_size(self, market_regime: str) -> float:
        """Determines the maximum notional size for a single trade."""
        base_size = self.config.get("base_notional_trade_size", 5000)
        size_multiplier = self.config.get("regime_size_multipliers", {
            "BULL_TREND": 1.0, "BEAR_TREND": 1.0, "SIDEWAYS_RANGE": 0.5
        })
        return base_size * size_multiplier.get(market_regime, 0.25)
