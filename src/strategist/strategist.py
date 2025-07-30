import asyncio
import time
from typing import Dict, Any, Optional, Union # Ensure Union is imported
import pandas as pd # Added for pd.isna

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

            # Incorporate volatility targeting
            vol_targeting_multiplier = self._get_volatility_targeting_multiplier(analyst_intelligence)
            adjusted_leverage_cap = self._apply_volatility_targeting(leverage_cap, vol_targeting_multiplier)
            adjusted_notional_size = self._apply_volatility_targeting(max_notional_size, vol_targeting_multiplier)

            strategist_params = {
                "timestamp": int(time.time() * 1000),
                "positional_bias": positional_bias,
                "max_allowable_leverage": adjusted_leverage_cap,
                "max_notional_trade_size": adjusted_notional_size,
                "volatility_multiplier": vol_targeting_multiplier,
                "base_leverage": leverage_cap,
                "base_notional_size": max_notional_size,
                "source_analyst_timestamp": self.last_analyst_timestamp
            }

            if self.state_manager: # Only set state if state_manager is available
                self.state_manager.set_state("strategist_params", strategist_params)
            self.logger.info(f"Strategy formulated. Bias: {positional_bias}, Leverage Cap: {adjusted_leverage_cap}x")
            self.logger.debug(f"Strategist Params: {strategist_params}")

        except Exception as e:
            self.logger.error(f"Error during strategy formulation: {e}", exc_info=True)

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

    @handle_errors(
        exceptions=(ValueError, TypeError, KeyError),
        default_return=1.0,
        context="get_volatility_targeting_multiplier"
    )
    def _get_volatility_targeting_multiplier(self, analyst_intelligence: Dict) -> float:
        """Extract volatility targeting multiplier from analyst intelligence."""
        try:
            # Get features from analyst intelligence
            features = analyst_intelligence.get("latest_features", {})
            
            # Check which volatility targeting method to use
            vol_method = self.config.get("volatility_targeting_method", "EWMA")
            
            multiplier_map = {
                "Simple": features.get("Vol_Target_Multiplier_Simple", 1.0),
                "EWMA": features.get("Vol_Target_Multiplier_EWMA", 1.0),
                "Adaptive": features.get("Vol_Target_Multiplier_Adaptive", 1.0),
                "Regime": features.get("Vol_Target_Multiplier_Regime", 1.0),
                "Kelly": features.get("Vol_Target_Multiplier_Kelly", 1.0),
                "Dynamic": features.get("Vol_Target_Multiplier_Dynamic", 1.0),
                "Parkinson": features.get("Vol_Target_Multiplier_Parkinson", 1.0)
            }
            
            multiplier = multiplier_map.get(vol_method, 1.0)
            
            # Ensure multiplier is reasonable
            if pd.isna(multiplier) or multiplier <= 0:
                multiplier = 1.0
                
            # Additional safety bounds
            max_vol_multiplier = self.config.get("max_volatility_multiplier", 3.0)
            min_vol_multiplier = self.config.get("min_volatility_multiplier", 0.1)
            
            multiplier = max(min_vol_multiplier, min(max_vol_multiplier, multiplier))
            
            self.logger.debug(f"Volatility targeting multiplier ({vol_method}): {multiplier}")
            return multiplier
            
        except Exception as e:
            self.logger.warning(f"Error extracting volatility multiplier: {e}")
            return 1.0

    @handle_errors(
        exceptions=(ValueError, TypeError),
        default_return=1.0,
        context="apply_volatility_targeting"
    )
    def _apply_volatility_targeting(self, base_value: float, vol_multiplier: float) -> float:
        """Apply volatility targeting to a base value (leverage or position size)."""
        try:
            adjusted_value = base_value * vol_multiplier
            
            # Ensure result is reasonable
            if adjusted_value <= 0 or pd.isna(adjusted_value):
                return base_value
                
            return adjusted_value
            
        except Exception as e:
            self.logger.warning(f"Error applying volatility targeting: {e}")
            return base_value

    @handle_errors(
        exceptions=(KeyError, TypeError, AttributeError),
        default_return="NEUTRAL",
        context="determine_positional_bias"
    )
    def _determine_positional_bias(self, market_regime: str, tech_analysis: Dict) -> str:
        """Determines the trading bias using a combination of regime, technicals, and enhanced indicators."""
        
        regime_bias_map = self.config.get("regime_to_bias_map", {"BULL_TREND": "LONG", "BEAR_TREND": "SHORT"})
        regime_bias = regime_bias_map.get(market_regime, "NEUTRAL")

        ma_bias = "NEUTRAL"
        vwap_bias = "NEUTRAL"
        trend_bias = "NEUTRAL"
        momentum_bias = "NEUTRAL"
        
        # Moving averages analysis
        mas = tech_analysis.get('moving_averages', {})
        if mas.get('sma_9', 0) > mas.get('sma_50', 0): 
            ma_bias = "LONG"
        elif mas.get('sma_9', 0) < mas.get('sma_50', 0): 
            ma_bias = "SHORT"

        # VWAP analysis
        vwap_ratio = tech_analysis.get('price_to_vwap_ratio', 1.0)
        if vwap_ratio > 1.02:  # Price 2% above VWAP
            vwap_bias = "LONG"
        elif vwap_ratio < 0.98:  # Price 2% below VWAP
            vwap_bias = "SHORT"

        # Enhanced trend indicators
        psar = tech_analysis.get('parabolic_sar', {})
        if psar.get('trend') == 'BULLISH':
            trend_bias = "LONG"
        elif psar.get('trend') == 'BEARISH':
            trend_bias = "SHORT"
            
        # SuperTrend confirmation
        supertrend = tech_analysis.get('supertrend', {})
        if supertrend.get('trend') == 'BULLISH' and trend_bias == "LONG":
            trend_bias = "LONG"  # Confirmed
        elif supertrend.get('trend') == 'BEARISH' and trend_bias == "SHORT":
            trend_bias = "SHORT"  # Confirmed
        elif supertrend.get('trend') in ['BULLISH', 'BEARISH']:
            trend_bias = "LONG" if supertrend.get('trend') == 'BULLISH' else "SHORT"

        # Enhanced momentum indicators
        williams_r = tech_analysis.get('williams_r', 0)
        mfi = tech_analysis.get('mfi', 50)
        roc = tech_analysis.get('roc', 0)
        
        momentum_signals = []
        if williams_r < -80:  # Oversold
            momentum_signals.append("LONG")
        elif williams_r > -20:  # Overbought
            momentum_signals.append("SHORT")
            
        if mfi < 20:  # Oversold
            momentum_signals.append("LONG")
        elif mfi > 80:  # Overbought
            momentum_signals.append("SHORT")
            
        if roc > 0:
            momentum_signals.append("LONG")
        elif roc < 0:
            momentum_signals.append("SHORT")
            
        # Momentum bias based on majority
        if momentum_signals.count("LONG") > momentum_signals.count("SHORT"):
            momentum_bias = "LONG"
        elif momentum_signals.count("SHORT") > momentum_signals.count("LONG"):
            momentum_bias = "SHORT"

        # Combine all biases with weights
        bias_weights = {
            regime_bias: self.config.get("regime_weight", 0.4),
            ma_bias: self.config.get("ma_weight", 0.2),
            vwap_bias: self.config.get("vwap_weight", 0.15),
            trend_bias: self.config.get("trend_weight", 0.15),
            momentum_bias: self.config.get("momentum_weight", 0.1)
        }
        
        long_weight = sum(weight for bias, weight in bias_weights.items() if bias == "LONG")
        short_weight = sum(weight for bias, weight in bias_weights.items() if bias == "SHORT")
        
        # Require a minimum threshold for directional bias
        min_threshold = self.config.get("bias_threshold", 0.6)
        
        if long_weight >= min_threshold:
            return "LONG"
        elif short_weight >= min_threshold:
            return "SHORT"
        else:
            return "NEUTRAL"

