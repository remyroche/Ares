import asyncio
import time
from typing import Dict, Any

from src.exchange.binance import BinanceExchange
from src.utils.logger import logger
from src.config import settings
from src.utils.state_manager import StateManager

class Tactician:
    """
    The Tactician is the final decision-making and execution module.
    It now acts as a strategy dispatcher, selecting the appropriate tactics
    (e.g., trend-following, mean-reversion) based on the market regime.
    """

    def __init__(self, exchange_client: BinanceExchange, state_manager: StateManager):
        self.exchange = exchange_client
        self.state_manager = state_manager
        self.logger = logger.getChild('Tactician')
        self.config = settings.get("tactician", {})
        self.trade_symbol = settings.trade_symbol
        self.last_strategist_timestamp = None
        
        self.current_position = self.state_manager.get_state("current_position", self._get_default_position())
        self.logger.info(f"Tactician initialized. Position: {self.current_position.get('direction')}")

    def _get_default_position(self):
        """Returns the default structure for an empty position."""
        return {"direction": None, "size": 0.0}

    async def start(self):
        """Starts the main tactician loop."""
        self.logger.info("Tactician started. Waiting for new strategist parameters...")
        while True:
            try:
                if self.state_manager.get_state("is_trading_paused", False):
                    await asyncio.sleep(10)
                    continue

                strategist_params = self.state_manager.get_state("strategist_params")
                
                if strategist_params and strategist_params.get("timestamp") != self.last_strategist_timestamp:
                    self.last_strategist_timestamp = strategist_params.get("timestamp")
                    self.logger.info("New strategist parameters detected. Running tactical assessment.")
                    
                    analyst_intelligence = self.state_manager.get_state("analyst_intelligence")
                    if analyst_intelligence:
                        await self.run_strategy_dispatcher(strategist_params, analyst_intelligence)
                
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                self.logger.info("Tactician task cancelled.")
                break
            except Exception as e:
                self.logger.error(f"An error occurred in the Tactician loop: {e}", exc_info=True)
                await asyncio.sleep(10)

    async def run_strategy_dispatcher(self, strat_params: Dict, analyst_intel: Dict):
        """
        Acts as a dispatcher, selecting and running the correct tactical logic
        based on the current market regime.
        """
        self.logger.info("--- Starting Tactical Assessment ---")
        
        if self.current_position.get("size", 0) > 0:
            exit_decision = self._check_exit_conditions(analyst_intel)
            if exit_decision:
                await self.execute_close_position(exit_decision['reason'])
                return
            self.logger.info("Holding open position. No exit conditions met.")
            return

        market_regime = analyst_intel.get("market_regime", "UNCERTAIN")
        self.logger.info(f"Market Regime: {market_regime}. Dispatching to appropriate tactical function.")

        entry_decision = None
        if market_regime in ["BULL_TREND", "BEAR_TREND"]:
            entry_decision = self._assess_trend_following_entry(strat_params, analyst_intel)
        elif market_regime == "SIDEWAYS_RANGE":
            entry_decision = self._assess_mean_reversion_entry(strat_params, analyst_intel)
        else:
            self.logger.info(f"No specific tactics for {market_regime} regime. Holding.")

        if entry_decision:
            await self.execute_open_position(entry_decision)
        else:
            self.logger.info("Holding. No entry conditions met for the current regime.")

    def _assess_trend_following_entry(self, strat_params: Dict, analyst_intel: Dict) -> Dict | None:
        """Checks rules for entering a position in a trending market."""
        self.logger.info("Assessing Trend-Following entry...")
        position_bias = strat_params.get("positional_bias", "NEUTRAL")
        confidence = analyst_intel.get("directional_confidence_score", 0.5)
        min_confidence = self.config.get("min_confidence_for_entry", 0.65)
        
        if position_bias == "NEUTRAL" or confidence < min_confidence:
            return None

        # Standard entry logic for trends
        return self._prepare_standard_order(position_bias, strat_params, analyst_intel)

    def _assess_mean_reversion_entry(self, strat_params: Dict, analyst_intel: Dict) -> Dict | None:
        """Checks rules for entering a position in a ranging market."""
        self.logger.info("Assessing Mean-Reversion entry...")
        tech_analysis = analyst_intel.get("technical_analysis", {})
        sr_levels = analyst_intel.get("support_resistance", {})
        price = tech_analysis.get("current_price")
        
        if not price or not sr_levels: return None

        # Find closest support and resistance
        support = max([lvl['level_price'] for lvl in sr_levels if lvl['level_price'] < price], default=None)
        resistance = min([lvl['level_price'] for lvl in sr_levels if lvl['level_price'] > price], default=None)

        if not support or not resistance: return None

        # Entry condition: price is very close to support or resistance
        proximity_pct = self.config.get("mean_reversion_proximity_pct", 0.005) # 0.5%
        
        position_bias = "NEUTRAL"
        if abs(price - support) / price < proximity_pct:
            position_bias = "LONG" # Buy near support
            self.logger.info(f"Price ({price}) is near support ({support}). Assessing LONG entry.")
        elif abs(price - resistance) / price < proximity_pct:
            position_bias = "SHORT" # Sell near resistance
            self.logger.info(f"Price ({price}) is near resistance ({resistance}). Assessing SHORT entry.")
        
        if position_bias != "NEUTRAL":
            return self._prepare_standard_order(position_bias, strat_params, analyst_intel, is_mean_reversion=True)
        
        return None

    def _prepare_standard_order(self, direction: str, strat_params: Dict, analyst_intel: Dict, is_mean_reversion: bool = False) -> Dict | None:
        """A helper function to calculate and structure a potential trade order."""
        tech_analysis = analyst_intel.get("technical_analysis", {})
        current_price = tech_analysis.get("current_price")
        atr = tech_analysis.get("atr")
        if not current_price or not atr: return None

        lss = analyst_intel.get("liquidation_safety_score", 0)
        leverage = self._determine_leverage(lss, strat_params["max_allowable_leverage"])
        
        # Use tighter stops for mean reversion
        sl_multiplier = self.config.get("atr_sl_multiplier_mean_reversion" if is_mean_reversion else "atr_sl_multiplier_trend", 1.5)
        tp_multiplier = self.config.get("atr_tp_multiplier_mean_reversion" if is_mean_reversion else "atr_tp_multiplier_trend", 2.0)

        if direction == "LONG":
            stop_loss = current_price - (atr * sl_multiplier)
            take_profit = current_price + (atr * tp_multiplier)
        else: # SHORT
            stop_loss = current_price + (atr * sl_multiplier)
            take_profit = current_price - (atr * tp_multiplier)

        quantity = self._calculate_position_size(current_price, stop_loss, leverage)
        if quantity <= 0: return None

        return {
            "direction": direction, "quantity": quantity, "leverage": leverage,
            "entry_price": current_price, "stop_loss": stop_loss, "take_profit": take_profit,
        }

    def _check_exit_conditions(self, analyst_intel: Dict) -> Dict | None:
        """Checks rules for exiting an existing position."""
        pos = self.current_position
        price = analyst_intel["technical_analysis"]["current_price"]
        
        if pos.get('direction') == "LONG":
            if price >= pos.get('take_profit', float('inf')): return {"reason": "Take Profit Hit"}
            if price <= pos.get('stop_loss', float('-inf')): return {"reason": "Stop Loss Hit"}
        elif pos.get('direction') == "SHORT":
            if price <= pos.get('take_profit', float('-inf')): return {"reason": "Take Profit Hit"}
            if price >= pos.get('stop_loss', float('inf')): return {"reason": "Stop Loss Hit"}

        return None

    def _calculate_position_size(self, current_price: float, stop_loss_price: float, leverage: int) -> float:
        """Calculates position size based on risk per trade and stop loss distance."""
        capital = self.state_manager.get_state("account_equity", settings.get("initial_equity", 10000))
        risk_per_trade_pct = self.config.get("risk_per_trade_pct", 0.01)
        risk_multiplier = self.state_manager.get_state("global_risk_multiplier", 1.0)
        
        max_risk_usd = capital * risk_per_trade_pct * risk_multiplier
        stop_loss_distance = abs(current_price - stop_loss_price)
        if stop_loss_distance == 0: return 0.0

        units = max_risk_usd / stop_loss_distance
        return round(units, 3)

    def _determine_leverage(self, lss: float, max_leverage_cap: int) -> int:
        """Determines leverage based on Liquidation Safety Score (LSS)."""
        initial_leverage = self.config.get("initial_leverage", 25)
        if lss <= 50: return initial_leverage
        
        scaled_leverage = initial_leverage + ((lss - 50) / 50) * (max_leverage_cap - initial_leverage)
        return min(max(initial_leverage, int(scaled_leverage)), max_leverage_cap)

    async def execute_open_position(self, decision: Dict):
        """Executes the logic to open a new position."""
        self.logger.info(f"Executing OPEN for {decision['direction']} position: Qty={decision['quantity']:.3f}, SL={decision['stop_loss']:.2f}, TP={decision['take_profit']:.2f}")
        try:
            order = await self.exchange.create_order(
                symbol=self.trade_symbol, side="BUY" if decision['direction'] == "LONG" else "SELL",
                order_type="MARKET", quantity=decision['quantity']
            )
            self.logger.info(f"Entry order placed successfully: {order}")
            
            self.current_position = {
                "direction": decision["direction"], "size": float(order.get('executedQty', decision['quantity'])),
                "entry_price": float(order.get('avgPrice', decision['entry_price'])),
                "stop_loss": decision["stop_loss"], "take_profit": decision["take_profit"],
                "leverage": decision["leverage"], "entry_timestamp": time.time()
            }
            self.state_manager.set_state("current_position", self.current_position)
            self.logger.info(f"New position state saved: {self.current_position}")
            
        except Exception as e:
            self.logger.error(f"Failed to execute trade entry: {e}", exc_info=True)

    async def execute_close_position(self, reason: str):
        """Executes the logic to close the current open position."""
        self.logger.warning(f"Executing CLOSE for {self.current_position['direction']} position. Reason: {reason}")
        try:
            order = await self.exchange.create_order(
                symbol=self.trade_symbol, side="SELL" if self.current_position['direction'] == "LONG" else "BUY",
                order_type="MARKET", quantity=self.current_position['size']
            )
            self.logger.info(f"Close order placed successfully: {order}")
            
            self.current_position = self._get_default_position()
            self.state_manager.set_state("current_position", self.current_position)
            self.logger.info("Position closed and state has been reset.")

        except Exception as e:
            self.logger.error(f"Failed to execute close order: {e}", exc_info=True)
