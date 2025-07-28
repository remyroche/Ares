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
    It synthesizes intelligence and strategy to make concrete trading decisions,
    implementing a sophisticated rule-based engine for entries, exits, and position management.
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
                strategist_params = self.state_manager.get_state("strategist_params")
                
                if strategist_params and strategist_params.get("timestamp") != self.last_strategist_timestamp:
                    self.last_strategist_timestamp = strategist_params.get("timestamp")
                    self.logger.info("New strategist parameters detected. Running tactical assessment.")
                    
                    analyst_intelligence = self.state_manager.get_state("analyst_intelligence")
                    if analyst_intelligence:
                        await self.run_rule_engine(strategist_params, analyst_intelligence)
                
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                self.logger.info("Tactician task cancelled.")
                break
            except Exception as e:
                self.logger.error(f"An error occurred in the Tactician loop: {e}", exc_info=True)
                await asyncio.sleep(10)

    async def run_rule_engine(self, strat_params: Dict, analyst_intel: Dict):
        """The main rule engine that determines the trading action."""
        self.logger.info("--- Starting Tactical Assessment ---")
        
        if self.current_position.get("size", 0) > 0:
            exit_decision = self._check_exit_conditions(analyst_intel)
            if exit_decision:
                await self.execute_close_position(exit_decision['reason'])
                return
            
            self.logger.info("Holding open position. No exit conditions met.")
            return

        entry_decision = self._check_entry_conditions(strat_params, analyst_intel)
        if entry_decision:
            await self.execute_open_position(entry_decision)
            return

        self.logger.info("Holding. No entry conditions met.")

    def _check_exit_conditions(self, analyst_intel: Dict) -> Dict | None:
        """Checks rules for exiting an existing position."""
        pos = self.current_position
        price = analyst_intel["technical_analysis"]["current_price"]
        
        if pos['direction'] == "LONG":
            if price >= pos['take_profit']: return {"reason": "Take Profit Hit"}
            if price <= pos['stop_loss']: return {"reason": "Stop Loss Hit"}
        elif pos['direction'] == "SHORT":
            if price <= pos['take_profit']: return {"reason": "Take Profit Hit"}
            if price >= pos['stop_loss']: return {"reason": "Stop Loss Hit"}

        confidence = analyst_intel['directional_confidence_score']
        conf_rev_thresh = self.config.get("confidence_reversal_threshold", 0.6)
        if (pos['direction'] == "LONG" and confidence < 0.5 and (1 - confidence) >= conf_rev_thresh) or \
           (pos['direction'] == "SHORT" and confidence > 0.5 and confidence >= conf_rev_thresh):
            return {"reason": f"Directional confidence reversal (score: {confidence:.2f})"}

        return None

    def _check_entry_conditions(self, strat_params: Dict, analyst_intel: Dict) -> Dict | None:
        """Checks rules for opening a new position."""
        position_bias = strat_params.get("positional_bias", "NEUTRAL")
        confidence = analyst_intel.get("directional_confidence_score", 0.5)
        lss = analyst_intel.get("liquidation_safety_score", 0)
        min_confidence = self.config.get("min_confidence_for_entry", 0.65)
        min_lss = self.config.get("min_lss_for_entry", 60)
        
        if position_bias == "NEUTRAL" or confidence < min_confidence or lss < min_lss:
            return None

        tech_analysis = analyst_intel.get("technical_analysis", {})
        current_price = tech_analysis.get("current_price")
        atr = tech_analysis.get("atr")
        if not current_price or not atr: return None

        leverage = self._determine_leverage(lss, strat_params["max_allowable_leverage"])
        sl_multiplier = self.config.get("atr_stop_loss_multiplier", 2.0)
        
        if position_bias == "LONG":
            stop_loss = current_price - (atr * sl_multiplier)
        else: # SHORT
            stop_loss = current_price + (atr * sl_multiplier)

        quantity = self._calculate_position_size(current_price, stop_loss, leverage)
        if quantity <= 0:
            self.logger.warning("Position size calculated to be zero or less. No trade placed.")
            return None

        tp_multiplier = self.config.get("atr_take_profit_multiplier", 3.0)
        if position_bias == "LONG":
            take_profit = current_price + (atr * tp_multiplier)
        else: # SHORT
            take_profit = current_price - (atr * tp_multiplier)

        return {
            "direction": position_bias, "quantity": quantity, "leverage": leverage,
            "entry_price": current_price, "stop_loss": stop_loss, "take_profit": take_profit,
        }

    def _calculate_position_size(self, current_price: float, stop_loss_price: float, leverage: int) -> float:
        """Calculates position size based on risk per trade and stop loss distance."""
        capital = self.state_manager.get_state("account_equity", settings.get("initial_equity", 10000))
        risk_per_trade_pct = self.config.get("risk_per_trade_pct", 0.01) # 1% risk
        
        if current_price == stop_loss_price: return 0.0

        max_risk_usd = capital * risk_per_trade_pct
        stop_loss_distance = abs(current_price - stop_loss_price)
        if stop_loss_distance == 0: return 0.0

        units = max_risk_usd / stop_loss_distance
        notional_value = units * current_price
        required_margin = notional_value / leverage

        if required_margin > capital:
            self.logger.warning(f"Required margin ({required_margin:.2f}) exceeds capital ({capital:.2f}). Reducing size.")
            units = (capital * leverage) / current_price
        
        return round(units, 3) # Return rounded to 3 decimal places for crypto

    def _determine_leverage(self, lss: float, max_leverage_cap: int) -> int:
        """Determines leverage based on Liquidation Safety Score (LSS)."""
        initial_leverage = self.config.get("initial_leverage", 25)
        if lss <= 50:
            scaled_leverage = initial_leverage
        else:
            # Scale leverage from initial_leverage up to max_leverage_cap as LSS goes from 50 to 100
            scaled_leverage = initial_leverage + ((lss - 50) / 50) * (max_leverage_cap - initial_leverage)
        
        return min(max(initial_leverage, int(scaled_leverage)), max_leverage_cap)

    async def execute_open_position(self, decision: Dict):
        """Executes the logic to open a new position."""
        self.logger.info(f"Executing OPEN for {decision['direction']} position: Qty={decision['quantity']:.3f}, SL={decision['stop_loss']:.2f}, TP={decision['take_profit']:.2f}")
        try:
            # Here you would set leverage on the exchange before placing the order
            # await self.exchange.set_leverage(self.trade_symbol, decision['leverage'])
            
            order = await self.exchange.create_order(
                symbol=self.trade_symbol,
                side="BUY" if decision['direction'] == "LONG" else "SELL",
                order_type="MARKET",
                quantity=decision['quantity']
            )
            self.logger.info(f"Entry order placed successfully: {order}")
            
            self.current_position = {
                "direction": decision["direction"],
                "size": float(order.get('executedQty', decision['quantity'])),
                "entry_price": float(order.get('avgPrice', decision['entry_price'])),
                "stop_loss": decision["stop_loss"],
                "take_profit": decision["take_profit"],
                "leverage": decision["leverage"],
                "entry_timestamp": time.time()
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
                symbol=self.trade_symbol,
                side="SELL" if self.current_position['direction'] == "LONG" else "BUY",
                order_type="MARKET",
                quantity=self.current_position['size']
            )
            self.logger.info(f"Close order placed successfully: {order}")
            
            self.current_position = self._get_default_position()
            self.state_manager.set_state("current_position", self.current_position)
            self.logger.info("Position closed and state has been reset.")

        except Exception as e:
            self.logger.error(f"Failed to execute close order: {e}", exc_info=True)

