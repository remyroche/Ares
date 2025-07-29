import asyncio
import time
from typing import Dict, Any

from src.exchange.binance import BinanceExchange
from src.utils.logger import logger
from src.config import settings
from src.utils.state_manager import StateManager

class Tactician:
    """
    ## CHANGE: Fully updated to integrate strategy-specific execution logic.
    ## This version replaces the broad regime-based assessment with a more precise
    ## signal-driven approach. It can now interpret and execute all specialized
    ## signals from the Analyst (e.g., 'SR_FADE_LONG', 'SR_BREAKOUT_SHORT') with
    ## the correct order types (MARKET, LIMIT, STOP_MARKET) and parameters.
    """

    def __init__(self, exchange_client: BinanceExchange, state_manager: StateManager):
        self.exchange = exchange_client
        self.state_manager = state_manager
        self.logger = logger.getChild('Tactician')
        self.config = settings.get("tactician", {})
        self.trade_symbol = settings.trade_symbol
        self.last_analyst_timestamp = None
        
        self.current_position = self.state_manager.get_state("current_position", self._get_default_position())
        self.logger.info(f"Tactician initialized. Position: {self.current_position.get('direction')}")

    def _get_default_position(self):
        """Returns the default structure for an empty position."""
        return {"direction": None, "size": 0.0}

    async def start(self):
        """Starts the main tactician loop."""
        self.logger.info("Tactician started. Waiting for new analyst intelligence...")
        while True:
            try:
                if self.state_manager.get_state("is_trading_paused", False):
                    await asyncio.sleep(10)
                    continue

                analyst_intelligence = self.state_manager.get_state("analyst_intelligence")
                
                if analyst_intelligence and analyst_intelligence.get("timestamp") != self.last_analyst_timestamp:
                    self.last_analyst_timestamp = analyst_intelligence.get("timestamp")
                    self.logger.info("New analyst intelligence detected. Running tactical assessment.")
                    
                    await self.run_tactical_assessment(analyst_intelligence)
                
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                self.logger.info("Tactician task cancelled.")
                break
            except Exception as e:
                self.logger.error(f"An error occurred in the Tactician loop: {e}", exc_info=True)
                await asyncio.sleep(10)

    async def run_tactical_assessment(self, analyst_intel: Dict):
        """
        Assesses the current market situation based on Analyst intelligence
        and decides whether to enter or exit a position.
        """
        self.logger.info("--- Starting Tactical Assessment ---")
        
        # 1. Check for exit conditions if a position is open
        if self.current_position.get("size", 0) > 0:
            exit_decision = self._check_exit_conditions(analyst_intel)
            if exit_decision:
                await self.execute_close_position(exit_decision['reason'])
                return
            self.logger.info("Holding open position. No exit conditions met.")
            return

        # 2. If no position is open, assess for a new entry
        entry_decision = self._prepare_trade_decision(analyst_intel)
        
        if entry_decision:
            await self.execute_open_position(entry_decision)
        else:
            self.logger.info("Holding. No entry conditions met.")

    def _prepare_trade_decision(self, analyst_intel: Dict) -> Dict | None:
        """
        Interprets the Analyst's signal and prepares a detailed trade execution plan.
        This is the core logic for translating a signal into actionable trade parameters.
        """
        signal = analyst_intel.get("prediction", "HOLD")
        confidence = analyst_intel.get("confidence", 0.0)
        current_candle = analyst_intel.get("technical_analysis", {})
        
        if not current_candle or 'ATR' not in current_candle:
            self.logger.warning("Cannot prepare trade decision: current candle or ATR data is missing.")
            return None

        min_confidence = self.config.get("min_confidence_for_entry", 0.65)
        if confidence < min_confidence:
            self.logger.info(f"Signal '{signal}' confidence ({confidence:.2f}) is below threshold ({min_confidence}). No action.")
            return None
            
        # --- Strategy-Specific Execution Logic ---
        match signal:
            case "BUY" | "SELL":
                order_type = 'MARKET'; side = 'buy' if signal == 'BUY' else 'sell'
                entry_price = None; stop_loss = self._calculate_atr_stop_loss(side, current_candle)
                take_profit = self._calculate_atr_take_profit(side, current_candle, stop_loss)
            case "SR_FADE_LONG":
                order_type = 'LIMIT'; side = 'buy'
                entry_price = current_candle['low']; stop_loss = entry_price - current_candle['ATR']
                take_profit = entry_price + (current_candle['ATR'] * self.config.get("sr_tp_multiplier", 2.0))
            case "SR_FADE_SHORT":
                order_type = 'LIMIT'; side = 'sell'
                entry_price = current_candle['high']; stop_loss = entry_price + current_candle['ATR']
                take_profit = entry_price - (current_candle['ATR'] * self.config.get("sr_tp_multiplier", 2.0))
            case "SR_BREAKOUT_LONG":
                order_type = 'STOP_MARKET'; side = 'buy'
                entry_price = current_candle['high']; stop_loss = current_candle['low']
                take_profit = entry_price + (current_candle['ATR'] * self.config.get("sr_tp_multiplier", 2.0))
            case "SR_BREAKOUT_SHORT":
                order_type = 'STOP_MARKET'; side = 'sell'
                entry_price = current_candle['low']; stop_loss = current_candle['high']
                take_profit = entry_price - (current_candle['ATR'] * self.config.get("sr_tp_multiplier", 2.0))
            case _:
                return None # No action for HOLD or unknown signals

        # --- Risk & Sizing Calculation ---
        lss = analyst_intel.get("liquidation_safety_score", 0)
        leverage = self._determine_leverage(lss, self.config.get("max_allowable_leverage", 50))
        quantity = self._calculate_position_size(current_candle['close'], stop_loss, leverage)
        
        if quantity <= 0:
            self.logger.warning("Position size calculated to be zero or less. Aborting trade.")
            return None

        return {
            "signal": signal, "side": side, "order_type": order_type,
            "quantity": quantity, "leverage": leverage,
            "entry_price": entry_price, "stop_loss": stop_loss, "take_profit": take_profit,
        }

    async def execute_open_position(self, decision: Dict):
        """Executes the logic to open a new position based on the prepared decision."""
        self.logger.info(f"Executing OPEN for {decision['side'].upper()} signal '{decision['signal']}': Qty={decision['quantity']:.3f}, Type={decision['order_type']}")
        try:
            order = await self.exchange.create_order(
                symbol=self.trade_symbol,
                side=decision['side'],
                type=decision['order_type'],
                amount=decision['quantity'],
                price=decision['entry_price'], # Used for LIMIT orders
                params={
                    'stopPrice': decision['entry_price'], # Used for STOP_MARKET orders
                    'takeProfitPrice': decision['take_profit'],
                    'stopLossPrice': decision['stop_loss']
                }
            )
            self.logger.info(f"Entry order placed successfully: {order}")
            
            # Update state after order placement
            self.current_position = {
                "direction": "LONG" if decision['side'] == 'buy' else "SHORT",
                "size": float(order.get('executedQty', decision['quantity'])),
                "entry_price": float(order.get('avgPrice', decision['entry_price'] or self.state_manager.get_state("analyst_intelligence")['technical_analysis']['current_price'])),
                "stop_loss": decision["stop_loss"],
                "take_profit": decision["take_profit"],
                "leverage": decision["leverage"],
                "entry_timestamp": time.time()
            }
            self.state_manager.set_state("current_position", self.current_position)
            self.logger.info(f"New position state saved: {self.current_position}")

        except Exception as e:
            self.logger.error(f"Failed to execute trade entry: {e}", exc_info=True)

    # ... (other helper methods like _check_exit_conditions, _calculate_position_size, etc. remain the same)
    def _check_exit_conditions(self, analyst_intel: Dict) -> Dict | None:
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
        capital = self.state_manager.get_state("account_equity", settings.get("initial_equity", 10000))
        risk_per_trade_pct = self.config.get("risk_per_trade_pct", 0.01)
        risk_multiplier = self.state_manager.get_state("global_risk_multiplier", 1.0)
        max_risk_usd = capital * risk_per_trade_pct * risk_multiplier
        stop_loss_distance = abs(current_price - stop_loss_price)
        if stop_loss_distance == 0: return 0.0
        units = max_risk_usd / stop_loss_distance
        return round(units, 3)

    def _determine_leverage(self, lss: float, max_leverage_cap: int) -> int:
        initial_leverage = self.config.get("initial_leverage", 25)
        if lss <= 50: return initial_leverage
        scaled_leverage = initial_leverage + ((lss - 50) / 50) * (max_leverage_cap - initial_leverage)
        return min(max(initial_leverage, int(scaled_leverage)), max_leverage_cap)

    async def execute_close_position(self, reason: str):
        self.logger.warning(f"Executing CLOSE for {self.current_position['direction']} position. Reason: {reason}")
        try:
            order = await self.exchange.create_order(
                symbol=self.trade_symbol,
                side="SELL" if self.current_position['direction'] == "LONG" else "BUY",
                type="MARKET",
                amount=self.current_position['size']
            )
            self.logger.info(f"Close order placed successfully: {order}")
            self.current_position = self._get_default_position()
            self.state_manager.set_state("current_position", self.current_position)
            self.logger.info("Position closed and state has been reset.")
        except Exception as e:
            self.logger.error(f"Failed to execute close order: {e}", exc_info=True)

    def _calculate_atr_stop_loss(self, side: str, candle: dict, multiplier: float = 1.5) -> float:
        atr = candle.get('ATR', 0)
        return candle['close'] - (atr * multiplier) if side == 'buy' else candle['close'] + (atr * multiplier)

    def _calculate_atr_take_profit(self, side: str, candle: dict, stop_loss: float, rr_ratio: float = 2.0) -> float:
        risk = abs(candle['close'] - stop_loss)
        return candle['close'] + (risk * rr_ratio) if side == 'buy' else candle['close'] - (risk * rr_ratio)
