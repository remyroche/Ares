import asyncio
import time
from typing import Dict, Any, Optional

from src.exchange.binance import BinanceExchange
from src.utils.logger import logger
from src.config import settings
from src.utils.state_manager import StateManager

class Tactician:
    """
    The Tactician translates the Analyst's rich intelligence into a high-level trading plan.
    It now uses detailed technical analysis (VWAP, MAs, etc.) to formulate its strategy.
    """

    def __init__(self, exchange_client: Optional[BinanceExchange] = None, state_manager: Optional[StateManager] = None):
        # Allow exchange_client and state_manager to be optional for ModelManager instantiation
        self.exchange = exchange_client
        self.state_manager = state_manager
        self.logger = logger.getChild('Tactician')
        self.config = settings.get("tactician", {})
        self.trade_symbol = settings.trade_symbol
        
        # Initialize current_position from state_manager if available, otherwise default
        self.current_position = self.state_manager.get_state("current_position", self._get_default_position()) if self.state_manager else self._get_default_position()
        self.logger.info(f"Tactician initialized. Position: {self.current_position.get('direction')}")
        self.last_analyst_timestamp = None


    def _get_default_position(self):
        """Returns the default structure for an empty position."""
        return {"direction": None, "size": 0.0}

    async def start(self):
        """Starts the main tactician loop."""
        if self.state_manager is None or self.exchange is None:
            self.logger.error("Tactician cannot start: StateManager or Exchange client not provided.")
            return

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
        signal = analyst_intel.get("ensemble_prediction", "HOLD") # Use ensemble_prediction
        confidence = analyst_intel.get("ensemble_confidence", 0.0) # Use ensemble_confidence
        technical_analysis_data = analyst_intel.get("technical_signals", {}) # Corrected key
        
        if not technical_analysis_data or 'current_price' not in technical_analysis_data or 'ATR' not in technical_analysis_data:
            self.logger.warning("Cannot prepare trade decision: current price or ATR data is missing from technical_signals.")
            return None

        min_confidence = self.config.get("min_confidence_for_entry", 0.65)
        if confidence < min_confidence:
            self.logger.info(f"Signal '{signal}' confidence ({confidence:.2f}) is below threshold ({min_confidence}). No action.")
            return None
            
        current_price = technical_analysis_data['current_price']
        current_atr = technical_analysis_data['ATR']

        # --- Strategy-Specific Execution Logic ---
        order_type = None; side = None; entry_price = None; stop_loss = None; take_profit = None

        if signal == "BUY":
            order_type = 'MARKET'; side = 'buy'
            stop_loss = self._calculate_atr_stop_loss(side, technical_analysis_data)
            take_profit = self._calculate_atr_take_profit(side, technical_analysis_data, stop_loss)
        elif signal == "SELL":
            order_type = 'MARKET'; side = 'sell'
            stop_loss = self._calculate_atr_stop_loss(side, technical_analysis_data)
            take_profit = self._calculate_atr_take_profit(side, technical_analysis_data, stop_loss)
        elif signal == "SR_FADE_LONG":
            order_type = 'LIMIT'; side = 'buy'
            entry_price = technical_analysis_data['low'] # Assuming 'low' is available from kline
            stop_loss = entry_price - current_atr
            take_profit = entry_price + (current_atr * self.config.get("sr_tp_multiplier", 2.0))
        elif signal == "SR_FADE_SHORT":
            order_type = 'LIMIT'; side = 'sell'
            entry_price = technical_analysis_data['high'] # Assuming 'high' is available from kline
            stop_loss = entry_price + current_atr
            take_profit = entry_price - (current_atr * self.config.get("sr_tp_multiplier", 2.0))
        elif signal == "SR_BREAKOUT_LONG":
            order_type = 'STOP_MARKET'; side = 'buy'
            entry_price = technical_analysis_data['high'] # Assuming 'high' is the breakout level
            stop_loss = technical_analysis_data['low'] # Stop below the breakout candle's low
            take_profit = entry_price + (current_atr * self.config.get("sr_tp_multiplier", 2.0))
        elif signal == "SR_BREAKOUT_SHORT":
            order_type = 'STOP_MARKET'; side = 'sell'
            entry_price = technical_analysis_data['low'] # Assuming 'low' is the breakout level
            stop_loss = technical_analysis_data['high'] # Stop above the breakout candle's high
            take_profit = entry_price - (current_atr * self.config.get("sr_tp_multiplier", 2.0))
        else:
            return None # No action for HOLD or unknown signals

        if order_type is None:
            return None # No valid signal processed

        # --- Risk & Sizing Calculation ---
        lss = analyst_intel.get("liquidation_risk_score", 0)
        max_allowable_leverage = analyst_intel.get("strategist_params", {}).get("max_allowable_leverage", settings.CONFIG['tactician']['initial_leverage'])
        leverage = self._determine_leverage(lss, max_allowable_leverage)
        
        # Use current_price for quantity calculation if entry_price is not set (e.g., for MARKET orders)
        price_for_sizing = entry_price if entry_price else current_price
        
        quantity = self._calculate_position_size(price_for_sizing, stop_loss, leverage)
        
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
                quantity=decision['quantity'],
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
                "entry_price": float(order.get('avgPrice', decision['entry_price'] or self.state_manager.get_state("analyst_intelligence")['technical_signals']['current_price'])),
                "stop_loss": decision["stop_loss"],
                "take_profit": decision["take_profit"],
                "leverage": decision["leverage"],
                "entry_timestamp": time.time()
            }
            self.state_manager.set_state("current_position", self.current_position)
            self.logger.info(f"New position state saved: {self.current_position}")

        except Exception as e:
            self.logger.error(f"Failed to execute trade entry: {e}", exc_info=True)

    def _check_exit_conditions(self, analyst_intel: Dict) -> Dict | None:
        pos = self.current_position
        price = analyst_intel["technical_signals"]["current_price"] # Corrected key
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
                quantity=self.current_position['size']
            )
            self.logger.info(f"Close order placed successfully: {order}")
            self.current_position = self._get_default_position()
            self.state_manager.set_state("current_position", self.current_position)
            self.logger.info("Position closed and state has been reset.")
        except Exception as e:
            self.logger.error(f"Failed to execute close order: {e}", exc_info=True)

    def _calculate_atr_stop_loss(self, side: str, candle: dict, multiplier: float = 1.5) -> float:
        atr = candle.get('ATR', 0)
        return candle['current_price'] - (atr * multiplier) if side == 'buy' else candle['current_price'] + (atr * multiplier)

    def _calculate_atr_take_profit(self, side: str, candle: dict, stop_loss: float, rr_ratio: float = 2.0) -> float:
        risk = abs(candle['current_price'] - stop_loss)
        return candle['current_price'] + (risk * rr_ratio) if side == 'buy' else candle['current_price'] - (risk * rr_ratio)
