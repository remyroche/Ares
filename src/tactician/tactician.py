import asyncio
import time
from typing import Dict, Any, Optional
import uuid # Import uuid for unique trade IDs
import datetime # Import datetime for timestamps

from src.exchange.binance import BinanceExchange
from src.utils.logger import logger
from src.config import settings, CONFIG # Import CONFIG for fees etc.
from src.utils.state_manager import StateManager
from src.supervisor.performance_reporter import PerformanceReporter # Import PerformanceReporter

class Tactician:
    """
    The Tactician translates the Analyst's rich intelligence into a high-level trading plan.
    It now uses detailed technical analysis (VWAP, MAs, etc.) to formulate its strategy.
    Also responsible for generating detailed trade logs.
    """

    def __init__(self, exchange_client: Optional[BinanceExchange] = None, 
                 state_manager: Optional[StateManager] = None,
                 performance_reporter: Optional[PerformanceReporter] = None): # Added performance_reporter
        
        self.exchange = exchange_client
        self.state_manager = state_manager
        self.performance_reporter = performance_reporter # Store reporter instance
        self.logger = logger.getChild('Tactician')
        self.config = settings.get("tactician", {})
        self.trade_symbol = settings.trade_symbol
        
        # Initialize current_position from state_manager if available, otherwise default
        self.current_position = self.state_manager.get_state("current_position", self._get_default_position()) if self.state_manager else self._get_default_position()
        self.logger.info(f"Tactician initialized. Position: {self.current_position.get('direction')}")
        self.last_analyst_timestamp = None


    def _get_default_position(self):
        """Returns the default structure for an empty position."""
        return {
            "direction": None, 
            "size": 0.0,
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "leverage": 1,
            "entry_timestamp": 0.0,
            "trade_id": None, # Add trade ID
            "entry_context": {} # Store all decision-making context at entry
        }

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
                await self.execute_close_position(exit_decision['reason'], analyst_intel) # Pass analyst_intel for exit price
                return
            self.logger.info("Holding open position. No exit conditions met.")
            return

        # 2. If no position is open, assess for a new entry
        entry_decision = self._prepare_trade_decision(analyst_intel)
        
        if entry_decision:
            await self.execute_open_position(entry_decision, analyst_intel) # Pass analyst_intel for entry context
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
        max_allowable_leverage = analyst_intel.get("strategist_params", {}).get("max_allowable_leverage", CONFIG['tactician']['initial_leverage'])
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

    async def execute_open_position(self, decision: Dict, analyst_intel: Dict): # Added analyst_intel
        """Executes the logic to open a new position based on the prepared decision."""
        self.logger.info(f"Executing OPEN for {decision['side'].upper()} signal '{decision['signal']}': Qty={decision['quantity']:.3f}, Type={decision['order_type']}")
        
        trade_id = str(uuid.uuid4()) # Generate unique trade ID
        entry_timestamp = time.time() # Capture entry timestamp
        
        try:
            order_response = await self.exchange.create_order(
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
            self.logger.info(f"Entry order placed successfully: {order_response}")
            
            # Extract actual executed price and quantity from order_response
            # This is a simplification; a real system would parse the full order response
            executed_qty = float(order_response.get('executedQty', decision['quantity']))
            avg_entry_price = float(order_response.get('avgPrice', decision['entry_price'] or analyst_intel['technical_signals']['current_price']))
            
            # Calculate entry fees (assuming taker fee for market orders, or maker for limit)
            # This needs to be refined based on actual order type and fill details
            entry_fees_usd = executed_qty * avg_entry_price * CONFIG['taker_fee'] # Assuming taker fee for simplicity

            # Store detailed entry context
            entry_context = {
                "MarketRegimeAtEntry": analyst_intel.get("market_regime"),
                "TacticianSignal": decision['signal'],
                "EnsemblePredictionAtEntry": analyst_intel.get("ensemble_prediction"),
                "EnsembleConfidenceAtEntry": analyst_intel.get("ensemble_confidence"),
                "BaseModelPredictionsAtEntry": analyst_intel.get("base_model_predictions", {}), # Added base model predictions
                "EnsembleWeightsAtEntry": analyst_intel.get("ensemble_weights", {}), # Added ensemble weights
                "DirectionalConfidenceAtEntry": analyst_intel.get("directional_confidence_score"),
                "MarketHealthScoreAtEntry": analyst_intel.get("market_health_score"),
                "LiquidationSafetyScoreAtEntry": analyst_intel.get("liquidation_risk_score"),
                "TrendStrengthAtEntry": analyst_intel.get("trend_strength"),
                "ADXValueAtEntry": analyst_intel.get("adx"),
                "RSIValueAtEntry": analyst_intel.get("technical_signals", {}).get("rsi"),
                "MACDHistogramValueAtEntry": analyst_intel.get("technical_signals", {}).get("macd", {}).get("histogram"),
                "PriceVsVWAPRatioAtEntry": analyst_intel.get("technical_signals", {}).get("price_to_vwap_ratio"),
                "VolumeDeltaAtEntry": analyst_intel.get("technical_signals", {}).get("volume_profile", {}).get("volume_delta"), # Assuming volume_delta is part of technical_signals
                "GlobalRiskMultiplierAtEntry": self.state_manager.get_state("global_risk_multiplier"),
                "AvailableAccountEquityAtEntry": self.state_manager.get_state("account_equity"),
                "TradingEnvironment": settings.trading_environment,
                "IsTradingPausedAtEntry": self.state_manager.get_state("is_trading_paused"),
                "KillSwitchActiveAtEntry": self.state_manager.is_kill_switch_active(),
                "ModelVersionID": self.state_manager.get_state("model_version_id", "champion") # Assuming model_version_id is set in state
            }

            # Update current position state with comprehensive details
            self.current_position = {
                "direction": "LONG" if decision['side'] == 'buy' else "SHORT",
                "size": executed_qty,
                "entry_price": avg_entry_price,
                "stop_loss": decision["stop_loss"],
                "take_profit": decision["take_profit"],
                "leverage": decision["leverage"],
                "entry_timestamp": entry_timestamp,
                "trade_id": trade_id,
                "entry_fees_usd": entry_fees_usd,
                "entry_context": entry_context # Store the collected context
            }
            self.state_manager.set_state("current_position", self.current_position)
            self.logger.info(f"New position state saved: {self.current_position}")

        except Exception as e:
            self.logger.error(f"Failed to execute trade entry: {e}", exc_info=True)
            # Reset current_position if entry failed to avoid stale state
            self.current_position = self._get_default_position()
            self.state_manager.set_state("current_position", self.current_position)

    def _check_exit_conditions(self, analyst_intel: Dict) -> Dict | None:
        pos = self.current_position
        price = analyst_intel["technical_signals"]["current_price"] # Corrected key
        if pos.get('direction') == "LONG":
            if price >= pos.get('take_profit', float('inf')): return {"reason": "Take Profit Hit", "exit_price": pos.get('take_profit')}
            if price <= pos.get('stop_loss', float('-inf')): return {"reason": "Stop Loss Hit", "exit_price": pos.get('stop_loss')}
        elif pos.get('direction') == "SHORT":
            if price <= pos.get('take_profit', float('-inf')): return {"reason": "Take Profit Hit", "exit_price": pos.get('take_profit')}
            if price >= pos.get('stop_loss', float('inf')): return {"reason": "Stop Loss Hit", "exit_price": pos.get('stop_loss')}
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

    async def execute_close_position(self, reason: str, analyst_intel: Dict): # Added analyst_intel
        """Executes the logic to close an open position and logs the trade."""
        
        # Retrieve current position details from state
        pos_details = self.current_position
        if not pos_details.get("trade_id"):
            self.logger.error("Attempted to close position but no trade_id found in current_position. Aborting log.")
            # Still attempt to close position on exchange
            return

        trade_id = pos_details["trade_id"]
        entry_timestamp = pos_details["entry_timestamp"]
        entry_price = pos_details["entry_price"]
        quantity = pos_details["size"]
        direction = pos_details["direction"]
        entry_fees_usd = pos_details.get("entry_fees_usd", 0.0)
        
        exit_timestamp = time.time() # Capture exit timestamp
        
        # Determine exit price based on reason or current market price
        exit_price = analyst_intel["technical_signals"]["current_price"] # Default to current price
        if reason in ["Take Profit Hit", "Stop Loss Hit"]:
            # If TP/SL hit, the exit_price should ideally be the TP/SL price
            # This is a simplification; in a real system, you'd get the actual fill price.
            if direction == "LONG" and reason == "Take Profit Hit":
                exit_price = pos_details["take_profit"]
            elif direction == "LONG" and reason == "Stop Loss Hit":
                exit_price = pos_details["stop_loss"]
            elif direction == "SHORT" and reason == "Take Profit Hit":
                exit_price = pos_details["take_profit"]
            elif direction == "SHORT" and reason == "Stop Loss Hit":
                exit_price = pos_details["stop_loss"]

        self.logger.warning(f"Executing CLOSE for {direction} position (ID: {trade_id}). Reason: {reason}. Exit Price: {exit_price:.2f}")
        
        try:
            order_response = await self.exchange.create_order(
                symbol=self.trade_symbol,
                side="SELL" if direction == "LONG" else "BUY",
                type="MARKET", # Always market order for closing
                quantity=quantity
            )
            self.logger.info(f"Close order placed successfully: {order_response}")
            
            # Recalculate actual exit price from order_response if available
            actual_exit_price = float(order_response.get('avgPrice', exit_price))
            exit_fees_usd = quantity * actual_exit_price * CONFIG['taker_fee'] # Assuming taker fee for simplicity

            # --- Calculate PnL and other metrics ---
            net_pnl_usd = 0.0
            pnl_percentage = 0.0
            if direction == "LONG":
                net_pnl_usd = (actual_exit_price - entry_price) * quantity - entry_fees_usd - exit_fees_usd
                pnl_percentage = (net_pnl_usd / (entry_price * quantity)) * 100 if (entry_price * quantity) != 0 else 0.0
            elif direction == "SHORT":
                net_pnl_usd = (entry_price - actual_exit_price) * quantity - entry_fees_usd - exit_fees_usd
                pnl_percentage = (net_pnl_usd / (entry_price * quantity)) * 100 if (entry_price * quantity) != 0 else 0.0

            trade_duration_seconds = exit_timestamp - entry_timestamp

            # --- Construct detailed trade log ---
            trade_log = {
                "TradeID": trade_id,
                "Token": self.trade_symbol.replace(settings.get("base_currency", "USDT"), ""), # Extract base token
                "Exchange": "Binance", # Hardcoded for now
                "Side": direction,
                "EntryTimestampUTC": datetime.datetime.fromtimestamp(entry_timestamp).isoformat(),
                "ExitTimestampUTC": datetime.datetime.fromtimestamp(exit_timestamp).isoformat(),
                "TradeDurationSeconds": trade_duration_seconds,
                "NetPnLUSD": net_pnl_usd,
                "PnLPercentage": pnl_percentage,
                "ExitReason": reason,
                "EntryPrice": entry_price,
                "ExitPrice": actual_exit_price,
                "QuantityBaseAsset": quantity,
                "NotionalSizeUSD": entry_price * quantity, # Notional at entry
                "LeverageUsed": pos_details.get("leverage"),
                "IntendedStopLossPrice": pos_details.get("stop_loss"),
                "IntendedTakeProfitPrice": pos_details.get("take_profit"),
                "ActualStopLossPrice": pos_details.get("stop_loss") if reason == "Stop Loss Hit" else None,
                "ActualTakeProfitPrice": pos_details.get("take_profit") if reason == "Take Profit Hit" else None,
                "OrderTypeEntry": pos_details.get("entry_context", {}).get("OrderTypeEntry"), # Assuming this was stored
                "OrderTypeExit": "MARKET", # Always market for closing
                "EntryFeesUSD": entry_fees_usd,
                "ExitFeesUSD": exit_fees_usd,
                "SlippageEntryPct": None, # Needs actual fill price vs intended price at entry
                "SlippageExitPct": None, # Needs actual fill price vs current market price at exit
                **pos_details.get("entry_context", {}) # Unpack all stored entry context
            }
            
            # Record the detailed trade log
            if self.performance_reporter:
                await self.performance_reporter.record_detailed_trade_log(trade_log)

            # Reset position after successful exit and logging
            self.current_position = self._get_default_position()
            self.state_manager.set_state("current_position", self.current_position)
            self.logger.info("Position closed and state has been reset.")

        except Exception as e:
            self.logger.error(f"Failed to execute close order or log trade: {e}", exc_info=True)
            # If close failed, don't reset position, allow supervisor to re-sync
            # or manual intervention.

    def _calculate_atr_stop_loss(self, side: str, candle: dict, multiplier: float = 1.5) -> float:
        atr = candle.get('ATR', 0)
        return candle['current_price'] - (atr * multiplier) if side == 'buy' else candle['current_price'] + (atr * multiplier)

    def _calculate_atr_take_profit(self, side: str, candle: dict, stop_loss: float, rr_ratio: float = 2.0) -> float:
        risk = abs(candle['current_price'] - stop_loss)
        return candle['current_price'] + (risk * rr_ratio) if side == 'buy' else candle['current_price'] - (risk * rr_ratio)
