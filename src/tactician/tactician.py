import asyncio
import time
from typing import Dict, Any, Optional, Union # Ensure Union is imported
import uuid # Import uuid for unique trade IDs
import datetime # Import datetime for timestamps
import numpy as np # Import numpy for numerical operations

from src.exchange.binance import BinanceExchange
from src.utils.logger import system_logger as logger # Fixed: Changed import to system_logger
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
        # Ensure state_manager is not None before calling get_state
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
            "entry_fees_usd": 0.0, # Initialize entry fees
            "entry_context": {} # Store all decision-making context at entry
        }

    async def start(self):
        """Starts the main tactician loop."""
        # Fixed: Explicitly check if state_manager and exchange are not None
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
                # Fixed: Ensure exit_price is a float before passing
                exit_price_val = exit_decision.get('exit_price')
                if exit_price_val is not None:
                    await self.execute_close_position(exit_decision['reason'], analyst_intel, float(exit_price_val))
                else:
                    self.logger.warning(f"Exit decision made but no valid exit_price provided for reason: {exit_decision['reason']}. Closing at current market price.")
                    await self.execute_close_position(exit_decision['reason'], analyst_intel) # Close at current market price
                return
            self.logger.info("Holding open position. No exit conditions met.")
            return

        # 2. If no position is open, assess for a new entry
        entry_decision = self._prepare_trade_decision(analyst_intel)
        
        if entry_decision:
            await self.execute_open_position(entry_decision, analyst_intel) # Pass analyst_intel for entry context
        else:
            self.logger.info("Holding. No entry conditions met.")

    def _prepare_trade_decision(self, analyst_intel: Dict) -> Union[Dict, None]: # Fixed: Union syntax
        """
        Interprets the Analyst's signal and prepares a detailed trade execution plan.
        This is the core logic for translating a signal into actionable trade parameters.
        Includes error handling for missing data.
        """
        signal = analyst_intel.get("ensemble_prediction", "HOLD")
        confidence = analyst_intel.get("ensemble_confidence", 0.0)
        technical_analysis_data = analyst_intel.get("technical_signals", {})
        
        # Essential data checks
        current_price = technical_analysis_data.get('current_price')
        current_atr = technical_analysis_data.get('ATR')

        if current_price is None or current_atr is None or current_atr <= 0:
            self.logger.warning(f"Cannot prepare trade decision: current price ({current_price}) or ATR ({current_atr}) data is missing or invalid from technical_signals.")
            return None

        min_confidence = self.config.get("min_confidence_for_entry", 0.65)
        if confidence < min_confidence:
            self.logger.info(f"Signal '{signal}' confidence ({confidence:.2f}) is below threshold ({min_confidence}). No action.")
            return None
            
        # --- Strategy-Specific Execution Logic ---
        order_type: Optional[str] = None # Fixed: Explicitly type as Optional[str]
        side: Optional[str] = None # Fixed: Explicitly type as Optional[str]
        entry_price: Optional[float] = None # Fixed: Explicitly type as Optional[float]
        stop_loss: Optional[float] = None # Fixed: Explicitly type as Optional[float]
        take_profit: Optional[float] = None # Fixed: Explicitly type as Optional[float]

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
            entry_price = technical_analysis_data.get('low')
            if entry_price is None: self.logger.warning("Missing 'low' for SR_FADE_LONG entry."); return None
            stop_loss = entry_price - current_atr
            take_profit = entry_price + (current_atr * self.config.get("sr_tp_multiplier", 2.0))
        elif signal == "SR_FADE_SHORT":
            order_type = 'LIMIT'; side = 'sell'
            entry_price = technical_analysis_data.get('high')
            if entry_price is None: self.logger.warning("Missing 'high' for SR_FADE_SHORT entry."); return None
            stop_loss = entry_price + current_atr
            take_profit = entry_price - (current_atr * self.config.get("sr_tp_multiplier", 2.0))
        elif signal == "SR_BREAKOUT_LONG":
            order_type = 'STOP_MARKET'; side = 'buy'
            entry_price = technical_analysis_data.get('high')
            if entry_price is None: self.logger.warning("Missing 'high' for SR_BREAKOUT_LONG entry."); return None
            stop_loss = technical_analysis_data.get('low')
            if stop_loss is None: self.logger.warning("Missing 'low' for SR_BREAKOUT_LONG stop loss."); return None
            take_profit = entry_price + (current_atr * self.config.get("sr_tp_multiplier", 2.0))
        elif signal == "SR_BREAKOUT_SHORT":
            order_type = 'STOP_MARKET'; side = 'sell'
            entry_price = technical_analysis_data.get('low')
            if entry_price is None: self.logger.warning("Missing 'low' for SR_BREAKOUT_SHORT entry."); return None
            stop_loss = technical_analysis_data.get('high')
            if stop_loss is None: self.logger.warning("Missing 'high' for SR_BREAKOUT_SHORT stop loss."); return None
            take_profit = entry_price - (current_atr * self.config.get("sr_tp_multiplier", 2.0))
        else:
            return None # No action for HOLD or unknown signals

        if order_type is None:
            self.logger.warning(f"No valid order type determined for signal: {signal}.")
            return None

        # --- Risk & Sizing Calculation ---
        lss = analyst_intel.get("liquidation_risk_score", 0)
        max_allowable_leverage = analyst_intel.get("strategist_params", {}).get("max_allowable_leverage", CONFIG['tactician']['initial_leverage'])
        leverage = self._determine_leverage(lss, max_allowable_leverage)
        
        price_for_sizing = entry_price if entry_price is not None else current_price # Fixed: Check entry_price for None
        
        # Fixed: Ensure stop_loss is not None before passing to _calculate_position_size
        if stop_loss is None:
            self.logger.warning("Stop loss is None, cannot calculate position size. Aborting trade.")
            return None

        try:
            quantity = self._calculate_position_size(price_for_sizing, stop_loss, leverage)
        except ZeroDivisionError:
            self.logger.error("ZeroDivisionError in _calculate_position_size. Stop loss price might be too close to current price.")
            return None
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}", exc_info=True)
            return None

        if quantity <= 0:
            self.logger.warning("Position size calculated to be zero or less. Aborting trade.")
            return None

        return {
            "signal": signal, "side": side, "order_type": order_type,
            "quantity": quantity, "leverage": leverage,
            "entry_price": entry_price, "stop_loss": stop_loss, "take_profit": take_profit,
        }

    async def execute_open_position(self, decision: Dict, analyst_intel: Dict):
        """Executes the logic to open a new position based on the prepared decision."""
        self.logger.info(f"Executing OPEN for {decision['side'].upper()} signal '{decision['signal']}': Qty={decision['quantity']:.3f}, Type={decision['order_type']}")
        
        trade_id = str(uuid.uuid4()) # Generate unique trade ID
        entry_timestamp = time.time() # Capture entry timestamp
        
        try:
            # Fixed: Ensure self.exchange is not None before calling create_order
            if self.exchange is None:
                self.logger.error("Exchange client is None. Cannot execute open position.")
                raise RuntimeError("Exchange client not initialized.")

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
            
            if order_response and order_response.get('status') == 'failed':
                self.logger.error(f"Order creation failed: {order_response.get('error')}")
                raise Exception(f"Order creation failed: {order_response.get('error')}")

            self.logger.info(f"Entry order placed successfully: {order_response}")
            
            executed_qty = float(order_response.get('executedQty', decision['quantity']))
            avg_entry_price = float(order_response.get('avgPrice', decision['entry_price'] or analyst_intel['current_price']))
            
            entry_fees_usd = executed_qty * avg_entry_price * CONFIG['taker_fee']

            entry_context = {
                "MarketRegimeAtEntry": analyst_intel.get("market_regime"),
                "TacticianSignal": decision['signal'],
                "EnsemblePredictionAtEntry": analyst_intel.get("ensemble_prediction"),
                "EnsembleConfidenceAtEntry": analyst_intel.get("ensemble_confidence"),
                "BaseModelPredictionsAtEntry": analyst_intel.get("base_model_predictions", {}),
                "EnsembleWeightsAtEntry": analyst_intel.get("ensemble_weights", {}),
                "DirectionalConfidenceAtEntry": analyst_intel.get("directional_confidence_score"),
                "MarketHealthScoreAtEntry": analyst_intel.get("market_health_score"),
                "LiquidationSafetyScoreAtEntry": analyst_intel.get("liquidation_risk_score"),
                "TrendStrengthAtEntry": analyst_intel.get("trend_strength"),
                "ADXValueAtEntry": analyst_intel.get("adx"),
                "RSIValueAtEntry": analyst_intel.get("technical_signals", {}).get("rsi"), # Revert to technical_signals path
                "MACDHistogramValueAtEntry": analyst_intel.get("technical_signals", {}).get("macd", {}).get("histogram"), # Revert
                "PriceVsVWAPRatioAtEntry": analyst_intel.get("technical_signals", {}).get("price_to_vwap_ratio"), # Revert
                "VolumeDeltaAtEntry": analyst_intel.get("volume_delta"), # This is now top-level in analyst_intel
                "GlobalRiskMultiplierAtEntry": self.state_manager.get_state("global_risk_multiplier") if self.state_manager else None, # Fixed: Check state_manager
                "AvailableAccountEquityAtEntry": self.state_manager.get_state("account_equity") if self.state_manager else None, # Fixed: Check state_manager
                "TradingEnvironment": settings.trading_environment,
                "IsTradingPausedAtEntry": self.state_manager.get_state("is_trading_paused") if self.state_manager else None, # Fixed: Check state_manager
                "KillSwitchActiveAtEntry": self.state_manager.is_kill_switch_active() if self.state_manager else None, # Fixed: Check state_manager
                "ModelVersionID": self.state_manager.get_state("model_version_id", "champion") if self.state_manager else None # Fixed: Check state_manager
            }

            # Fixed: Ensure self.state_manager is not None before setting state
            if self.state_manager:
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
                    "entry_context": entry_context
                }
                self.state_manager.set_state("current_position", self.current_position)
                self.logger.info(f"New position state saved: {self.current_position}")
            else:
                self.logger.error("State manager is None. Cannot save current position state.")


        except Exception as e:
            self.logger.error(f"Failed to execute trade entry or update state: {e}", exc_info=True)
            # Fixed: Ensure self.state_manager is not None before setting state
            if self.state_manager:
                self.current_position = self._get_default_position()
                self.state_manager.set_state("current_position", self.current_position)

    def _check_exit_conditions(self, analyst_intel: Dict) -> Union[Dict, None]: # Fixed: Union syntax
        pos = self.current_position
        current_price = analyst_intel.get("current_price") # Using top-level current_price from analyst_intel
        if current_price is None:
            self.logger.warning("Current price not available for exit condition check.")
            return None

        if pos.get('direction') == "LONG":
            if current_price >= pos.get('take_profit', float('inf')): return {"reason": "Take Profit Hit", "exit_price": pos.get('take_profit')}
            if current_price <= pos.get('stop_loss', float('-inf')): return {"reason": "Stop Loss Hit", "exit_price": pos.get('stop_loss')}
        elif pos.get('direction') == "SHORT":
            if current_price <= pos.get('take_profit', float('-inf')): return {"reason": "Take Profit Hit", "exit_price": pos.get('take_profit')}
            if current_price >= pos.get('stop_loss', float('inf')): return {"reason": "Stop Loss Hit", "exit_price": pos.get('stop_loss')}
        return None

    def _calculate_position_size(self, current_price: float, stop_loss_price: float, leverage: int) -> float:
        # Fixed: Ensure self.state_manager is not None before calling get_state
        capital = self.state_manager.get_state("account_equity", settings.get("initial_equity", 10000)) if self.state_manager else settings.get("initial_equity", 10000)
        risk_per_trade_pct = self.config.get("risk_per_trade_pct", 0.01)
        risk_multiplier = self.state_manager.get_state("global_risk_multiplier", 1.0) if self.state_manager else 1.0
        max_risk_usd = capital * risk_per_trade_pct * risk_multiplier
        
        # Handle potential division by zero if stop_loss_price is too close to current_price
        stop_loss_distance = abs(current_price - stop_loss_price)
        if stop_loss_distance == 0: 
            self.logger.warning("Stop loss distance is zero, cannot calculate position size. Returning 0.")
            return 0.0
        
        units = max_risk_usd / stop_loss_distance
        return round(units, 3)

    def _determine_leverage(self, lss: float, max_leverage_cap: int) -> int:
        initial_leverage = self.config.get("initial_leverage", 25)
        if lss <= 50: return initial_leverage
        scaled_leverage = initial_leverage + ((lss - 50) / 50) * (max_leverage_cap - initial_leverage)
        return min(max(initial_leverage, int(scaled_leverage)), max_leverage_cap)

    async def execute_close_position(self, reason: str, analyst_intel: Dict, exit_price_override: Optional[float] = None):
        """Executes the logic to close an open position and logs the trade."""
        
        pos_details = self.current_position
        if not pos_details.get("trade_id"):
            self.logger.error("Attempted to close position but no trade_id found in current_position. Aborting log.")
            return # Don't proceed with logging if essential data is missing

        trade_id = pos_details["trade_id"]
        entry_timestamp = pos_details["entry_timestamp"]
        entry_price = pos_details["entry_price"]
        quantity = pos_details["size"]
        direction = pos_details["direction"]
        entry_fees_usd = pos_details.get("entry_fees_usd", 0.0)
        
        exit_timestamp = time.time()
        
        exit_price = exit_price_override if exit_price_override is not None else analyst_intel.get("current_price")
        
        if exit_price is None:
            self.logger.error(f"Exit price not available for trade {trade_id}. Cannot close position or log accurately.")
            return # Cannot proceed without exit price

        self.logger.warning(f"Executing CLOSE for {direction} position (ID: {trade_id}). Reason: {reason}. Exit Price: {exit_price:.2f}")
        
        try:
            # Fixed: Ensure self.exchange is not None before calling create_order
            if self.exchange is None:
                self.logger.error("Exchange client is None. Cannot execute close position.")
                raise RuntimeError("Exchange client not initialized.")

            order_response = await self.exchange.create_order(
                symbol=self.trade_symbol,
                side="SELL" if direction == "LONG" else "BUY",
                type="MARKET",
                quantity=quantity
            )
            
            if order_response and order_response.get('status') == 'failed':
                self.logger.error(f"Closing order failed: {order_response.get('error')}")
                raise Exception(f"Closing order failed: {order_response.get('error')}")

            self.logger.info(f"Close order placed successfully: {order_response}")
            
            actual_exit_price = float(order_response.get('avgPrice', exit_price))
            exit_fees_usd = quantity * actual_exit_price * CONFIG['taker_fee']

            net_pnl_usd = 0.0
            pnl_percentage = 0.0
            if direction == "LONG":
                net_pnl_usd = (actual_exit_price - entry_price) * quantity - entry_fees_usd - exit_fees_usd
                pnl_percentage = (net_pnl_usd / (entry_price * quantity)) * 100 if (entry_price * quantity) != 0 else 0.0
            elif direction == "SHORT":
                net_pnl_usd = (entry_price - actual_exit_price) * quantity - entry_fees_usd - exit_fees_usd
                pnl_percentage = (net_pnl_usd / (entry_price * quantity)) * 100 if (entry_price * quantity) != 0 else 0.0

            trade_duration_seconds = exit_timestamp - entry_timestamp

            trade_log = {
                "TradeID": trade_id,
                "Token": self.trade_symbol.replace(settings.get("base_currency", "USDT"), ""),
                "Exchange": "Binance",
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
                "NotionalSizeUSD": entry_price * quantity,
                "LeverageUsed": pos_details.get("leverage"),
                "IntendedStopLossPrice": pos_details.get("stop_loss"),
                "IntendedTakeProfitPrice": pos_details.get("take_profit"),
                "ActualStopLossPrice": pos_details.get("stop_loss") if reason == "Stop Loss Hit" else None,
                "ActualTakeProfitPrice": pos_details.get("take_profit") if reason == "Take Profit Hit" else None,
                "OrderTypeEntry": pos_details.get("entry_context", {}).get("OrderTypeEntry"),
                "OrderTypeExit": "MARKET",
                "EntryFeesUSD": entry_fees_usd,
                "ExitFeesUSD": exit_fees_usd,
                "SlippageEntryPct": None,
                "SlippageExitPct": None,
                **pos_details.get("entry_context", {})
            }
            
            # Fixed: Ensure self.performance_reporter is not None before calling record_detailed_trade_log
            if self.performance_reporter:
                await self.performance_reporter.record_detailed_trade_log(trade_log)
            else:
                self.logger.warning("Performance reporter is None. Cannot record detailed trade log.")

            # Fixed: Ensure self.state_manager is not None before setting state
            if self.state_manager:
                self.current_position = self._get_default_position()
                self.state_manager.set_state("current_position", self.current_position)
                self.logger.info("Position closed and state has been reset.")
            else:
                self.logger.error("State manager is None. Cannot reset position state.")

        except Exception as e:
            self.logger.error(f"Failed to execute close order or log trade: {e}", exc_info=True)
            # If close failed, don't reset position, allow supervisor to re-sync
            # or manual intervention.

    def _calculate_atr_stop_loss(self, side: str, candle: Dict[str, Any], multiplier: float = 1.5) -> float: # Fixed: Type hint for candle
        atr = candle.get('ATR', 0)
        return candle['current_price'] - (atr * multiplier) if side == 'buy' else candle['current_price'] + (atr * multiplier)

    def _calculate_atr_take_profit(self, side: str, candle: Dict[str, Any], stop_loss: float, rr_ratio: float = 2.0) -> float: # Fixed: Type hint for candle
        risk = abs(candle['current_price'] - stop_loss)
        return candle['current_price'] + (risk * rr_ratio) if side == 'buy' else candle['current_price'] - (risk * rr_ratio)
