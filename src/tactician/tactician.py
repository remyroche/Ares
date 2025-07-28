# src/supervisor/monitoring.py

import json
import logging
import time
from datetime import datetime

from google.cloud.firestore import Client

from src.database.firestore_manager import FirestoreManager

logger = logging.getLogger(__name__)


class Monitoring:
    def __init__(self, firestore_manager: FirestoreManager, log_file="monitoring_log.json"):
        self.firestore_manager = firestore_manager
        self.log_file = log_file
        self.start_time = time.time()

    def record_heartbeat(self):
        """Records a heartbeat to indicate the bot is alive and running."""
        uptime = time.time() - self.start_time
        heartbeat_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "alive",
            "uptime_seconds": uptime,
        }
        self._log_to_file(heartbeat_data)
        self.firestore_manager.db.collection("monitoring").document("heartbeat").set(heartbeat_data)
        logger.info("Heartbeat recorded.")

    def record_trade(self, trade_data: dict):
        """Records the details of a trade."""
        self._log_to_file(trade_data)
        self.firestore_manager.db.collection("trades").add(trade_data)
        logger.info(f"Trade recorded: {trade_data}")

    def record_error(self, error_message: str):
        """Records an error."""
        error_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "error": error_message,
        }
        self._log_to_file(error_data)
        self.firestore_manager.db.collection("errors").add(error_data)
        logger.error(f"Error recorded: {error_message}")

    def record_performance_metrics(self, metrics: dict):
        """Records performance metrics."""
        performance_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
        }
        self._log_to_file(performance_data)
        self.firestore_manager.db.collection("performance").add(performance_data)
        logger.info(f"Performance metrics recorded: {metrics}")

    def _log_to_file(self, data: dict):
        """Logs data to a local JSON file."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(data) + "\n")

```python
# src/supervisor/main.py

import asyncio
import logging
import time

# The user-provided code had several imports for components not defined in the current context
# (Sentinel, Analyst, Strategist, Tactician, PaperTrader, StateManager, db_manager).
# We are assuming these exist in the user's project structure.
from src.config import Config
from src.database.firestore_manager import FirestoreManager, db_manager
from src.supervisor.ab_tester import ABTester
from src.supervisor.performance_reporter import PerformanceReporter
from src.supervisor.risk_allocator import RiskAllocator
from src.supervisor.monitoring import Monitoring
from src.sentinel.sentinel import Sentinel
from src.analyst.analyst import Analyst
from src.strategist.strategist import Strategist
from src.tactician.tactician import Tactician
from src.paper_trader import PaperTrader
from src.utils.state_manager import StateManager


logger = logging.getLogger(__name__)


class Supervisor:
    """
    The central, real-time orchestrator of the Ares Trading Bot.
    It initializes, manages, and connects all the core components of the
    trading pipeline, ensuring they run concurrently and communicate efficiently.
    """
    def __init__(self):
        self.logger = logger
        self.state_manager = StateManager('ares_state.json')
        self.state = self.state_manager.load_state()
        self.config = Config()
        self.firestore_manager = FirestoreManager(self.config)
        self.risk_allocator = RiskAllocator(self.config, self.firestore_manager)
        self.performance_reporter = PerformanceReporter(self.firestore_manager)
        self.ab_tester = ABTester(self.firestore_manager)
        self.monitoring = Monitoring(self.firestore_manager)
        
        # Initialize the core real-time components
        self.sentinel = Sentinel()
        self.analyst = Analyst()
        self.strategist = Strategist(self.state)
        
        # Initialize the trader based on the configured mode
        if self.config.PAPER_TRADING:
            self.trader = PaperTrader(self.state)
            self.logger.info("Paper Trader initialized.")
        else:
            self.trader = None # Live trading not implemented
            self.logger.error("Live trading mode is not fully implemented yet.")
            # In a real scenario, you might want to raise an exception or handle this differently
            # raise NotImplementedError("Live trading not implemented.")

        if self.trader:
            # The new Tactician needs more than just the trader.
            # This part needs to be adapted to the new Tactician's signature.
            self.tactician = Tactician(state=self.state, config=self.config, firestore_manager=self.firestore_manager, trader=self.trader)
        else:
            self.tactician = None
        
        self.running = False
        
        # Asynchronous queues for communication between components
        self.market_data_queue = asyncio.Queue(maxsize=100)
        self.analysis_queue = asyncio.Queue(maxsize=100)
        self.signal_queue = asyncio.Queue(maxsize=50)
        self.order_queue = asyncio.Queue(maxsize=50) # Queue for Tactician's final orders

    async def start(self):
        """Starts all bot components and the main processing loop."""
        self.logger.info("Supervisor starting all components...")
        self.running = True

        # Initialize the database manager asynchronously if it has an init method
        if hasattr(db_manager, 'initialize') and asyncio.iscoroutinefunction(db_manager.initialize):
            await db_manager.initialize()

        # Create a list of all concurrent tasks to run
        tasks = [
            asyncio.create_task(self.sentinel.run(self.market_data_queue)),
            asyncio.create_task(self.analyst.run(self.market_data_queue, self.analysis_queue)),
            asyncio.create_task(self.strategist.run(self.analysis_queue, self.signal_queue)),
        ]
        
        if self.tactician:
             # The tactician now produces orders, another component would execute them.
             tasks.append(asyncio.create_task(self.tactician.run(self.signal_queue, self.order_queue)))

        # A new component would be needed to consume from order_queue and execute trades
        # For example: tasks.append(asyncio.create_task(self.trader.run(self.order_queue)))

        # If in paper trading mode, also run the simulation engine
        if isinstance(self.trader, PaperTrader):
            tasks.append(asyncio.create_task(self.trader.run_simulation()))

        try:
            # Run all component tasks concurrently
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.info("Supervisor tasks cancelled. Beginning graceful shutdown...")
        finally:
            self.running = False
            # Ensure all tasks are properly cancelled on exit
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait for all tasks to acknowledge cancellation
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Save the final state of the bot
            self.state_manager.save_state(self.state)
            self.logger.info("All components have been shut down and state has been saved.")
```python
# exchange/binance.py

import time
import logging
from functools import wraps
import asyncio

from binance.client import Client
from binance import AsyncClient
from binance.exceptions import BinanceAPIException, BinanceRequestException

from src.config import Config

logger = logging.getLogger(__name__)


def handle_binance_errors(func):
    """A decorator to handle Binance API errors with retries and exponential backoff."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        # The first argument is the class instance 'self'
        self_instance = args[0]
        retries = self_instance.config.BINANCE_API_RETRIES
        delay = 1
        for i in range(retries):
            try:
                return await func(*args, **kwargs)
            except (BinanceAPIException, BinanceRequestException) as e:
                if e.status_code == 429:  # Rate limit exceeded
                    logger.warning(f"Rate limit exceeded. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    delay *= 2
                elif e.status_code >= 500:  # Server-side error
                    logger.warning(f"Binance server error. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"Binance API error: {e}")
                    raise
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
                raise
        raise Exception("Failed to execute Binance API call after several retries.")

    return wrapper


class BinanceExchange:
    def __init__(self, config: Config):
        self.config = config
        self.client = None

    async def initialize(self):
        """Asynchronously initialize the Binance client."""
        if self.config.PAPER_TRADING:
            self.client = await AsyncClient.create(self.config.BINANCE_TESTNET_API_KEY, self.config.BINANCE_TESTNET_API_SECRET, tld='com', testnet=True)
        else:
            self.client = await AsyncClient.create(self.config.BINANCE_API_KEY, self.config.BINANCE_API_SECRET)
        logger.info("Binance AsyncClient initialized.")

    @handle_binance_errors
    async def get_account_balance(self):
        return await self.client.get_account()

    @handle_binance_errors
    async def get_klines(self, symbol, interval, limit):
        return await self.client.get_klines(symbol=symbol, interval=interval, limit=limit)

    @handle_binance_errors
    async def create_order(self, symbol, side, type, quantity, price=None):
        params = {
            "symbol": symbol,
            "side": side,
            "type": type,
            "quantity": quantity,
        }
        if price:
            params["price"] = price
            params["timeInForce"] = "GTC"

        return await self.client.create_order(**params)

    @handle_binance_errors
    async def get_order_status(self, symbol, order_id):
        return await self.client.get_order(symbol=symbol, orderId=order_id)

    async def handle_partial_fill(self, order):
        """Handles partially filled orders by creating a new order for the remaining amount."""
        if order['status'] == 'PARTIALLY_FILLED':
            logger.info(f"Order {order['orderId']} is partially filled. Executed quantity: {order['executedQty']}")
            remaining_qty = float(order['origQty']) - float(order['executedQty'])
            logger.info(f"Creating a new order for the remaining quantity: {remaining_qty}")
            await self.create_order(
                symbol=order['symbol'],
                side=order['side'],
                type=order['type'],
                quantity=remaining_qty,
                price=order['price']
            )
    
    async def close(self):
        if self.client:
            await self.client.close_connection()
            logger.info("Binance AsyncClient connection closed.")

```python
# backtesting/ares_backtester.py

import pandas as pd


class AresBacktester:
    def __init__(self, strategy, data, initial_capital=10000, fee=0.001, slippage=0.0005):
        self.strategy = strategy
        self.data = data
        self.initial_capital = initial_capital
        self.fee = fee
        self.slippage = slippage
        self.positions = pd.DataFrame(index=data.index).fillna(0.0)
        self.portfolio = pd.DataFrame(index=data.index).fillna(0.0)

    def run(self):
        self.portfolio['holdings'] = 0.0
        self.portfolio['cash'] = self.initial_capital
        self.portfolio['total'] = self.initial_capital
        
        # This part of the logic requires a 'signal' variable which is not defined
        # in the current context. Assuming it comes from the strategy.
        # signals = self.strategy.generate_signals(self.data) # Example
        
        # for i in range(len(self.data)):
        #     signal = signals[i] # Example
        #     # Apply fees and slippage to trades
        #     if signal == 1:  # Buy
        #         # Slippage
        #         buy_price = self.data['close'][i] * (1 + self.slippage)
        #         # Fee
        #         cost = buy_price * 100 * (1 + self.fee)
        #         self.portfolio.loc[self.data.index[i], 'cash'] -= cost
        #         self.portfolio.loc[self.data.index[i], 'holdings'] += 100 * buy_price
        #     elif signal == -1:  # Sell
        #         # Slippage
        #         sell_price = self.data['close'][i] * (1 - self.slippage)
        #         # Fee
        #         revenue = sell_price * 100 * (1 - self.fee)
        #         self.portfolio.loc[self.data.index[i], 'cash'] += revenue
        #         self.portfolio.loc[self.data.index[i], 'holdings'] -= 100 * sell_price

        #     # ... (rest of the logic)

        return self.portfolio

```python
# src/config.py

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # General
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    SUPERVISOR_SLEEP_INTERVAL = int(os.getenv("SUPERVISOR_SLEEP_INTERVAL", 60))
    SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
    INITIAL_EQUITY = float(os.getenv("INITIAL_EQUITY", 10000.0))

    # Binance
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
    BINANCE_TESTNET_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY")
    BINANCE_TESTNET_API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET")
    BINANCE_API_RETRIES = int(os.getenv("BINANCE_API_RETRIES", 5))

    # Trading
    PAPER_TRADING = os.getenv("PAPER_TRADING", "False").lower() in ("true", "1", "t")
    RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", 0.01)) # 1% risk per trade

    # Firestore
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    FIRESTORE_PROJECT_ID = os.getenv("FIRESTORE_PROJECT_ID")

    # Tactician default parameters (can be overridden by Supervisor)
    BEST_PARAMS = {
        "trade_entry_threshold": 0.7,
        "sl_atr_multiplier": 1.5,
        "take_profit_rr": 2.0,
    }
    
    # General Trading default parameters
    GENERAL_TRADING = {
        "confidence_wrong_direction_thresholds": [0.5, 0.6]
    }

    # Tactician specific config
    TACTICIAN_CONFIG = {
        "laddering": {
            "initial_leverage": 25,
            "min_lss_for_ladder": 70,
            "min_confidence_for_ladder": 0.75,
            "ladder_step_leverage_increase": 5,
            "max_ladder_steps": 3,
        },
        "risk_management": {
            "min_lss_for_entry": 60,
        }
    }
    
    # Strategist specific config
    STRATEGIST_CONFIG = {
        "max_leverage_cap_default": 100
    }


```python
# src/tactician/tactician.py

import logging
import asyncio

from src.config import Config
from src.database.firestore_manager import FirestoreManager

logger = logging.getLogger(__name__)


class Tactician:
    """
    The Tactician module, the "brain" of the system, deciding when, where, and how to place each order.
    It uses a sophisticated rule-based system, with parameters optimized by the Supervisor,
    to manage individual orders and implement dynamic laddering.
    """

    def __init__(self, state: dict, config: Config, firestore_manager: FirestoreManager, trader):
        self.state = state
        self.config = config
        self.firestore_manager = firestore_manager
        self.trader = trader # The trader object (Paper or Live) for executing orders
        self.logger = logging.getLogger(__name__)
        self.current_position = self.state.get('current_position', self._get_default_position())

    def _get_default_position(self):
        """Returns the default structure for an empty position."""
        return {
            "symbol": None, "direction": None, "size": 0.0, "entry_price": 0.0,
            "unrealized_pnl": 0.0, "current_leverage": 0, "ladder_steps": 0,
            "stop_loss": None, "take_profit": None, "liquidation_price": 0.0,
            "entry_confidence": 0.0, "entry_lss": 0.0
        }

    async def run(self, signal_queue: asyncio.Queue, order_queue: asyncio.Queue):
        """The main loop for the Tactician to process intelligence signals."""
        self.logger.info("Tactician is running and waiting for intelligence signals...")
        while True:
            try:
                # Wait for a combined intelligence package from the Strategist
                intelligence_package = await signal_queue.get()
                
                # Decide on an action based on the intelligence
                decision = self._determine_action(intelligence_package)
                
                # If a decision to act is made, put it on the order queue for execution
                if decision and decision.get("action") not in ["HOLD", "UNKNOWN"]:
                    self.logger.info(f"Tactician decided to act: {decision['action']}. Placing on order queue.")
                    await order_queue.put(decision)
                else:
                    self.logger.info(f"Tactician decided to HOLD. Reason: {decision.get('reason', 'N/A')}")
                
                signal_queue.task_done()
            except asyncio.CancelledError:
                self.logger.info("Tactician run loop cancelled.")
                break
            except Exception as e:
                self.logger.error(f"An error occurred in the Tactician run loop: {e}", exc_info=True)


    def _determine_action(self, state: dict):
        """
        Determines the trading action based on rules and current market intelligence.
        This is the main rule engine.
        """
        self.logger.info("Tactician: Determining action based on rule-based engine...")

        # If a position is open, first check for exit or management actions
        if self.current_position.get("size", 0) > 0:
            # Check for exit conditions (TP/SL, reversal, etc.)
            exit_decision = self._check_exit_conditions(state)
            if exit_decision:
                self._close_position(state['current_price'], exit_decision['reason'])
                return exit_decision

            # Check for laddering conditions (add to profitable position)
            ladder_decision = self._check_laddering_conditions(state)
            if ladder_decision:
                self._add_to_ladder(state['current_price'], ladder_decision)
                return ladder_decision
            
            return {"action": "HOLD", "reason": "Position open, no laddering or exit conditions met."}

        # If no position is open, check for entry conditions
        entry_decision = self._check_entry_conditions(state)
        if entry_decision:
            self._update_position(state['current_price'], entry_decision)
            return entry_decision

        return {"action": "HOLD", "reason": "No entry conditions met."}

    def _check_exit_conditions(self, state: dict) -> dict | None:
        """Checks all rules for exiting an existing position."""
        pos = self.current_position
        price = state['current_price']
        
        # Rule: Take Profit or Stop Loss Hit
        if pos['direction'] == "LONG":
            if price >= pos['take_profit']: return {"action": "CLOSE_POSITION", "reason": "Take Profit Hit."}
            if price <= pos['stop_loss']: return {"action": "CLOSE_POSITION", "reason": "Stop Loss Hit."}
        elif pos['direction'] == "SHORT":
            if price <= pos['take_profit']: return {"action": "CLOSE_POSITION", "reason": "Take Profit Hit."}
            if price >= pos['stop_loss']: return {"action": "CLOSE_POSITION", "reason": "Stop Loss Hit."}

        # Rule: Confidence Reversal
        conf_rev_thresh = self.config.GENERAL_TRADING["confidence_wrong_direction_thresholds"][1]
        if (pos['direction'] == "LONG" and state['directional_prediction'] == "SELL" and state['directional_confidence_score'] >= conf_rev_thresh) or \
           (pos['direction'] == "SHORT" and state['directional_prediction'] == "BUY" and state['directional_confidence_score'] >= conf_rev_thresh):
            return {"action": "CLOSE_POSITION", "reason": f"Directional reversal signal with confidence {state['directional_confidence_score']:.2f}."}

        # Rule: LSS Deterioration
        if state['liquidation_safety_score'] < pos['entry_lss'] * 0.8 and state['liquidation_safety_score'] < 50:
            return {"action": "CLOSE_POSITION", "reason": f"Liquidation Safety Score deteriorated to {state['liquidation_safety_score']:.2f}."}

        return None # No exit conditions met

    def _check_laddering_conditions(self, state: dict) -> dict | None:
        """Checks all rules for adding to an existing profitable position."""
        pos = self.current_position
        price = state['current_price']
        
        is_profitable = (pos["direction"] == "LONG" and price > pos["entry_price"]) or \
                        (pos["direction"] == "SHORT" and price < pos["entry_price"])
        
        if not is_profitable:
            return None

        # Laddering configuration
        ladder_cfg = self.config.TACTICIAN_CONFIG['laddering']
        max_steps = ladder_cfg['max_ladder_steps']
        min_conf = ladder_cfg['min_confidence_for_ladder']
        min_lss = ladder_cfg['min_lss_for_ladder']

        # Rule: Check if laddering is viable
        confidence_increased = state['directional_confidence_score'] > pos['entry_confidence'] + 0.1
        lss_increased = state['liquidation_safety_score'] > pos['entry_lss'] + 10
        
        can_ladder = (confidence_increased or state['directional_confidence_score'] >= min_conf) and \
                     (lss_increased or state['liquidation_safety_score'] >= min_lss) and \
                     (pos['ladder_steps'] < max_steps)

        if not can_ladder:
            return None
        
        # Calculate new order details
        new_leverage = min(pos["current_leverage"] + ladder_cfg['ladder_step_leverage_increase'], state['Max_Allowable_Leverage_Cap'])
        sl_atr_mult = self.config.BEST_PARAMS['sl_atr_multiplier']
        tp_rr = self.config.BEST_PARAMS['take_profit_rr']

        if pos["direction"] == "LONG":
            new_sl = price - (state['current_atr'] * sl_atr_mult)
            new_tp = price + (state['current_atr'] * sl_atr_mult * tp_rr)
        else: # SHORT
            new_sl = price + (state['current_atr'] * sl_atr_mult)
            new_tp = price - (state['current_atr'] * sl_atr_mult * tp_rr)
        
        additional_units, _ = self._calculate_position_size(state['current_equity'], price, new_sl, new_leverage)

        if additional_units > 0:
            return {
                "action": "ADD_TO_POSITION", "symbol": self.config.SYMBOL, "direction": pos["direction"],
                "order_type": "MARKET", "quantity": additional_units, "leverage": new_leverage,
                "stop_loss": new_sl, "take_profit": new_tp,
                "reason": f"Laddering up: Confidence {state['directional_confidence_score']:.2f}, LSS {state['liquidation_safety_score']:.2f}."
            }
        return None


    def _check_entry_conditions(self, state: dict) -> dict | None:
        """Checks all rules for opening a new position."""
        price = state['current_price']
        pred = state['directional_prediction']
        conf = state['directional_confidence_score']
        lss = state['liquidation_safety_score']
        
        # Rule: Confidence and LSS thresholds
        if conf < self.config.BEST_PARAMS['trade_entry_threshold'] or lss < self.config.TACTICIAN_CONFIG['risk_management']['min_lss_for_entry']:
            return None

        # Rule: Check against positional bias
        bias = state.get("Positional_Bias", "NEUTRAL")
        if (bias == "LONG" and pred == "SELL") or (bias == "SHORT" and pred == "BUY"):
            return None

        # Calculate order details
        leverage = self._determine_leverage(lss, state['Max_Allowable_Leverage_Cap'])
        sl_atr_mult = self.config.BEST_PARAMS['sl_atr_multiplier']
        tp_rr = self.config.BEST_PARAMS['take_profit_rr']

        if pred == "BUY":
            direction = "LONG"
            stop_loss = price - (state['current_atr'] * sl_atr_mult)
            take_profit = price + (state['current_atr'] * sl_atr_mult * tp_rr)
        elif pred == "SELL":
            direction = "SHORT"
            stop_loss = price + (state['current_atr'] * sl_atr_mult)
            take_profit = price - (state['current_atr'] * sl_atr_mult * tp_rr)
        else:
            return None # HOLD signal

        units, _ = self._calculate_position_size(state['current_equity'], price, stop_loss, leverage)

        if units > 0:
            return {
                "action": "PLACE_ORDER", "symbol": self.config.SYMBOL, "direction": direction,
                "order_type": "MARKET", "quantity": units, "leverage": leverage,
                "stop_loss": stop_loss, "take_profit": take_profit,
                "reason": f"New entry signal: {direction} with confidence {conf:.2f} and LSS {lss:.2f}.",
                "entry_confidence": conf, "entry_lss": lss
            }
        return None

    def _calculate_position_size(self, capital: float, current_price: float, stop_loss_price: float, leverage: float):
        """Calculates position size based on risk per trade and stop loss distance."""
        if stop_loss_price is None or current_price == stop_loss_price:
            return 0.0, 0.0

        risk_per_trade_pct = self.config.RISK_PER_TRADE_PCT
        max_risk_usd = capital * risk_per_trade_pct
        stop_loss_distance = abs(current_price - stop_loss_price)

        if stop_loss_distance == 0: return 0.0, 0.0

        units = max_risk_usd / stop_loss_distance
        notional_value = units * current_price
        required_margin = notional_value / leverage

        if required_margin > capital:
            units = (capital * leverage) / current_price
            notional_value = units * current_price
            self.logger.info(f"Adjusted position size due to capital limits. New units: {units:.4f}")

        return units, notional_value

    def _determine_leverage(self, lss: float, max_leverage_cap: int):
        """Determines leverage based on Liquidation Safety Score (LSS)."""
        initial_leverage = self.config.TACTICIAN_CONFIG['laddering']['initial_leverage']
        if lss <= 50:
            scaled_leverage = initial_leverage
        else:
            scaled_leverage = initial_leverage + (lss - 50) / 50 * (max_leverage_cap - initial_leverage)
        
        return min(max(initial_leverage, int(scaled_leverage)), max_leverage_cap)

    def _update_position(self, entry_price, decision):
        """Updates the internal state for a new position."""
        self.current_position = {
            "symbol": decision["symbol"], "direction": decision["direction"], "size": decision["quantity"],
            "entry_price": entry_price, "current_leverage": decision["leverage"],
            "stop_loss": decision["stop_loss"], "take_profit": decision["take_profit"],
            "entry_confidence": decision["entry_confidence"], "entry_lss": decision["entry_lss"],
            "ladder_steps": 0, "unrealized_pnl": 0.0, "liquidation_price": 0.0
        }
        self.logger.info(f"Position opened: {self.current_position}")

    def _add_to_ladder(self, current_price, decision):
        """Updates the internal state when adding to a position."""
        pos = self.current_position
        additional_size = decision['quantity']
        total_notional_old = pos["size"] * pos["entry_price"]
        total_notional_new = additional_size * current_price
        new_total_size = pos["size"] + additional_size

        pos["entry_price"] = (total_notional_old + total_notional_new) / new_total_size
        pos["size"] = new_total_size
        pos["current_leverage"] = decision["leverage"]
        pos["stop_loss"] = decision["stop_loss"]
        pos["take_profit"] = decision["take_profit"]
        pos["ladder_steps"] += 1
        self.logger.info(f"Position laddered: {self.current_position}")

    def _close_position(self, exit_price, reason):
        """Resets the internal state when a position is closed."""
        self.logger.info(f"Position closed at {exit_price}. Reason: {reason}")
        self.current_position = self._get_default_position()
