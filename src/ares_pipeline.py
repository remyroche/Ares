# src/ares_pipeline.py
import time
import datetime
import os
import sys
import signal # For graceful shutdown
import pandas as pd
import numpy as np # For simulating data
import asyncio # For running async Firestore operations

# Import core configuration and utilities
from config import CONFIG, PIPELINE_PID_FILE, RESTART_FLAG_FILE, INITIAL_EQUITY, SYMBOL, INTERVAL
from utils.logger import system_logger # Centralized logger
from emails.ares_mailer import send_email # For pipeline alerts

# Import hierarchical modules
from analyst.analyst import Analyst
from tactician.tactician import Tactician
from strategist.strategist import Strategist
from supervisor.supervisor import Supervisor
from sentinel.sentinel import Sentinel

# Import Binance client (now enhanced for live data)
from exchange.binance import BinanceFuturesAPI # Assuming exchange/binance.py is now in root or accessible

# Import Firestore Manager
from database.firestore_manager import FirestoreManager # New import

class AresPipeline:
    """
    The main orchestration class for Project Ares.
    It manages the real-time loop, inter-module communication, and system state.
    """
    def __init__(self, config=CONFIG):
        self.config = config
        self.logger = system_logger.getChild('AresPipeline') # Child logger for pipeline
        
        # Initialize Firestore Manager
        self.firestore_manager = FirestoreManager(
            config=self.config,
            app_id=os.environ.get('__app_id'), # Passed by Canvas environment
            firebase_config_str=os.environ.get('__firebase_config'), # Passed by Canvas environment
            initial_auth_token=os.environ.get('__initial_auth_token') # Passed by Canvas environment
        )

        # Initialize Binance Client for live data and trading
        live_config = self.config.get("live_trading", {})
        self.live_trading_enabled = live_config.get("enabled", False)
        self.binance_client = BinanceFuturesAPI(
            api_key=live_config.get("api_key"),
            api_secret=live_config.get("api_secret"),
            testnet=live_config.get("testnet", True),
            symbol=SYMBOL,
            interval=INTERVAL,
            config=self.config # Pass full config for WebSocket streams
        )

        # Initialize all core modules, passing FirestoreManager where needed
        self.analyst = Analyst(config, firestore_manager=self.firestore_manager)
        self.strategist = Strategist(config) # Strategist doesn't directly use Firestore for now
        self.tactician = Tactician(config, firestore_manager=self.firestore_manager)
        self.supervisor = Supervisor(config, firestore_manager=self.firestore_manager)
        self.sentinel = Sentinel(config)

        # Internal state variables
        self.current_equity = INITIAL_EQUITY # This will be updated by live balance if live trading
        self.trade_logs_today = [] # List of completed trades for the current day
        self.daily_pnl_per_regime = {} # P&L aggregated by regime for the current day
        self.historical_daily_pnl_data = pd.DataFrame(columns=['Date', 'NetPnL']) # For Supervisor's allocation
        
        self.last_strategist_update_time = datetime.datetime.min # Track last update for periodic calls
        self.last_supervisor_update_time = datetime.datetime.min
        self.last_sentinel_check_time = datetime.datetime.min

        # Load initial system state and historical data
        asyncio.run(self._initial_setup()) # Run async setup

    async def _initial_setup(self):
        """Performs initial setup, including connecting to Binance and loading historical data."""
        self.logger.info("AresPipeline: Initializing system setup...")

        if self.live_trading_enabled:
            self.logger.info("Live trading is ENABLED. Connecting to Binance API...")
            # Start WebSocket data streams
            self.binance_client.start_data_streams()
            if self.config["live_trading"].get("websocket_streams", {}).get("userData", False):
                self.binance_client.start_user_data_stream()
            
            # Give some time for initial WebSocket data to populate buffers
            await asyncio.sleep(5) 
            
            # Fetch initial account balance and position
            account_balance = self.binance_client.get_latest_account_balance()
            if 'USDT' in account_balance:
                self.current_equity = account_balance['USDT']['free'] + account_balance['USDT']['locked']
                self.logger.info(f"Initial live equity from Binance: ${self.current_equity:,.2f}")
            else:
                self.logger.warning("Could not fetch USDT balance from live account. Using initial equity from config.")
                self.current_equity = INITIAL_EQUITY # Fallback

            # Update Tactician's initial position based on live data
            live_position = self.binance_client.get_latest_position(SYMBOL)
            if live_position and live_position.get('positionAmt', 0) != 0:
                self.tactician.current_position = {
                    "symbol": SYMBOL,
                    "direction": "LONG" if live_position['positionAmt'] > 0 else "SHORT",
                    "size": abs(live_position['positionAmt']),
                    "entry_price": live_position['entryPrice'],
                    "unrealized_pnl": live_position['unrealizedPnl'],
                    "current_leverage": live_position['leverage'],
                    "ladder_steps": 0, # Cannot determine from live data, assume 0 for now
                    "stop_loss": None, # Needs to be managed via user data stream or separate query
                    "take_profit": None # Needs to be managed via user data stream or separate query
                }
                self.logger.info(f"Detected existing live position: {self.tactician.current_position}")
            else:
                self.logger.info("No existing live position detected.")

        else:
            self.logger.info("Live trading is DISABLED. Using simulated data for pipeline operation.")
            # Ensure dummy data files exist for simulation
            from analyst.data_utils import create_dummy_data
            create_dummy_data(KLINES_FILENAME, 'klines')
            create_dummy_data(AGG_TRADES_FILENAME, 'agg_trades')
            create_dummy_data(CONFIG['FUTURES_FILENAME'], 'futures') # Use CONFIG['FUTURES_FILENAME']

        # Load and prepare historical data for Analyst and Strategist (from CSVs for now)
        # This is still done from CSVs as fetching full history via REST is slow and for training.
        self.logger.info("Loading and preparing historical data for Analyst and Strategist (from CSVs)...")
        if not self.analyst.load_and_prepare_historical_data():
            self.logger.error("Analyst failed to load historical data. System might not function correctly.")
        
        if not self.strategist.load_historical_data_htf():
            self.logger.error("Strategist failed to load HTF data. Macro analysis might be limited.")
        
        # Load historical daily P&L from Firestore first, then CSV as fallback
        if self.firestore_manager.firestore_enabled:
            firestore_pnl_data = await self.firestore_manager.get_collection(
                self.config['supervisor']['daily_summary_log_filename'].split('/')[-1].replace('.csv', ''), # Collection name
                is_public=False # Assuming daily summary is private per user
            )
            if firestore_pnl_data:
                # Convert Firestore docs to DataFrame, handling potential missing fields
                df_pnl = pd.DataFrame(firestore_pnl_data)
                if 'Date' in df_pnl.columns and 'NetPnL' in df_pnl.columns:
                    df_pnl['Date'] = pd.to_datetime(df_pnl['Date'])
                    self.historical_daily_pnl_data = df_pnl[['Date', 'NetPnL']].sort_values('Date').reset_index(drop=True)
                    self.logger.info(f"Loaded {len(self.historical_daily_pnl_data)} historical daily P&L records from Firestore.")
                else:
                    self.logger.warning("Firestore daily P&L data missing 'Date' or 'NetPnL' columns.")
            else:
                self.logger.info("No historical daily P&L data found in Firestore. Checking CSV.")

        if self.historical_daily_pnl_data.empty and os.path.exists(self.supervisor.daily_summary_log_filename):
            try:
                self.historical_daily_pnl_data = pd.read_csv(
                    self.supervisor.daily_summary_log_filename, 
                    parse_dates=['Date']
                )[['Date', 'NetPnL']].set_index('Date').reset_index()
                self.logger.info(f"Loaded {len(self.historical_daily_pnl_data)} historical daily P&L records from CSV.")
            except Exception as e:
                self.logger.error(f"Error loading historical daily P&L data from CSV: {e}")
                self.historical_daily_pnl_data = pd.DataFrame(columns=['Date', 'NetPnL'])

        # Load last known BEST_PARAMS from Firestore
        if self.firestore_manager.firestore_enabled:
            last_params_doc = await self.firestore_manager.get_document(
                self.config['firestore']['optimized_params_collection'],
                doc_id='latest', # Assuming 'latest' document stores current best params
                is_public=True # Optimized params might be public
            )
            if last_params_doc and 'params' in last_params_doc:
                self.logger.info("Loaded latest optimized parameters from Firestore.")
                # Update CONFIG.BEST_PARAMS with loaded values
                # This requires a helper function to deep update a dictionary
                self._deep_update_dict(self.config['BEST_PARAMS'], last_params_doc['params'])
                self.logger.info(f"Updated CONFIG.BEST_PARAMS with Firestore values: {self.config['BEST_PARAMS']}")
            else:
                self.logger.info("No latest optimized parameters found in Firestore. Using default BEST_PARAMS.")

        # Initialize strategist params once at startup
        initial_klines_for_strategist = self.binance_client.get_latest_klines(num_klines=500) if self.live_trading_enabled else pd.DataFrame()
        if initial_klines_for_strategist.empty: # Fallback if live data not ready or not enabled
            initial_klines_for_strategist = load_klines_data(CONFIG['KLINES_FILENAME']).tail(500)

        initial_price_for_strategist = initial_klines_for_strategist['close'].iloc[-1] if not initial_klines_for_strategist.empty else 0.0

        if initial_price_for_strategist != 0.0:
            self.strategist_params = self.strategist.get_strategist_parameters(
                analyst_market_health_score=self.analyst.market_health_analyzer.get_market_health_score(initial_klines_for_strategist),
                current_price=initial_price_for_strategist
            )
        else:
            self.strategist_params = { # Default if no data
                "Trading_Range": {"low": 0.0, "high": float('inf')},
                "Max_Allowable_Leverage_Cap": self.config['strategist'].get("max_leverage_cap_default", 100),
                "Positional_Bias": "NEUTRAL"
            }
        
        # Initialize previous states for Sentinel
        self.sentinel.previous_analyst_intelligence = {}
        self.sentinel.previous_strategist_params = self.strategist_params
        self.sentinel.previous_tactician_decision = {"action": "INITIAL_STARTUP"}

        self.logger.info("AresPipeline: Initial setup complete.")

    def _deep_update_dict(self, target_dict, source_dict):
        """Recursively updates a dictionary."""
        for key, value in source_dict.items():
            if isinstance(value, dict) and key in target_dict and isinstance(target_dict[key], dict):
                self._deep_update_dict(target_dict[key], value)
            else:
                target_dict[key] = value

    def _get_real_time_market_data(self):
        """
        Fetches real-time market data from the Binance client's buffers (live) or CSVs (simulated).
        """
        self.logger.debug("Fetching real-time market data...")
        try:
            if self.live_trading_enabled:
                klines = self.binance_client.get_latest_klines(num_klines=500) # Get enough for indicators
                agg_trades = self.binance_client.get_latest_agg_trades(num_trades=1000)
                order_book = self.binance_client.get_latest_order_book()
                
                # Get latest account and position info from client's user data buffer
                account_balance = self.binance_client.get_latest_account_balance()
                current_position_live = self.binance_client.get_latest_position(SYMBOL)

                current_price = klines['close'].iloc[-1] if not klines.empty else 0.0
                current_volume = klines['volume'].iloc[-1] if not klines.empty else 0.0
                # ATR will be calculated by Analyst's feature engine
                current_atr = 0.0 

                pos_notional = 0.0
                liq_price = 0.0

                if current_position_live and current_position_live.get('positionAmt', 0) != 0:
                    pos_notional = abs(current_position_live['positionAmt']) * current_price
                    liq_price = current_position_live['liquidationPrice']
                    # Update Tactician's internal position state with live data
                    self.tactician.current_position.update({
                        "unrealized_pnl": current_position_live['unrealizedPnl'],
                        "liquidation_price": liq_price # Add liq price to tactician's state
                    })
                
                # Update current equity from live balance
                if 'USDT' in account_balance:
                    self.current_equity = account_balance['USDT']['free'] + account_balance['USDT']['locked']
                
                # Futures data (funding rate, open interest) will be derived from klines or separate stream
                # For now, we'll use the klines df for funding rate if available, otherwise simulate.
                futures_data = pd.DataFrame([{'fundingRate': 0.0, 'openInterest': 0.0}]) # Placeholder
                if 'fundingRate' in klines.columns: # If klines has funding rate (unlikely directly)
                     futures_data['fundingRate'] = klines['fundingRate'].iloc[-1]
                if 'openInterest' in klines.columns: # If klines has open interest
                     futures_data['openInterest'] = klines['openInterest'].iloc[-1]
                
                # Ensure futures_data has an index for merging later
                futures_data.index = [klines.index[-1]] if not klines.empty else [pd.Timestamp.now()]
                futures_data.index.name = 'timestamp'


                return klines, agg_trades, futures_data, order_book, current_price, current_atr, pos_notional, liq_price
            else:
                # Simulated data for non-live mode
                # Load from CSVs as before
                klines = self.analyst.load_and_prepare_historical_data_for_sim().tail(500) # Use Analyst's internal data for consistency
                agg_trades = self.analyst._historical_agg_trades.tail(1000) if self.analyst._historical_agg_trades is not None else pd.DataFrame()
                futures = self.analyst._historical_futures.tail(1) if self.analyst._historical_futures is not None else pd.DataFrame()
                
                current_price = klines['close'].iloc[-1] if not klines.empty else 0.0
                current_volume = klines['volume'].iloc[-1] if not klines.empty else 0.0
                current_order_book = simulate_order_book_data(current_price)

                current_atr = klines['ATR'].iloc[-1] if 'ATR' in klines.columns else 0.0 # Analyst will add this
                
                # Simulate position for non-live mode
                pos_notional = self.tactician.current_position["size"] * self.tactician.current_position["entry_price"] if self.tactician.current_position["size"] != 0 else 0.0
                liq_price = self.tactician.current_position["entry_price"] * (1 - 0.05 * np.sign(self.tactician.current_position["size"])) if self.tactician.current_position["size"] != 0 else 0.0 # Dummy liq price

                return klines, agg_trades, futures, current_order_book, current_price, current_atr, pos_notional, liq_price
        except Exception as e:
            self.logger.error(f"Error fetching real-time market data: {e}", exc_info=True)
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, 0.0, 0.0, 0.0, 0.0


    async def _update_system_state_from_trade(self, trade_action_result: dict, current_price: float):
        """
        Updates the pipeline's internal state (equity, trade logs) based on a Tactician's trade action.
        Also interacts with the live exchange for order execution and saves to Firestore.
        :param trade_action_result: The dictionary returned by Tactician.process_intelligence.
        :param current_price: The current market price at the time of action.
        """
        action_type = trade_action_result.get("action")
        details = trade_action_result.get("details", {})
        
        symbol = details.get("symbol", SYMBOL) # Default to SYMBOL from config

        if action_type == "ORDER_PLACED":
            if self.live_trading_enabled:
                try:
                    order_response = await self.binance_client.place_order(
                        symbol=symbol,
                        side=details["direction"],
                        type=details["order_type"],
                        quantity=details["quantity"],
                        price=details.get("price"), # For LIMIT orders
                        stop_price=details.get("stop_loss"), # For STOP_MARKET
                        time_in_force="GTC"
                    )
                    if order_response and "orderId" in order_response:
                        self.logger.info(f"Live order placed: {order_response['orderId']} - {order_response['status']}")
                        # Update Tactician's internal position with actual entry price from response
                        self.tactician._update_position(
                            symbol=symbol,
                            direction=details["direction"],
                            size=float(order_response.get("executedQty", details["quantity"])),
                            entry_price=float(order_response.get("avgPrice", current_price)), # Use avgPrice if available
                            leverage=details["leverage"],
                            stop_loss=details.get("stop_loss"),
                            take_profit=details.get("take_profit")
                        )
                    else:
                        self.logger.error(f"Failed to place live order: {order_response}")
                        send_email("Ares Live Trading Error: Order Placement Failed", f"Failed to place order: {order_response}")
                except Exception as e:
                    self.logger.critical(f"Exception during live order placement: {e}", exc_info=True)
                    send_email("Ares Live Trading Critical Error", f"Exception placing order: {e}")
            else: # Simulation mode
                self.logger.info(f"Simulating PLACE_ORDER: {details['direction']} {details['quantity']:.4f} {symbol}")
                self.tactician._update_position(
                    symbol=symbol,
                    direction=details["direction"],
                    size=details["quantity"],
                    entry_price=current_price,
                    leverage=details["leverage"],
                    stop_loss=details.get("stop_loss"),
                    take_profit=details.get("take_profit")
                )

        elif action_type == "LADDER_UPDATED":
            if self.live_trading_enabled:
                try:
                    order_response = await self.binance_client.place_order(
                        symbol=symbol,
                        side=details["direction"],
                        type=details["order_type"], # Usually MARKET
                        quantity=details["quantity"], # This is the *additional* quantity
                        price=details.get("price"),
                        stop_price=details.get("stop_loss"),
                        time_in_force="GTC"
                    )
                    if order_response and "orderId" in order_response:
                        self.logger.info(f"Live ladder order placed: {order_response['orderId']} - {order_response['status']}")
                        self.tactician._add_to_ladder(
                            symbol=symbol,
                            direction=details["direction"],
                            current_price=float(order_response.get("avgPrice", current_price)),
                            additional_size=float(order_response.get("executedQty", details["quantity"])),
                            new_leverage=details["leverage"],
                            new_stop_loss=details.get("stop_loss"),
                            new_take_profit=details.get("take_profit")
                        )
                    else:
                        self.logger.error(f"Failed to place live ladder order: {order_response}")
                        send_email("Ares Live Trading Error: Ladder Order Failed", f"Failed to place ladder order: {order_response}")
                except Exception as e:
                    self.logger.critical(f"Exception during live ladder order placement: {e}", exc_info=True)
                    send_email("Ares Live Trading Critical Error", f"Exception placing ladder order: {e}")
            else: # Simulation mode
                self.logger.info(f"Simulating ADD_TO_LADDER: Add {details['quantity']:.4f} {symbol}")
                self.tactician._add_to_ladder(
                    symbol=symbol,
                    direction=details["direction"],
                    current_price=current_price,
                    additional_size=details["quantity"],
                    new_leverage=details["leverage"],
                    new_stop_loss=details.get("stop_loss"),
                    new_take_profit=details.get("take_profit")
                )

        elif action_type == "POSITION_CLOSED":
            if self.live_trading_enabled:
                try:
                    current_live_pos = await self.binance_client.get_latest_position(symbol) # Ensure latest live pos
                    if current_live_pos and current_live_pos.get('positionAmt', 0) != 0:
                        close_side = "SELL" if current_live_pos['positionAmt'] > 0 else "BUY"
                        close_quantity = abs(current_live_pos['positionAmt'])
                        
                        order_response = await self.binance_client.place_order(
                            symbol=symbol,
                            side=close_side,
                            type="MARKET",
                            quantity=close_quantity,
                            reduce_only=True
                        )
                        if order_response and "orderId" in order_response:
                            self.logger.info(f"Live position close order placed: {order_response['orderId']} - {order_response['status']}")
                            self.tactician._close_position(current_price, details.get("reason", "Live Close"))
                            await self._log_completed_trade(symbol, current_live_pos, current_price, details.get("reason", "Live Close"))
                        else:
                            self.logger.error(f"Failed to place live close order: {order_response}")
                            send_email("Ares Live Trading Error: Close Order Failed", f"Failed to close position: {order_response}")
                    else:
                        self.logger.warning("Attempted to close position but no live position found.")
                except Exception as e:
                    self.logger.critical(f"Exception during live position close: {e}", exc_info=True)
                    send_email("Ares Live Trading Critical Error", f"Exception closing position: {e}")
            else: # Simulation mode
                self.logger.info(f"Simulating CLOSE_POSITION: {details.get('reason', 'Unknown')} at {current_price:.2f}")
                await self._log_completed_trade(symbol, self.tactician.current_position, current_price, details.get("reason", "Simulated Close"))
                self.tactician._close_position(current_price, details.get("reason", "Simulated Close"))
        
        elif action_type == "ORDER_CANCELLED":
            if self.live_trading_enabled:
                try:
                    order_response = await self.binance_client.cancel_order(
                        symbol=symbol,
                        order_id=details.get("order_id"),
                        orig_client_order_id=details.get("orig_client_order_id")
                    )
                    if order_response and order_response.get("status") == "CANCELED":
                        self.logger.info(f"Live order cancelled: {order_response['orderId']}")
                    else:
                        self.logger.error(f"Failed to cancel live order: {order_response}")
                        send_email("Ares Live Trading Error: Order Cancellation Failed", f"Failed to cancel order: {order_response}")
                except Exception as e:
                    self.logger.critical(f"Exception during live order cancellation: {e}", exc_info=True)
                    send_email("Ares Live Trading Critical Error", f"Exception cancelling order: {e}")
            else:
                self.logger.info(f"Simulating CANCEL_ORDER: {details.get('order_id', 'N/A')}")
        
        elif action_type == "HOLD":
            self.logger.info("Tactician decided to HOLD.")

    async def _log_completed_trade(self, symbol: str, closed_position_info: dict, exit_price: float, exit_reason: str):
        """
        Logs a completed trade to the daily trade logs and saves to Firestore.
        This is called when a position is closed.
        """
        if closed_position_info["size"] == 0:
            return # No actual position was closed

        # Calculate P&L for logging
        simulated_pnl = 0.0
        if closed_position_info["entry_price"] != 0:
            pnl_per_unit = (exit_price - closed_position_info["entry_price"]) if closed_position_info["direction"] == "LONG" else \
                           (closed_position_info["entry_price"] - exit_price)
            simulated_pnl = pnl_per_unit * closed_position_info["size"] * closed_position_info["current_leverage"]
        
        # Get market state at entry (simplified for now)
        market_state_at_entry = self.analyst.get_intelligence(
            self.binance_client.get_latest_klines(num_klines=500), # Use live klines for this
            self.binance_client.get_latest_agg_trades(num_trades=1000),
            pd.DataFrame([{'fundingRate': 0.0, 'openInterest': 0.0}], index=[pd.Timestamp.now()], columns=['fundingRate', 'openInterest']), # Dummy futures
            self.binance_client.get_latest_order_book(),
            0, 0 # No open position for this call
        )['market_regime']

        trade_log_entry = {
            "trade_id": str(self.tactician.trade_id_counter), # Use the counter for the trade that just completed
            "timestamp": datetime.datetime.now().isoformat(),
            "asset": symbol,
            "direction": closed_position_info["direction"],
            "market_state_at_entry": market_state_at_entry,
            "entry_price": closed_position_info["entry_price"],
            "exit_price": exit_price,
            "position_size": closed_position_info["size"],
            "leverage_used": closed_position_info["current_leverage"],
            "confidence_score_at_entry": 0.0, # Placeholder, would need to store at entry
            "lss_at_entry": 0.0, # Placeholder, would need to store at entry
            "fees_paid": abs(simulated_pnl) * 0.0005, # Simulate fees
            "funding_rate_pnl": 0.0, # Placeholder
            "realized_pnl_usd": simulated_pnl,
            "exit_reason": exit_reason
        }
        self.trade_logs_today.append(trade_log_entry)
        self.current_equity += simulated_pnl # Update overall equity

        # Aggregate P&L by regime for Supervisor
        regime = trade_log_entry['market_state_at_entry']
        if regime not in self.daily_pnl_per_regime:
            self.daily_pnl_per_regime[regime] = 0.0
        self.daily_pnl_per_regime[regime] += simulated_pnl
        self.logger.info(f"Trade logged. Equity updated to ${self.current_equity:,.2f}. P&L: ${simulated_pnl:,.2f}")

        # Save to Firestore
        if self.firestore_manager.firestore_enabled:
            await self.firestore_manager.add_document(
                self.config['firestore']['trade_logs_collection'],
                trade_log_entry,
                is_public=False # Trade logs are private per user
            )


    def _handle_graceful_shutdown(self, signum, frame):
        """Handles signals for graceful shutdown."""
        self.logger.critical(f"\nReceived signal {signum}. Initiating graceful shutdown...")
        self.logger.info("Performing graceful shutdown tasks...")
        
        if self.live_trading_enabled:
            self.logger.info("Stopping Binance client streams...")
            self.binance_client.stop_all_streams()
            # You might also want to close any open positions or cancel orders here
            # self.binance_client.cancel_all_open_orders(SYMBOL)
            # self.binance_client.close_position(SYMBOL, self.tactician.current_position['direction'], self.tactician.current_position['size'])
        
        self.logger.info("AresPipeline: Shutdown complete.")
        self._remove_pid_file()
        sys.exit(0)

    def _write_pid_file(self):
        """Writes the current process ID to a file."""
        try:
            with open(PIPELINE_PID_FILE, 'w') as f:
                f.write(f"{os.getpid()}")
            self.logger.info(f"PID {os.getpid()} written to {PIPELINE_PID_FILE}")
        except Exception as e:
            self.logger.error(f"Error writing PID file: {e}")

    def _remove_pid_file(self):
        """Removes the PID file."""
        try:
            if os.path.exists(PIPELINE_PID_FILE):
                os.remove(PIPELINE_PID_FILE)
                self.logger.info(f"PID file {PIPELINE_PID_FILE} removed.")
        except Exception as e:
            self.logger.error(f"Error removing PID file: {e}")

    async def run_async(self):
        """
        Starts the main real-time operational loop of the Ares system (async version).
        """
        self.logger.info("--- Ares Pipeline Starting ---")
        self._write_pid_file()

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_graceful_shutdown)  # Ctrl+C
        signal.signal(signal.SIGTERM, self._handle_graceful_shutdown) # Termination signal

        loop_interval = self.config['pipeline'].get("loop_interval_seconds", 10)
        strategist_update_interval = self.config['pipeline'].get("strategist_update_interval_minutes", 1440) * 60 # to seconds
        supervisor_update_interval = self.config['pipeline'].get("supervisor_update_interval_minutes", 1440) * 60 # to seconds
        sentinel_check_interval = self.config['pipeline'].get("sentinel_check_interval_seconds", 30)

        self.logger.info(f"Main loop interval: {loop_interval} seconds.")
        self.logger.info(f"Strategist updates every: {strategist_update_interval / 60} minutes.")
        self.logger.info(f"Supervisor updates every: {supervisor_update_interval / 60} minutes.")
        self.logger.info(f"Sentinel checks every: {sentinel_check_interval} seconds.")
        
        # Set last update times to current time for initial run logic
        self.last_strategist_update_time = datetime.datetime.now()
        self.last_supervisor_update_time = datetime.datetime.now()
        self.last_sentinel_check_time = datetime.datetime.now()


        try:
            while True:
                current_time = datetime.datetime.now()
                current_date = current_time.date()

                self.logger.info(f"\n--- Pipeline Loop: {current_time.strftime('%Y-%m-%d %H:%M:%S')} ---")

                # 1. Fetch Real-Time Market Data
                klines, agg_trades, futures, order_book, current_price, current_atr, pos_notional, liq_price = await self._get_real_time_market_data()
                
                if klines.empty or current_price == 0.0:
                    self.logger.warning("Skipping current loop iteration due to missing or invalid market data.")
                    await asyncio.sleep(loop_interval)
                    continue
                
                current_market_data_for_tactician = {
                    "current_price": current_price,
                    "current_volume": klines['volume'].iloc[-1],
                    "current_atr": current_atr # This will be updated by Analyst's features
                }

                # 2. Strategist Update (e.g., daily)
                if (current_time - self.last_strategist_update_time).total_seconds() >= strategist_update_interval:
                    self.logger.info("Triggering Strategist update...")
                    market_health_for_strategist = self.analyst.market_health_analyzer.get_market_health_score(klines)
                    self.strategist_params = self.strategist.get_strategist_parameters(
                        analyst_market_health_score=market_health_for_strategist,
                        current_price=current_price
                    )
                    self.last_strategist_update_time = current_time

                # 3. Analyst Intelligence
                analyst_intelligence = self.analyst.get_intelligence(
                    klines, agg_trades, futures, order_book, pos_notional, liq_price
                )
                if 'ATR' in analyst_intelligence.get('features', {}):
                    current_market_data_for_tactician['current_atr'] = analyst_intelligence['features']['ATR']


                # 4. Tactician Decision
                tactician_decision_result = await self.tactician.process_intelligence(
                    analyst_intelligence, self.strategist_params, current_market_data_for_tactician
                )
                
                # Update pipeline state and execute live trades based on Tactician's action
                await self._update_system_state_from_trade(tactician_decision_result, current_price)

                # 5. Supervisor Update (e.g., daily at end of day)
                if (current_time - self.last_supervisor_update_time).total_seconds() >= supervisor_update_interval:
                    self.logger.info(f"Triggering Supervisor daily update for {self.last_supervisor_update_time.date()}...")
                    supervisor_output = await self.supervisor.orchestrate_supervision(
                        current_date=self.last_supervisor_update_time.date(), # Report for the *previous* day
                        total_equity=self.current_equity,
                        daily_trade_logs=self.trade_logs_today,
                        daily_pnl_per_regime=self.daily_pnl_per_regime,
                        historical_daily_pnl_data=self.historical_daily_pnl_data
                    )
                    # Update historical P&L for next supervisor run
                    new_daily_pnl_entry = pd.DataFrame([{'Date': self.last_supervisor_update_time.date(), 'NetPnL': supervisor_output['daily_summary']['NetPnL']}])
                    new_daily_pnl_entry['Date'] = pd.to_datetime(new_daily_pnl_entry['Date'])
                    
                    self.historical_daily_pnl_data = pd.concat([self.historical_daily_pnl_data, new_daily_pnl_entry]) \
                                                        .drop_duplicates(subset=['Date']) \
                                                        .sort_values('Date') \
                                                        .reset_index(drop=True)
                    
                    self.current_equity = supervisor_output["allocated_capital"]
                    self.trade_logs_today = []
                    self.daily_pnl_per_regime = {}
                    self.last_supervisor_update_time = current_time
                
                # 6. Sentinel Checks
                if (current_time - self.last_sentinel_check_time).total_seconds() >= sentinel_check_interval:
                    self.logger.info("Triggering Sentinel checks...")
                    self.sentinel.run_checks(
                        current_analyst_intelligence=analyst_intelligence,
                        current_strategist_params=self.strategist_params,
                        current_tactician_decision=tactician_decision_result,
                        latest_trade_data=self.trade_logs_today[-1] if self.trade_logs_today else None,
                        average_trade_size=self.supervisor.get_current_allocated_capital() * self.config['tactician']['risk_management']['risk_per_trade_pct']
                    )
                    self.last_sentinel_check_time = current_time

                self.logger.info(f"Current Equity: ${self.current_equity:,.2f}")
                self.logger.info(f"Allocated Capital: ${self.supervisor.get_current_allocated_capital():,.2f}")
                self.logger.info(f"Current Position: {self.tactician.current_position['direction']} {self.tactician.current_position['size']:.4f} {self.tactician.current_position['symbol'] if self.tactician.current_position['symbol'] else ''} @ {self.tactician.current_position['entry_price']:.2f}x{self.tactician.current_position['current_leverage']}")
                
                await asyncio.sleep(loop_interval)

        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt detected. Exiting pipeline.")
        except Exception as e:
            self.logger.critical(f"An unhandled error occurred in the main pipeline loop: {e}", exc_info=True)
            self._send_alert("Ares Critical Error: Unhandled Pipeline Exception", f"An unhandled exception occurred:\n\n{e}\n\n{sys.exc_info()}")
        finally:
            self.logger.info("--- Ares Pipeline Shutting Down ---")
            self._remove_pid_file()
            if self.live_trading_enabled:
                self.binance_client.stop_all_streams()


# --- Main execution block ---
if __name__ == "__main__":
    # Ensure logs and reports directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("models/analyst", exist_ok=True) # For analyst model storage
    os.makedirs("models/tactician", exist_ok=True) # For tactician model storage

    # Initialize logger
    from importlib import reload
    import utils.logger
    reload(utils.logger) # Reload logger to ensure fresh config for demo
    system_logger = utils.logger.system_logger

    pipeline = AresPipeline()
    asyncio.run(pipeline.run_async()) # Run the async main loop
