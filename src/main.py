# src/main.py
import time
import datetime
import os
import sys
import signal # For graceful shutdown
import pandas as pd
import numpy as np # For simulating data

# Import core configuration and utilities
from config import CONFIG, KLINES_FILENAME, AGG_TRADES_FILENAME, FUTURES_FILENAME, PIPELINE_PID_FILE, RESTART_FLAG_FILE, INITIAL_EQUITY
from utils.logger import system_logger # Centralized logger
from emails.ares_mailer import send_email # For pipeline alerts

# Import hierarchical modules
from analyst.analyst import Analyst
from tactician.tactician import Tactician
from strategist.strategist import Strategist
from supervisor.supervisor import Supervisor
from sentinel.sentinel import Sentinel

# Import data utilities for fetching/simulating real-time data
from analyst.data_utils import load_klines_data, load_agg_trades_data, load_futures_data, simulate_order_book_data

class AresPipeline:
    """
    The main orchestration class for Project Ares.
    It manages the real-time loop, inter-module communication, and system state.
    """
    def __init__(self, config=CONFIG):
        self.config = config
        self.logger = system_logger.getChild('AresPipeline') # Child logger for pipeline
        
        # Initialize all core modules
        self.analyst = Analyst(config)
        self.strategist = Strategist(config)
        self.tactician = Tactician(config)
        self.supervisor = Supervisor(config)
        self.sentinel = Sentinel(config)

        # Internal state variables
        self.current_equity = INITIAL_EQUITY
        self.trade_logs_today = [] # List of completed trades for the current day
        self.daily_pnl_per_regime = {} # P&L aggregated by regime for the current day
        self.historical_daily_pnl_data = pd.DataFrame(columns=['Date', 'NetPnL']) # For Supervisor's allocation
        
        self.last_strategist_update_time = datetime.datetime.min # Track last update for periodic calls
        self.last_supervisor_update_time = datetime.datetime.min
        self.last_sentinel_check_time = datetime.datetime.min

        # Load historical data for Analyst and Strategist once at startup
        self._initial_data_load()

    def _initial_data_load(self):
        """Loads and prepares initial historical data for Analyst and Strategist."""
        self.logger.info("AresPipeline: Initializing historical data for modules...")
        if not self.analyst.load_and_prepare_historical_data():
            self.logger.error("Analyst failed to load historical data. System might not function correctly.")
            # Consider triggering a shutdown here if data is critical for startup
        
        if not self.strategist.load_historical_data_htf():
            self.logger.error("Strategist failed to load HTF data. Macro analysis might be limited.")
        
        # Load any existing daily summary log for supervisor's historical PnL
        if os.path.exists(self.supervisor.daily_summary_log_filename):
            try:
                self.historical_daily_pnl_data = pd.read_csv(
                    self.supervisor.daily_summary_log_filename, 
                    parse_dates=['Date'], 
                    index_col='Date'
                )[['NetPnL']].reset_index() # Keep only Date and NetPnL
                self.logger.info(f"Loaded {len(self.historical_daily_pnl_data)} historical daily P&L records.")
            except Exception as e:
                self.logger.error(f"Error loading historical daily P&L data: {e}")
                self.historical_daily_pnl_data = pd.DataFrame(columns=['Date', 'NetPnL'])

    def _get_real_time_market_data(self):
        """
        Simulates fetching real-time market data (klines, agg trades, futures, order book).
        In a live system, this would connect to exchange APIs (e.g., BinanceFuturesAPI).
        """
        self.logger.debug("Fetching real-time market data...")
        try:
            # Simulate fetching the latest N candles/trades/futures data
            # In a real system, you'd fetch only new data since last update
            latest_klines = load_klines_data(KLINES_FILENAME).tail(self.config['analyst']['market_health_analyzer']['ma_periods'][-1] + 50) # Enough for indicators
            latest_agg_trades = load_agg_trades_data(AGG_TRADES_FILENAME).tail(200) # Recent trades
            latest_futures = load_futures_data(FUTURES_FILENAME).tail(1) # Latest funding/OI

            if latest_klines.empty:
                self.logger.error("No real-time k-lines data available.")
                return None, None, None, None, None, None

            current_price = latest_klines['close'].iloc[-1]
            current_volume = latest_klines['volume'].iloc[-1]
            current_atr = latest_klines['ATR'].iloc[-1] if 'ATR' in latest_klines.columns else 0 # From Analyst features

            current_order_book = simulate_order_book_data(current_price) # Simulated

            # Simulate current position details (would come from exchange API)
            current_position_notional = self.tactician.current_position["size"] * self.tactician.current_position["entry_price"]
            current_liquidation_price = self.tactician.current_position["entry_price"] * (1 - 0.05 * np.sign(self.tactician.current_position["size"])) if self.tactician.current_position["size"] != 0 else 0.0 # Dummy liq price

            return latest_klines, latest_agg_trades, latest_futures, current_order_book, current_price, current_atr, current_position_notional, current_liquidation_price
        except Exception as e:
            self.logger.error(f"Error fetching real-time market data: {e}", exc_info=True)
            return None, None, None, None, None, None, None, None

    def _update_system_state_from_trade(self, trade_action_result: dict):
        """
        Updates the pipeline's internal state (equity, trade logs) based on a Tactician's trade action.
        :param trade_action_result: The dictionary returned by Tactician.process_intelligence.
        """
        action_type = trade_action_result.get("action")
        details = trade_action_result.get("details", {})

        if action_type == "POSITION_CLOSED":
            # Simulate P&L from the closed trade (Tactician already prints it)
            # For accurate P&L, you'd need to fetch actual exchange P&L.
            # Here, we'll get it from Tactician's internal state before it's reset.
            closed_position_info = self.tactician.current_position # This will be the *old* position
            
            # This is a hacky way to get P&L for logging after position reset
            # In a real system, Tactician would return the full trade log entry on close.
            simulated_pnl = 0.0
            if closed_position_info["size"] != 0 and details.get("reason") in ["Take Profit", "Stop Loss/Reversal"]:
                exit_price = details.get("current_price", closed_position_info["entry_price"]) # Assuming current_price is passed
                if closed_position_info["direction"] == "LONG":
                    simulated_pnl = (exit_price - closed_position_info["entry_price"]) * closed_position_info["size"] * closed_position_info["current_leverage"]
                else: # SHORT
                    simulated_pnl = (closed_position_info["entry_price"] - exit_price) * closed_position_info["size"] * closed_position_info["current_leverage"]
            
            trade_log_entry = {
                "Trade ID": self.tactician.trade_id_counter, # Use the counter before it increments for next trade
                "Entry/Exit Timestamps": datetime.datetime.now().isoformat(),
                "Asset": closed_position_info["symbol"],
                "Direction": closed_position_info["direction"],
                "Market State at Entry": self.analyst.get_intelligence(
                    load_klines_data(KLINES_FILENAME).tail(10), # Dummy data for state at entry
                    load_agg_trades_data(AGG_TRADES_FILENAME).tail(50),
                    load_futures_data(FUTURES_FILENAME).tail(1),
                    simulate_order_book_data(closed_position_info["entry_price"]),
                    0, 0 # No open position for this call
                )['market_regime'], # Get regime at entry (simplified)
                "Entry Price": closed_position_info["entry_price"],
                "Exit Price": details.get("current_price", closed_position_info["entry_price"]),
                "Position Size": closed_position_info["size"],
                "Leverage Used": closed_position_info["current_leverage"],
                "Confidence Score & LSS at Entry": {"conf": 0.0, "lss": 0.0}, # Placeholder, would be captured at entry
                "Fees Paid": abs(simulated_pnl) * 0.0005, # Simulate fees
                "Funding Rate Paid/Received": 0.0, # Placeholder
                "Realized P&L ($)": simulated_pnl,
                "Exit Reason": details.get("reason", "Unknown")
            }
            self.trade_logs_today.append(trade_log_entry)
            self.current_equity += simulated_pnl # Update overall equity

            # Aggregate P&L by regime for Supervisor
            regime = trade_log_entry['Market State at Entry']
            if regime not in self.daily_pnl_per_regime:
                self.daily_pnl_per_regime[regime] = 0.0
            self.daily_pnl_per_regime[regime] += simulated_pnl

            self.logger.info(f"Trade closed. Equity updated to ${self.current_equity:,.2f}. P&L: ${simulated_pnl:,.2f}")
        
        elif action_type == "ORDER_PLACED" or action_type == "LADDER_UPDATED":
            # Store initial trade details for eventual logging when closed
            # For now, just log the action
            self.logger.info(f"Tactician action: {action_type}. Details: {details}")

    def _handle_graceful_shutdown(self, signum, frame):
        """Handles signals for graceful shutdown."""
        self.logger.critical(f"\nReceived signal {signum}. Initiating graceful shutdown...")
        # In a real system:
        # 1. Close all open positions.
        # 2. Cancel all open orders.
        # 3. Save any critical state.
        # 4. Remove PID file.
        self.logger.info("Performing graceful shutdown tasks (placeholder)...")
        if os.path.exists(PIPELINE_PID_FILE):
            os.remove(PIPELINE_PID_FILE)
            self.logger.info(f"Removed PID file: {PIPELINE_PID_FILE}")
        self.logger.info("AresPipeline: Shutdown complete.")
        sys.exit(0)

    def _write_pid_file(self):
        """Writes the current process ID to a file."""
        try:
            with open(PIPELINE_PID_FILE, 'w') as f:
                f.write(str(os.getpid()))
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

    def run(self):
        """
        Starts the main real-time operational loop of the Ares system.
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

        # Initialize strategist params once at startup
        current_market_data_for_strategist = self._get_real_time_market_data()
        if current_market_data_for_strategist[4] is not None: # current_price
            self.strategist_params = self.strategist.get_strategist_parameters(
                analyst_market_health_score=self.analyst.market_health_analyzer.get_market_health_score(current_market_data_for_strategist[0]), # Pass latest klines
                current_price=current_market_data_for_strategist[4]
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


        try:
            while True:
                current_time = datetime.datetime.now()
                current_date = current_time.date()

                self.logger.info(f"\n--- Pipeline Loop: {current_time.strftime('%Y-%m-%d %H:%M:%S')} ---")

                # 1. Fetch Real-Time Market Data
                klines, agg_trades, futures, order_book, current_price, current_atr, pos_notional, liq_price = self._get_real_time_market_data()
                if klines is None:
                    self.logger.warning("Skipping current loop iteration due to missing market data.")
                    time.sleep(loop_interval)
                    continue
                
                current_market_data = {
                    "current_price": current_price,
                    "current_volume": klines['volume'].iloc[-1],
                    "current_atr": current_atr # Passed to Tactician
                }

                # 2. Strategist Update (e.g., daily)
                if (current_time - self.last_strategist_update_time).total_seconds() >= strategist_update_interval:
                    self.logger.info("Triggering Strategist update...")
                    self.strategist_params = self.strategist.get_strategist_parameters(
                        analyst_market_health_score=self.analyst.market_health_analyzer.get_market_health_score(klines),
                        current_price=current_price
                    )
                    self.last_strategist_update_time = current_time

                # 3. Analyst Intelligence
                analyst_intelligence = self.analyst.get_intelligence(
                    klines, agg_trades, futures, order_book, pos_notional, liq_price
                )

                # 4. Tactician Decision
                tactician_decision_result = self.tactician.process_intelligence(
                    analyst_intelligence, self.strategist_params, current_market_data
                )
                
                # Update pipeline state based on Tactician's action
                self._update_system_state_from_trade(tactician_decision_result)

                # 5. Supervisor Update (e.g., daily at end of day)
                # This needs to be triggered at a specific time (e.g., 00:00 UTC) or after a full day of data.
                # For this demo, we'll trigger it if a new day starts based on loop's current_date.
                if current_date > self.last_supervisor_update_time.date():
                    self.logger.info(f"Triggering Supervisor daily update for {self.last_supervisor_update_time.date()}...")
                    supervisor_output = self.supervisor.orchestrate_supervision(
                        current_date=self.last_supervisor_update_time.date(), # Report for the *previous* day
                        total_equity=self.current_equity,
                        daily_trade_logs=self.trade_logs_today,
                        daily_pnl_per_regime=self.daily_pnl_per_regime,
                        historical_daily_pnl_data=self.historical_daily_pnl_data
                    )
                    # Update historical P&L for next supervisor run
                    self.historical_daily_pnl_data = pd.concat([
                        self.historical_daily_pnl_data, 
                        pd.DataFrame([{'Date': self.last_supervisor_update_time.date(), 'NetPnL': supervisor_output['daily_summary']['NetPnL']}])
                    ]).drop_duplicates(subset=['Date']).sort_values('Date')
                    
                    self.current_equity = supervisor_output["allocated_capital"] # Supervisor updates equity
                    self.trade_logs_today = [] # Reset trade logs for new day
                    self.daily_pnl_per_regime = {} # Reset daily P&L by regime
                    self.last_supervisor_update_time = current_time # Update last supervisor run time
                
                # Set last_supervisor_update_time to current_time if it's the first run or a new day
                if self.last_supervisor_update_time == datetime.datetime.min:
                    self.last_supervisor_update_time = current_time


                # 6. Sentinel Checks
                if (current_time - self.last_sentinel_check_time).total_seconds() >= sentinel_check_interval:
                    self.logger.info("Triggering Sentinel checks...")
                    # Pass the latest states of the modules to Sentinel
                    self.sentinel.run_checks(
                        current_analyst_intelligence=analyst_intelligence,
                        current_strategist_params=self.strategist_params,
                        current_tactician_decision=tactician_decision_result, # Pass the raw decision result
                        latest_trade_data=self.trade_logs_today[-1] if self.trade_logs_today else None, # Pass last trade if any
                        average_trade_size=self.supervisor.get_current_allocated_capital() * self.config['tactician']['risk_management']['risk_per_trade_pct'] # Example average trade size
                    )
                    self.last_sentinel_check_time = current_time

                self.logger.info(f"Current Equity: ${self.current_equity:,.2f}")
                self.logger.info(f"Allocated Capital: ${self.supervisor.get_current_allocated_capital():,.2f}")
                self.logger.info(f"Current Position: {self.tactician.current_position['direction']} {self.tactician.current_position['size']:.4f} {self.tactician.current_position['symbol'] if self.tactician.current_position['symbol'] else ''}")
                
                time.sleep(loop_interval)

        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt detected. Exiting pipeline.")
        except Exception as e:
            self.logger.critical(f"An unhandled error occurred in the main pipeline loop: {e}", exc_info=True)
            self._send_alert("Ares Critical Error: Unhandled Pipeline Exception", f"An unhandled exception occurred:\n\n{e}\n\n{sys.exc_info()}")
        finally:
            self.logger.info("--- Ares Pipeline Shutting Down ---")
            self._remove_pid_file()


# --- Main execution block ---
if __name__ == "__main__":
    # Ensure logs directory exists before logger initialization
    os.makedirs("logs", exist_ok=True)
    os.makedirs("reports", exist_ok=True) # Ensure reports dir for supervisor

    # Initial setup for dummy data if needed
    from analyst.data_utils import create_dummy_data
    create_dummy_data(KLINES_FILENAME, 'klines')
    create_dummy_data(AGG_TRADES_FILENAME, 'agg_trades')
    create_dummy_data(FUTURES_FILENAME, 'futures')

    pipeline = AresPipeline()
    pipeline.run()

