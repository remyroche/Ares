import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union # Added import for Dict, Any, Optional
import pandas as pd
import numpy as np
import os
import logging # Ensure logging is imported for logging levels

# Import the scheduler
from apscheduler.schedulers.asyncio import AsyncIOScheduler # Assuming apscheduler is installed

from .dynamic_weighter import DynamicWeighter
from src.utils.model_manager import ModelManager
from src.exchange.binance import BinanceExchange
from src.utils.logger import system_logger
from src.config import settings, CONFIG
from src.utils.state_manager import StateManager
# FirestoreManager import removed - using SQLite only
from src.supervisor.performance_monitor import PerformanceMonitor
from backtesting.ares_backtester import Backtester # Re-added this import to match original
from src.strategist.strategist import Strategist
from src.tactician.tactician import Tactician # Import Tactician
from src.analyst.analyst import Analyst # Import Analyst
from src.sentinel.sentinel import Sentinel # Import Sentinel
from src.paper_trader import PaperTrader # Import PaperTrader
from src.emails.ares_mailer import AresMailer # Import AresMailer
from src.training.training_manager import TrainingManager
from src.utils.error_handler import get_logged_exceptions
from src.utils.error_handler import (
    handle_errors,
    handle_data_processing_errors,
    handle_file_operations,
    handle_network_operations,
    handle_type_conversions,
    error_context,
    ErrorRecoveryStrategies
)

class Supervisor:
    """
    The Supervisor acts as the high-level risk and performance manager for the system.
    It runs independently, monitoring overall portfolio health and enforcing risk policies
    by adjusting parameters or pausing trading when necessary.
    Now includes a daily scheduled task for error reporting.
    """

    def __init__(self, exchange_client: BinanceExchange, state_manager: StateManager, db_manager):
        self.exchange = exchange_client
        self.state_manager = state_manager
        self.db_manager = db_manager
        self.logger = system_logger.getChild('Supervisor')
        self.config = settings.get("supervisor", {}) # Corrected config init
        self.global_config = CONFIG
        self.data = self.load_data() # This might be for backtesting data, not live operations

        # Initialize AresMailer for sending error reports
        self.ares_mailer = AresMailer(config=self.global_config)

        # These lines were in the original supervisor.py and are re-added.
        # ensemble_orchestrator and data_fetcher are initialized to None to prevent NameErrors,
        # as they are not defined globally in the provided file set.
        self.ensemble_orchestrator: Any = None # Placeholder, assuming it would be set up elsewhere if used
        self.data_fetcher: Any = None # Placeholder, assuming it would be set up elsewhere if used
        self.dynamic_weighter = DynamicWeighter(self.global_config) # Corrected config reference to self.global_config
        # Fixed: Removed 'config=' argument as ModelManager does not accept it
        self.model_manager = ModelManager(firestore_manager=self.db_manager) 
        
        self.prediction_history = pd.DataFrame() # For dynamic weighter

        # State is loaded from file by StateManager's constructor.
        self.state_manager.set_state_if_not_exists("global_peak_equity", settings.get("initial_equity", 10000))
        self.state_manager.set_state_if_not_exists("is_trading_paused", False)
        self.state_manager.set_state_if_not_exists("global_risk_multiplier", 1.0)
        self.state_manager.set_state_if_not_exists("last_retrain_timestamp", datetime.now().isoformat())
        self.state_manager.set_state_if_not_exists("current_position", self.state_manager._get_default_position_structure()) # Ensure default position structure is set

        self.performance_monitor = PerformanceMonitor(config=settings, firestore_manager=self.db_manager)
        
        # Fixed: Explicitly type these as Optional
        self.sentinel: Optional[Sentinel] = None
        self.analyst: Optional[Analyst] = None
        self.strategist: Optional[Strategist] = None
        self.tactician: Optional[Tactician] = None
        
        # Initialize TrainingManager for monthly retraining
        self.training_manager = TrainingManager(self.db_manager)

        # Determine the actual trading client (PaperTrader or live exchange_client)
        if settings.trading_environment == "PAPER":
            self.trader: Union[PaperTrader, BinanceExchange] = PaperTrader(initial_equity=settings.initial_equity) # Fixed: Union type
            self.logger.info("Paper Trader initialized for simulation.")
        elif settings.trading_environment == "LIVE":
            self.trader = exchange_client # Use the live exchange client passed from main
            self.logger.info("Live Trader (BinanceExchange) initialized for live operations.")
        else:
            self.trader = None # Fixed: Explicitly allow None if environment is invalid
            self.logger.error(f"Unknown trading environment: '{settings.trading_environment}'. Trading will be disabled.")
            raise ValueError(f"Invalid TRADING_ENVIRONMENT: {settings.trading_environment}") # Halt if invalid

        # Initialize the core real-time components, getting instances from ModelManager
        if self.trader:
            # Fixed: Pass self.trader and self.state_manager to constructors
            self.sentinel = Sentinel(self.trader, self.state_manager) 
            self.analyst = self.model_manager.get_analyst() # Get Analyst instance from ModelManager
            self.strategist = self.model_manager.get_strategist() # Get Strategist instance from ModelManager
            # Pass performance_reporter to Tactician
            self.tactician = self.model_manager.get_tactician(performance_reporter=self.performance_monitor) 

            # Ensure the Analyst, Strategist, Tactician instances from ModelManager
            # have their exchange_client and state_manager set if they need it for live ops.
            # This is a critical point for dependency injection.
            # For the training pipeline, these are mostly placeholders.
            if self.analyst: # Fixed: Check if not None
                if self.analyst.exchange is None: self.analyst.exchange = self.trader
                if self.analyst.state_manager is None: self.analyst.state_manager = self.state_manager

            if self.strategist: # Fixed: Check if not None
                if self.strategist.exchange is None: self.strategist.exchange = self.trader
                if self.strategist.state_manager is None: self.strategist.state_manager = self.state_manager

            if self.tactician: # Fixed: Check if not None
                if self.tactician.exchange is None: self.tactician.exchange = self.trader
                if self.tactician.state_manager is None: self.tactician.state_manager = self.state_manager


        else:
            self.logger.critical("Core trading components not initialized due to invalid trading environment.")
            
        # Fixed: Add type annotations for queues
        self.market_data_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self.analysis_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self.signal_queue: asyncio.Queue = asyncio.Queue(maxsize=50)

        self.scheduler = AsyncIOScheduler() # Corrected class name

    def start_background_tasks(self):
        """
        This method starts a background scheduler that will trigger the
        run_daily_tasks method every 24 hours, ensuring that model weights
        are periodically and automatically adjusted and error reports are sent.
        """
        # Ensure the job is added only once
        if not self.scheduler.get_job('daily_tasks_job'):
            self.scheduler.add_job(self.run_daily_tasks, 'interval', days=1, id='daily_tasks_job', next_run_time=datetime.now()) # Run immediately on startup, then daily
            self.scheduler.start()
            self.logger.info("Supervisor background task scheduler started.")
        else:
            self.logger.info("Daily tasks job already scheduled.")


    def stop_background_tasks(self):
        """Stops the background scheduler gracefully."""
        self.logger.info("Stopping Supervisor background task scheduler.")
        self.scheduler.shutdown()

    @handle_file_operations(
        default_return=pd.DataFrame(),
        context="load_data"
    )
    def load_data(self) -> pd.DataFrame: # Fixed: Return type hint
        """Loads historical data for backtesting and analysis. (Placeholder for live system)"""
        try:
            # In a live system, this data might not be needed by Supervisor directly
            # or would be fetched/streamed. This is primarily for backtesting context.
            data_path = self.global_config.get("data_path", "data/historical/BTC_USDT-1h.csv")
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                self.logger.info(f"Successfully loaded historical data from {data_path}.")
                return df
            else:
                self.logger.warning(f"Data file not found at {data_path}. Backtesting features will be unavailable.")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error loading historical data in Supervisor: {e}", exc_info=True)
            return pd.DataFrame()

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="supervisor_start"
    )
    async def start(self):
        """Starts the main supervisor monitoring loop, ensuring state is recovered on startup."""
        self.logger.info("Supervisor starting up...")
        
        # On startup, synchronize with the exchange to recover any active state.
        self.logger.info("Attempting to synchronize state with exchange on startup...")
        await self._synchronize_exchange_state()
        self.logger.info("Initial state synchronization complete.")

        # Start background tasks for daily operations (dynamic weighting, error reports)
        self.start_background_tasks()

        self.logger.info("Supervisor started. Monitoring overall system performance and risk.")
        
        check_interval = self.config.get("check_interval_seconds", 300)
        retrain_interval_days = self.global_config.get('supervisor', {}).get("retrain_interval_days", 30)

        while True:
            try:
                await asyncio.sleep(check_interval)
                self.logger.info("--- Running Supervisor Health Check ---")
                
                # Periodically synchronize state with the exchange
                await self._synchronize_exchange_state()
                
                await self._check_performance_and_risk()

                current_equity = self.state_manager.get_state("account_equity")
                peak_equity = self.state_manager.get_state("global_peak_equity") # Use global_peak_equity
                
                # Ensure equity values are valid before calculating drawdown
                if current_equity is None or peak_equity is None or peak_equity <= 0:
                    self.logger.warning("Equity data missing or invalid for performance monitoring. Skipping.")
                    live_metrics = {} # Provide empty metrics if data is bad
                else:
                    # Calculate live metrics based on state manager data
                    # These are simplified for the supervisor's view
                    live_metrics = {
                        'Final Equity': current_equity,
                        'Max Drawdown (%)': ((peak_equity - current_equity) / peak_equity * 100) if peak_equity > 0 else 0,
                        'Total Trades': self.state_manager.get_state("total_trades", 0),
                        'Profit Factor': self.state_manager.get_state("live_profit_factor", 0),
                        'Sharpe Ratio': self.state_manager.get_state("live_sharpe_ratio", 0),
                        'Win Rate (%)': self.state_manager.get_state("live_win_rate", 0)
                    }
                
                await self.performance_monitor.monitor_performance(live_metrics)
                await self._check_for_retraining(retrain_interval_days)
                await self.run_daily_profit_sweep() # Add daily profit sweep to main loop

            except asyncio.CancelledError:
                self.logger.info("Supervisor task cancelled.")
                break
            except Exception as e:
                self.logger.error(f"An error occurred in the Supervisor loop: {e}", exc_info=True)

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="run_daily_tasks"
    )
    async def run_daily_tasks(self):
        """
        Daily task for dynamic weight adjustment and error reporting.
        This function is intended to be run once per day.
        """
        self.logger.info("Running daily supervisor tasks...")
        
        # 1. Run Dynamic Weight Adjustment (if ensemble_orchestrator is available)
        # In a live system, self.ensemble_orchestrator would be passed from ModelManager
        # or initialized here if Supervisor is the orchestrator of all models.
        # For now, we'll assume it's set up if dynamic weighting is active.
        if self.ensemble_orchestrator: # Placeholder, ensure ensemble_orchestrator is properly initialized in live mode
            try:
                self.dynamic_weighter.run_daily_adjustment(self.ensemble_orchestrator, self.prediction_history)
                new_weights = self.ensemble_orchestrator.get_current_weights()
                self.model_manager.save_ensemble_weights(new_weights)
            except Exception as e:
                self.logger.error(f"Error during daily dynamic weight adjustment: {e}", exc_info=True)
        else:
            self.logger.warning("Ensemble Orchestrator not available. Skipping dynamic weight adjustment.")

        # 2. Run Daily Error Report
        await self._run_daily_error_report()
        
        self.logger.info("Daily tasks complete.")

    @handle_errors(
        exceptions=(KeyError, TypeError, ValueError),
        default_return=None,
        context="store_prediction_results"
    )
    def _store_prediction_results(self, asset: str, prediction_output: Dict[str, Any], actual_outcome: Any): # Fixed: Type hints
        """Appends prediction results to the history for the weighter to use."""
        new_record = {
            'timestamp': pd.Timestamp.now(tz='UTC'),
            'asset': asset,
            'regime': prediction_output.get('regime'),
            'final_prediction': prediction_output.get('prediction'),
            'actual': actual_outcome,
            **prediction_output.get('base_predictions', {})
        }
        # Ensure prediction_history is initialized and is a DataFrame
        if self.prediction_history.empty:
            self.prediction_history = pd.DataFrame([new_record])
        else:
            self.prediction_history = pd.concat([self.prediction_history, pd.DataFrame([new_record])], ignore_index=True)


    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="run_daily_error_report"
    )
    async def _run_daily_error_report(self):
        """
        Fetches all logged exceptions and sends a summary email.
        """
        self.logger.info("Generating daily error report...")
        try:
            # Fetch exceptions logged since the last report (or within a recent window)
            # For simplicity, let's fetch all logged exceptions for now.
            # In a real system, you might track the last report timestamp and fetch new ones.
            recent_errors = get_logged_exceptions(limit=50) # Fetch up to 50 recent errors

            if not recent_errors:
                self.logger.info("No new errors to report today.")
                await self.ares_mailer.send_alert(
                    f"Ares Daily Error Report - {datetime.now().strftime('%Y-%m-%d')} (No New Errors)",
                    "No new errors were detected in the Ares trading bot since the last report."
                )
                return

            error_summary = {}
            for error in recent_errors:
                error_type = error.get('type', 'UnknownError')
                error_summary[error_type] = error_summary.get(error_type, 0) + 1
            
            report_body = f"Ares Daily Error Report - {datetime.now().strftime('%Y-%m-%d')}\n\n"
            report_body += f"Total new errors detected: {len(recent_errors)}\n\n"
            report_body += "Summary by Error Type:\n"
            for err_type, count in error_summary.items():
                report_body += f"- {err_type}: {count} occurrences\n"
            
            report_body += "\n--- Last 5 Unique Error Messages (for context) ---\n"
            unique_messages = []
            for error in reversed(recent_errors): # Get most recent unique messages
                msg = error.get('message', 'N/A')
                if msg not in unique_messages:
                    unique_messages.append(msg)
                    report_body += f"\nType: {error.get('type')}\nMessage: {msg}\n"
                if len(unique_messages) >= 5:
                    break
            
            report_subject = f"Ares Daily Error Report - {datetime.now().strftime('%Y-%m-%d')} ({len(recent_errors)} Errors)"
            await self.ares_mailer.send_alert(report_subject, report_body)
            self.logger.info("Daily error report email sent successfully.")

            # Optional: Clear the error log file after reporting to avoid reporting old errors repeatedly
            # Or implement a more sophisticated tracking of reported errors.
            # with open(self.ares_mailer.error_log_file, 'w') as f:
            #     f.write("") # Clear file

        except Exception as e:
            self.logger.error(f"Failed to generate or send daily error report: {e}", exc_info=True)

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="synchronize_exchange_state"
    )
    async def _synchronize_exchange_state(self):
        """
        Fetches the current account equity and open positions from the exchange
        and updates the persistent state. This is key for crash recovery.
        """
        try:
            # 1. Update account equity and peak equity
            account_info = await self.exchange.get_account_info()
            current_equity = float(account_info.get('totalWalletBalance', 0))
            
            if current_equity > 0:
                self.state_manager.set_state("account_equity", current_equity)
                self.logger.debug(f"Updated account equity: ${current_equity:,.2f}")

                peak_equity = self.state_manager.get_state("global_peak_equity") # Use global_peak_equity from state
                if current_equity > peak_equity:
                    self.state_manager.set_state("global_peak_equity", current_equity)
                    self.logger.info(f"New peak equity reached: ${current_equity:,.2f}")
            else:
                self.logger.warning("Could not retrieve a valid account balance.")

            # 2. Update open positions state for crash recovery
            open_positions = await self.exchange.get_open_positions()
            symbol = settings.trade_symbol # Use settings.trade_symbol
            active_position_on_exchange = None
            
            for position in open_positions:
                if position.get('symbol') == symbol and float(position.get('positionAmt', 0)) != 0:
                    # Capture more details for active_position
                    active_position_on_exchange = {
                        "symbol": position['symbol'],
                        "amount": float(position.get('positionAmt', 0)),
                        "entry_price": float(position.get('entryPrice', 0)),
                        "leverage": int(position.get('leverage', 1)),
                        "direction": "LONG" if float(position.get('positionAmt', 0)) > 0 else "SHORT",
                        "trade_id": self.state_manager.get_state("current_position", {}).get("trade_id"),
                        "entry_timestamp": self.state_manager.get_state("current_position", {}).get("entry_timestamp"),
                        "stop_loss": self.state_manager.get_state("current_position", {}).get("stop_loss"),
                        "take_profit": self.state_manager.get_state("current_position", {}).get("take_profit"),
                        "entry_fees_usd": self.state_manager.get_state("current_position", {}).get("entry_fees_usd", 0.0),
                        "entry_context": self.state_manager.get_state("current_position", {}).get("entry_context", {})
                    }
                    self.logger.debug(f"Found active position on exchange for {symbol}.")
                    break 

            # Synchronize the state file with what's on the exchange
            current_state_position = self.state_manager.get_state('current_position') # Use 'current_position'
            
            # Only update if there's a meaningful change or new position found
            if active_position_on_exchange != current_state_position:
                self.logger.info(f"State mismatch or update: Synchronizing position state with exchange. New state: {active_position_on_exchange}")
                self.state_manager.set_state('current_position', active_position_on_exchange) # Update 'current_position'

        except Exception as e:
            self.logger.error(f"Failed to synchronize state with exchange: {e}", exc_info=True)

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="check_performance_and_risk"
    )
    async def _check_performance_and_risk(self):
        """Calculates drawdown and adjusts risk parameters or pauses trading if necessary."""
        peak_equity = self.state_manager.get_state("global_peak_equity")
        current_equity = self.state_manager.get_state("account_equity")

        if not peak_equity or not current_equity or peak_equity == 0:
            self.logger.warning("Cannot check performance; equity data is missing or peak equity is zero.")
            return

        drawdown = (peak_equity - current_equity) / peak_equity
        self.logger.info(f"Current Drawdown: {drawdown:.2%}")

        pause_threshold = self.config.get("pause_trading_drawdown_pct", 0.20)
        risk_reduction_threshold = self.config.get("risk_reduction_drawdown_pct", 0.10)

        if drawdown >= pause_threshold:
            if not self.state_manager.get_state("is_trading_paused"):
                await self._pause_trading(f"Drawdown of {drawdown:.2%} exceeded pause threshold of {pause_threshold:.2%}")
            return

        if drawdown >= risk_reduction_threshold:
            new_risk_multiplier = 0.5
            if self.state_manager.get_state("global_risk_multiplier") != new_risk_multiplier:
                self.logger.warning(f"Drawdown of {drawdown:.2%} exceeded risk reduction threshold. Reducing risk multiplier to {new_risk_multiplier}.")
                self.state_manager.set_state("global_risk_multiplier", new_risk_multiplier)
        else:
            if self.state_manager.get_state("global_risk_multiplier") != 1.0:
                self.logger.info("Performance has recovered. Restoring global risk multiplier to 1.0.")
                self.state_manager.set_state("global_risk_multiplier", 1.0)
        
        if self.state_manager.get_state("is_trading_paused") and drawdown < pause_threshold * 0.9: # Add a buffer for resuming
            await self._resume_trading()

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="pause_trading"
    )
    async def _pause_trading(self, reason: str):
        """Pauses all new trading activity."""
        self.logger.critical(f"PAUSING ALL TRADING. Reason: {reason}")
        self.state_manager.set_state("is_trading_paused", True)

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="resume_trading"
    )
    async def _resume_trading(self):
        """Resumes trading activity."""
        self.logger.info("Resuming trading activity. Drawdown has recovered to an acceptable level.")
        self.state_manager.set_state("is_trading_paused", False)

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="check_for_retraining"
    )
    async def _check_for_retraining(self, retrain_interval_days: int):
        """Checks if it's time to trigger a system retraining."""
        last_retrain_timestamp_str = self.state_manager.get_state("last_retrain_timestamp")
        try:
            last_retrain_datetime = datetime.fromisoformat(last_retrain_timestamp_str)
        except (TypeError, ValueError):
            self.logger.warning("Invalid last_retrain_timestamp in state. Resetting to now.")
            last_retrain_datetime = datetime.now()
            self.state_manager.set_state("last_retrain_timestamp", last_retrain_datetime.isoformat())

        if datetime.now() >= last_retrain_datetime + timedelta(days=retrain_interval_days):
            self.logger.info(f"Retraining interval of {retrain_interval_days} days has passed. Triggering system retraining and validation.")
            await self._trigger_retraining()
            self.state_manager.set_state("last_retrain_timestamp", datetime.now().isoformat())
        else:
            next_retrain_due = last_retrain_datetime + timedelta(days=retrain_interval_days)
            time_until = next_retrain_due - datetime.now()
            self.logger.info(f"Next system retraining due in: {time_until.days} days, {time_until.seconds // 3600} hours.")


    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="trigger_retraining"
    )
    async def _trigger_retraining(self):
        """Triggers the full system retraining and validation pipeline using the TrainingManager."""
        self.logger.info("Initiating full system retraining pipeline via TrainingManager...")
        try:
            # Get the current trading symbol from settings
            symbol = settings.trade_symbol
            exchange_name = "BINANCE"  # Default to BINANCE
            
            # Run full training pipeline for the current symbol
            success = await self.training_manager.run_full_training(symbol, exchange_name)
            
            if success:
                self.logger.info(f"Monthly retraining completed successfully for {symbol}")
                # Send email notification of successful retraining
                await self.ares_mailer.send_email(
                    subject=f"Ares Monthly Retraining Complete - {symbol}",
                    body=f"Monthly retraining pipeline completed successfully for {symbol} on {exchange_name}.\n\nTraining session details:\n{self.training_manager.current_training_session}"
                )
            else:
                self.logger.error(f"Monthly retraining failed for {symbol}")
                # Send email notification of failed retraining
                await self.ares_mailer.send_email(
                    subject=f"Ares Monthly Retraining Failed - {symbol}",
                    body=f"Monthly retraining pipeline failed for {symbol} on {exchange_name}.\n\nError details:\n{self.training_manager.current_training_session.get('error', 'Unknown error')}"
                )
                
        except Exception as e:
            self.logger.error(f"Failed to trigger retraining: {e}", exc_info=True)
            self.logger.critical("Automated retraining failed to trigger. Manual intervention may be required.")
            # Send email notification of critical error
            await self.ares_mailer.send_email(
                subject="Ares Monthly Retraining Critical Error",
                body=f"Critical error during monthly retraining:\n{str(e)}"
            )

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="run_daily_profit_sweep"
    )
    async def run_daily_profit_sweep(self):
        """
        Daily profit sweep: Check USDT balance, move 10% of gains to spot/euro,
        and send withdrawal alerts when threshold is reached.
        """
        self.logger.info("Running daily profit sweep...")
        
        try:
            # Get current balances
            perps_balance = await self.trader.get_account_balance('USDT')
            spot_balance = await self.trader.get_spot_balance('USDT')
            euro_balance = await self.trader.get_spot_balance('EUR')
            
            # Load highest historical balance
            highest_balance = self.state_manager.get_state("highest_usdt_balance", 0)
            
            action = ""
            sweep_amount = 0
            
            if perps_balance > highest_balance:
                gain = perps_balance - highest_balance
                sweep_amount = gain * 0.10  # Move 10% of gains
                
                # Move to spot account
                await self.trader.transfer_to_spot('USDT', sweep_amount)
                
                # Update highest balance
                self.state_manager.set_state("highest_usdt_balance", perps_balance)
                action = f"Swept {sweep_amount:.2f} USDT to spot account"
                self.logger.info(f"Profit sweep: {action}")
            else:
                action = "No sweep - balance not higher than peak"
                self.logger.info("Profit sweep: No action taken")
            
            # Log daily balances
            await self._log_daily_balances(perps_balance, spot_balance, euro_balance, action)
            
            # Check withdrawal threshold
            total_withdrawable = spot_balance + euro_balance
            withdrawal_threshold = self.config.get("withdrawable_amount_threshold", 1000)  # Default $1000
            
            if total_withdrawable >= withdrawal_threshold:
                await self._send_withdrawal_alert(total_withdrawable, spot_balance, euro_balance)
                
        except Exception as e:
            self.logger.error(f"Error during daily profit sweep: {e}", exc_info=True)

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="log_daily_balances"
    )
    async def _log_daily_balances(self, perps_balance: float, spot_balance: float, euro_balance: float, action: str):
        """Log daily account balances to CSV file."""
        import csv
        from datetime import datetime
        
        balance_file = "data/daily_balances.csv"
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(balance_file), exist_ok=True)
        
        # Create file with headers if it doesn't exist
        file_exists = os.path.exists(balance_file)
        
        with open(balance_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            if not file_exists:
                writer.writerow(['Date', 'Perps_USDT', 'Spot_USDT', 'Spot_EUR', 'Action'])
            
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d'),
                f"{perps_balance:.2f}",
                f"{spot_balance:.2f}",
                f"{euro_balance:.2f}",
                action
            ])
        
        self.logger.info(f"Daily balances logged to {balance_file}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="send_withdrawal_alert"
    )
    async def _send_withdrawal_alert(self, total_withdrawable: float, spot_balance: float, euro_balance: float):
        """Send email alert for manual withdrawal."""
        subject = "Ares: Manual Withdrawal Alert"
        body = f"""
        Withdrawal threshold reached!
        
        Total withdrawable amount: ${total_withdrawable:.2f}
        - Spot USDT: ${spot_balance:.2f}
        - Spot EUR: ${euro_balance:.2f}
        
        Please manually withdraw these funds from your exchange account.
        """
        
        try:
            await self.ares_mailer.send_alert(subject, body)
            self.logger.info(f"Withdrawal alert sent for ${total_withdrawable:.2f}")
        except Exception as e:
            self.logger.error(f"Failed to send withdrawal alert: {e}", exc_info=True)
