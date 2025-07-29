# src/supervisor/supervisor.py

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
import pandas as pd
import numpy as np
import os

from .dynamic_weighter import DynamicWeighter
from src.utils.model_manager import ModelManager # Assuming this exists for saving/loading weights
from src.exchange.binance import BinanceExchange
from src.utils.logger import logger
from src.config import settings, CONFIG
from src.utils.state_manager import StateManager
from src.database.firestore_manager import FirestoreManager
from src.supervisor.performance_monitor import PerformanceMonitor
from backtesting.ares_backtester import Backtester
from src.strategist.strategist import Strategist

class Supervisor:
    """
    The Supervisor acts as the high-level risk and performance manager for the system.
    It runs independently, monitoring overall portfolio health and enforcing risk policies
    by adjusting parameters or pausing trading when necessary.
    """

    def __init__(self, exchange_client: BinanceExchange, state_manager: StateManager, firestore_manager: FirestoreManager):
        """
        Initializes the Supervisor.

        Args:
            exchange_client (BinanceExchange): The exchange client for interacting with the exchange.
            state_manager (StateManager): The state manager for persisting and retrieving system state.
            firestore_manager (FirestoreManager): The Firestore manager for database operations.
        """
        self.exchange = exchange_client
        self.state_manager = state_manager
        self.firestore_manager = firestore_manager
        self.logger = logger.getChild('Supervisor')
        self.config = settings.get("supervisor", {})
        self.global_config = CONFIG
        self.data = self.load_data()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.ensemble_orchestrator = ensemble_orchestrator
        self.data_fetcher = data_fetcher
        self.dynamic_weighter = DynamicWeighter(config)
        self.model_manager = ModelManager(config) # For persisting weights
        
        # In a real system, this would be loaded from a persistent database
        self.prediction_history = pd.DataFrame()

        
        # State is loaded from file by StateManager's constructor.
        # Initialize default values only if they don't exist in the loaded state.
        self.state_manager.set_state_if_not_exists("peak_equity", settings.get("initial_equity", 10000))
        self.state_manager.set_state_if_not_exists("is_trading_paused", False)
        self.state_manager.set_state_if_not_exists("global_risk_multiplier", 1.0)
        self.state_manager.set_state_if_not_exists("last_retrain_timestamp", datetime.now().isoformat())
        self.state_manager.set_state_if_not_exists("active_position", None) # To track live trades

        self.performance_monitor = PerformanceMonitor(config=settings, firestore_manager=self.firestore_manager)
        self.strategist = Strategist(self.global_config)

        self.scheduler = AsyncIOScheduler()

    def start_background_tasks(self):
        """
        This method starts a background scheduler that will trigger the
        run_daily_tasks method every 24 hours, ensuring that model weights
        are periodically and automatically adjusted.
        """
        self.scheduler.add_job(self.run_daily_tasks, 'interval', days=1, id='daily_tasks_job')
        self.scheduler.start()
        self.logger.info("Supervisor background task scheduler started.")

    def stop_background_tasks(self):
        """Stops the background scheduler gracefully."""
        self.logger.info("Stopping Supervisor background task scheduler.")
        self.scheduler.shutdown()

    def load_data(self):
        """Loads historical data for backtesting and analysis."""
        try:
            # This assumes a single data file for now, can be parameterized
            data_path = self.global_config.get("data_path", "data/historical/BTC_USDT-1h.csv")
            df = pd.read_csv(data_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            self.logger.info(f"Successfully loaded historical data from {data_path}.")
            return df
        except FileNotFoundError:
            self.logger.error(f"Data file not found at {data_path}. Backtesting features will be unavailable.")
            return pd.DataFrame()

    async def start(self):
        """Starts the main supervisor monitoring loop, ensuring state is recovered on startup."""
        self.logger.info("Supervisor starting up...")
        
        # On startup, synchronize with the exchange to recover any active state.
        self.logger.info("Attempting to synchronize state with exchange on startup...")
        await self._synchronize_exchange_state()
        self.logger.info("Initial state synchronization complete.")

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
                peak_equity = self.state_manager.get_state("peak_equity")
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

            except asyncio.CancelledError:
                self.logger.info("Supervisor task cancelled.")
                break
            except Exception as e:
                self.logger.error(f"An error occurred in the Supervisor loop: {e}", exc_info=True)

    def run_daily_tasks(self):
        """
        Daily task for dynamic weight adjustment.
        This function is intended to be run once per day to adapt the model
        weights based on their performance over the last week.
        """
        self.logger.info("Running daily supervisor tasks...")
        
        # 1. Run Dynamic Weight Adjustment
        self.dynamic_weighter.run_daily_adjustment(self.ensemble_orchestrator, self.prediction_history)
        
        # 2. Persist the new weights
        new_weights = self.ensemble_orchestrator.get_current_weights()
        self.model_manager.save_ensemble_weights(new_weights)
        
        self.logger.info("Daily tasks complete.")

    def _store_prediction_results(self, asset, prediction_output, actual_outcome):
        """Appends prediction results to the history for the weighter to use."""
        new_record = {
            'timestamp': pd.Timestamp.now(tz='UTC'),
            'asset': asset,
            'regime': prediction_output.get('regime'),
            'final_prediction': prediction_output.get('prediction'),
            'actual': actual_outcome,
            **prediction_output.get('base_predictions', {})
        }
        self.prediction_history = pd.concat([self.prediction_history, pd.DataFrame([new_record])], ignore_index=True)


    
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

                peak_equity = self.state_manager.get_state("peak_equity")
                if current_equity > peak_equity:
                    self.state_manager.set_state("peak_equity", current_equity)
                    self.logger.info(f"New peak equity reached: ${current_equity:,.2f}")
            else:
                self.logger.warning("Could not retrieve a valid account balance.")

            # 2. Update open positions state for crash recovery
            open_positions = await self.exchange.get_open_positions()
            symbol = self.global_config.get('trading', {}).get('symbol', 'BTCUSDT')
            active_position_on_exchange = None
            
            for position in open_positions:
                # Find the position for the symbol we are trading
                if position.get('symbol') == symbol and float(position.get('positionAmt', 0)) != 0:
                    active_position_on_exchange = {
                        "symbol": position['symbol'],
                        "amount": float(position['positionAmt']),
                        "entry_price": float(position['entryPrice']),
                        "leverage": int(position.get('leverage', 1)),
                        "side": "long" if float(position['positionAmt']) > 0 else "short"
                    }
                    self.logger.debug(f"Found active position on exchange for {symbol}.")
                    break 

            # Synchronize the state file with what's on the exchange
            current_state_position = self.state_manager.get_state('active_position')
            
            if active_position_on_exchange != current_state_position:
                self.logger.info(f"State mismatch or update: Synchronizing position state with exchange. New state: {active_position_on_exchange}")
                self.state_manager.set_state('active_position', active_position_on_exchange)

        except Exception as e:
            self.logger.error(f"Failed to synchronize state with exchange: {e}", exc_info=True)


    async def _check_performance_and_risk(self):
        """Calculates drawdown and adjusts risk parameters or pauses trading if necessary."""
        peak_equity = self.state_manager.get_state("peak_equity")
        current_equity = self.state_manager.get_state("account_equity")

        if not peak_equity or not current_equity or peak_equity == 0:
            self.logger.warning("Cannot check performance; equity data is missing.")
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
        
        if self.state_manager.get_state("is_trading_paused") and drawdown < pause_threshold:
            await self._resume_trading()

    async def _pause_trading(self, reason: str):
        """Pauses all new trading activity."""
        self.logger.critical(f"PAUSING ALL TRADING. Reason: {reason}")
        self.state_manager.set_state("is_trading_paused", True)

    async def _resume_trading(self):
        """Resumes trading activity."""
        self.logger.info("Resuming trading activity. Drawdown has recovered to an acceptable level.")
        self.state_manager.set_state("is_trading_paused", False)

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

    async def _trigger_retraining(self):
        """Triggers the full system retraining and validation pipeline."""
        self.logger.info("Initiating full system retraining pipeline...")
        # Placeholder for actual model training logic
        self.logger.info("System retraining complete (placeholder). Starting validation.")
        
        # Validate the new model using walk-forward and Monte Carlo analysis
        await self.run_walk_forward_analysis()
        await self.run_monte_carlo_simulation()

    async def run_walk_forward_analysis(self, train_months=6, test_months=2):
        """Performs walk-forward analysis of the current strategy."""
        if self.data.empty:
            self.logger.error("Cannot run walk-forward analysis: No historical data.")
            return

        self.logger.info("Starting Walk-Forward Analysis...")
        wfa_config = self.global_config.get('walk_forward_analysis', {})
        train_period = pd.DateOffset(months=wfa_config.get('train_months', train_months))
        test_period = pd.DateOffset(months=wfa_config.get('test_months', test_months))
        
        start_date = self.data.index.min()
        end_date = self.data.index.max()
        
        current_start = start_date
        fold = 1
        all_results = []

        while current_start + train_period + test_period <= end_date:
            train_end = current_start + train_period
            test_end = train_end + test_period
            
            train_set = self.data.loc[current_start:train_end]
            test_set = self.data.loc[train_end:test_end]
            
            self.logger.info(f"Fold {fold}: Training on {len(train_set)} candles, Testing on {len(test_set)} candles.")
            
            # In a real system, you would retrain your model on `train_set` here.
            # For now, we use the same strategy instance.
            strategy_config = self.global_config.get('default_strategy', {})
            strategy = self.strategist.get_strategy_by_name(strategy_config.get("strategy"))

            if not strategy:
                self.logger.error("Walk-forward analysis failed: Strategy not found.")
                return

            backtester = Backtester(
                data=test_set.copy(),
                strategy=strategy,
                leverage=strategy_config.get("leverage", 1),
                fee_rate=self.global_config.get('backtesting', {}).get('fee_rate', 0.0005)
            )
            results = backtester.run()
            self.logger.info(f"Fold {fold} Results: {results}")
            all_results.append(results)
            
            current_start += test_period
            fold += 1
        
        self.logger.info(f"Walk-Forward Analysis complete. Total folds: {len(all_results)}")
        # Here you would aggregate and analyze `all_results`
        return all_results

    async def run_monte_carlo_simulation(self, num_simulations=1000):
        """Runs a Monte Carlo simulation on the strategy's historical performance."""
        if self.data.empty:
            self.logger.error("Cannot run Monte Carlo simulation: No historical data.")
            return
            
        self.logger.info(f"Starting Monte Carlo Simulation ({num_simulations} iterations)...")
        
        # First, run a full backtest to get the sequence of trades
        strategy_config = self.global_config.get('default_strategy', {})
        strategy = self.strategist.get_strategy_by_name(strategy_config.get("strategy"))
        if not strategy:
            self.logger.error("Monte Carlo simulation failed: Strategy not found.")
            return

        backtester = Backtester(
            data=self.data.copy(),
            strategy=strategy,
            leverage=strategy_config.get("leverage", 1),
            fee_rate=self.global_config.get('backtesting', {}).get('fee_rate', 0.0005)
        )
        backtester.run()
        trades = backtester.trades
        
        if not trades:
            self.logger.warning("No trades were generated in the backtest. Cannot run Monte Carlo simulation.")
            return

        trade_pnls = [trade['pnl'] for trade in trades]
        final_equities = []

        for i in range(num_simulations):
            np.random.shuffle(trade_pnls)
            equity = settings.get("initial_equity", 10000)
            for pnl in trade_pnls:
                equity *= (1 + pnl)
            final_equities.append(equity)
        
        mean_equity = np.mean(final_equities)
        std_dev_equity = np.std(final_equities)
        percentile_5 = np.percentile(final_equities, 5)
        
        self.logger.info("Monte Carlo Simulation Complete.")
        self.logger.info(f"  - Average Final Equity: ${mean_equity:,.2f}")
        self.logger.info(f"  - Std Dev of Final Equity: ${std_dev_equity:,.2f}")
        self.logger.info(f"  - 5th Percentile of Final Equity: ${percentile_5:,.2f}")
        
        return {"mean": mean_equity, "std_dev": std_dev_equity, "percentile_5": percentile_5}
