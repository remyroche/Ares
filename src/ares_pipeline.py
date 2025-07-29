# src/ares_pipeline.py

import asyncio
import datetime
import os
import signal
import sys
import pandas as pd

from src.config import CONFIG, settings
from src.database.firestore_manager import FirestoreManager
from src.exchange.binance import BinanceExchange
from src.supervisor.supervisor import Supervisor as LocalSupervisor
from src.utils.logger import logger
from src.utils.state_manager import StateManager
from src.utils.model_manager import ModelManager
from src.analyst.analyst import Analyst
from src.strategist.strategist import Strategist
from src.tactician.tactician import Tactician
from src.sentinel.sentinel import Sentinel
from src.emails.ares_mailer import send_email


class AresPipeline:
    """
    The core trading pipeline that analyzes market data, generates signals,
    and executes trades, adhering to both local and global risk controls.
    """
    def __init__(self, exchange_client: BinanceExchange, state_manager: StateManager, firestore_manager: FirestoreManager, supervisor: LocalSupervisor):
        self.exchange = exchange_client
        self.state_manager = state_manager
        self.firestore_manager = firestore_manager
        self.supervisor = supervisor
        self.logger = logger.getChild(f"AresPipeline-{self.exchange.symbol}")
        self.config = CONFIG
        
        # Initialize core logic and safety components
        self.model_manager = ModelManager(firestore_manager=self.firestore_manager)
        self.analyst = self.model_manager.get_analyst()
        self.strategist = self.model_manager.get_strategist()
        self.tactician = self.model_manager.get_tactician()
        self.sentinel = Sentinel(self.config)

        self.last_sentinel_check_time = datetime.datetime.min

    async def start(self):
        """Starts the main trading loop for the pipeline."""
        self.logger.info(f"Ares Pipeline for {self.exchange.symbol} starting up...")
        self._write_pid_file()
        self._setup_signal_handlers()
        await self._initial_setup()

        loop_interval = self.config['pipeline'].get("loop_interval_seconds", 10)
        self.logger.info(f"Ares Pipeline for {self.exchange.symbol} is now running.")

        while True:
            try:
                await asyncio.sleep(loop_interval)
                await self._execute_main_cycle()
            except asyncio.CancelledError:
                self.logger.info("Ares Pipeline task cancelled.")
                break
            except Exception as e:
                self.logger.error(f"An unhandled error occurred in the AresPipeline loop: {e}", exc_info=True)

    async def _initial_setup(self):
        """Performs initial setup for the pipeline."""
        self.logger.info("Performing initial setup...")
        if self.supervisor.data.empty:
            self.logger.warning("No historical data available from supervisor for initial setup.")
        else:
            self.analyst.data = self.supervisor.data.copy()
            self.logger.info("Analyst has been provided with historical data.")
        self.logger.info("Initial setup complete.")

    async def _execute_main_cycle(self):
        """The main operational loop of the Ares system."""
        
        await self._check_for_model_promotion()
        
        global_status = self.state_manager.get_state("global_trading_status", "RUNNING")
        is_locally_paused = self.state_manager.get_state("is_trading_paused", False)

        if global_status == "PAUSED" or is_locally_paused:
            status_origin = "Global Manager" if global_status == "PAUSED" else "Local Supervisor"
            self.logger.warning(f"Trading is PAUSED by {status_origin}. Checking for open positions to close.")
            await self._handle_pause_state()
            return

        try:
            klines = await self.exchange.get_latest_klines()
            if klines.empty:
                self.logger.warning("Received empty klines data, skipping analysis cycle.")
                return
        except Exception as e:
            self.logger.error(f"Failed to fetch latest klines: {e}")
            return
        
        intelligence = self.analyst.get_intelligence(klines)
        strategy_decision = self.strategist.generate_strategy(intelligence)
        
        # --- Champion Model Logic ---
        champion_signal = self.tactician.get_signal(strategy_decision)
        self.sentinel.run_checks(intelligence, self.state_manager.get_state("active_position"), champion_signal)
        if self.sentinel.is_trade_halted():
            self.logger.critical(f"SENTINEL HALT (Champion): {self.sentinel.get_halt_reason()}.")
        elif champion_signal:
            await self._execute_trade_action(champion_signal, klines, self.exchange, self.tactician)

        # --- A/B Testing Logic for Challenger Model ---
        if hasattr(self.supervisor, 'ab_tester') and self.supervisor.ab_tester.ab_test_active:
            challenger_tactician = self.supervisor.ab_tester.challenger_tactician
            paper_trader = self.supervisor.ab_tester.challenger_paper_trader
            
            challenger_signal = challenger_tactician.get_signal(strategy_decision)
            # Run sentinel for challenger on its paper-traded position
            self.sentinel.run_checks(intelligence, paper_trader.get_position(), challenger_signal)
            if self.sentinel.is_trade_halted():
                 self.logger.warning(f"SENTINEL HALT (Challenger): {self.sentinel.get_halt_reason()}.")
            elif challenger_signal:
                await self._execute_trade_action(challenger_signal, klines, paper_trader, challenger_tactician)

    async def _execute_trade_action(self, signal, klines, client, tactician_instance):
        """Executes a trade signal on a given client (live or paper)."""
        active_position = client.get_position(self.exchange.symbol) if hasattr(client, 'get_position') else self.state_manager.get_state("active_position")

        if active_position and signal['action'] == 'ENTRY':
            self.logger.info(f"Client is already in a position for {self.exchange.symbol}. Skipping new entry signal.")
            return

        trade_size_usd = await self._calculate_trade_size()
        if trade_size_usd > 0 and signal['action'] == 'ENTRY':
            self.logger.info(f"Placing trade for {self.exchange.symbol} with size ${trade_size_usd:,.2f} on client {type(client).__name__}")
            try:
                current_price = klines['close'].iloc[-1]
                quantity = trade_size_usd / current_price
                
                order_response = await client.place_order(
                    symbol=self.exchange.symbol, side=signal['side'], order_type='MARKET',
                    quantity=round(quantity, self.exchange.qty_precision)
                )
                if order_response:
                    self.logger.info(f"Successfully placed trade: {order_response}")
                    if client == self.exchange: # If live trade, sync state
                        await self.supervisor._synchronize_exchange_state()
            except Exception as e:
                self.logger.error(f"Error placing trade for {self.exchange.symbol}: {e}", exc_info=True)
        elif signal['action'] == 'EXIT':
             # Logic to close a position
             pass

    async def _handle_pause_state(self):
        """Checks if there is an open position and closes it if the system is paused."""
        active_position = self.state_manager.get_state("active_position")
        if active_position:
            self.logger.critical(f"System is paused. Closing open position for {active_position['symbol']}.")
            try:
                await self.exchange.close_position(symbol=active_position['symbol'])
                self.logger.info(f"Close order for {active_position['symbol']} placed successfully.")
                await self.supervisor._synchronize_exchange_state()
            except Exception as e:
                self.logger.error(f"Failed to close position for {active_position['symbol']} during pause state: {e}", exc_info=True)

    async def _calculate_trade_size(self) -> float:
        """Calculates trade size in USD, respecting global and local risk limits."""
        risk_params = self.state_manager.get_state("global_risk_params", {})
        max_allocation_per_pair = risk_params.get("max_allocation_per_pair_usd", 0)
        if max_allocation_per_pair <= 0: return 0.0

        available_balance = self.state_manager.get_state("account_balance", 0)
        leverage = self.state_manager.get_state("leverage", 1)
        
        max_size_by_balance = available_balance * leverage
        trade_size = min(max_allocation_per_pair, max_size_by_balance)
        
        if trade_size <= 0: return 0.0
        self.logger.info(f"Calculated trade size: ${trade_size:,.2f}")
        return trade_size

    async def _check_for_model_promotion(self):
        """Checks for a flag to promote a challenger model to champion."""
        flag_file = self.config.get("PROMOTE_CHALLENGER_FLAG_FILE")
        if flag_file and os.path.exists(flag_file):
            self.logger.critical("Promote challenger flag detected! Initiating model hot-swap...")
            if self.model_manager.promote_challenger_to_champion():
                self.logger.info("Model promotion successful. New champion is live.")
                send_email("Ares Alert: Model Promoted", f"Challenger for {self.exchange.symbol} is now champion.")
                self.analyst = self.model_manager.get_analyst()
                self.strategist = self.model_manager.get_strategist()
                self.tactician = self.model_manager.get_tactician()
            else:
                self.logger.error("Model promotion FAILED.")
                send_email("Ares Alert: Model Promotion FAILED", f"Failed for {self.exchange.symbol}.")
            os.remove(flag_file)

    def _write_pid_file(self):
        """Writes the process ID to a file for monitoring."""
        pid_file = self.config.get('PIPELINE_PID_FILE', f'ares_{self.exchange.symbol}.pid')
        try:
            with open(pid_file, 'w') as f:
                f.write(str(os.getpid()))
        except Exception as e:
            self.logger.error(f"Error writing PID file: {e}")

    def _remove_pid_file(self):
        """Removes the PID file on clean shutdown."""
        pid_file = self.config.get('PIPELINE_PID_FILE', f'ares_{self.exchange.symbol}.pid')
        if os.path.exists(pid_file):
            os.remove(pid_file)

    def _handle_graceful_shutdown(self, signum, frame):
        """Handles SIGINT and SIGTERM for a clean shutdown."""
        self.logger.critical(f"Received signal {signum}. Initiating graceful shutdown...")
        self._remove_pid_file()
        # The main_launcher will cancel the asyncio tasks.
        sys.exit(0)

    def _setup_signal_handlers(self):
        """Sets up signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._handle_graceful_shutdown)
        signal.signal(signal.SIGTERM, self._handle_graceful_shutdown)
