# src/ares_pipeline.py
import time
import datetime
import os
import sys
import signal
import pandas as pd
import numpy as np
import asyncio

from src.config import CONFIG
from src.utils.logger import system_logger
from src.emails.ares_mailer import send_email
from src.supervisor.main import Supervisor
from src.sentinel.sentinel import Sentinel
from src.exchange.binance import BinanceFuturesAPI
from src.database.firestore_manager import FirestoreManager
from src.utils.model_manager import ModelManager
from src.analyst.data_utils import load_klines_data, simulate_order_book_data

class AresPipeline:
    """
    The main orchestration class for a single Ares trading bot instance.
    It manages the real-time loop, inter-module communication, and system state.
    """
    def __init__(self, config=CONFIG):
        self.config = config
        self.logger = system_logger.getChild(f"AresPipeline.{self.config.get('SYMBOL', 'DEFAULT')}")
        
        self.firestore_manager = FirestoreManager(config=self.config)
        self.model_manager = ModelManager(firestore_manager=self.firestore_manager)

        # Modules are now accessed via the ModelManager for hot-swapping
        self.analyst = self.model_manager.analyst
        self.strategist = self.model_manager.strategist
        self.tactician = self.model_manager.tactician
        
        self.supervisor = Supervisor(config, firestore_manager=self.firestore_manager)
        self.sentinel = Sentinel(config)

        live_config = self.config.get("live_trading", {})
        self.live_trading_enabled = live_config.get("enabled", False)
        self.binance_client = BinanceFuturesAPI(
            api_key=live_config.get("api_key"), api_secret=live_config.get("api_secret"),
            testnet=live_config.get("testnet", True), symbol=self.config['SYMBOL'],
            interval=self.config['INTERVAL'], config=self.config
        )
        
        self.current_equity = self.config['INITIAL_EQUITY']
        self.trade_logs_today = []
        self.daily_pnl_per_regime = {}
        self.historical_daily_pnl_data = pd.DataFrame(columns=['Date', 'NetPnL'])
        self.last_strategist_update_time = datetime.datetime.min
        self.last_supervisor_update_time = datetime.datetime.min
        self.last_sentinel_check_time = datetime.datetime.min

    async def _initial_setup(self):
        """Performs initial setup, connecting to Binance and loading historical data."""
        self.logger.info("Initializing system setup...")

        if self.live_trading_enabled:
            self.logger.info("Live trading ENABLED. Connecting to Binance API...")
            self.binance_client.start_data_streams()
            if self.config["live_trading"].get("websocket_streams", {}).get("userData", False):
                self.binance_client.start_user_data_stream()
            
            await asyncio.sleep(5) # Allow buffers to populate
            
            account_balance = self.binance_client.get_latest_account_balance()
            if 'USDT' in account_balance:
                self.current_equity = account_balance['USDT']['free'] + account_balance['USDT']['locked']
            
            live_position = self.binance_client.get_latest_position(self.config['SYMBOL'])
            if live_position and live_position.get('positionAmt', 0) != 0:
                self.tactician.current_position = {
                    "symbol": self.config['SYMBOL'],
                    "direction": "LONG" if live_position['positionAmt'] > 0 else "SHORT",
                    "size": abs(live_position['positionAmt']),
                    "entry_price": live_position['entryPrice'],
                    "unrealized_pnl": live_position['unrealizedPnl'],
                    "current_leverage": live_position['leverage'],
                    "ladder_steps": 0, "stop_loss": None, "take_profit": None
                }
                self.logger.info(f"Detected existing live position: {self.tactician.current_position}")
        else:
            self.logger.info("Live trading DISABLED. Using simulated data.")

        self.logger.info("Loading and preparing historical data for Analyst and Strategist...")
        if not await self.analyst.load_and_prepare_historical_data():
            self.logger.error("Analyst failed to load historical data.")
        
        if not self.strategist.load_historical_data_htf():
            self.logger.error("Strategist failed to load HTF data.")
        
        self.logger.info("Initial setup complete.")

    def _get_real_time_market_data(self):
        """Fetches real-time market data from Binance (live) or CSVs (simulated)."""
        self.logger.debug("Fetching real-time market data...")
        try:
            if self.live_trading_enabled:
                klines = self.binance_client.get_latest_klines(num_klines=500)
                agg_trades = self.binance_client.get_latest_agg_trades(num_trades=1000)
                order_book = self.binance_client.get_latest_order_book()
                account_balance = self.binance_client.get_latest_account_balance()
                current_position_live = self.binance_client.get_latest_position(self.config['SYMBOL'])
                
                if klines.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, 0.0, 0.0, 0.0, 0.0
                
                current_price = klines['close'].iloc[-1]
                if 'USDT' in account_balance: self.current_equity = account_balance['USDT']['free'] + account_balance['USDT']['locked']
                
                pos_notional = liq_price = 0.0
                if current_position_live and current_position_live.get('positionAmt', 0) != 0:
                    pos_notional = abs(current_position_live['positionAmt']) * current_price
                    liq_price = current_position_live['liquidationPrice']
                
                return klines, agg_trades, pd.DataFrame(), order_book, current_price, 0.0, pos_notional, liq_price
            else:
                klines = self.analyst.load_and_prepare_historical_data_for_sim().tail(500)
                current_price = klines['close'].iloc[-1] if not klines.empty else 0.0
                order_book = simulate_order_book_data(current_price)
                return klines, pd.DataFrame(), pd.DataFrame(), order_book, current_price, 0.0, 0.0, 0.0
        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}", exc_info=True)
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, 0.0, 0.0, 0.0, 0.0

    async def _execute_trade_action(self, tactician_instance, paper_trader_instance, decision, current_price):
        """Helper to execute a trade decision on either a live client or paper trader."""
        action_type = decision.get("action")
        details = decision.get("details", {})
        symbol = details.get("symbol", self.config['SYMBOL'])
        
        client = self.binance_client if not paper_trader_instance else paper_trader_instance

        if action_type == "ORDER_PLACED":
            order_response = client.place_order(
                symbol=symbol, side=details["direction"], type=details["order_type"],
                quantity=details["quantity"], price=details.get("price")
            )
            # Update position based on response
            if order_response and order_response.get('status') == 'FILLED':
                 tactician_instance._update_position(
                    symbol=symbol, direction=details["direction"], size=details["quantity"],
                    entry_price=current_price, leverage=details["leverage"],
                    stop_loss=details.get("stop_loss"), take_profit=details.get("take_profit"),
                    entry_confidence=details.get("entry_confidence", 0.0), entry_lss=details.get("entry_lss", 0.0)
                )

        elif action_type == "POSITION_CLOSED":
            # For live trading, closing logic is more complex (getting position side, etc.)
            # This is a simplified representation.
            position_to_log = tactician_instance.current_position.copy()
            if not paper_trader_instance: # If live
                await self._log_completed_trade(symbol, position_to_log, current_price, details.get("reason"))
            tactician_instance._close_position(current_price, details.get("reason"))

    async def _log_completed_trade(self, symbol: str, closed_position_info: dict, exit_price: float, exit_reason: str):
        # ... (implementation remains the same) ...
        pass

    def _handle_graceful_shutdown(self, signum, frame):
        self.logger.critical(f"Received signal {signum}. Initiating graceful shutdown...")
        if self.live_trading_enabled:
            self.binance_client.stop_all_streams()
        self._remove_pid_file()
        sys.exit(0)

    def _write_pid_file(self):
        try:
            with open(self.config['PIPELINE_PID_FILE'], 'w') as f:
                f.write(f"{os.getpid()}")
        except Exception as e:
            self.logger.error(f"Error writing PID file: {e}")

    def _remove_pid_file(self):
        if os.path.exists(self.config['PIPELINE_PID_FILE']):
            os.remove(self.config['PIPELINE_PID_FILE'])

    async def run_async(self):
        """The main real-time operational loop of the Ares system."""
        self.logger.info(f"--- Ares Pipeline Starting for {self.config['SYMBOL']} ---")
        self._write_pid_file()
        signal.signal(signal.SIGINT, self._handle_graceful_shutdown)
        signal.signal(signal.SIGTERM, self._handle_graceful_shutdown)

        await self._initial_setup()

        loop_interval = self.config['pipeline'].get("loop_interval_seconds", 10)
        
        while True:
            current_time = datetime.datetime.now()
            
            # --- Check for Hot-Swap Signal ---
            if os.path.exists(self.config.get("PROMOTE_CHALLENGER_FLAG_FILE")):
                self.logger.critical("Promote challenger flag detected! Initiating model hot-swap...")
                if self.model_manager.promote_challenger_to_champion():
                    send_email("Ares Alert: Model Promoted Successfully", "The challenger model is now live.")
                else:
                    send_email("Ares Alert: Model Promotion FAILED", "System continues on the old model.")
                os.remove(self.config.get("PROMOTE_CHALLENGER_FLAG_FILE"))

            # Get latest modules from model manager in case of hot-swap
            self.analyst = self.model_manager.analyst
            self.strategist = self.model_manager.strategist
            self.tactician = self.model_manager.tactician
            
            # --- Main Loop Logic ---
            klines, agg_trades, futures, order_book, current_price, _, pos_notional, liq_price = self._get_real_time_market_data()
            if klines.empty:
                await asyncio.sleep(loop_interval)
                continue

            # Generate intelligence with the champion model
            analyst_intelligence = self.analyst.get_intelligence(klines, agg_trades, futures, order_book, pos_notional, liq_price)
            
            # Get champion's decision
            tactician_decision = await self.tactician.process_intelligence(analyst_intelligence, self.strategist.get_strategist_parameters(0, current_price), {"current_price": current_price, "current_equity": self.current_equity})
            
            # Execute champion's decision (live trading)
            await self._execute_trade_action(self.tactician, None, tactician_decision, current_price)

            # --- A/B Testing Logic ---
            if self.supervisor.ab_tester.ab_test_active:
                challenger_tactician = self.supervisor.ab_tester.challenger_tactician
                paper_trader = self.supervisor.ab_tester.challenger_paper_trader
                
                # The challenger uses the same intelligence but its own tactician/parameters
                challenger_decision = await challenger_tactician.process_intelligence(analyst_intelligence, self.strategist.get_strategist_parameters(0, current_price), {"current_price": current_price, "current_equity": paper_trader.equity})
                
                # Execute challenger's decision on the paper trader
                await self._execute_trade_action(challenger_tactician, paper_trader, challenger_decision, current_price)

            # --- Periodic Supervisor & Sentinel Checks ---
            if (current_time - self.last_supervisor_update_time).total_seconds() >= self.config['pipeline'].get("supervisor_update_interval_minutes", 1440) * 60:
                await self.supervisor.orchestrate_supervision(current_time.date(), self.current_equity, self.trade_logs_today, self.daily_pnl_per_regime, self.historical_daily_pnl_data)
                self.last_supervisor_update_time = current_time

            if (current_time - self.last_sentinel_check_time).total_seconds() >= self.config['pipeline'].get("sentinel_check_interval_seconds", 30):
                self.sentinel.run_checks(analyst_intelligence, {}, tactician_decision) # Simplified for brevity
                self.last_sentinel_check_time = current_time

            await asyncio.sleep(loop_interval)

if __name__ == "__main__":
    # This allows running a single pipeline instance directly for debugging
    # The symbol would be taken from the main config file.
    pipeline = AresPipeline()
    asyncio.run(pipeline.run_async())
