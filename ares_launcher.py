#!/usr/bin/env python3
"""
Ares Comprehensive Launcher

This script provides a unified interface for launching the Ares trading bot with various modes:
1. Paper trading for robust trade information and performance metrics
2. Enhanced backtesting with efficiency optimizations for large datasets (uses existing data)
3. Enhanced model training with efficiency optimizations for large datasets (uses existing data)
4. Live trading for production
5. Portfolio management for multi-token trading
6. HMM regime classification and ML model training

Usage:
    # Paper trading (robust trade info and performance metrics)
    python ares_launcher.py paper --symbol ETHUSDT --exchange BINANCE

    # Challenger paper trading (with challenger model)
    python ares_launcher.py challenger --symbol ETHUSDT --exchange BINANCE

    # Enhanced backtesting with efficiency optimizations (uses existing data)
    python ares_launcher.py backtest --symbol ETHUSDT --exchange BINANCE

    # Enhanced model training with efficiency optimizations (uses existing data)
    python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE

    # Model training with backtesting and paper trading
    python ares_launcher.py model_trainer --symbol ETHUSDT --exchange BINANCE

    # Live trading for single token
    python ares_launcher.py live --symbol ETHUSDT --exchange BINANCE

    # Data loading (klines, aggtrades, futures) without backtesting
    python ares_launcher.py load --symbol ETHUSDT --exchange BINANCE
    python ares_launcher.py load --symbol ETHUSDT --exchange MEXC
    python ares_launcher.py load --symbol ETHUSDT --exchange GATEIO

    # Unified Regime Classifier operations
    python ares_launcher.py regime --regime-subcommand load --symbol ETHUSDT --exchange BINANCE
    python ares_launcher.py regime --regime-subcommand train --symbol ETHUSDT --exchange BINANCE
    python ares_launcher.py regime --regime-subcommand train_blank --symbol ETHUSDT --exchange BINANCE

    # Live trading for all tokens + portfolio manager
    python ares_launcher.py portfolio

    # GUI only
    python ares_launcher.py gui

    # GUI + specific mode
    python ares_launcher.py gui --mode paper --symbol ETHUSDT --exchange BINANCE
"""

import argparse
import asyncio
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Try to import requests for GUI health checks
try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from src.config import CONFIG
from src.utils.comprehensive_logger import (
    setup_comprehensive_logging,
)
from src.utils.error_handler import handle_errors
from src.utils.logger import (
    ensure_comprehensive_logging_available,
)
from src.utils.signal_handler import setup_signal_handlers

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class AresLauncher:
    """Comprehensive launcher for Ares trading bot."""

    def __init__(self):
        # Initialize comprehensive logging
        self.comprehensive_logger = setup_comprehensive_logging(CONFIG)

        # Ensure comprehensive logging is available for all existing logging calls
        ensure_comprehensive_logging_available()

        self.logger = self.comprehensive_logger.get_component_logger("AresLauncher")
        self.global_logger = self.comprehensive_logger.get_global_logger()
        self.processes = []  # Track subprocesses for cleanup
        self.gui_process = None
        self.portfolio_process = None
        self.signal_handler = None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="setup_logging",
    )
    def setup_logging(self):
        """Setup comprehensive logging for the launcher."""
        # Comprehensive logging is already set up in __init__
        # Log launcher startup information
        self.comprehensive_logger.log_launcher_start("INITIALIZATION")

        # Log to both component logger and global logger
        self.logger.info("=" * 80)
        self.logger.info("🚀 ARES COMPREHENSIVE LAUNCHER")
        self.logger.info("=" * 80)
        self.logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Log directory: {self.comprehensive_logger.log_dir}")
        self.logger.info(f"Log level: {CONFIG.get('logging', {}).get('level', 'INFO')}")
        if self.global_logger:
            self.logger.info(
                f"Global log file: ares_global_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            )
        self.logger.info("=" * 80)

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="setup_signal_handling",
    )
    def setup_signal_handling(self):
        """Set up centralized signal handling."""
        self.signal_handler = setup_signal_handlers()
        self.signal_handler.register_shutdown_callback(self.cleanup)
        self.logger.info("✅ Centralized signal handling set up")

    @handle_errors(exceptions=(Exception,), default_return=None, context="cleanup")
    def cleanup(self):
        """Cleanup processes on exit."""
        self.logger.info("🧹 Cleaning up processes...")

        # Terminate GUI process
        if self.gui_process and self.gui_process.poll() is None:
            self.logger.info("🔄 Terminating GUI process...")
            self.gui_process.terminate()
            try:
                self.gui_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.gui_process.kill()

        # Terminate portfolio process
        if self.portfolio_process and self.portfolio_process.poll() is None:
            self.logger.info("🔄 Terminating portfolio process...")
            self.portfolio_process.terminate()
            try:
                self.portfolio_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.portfolio_process.kill()

        # Terminate any other tracked processes
        for process in self.processes:
            if process.poll() is None:
                self.logger.info(f"🔄 Terminating process {process.pid}...")
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()

        self.logger.info("✅ Cleanup completed")

    @handle_errors(exceptions=(Exception,), default_return=False, context="launch_gui")
    def launch_gui(
        self,
        mode: str | None = None,
        symbol: str | None = None,
        exchange: str | None = None,
    ):
        """Launch the GUI server."""
        self.logger.info("🚀 Launching GUI server...")

        # Build the command
        cmd = [sys.executable, "GUI/api_server.py"]

        # Add mode-specific arguments if provided
        if mode and symbol and exchange:
            cmd.extend(["--mode", mode, "--symbol", symbol, "--exchange", exchange])

        self.gui_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.processes.append(self.gui_process)
        self.logger.info(f"✅ GUI server started with PID {self.gui_process.pid}")

        # Wait a moment for the server to start
        time.sleep(2)

        # Check if the server is running
        if self.gui_process.poll() is None:
            self.logger.info("✅ GUI server is running")
            return True
        stdout, stderr = self.gui_process.communicate()
        self.logger.error(f"❌ GUI server failed to start: {stderr}")
        return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="launch_portfolio_manager",
    )
    def launch_portfolio_manager(self):
        """Launch the portfolio manager."""
        self.logger.info("🚀 Launching portfolio manager...")

        self.portfolio_process = subprocess.Popen(
            [sys.executable, "src/supervisor/global_portfolio_manager.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.processes.append(self.portfolio_process)
        self.logger.info(
            f"✅ Portfolio manager started with PID {self.portfolio_process.pid}",
        )
        return True

    def _run_unified_training(
        self,
        symbol: str,
        exchange: str,
        training_mode: str,
        lookback_days: int,
        with_gui: bool = False,
    ):
        """Run unified training with enhanced training manager."""
        # Set environment variable for blank training mode
        import os

        if training_mode == "blank":
            os.environ["BLANK_TRAINING_MODE"] = "1"
            print("🧪 BLANK TRAINING MODE: Set BLANK_TRAINING_MODE=1")

        mode_display = f"{training_mode} training"
        print(f"🚀 Starting {mode_display} for {symbol} on {exchange}")
        self.logger.info(f"🚀 Starting {mode_display} for {symbol} on {exchange}")

        @handle_errors(
            exceptions=(Exception,),
            default_return=False,
            context="enhanced_training_pipeline",
        )
        async def run_enhanced_training():
            """Execute enhanced training using EnhancedTrainingManager with comprehensive error handling."""
            from src.database.sqlite_manager import SQLiteManager
            from src.training.enhanced_training_manager import EnhancedTrainingManager
            from src.utils.logger import system_logger

            logger = system_logger.getChild("EnhancedTrainingPipeline")
            
            logger.info("=" * 80)
            logger.info("🚀 ENHANCED TRAINING PIPELINE START")
            logger.info("=" * 80)
            logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"🎯 Symbol: {symbol}")
            logger.info(f"🏢 Exchange: {exchange}")
            logger.info(f"📊 Training Mode: {training_mode}")
            logger.info(f"📈 Lookback Days: {lookback_days}")
            print("=" * 80)
            print("🚀 ENHANCED TRAINING PIPELINE START")
            print("=" * 80)

            try:
                # Initialize database manager
                logger.info("📊 STEP 0: Initializing Database Manager...")
                print("   📊 Setting up database manager...")
                
                default_config = {
                    "database": {
                        "sqlite_path": "data/ares.db",
                        "backup_enabled": True,
                        "max_connections": 10,
                        "timeout": 30,
                        "check_same_thread": False,
                    },
                }
                
                db_manager = SQLiteManager(default_config)
                await db_manager.initialize()
                logger.info("✅ Database manager initialized successfully")
                print("   ✅ Database manager initialized successfully")

                # Initialize enhanced training manager
                logger.info("🤖 STEP 1: Initializing Enhanced Training Manager...")
                print("   🤖 Initializing enhanced training manager...")
                
                training_config = {
                    "enhanced_training_manager": {
                        "enhanced_training_interval": 3600,
                        "max_enhanced_training_history": 100,
                        "enable_advanced_model_training": True,
                        "enable_ensemble_training": True,
                        "enable_multi_timeframe_training": True,
                        "enable_adaptive_training": True,
                        # Light parameters for blank training mode
                        "blank_training_mode": training_mode == "blank",
                        "max_trials": 3 if training_mode == "blank" else 200,
                        "n_trials": 5 if training_mode == "blank" else 100,
                        "lookback_days": lookback_days,
                    },
                    "database": default_config["database"],
                }

                training_manager = EnhancedTrainingManager(training_config)
                logger.info("✅ Enhanced training manager initialized successfully")
                print("   ✅ Enhanced training manager initialized successfully")

                # Execute the enhanced training
                logger.info("🚀 STEP 2: Executing Enhanced Training Pipeline...")
                print("   🚀 Starting enhanced training pipeline...")

                # Initialize the training manager
                if not await training_manager.initialize():
                    logger.error("❌ Failed to initialize enhanced training manager")
                    print("❌ Failed to initialize enhanced training manager")
                    return False

                # Prepare training input
                training_input = {
                    "enhanced_training_type": f"{training_mode}_training",
                    "model_architecture": "enhanced_ensemble",
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": "1m",
                    "lookback_days": lookback_days,
                    "training_mode": training_mode,
                }

                # Execute the enhanced training
                success = await training_manager.execute_enhanced_training(
                    training_input,
                )

                if success:
                    logger.info("=" * 80)
                    logger.info("🎉 ENHANCED TRAINING PIPELINE COMPLETED SUCCESSFULLY")
                    logger.info("=" * 80)
                    logger.info(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    logger.info(f"🎯 Symbol: {symbol}")
                    logger.info(f"🏢 Exchange: {exchange}")
                    logger.info(f"📊 Training Mode: {training_mode}")
                    print("=" * 80)
                    print("🎉 ENHANCED TRAINING PIPELINE COMPLETED SUCCESSFULLY")
                    print("=" * 80)
                    print("   ✅ Enhanced training completed successfully!")
                    return True
                else:
                    logger.error("❌ Enhanced training pipeline failed")
                    print("❌ Enhanced training pipeline failed")
                    return False

            except Exception as e:
                logger.error(f"💥 ENHANCED TRAINING PIPELINE FAILED: {str(e)}")
                logger.error(f"📋 Error details: {type(e).__name__}: {str(e)}")
                print(f"💥 ENHANCED TRAINING PIPELINE FAILED: {str(e)}")
                print(f"📋 Error details: {type(e).__name__}: {str(e)}")
                return False

            finally:
                # Cleanup
                try:
                    if 'db_manager' in locals():
                        await db_manager.stop()
                        logger.info("🧹 Database manager cleaned up successfully")
                except Exception as cleanup_error:
                    logger.warning(f"⚠️ Database cleanup warning: {cleanup_error}")

        # Run the async training
        print("🔄 Starting async training execution...")
        print("⏳ Training is running... This may take several minutes.")
        print("📊 You can monitor progress in the logs directory.")

        success = asyncio.run(run_enhanced_training())

        if success:
            self.logger.info(f"✅ {mode_display} completed successfully")
            print(f"✅ {mode_display} completed successfully")
            print("🎉 Training pipeline finished!")
            return True
        self.logger.error(f"❌ {mode_display} failed")
        print(f"❌ {mode_display} failed")
        print("💥 Training pipeline encountered an error.")
        return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_enhanced_blank_training",
    )
    def run_enhanced_blank_training(
        self,
        symbol: str,
        exchange: str,
        with_gui: bool = False,
    ):
        """Run enhanced blank training using unified training method."""
        # Note: The system still processes all available data files during consolidation,
        # but then filters to the specified lookback period. For blank training,
        # we use a smaller lookback period to reduce processing time.
        return self._run_unified_training(
            symbol=symbol,
            exchange=exchange,
            training_mode="blank",
            lookback_days=30,  # 30 days for blank training (minimal dataset)
            with_gui=with_gui,
        )

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_backtesting",
    )
    def run_backtesting(self, symbol: str, exchange: str, with_gui: bool = False):
        """Run enhanced backtesting using unified training method."""
        return self._run_unified_training(
            symbol=symbol,
            exchange=exchange,
            training_mode="backtesting",
            lookback_days=730,  # 2 years for comprehensive backtesting
            with_gui=with_gui,
        )

    def _run_unified_trading(
        self,
        symbol: str,
        exchange: str,
        trading_mode: str,
        with_gui: bool = False,
    ):
        """Unified trading method for both paper and live trading modes."""
        mode_display = "paper trading" if trading_mode == "PAPER" else "live trading"
        self.logger.info(f"📊 Running {mode_display} for {symbol} on {exchange}")
        print(f"📊 Running {mode_display} for {symbol} on {exchange}")
        print("=" * 80)

        if with_gui:
            if not self.launch_gui(trading_mode.lower(), symbol, exchange):
                return False

        try:
            # Set environment variable for trading mode
            import os

            os.environ["TRADING_MODE"] = trading_mode

            # Run the same pipeline but with different trading mode
            process = subprocess.Popen(
                [sys.executable, "src/ares_pipeline.py", symbol, exchange],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
                env=dict(
                    os.environ,
                    TRADING_MODE=trading_mode,
                ),  # Pass environment variable
            )
            self.processes.append(process)

            # Read output in real-time
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    print(output.strip())  # Print to terminal in real-time
                    self.logger.info(output.strip())  # Also log it

            # Get the final return code
            return_code = process.poll()

            if return_code == 0:
                self.logger.info(f"✅ {mode_display} completed successfully")
                print(f"✅ {mode_display} completed successfully")
                return True
            self.logger.error(
                f"❌ {mode_display} failed with return code: {return_code}",
            )
            print(f"❌ {mode_display} failed with return code: {return_code}")
            return False

        except Exception as e:
            self.logger.error(f"❌ Failed to run {mode_display}: {e}")
            print(f"❌ Failed to run {mode_display}: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_paper_trading",
    )
    def run_paper_trading(self, symbol: str, exchange: str, with_gui: bool = False):
        """Run paper trading using unified trading method."""
        return self._run_unified_trading(
            symbol=symbol,
            exchange=exchange,
            trading_mode="PAPER",
            with_gui=with_gui,
        )

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_live_trading",
    )
    def run_live_trading(self, symbol: str, exchange: str, with_gui: bool = False):
        """Run live trading using unified trading method."""
        return self._run_unified_trading(
            symbol=symbol,
            exchange=exchange,
            trading_mode="LIVE",
            with_gui=with_gui,
        )

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_model_trainer",
    )
    def run_model_trainer(self, symbol: str, exchange: str, with_gui: bool = False):
        """Run model training with optional GUI."""
        self.logger.info(f"🧠 Running model training for {symbol} on {exchange}")

        if with_gui:
            if not self.launch_gui("training", symbol, exchange):
                return False

        try:
            # Run the model training script
            process = subprocess.Popen(
                [sys.executable, "scripts/training_cli.py", symbol, exchange],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.processes.append(process)

            stdout, stderr = process.communicate()

            if process.returncode == 0:
                self.logger.info("✅ Model training completed successfully")
                return True
            self.logger.error(f"❌ Model training failed: {stderr}")
            return False

        except Exception as e:
            self.logger.error(f"❌ Failed to run model training: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_portfolio_trading",
    )
    def run_portfolio_trading(self, with_gui: bool = False):
        """Run portfolio trading with optional GUI."""
        self.logger.info("📈 Running portfolio trading")

        if with_gui:
            if not self.launch_gui("portfolio"):
                return False

        # Launch portfolio manager
        if not self.launch_portfolio_manager():
            return False

        # Launch individual trading bots for each supported token
        supported_tokens = CONFIG.get("SUPPORTED_TOKENS", {}).get(
            "BINANCE",
            ["ETHUSDT"],
        )

        for token in supported_tokens:
            self.logger.info(f"🚀 Launching trading bot for {token}")
            try:
                process = subprocess.Popen(
                    [sys.executable, "src/ares_pipeline.py", token, "BINANCE"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self.processes.append(process)
                self.logger.info(
                    f"✅ Trading bot for {token} started with PID {process.pid}",
                )
            except Exception as e:
                self.logger.error(f"❌ Failed to launch trading bot for {token}: {e}")

        return True

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_gui_only",
    )
    def run_gui_only(self):
        """Run GUI only mode."""
        self.logger.info("🖥️ Running GUI only mode")
        return self.launch_gui()

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_gui_with_mode",
    )
    def run_gui_with_mode(self, mode: str, symbol: str, exchange: str):
        """Run GUI with specific mode."""
        self.logger.info(f"🖥️ Running GUI with mode: {mode}")
        return self.launch_gui(mode, symbol, exchange)

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="wait_for_user_input",
    )
    def wait_for_user_input(self):
        """Wait for user input to stop the launcher."""
        self.logger.info("⏸️ Press Enter to stop the launcher...")
        try:
            input()
        except KeyboardInterrupt:
            self.logger.info("🛑 Received keyboard interrupt")
        finally:
            self.cleanup()

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_challenger_trading",
    )
    def run_challenger_trading(
        self,
        symbol: str,
        exchange: str,
        with_gui: bool = False,
    ):
        """Run challenger trading with optional GUI."""
        self.logger.info(f"🏆 Running challenger trading for {symbol} on {exchange}")

        if with_gui:
            if not self.launch_gui("challenger", symbol, exchange):
                return False

        try:
            # Run the challenger trading script
            process = subprocess.Popen(
                [sys.executable, "scripts/setup_challenger_model.py", symbol, exchange],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.processes.append(process)

            stdout, stderr = process.communicate()

            if process.returncode == 0:
                self.logger.info("✅ Challenger trading completed successfully")
                return True
            self.logger.error(f"❌ Challenger trading failed: {stderr}")
            return False

        except Exception as e:
            self.logger.error(f"❌ Failed to run challenger trading: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_comprehensive_training",
    )
    def run_comprehensive_training(
        self,
        symbol: str,
        exchange: str,
        with_gui: bool = False,
    ):
        """Run enhanced comprehensive training pipeline with efficiency optimizations."""
        self.logger.info(
            f"🧠 Running enhanced comprehensive training for {symbol} on {exchange}",
        )

        if with_gui:
            if not self.launch_gui("blank", symbol, exchange):
                return False

        try:
            # Run multi-timeframe training with blank mode for quick testing
            print(
                f"🚀 Starting multi-timeframe blank training for {symbol} on {exchange}...",
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    "scripts/run_multi_timeframe_training.py",
                    "--symbol",
                    symbol,
                    "--quick-test",  # Use limited data and parameters for quick testing
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
            )
            self.processes.append(process)

            # Read output in real-time
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    print(output.strip())  # Print to terminal in real-time
                    self.logger.info(output.strip())  # Also log it

            # Get the final return code
            return_code = process.poll()

            if return_code == 0:
                self.logger.info(
                    "✅ Enhanced comprehensive training completed successfully",
                )
                print("✅ Enhanced comprehensive training completed successfully")
                return True
            self.logger.error(
                f"❌ Enhanced comprehensive training failed with return code: {return_code}",
            )
            print(
                f"❌ Enhanced comprehensive training failed with return code: {return_code}",
            )
            return False

        except Exception as e:
            self.logger.error(f"❌ Failed to run enhanced comprehensive training: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_multi_timeframe_training",
    )
    def run_multi_timeframe_training(
        self,
        symbol: str,
        exchange: str,
        with_gui: bool = False,
        quick_test: bool = False,
    ):
        """Run multi-timeframe training with ensemble creation."""
        self.logger.info(
            f"🎯 Running multi-timeframe training for {symbol} on {exchange}",
        )

        if with_gui:
            if not self.launch_gui("multi-timeframe", symbol, exchange):
                return False

        try:
            # Run the multi-timeframe training script
            print(f"🚀 Starting multi-timeframe training for {symbol} on {exchange}...")

            cmd = [
                sys.executable,
                "scripts/run_multi_timeframe_training.py",
                "--symbol",
                symbol,
                "--timeframes",
                "1h,4h,1d",  # Default timeframes
            ]

            # Add quick-test flag for blank mode
            if quick_test:
                cmd.append("--quick-test")
                self.logger.info(
                    "🧪 Running in quick-test mode (limited data/parameters)",
                )

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
            )
            self.processes.append(process)

            # Read output in real-time
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    print(output.strip())  # Print to terminal in real-time
                    self.logger.info(output.strip())  # Also log it

            # Get the final return code
            return_code = process.poll()

            if return_code == 0:
                self.logger.info("✅ Multi-timeframe training completed successfully")
                print("✅ Multi-timeframe training completed successfully")
                return True
            self.logger.error(
                f"❌ Multi-timeframe training failed with return code: {return_code}",
            )
            print(f"❌ Multi-timeframe training failed with return code: {return_code}")
            return False

        except Exception as e:
            self.logger.error(f"❌ Failed to run multi-timeframe training: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_blank_training",
    )
    def run_blank_training(
        self,
        symbol: str,
        exchange: str,
        with_gui: bool = False,
    ):
        """Run standard blank training pipeline for quick testing."""
        self.logger.info(f"🧪 Running blank training for {symbol} on {exchange}")

        if with_gui:
            if not self.launch_gui("blank", symbol, exchange):
                return False

        try:
            # Run the standard blank training script
            print(f"🚀 Starting blank training for {symbol} on {exchange}...")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "scripts/blank_training_run.py",
                    "--symbol",
                    symbol,
                    "--exchange",
                    exchange,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
            )
            self.processes.append(process)

            # Read output in real-time
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    print(output.strip())  # Print to terminal in real-time
                    self.logger.info(output.strip())  # Also log it

            # Get the final return code
            return_code = process.poll()

            if return_code == 0:
                self.logger.info("✅ Blank training completed successfully")
                print("✅ Blank training completed successfully")
                return True
            self.logger.error(
                f"❌ Blank training failed with return code: {return_code}",
            )
            print(f"❌ Blank training failed with return code: {return_code}")
            return False

        except Exception as e:
            self.logger.error(f"❌ Failed to run blank training: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_enhanced_blank_training",
    )
    def run_enhanced_blank_training(
        self,
        symbol: str,
        exchange: str,
        with_gui: bool = False,
    ):
        """Run enhanced blank training using unified training method."""
        # Note: The system still processes all available data files during consolidation,
        # but then filters to the specified lookback period. For blank training,
        # we use a smaller lookback period to reduce processing time.
        return self._run_unified_training(
            symbol=symbol,
            exchange=exchange,
            training_mode="blank",
            lookback_days=30,  # 30 days for blank training (minimal dataset)
            with_gui=with_gui,
        )

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_data_loading",
    )
    def run_data_loading(
        self,
        symbol: str,
        exchange: str,
        lookback_days: int = 730,
    ) -> bool:
        """Run data loading and consolidation for the specified symbol and exchange."""
        try:
            self.logger.info(f"🔄 Starting data loading for {symbol} on {exchange}")

            # Set environment variable for blank training mode
            import os

            os.environ["BLANK_TRAINING_MODE"] = "1"

            # Step 1: Download data using optimized downloader
            self.logger.info("📥 Step 1: Downloading data...")
            download_script = "backtesting/ares_data_downloader_optimized.py"

            if not os.path.exists(download_script):
                self.logger.error(f"❌ Download script not found: {download_script}")
                return False

            # Run the download script
            download_cmd = [
                sys.executable,
                download_script,
                "--symbol",
                symbol,
                "--exchange",
                exchange,
                "--lookback-years",
                str(lookback_days // 365),
            ]

            self.logger.info(f"🔧 Running download command: {' '.join(download_cmd)}")
            # Pass environment with BLANK_TRAINING_MODE set
            env = os.environ.copy()
            env["BLANK_TRAINING_MODE"] = "1"
            download_result = subprocess.run(
                download_cmd,
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )

            if download_result.returncode != 0:
                self.logger.error(f"❌ Download failed: {download_result.stderr}")
                return False

            self.logger.info("✅ Data download completed successfully")

            # Step 2: Consolidate data using step1_data_collection
            self.logger.info("🔄 Step 2: Consolidating data...")
            consolidate_script = "src/training/steps/step1_data_collection.py"

            if not os.path.exists(consolidate_script):
                self.logger.error(
                    f"❌ Consolidation script not found: {consolidate_script}",
                )
                return False

            # Run the consolidation script
            consolidate_cmd = [
                sys.executable,
                consolidate_script,
                symbol,
                exchange,  # This should be BINANCE
                "1000",  # min_data_points
                "data_cache",  # data_dir
                str(lookback_days),  # Pass lookback period as positional argument
            ]

            self.logger.info(
                f"🔧 Running consolidation command: {' '.join(consolidate_cmd)}",
            )
            # Pass environment with BLANK_TRAINING_MODE set
            consolidate_result = subprocess.run(
                consolidate_cmd,
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )

            if consolidate_result.returncode != 0:
                self.logger.error(
                    f"❌ Consolidation failed: {consolidate_result.stderr}",
                )
                return False

            self.logger.info("✅ Data consolidation completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Data loading failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_regime_operations",
    )
    async def run_regime_operations(
        self,
        symbol: str,
        exchange: str,
        subcommand: str,
        with_gui: bool = False,
    ):
        """Run regime operations (HMM labeling or ML training) with optional GUI."""
        self.logger.info(f"🧠 Running regime operations for {symbol} on {exchange}")
        self.logger.info(f"📋 Subcommand: {subcommand}")
        self.logger.info(f"🖥️ GUI mode: {with_gui}")
        self.logger.info(
            f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        )

        if with_gui:
            if not self.launch_gui("regime", symbol, exchange):
                return False

        try:
            self.logger.info("📦 Importing required modules...")
            # Import UnifiedRegimeClassifier
            from src.analyst.unified_regime_classifier import UnifiedRegimeClassifier
            from src.config import CONFIG

            self.logger.info("✅ Modules imported successfully")

            self.logger.info("🔧 Initializing unified regime classifier...")
            # Initialize unified regime classifier
            regime_classifier = UnifiedRegimeClassifier(CONFIG)
            self.logger.info("✅ Unified regime classifier initialized successfully")

            if subcommand == "load":
                print(
                    f"🚀 Starting unified regime classifier training for {symbol} on {exchange}...",
                )

                # Load historical data from data directory
                data_file = f"data/{symbol}_1h.csv"
                if not os.path.exists(data_file):
                    self.logger.error(f"❌ Data file not found: {data_file}")
                    print(f"❌ Data file not found: {data_file}")
                    print(
                        "Please run data loading first: python ares_launcher.py load --symbol ETHUSDT --exchange BINANCE",
                    )
                    return False

                from src.analyst.data_utils import load_klines_data

                historical_data = load_klines_data(data_file)

                if historical_data is None or historical_data.empty:
                    self.logger.error("❌ Failed to load historical data")
                    print("❌ Failed to load historical data")
                    return False

                # Train unified regime classifier
                success = await regime_classifier.train_complete_system(historical_data)

                if success:
                    self.logger.info(
                        "✅ Unified regime classifier training completed successfully",
                    )
                    print(
                        "✅ Unified regime classifier training completed successfully",
                    )
                    return True
                self.logger.error("❌ Unified regime classifier training failed")
                print("❌ Unified regime classifier training failed")
                return False

            if subcommand == "train":
                self.logger.info(
                    f"🚀 Starting unified regime classifier training for {symbol} on {exchange} (2 years data)...",
                )
                self.logger.info("📊 Training configuration:")
                self.logger.info(f"   - Symbol: {symbol}")
                self.logger.info(f"   - Exchange: {exchange}")
                self.logger.info("   - Lookback years: 2")
                self.logger.info("   - Target timeframe: 1h")
                print(
                    f"🚀 Starting unified regime classifier training for {symbol} on {exchange} (2 years data)...",
                )

                # Load historical data from data directory
                data_file = f"data/{symbol}_1h.csv"
                if not os.path.exists(data_file):
                    self.logger.error(f"❌ Data file not found: {data_file}")
                    print(f"❌ Data file not found: {data_file}")
                    print(
                        "Please run data loading first: python ares_launcher.py load --symbol ETHUSDT --exchange BINANCE",
                    )
                    return False

                from src.analyst.data_utils import load_klines_data

                historical_data = load_klines_data(data_file)

                if historical_data is None or historical_data.empty:
                    self.logger.error("❌ Failed to load historical data")
                    print("❌ Failed to load historical data")
                    return False

                # Train unified regime classifier
                success = await regime_classifier.train_complete_system(historical_data)

                if success:
                    self.logger.info(
                        "✅ Unified regime classifier training completed successfully",
                    )
                    print(
                        "✅ Unified regime classifier training completed successfully",
                    )
                    return True
                self.logger.error("❌ Unified regime classifier training failed")
                print("❌ Unified regime classifier training failed")
                return False

            if subcommand == "train_blank":
                print(
                    f"🚀 Starting unified regime classifier training for {symbol} on {exchange} (30 days data)...",
                )

                # Load historical data
                from src.analyst.data_utils import load_klines_data

                # Load 30 days of data for quick training
                historical_data = await load_klines_data(
                    symbol,
                    exchange,
                    lookback_days=30,
                )

                if historical_data is None or historical_data.empty:
                    self.logger.error("❌ Failed to load historical data")
                    print("❌ Failed to load historical data")
                    return False

                # Train unified regime classifier
                success = await regime_classifier.train_complete_system(historical_data)

                if success:
                    self.logger.info(
                        "✅ Unified regime classifier training completed successfully",
                    )
                    print(
                        "✅ Unified regime classifier training completed successfully",
                    )
                    return True
                self.logger.error("❌ Unified regime classifier training failed")
                print("❌ Unified regime classifier training failed")
                return False

            self.logger.error(f"❌ Unknown regime subcommand: {subcommand}")
            print(f"❌ Unknown regime subcommand: {subcommand}")
            return False

        except Exception as e:
            self.logger.error(f"❌ Failed to run regime operations: {e}")
            print(f"❌ Failed to run regime operations: {e}")
            return False


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ares Trading Bot Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ares_launcher.py paper --symbol ETHUSDT --exchange BINANCE
  python ares_launcher.py backtest --symbol ETHUSDT --exchange BINANCE --gui
  python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --gui
  python ares_launcher.py live --symbol ETHUSDT --exchange BINANCE
  python ares_launcher.py load --symbol ETHUSDT --exchange BINANCE
  python ares_launcher.py load --symbol ETHUSDT --exchange MEXC
  python ares_launcher.py load --symbol ETHUSDT --exchange GATEIO
  python ares_launcher.py portfolio --gui
  python ares_launcher.py gui --mode paper --symbol ETHUSDT --exchange BINANCE
        """,
    )

    parser.add_argument(
        "command",
        choices=[
            "paper",
            "challenger",
            "backtest",
            "model_trainer",
            "live",
            "portfolio",
            "gui",
            "blank",
            "multi-timeframe",
            "load",
            "regime",
        ],
        help="The command to execute",
    )

    parser.add_argument(
        "--symbol",
        type=str,
        help="Trading symbol (e.g., ETHUSDT)",
    )

    parser.add_argument(
        "--exchange",
        type=str,
        default="BINANCE",
        help="Exchange name (default: BINANCE, supported: BINANCE, MEXC, GATEIO)",
    )

    parser.add_argument(
        "--interval",
        type=str,
        default="1m",
        help="K-line interval for data loading (default: 1m)",
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch with GUI",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "paper",
            "challenger",
            "backtest",
            "model_trainer",
            "live",
            "portfolio",
            "load",
        ],
        help="Mode for GUI (required when using gui command with mode)",
    )

    parser.add_argument(
        "--regime-subcommand",
        type=str,
        choices=["load", "train", "train_blank"],
        help="Regime subcommand: 'load' to train unified regime classifier on 2 years data, 'train' to train on 2 years data, 'train_blank' to train on 30 days data",
    )

    parser.add_argument(
        "--optimized",
        action="store_true",
        default=True,
        help="Use optimized data downloader with parallel processing (default: True)",
    )
    parser.add_argument(
        "--no-optimized",
        action="store_true",
        help="Use standard data downloader (disable optimization)",
    )

    parser.add_argument(
        "--blank-mode",
        action="store_true",
        help="Use blank mode (30 days of data) for quick testing instead of 2 years",
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate parsed arguments."""
    if args.command == "gui" and args.mode:
        if not args.symbol or not args.exchange:
            raise ValueError(
                "Symbol and exchange are required when using gui with mode",
            )

    commands_requiring_symbol = [
        "paper",
        "challenger",
        "backtest",
        "model_trainer",
        "live",
        "blank",
        "multi-timeframe",
        "load",
    ]

    if args.command in commands_requiring_symbol:
        if not args.symbol:
            raise ValueError(f"Symbol is required for {args.command} command")


def initialize_launcher() -> tuple[AresLauncher, object]:
    """Initialize launcher with signal handling."""
    signal_handler = setup_signal_handlers()
    launcher = AresLauncher()
    launcher.setup_logging()

    # Add cleanup callback to signal handler
    signal_handler.register_shutdown_callback(launcher.cleanup)

    return launcher, signal_handler


def execute_command(launcher: AresLauncher, args: argparse.Namespace) -> bool:
    """Execute the requested command based on parsed arguments."""
    print(f"🔍 DEBUG: Executing command: {args.command}")
    print(f"🔍 DEBUG: Symbol: {args.symbol}, Exchange: {args.exchange}")

    command_handlers = {
        "backtest": lambda: launcher.run_backtesting(
            args.symbol,
            args.exchange,
            with_gui=args.gui,
        ),
        "paper": lambda: launcher.run_paper_trading(
            args.symbol,
            args.exchange,
            with_gui=args.gui,
        ),
        "challenger": lambda: launcher.run_challenger_trading(
            args.symbol,
            args.exchange,
            with_gui=args.gui,
        ),
        "model_trainer": lambda: launcher.run_model_trainer(
            args.symbol,
            args.exchange,
            with_gui=args.gui,
        ),
        "live": lambda: launcher.run_live_trading(
            args.symbol,
            args.exchange,
            with_gui=args.gui,
        ),
        "portfolio": lambda: launcher.run_portfolio_trading(with_gui=args.gui),
        "blank": lambda: launcher.run_enhanced_blank_training(
            args.symbol,
            args.exchange,
            with_gui=args.gui,
        ),
        "multi-timeframe": lambda: launcher.run_multi_timeframe_training(
            args.symbol,
            args.exchange,
            with_gui=args.gui,
        ),
        "load": lambda: launcher.run_data_loading(
            args.symbol,
            args.exchange,
            lookback_days=730
            if not args.blank_mode
            else 30,  # Use 730 for standard, 30 for blank
        ),
        "regime": lambda: asyncio.run(
            launcher.run_regime_operations(
                args.symbol,
                args.exchange,
                args.regime_subcommand,
                with_gui=args.gui,
            ),
        ),
    }

    if args.command in command_handlers:
        print(f"🔍 DEBUG: Found command handler for '{args.command}'")
        success = command_handlers[args.command]()
        print(f"🔍 DEBUG: Command execution result: {success}")
        if not success:
            return False
        if args.gui:
            launcher.wait_for_user_input()
        return True

    if args.command == "gui":
        return execute_gui_command(launcher, args)

    print(f"❌ ERROR: Unknown command: {args.command}")
    return False


def execute_gui_command(launcher: AresLauncher, args: argparse.Namespace) -> bool:
    """Execute GUI-specific commands."""
    if args.mode:
        if not args.symbol or not args.exchange:
            launcher.logger.error(
                "❌ Symbol and exchange are required when mode is specified",
            )
            return False
        success = launcher.run_gui_with_mode(args.mode, args.symbol, args.exchange)
        if not success:
            return False
        launcher.wait_for_user_input()
        return True
    success = launcher.run_gui_only()
    if not success:
        return False
    launcher.wait_for_user_input()
    return True


@handle_errors(exceptions=(Exception,), default_return=1, context="main")
def main():
    """Main entry point for the Ares launcher."""
    try:
        # Parse and validate arguments
        args = parse_arguments()
        validate_arguments(args)

        # Initialize launcher (this sets up comprehensive logging)
        launcher, signal_handler = initialize_launcher()

        # Log command execution
        command_info = f"Command: {args.command}"
        if hasattr(args, "symbol") and args.symbol:
            command_info += f" - Symbol: {args.symbol}"
        if hasattr(args, "exchange") and args.exchange:
            command_info += f" - Exchange: {args.exchange}"

        launcher.comprehensive_logger.log_launcher_start(
            args.command,
            getattr(args, "symbol", None),
            getattr(args, "exchange", None),
        )

        # Execute the requested command
        success = execute_command(launcher, args)

        if success:
            launcher.comprehensive_logger.log_launcher_end(0)
            return 0
        launcher.comprehensive_logger.log_launcher_end(1)
        return 1

    except Exception as e:
        # Log error if launcher is available
        if "launcher" in locals():
            launcher.comprehensive_logger.log_error(
                f"Main function exception: {e}",
                exc_info=True,
            )
            launcher.comprehensive_logger.log_launcher_end(1)
        else:
            print(f"💥 ERROR: Exception in main: {e}")
            import traceback

            traceback.print_exc()
        return 1
    finally:
        # Ensure cleanup happens
        if "launcher" in locals():
            launcher.cleanup()


if __name__ == "__main__":
    main()
