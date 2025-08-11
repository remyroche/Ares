#!/usr/bin/env python3
"""
Ares Comprehensive Launcher

This script provides a unified interface for launching the Ares trading bot with various modes:
1. Paper trading for robust trade information and performance metrics
2. Enhanced backtesting with cached wavelet features for efficiency (uses existing data)
3. Enhanced model training with efficiency optimizations for large datasets (uses existing data)
4. Live trading for production
5. Portfolio management for multi-token trading
6. HMM regime classification and ML model training
7. Wavelet feature precomputation for fast backtesting

Usage:
    # Paper trading (robust trade info and performance metrics)
    python ares_launcher.py paper --symbol ETHUSDT --exchange BINANCE

    # Challenger paper trading (with challenger model)
    python ares_launcher.py challenger --symbol ETHUSDT --exchange BINANCE

    # Enhanced backtesting with cached wavelet features (uses existing data)
    python ares_launcher.py backtest --symbol ETHUSDT --exchange BINANCE

    # Enhanced model training with efficiency optimizations (uses existing data)
    python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE

    # Multi-timeframe ensemble training (trains models on 1m, 5m, 15m, 1h, 4h, 1d and creates ensembles)
    python ares_launcher.py multi-timeframe --symbol ETHUSDT --exchange BINANCE

    # Live trading for single token
    python ares_launcher.py live --symbol ETHUSDT --exchange BINANCE

    # Data loading (klines, aggtrades, futures) without backtesting
    python ares_launcher.py load --symbol ETHUSDT --exchange BINANCE
    python ares_launcher.py load --symbol ETHUSDT --exchange MEXC
    python ares_launcher.py load --symbol ETHUSDT --exchange GATEIO

    # Wavelet feature precomputation for fast backtesting
    python ares_launcher.py precompute --symbol ETHUSDT --exchange BINANCE

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
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import logging

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
from src.utils.observability import init_observability

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

        # Initialize observability backends (Sentry/OTLP) if configured
        try:
            init_observability({})
        except Exception as _obs_exc:
            logging.getLogger(__name__).warning(f"Observability init skipped: {_obs_exc}")

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

        # Prefer unified start script which runs API and frontend
        script_path = Path("GUI/start.sh")
        env = os.environ.copy()
        # Allow user to override ports via env
        env.setdefault("API_PORT", env.get("API_PORT", "8000"))
        env.setdefault("FRONTEND_PORT", env.get("FRONTEND_PORT", "3000"))
        # If a remote API is used, VITE_API_BASE_URL can be provided by the user
        # Otherwise Vite proxy will forward /api to API_PORT

        if script_path.exists():
            cmd = ["bash", str(script_path)]
        else:
            # Fallback: start API only (legacy behaviour)
            cmd = [sys.executable, "GUI/api_server.py"]
            # Pass optional mode args if provided and using api_server directly
            if mode and symbol and exchange:
                cmd.extend(["--mode", mode, "--symbol", symbol, "--exchange", exchange])

        self.gui_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        self.processes.append(self.gui_process)
        self.logger.info(f"✅ GUI process started with PID {self.gui_process.pid}")

        # Wait a moment for the server(s) to start
        time.sleep(3)

        # Health check: if requests is available, ping frontend then API
        if self.gui_process.poll() is None:
            if REQUESTS_AVAILABLE:
                try:
                    fp = int(env.get("FRONTEND_PORT", "3000"))
                    ap = int(env.get("API_PORT", "8000"))
                    requests.get(f"http://localhost:{fp}", timeout=2)
                    requests.get(f"http://localhost:{ap}/docs", timeout=2)
                    self.logger.info("✅ GUI (frontend+API) appears healthy")
                except Exception as _hc_exc:
                    self.logger.warning(f"GUI health check skipped/failed: {_hc_exc}")
            self.logger.info("✅ GUI server is running")
            return True

        stdout, stderr = self.gui_process.communicate()
        self.logger.error(f"❌ GUI start failed. STDERR: {stderr}\nSTDOUT: {stdout}")
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
        # Set environment variable for training mode
        import os

        if training_mode == "blank":
            os.environ["BLANK_TRAINING_MODE"] = "1"
            os.environ["FULL_TRAINING_MODE"] = "0"
            print("🧪 BLANK TRAINING MODE: Set BLANK_TRAINING_MODE=1")
        elif training_mode == "full":
            os.environ["FULL_TRAINING_MODE"] = "1"
            os.environ["BLANK_TRAINING_MODE"] = "0"
            print("🚀 FULL TRAINING MODE: Set FULL_TRAINING_MODE=1")

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
                
                # Set training parameters based on mode
                if training_mode == "blank":
                    max_trials = 3
                    n_trials = 5
                elif training_mode == "full":
                    max_trials = 200
                    n_trials = 100
                else:
                    max_trials = 200
                    n_trials = 100
                
                training_config = {
                    "enhanced_training_manager": {
                        "enhanced_training_interval": 3600,
                        "max_enhanced_training_history": 100,
                        "enable_advanced_model_training": True,
                        "enable_ensemble_training": True,
                        "enable_multi_timeframe_training": True,
                        "enable_adaptive_training": True,
                        # Parameters based on training mode
                        "blank_training_mode": training_mode == "blank",
                        "full_training_mode": training_mode == "full",
                        "max_trials": max_trials,
                        "n_trials": n_trials,
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
            lookback_days=60,  # 60 days for blank training (expanded for better regime coverage)
            with_gui=with_gui,
        )

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_full_training",
    )
    def run_full_training(
        self,
        symbol: str,
        exchange: str,
        with_gui: bool = False,
    ):
        """Run full training using unified training method with full parameters."""
        # Full training uses complete dataset and full training parameters
        return self._run_unified_training(
            symbol=symbol,
            exchange=exchange,
            training_mode="full",
            lookback_days=1095,  # 1095 days for full training (3 years)
            with_gui=with_gui,
        )

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="resume_training",
    )
    def resume_training(
        self,
        symbol: str,
        exchange: str,
        with_gui: bool = False,
    ):
        """Resume training from the last checkpoint."""
        self.logger.info(f"🔄 Resuming training for {symbol} on {exchange}")
        print(f"🔄 Resuming training for {symbol} on {exchange}")
        
        # Check if checkpoint exists
        checkpoint_file = Path("checkpoints/training_progress.json")
        if not checkpoint_file.exists():
            self.logger.error("❌ No checkpoint found to resume from")
            print("❌ No checkpoint found to resume from")
            return False
        
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            training_mode = checkpoint_data.get("training_mode", "blank")
            lookback_days = checkpoint_data.get("lookback_days", 30)
            last_step = checkpoint_data.get("current_step", "")
            
            self.logger.info(f"📂 Found checkpoint: {last_step}")
            print(f"📂 Found checkpoint: {last_step}")
            
            return self._run_unified_training(
                symbol=symbol,
                exchange=exchange,
                training_mode=training_mode,
                lookback_days=lookback_days,
                with_gui=with_gui,
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to resume training: {e}")
            print(f"❌ Failed to resume training: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="precompute_wavelet_features",
    )
    def precompute_wavelet_features(self, symbol: str, exchange: str) -> bool:
        """Precompute wavelet features for backtesting if they don't exist."""
        self.logger.info(f"🔧 Precomputing wavelet features for {symbol} on {exchange}")
        print(f"🔧 Precomputing wavelet features for {symbol} on {exchange}")

        try:
            # Import the precomputation system
            import asyncio
            from src.training.steps.precompute_wavelet_features import WaveletFeaturePrecomputer
            from src.config import CONFIG

            # Initialize precomputer
            precomputer = WaveletFeaturePrecomputer(CONFIG)
            init_success = asyncio.run(precomputer.initialize())
            
            if not init_success:
                self.logger.error("❌ Failed to initialize wavelet precomputer")
                return False

            # Check if cache already exists
            cache_dir = CONFIG.get("wavelet_cache", {}).get("cache_dir", "data/wavelet_cache")
            import os
            if os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0:
                self.logger.info("✅ Wavelet features already cached, skipping precomputation")
                print("✅ Wavelet features already cached, skipping precomputation")
                return True

            # Data path for precomputation
            data_path = f"data_cache/aggtrades_{exchange}_{symbol}_consolidated.parquet"
            
            if not os.path.exists(data_path):
                self.logger.error(f"❌ Consolidated data file not found: {data_path}")
                self.logger.error("Please run data loading first")
                return False

            # Precompute features
            self.logger.info("🚀 Starting wavelet feature precomputation...")
            print("🚀 Starting wavelet feature precomputation...")
            
            success = asyncio.run(precomputer.precompute_dataset(
                data_path=data_path,
                symbol=symbol
            ))

            if success:
                self.logger.info("✅ Wavelet feature precomputation completed successfully")
                print("✅ Wavelet feature precomputation completed successfully")
                
                # Print statistics
                stats = precomputer.get_precomputation_stats()
                print(f"📊 Precomputation Statistics: {stats}")
                return True
            else:
                self.logger.error("❌ Wavelet feature precomputation failed")
                print("❌ Wavelet feature precomputation failed")
                return False

        except Exception as e:
            self.logger.error(f"❌ Failed to precompute wavelet features: {e}")
            print(f"❌ Failed to precompute wavelet features: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_backtesting",
    )
    def run_backtesting(self, symbol: str, exchange: str, with_gui: bool = False):
        """Run enhanced backtesting using cached wavelet features by default."""
        self.logger.info(f"📊 Running backtesting with cached wavelet features for {symbol} on {exchange}")
        print(f"📊 Running backtesting with cached wavelet features for {symbol} on {exchange}")
        print("=" * 80)

        if with_gui:
            if not self.launch_gui("backtesting", symbol, exchange):
                return False

        try:
            # First, ensure wavelet features are precomputed
            if not self.precompute_wavelet_features(symbol, exchange):
                self.logger.warning("⚠️ Wavelet precomputation failed, continuing with direct computation")
                print("⚠️ Wavelet precomputation failed, continuing with direct computation")

            # Import and use the cached backtesting system
            import asyncio
            from src.training.steps.backtesting_with_cached_features import BacktestingWithCachedFeatures
            from src.config import CONFIG

            # Initialize backtesting with cached features
            backtester = BacktestingWithCachedFeatures(CONFIG)
            
            # Initialize the backtesting system
            init_success = asyncio.run(backtester.initialize())
            if not init_success:
                self.logger.error("❌ Failed to initialize backtesting system")
                return False

            # Load data for backtesting
            data_path = f"data_cache/aggtrades_{exchange}_{symbol}_consolidated.parquet"
            volume_path = f"data_cache/volume_{exchange}_{symbol}_consolidated.parquet"
            
            # Check if consolidated data exists
            import os
            if not os.path.exists(data_path):
                self.logger.error(f"❌ Consolidated data file not found: {data_path}")
                self.logger.error("Please run data loading first: python ares_launcher.py load --symbol ETHUSDT --exchange BINANCE")
                return False

            # Load data
            price_data = asyncio.run(backtester._load_backtest_data(data_path))
            volume_data = asyncio.run(backtester._load_volume_data(volume_path)) if os.path.exists(volume_path) else None

            if price_data is None or price_data.empty:
                self.logger.error("❌ Failed to load price data for backtesting")
                return False

            # Run backtest with cached features
            self.logger.info(f"🚀 Starting backtest with {len(price_data)} data points")
            print(f"🚀 Starting backtest with {len(price_data)} data points")
            
            # Strategy configuration
            strategy_config = {
                "strategy_type": "wavelet_energy_entropy",
                "parameters": {
                    "energy_threshold": 0.5,
                    "entropy_threshold": 0.3,
                    "use_cached_features": True
                }
            }

            # Run the backtest
            results = asyncio.run(backtester.run_backtest(
                price_data=price_data,
                volume_data=volume_data,
                strategy_config=strategy_config
            ))

            if "error" in results:
                self.logger.error(f"❌ Backtesting failed: {results['error']}")
                print(f"❌ Backtesting failed: {results['error']}")
                return False

            # Print results
            strategy_results = results.get("strategy_results", {})
            print("=" * 80)
            print("📊 BACKTESTING RESULTS")
            print("=" * 80)
            print(f"Total Return: {strategy_results.get('total_return', 0):.4f}")
            print(f"Sharpe Ratio: {strategy_results.get('sharpe_ratio', 0):.4f}")
            print(f"Max Drawdown: {strategy_results.get('max_drawdown', 0):.4f}")
            print(f"Win Rate: {strategy_results.get('win_rate', 0):.2%}")
            print(f"Signal Count: {strategy_results.get('signal_count', 0)}")
            print(f"Feature Count: {results.get('feature_count', 0)}")
            print("=" * 80)

            # Print performance stats
            stats = backtester.get_performance_stats()
            print("📈 PERFORMANCE STATISTICS")
            print(f"Cache Hit Rate: {stats.get('cache_hit_rate', 0):.2%}")
            print(f"Avg Backtest Time: {stats.get('avg_backtest_time', 0):.3f}s")
            print(f"Avg Feature Load Time: {stats.get('avg_feature_load_time', 0):.3f}s")
            print(f"Iterations Completed: {stats.get('iterations_completed', 0)}")
            print("=" * 80)

            self.logger.info("✅ Backtesting with cached wavelet features completed successfully")
            print("✅ Backtesting with cached wavelet features completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to run backtesting: {e}")
            print(f"❌ Failed to run backtesting: {e}")
            return False

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

    # REMOVED: run_model_trainer method - Use blank command with step5_analyst_specialist_training instead

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
            lookback_days=60,  # 60 days for blank training (expanded for better regime coverage)
            with_gui=with_gui,
        )

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_step_based_training",
    )
    def run_step_based_training(
        self,
        symbol: str,
        exchange: str,
        start_step: str = "step2_market_regime_classification",
        force_rerun: bool = False,
        with_gui: bool = False,
    ):
        """Run enhanced 16-step training pipeline using the step orchestrator."""
        self.logger.info(f"🚀 Running enhanced 16-step training pipeline for {symbol} on {exchange}")
        self.logger.info(f"Starting from step: {start_step}")
        
        # Ensure BLANK_TRAINING_MODE is set for step-based blank training
        import os
        os.environ["BLANK_TRAINING_MODE"] = "1"
        os.environ["FULL_TRAINING_MODE"] = "0"
        self.logger.info("🧪 BLANK TRAINING MODE: Set BLANK_TRAINING_MODE=1 for step-based training")
        
        # Prevent blank mode from being used with step1_data_collection
        if start_step == "step1_data_collection":
            # Check if we're in blank mode (30 days lookback)
            blank_mode = os.environ.get("BLANK_TRAINING_MODE", "0") == "1"
            if blank_mode:
                self.logger.error("❌ Cannot use blank mode with step1_data_collection")
                self.logger.error("Blank mode is designed for quick testing with limited data")
                self.logger.error("step1_data_collection processes all available data files")
                self.logger.error("Use one of the following instead:")
                self.logger.error("  - python ares_launcher.py load --symbol ETHUSDT --exchange BINANCE (for full data)")
                self.logger.error("  - python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step2_market_regime_classification (for blank mode)")
                return False
        
        if with_gui:
            if not self.launch_gui("training", symbol, exchange):
                return False

        try:
            # Import the step orchestrator
            import os
            from src.training.step_orchestrator import StepOrchestrator
            from src.config import CONFIG
            
            # Check if starting from step2, use pre-consolidated data
            if start_step == "step2_market_regime_classification":
                self.logger.info("📁 Using pre-consolidated data for step2")
                
                # Check for consolidated data file
                consolidated_file = f"data_cache/aggtrades_{exchange}_{symbol}_consolidated.parquet"
                if not os.path.exists(consolidated_file):
                    self.logger.error(f"❌ Consolidated data file not found: {consolidated_file}")
                    self.logger.error("Please run data loading first or ensure consolidated data exists")
                    return False
                
                self.logger.info(f"✅ Found consolidated data: {consolidated_file}")
            
            # Initialize step orchestrator
            orchestrator = StepOrchestrator(symbol, exchange)
            
            # Run the step-based training using the orchestrator
            import asyncio
            success = asyncio.run(orchestrator.execute_from_step(
                start_step=start_step,
                config=CONFIG,
                force_rerun=force_rerun
            ))
            
            if success:
                self.logger.info("✅ Enhanced 16-step training pipeline completed successfully")
                return True
            else:
                self.logger.error("❌ Enhanced 16-step training pipeline failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to run enhanced training pipeline: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_step_based_full_training",
    )
    def run_step_based_full_training(
        self,
        symbol: str,
        exchange: str,
        start_step: str = "step2_market_regime_classification",
        force_rerun: bool = False,
        with_gui: bool = False,
    ):
        """Run step-based full training starting from a specific step with full parameters."""
        self.logger.info(f"🚀 Running step-based full training for {symbol} on {exchange}")
        self.logger.info(f"Starting from step: {start_step}")
        self.logger.info("📊 Using full parameters (1095 days lookback, full training parameters)")
        
        if with_gui:
            if not self.launch_gui("training", symbol, exchange):
                return False

        try:
            # Import the step orchestrator
            import os
            from src.training.step_orchestrator import StepOrchestrator
            from src.config import CONFIG
            
            # Set environment variable for full training mode
            os.environ["FULL_TRAINING_MODE"] = "1"
            os.environ["BLANK_TRAINING_MODE"] = "0"  # Ensure blank mode is off
            
            # Initialize step orchestrator
            orchestrator = StepOrchestrator(symbol, exchange)
            
            # Check if starting from step2, use pre-consolidated data
            if start_step == "step2_market_regime_classification":
                self.logger.info("📁 Using pre-consolidated data for step2")
                
                # Check for consolidated data file
                consolidated_file = f"data_cache/aggtrades_{exchange}_{symbol}_consolidated.parquet"
                if not os.path.exists(consolidated_file):
                    self.logger.error(f"❌ Consolidated data file not found: {consolidated_file}")
                    self.logger.error("Please run data loading first or ensure consolidated data exists")
                    return False
                
                self.logger.info(f"✅ Found consolidated data: {consolidated_file}")
            
            # Run the step-based training
            import asyncio
            success = asyncio.run(orchestrator.execute_from_step(
                start_step=start_step,
                config=CONFIG,
                force_rerun=force_rerun
            ))
            
            if success:
                self.logger.info("✅ Step-based full training completed successfully")
                return True
            else:
                self.logger.error("❌ Step-based full training failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to run step-based full training: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_data_loading",
    )
    def run_data_loading(
        self,
        symbol: str,
        exchange: str,
        lookback_days: int = 1095,
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
                str(CONFIG.get("DATA_CONFIG", {}).get("exclude_recent_days", 0)),  # Exclude recent days
            ]

            self.logger.info(
                f"🔧 Running consolidation command: {' '.join(consolidate_cmd)}",
            )
            # Pass environment with BLANK_TRAINING_MODE set
            self.logger.info("🔄 Starting consolidation subprocess...")
            consolidate_result = subprocess.run(
                consolidate_cmd,
                capture_output=True,
                text=True,
                env=env,
                check=False,
                timeout=1800,  # 30 minute timeout for large datasets
            )
            self.logger.info(f"🔄 Consolidation subprocess completed with return code: {consolidate_result.returncode}")

            if consolidate_result.returncode != 0:
                self.logger.error(
                    f"❌ Consolidation failed: {consolidate_result.stderr}",
                )
                return False

            self.logger.info("✅ Data consolidation completed successfully")

            # Step 3: Convert consolidated data to ETHUSDT_1h.csv format
            if symbol == "ETHUSDT" and exchange == "BINANCE":
                self.logger.info("🔄 Step 3: Converting data to ETHUSDT_1h.csv format...")
                from src.analyst.data_utils import create_ethusdt_1h_csv
                
                conversion_success = create_ethusdt_1h_csv()
                if conversion_success:
                    self.logger.info("✅ Data conversion completed successfully")
                else:
                    self.logger.warning("⚠️ Data conversion failed, but continuing...")
            else:
                self.logger.info(f"⏭️ Skipping ETHUSDT_1h.csv conversion for {symbol} on {exchange}")

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
            regime_classifier = UnifiedRegimeClassifier(CONFIG, exchange, symbol)
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
                    lookback_days=60,
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
  python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step3_regime_data_splitting
  python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step1_data_collection
  python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step2_market_regime_classification
  python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step3_regime_data_splitting
  python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step4_analyst_labeling_feature_engineering
  python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step5_analyst_specialist_training
  python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step6_analyst_enhancement
  python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step7_analyst_ensemble_creation
  python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step8_tactician_labeling
  python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step9_tactician_specialist_training
  python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step10_tactician_ensemble_creation
  python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step11_confidence_calibration
  python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step12_final_parameters_optimization
  python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step13_walk_forward_validation
  python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step14_monte_carlo_validation
  python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step15_ab_testing
  python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step16_saving
  python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step2_market_regime_classification --force-rerun
  python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step5_analyst_specialist_training --force-rerun --gui
  python ares_launcher.py model_trainer --symbol ETHUSDT --exchange BINANCE --step step4_analyst_labeling_feature_engineering
  python ares_launcher.py live --symbol ETHUSDT --exchange BINANCE
  python ares_launcher.py load --symbol ETHUSDT --exchange BINANCE
  python ares_launcher.py load --symbol ETHUSDT --exchange MEXC
  python ares_launcher.py load --symbol ETHUSDT --exchange GATEIO
  python ares_launcher.py portfolio --gui
  python ares_launcher.py gui --mode paper --symbol ETHUSDT --exchange BINANCE
  python ares_launcher.py precompute --symbol ETHUSDT --exchange BINANCE
        """,
    )

    parser.add_argument(
        "command",
        choices=[
            "paper",
            "challenger",
            "backtest",
            # "model_trainer",  # REMOVED: Use blank command with step5_analyst_specialist_training instead
            "live",
            "portfolio",
            "gui",
            "blank",
            "full",
            "multi-timeframe",
            "load",
            "regime",
            "precompute",
            "resume",
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
            # "model_trainer",  # REMOVED: Use blank command with step5_analyst_specialist_training instead
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

    parser.add_argument(
        "--step",
        type=str,
        help="Start training from a specific step (e.g., step2_market_regime_classification)",
    )

    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force rerun of completed steps",
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
        # "model_trainer",  # REMOVED: Use blank command with step5_analyst_specialist_training instead
        "live",
        "blank",
        "full",
        "multi-timeframe",
        "load",
        "precompute",
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
        "blank": lambda: launcher.run_step_based_training(
            args.symbol,
            args.exchange,
            start_step=args.step or "step2_market_regime_classification",
            force_rerun=args.force_rerun,
            with_gui=args.gui,
        ),
        "full": lambda: launcher.run_step_based_full_training(
            args.symbol,
            args.exchange,
            start_step=args.step or "step2_market_regime_classification",
            force_rerun=args.force_rerun,
            with_gui=args.gui,
        ),
        "live": lambda: launcher.run_live_trading(
            args.symbol,
            args.exchange,
            with_gui=args.gui,
        ),
        "portfolio": lambda: launcher.run_portfolio_trading(with_gui=args.gui),
        "multi-timeframe": lambda: launcher.run_multi_timeframe_training(
            args.symbol,
            args.exchange,
            with_gui=args.gui,
        ),
        "load": lambda: launcher.run_data_loading(
            args.symbol,
            args.exchange,
            lookback_days=1095
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
        "precompute": lambda: launcher.precompute_wavelet_features(
            args.symbol,
            args.exchange,
        ),
        "resume": lambda: launcher.resume_training(
            args.symbol,
            args.exchange,
            args.gui,
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
