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

    # Light training (30 days for quick testing and development)
    python ares_launcher.py light --symbol ETHUSDT --exchange BINANCE

    # Show available training modes and their configurations
    python ares_launcher.py modes

    # Start from step02 with existing data (no new downloads)
    python ares_launcher.py step02 --symbol ETHUSDT --exchange BINANCE

    # Step-based training with validation (new steps 1-21)
    python ares_launcher.py step2_5 --symbol ETHUSDT --exchange BINANCE --training-mode blank
    python ares_launcher.py step3_5 --symbol ETHUSDT --exchange BINANCE --training-mode blank
    python ares_launcher.py step9_5 --symbol ETHUSDT --exchange BINANCE --training-mode blank
    python ares_launcher.py step18 --symbol ETHUSDT --exchange BINANCE --training-mode full
    python ares_launcher.py step21 --symbol ETHUSDT --exchange BINANCE --training-mode full

    # Multi-timeframe ensemble training (trains models on 1m, 5m, 15m, 1h, 4h, 1d and creates ensembles)
    python ares_launcher.py multi-timeframe --symbol ETHUSDT --exchange BINANCE

    # Individual pipeline execution (organized structure)
    python ares_launcher.py data-collection --symbol ETHUSDT --exchange BINANCE
    python ares_launcher.py market-analysis --symbol ETHUSDT --exchange BINANCE
    python ares_launcher.py model-training --symbol ETHUSDT --exchange BINANCE
    python ares_launcher.py optimisation --symbol ETHUSDT --exchange BINANCE
    python ares_launcher.py backtesting --symbol ETHUSDT --exchange BINANCE

    # Run all pipelines in sequence
    python ares_launcher.py all-pipelines --symbol ETHUSDT --exchange BINANCE

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
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Import common operations
from src.utils.common_operations import (
    format_datetime,
    get_current_datetime,
)

# Try to import requests for GUI health checks
try:
    import requests

    REQUESTS_AVAILABLE=True
except ImportError:
    REQUESTS_AVAILABLE = False

from src.config import CONFIG
from src.config.constants import (
    DEFAULT_LOOKBACK_DAYS,
)
from src.config.training_modes import (
    get_intensity_comparison,
    get_intensity_percentage,
    get_mode_recommendations,
    get_training_config_dict,
    get_training_input_dict,
    get_training_mode_config,
    list_available_modes,
)
from src.utils.comprehensive_logger import (
    setup_comprehensive_logging,
)
# Simple handle_errors decorator to avoid circular imports
def handle_errors(exceptions=(Exception,), default_return=None, context="operation"):
    """Simple error handling decorator."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logging.error(f"Error in {context}: {e}")
                return default_return
        return wrapper
    return decorator
from src.utils.logger import (
    ensure_comprehensive_logging_available,
)
from src.utils.observability import init_observability
from src.utils.simple_signal_handler import setup_signal_handlers

# Add the project root to the Python path
project_root=Path(__file__).parent
sys.path.insert(0, str(project_root))


class AresLauncher:
    """Comprehensive launcher for Ares trading bot."""

    def __init__(self):
        # Initialize comprehensive logging
        self.comprehensive_logger=setup_comprehensive_logging(CONFIG)

        # Ensure comprehensive logging is available for all existing logging calls
        ensure_comprehensive_logging_available()

        # Initialize observability backends (Sentry/OTLP) if configured
        try:
            init_observability({})
        except Exception as _obs_exc:
            logging.getLogger(__name__).warning(
                f"Observability init skipped: {_obs_exc}",
            )

        self.logger=self.comprehensive_logger.get_component_logger("AresLauncher")
        self.global_logger=self.comprehensive_logger.get_global_logger()
        self.full_log_path=getattr(
            self.comprehensive_logger, "get_full_log_path", lambda: None,
        )()
        self.trades_log_path=getattr(
            self.comprehensive_logger, "get_trades_log_path", lambda: None,
        )()
        self.backtest_log_path=getattr(
            self.comprehensive_logger, "get_backtest_log_path", lambda: None,
        )()
        self.processes=[]  # Track subprocesses for cleanup
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
        self.logger.info(f"Start time: {format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Log directory: {self.comprehensive_logger.log_dir}")
        self.logger.info(f"Log level: {CONFIG.get('logging', {}).get('level', 'INFO')}")
        if self.global_logger:
            self.logger.info(
                f"Global log file: ares_global_{format_datetime(get_current_datetime(), '%Y%m%d_%H%M%S')}.log",
            )
        if self.full_log_path:
            self.logger.info(f"Full run log: {self.full_log_path}")
        if self.trades_log_path:
            self.logger.info(f"Trades log: {self.trades_log_path}")
        if self.backtest_log_path:
            self.logger.info(f"Backtest log: {self.backtest_log_path}")
        self.logger.info("=" * 80)

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="setup_signal_handling",
    )
    def setup_signal_handling(self):
        """Set up centralized signal handling."""
        self.signal_handler=setup_signal_handlers()
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
        mode: str | None=None,
        symbol: str | None=None,
        exchange: str | None=None,
    ):
        """Launch the GUI server."""
        self.logger.info("🚀 Launching GUI server...")

        # Prefer unified start script which runs API and frontend
        script_path=Path("GUI/start.sh")
        env=os.environ.copy()
        # Allow user to override ports via env
        env.setdefault("API_PORT", env.get("API_PORT", "8000"))
        env.setdefault("FRONTEND_PORT", env.get("FRONTEND_PORT", "3000"))
        # If a remote API is used, VITE_API_BASE_URL can be provided by the user
        # Otherwise Vite proxy will forward /api to API_PORT

        if script_path.exists():
            cmd=["bash", str(script_path)]
        else:
            # Fallback: start API only (legacy behaviour)
            cmd=[sys.executable, "GUI/api_server.py"]
            # Pass optional mode args if provided and using api_server directly
            if mode and symbol and exchange:
                cmd.extend(["--mode", mode, "--symbol", symbol, "--exchange", exchange])

        self.gui_process=subprocess.Popen(
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
                    fp=int(env.get("FRONTEND_PORT", "3000"))
                    ap=int(env.get("API_PORT", "8000"))
                    requests.get(f"http://localhost:{fp}", timeout=2)
                    requests.get(f"http://localhost:{ap}/docs", timeout=2)
                    self.logger.info("✅ GUI (frontend+API) appears healthy")
                except Exception as _hc_exc:
                    self.logger.warning(f"GUI health check skipped/failed: {_hc_exc}")
            self.logger.info("✅ GUI server is running")
            return True

        stdout, stderr=self.gui_process.communicate()
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

        self.portfolio_process=subprocess.Popen(
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

    def _normalize_step_name(self, step_name: str | None) -> str | None:
        """Normalize legacy step names to the current ones used by the orchestrator."""
        if not step_name:
            return None
        mapping={
            # Legacy -> Current
            "step2_market_regime_classification": "step2_processing_labeling_feature_engineering",
            "step3_regime_data_splitting": "step4_regime_data_splitting",
            "step4_analyst_labeling_feature_engineering": "step2_processing_labeling_feature_engineering",
        }
        normalized=mapping.get(step_name, step_name)
        if normalized != step_name:
            self.logger.info(
                f"🔁 Normalized requested step '{step_name}' -> '{normalized}'",
            )
        return normalized

    def _clear_checkpoint_files(
        self, symbol: str, exchange: str, timeframe: str="1m",
    ) -> None:
        """Remove enhanced training checkpoints to guarantee a fresh start."""
        try:
            ns_dir=Path("checkpoints") / exchange / symbol / timeframe
            target_file=ns_dir / "training_progress.json"
            if target_file.exists():
                target_file.unlink()
                self.logger.info(f"🗑️  Cleared checkpoint: {target_file}")

            # Also clear any individual step checkpoint files
            for checkpoint_file in ns_dir.glob("*.json"):
                if checkpoint_file.name != "training_progress.json":
                    checkpoint_file.unlink()
                    self.logger.info(f"🗑️  Cleared step checkpoint: {checkpoint_file}")

        except OSError as e:
            self.logger.warning(f"Failed to clear checkpoint: {e}")

    def _force_fresh_start_from_step(self, orchestrator, start_step: str) -> None:
        """Clear progress from the specified start step onward to enforce a fresh run."""
        try:
            steps=orchestrator.list_available_steps()
            if start_step not in steps:
                self.logger.warning(
                    f"Cannot clear progress: step '{start_step}' is not in available steps",
                )
                return

            # Find the index of the starting step
            try:
                start_index=steps.index(start_step)
            except ValueError:
                self.logger.warning(f"⚠️ Unknown step {start_step}, clearing all progress")
                start_index=0

            # Clear progress for the starting step and all subsequent steps
            steps_to_clear = steps[start_index:]

            for step in steps_to_clear:
                if orchestrator.clear_progress(step):
                    self.logger.info(f"🧹 Cleared progress for '{step}' (force)")
                else:
                    self.logger.warning(f"⚠️ Failed to clear progress for '{step}'")

            self.logger.info(f"✅ Cleared progress for {len(steps_to_clear)} steps: {steps_to_clear}")

        except OSError as e:
            self.logger.warning(
                f"Failed clearing progress from step '{start_step}': {e}",
            )

    def _run_unified_training(
        self,
        symbol: str,
        exchange: str,
        training_mode: str,
        lookback_days: int=None,
        with_gui: bool=False,
    ):
        """Run unified training with enhanced training manager using centralized mode configuration."""
        # Get the training mode configuration
        try:
            mode_config=get_training_mode_config(training_mode)
        except ValueError as e:
            self.logger.exception(f"❌ Invalid training mode: {e}")
            print(f"❌ Invalid training mode: {e}")
            return False

        # Use the mode's default lookback_days if not provided
        if lookback_days is None:
            lookback_days=mode_config.lookback_days

        # Set environment variables for training mode
        import os

        if training_mode == "light":
            os.environ["LIGHT_TRAINING_MODE"] = "1"
            os.environ["BLANK_TRAINING_MODE"] = "0"
            os.environ["FULL_TRAINING_MODE"] = "0"
            print("💡 LIGHT TRAINING MODE: Set LIGHT_TRAINING_MODE=1")
        elif training_mode== "blank":
            os.environ["BLANK_TRAINING_MODE"] = "1"
            os.environ["LIGHT_TRAINING_MODE"] = "0"
            os.environ["FULL_TRAINING_MODE"] = "0"
            print("🧪 BLANK TRAINING MODE: Set BLANK_TRAINING_MODE=1")
        elif training_mode== "full":
            os.environ["FULL_TRAINING_MODE"] = "1"
            os.environ["LIGHT_TRAINING_MODE"] = "0"
            os.environ["BLANK_TRAINING_MODE"] = "0"
            print("🚀 FULL TRAINING MODE: Set FULL_TRAINING_MODE=1")

        mode_display=f"{training_mode} training"
        intensity_pct = f"{get_intensity_percentage(training_mode)*100:.0f}%"
        print(f"🚀 Starting {mode_display} for {symbol} on {exchange}")
        print(f"📊 Mode Configuration ({intensity_pct} of full intensity):")
        print(f"   • Lookback Days: {lookback_days}")
        print(f"   • Max Trials: {mode_config.max_trials}")
        print(f"   • N Trials: {mode_config.n_trials}")
        print(f"   • Computational Intensity: {mode_config.computational_intensity}")
        print(f"   • Estimated Duration: {mode_config.estimated_duration_minutes} minutes")
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

            logger=system_logger.getChild("EnhancedTrainingPipeline")

            logger.info("=" * 80)
            logger.info("🚀 ENHANCED TRAINING PIPELINE START")
            logger.info("=" * 80)
            logger.info(
                f"📅 Started at: {format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S')}",
            )
            logger.info(f"🎯 Symbol: {symbol}")
            logger.info(f"🏢 Exchange: {exchange}")
            logger.info(f"📊 Training Mode: {training_mode}")
            logger.info(f"📈 Lookback Days: {lookback_days}")
            logger.info(f"⚙️ Max Trials: {mode_config.max_trials}")
            logger.info(f"⚙️ N Trials: {mode_config.n_trials}")
            logger.info(f"⚙️ Computational Intensity: {mode_config.computational_intensity}")
            print("=" * 80)
            print("🚀 ENHANCED TRAINING PIPELINE START")
            print("=" * 80)

            try:
                # Initialize database manager
                logger.info("📊 STEP 0: Initializing Database Manager...")
                print("   📊 Setting up database manager...")

                default_config={
                    "database": {
                        "sqlite_path": "data/ares.db",
                        "backup_enabled": True,
                        "max_connections": 10,
                        "timeout": 30,
                        "check_same_thread": False,
                    },
                }

                db_manager=SQLiteManager(default_config)
                await db_manager.initialize()
                logger.info("✅ Database manager initialized successfully")
                print("   ✅ Database manager initialized successfully")

                # Initialize enhanced training manager
                logger.info("🤖 STEP 1: Initializing Enhanced Training Manager...")
                print("   🤖 Initializing enhanced training manager...")

                # Get training configuration from centralized mode configuration
                training_config=get_training_config_dict(training_mode)
                training_config["database"] = default_config["database"]

                # Override lookback_days if provided
                if lookback_days != mode_config.lookback_days:
                    training_config["enhanced_training_manager"]["lookback_days"] = lookback_days
                    logger.info(f"📈 Overriding lookback_days to: {lookback_days}")

                training_manager=EnhancedTrainingManager(training_config)
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

                # Prepare training input using centralized configuration
                training_input=get_training_input_dict(
                    mode=training_mode,
                    symbol=symbol,
                    exchange=exchange,
                    timestamp=format_datetime(get_current_datetime(), "%Y-%m-%dT%H:%M:%S"),
                    lookback_days=lookback_days,
                )

                # Execute the enhanced training
                success=await training_manager.execute_enhanced_training(
                    training_input,
                )

                if success:
                    logger.info("=" * 80)
                    logger.info("🎉 ENHANCED TRAINING PIPELINE COMPLETED SUCCESSFULLY")
                    logger.info("=" * 80)
                    logger.info(
                        f"📅 Completed at: {format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S')}",
                    )
                    logger.info(f"🎯 Symbol: {symbol}")
                    logger.info(f"🏢 Exchange: {exchange}")
                    logger.info(f"📊 Training Mode: {training_mode}")
                    logger.info(f"📈 Lookback Days: {lookback_days}")
                    print("=" * 80)
                    print("🎉 ENHANCED TRAINING PIPELINE COMPLETED SUCCESSFULLY")
                    print("=" * 80)
                    print("   ✅ Enhanced training completed successfully!")
                    return True
                logger.error("❌ Enhanced training pipeline failed")
                print("❌ Enhanced training pipeline failed")
                return False

            except Exception as e:
                logger.exception(f"💥 ENHANCED TRAINING PIPELINE FAILED: {str(e)}")
                logger.exception(f"📋 Error details: {type(e).__name__}: {str(e)}")
                print(f"💥 ENHANCED TRAINING PIPELINE FAILED: {str(e)}")
                print(f"📋 Error details: {type(e).__name__}: {str(e)}")
                return False

            finally:
                # Cleanup
                try:
                    if "db_manager" in locals():
                        await db_manager.stop()
                        logger.info("🧹 Database manager cleaned up successfully")
                except Exception as cleanup_error:
                    logger.warning(f"⚠️ Database cleanup warning: {cleanup_error}")

        # Run the async training
        print("🔄 Starting async training execution...")
        print("⏳ Training is running... This may take several minutes.")
        print("📊 You can monitor progress in the logs directory.")

        success=asyncio.run(run_enhanced_training())

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
        with_gui: bool=False,
    ):
        """Run enhanced blank training using unified training method."""
        # Note: The system still processes all available data files during consolidation,
        # but then filters to the specified lookback period. For blank training,
        # we use a smaller lookback period to reduce processing time.
        return self._run_unified_training(
            symbol=symbol,
            exchange=exchange,
            training_mode="blank",
            with_gui=with_gui,
        )

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_light_training",
    )
    def run_light_training(
        self,
        symbol: str,
        exchange: str,
        with_gui: bool=False,
    ):
        """Run light training using unified training method with 30 days."""
        # Light training uses only 30 days for very quick testing
        return self._run_unified_training(
            symbol=symbol,
            exchange=exchange,
            training_mode="light",
            with_gui=with_gui,
        )

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="show_training_modes",
    )
    def show_training_modes(self):
        """Display available training modes and their configurations."""
        print("=" * 80)
        print("🎯 AVAILABLE TRAINING MODES")
        print("=" * 80)

        # Show intensity comparison table
        print("\n📊 INTENSITY COMPARISON")
        print("-" * 80)
        comparison=get_intensity_comparison()

        # Print header
        print(f"{'Mode':<8} {'Intensity':<12} {'Max Trials':<12} {'N Trials':<10} {'Duration':<10} {'Lookback':<10}")
        print("-" * 80)

        for mode, data in comparison.items():
            intensity_pct=f"{data['intensity_percentage']*100:.0f}%"
            print(f"{mode:<8} {intensity_pct:<12} {data['max_trials']:<12} {data['n_trials']:<10} {data['estimated_duration_minutes']:<10}min {data['lookback_days']:<10}days")

        print("\n" + "=" * 80)
        print("📋 DETAILED MODE CONFIGURATIONS")
        print("=" * 80)

        modes=list_available_modes()
        recommendations=get_mode_recommendations()

        for mode_name, description in modes.items():
            try:
                config=get_training_mode_config(mode_name)
                recommendation=recommendations.get(mode_name, "No specific recommendation available.")
                intensity_pct=f"{get_intensity_percentage(mode_name)*100:.0f}%"

                print(f"\n📊 {mode_name.upper()} MODE ({intensity_pct} of full intensity)")
                print(f"   Description: {description}")
                print(f"   Lookback Days: {config.lookback_days}")
                print(f"   Max Trials: {config.max_trials}")
                print(f"   N Trials: {config.n_trials}")
                print(f"   Exclude Recent Days: {config.exclude_recent_days}")
                print(f"   Min Data Points: {config.min_data_points}")
                print(f"   Computational Intensity: {config.computational_intensity}")
                print(f"   Estimated Duration: {config.estimated_duration_minutes} minutes")
                print(f"   Advanced Model Training: {'✅' if config.enable_advanced_model_training else '❌'}")
                print(f"   Ensemble Training: {'✅' if config.enable_ensemble_training else '❌'}")
                print(f"   Multi-timeframe Training: {'✅' if config.enable_multi_timeframe_training else '❌'}")
                print(f"   Adaptive Training: {'✅' if config.enable_adaptive_training else '❌'}")
                print(f"   Recommendation: {recommendation}")

            except ValueError as e:
                print(f"\n❌ Error loading {mode_name} mode: {e}")

        print("\n" + "=" * 80)
        print("💡 USAGE EXAMPLES")
        print("=" * 80)
        print("  python ares_launcher.py light --symbol ETHUSDT --exchange BINANCE")
        print("  python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE")
        print("  python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE")
        print("  python ares_launcher.py light --symbol ETHUSDT --exchange BINANCE --lookback-days 15")
        print("  python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --lookback-days 90")
        print("  python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --lookback-days 365")
        print("=" * 80)

        return True

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_full_training",
    )
    def run_full_training(
        self,
        symbol: str,
        exchange: str,
        with_gui: bool=False,
    ):
        """Run full training using unified training method with full parameters."""
        # Full training uses complete dataset and full training parameters
        return self._run_unified_training(
            symbol=symbol,
            exchange=exchange,
            training_mode="full",
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
        with_gui: bool=False,
    ):
        """Resume training from the last checkpoint."""
        self.logger.info(f"🔄 Resuming training for {symbol} on {exchange}")
        print(f"🔄 Resuming training for {symbol} on {exchange}")

        # Check if checkpoint exists
        checkpoint_file=Path("checkpoints/training_progress.json")
        if not checkpoint_file.exists():
            self.logger.error("❌ No checkpoint found to resume from")
            print("❌ No checkpoint found to resume from")
            return False

        try:
            with open(checkpoint_file) as f:
                checkpoint_data=json.load(f)

            training_mode=checkpoint_data.get("training_mode", "blank")
            lookback_days=checkpoint_data.get("lookback_days", 30)
            last_step=checkpoint_data.get("current_step", "")

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
            self.logger.exception(f"❌ Failed to resume training: {e}")
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

            from src.config import CONFIG
            from src.training.steps.precompute_wavelet_features import (
                WaveletFeaturePrecomputer,
            )

            # Initialize precomputer
            precomputer=WaveletFeaturePrecomputer(CONFIG)
            init_success=asyncio.run(precomputer.initialize())

            if not init_success:
                self.logger.error("❌ Failed to initialize wavelet precomputer")
                return False

            # Check if cache already exists
            cache_dir=CONFIG.get("wavelet_cache", {}).get(
                "cache_dir", "data/wavelet_cache",
            )
            import os

            if os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0:
                self.logger.info(
                    "✅ Wavelet features already cached, skipping precomputation",
                )
                print("✅ Wavelet features already cached, skipping precomputation")
                return True

            # Data path for precomputation
            data_path=f"data_cache/aggtrades_{exchange}_{symbol}_consolidated.parquet"

            if not os.path.exists(data_path):
                self.logger.error(f"❌ Consolidated data file not found: {data_path}")
                self.logger.error("Please run data loading first")
                return False

            # Precompute features
            self.logger.info("🚀 Starting wavelet feature precomputation...")
            print("🚀 Starting wavelet feature precomputation...")

            success=asyncio.run(
                precomputer.precompute_dataset(data_path=data_path, symbol=symbol),
            )

            if success:
                self.logger.info(
                    "✅ Wavelet feature precomputation completed successfully",
                )
                print("✅ Wavelet feature precomputation completed successfully")

                # Print statistics
                stats=precomputer.get_precomputation_stats()
                print(f"📊 Precomputation Statistics: {stats}")
                return True
            self.logger.error("❌ Wavelet feature precomputation failed")
            print("❌ Wavelet feature precomputation failed")
            return False

        except Exception as e:
            self.logger.exception(f"❌ Failed to precompute wavelet features: {e}")
            print(f"❌ Failed to precompute wavelet features: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_backtesting",
    )
    def run_backtesting(self, symbol: str, exchange: str, with_gui: bool=False):
        """Run enhanced backtesting with comprehensive validation and error handling."""
        self.logger.info(
            f"📊 Running enhanced backtesting for {symbol} on {exchange}",
        )
        print(
            f"📊 Running enhanced backtesting for {symbol} on {exchange}",
        )
        print("=" * 80)

        if with_gui and not self.launch_gui("backtesting", symbol, exchange):
            return False

        try:
            # Import enhanced backtesting components
            import asyncio
            from src.training.steps.backtesting import run_backtesting_pipeline, BacktestingPipelineConfig
            from src.utils.common_operations import safe_file_exists, format_datetime, get_current_datetime

            # Enhanced configuration for backtesting
            enhanced_config = {
                'force_rerun': True,
                'walk_forward_validation': True,
                'monte_carlo_validation': True,
                'ab_testing': True,
                'model_saving': True,
                'random_state': 42,
                
                # Enhanced validation settings
                'enable_validation': True,
                'strict_validation': False,
                'validate_data_quality': True,
                
                # Error handling
                'retry_failed_steps': True,
                'max_retries': 3,
                'timeout_seconds': 3600,
                
                # Performance monitoring
                'enable_performance_monitoring': True,
                'log_detailed_metrics': True,
            }

            # Pre-flight validation
            self.logger.info("🔍 Running pre-flight validation...")
            
            # Validate data directory and files
            data_dir = "data_cache"
            required_files = [
                f"aggtrades_{exchange}_{symbol}_consolidated.parquet",
                f"volume_{exchange}_{symbol}_consolidated.parquet"
            ]
            
            missing_files = []
            for file_name in required_files:
                file_path = f"{data_dir}/{file_name}"
                if not safe_file_exists(file_path):
                    missing_files.append(file_name)
            
            if missing_files:
                self.logger.error(f"❌ Missing required data files: {missing_files}")
                self.logger.error("Please run data loading first: python ares_launcher.py load --symbol ETHUSDT --exchange BINANCE")
                return False
            
            self.logger.info("✅ Pre-flight validation passed")

            # Run enhanced backtesting pipeline
            self.logger.info("🚀 Starting enhanced backtesting pipeline...")
            print("🚀 Starting enhanced backtesting pipeline...")
            print(f"📅 Started at: {format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S')}")

            success = asyncio.run(
                run_backtesting_pipeline(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe="1m",
                    data_dir=data_dir,
                    **enhanced_config
                )
            )

            if success:
                self.logger.info("🎉 Enhanced backtesting completed successfully!")
                print("🎉 Enhanced backtesting completed successfully!")
                print("=" * 80)
                print("✅ Enhanced backtesting results:")
                print("   ✅ Comprehensive validation passed")
                print("   ✅ Walk forward validation completed")
                print("   ✅ Monte Carlo validation completed")
                print("   ✅ A/B testing completed")
                print("   ✅ Model saving completed")
                print("   ✅ Performance monitoring completed")
                print("=" * 80)
                
                # Print performance summary
                print("📈 ENHANCED BACKTESTING SUMMARY")
                print(f"Symbol: {symbol}")
                print(f"Exchange: {exchange}")
                print(f"Timeframe: 1m")
                print(f"Validation: Comprehensive")
                print(f"Error Handling: Enhanced")
                print(f"Performance Monitoring: Enabled")
                print("=" * 80)
                
                return True
            else:
                self.logger.error("❌ Enhanced backtesting failed!")
                print("❌ Enhanced backtesting failed!")
                print("Please check the logs for detailed error information")
                return False

        except Exception as e:
            self.logger.exception(f"❌ Failed to run enhanced backtesting: {e}")
            print(f"❌ Failed to run enhanced backtesting: {e}")
            return False

    def _run_unified_trading(
        self,
        symbol: str,
        exchange: str,
        trading_mode: str,
        with_gui: bool=False,
    ):
        """Unified trading method for both paper and live trading modes."""
        mode_display="paper trading" if trading_mode == "PAPER" else "live trading"
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
            process=subprocess.Popen(
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
                output=process.stdout.readline()
                if output== "" and process.poll() is not None:
                    break
                if output:
                    print(output.strip())  # Print to terminal in real-time
                    self.logger.info(output.strip())  # Also log it

            # Get the final return code
            return_code=process.poll()

            if return_code== 0:
                self.logger.info(f"✅ {mode_display} completed successfully")
                print(f"✅ {mode_display} completed successfully")
                return True
            self.logger.error(
                f"❌ {mode_display} failed with return code: {return_code}",
            )
            print(f"❌ {mode_display} failed with return code: {return_code}")
            return False

        except Exception as e:
            self.logger.exception(f"❌ Failed to run {mode_display}: {e}")
            print(f"❌ Failed to run {mode_display}: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_paper_trading",
    )
    def run_paper_trading(self, symbol: str, exchange: str, with_gui: bool=False):
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
    def run_live_trading(self, symbol: str, exchange: str, with_gui: bool=False):
        """Run live trading using unified trading method."""
        return self._run_unified_trading(
            symbol=symbol,
            exchange=exchange,
            trading_mode="LIVE",
            with_gui=with_gui,
        )

    # REMOVED: run_model_trainer method - Use blank command with step5_hmm_based_training instead

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_portfolio_trading",
    )
    def run_portfolio_trading(self, with_gui: bool=False):
        """Run portfolio trading with optional GUI."""
        self.logger.info("📈 Running portfolio trading")

        if with_gui and not self.launch_gui("portfolio"):
            return False

        # Launch portfolio manager
        if not self.launch_portfolio_manager():
            return False

        # Launch individual trading bots for each supported token
        supported_tokens=CONFIG.get("SUPPORTED_TOKENS", {}).get(
            "BINANCE",
            ["ETHUSDT"],
        )

        for token in supported_tokens:
            self.logger.info(f"🚀 Launching trading bot for {token}")
            try:
                process=subprocess.Popen(
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
                self.logger.exception(f"❌ Failed to launch trading bot for {token}: {e}")

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
        with_gui: bool=False,
    ):
        """Run challenger trading with optional GUI."""
        self.logger.info(f"🏆 Running challenger trading for {symbol} on {exchange}")

        if with_gui and not self.launch_gui("challenger", symbol, exchange):
            return False

        try:
            # Run the challenger trading script
            process=subprocess.Popen(
                [sys.executable, "scripts/setup_challenger_model.py", symbol, exchange],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.processes.append(process)

            stdout, stderr=process.communicate()

            if process.returncode== 0:
                self.logger.info("✅ Challenger trading completed successfully")
                return True
            self.logger.error(f"❌ Challenger trading failed: {stderr}")
            return False

        except Exception as e:
            self.logger.exception(f"❌ Failed to run challenger trading: {e}")
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
        with_gui: bool=False,
    ):
        """Run enhanced comprehensive training pipeline with efficiency optimizations."""
        self.logger.info(
            f"🧠 Running enhanced comprehensive training for {symbol} on {exchange}",
        )

        if with_gui and not self.launch_gui("blank", symbol, exchange):
            return False

        try:
            # Run multi-timeframe training with blank mode for quick testing
            print(
                f"🚀 Starting multi-timeframe blank training for {symbol} on {exchange}...",
            )
            process=subprocess.Popen(
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
                output=process.stdout.readline()
                if output== "" and process.poll() is not None:
                    break
                if output:
                    print(output.strip())  # Print to terminal in real-time
                    self.logger.info(output.strip())  # Also log it

            # Get the final return code
            return_code=process.poll()

            if return_code== 0:
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
            self.logger.exception(f"❌ Failed to run enhanced comprehensive training: {e}")
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
        with_gui: bool=False,
        quick_test: bool=False,
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

            cmd=[
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

            process=subprocess.Popen(
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
                output=process.stdout.readline()
                if output== "" and process.poll() is not None:
                    break
                if output:
                    print(output.strip())  # Print to terminal in real-time
                    self.logger.info(output.strip())  # Also log it

            # Get the final return code
            return_code=process.poll()

            if return_code== 0:
                self.logger.info("✅ Multi-timeframe training completed successfully")
                print("✅ Multi-timeframe training completed successfully")
                return True
            self.logger.error(
                f"❌ Multi-timeframe training failed with return code: {return_code}",
            )
            print(f"❌ Multi-timeframe training failed with return code: {return_code}")
            return False

        except Exception as e:
            self.logger.exception(f"❌ Failed to run multi-timeframe training: {e}")
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
        with_gui: bool=False,
    ):
        """Run standard blank training pipeline for quick testing."""
        self.logger.info(f"🧪 Running blank training for {symbol} on {exchange}")

        if with_gui and not self.launch_gui("blank", symbol, exchange):
            return False

        try:
            # Run the standard blank training script
            print(f"🚀 Starting blank training for {symbol} on {exchange}...")
            process=subprocess.Popen(
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
                output=process.stdout.readline()
                if output== "" and process.poll() is not None:
                    break
                if output:
                    print(output.strip())  # Print to terminal in real-time
                    self.logger.info(output.strip())  # Also log it

            # Get the final return code
            return_code=process.poll()

            if return_code== 0:
                self.logger.info("✅ Blank training completed successfully")
                print("✅ Blank training completed successfully")
                return True
            self.logger.error(
                f"❌ Blank training failed with return code: {return_code}",
            )
            print(f"❌ Blank training failed with return code: {return_code}")
            return False

        except Exception as e:
            self.logger.exception(f"❌ Failed to run blank training: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_data_collection_pipeline",
    )
    def run_data_collection_pipeline(
        self,
        symbol: str,
        exchange: str,
        with_gui: bool=False,
    ):
        """Run enhanced data collection pipeline with comprehensive logging and monitoring."""
        self.logger.info(f"📊 Running enhanced data collection pipeline for {symbol} on {exchange}")
        print("=" * 80)
        print("🚀 ENHANCED DATA COLLECTION PIPELINE")
        print("=" * 80)
        print(f"ℹ️ Symbol: {symbol}")
        print(f"ℹ️ Exchange: {exchange}")
        print(f"ℹ️ GUI Mode: {with_gui}")
        print(f"ℹ️ Start Time: {format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Pre-flight validation
        validation_success = self._validate_data_collection_prerequisites(symbol, exchange)
        if not validation_success:
            self.logger.error("❌ Pre-flight validation failed")
            print("❌ Pre-flight validation failed - cannot proceed with data collection")
            return False

        if with_gui and not self.launch_gui("data-collection", symbol, exchange):
            return False

        try:
            # Run the enhanced data collection pipeline
            print(f"🚀 Starting enhanced data collection pipeline for {symbol} on {exchange}...")
            
            # Set up environment with correct Python path and enhanced logging
            env = os.environ.copy()
            env['PYTHONPATH'] = str(project_root)
            env['DATA_COLLECTION_MODE'] = 'enhanced'
            env['SYMBOL'] = symbol
            env['EXCHANGE'] = exchange
            
            process=subprocess.Popen(
                [
                    sys.executable,
                    "standalone_data_collection.py",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
                env=env,
                cwd=str(project_root)
            )
            self.processes.append(process)

            # Enhanced real-time output monitoring with progress tracking
            line_count = 0
            error_count = 0
            warning_count = 0
            success_count = 0
            progress_indicators = []
            
            print("📊 Monitoring pipeline execution...")
            print("=" * 80)
            
            while True:
                output=process.stdout.readline()
                if output== "" and process.poll() is not None:
                    break
                if output:
                    line_count += 1
                    output_stripped = output.strip()
                    
                    # Monitor for different types of messages
                    if "ERROR" in output_stripped or "❌" in output_stripped:
                        error_count += 1
                    elif "WARNING" in output_stripped or "⚠️" in output_stripped:
                        warning_count += 1
                    elif "SUCCESS" in output_stripped or "✅" in output_stripped:
                        success_count += 1
                    elif "Progress:" in output_stripped:
                        progress_indicators.append(output_stripped)
                    
                    print(output_stripped)  # Print to terminal in real-time
                    self.logger.info(output_stripped)  # Also log it
                    
                    # Progress indicator every 25 lines
                    if line_count % 25 == 0:
                        print(f"📊 Progress: {line_count} lines processed, {success_count} successes, {warning_count} warnings, {error_count} errors")

            # Get the final return code
            return_code=process.poll()
            
            # Enhanced result reporting
            print("=" * 80)
            print("📊 DATA COLLECTION PIPELINE RESULTS")
            print("=" * 80)
            print(f"ℹ️ Total lines processed: {line_count}")
            print(f"✅ Total successes: {success_count}")
            print(f"⚠️ Total warnings: {warning_count}")
            print(f"❌ Total errors: {error_count}")
            print(f"ℹ️ Return code: {return_code}")
            print("=" * 80)

            if return_code== 0:
                self.logger.info("✅ Enhanced data collection pipeline completed successfully")
                print("✅ Enhanced data collection pipeline completed successfully")
                print("🎉 All data collection steps completed with validation!")
                
                # Show final progress summary
                if progress_indicators:
                    print("📊 Final Progress Summary:")
                    for indicator in progress_indicators[-3:]:  # Show last 3 progress indicators
                        print(f"   {indicator}")
                
                return True
            else:
                self.logger.error(
                    f"❌ Enhanced data collection pipeline failed with return code: {return_code}",
                )
                print(f"❌ Enhanced data collection pipeline failed with return code: {return_code}")
                if error_count > 0:
                    print(f"💥 Pipeline encountered {error_count} errors during execution")
                return False

        except Exception as e:
            self.logger.exception(f"❌ Failed to run enhanced data collection pipeline: {e}")
            print(f"❌ Failed to run enhanced data collection pipeline: {e}")
            return False
        finally:
            # Cleanup environment variables
            import os
            os.environ.pop("DATA_COLLECTION_MODE", None)
            os.environ.pop("SYMBOL", None)
            os.environ.pop("EXCHANGE", None)

    def _validate_data_collection_prerequisites(self, symbol: str, exchange: str) -> bool:
        """Validate prerequisites for data collection pipeline execution."""
        self.logger.info("🔍 Validating data collection prerequisites...")
        print("🔍 Validating data collection prerequisites...")
        
        try:
            from src.utils.common_operations import safe_file_exists, ensure_directory
            
            # Check required directories
            required_dirs = [
                "data_cache",
                "log"
            ]
            
            for dir_path in required_dirs:
                if not safe_file_exists(dir_path):
                    self.logger.warning(f"⚠️ Creating missing directory: {dir_path}")
                    ensure_directory(dir_path)
                else:
                    self.logger.info(f"✅ Directory exists: {dir_path}")
            
            # Check for standalone data collection script
            script_path = "standalone_data_collection.py"
            if not safe_file_exists(script_path):
                self.logger.error(f"❌ Data collection script not found: {script_path}")
                print(f"❌ Data collection script not found: {script_path}")
                return False
            else:
                self.logger.info(f"✅ Data collection script found: {script_path}")
            
            # Check Python environment
            try:
                import pandas as pd
                import numpy as np
                self.logger.info("✅ Required Python packages available")
            except ImportError as e:
                self.logger.error(f"❌ Missing required Python package: {e}")
                print(f"❌ Missing required Python package: {e}")
                return False
            
            self.logger.info("✅ Data collection prerequisites validation completed")
            print("✅ Data collection prerequisites validation completed")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Prerequisites validation failed: {e}")
            print(f"❌ Prerequisites validation failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_market_analysis_pipeline",
    )
    def run_market_analysis_pipeline(
        self,
        symbol: str,
        exchange: str,
        with_gui: bool=False,
    ):
        """Run market analysis pipeline."""
        self.logger.info(f"📊 Running market analysis pipeline for {symbol} on {exchange}")

        if with_gui and not self.launch_gui("market-analysis", symbol, exchange):
            return False

        try:
            # Run the market analysis pipeline
            print(f"🚀 Starting market analysis pipeline for {symbol} on {exchange}...")
            process=subprocess.Popen(
                [
                    sys.executable,
                    "src/training/steps/market_analysis/step03_market_analysis_main.py",
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
                output=process.stdout.readline()
                if output== "" and process.poll() is not None:
                    break
                if output:
                    print(output.strip())  # Print to terminal in real-time
                    self.logger.info(output.strip())  # Also log it

            # Get the final return code
            return_code=process.poll()

            if return_code== 0:
                self.logger.info("✅ Market analysis pipeline completed successfully")
                print("✅ Market analysis pipeline completed successfully")
                return True
            self.logger.error(
                f"❌ Market analysis pipeline failed with return code: {return_code}",
            )
            print(f"❌ Market analysis pipeline failed with return code: {return_code}")
            return False

        except Exception as e:
            self.logger.exception(f"❌ Failed to run market analysis pipeline: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_model_training_pipeline",
    )
    def run_model_training_pipeline(
        self,
        symbol: str,
        exchange: str,
        with_gui: bool=False,
    ):
        """Run model training pipeline."""
        self.logger.info(f"📊 Running model training pipeline for {symbol} on {exchange}")

        if with_gui and not self.launch_gui("model-training", symbol, exchange):
            return False

        try:
            # Run the model training pipeline
            print(f"🚀 Starting model training pipeline for {symbol} on {exchange}...")
            process=subprocess.Popen(
                [
                    sys.executable,
                    "src/training/steps/model_training/step09_model_training_main.py",
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
                output=process.stdout.readline()
                if output== "" and process.poll() is not None:
                    break
                if output:
                    print(output.strip())  # Print to terminal in real-time
                    self.logger.info(output.strip())  # Also log it

            # Get the final return code
            return_code=process.poll()

            if return_code== 0:
                self.logger.info("✅ Model training pipeline completed successfully")
                print("✅ Model training pipeline completed successfully")
                return True
            self.logger.error(
                f"❌ Model training pipeline failed with return code: {return_code}",
            )
            print(f"❌ Model training pipeline failed with return code: {return_code}")
            return False

        except Exception as e:
            self.logger.exception(f"❌ Failed to run model training pipeline: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_optimisation_pipeline",
    )
    def run_optimisation_pipeline(
        self,
        symbol: str,
        exchange: str,
        with_gui: bool=False,
    ):
        """Run enhanced optimisation pipeline with comprehensive validation and protection.
        
        Features:
        - 🎯 Comprehensive logging with emojis for easy troubleshooting
        - 📊 Real-time progress tracking and monitoring
        - 🔍 Detailed error reporting and quality issue flagging
        - ✅ Step-by-step validation with detailed reporting
        - 📈 Performance metrics tracking
        - 🛡️ Data quality monitoring throughout the process
        """
        self.logger.info(f"📊 Running enhanced optimisation pipeline for {symbol} on {exchange}")
        print("=" * 80)
        print("🚀 ENHANCED OPTIMISATION PIPELINE")
        print("=" * 80)
        print(f"🎯 Symbol: {symbol}")
        print(f"🏢 Exchange: {exchange}")
        print(f"🖥️ GUI Mode: {with_gui}")
        print(f"⏰ Start Time: {format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Initialize monitoring variables
        pipeline_start_time = time.time()
        error_count = 0
        warning_count = 0
        progress_updates = 0

        # Pre-flight validation
        validation_success = self._validate_optimisation_prerequisites(symbol, exchange)
        if not validation_success:
            self.logger.error("❌ Pre-flight validation failed")
            print("❌ Pre-flight validation failed - cannot proceed with optimisation")
            return False

        if with_gui and not self.launch_gui("optimisation", symbol, exchange):
            return False

        try:
            # Enhanced optimisation pipeline with comprehensive error handling
            print(f"🚀 Starting enhanced optimisation pipeline for {symbol} on {exchange}...")
            
            # Set environment variables for enhanced pipeline
            import os
            os.environ["OPTIMISATION_MODE"] = "enhanced"
            os.environ["SYMBOL"] = symbol
            os.environ["EXCHANGE"] = exchange
            
            process=subprocess.Popen(
                [
                    sys.executable,
                    "src/training/steps/optimisation/step16_optimisation_main.py",
                    "--symbol", symbol,
                    "--exchange", exchange,
                    "--enhanced-mode"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
                env=dict(os.environ, OPTIMISATION_MODE="enhanced", SYMBOL=symbol, EXCHANGE=exchange),
            )
            self.processes.append(process)

            # Read output in real-time with enhanced monitoring
            line_count = 0
            process_error_count = 0
            process_warning_count = 0
            quality_issues = []
            step_progress = {}
            
            def log_quality_issue(issue_type: str, description: str, severity: str = "WARNING"):
                """Log quality issues with detailed reporting."""
                timestamp = format_datetime(get_current_datetime(), '%H:%M:%S')
                issue = {
                    'timestamp': timestamp,
                    'type': issue_type,
                    'description': description,
                    'severity': severity
                }
                quality_issues.append(issue)
                
                if severity == "ERROR":
                    self.logger.error(f"🔴 [{timestamp}] QUALITY ISSUE - {issue_type}: {description}")
                    print(f"🔴 [{timestamp}] QUALITY ISSUE - {issue_type}: {description}")
                elif severity == "WARNING":
                    self.logger.warning(f"🟡 [{timestamp}] QUALITY ISSUE - {issue_type}: {description}")
                    print(f"🟡 [{timestamp}] QUALITY ISSUE - {issue_type}: {description}")
                else:
                    self.logger.info(f"🔵 [{timestamp}] QUALITY ISSUE - {issue_type}: {description}")
                    print(f"🔵 [{timestamp}] QUALITY ISSUE - {issue_type}: {description}")
            
            def update_step_progress(step_name: str, status: str):
                """Update step progress tracking."""
                if step_name not in step_progress:
                    step_progress[step_name] = {'start_time': time.time(), 'status': 'STARTING'}
                
                step_progress[step_name]['status'] = status
                step_progress[step_name]['last_update'] = time.time()
                
                if status == "COMPLETED":
                    duration = time.time() - step_progress[step_name]['start_time']
                    self.logger.info(f"✅ [{step_name}] Completed in {duration:.2f}s")
                    print(f"✅ [{step_name}] Completed in {duration:.2f}s")
                elif status == "FAILED":
                    duration = time.time() - step_progress[step_name]['start_time']
                    self.logger.error(f"❌ [{step_name}] Failed after {duration:.2f}s")
                    print(f"❌ [{step_name}] Failed after {duration:.2f}s")
                    log_quality_issue("STEP_FAILURE", f"{step_name} failed", "ERROR")
            
            while True:
                output=process.stdout.readline()
                if output== "" and process.poll() is not None:
                    break
                if output:
                    line_count += 1
                    output_stripped = output.strip()
                    
                    # Monitor for errors and warnings
                    if "ERROR" in output_stripped or "❌" in output_stripped:
                        process_error_count += 1
                        log_quality_issue("PROCESS_ERROR", output_stripped, "ERROR")
                    elif "WARNING" in output_stripped or "⚠️" in output_stripped:
                        process_warning_count += 1
                        log_quality_issue("PROCESS_WARNING", output_stripped, "WARNING")
                    
                    # Track step progress
                    if "STEP" in output_stripped and "STARTING" in output_stripped:
                        # Extract step name from output
                        if "Confidence Calibration" in output_stripped:
                            update_step_progress("Confidence Calibration", "STARTING")
                        elif "Final Parameters Optimization" in output_stripped:
                            update_step_progress("Final Parameters Optimization", "STARTING")
                    elif "STEP" in output_stripped and "COMPLETED" in output_stripped:
                        if "Confidence Calibration" in output_stripped:
                            update_step_progress("Confidence Calibration", "COMPLETED")
                        elif "Final Parameters Optimization" in output_stripped:
                            update_step_progress("Final Parameters Optimization", "COMPLETED")
                    elif "STEP" in output_stripped and "FAILED" in output_stripped:
                        if "Confidence Calibration" in output_stripped:
                            update_step_progress("Confidence Calibration", "FAILED")
                        elif "Final Parameters Optimization" in output_stripped:
                            update_step_progress("Final Parameters Optimization", "FAILED")
                    
                    print(output_stripped)  # Print to terminal in real-time
                    self.logger.info(output_stripped)  # Also log it
                    
                    # Enhanced progress indicator every 25 lines
                    if line_count % 25 == 0:
                        progress_updates += 1
                        elapsed_time = time.time() - pipeline_start_time
                        print(f"📊 Progress Update #{progress_updates}: {line_count} lines processed, {process_error_count} errors, {process_warning_count} warnings, {elapsed_time:.1f}s elapsed")
                        self.logger.info(f"📊 Progress Update #{progress_updates}: {line_count} lines, {process_error_count} errors, {process_warning_count} warnings")

            # Get the final return code
            return_code=process.poll()
            total_pipeline_time = time.time() - pipeline_start_time
            
            # Enhanced result reporting
            print("\n" + "=" * 80)
            print("📊 ENHANCED OPTIMISATION PIPELINE RESULTS")
            print("=" * 80)
            print(f"🎯 Symbol: {symbol}")
            print(f"🏢 Exchange: {exchange}")
            print(f"⏱️ Total Duration: {total_pipeline_time:.2f} seconds")
            print(f"📈 Total lines processed: {line_count}")
            print(f"📊 Progress updates: {progress_updates}")
            print(f"❌ Total errors: {process_error_count}")
            print(f"⚠️ Total warnings: {process_warning_count}")
            print(f"🔍 Quality issues: {len(quality_issues)}")
            print(f"🔢 Return code: {return_code}")
            
            # Step progress summary
            if step_progress:
                print("\n📋 STEP PROGRESS SUMMARY:")
                for step_name, progress in step_progress.items():
                    status_emoji = "✅" if progress['status'] == "COMPLETED" else "❌" if progress['status'] == "FAILED" else "🔄"
                    duration = time.time() - progress['start_time']
                    print(f"   {status_emoji} {step_name}: {progress['status']} ({duration:.2f}s)")
            
            # Quality issues summary
            if quality_issues:
                print("\n🔍 QUALITY ISSUES SUMMARY:")
                for issue in quality_issues:
                    severity_emoji = "🔴" if issue['severity'] == "ERROR" else "🟡" if issue['severity'] == "WARNING" else "🔵"
                    print(f"   {severity_emoji} [{issue['timestamp']}] {issue['type']}: {issue['description']}")
            
            print("=" * 80)

            if return_code== 0:
                self.logger.info("✅ Enhanced optimisation pipeline completed successfully")
                print("✅ Enhanced optimisation pipeline completed successfully")
                print("🎉 All optimisation steps completed with validation!")
                
                # Save success summary
                try:
                    success_summary_file = f"data_cache/optimisation_success_summary_{symbol}_{exchange}.json"
                    success_data = {
                        'symbol': symbol,
                        'exchange': exchange,
                        'execution_time': format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S'),
                        'total_duration_seconds': total_pipeline_time,
                        'lines_processed': line_count,
                        'progress_updates': progress_updates,
                        'error_count': process_error_count,
                        'warning_count': process_warning_count,
                        'quality_issues': quality_issues,
                        'step_progress': step_progress,
                        'return_code': return_code,
                        'success': True
                    }
                    
                    import json
                    with open(success_summary_file, 'w') as f:
                        json.dump(success_data, f, indent=2, default=str)
                    
                    self.logger.info(f"💾 Success summary saved to: {success_summary_file}")
                    print(f"💾 Success summary saved to: {success_summary_file}")
                except Exception as save_error:
                    self.logger.warning(f"⚠️ Could not save success summary: {save_error}")
                
                return True
            else:
                self.logger.error(
                    f"❌ Enhanced optimisation pipeline failed with return code: {return_code}",
                )
                print(f"❌ Enhanced optimisation pipeline failed with return code: {return_code}")
                if process_error_count > 0:
                    print(f"💥 Pipeline encountered {process_error_count} errors during execution")
                
                # Save failure summary
                try:
                    failure_summary_file = f"data_cache/optimisation_failure_summary_{symbol}_{exchange}.json"
                    failure_data = {
                        'symbol': symbol,
                        'exchange': exchange,
                        'execution_time': format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S'),
                        'total_duration_seconds': total_pipeline_time,
                        'lines_processed': line_count,
                        'progress_updates': progress_updates,
                        'error_count': process_error_count,
                        'warning_count': process_warning_count,
                        'quality_issues': quality_issues,
                        'step_progress': step_progress,
                        'return_code': return_code,
                        'success': False
                    }
                    
                    import json
                    with open(failure_summary_file, 'w') as f:
                        json.dump(failure_data, f, indent=2, default=str)
                    
                    self.logger.info(f"💾 Failure summary saved to: {failure_summary_file}")
                    print(f"💾 Failure summary saved to: {failure_summary_file}")
                except Exception as save_error:
                    self.logger.warning(f"⚠️ Could not save failure summary: {save_error}")
                
                return False

        except Exception as e:
            total_pipeline_time = time.time() - pipeline_start_time
            self.logger.exception(f"❌ Failed to run enhanced optimisation pipeline: {e}")
            print(f"\n💥 ENHANCED OPTIMISATION PIPELINE FAILED WITH EXCEPTION!")
            print("=" * 80)
            print(f"⏱️ Duration before failure: {total_pipeline_time:.2f} seconds")
            print(f"❌ Error: {str(e)}")
            print(f"🔍 Error Type: {type(e).__name__}")
            print("=" * 80)
            
            # Save exception summary
            try:
                exception_summary_file = f"data_cache/optimisation_exception_summary_{symbol}_{exchange}.json"
                exception_data = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'execution_time': format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S'),
                    'duration_before_failure': total_pipeline_time,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'success': False
                }
                
                import json
                with open(exception_summary_file, 'w') as f:
                    json.dump(exception_data, f, indent=2, default=str)
                
                self.logger.info(f"💾 Exception summary saved to: {exception_summary_file}")
                print(f"💾 Exception summary saved to: {exception_summary_file}")
            except Exception as save_error:
                self.logger.warning(f"⚠️ Could not save exception summary: {save_error}")
            
            return False
        finally:
            # Cleanup environment variables
            import os
            os.environ.pop("OPTIMISATION_MODE", None)
            os.environ.pop("SYMBOL", None)
            os.environ.pop("EXCHANGE", None)
            
            # Final cleanup logging
            total_time = time.time() - pipeline_start_time
            self.logger.info(f"🧹 Optimisation pipeline cleanup completed after {total_time:.2f}s")
            print(f"🧹 Optimisation pipeline cleanup completed after {total_time:.2f}s")

    def _validate_optimisation_prerequisites(self, symbol: str, exchange: str) -> bool:
        """Validate prerequisites for optimisation pipeline execution."""
        self.logger.info("🔍 Validating optimisation prerequisites...")
        print("🔍 Validating optimisation prerequisites...")
        
        try:
            from src.utils.common_operations import safe_file_exists, ensure_directory
            from src.utils.data_quality_framework import DataQualityFramework
            
            # Initialize data quality framework
            dq_framework = DataQualityFramework()
            
            # Check required directories
            required_dirs = [
                "data_cache",
                "models",
                "checkpoints",
                "log"
            ]
            
            for dir_path in required_dirs:
                if not safe_file_exists(dir_path):
                    self.logger.warning(f"⚠️ Creating missing directory: {dir_path}")
                    ensure_directory(dir_path)
                else:
                    self.logger.info(f"✅ Directory exists: {dir_path}")
            
            # Check for required data files
            required_data_files = [
                f"data_cache/aggtrades_{exchange}_{symbol}_consolidated.parquet",
                f"data_cache/volume_{exchange}_{symbol}_consolidated.parquet"
            ]
            
            missing_files = []
            for file_path in required_data_files:
                if not safe_file_exists(file_path):
                    missing_files.append(file_path)
                else:
                    self.logger.info(f"✅ Data file exists: {file_path}")
            
            if missing_files:
                self.logger.error(f"❌ Missing required data files: {missing_files}")
                print(f"❌ Missing required data files:")
                for file_path in missing_files:
                    print(f"   • {file_path}")
                print("💡 Please run data collection first:")
                print(f"   python ares_launcher.py load --symbol {symbol} --exchange {exchange}")
                return False
            
            # Check for previous step outputs
            previous_step_files = [
                f"models/{symbol}_{exchange}_tactician_specialist.pkl",
                f"models/{symbol}_{exchange}_confidence_calibration.json"
            ]
            
            missing_previous = []
            for file_path in previous_step_files:
                if not safe_file_exists(file_path):
                    missing_previous.append(file_path)
            
            if missing_previous:
                self.logger.warning(f"⚠️ Some previous step outputs missing: {missing_previous}")
                print("⚠️ Some previous step outputs are missing - optimisation will use defaults")
            
            self.logger.info("✅ Optimisation prerequisites validation completed")
            print("✅ Optimisation prerequisites validation completed")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Prerequisites validation failed: {e}")
            print(f"❌ Prerequisites validation failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_backtesting_pipeline",
    )
    def run_backtesting_pipeline(
        self,
        symbol: str,
        exchange: str,
        with_gui: bool=False,
    ):
        """Run backtesting pipeline."""
        self.logger.info(f"📊 Running backtesting pipeline for {symbol} on {exchange}")

        if with_gui and not self.launch_gui("backtesting", symbol, exchange):
            return False

        try:
            # Run the backtesting pipeline
            print(f"🚀 Starting backtesting pipeline for {symbol} on {exchange}...")
            process=subprocess.Popen(
                [
                    sys.executable,
                    "src/training/steps/backtesting/step18_backtesting_main.py",
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
                output=process.stdout.readline()
                if output== "" and process.poll() is not None:
                    break
                if output:
                    print(output.strip())  # Print to terminal in real-time
                    self.logger.info(output.strip())  # Also log it

            # Get the final return code
            return_code=process.poll()

            if return_code== 0:
                self.logger.info("✅ Backtesting pipeline completed successfully")
                print("✅ Backtesting pipeline completed successfully")
                return True
            self.logger.error(
                f"❌ Backtesting pipeline failed with return code: {return_code}",
            )
            print(f"❌ Backtesting pipeline failed with return code: {return_code}")
            return False

        except Exception as e:
            self.logger.exception(f"❌ Failed to run backtesting pipeline: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_all_pipelines",
    )
    def run_all_pipelines(
        self,
        symbol: str,
        exchange: str,
        with_gui: bool=False,
    ):
        """Run all pipelines in sequence."""
        self.logger.info(f"📊 Running all pipelines for {symbol} on {exchange}")

        if with_gui and not self.launch_gui("all-pipelines", symbol, exchange):
            return False

        try:
            # Run the all pipelines orchestrator
            print(f"🚀 Starting all pipelines for {symbol} on {exchange}...")
            process=subprocess.Popen(
                [
                    sys.executable,
                    "src/training/steps/run_all_pipelines.py",
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
                output=process.stdout.readline()
                if output== "" and process.poll() is not None:
                    break
                if output:
                    print(output.strip())  # Print to terminal in real-time
                    self.logger.info(output.strip())  # Also log it

            # Get the final return code
            return_code=process.poll()

            if return_code== 0:
                self.logger.info("✅ All pipelines completed successfully")
                print("✅ All pipelines completed successfully")
                return True
            self.logger.error(
                f"❌ All pipelines failed with return code: {return_code}",
            )
            print(f"❌ All pipelines failed with return code: {return_code}")
            return False

        except Exception as e:
            self.logger.exception(f"❌ Failed to run all pipelines: {e}")
            return False



    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_step_based_training",
    )
    def run_step_based_training(
        self,
        symbol: str,
        exchange: str,
        start_step: str="step2_processing_labeling_feature_engineering",
        force_rerun: bool=False,
        with_gui: bool=False,
    ):
        """Run enhanced 16-step training pipeline using the step orchestrator."""
        self.logger.info(
            f"🚀 Running enhanced 16-step training pipeline for {symbol} on {exchange}",
        )
        return self._run_step_pipeline(
            symbol=symbol,
            exchange=exchange,
            start_step=start_step,
            force_rerun=force_rerun,
            with_gui=with_gui,
            training_mode="blank",
        )

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_step_based_full_training",
    )
    def run_step_based_full_training(
        self,
        symbol: str,
        exchange: str,
        start_step: str="step2_processing_labeling_feature_engineering",
        force_rerun: bool=False,
        with_gui: bool=False,
    ):
        """Run step-based full training starting from a specific step with full parameters."""
        self.logger.info(
            f"🚀 Running step-based full training for {symbol} on {exchange}",
        )
        self.logger.info(
            "📊 Using full parameters (730 days lookback, full training parameters)",
        )
        return self._run_step_pipeline(
            symbol=symbol,
            exchange=exchange,
            start_step=start_step,
            force_rerun=force_rerun,
            with_gui=with_gui,
            training_mode="full",
        )

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_step_based_training_with_validation",
    )
    async def run_step_based_training_with_validation(
        self,
        symbol: str,
        exchange: str,
        start_step: str,
        training_mode: str="blank",
        force_rerun: bool=False,
        with_gui: bool=False,
    ):
        """Run step-based training with comprehensive validation of previous steps."""
        self.logger.info(
            f"🚀 Running step-based training with validation for {symbol} on {exchange}",
        )
        self.logger.info(f"📊 Starting from: {start_step}")
        self.logger.info(f"🎯 Training mode: {training_mode}")

        # Validate previous steps before proceeding
        validation_success=await self._validate_previous_steps(symbol, exchange, start_step)
        if not validation_success:
            self.logger.error(f"❌ Cannot start from {start_step} - previous step validation failed")
            return False

        # Run the step pipeline with the specified training mode
        return self._run_step_pipeline(
            symbol=symbol,
            exchange=exchange,
            start_step=start_step,
            force_rerun=force_rerun,
            with_gui=with_gui,
            training_mode=training_mode,
        )

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_step2_with_existing_data",
    )
    async def run_step2_with_existing_data(
        self,
        symbol: str,
        exchange: str,
        start_step: str="step2_feature_engineering",
        force_rerun: bool=False,
        with_gui: bool=False,
    ):
        """Run step02 with existing data from step01 and step1_5 without triggering new downloads."""
        self.logger.info(
            f"🚀 Running step02 with existing data for {symbol} on {exchange}",
        )
        self.logger.info(
            "📊 Using existing data from step01 and step1_5 - no new downloads",
        )

        # Use existing validator orchestrator to validate step01 and step1_5
        try:
            from src.utils.validator_orchestrator import ValidatorOrchestrator

            # Create validator orchestrator
            validator_orchestrator=ValidatorOrchestrator()

            # Prepare training input for validation
            training_input={
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": "1m",
                "data_dir": "data_cache",
            }

            # Empty pipeline state since we're checking existing data
            pipeline_state={}

            # Validate step01 and step1_5 using existing validators
            self.logger.info("🔍 Validating step1_data_collection using existing validator")
            step1_result=await validator_orchestrator.run_step_validator(
                "step1_data_collection", training_input, pipeline_state, CONFIG,
            )

            self.logger.info("🔍 Validating step1_5_data_converter using existing validator")
            step1_5_result=await validator_orchestrator.run_step_validator(
                "step1_5_data_converter", training_input, pipeline_state, CONFIG,
            )

            # Print validation report
            self._print_step2_validation_report(step1_result, step1_5_result, symbol, exchange)

            # Check if we can proceed
            step1_passed=step1_result.get("validation_passed", False)
            step1_5_passed=step1_5_result.get("validation_passed", False)
            can_start=step1_passed and step1_5_passed

            if not can_start:
                self.logger.error("❌ Cannot start from step02 - data validation failed")
                self.logger.error("Please run step01 and step1_5 first to collect and process data")
                return False

            # Log warnings if any issues found
            step1_warnings=step1_result.get("warnings", [])
            step1_5_warnings=step1_5_result.get("warnings", [])
            total_warnings=len(step1_warnings) + len(step1_5_warnings)

            if total_warnings > 0:
                self.logger.warning(f"⚠️ Data validation found {total_warnings} warnings - proceeding with existing data")
                for warning in step1_warnings:
                    self.logger.warning(f"   • Step1: {warning}")
                for warning in step1_5_warnings:
                    self.logger.warning(f"   • Step1_5: {warning}")

            self.logger.info("✅ Data validation passed - proceeding with existing data")

        except Exception as e:
            self.logger.warning(f"⚠️ Could not run existing validators: {e}")
            self.logger.warning("Proceeding with basic file existence check")

            # Fallback to basic check
            consolidated_file=(
                f"data_cache/aggtrades_{exchange}_{symbol}_consolidated.parquet"
            )
            if not os.path.exists(consolidated_file):
                self.logger.exception(
                    f"❌ Consolidated data file not found: {consolidated_file}",
                )
                self.logger.exception(
                    "Please run data loading first or ensure consolidated data exists",
                )
                return False
            self.logger.info(f"✅ Found consolidated data: {consolidated_file}")

        return self._run_step_pipeline(
            symbol=symbol,
            exchange=exchange,
            start_step=start_step,
            force_rerun=force_rerun,
            with_gui=with_gui,
            training_mode="blank",  # Use blank mode for step02 with existing data
        )

    async def _validate_previous_steps(self, symbol: str, exchange: str, start_step: str) -> bool:
        """Validate all previous steps before starting from a specific step."""
        self.logger.info(f"🔍 Validating previous steps before starting from {start_step}")

        try:
            from src.utils.step_dependency_validator import StepDependencyValidator
            from src.utils.validator_orchestrator import ValidatorOrchestrator

            # Create validator orchestrator and dependency validator
            validator_orchestrator=ValidatorOrchestrator()
            dependency_validator=StepDependencyValidator()

            # Prepare training input for validation
            training_input={
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": "1m",
                "data_dir": "data_cache",
            }

            # Get step dependencies
            step_dependencies=dependency_validator.step_dependencies

            # Find all steps that need to be validated
            steps_to_validate = self._get_required_steps(start_step, step_dependencies)

            if not steps_to_validate:
                self.logger.info(f"✅ No previous steps to validate for {start_step}")
                return True

            self.logger.info(f"🔍 Validating {len(steps_to_validate)} previous steps: {steps_to_validate}")

            # Validate each required step
            validation_results={}
            all_passed = True

            for step in steps_to_validate:
                self.logger.info(f"🔍 Validating {step}...")
                try:
                    result=await validator_orchestrator.run_step_validator(
                        step, training_input, {}, CONFIG,
                    )
                    validation_results[step] = result

                    if result.get("validation_passed", False):
                        self.logger.info(f"✅ {step} validation passed")
                    else:
                        self.logger.error(f"❌ {step} validation failed: {result.get('error', 'Unknown error')}")
                        all_passed=False

                except Exception as e:
                    self.logger.exception(f"❌ Error validating {step}: {e}")
                    validation_results[step] = {"validation_passed": False, "error": str(e)}
                    all_passed=False

            # Print validation report
            self._print_validation_report(validation_results, symbol, exchange, start_step)

            return all_passed

        except Exception as e:
            self.logger.exception(f"❌ Error in step validation: {e}")
            return False

    def _get_required_steps(self, start_step: str, step_dependencies: dict) -> list:
        """Get all steps that need to be validated before starting from a specific step."""
        required_steps=[]

        # Use a simple approach: validate all steps that come before the start step
        step_order = [
            "step1_data_collection",           # Download and prepare market data
            "step1_5_data_converter",          # Convert data to unified format
            "step2_data_reading",              # Read and validate data quality
            "step2_5_sr_optimization",         # S/R detection optimization
            "step3_hmm_regime_discovery",      # Define HMM regime clusters (with basic features)
            "step3_5_final_regime_clustering", # Final regime clustering
            "step4_triple_barrier_method",     # Apply triple barrier method
            "step4_regime_data_splitting",     # Regime data splitting (legacy step)
            "step5_labeling",                  # Create labels
            "step6_feature_engineering",       # Complete feature engineering (simple + advanced)
            "step7_enhanced_matrix_operations", # Enhanced matrix operations for analysis
            "step8_regime_data_splitting",     # Split data by regimes
            "step9_hmm_based_training",        # HMM-based model training
            "step9_5_hmm_lm_generalist_training", # HMM LM generalist training
            "step10_unified_regime_intelligence", # Unified regime intelligence
            "step11_analyst_creation",         # Analyst creation (NEW STEP)
            "step12_analyst_enhancement",      # Analyst enhancement
            "step13_analyst_ensemble_creation", # Analyst ensemble creation
            "step14_tactician_labeling",       # Tactician labeling
            "step15_tactician_specialist_training", # Tactician specialist training
            "step16_confidence_calibration",   # Confidence calibration
            "step17_final_parameters_optimization", # Final parameters optimization
            "step18_walk_forward_validation",  # Walk forward validation
            "step19_monte_carlo_validation",   # Monte Carlo validation
            "step20_ab_testing",               # A/B testing
            "step21_saving",                   # Save final models
        ]

        try:
            start_index=step_order.index(start_step)
            required_steps=step_order[:start_index]
        except ValueError:
            self.logger.warning(f"⚠️ Unknown step {start_step}, skipping validation")
            return []

        return required_steps

    def _print_validation_report(self, validation_results: dict, symbol: str, exchange: str, start_step: str):
        """Print a formatted validation report."""
        print("\n" + "="*80)
        print("📊 STEP VALIDATION REPORT")
        print(f"🎯 Symbol: {symbol}")
        print(f"🏢 Exchange: {exchange}")
        print(f"🚀 Starting from: {start_step}")
        print("="*80)

        all_passed=True
        for step, result in validation_results.items():
            passed=result.get("validation_passed", False)
            status="✅ PASSED" if passed else "❌ FAILED"
            print(f"{step:<35} {status}")

            if not passed:
                all_passed=False
                error = result.get("error", "Unknown error")
                print(f"   Error: {error}")

        print("="*80)
        if all_passed:
            print("🎉 All previous steps validated successfully!")
        else:
            print("❌ Some previous steps failed validation")
        print("="*80)

    def _print_step2_validation_report(self, step1_result: dict, step1_5_result: dict, symbol: str, exchange: str):
        """Print a formatted validation report for step02 readiness."""
        print("\n" + "="*80)
        print("📊 DATA VALIDATION REPORT FOR STEP2")
        print(f"🎯 Symbol: {symbol}")
        print(f"🏢 Exchange: {exchange}")
        print("="*80)

        # Step1 status
        step1_passed=step1_result.get("validation_passed", False)
        step1_status="✅ PASSED" if step1_passed else "❌ FAILED"
        step1_warnings = step1_result.get("warnings", [])
        print(f"📁 Step1 Data Collection: {step1_status}")
        if step1_warnings:
            print(f"   ⚠️  Found {len(step1_warnings)} warnings")
            for warning in step1_warnings:
                print(f"     • {warning}")

        # Step1_5 status
        step1_5_passed=step1_5_result.get("validation_passed", False)
        step1_5_status="✅ PASSED" if step1_5_passed else "❌ FAILED"
        step1_5_warnings = step1_5_result.get("warnings", [])
        print(f"🔄 Step1_5 Data Converter: {step1_5_status}")
        if step1_5_warnings:
            print(f"   ⚠️  Found {len(step1_5_warnings)} warnings")
            for warning in step1_5_warnings:
                print(f"     • {warning}")

        # Show validation details if available
        if step1_result.get("details"):
            print(f"   📋 Step1 Details: {step1_result['details']}")
        if step1_5_result.get("details"):
            print(f"   📋 Step1_5 Details: {step1_5_result['details']}")

        # Overall assessment
        can_start=step1_passed and step1_5_passed
        if can_start:
            print("\n✅ READY TO START FROM STEP2")
            print("   Proceeding with existing data...")
        else:
            print("\n❌ NOT READY FOR STEP2")
            print("   Data validation failed - missing or invalid data")

        print("="*80 + "\n")

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="run_data_loading",
    )
    def run_data_loading(
        self,
        symbol: str,
        exchange: str,
        lookback_days: int=DEFAULT_LOOKBACK_DAYS,
    ) -> bool:
        """Run data loading and consolidation for the specified symbol and exchange."""
        try:
            self.logger.info(f"🔄 Starting data loading for {symbol} on {exchange}")

            # Set environment variable for blank training mode
            import os

            os.environ["BLANK_TRAINING_MODE"] = "1"

            # Step 1: Download data using optimized downloader
            self.logger.info("📥 Step 1: Downloading data...")
            download_script="backtesting/ares_data_downloader_optimized.py"

            if not os.path.exists(download_script):
                self.logger.error(f"❌ Download script not found: {download_script}")
                return False

            # Run the download script
            download_cmd=[
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
            env=os.environ.copy()
            env["BLANK_TRAINING_MODE"] = "1"
            download_result=subprocess.run(
                download_cmd,
                env=env,
                check=False,
            )

            if download_result.returncode != 0:
                self.logger.error(f"❌ Download failed: {download_result.stderr}")
                return False

            self.logger.info("✅ Data download completed successfully")

            # Step 2: Consolidate data using step1_data_collection
            self.logger.info("🔄 Step 2: Consolidating data...")
            consolidate_script="src/training/steps/step1_data_collection.py"

            if not os.path.exists(consolidate_script):
                self.logger.error(
                    f"❌ Consolidation script not found: {consolidate_script}",
                )
                return False

            # Run the consolidation script
            consolidate_cmd=[
                sys.executable,
                consolidate_script,
                symbol,
                exchange,  # This should be BINANCE
                "1000",  # min_data_points
                "data_cache",  # data_dir
                str(lookback_days),  # Pass lookback period as positional argument
                str(
                    CONFIG.get("DATA_CONFIG", {}).get("exclude_recent_days", 0),
                ),  # Exclude recent days
            ]

            self.logger.info(
                f"🔧 Running consolidation command: {' '.join(consolidate_cmd)}",
            )
            # Pass environment with BLANK_TRAINING_MODE set
            self.logger.info("🔄 Starting consolidation subprocess...")
            consolidate_result=subprocess.run(
                consolidate_cmd,
                env=env,
                check=False,
                timeout=1800,  # 30 minute timeout for large datasets
            )
            self.logger.info(
                f"🔄 Consolidation subprocess completed with return code: {consolidate_result.returncode}",
            )

            if consolidate_result.returncode != 0:
                self.logger.error(
                    f"❌ Consolidation failed: {consolidate_result.stderr}",
                )
                return False

            self.logger.info("✅ Data consolidation completed successfully")

            # Step 3: Convert consolidated data to ETHUSDT_1h.csv format
            if symbol== "ETHUSDT" and exchange == "BINANCE":
                self.logger.info(
                    "🔄 Step 3: Converting data to ETHUSDT_1h.csv format...",
                )
                from src.analyst.data_utils import create_ethusdt_1h_csv

                conversion_success=create_ethusdt_1h_csv()
                if conversion_success:
                    self.logger.info("✅ Data conversion completed successfully")
                else:
                    self.logger.warning("⚠️ Data conversion failed, but continuing...")
            else:
                self.logger.info(
                    f"⏭️ Skipping ETHUSDT_1h.csv conversion for {symbol} on {exchange}",
                )

            return True

        except Exception as e:
            self.logger.exception(f"❌ Data loading failed: {e}")
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
        with_gui: bool=False,
    ):
        """Run regime operations (HMM labeling or ML training) with optional GUI."""
        self.logger.info(f"🧠 Running regime operations for {symbol} on {exchange}")
        self.logger.info(f"📋 Subcommand: {subcommand}")
        self.logger.info(f"🖥️ GUI mode: {with_gui}")
        self.logger.info(
            f"⏰ Start time: {format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S')}",
        )

        if with_gui and not self.launch_gui("regime", symbol, exchange):
            return False

        try:
            self.logger.info("📦 Importing required modules...")
            # Import UnifiedRegimeClassifier
            from src.analyst.unified_regime_classifier import UnifiedRegimeClassifier
            from src.config import CONFIG

            self.logger.info("✅ Modules imported successfully")

            self.logger.info("🔧 Initializing unified regime classifier...")
            # Initialize unified regime classifier
            regime_classifier=UnifiedRegimeClassifier(CONFIG, exchange, symbol)
            self.logger.info("✅ Unified regime classifier initialized successfully")

            if subcommand== "load":
                print(
                    f"🚀 Starting unified regime classifier training for {symbol} on {exchange}...",
                )

                # Load historical data from data directory
                data_file=f"data/{symbol}_1h.csv"
                if not os.path.exists(data_file):
                    self.logger.error(f"❌ Data file not found: {data_file}")
                    print(f"❌ Data file not found: {data_file}")
                    print(
                        "Please run data loading first: python ares_launcher.py load --symbol ETHUSDT --exchange BINANCE",
                    )
                    return False

                from src.analyst.data_utils import load_klines_data

                historical_data=load_klines_data(data_file)

                if historical_data is None or historical_data.empty:
                    self.logger.error("❌ Failed to load historical data")
                    print("❌ Failed to load historical data")
                    return False

                # Train unified regime classifier
                success=await regime_classifier.train_complete_system(historical_data)

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

            if subcommand== "train":
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
                data_file=f"data/{symbol}_1h.csv"
                if not os.path.exists(data_file):
                    self.logger.error(f"❌ Data file not found: {data_file}")
                    print(f"❌ Data file not found: {data_file}")
                    print(
                        "Please run data loading first: python ares_launcher.py load --symbol ETHUSDT --exchange BINANCE",
                    )
                    return False

                from src.analyst.data_utils import load_klines_data

                historical_data=load_klines_data(data_file)

                if historical_data is None or historical_data.empty:
                    self.logger.error("❌ Failed to load historical data")
                    print("❌ Failed to load historical data")
                    return False

                # Train unified regime classifier
                success=await regime_classifier.train_complete_system(historical_data)

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

            if subcommand== "train_blank":
                print(
                    f"🚀 Starting unified regime classifier training for {symbol} on {exchange} (30 days data)...",
                )

                # Load historical data
                from src.analyst.data_utils import load_klines_data

                # Load 30 days of data for quick training
                historical_data=await load_klines_data(
                    symbol,
                    exchange,
                    lookback_days=60,
                )

                if historical_data is None or historical_data.empty:
                    self.logger.error("❌ Failed to load historical data")
                    print("❌ Failed to load historical data")
                    return False

                # Train unified regime classifier
                success=await regime_classifier.train_complete_system(historical_data)

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
            self.logger.exception(f"❌ Failed to run regime operations: {e}")
            print(f"❌ Failed to run regime operations: {e}")
            return False

    def _run_step_pipeline(
        self,
        symbol: str,
        exchange: str,
        start_step: str,
        force_rerun: bool,
        with_gui: bool,
        training_mode: str,
    ) -> bool:
        """Common implementation for step-based training (blank/full) to reduce duplication."""
        # Normalize and log step
        start_step=self._normalize_step_name(start_step)
        self.logger.info(f"Starting from step: {start_step}")

        import os

        from src.config import CONFIG
        from src.training.step_orchestrator import StepOrchestrator

        # Set training mode environment
        if training_mode== "light":
            os.environ["LIGHT_TRAINING_MODE"] = "1"
            os.environ["BLANK_TRAINING_MODE"] = "0"
            os.environ["FULL_TRAINING_MODE"] = "0"
            self.logger.info(
                "💡 LIGHT TRAINING MODE: Set LIGHT_TRAINING_MODE=1 for step-based training (30 days)",
            )
        elif training_mode== "blank":
            os.environ["BLANK_TRAINING_MODE"] = "1"
            os.environ["LIGHT_TRAINING_MODE"] = "0"
            os.environ["FULL_TRAINING_MODE"] = "0"
            self.logger.info(
                "🧪 BLANK TRAINING MODE: Set BLANK_TRAINING_MODE=1 for step-based training (180 days)",
            )
        elif training_mode== "full":
            os.environ["FULL_TRAINING_MODE"] = "1"
            os.environ["LIGHT_TRAINING_MODE"] = "0"
            os.environ["BLANK_TRAINING_MODE"] = "0"
            self.logger.info(
                "📊 FULL TRAINING MODE: Set FULL_TRAINING_MODE=1 for step-based training (730 days)",
            )

        # Prevent blank mode with step01 data collection
        if training_mode== "blank" and start_step == "step1_data_collection":
            self.logger.error("❌ Cannot use blank mode with step1_data_collection")
            self.logger.error(
                "Blank mode is designed for quick testing with limited data",
            )
            self.logger.error(
                "step1_data_collection processes all available data files",
            )
            self.logger.error("Use one of the following instead:")
            self.logger.error(
                "  - python ares_launcher.py load --symbol ETHUSDT --exchange BINANCE (for full data)",
            )
            self.logger.error(
                "  - python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step2_processing_labeling_feature_engineering (for blank mode)",
            )
            return False

        if with_gui and not self.launch_gui("training", symbol, exchange):
            return False

        try:
            # Initialize step orchestrator
            orchestrator=StepOrchestrator(symbol, exchange)

            # When forcing, set env flags and clear progress/checkpoints from the start step
            if force_rerun:
                # Set FORCE for fresh runs; EnhancedTrainingManager recognizes this env flag.
                os.environ["FORCE"] = "1"
                self._force_fresh_start_from_step(orchestrator, start_step)
                self._clear_checkpoint_files(symbol, exchange, timeframe="1m")

            # Validation is now handled by EnhancedTrainingManager
            self.logger.info("🔍 Step validation will be performed by EnhancedTrainingManager")

            # Run the step-based training using the orchestrator
            import asyncio

            success=asyncio.run(
                orchestrator.execute_from_step(
                    start_step=start_step, config=CONFIG, force_rerun=force_rerun,
                ),
            )

            if success:
                self.logger.info(
                    "✅ Step-based training pipeline completed successfully",
                )
                return True
            self.logger.error("❌ Step-based training pipeline failed")
            return False

        except Exception as e:
            self.logger.exception(f"❌ Failed to run step-based training pipeline: {e}")
            return False


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser=argparse.ArgumentParser(
        description="Ares Trading Bot Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ares_launcher.py paper --symbol ETHUSDT --exchange BINANCE
  python ares_launcher.py backtest --symbol ETHUSDT --exchange BINANCE --gui
  python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --gui
  python ares_launcher.py light --symbol ETHUSDT --exchange BINANCE
      python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step4_regime_data_splitting
    python ares_launcher.py step02 --symbol ETHUSDT --exchange BINANCE  # Start from step02 with existing data
    python ares_launcher.py step02 --symbol ETHUSDT --exchange BINANCE --step step2_data_reading  # Specific step02

    # New step-based commands with validation
    python ares_launcher.py step01 --symbol ETHUSDT --exchange BINANCE --training-mode light
    python ares_launcher.py step2_5 --symbol ETHUSDT --exchange BINANCE --training-mode blank
    python ares_launcher.py step3_5 --symbol ETHUSDT --exchange BINANCE --training-mode blank
    python ares_launcher.py step04 --symbol ETHUSDT --exchange BINANCE --training-mode blank
    python ares_launcher.py step08 --symbol ETHUSDT --exchange BINANCE --training-mode full
    python ares_launcher.py step9_5 --symbol ETHUSDT --exchange BINANCE --training-mode blank
    python ares_launcher.py step05 --symbol ETHUSDT --exchange BINANCE --training-mode light --force
    python ares_launcher.py step10 --symbol ETHUSDT --exchange BINANCE --training-mode blank --gui
    python ares_launcher.py step18 --symbol ETHUSDT --exchange BINANCE --training-mode full
    python ares_launcher.py step21 --symbol ETHUSDT --exchange BINANCE --training-mode full

    # Individual pipeline execution (organized structure)
    python ares_launcher.py data-collection --symbol ETHUSDT --exchange BINANCE
    python ares_launcher.py market-analysis --symbol ETHUSDT --exchange BINANCE
    python ares_launcher.py model-training --symbol ETHUSDT --exchange BINANCE
    python ares_launcher.py optimisation --symbol ETHUSDT --exchange BINANCE
    python ares_launcher.py backtesting --symbol ETHUSDT --exchange BINANCE
    python ares_launcher.py all-pipelines --symbol ETHUSDT --exchange BINANCE

    # Legacy step-based commands (still supported)
    python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step1_data_collection
    python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step2_processing_labeling_feature_engineering
    python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step2_5_sr_optimization
    python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step3_feature_engineering
    python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step3_5_final_regime_clustering
    python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step4_regime_data_splitting
    python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step5_hmm_based_training
    python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step6_analyst_enhancement
    python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step7_analyst_ensemble_creation
    python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step8_tactician_labeling
    python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step9_tactician_specialist_training
    python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step9_5_hmm_lm_generalist_training
    python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step10_unified_regime_intelligence
    python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step11_analyst_creation
    python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step12_analyst_enhancement
    python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step13_analyst_ensemble_creation
    python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step14_tactician_labeling
    python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step15_tactician_specialist_training
    python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step16_confidence_calibration
    python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step17_final_parameters_optimization
    python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step18_walk_forward_validation
    python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step19_monte_carlo_validation
    python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step20_ab_testing
    python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step21_saving
    python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step2_processing_labeling_feature_engineering --force
    python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE --step step5_hmm_based_training --force --gui
  python ares_launcher.py live --symbol ETHUSDT --exchange BINANCE
  python ares_launcher.py load --symbol ETHUSDT --exchange BINANCE  # Safe: only downloads missing data
  python ares_launcher.py load --symbol ETHUSDT --exchange MEXC    # Safe: only downloads missing data
  python ares_launcher.py load --symbol ETHUSDT --exchange GATEIO  # Safe: only downloads missing data
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
            # "model_trainer",  # REMOVED: Use blank command with step5_hmm_based_training instead
            "live",
            "portfolio",
            "gui",
            "light",
            "blank",
            "step02",  # New command to start from step02 with existing data
            "full",
            "multi-timeframe",
            "load",
            "regime",
            "precompute",
            "resume",
            "modes",  # Show available training modes
            # New pipeline commands (organized structure)
            "data-collection", "market-analysis", "model-training", "optimisation", "backtesting", "all-pipelines",
            # New step-based commands
            "step01", "step1_5", "step02", "step2_5", "step03", "step3_5", "step04", "step05", "step06", "step07", "step08",
            "step8_5", "step09", "step9_5", "step10", "step11", "step12", "step13", "step14", "step15", "step16", "step17",
            "step18", "step19", "step20", "step21",
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
            # "model_trainer",  # REMOVED: Use blank command with step5_hmm_based_training instead
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
        help=(
            "Start training from a specific step (e.g., step2_processing_labeling_feature_engineering). "
            "Legacy names like step3_regime_data_splitting are accepted and auto-normalized."
        ),
    )

    parser.add_argument(
        "--training-mode",
        type=str,
        choices=["light", "blank", "full"],
        default="blank",
        help="Training mode for step-based commands: light (30 days), blank (180 days), full (730 days). Default: blank",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a fresh run starting from the specified step (clears progress and checkpoints from that step). Not available for 'load' command.",
    )

    # Backward compatibility alias
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="[Deprecated] Use --force instead. Force rerun of completed steps. Not available for 'load' command.",
    )

    parser.add_argument(
        "--lookback-days",
        type=int,
        help="Override the default lookback period for the training mode (in days). Use with caution.",
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate parsed arguments."""
    if args.command== "gui" and args.mode:
        if not args.symbol or not args.exchange:
            msg = "Symbol and exchange are required when using gui with mode"
            raise ValueError(
                msg,
            )

    # Validate that force flag is not used with load command
    force_flag=bool(
        getattr(args, "force", False) or getattr(args, "force_rerun", False),
    )
    if args.command== "load" and force_flag:
        msg = (
            "The --force flag is not available for the 'load' command. "
            "The load command is designed to be safe and only downloads missing data."
        )
        raise ValueError(
            msg,
        )

    commands_requiring_symbol=[
        "paper",
        "challenger",
        "backtest",
        # "model_trainer",  # REMOVED: Use blank command with step5_hmm_based_training instead
        "live",
        "light",
        "blank",
        "full",
        "multi-timeframe",
        "load",
        "precompute",
        # New pipeline commands (organized structure)
        "data-collection", "market-analysis", "model-training", "optimisation", "backtesting", "all-pipelines",
        # Step-based commands
        "step01", "step1_5", "step02", "step2_5", "step03", "step3_5", "step04", "step05", "step06", "step07", "step08",
        "step8_5", "step09", "step9_5", "step10", "step11", "step12", "step13", "step14", "step15", "step16", "step17",
        "step18", "step19", "step20", "step21",
    ]

    if args.command in commands_requiring_symbol and not args.symbol:
        msg = f"Symbol is required for {args.command} command"
        raise ValueError(msg)


def initialize_launcher() -> tuple[AresLauncher, object]:
    """Initialize launcher with signal handling."""
    signal_handler=setup_signal_handlers()
    launcher=AresLauncher()
    launcher.setup_logging()

    # Add cleanup callback to signal handler
    signal_handler.register_shutdown_callback(launcher.cleanup)

    return launcher, signal_handler


def execute_command(launcher: AresLauncher, args: argparse.Namespace) -> bool:
    """Execute the requested command based on parsed arguments."""
    print(f"🔍 DEBUG: Executing command: {args.command}")
    print(f"🔍 DEBUG: Symbol: {args.symbol}, Exchange: {args.exchange}")

    # Normalize input step name to current naming, and collapse force flags
    normalized_step=launcher._normalize_step_name(getattr(args, "step", None))
    force_flag=bool(
        getattr(args, "force", False) or getattr(args, "force_rerun", False),
    )

    command_handlers={
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
        "blank": lambda: launcher._run_unified_training(
            args.symbol,
            args.exchange,
            training_mode="blank",
            lookback_days=getattr(args, "lookback_days", None),
            with_gui=args.gui,
        ),
        "light": lambda: launcher._run_unified_training(
            args.symbol,
            args.exchange,
            training_mode="light",
            lookback_days=getattr(args, "lookback_days", None),
            with_gui=args.gui,
        ),
        "step02": lambda: asyncio.run(
            launcher.run_step2_with_existing_data(
                args.symbol,
                args.exchange,
                start_step=normalized_step or "step2_data_reading",
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        # New step-based commands with validation
        "step01": lambda: asyncio.run(
            launcher.run_step_based_training_with_validation(
                args.symbol,
                args.exchange,
                start_step="step1_data_collection",
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        "step1_5": lambda: asyncio.run(
            launcher.run_step_based_training_with_validation(
                args.symbol,
                args.exchange,
                start_step="step1_5_data_converter",
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        "step2_5": lambda: asyncio.run(
            launcher.run_step_based_training_with_validation(
                args.symbol,
                args.exchange,
                start_step="step2_5_sr_optimization",
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        "step03": lambda: asyncio.run(
            launcher.run_step_based_training_with_validation(
                args.symbol,
                args.exchange,
                start_step="step3_hmm_regime_discovery",
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        "step3_5": lambda: asyncio.run(
            launcher.run_step_based_training_with_validation(
                args.symbol,
                args.exchange,
                start_step="step3_5_final_regime_clustering",
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        "step04": lambda: asyncio.run(
            launcher.run_step_based_training_with_validation(
                args.symbol,
                args.exchange,
                start_step="step4_regime_data_splitting",
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        "step05": lambda: asyncio.run(
            launcher.run_step_based_training_with_validation(
                args.symbol,
                args.exchange,
                start_step="step5_labeling",
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        "step06": lambda: asyncio.run(
            launcher.run_step_based_training_with_validation(
                args.symbol,
                args.exchange,
                start_step="step6_feature_engineering",
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        "step07": lambda: asyncio.run(
            launcher.run_step_based_training_with_validation(
                args.symbol,
                args.exchange,
                start_step="step7_regime_data_splitting",
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        "step08": lambda: asyncio.run(
            launcher.run_step_based_training_with_validation(
                args.symbol,
                args.exchange,
                start_step="step8_hmm_based_training",
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        "step8_5": lambda: asyncio.run(
            launcher.run_step_based_training_with_validation(
                args.symbol,
                args.exchange,
                start_step="step8_5_unified_regime_intelligence",
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        "step09": lambda: asyncio.run(
            launcher.run_step_based_training_with_validation(
                args.symbol,
                args.exchange,
                start_step="step9_hmm_based_training",
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        "step9_5": lambda: asyncio.run(
            launcher.run_step_based_training_with_validation(
                args.symbol,
                args.exchange,
                start_step="step9_5_hmm_lm_generalist_training",
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        "step10": lambda: asyncio.run(
            launcher.run_step_based_training_with_validation(
                args.symbol,
                args.exchange,
                start_step="step10_unified_regime_intelligence",
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        "step10": lambda: asyncio.run(
            launcher.run_step_based_training_with_validation(
                args.symbol,
                args.exchange,
                start_step="step10_tactician_labeling",
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        "step11": lambda: asyncio.run(
            launcher.run_step_based_training_with_validation(
                args.symbol,
                args.exchange,
                start_step="step11_tactician_specialist_training",
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        "step12": lambda: asyncio.run(
            launcher.run_step_based_training_with_validation(
                args.symbol,
                args.exchange,
                start_step="step12_confidence_calibration",
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        "step13": lambda: asyncio.run(
            launcher.run_step_based_training_with_validation(
                args.symbol,
                args.exchange,
                start_step="step13_final_parameters_optimization",
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        "step14": lambda: asyncio.run(
            launcher.run_step_based_training_with_validation(
                args.symbol,
                args.exchange,
                start_step="step14_walk_forward_validation",
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        "step15": lambda: asyncio.run(
            launcher.run_step_based_training_with_validation(
                args.symbol,
                args.exchange,
                start_step="step15_monte_carlo_validation",
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        "step16": lambda: asyncio.run(
            launcher.run_step_based_training_with_validation(
                args.symbol,
                args.exchange,
                start_step="step16_ab_testing",
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        "step17": lambda: asyncio.run(
            launcher.run_step_based_training_with_validation(
                args.symbol,
                args.exchange,
                start_step="step17_final_parameters_optimization",
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        "step18": lambda: asyncio.run(
            launcher.run_step_based_training_with_validation(
                args.symbol,
                args.exchange,
                start_step="step18_walk_forward_validation",
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        "step19": lambda: asyncio.run(
            launcher.run_step_based_training_with_validation(
                args.symbol,
                args.exchange,
                start_step="step19_monte_carlo_validation",
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        "step20": lambda: asyncio.run(
            launcher.run_step_based_training_with_validation(
                args.symbol,
                args.exchange,
                start_step="step20_ab_testing",
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        "step21": lambda: asyncio.run(
            launcher.run_step_based_training_with_validation(
                args.symbol,
                args.exchange,
                start_step="step21_saving",
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            ),
        ),
        "full": lambda: launcher._run_unified_training(
            args.symbol,
            args.exchange,
            training_mode="full",
            lookback_days=getattr(args, "lookback_days", None),
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
            lookback_days=DEFAULT_LOOKBACK_DAYS
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
        "modes": lambda: launcher.show_training_modes(),
        # New pipeline commands (organized structure)
        "data-collection": lambda: launcher.run_data_collection_pipeline(
            args.symbol,
            args.exchange,
            with_gui=args.gui,
        ),
        "market-analysis": lambda: launcher.run_market_analysis_pipeline(
            args.symbol,
            args.exchange,
            with_gui=args.gui,
        ),
        "model-training": lambda: launcher.run_model_training_pipeline(
            args.symbol,
            args.exchange,
            with_gui=args.gui,
        ),
        "optimisation": lambda: launcher.run_optimisation_pipeline(
            args.symbol,
            args.exchange,
            with_gui=args.gui,
        ),
        "backtesting": lambda: launcher.run_backtesting_pipeline(
            args.symbol,
            args.exchange,
            with_gui=args.gui,
        ),
        "all-pipelines": lambda: launcher.run_all_pipelines(
            args.symbol,
            args.exchange,
            with_gui=args.gui,
        ),
    }

    if args.command in command_handlers:
        print(f"🔍 DEBUG: Found command handler for '{args.command}'")
        success=command_handlers[args.command]()
        print(f"🔍 DEBUG: Command execution result: {success}")
        if not success:
            return False
        if args.gui:
            launcher.wait_for_user_input()
        return True

    if args.command== "gui":
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
        success=launcher.run_gui_with_mode(args.mode, args.symbol, args.exchange)
        if not success:
            return False
        launcher.wait_for_user_input()
        return True
    success=launcher.run_gui_only()
    if not success:
        return False
    launcher.wait_for_user_input()
    return True


@handle_errors(exceptions=(Exception,), default_return=1, context="main")
def main():
    """Main entry point for the Ares launcher."""
    try:
        # Parse and validate arguments
        args=parse_arguments()
        validate_arguments(args)

        # Initialize launcher (this sets up comprehensive logging)
        launcher, signal_handler=initialize_launcher()

        # Log command execution
        command_info=f"Command: {args.command}"
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
        success=execute_command(launcher, args)

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


if __name__== "__main__":
    main()
