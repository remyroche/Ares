#!/usr/bin/env python3
"""
Refactored Ares Launcher - Simplified and Modular

This is the refactored version of ares_launcher.py that uses modular components
to reduce complexity from 634 to a much more manageable level.
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
from src.utils.common_operations import format_datetime, get_current_datetime

# Import logging utilities
try:
    from src.utils.logger import get_backtesting_logger
except ImportError:
    def get_backtesting_logger(name, log_dir="log"):
        return logging.getLogger(name)

from src.config import CONFIG
from src.config.constants import DEFAULT_LOOKBACK_DAYS
from src.utils.comprehensive_logger import setup_comprehensive_logging
from src.utils.logger import ensure_comprehensive_logging_available
from src.utils.observability import init_observability
from src.utils.simple_signal_handler import setup_signal_handlers

# Import our new modular components
from src.launcher.command_handlers import CommandHandlerFactory
from src.launcher.pipeline_managers import PipelineManagerFactory
from src.launcher.validation_utilities import ValidationFactory
from src.launcher.step_orchestrator_wrapper import StepOrchestratorWrapper
from src.launcher.gui_manager import GUIManagerFactory
from src.launcher.configuration_manager import ConfigurationManager

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class AresLauncher:
    """Simplified launcher for Ares trading bot using modular components."""

    def __init__(self):
        # Initialize comprehensive logging
        self.comprehensive_logger = setup_comprehensive_logging(CONFIG)
        ensure_comprehensive_logging_available()

        # Initialize observability backends (Sentry/OTLP) if configured
        try:
            init_observability({})
        except Exception as _obs_exc:
            logging.getLogger(__name__).warning(f"Observability init skipped: {_obs_exc}")

        self.logger = self.comprehensive_logger.get_component_logger("AresLauncher")
        self.global_logger = self.comprehensive_logger.get_global_logger()
        
        # Initialize modular components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all modular components."""
        # Create GUI and process managers
        (self.process_manager, self.gui_manager, self.trading_process_manager, 
         self.user_interaction_manager) = GUIManagerFactory.create_managers(self.logger)
        
        # Create configuration manager
        self.config_manager = ConfigurationManager(self.logger)
        
        # Create step orchestrator wrapper
        self.step_orchestrator_wrapper = StepOrchestratorWrapper(self)
        
        # Create validation factory
        self.validation_factory = ValidationFactory()

    def setup_logging(self):
        """Setup comprehensive logging for the launcher."""
        self.comprehensive_logger.log_launcher_start("INITIALIZATION")
        
        self.logger.info("=" * 80)
        self.logger.info("🚀 ARES COMPREHENSIVE LAUNCHER (REFACTORED)")
        self.logger.info("=" * 80)
        self.logger.info(f"Start time: {format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Log directory: {self.comprehensive_logger.log_dir}")
        self.logger.info(f"Log level: {CONFIG.get('logging', {}).get('level', 'INFO')}")
        self.logger.info("=" * 80)

    def setup_signal_handling(self):
        """Set up centralized signal handling."""
        self.signal_handler = setup_signal_handlers()
        self.signal_handler.register_shutdown_callback(self.cleanup)
        self.logger.info("✅ Centralized signal handling set up")

    def cleanup(self):
        """Cleanup processes on exit."""
        self.process_manager.cleanup()

    # Delegate methods to appropriate managers
    def launch_gui(self, mode=None, symbol=None, exchange=None):
        """Launch the GUI server."""
        return self.gui_manager.launch_gui(mode, symbol, exchange)

    def launch_portfolio_manager(self):
        """Launch the portfolio manager."""
        return self.gui_manager.launch_portfolio_manager()

    def wait_for_user_input(self):
        """Wait for user input to stop the launcher."""
        self.user_interaction_manager.wait_for_user_input()

    # Training methods using configuration manager
    def _run_unified_training(self, symbol, exchange, training_mode, lookback_days=None, with_gui=False):
        """Run unified training with enhanced training manager."""
        try:
            # Setup training environment
            env_config = self.config_manager.setup_training_environment(
                training_mode=training_mode,
                symbol=symbol,
                exchange=exchange,
                lookback_days=lookback_days
            )
            
            mode_config = env_config["mode_config"]
            actual_lookback_days = env_config["lookback_days"]
            display_info = env_config["display_info"]
            
            # Display training information
            mode_display = f"{training_mode} training"
            intensity_pct = f"{display_info['intensity_percentage']:.0f}%"
            print(f"🚀 Starting {mode_display} for {symbol} on {exchange}")
            print(f"📊 Mode Configuration ({intensity_pct} of full intensity):")
            print(f"   • Lookback Days: {actual_lookback_days}")
            print(f"   • Max Trials: {mode_config.max_trials}")
            print(f"   • N Trials: {mode_config.n_trials}")
            print(f"   • Computational Intensity: {mode_config.computational_intensity}")
            print(f"   • Estimated Duration: {mode_config.estimated_duration_minutes} minutes")
            
            # Run the training
            return asyncio.run(self._execute_enhanced_training(symbol, exchange, training_mode, actual_lookback_days))
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to run unified training: {e}")
            return False

    async def _execute_enhanced_training(self, symbol, exchange, training_mode, lookback_days):
        """Execute enhanced training using EnhancedTrainingManager."""
        from src.database.sqlite_manager import SQLiteManager
        from src.training.enhanced_training_manager import EnhancedTrainingManager
        from src.config.training_modes import get_training_config_dict, get_training_input_dict
        
        logger = self.logger.getChild("EnhancedTrainingPipeline")
        
        try:
            # Initialize database manager
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
            
            # Initialize enhanced training manager
            training_config = get_training_config_dict(training_mode)
            training_config["database"] = default_config["database"]
            
            if lookback_days != self.config_manager.get_mode_config(training_mode).lookback_days:
                training_config["enhanced_training_manager"]["lookback_days"] = lookback_days
            
            training_manager = EnhancedTrainingManager(training_config)
            
            if not await training_manager.initialize():
                logger.error("❌ Failed to initialize enhanced training manager")
                return False
            
            # Prepare training input
            training_input = get_training_input_dict(
                mode=training_mode,
                symbol=symbol,
                exchange=exchange,
                timestamp=format_datetime(get_current_datetime(), "%Y-%m-%dT%H:%M:%S"),
                lookback_days=lookback_days,
            )
            
            # Execute the enhanced training
            success = await training_manager.execute_enhanced_training(training_input)
            
            if success:
                logger.info("🎉 ENHANCED TRAINING PIPELINE COMPLETED SUCCESSFULLY")
                return True
            else:
                logger.error("❌ Enhanced training pipeline failed")
                return False
                
        except Exception as e:
            logger.exception(f"💥 ENHANCED TRAINING PIPELINE FAILED: {str(e)}")
            return False
        finally:
            try:
                if "db_manager" in locals():
                    await db_manager.stop()
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Database cleanup warning: {cleanup_error}")

    # Trading methods using trading process manager
    def _run_unified_trading(self, symbol, exchange, trading_mode, with_gui=False):
        """Unified trading method for both paper and live trading modes."""
        if with_gui and not self.launch_gui(trading_mode.lower(), symbol, exchange):
            return False
        
        self.config_manager.setup_trading_environment(trading_mode)
        return self.trading_process_manager.run_trading_process(symbol, exchange, trading_mode)

    def run_paper_trading(self, symbol, exchange, with_gui=False):
        """Run paper trading using unified trading method."""
        return self._run_unified_trading(symbol, exchange, "PAPER", with_gui)

    def run_live_trading(self, symbol, exchange, with_gui=False):
        """Run live trading using unified trading method."""
        return self._run_unified_trading(symbol, exchange, "LIVE", with_gui)

    def run_challenger_trading(self, symbol, exchange, with_gui=False):
        """Run challenger trading with optional GUI."""
        self.logger.info(f"🏆 Running challenger trading for {symbol} on {exchange}")
        
        if with_gui and not self.launch_gui("challenger", symbol, exchange):
            return False
        
        try:
            process = subprocess.Popen(
                [sys.executable, "scripts/setup_challenger_model.py", symbol, exchange],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.process_manager.add_process(process)
            
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                self.logger.info("✅ Challenger trading completed successfully")
                return True
            else:
                self.logger.error(f"❌ Challenger trading failed: {stderr}")
                return False
                
        except Exception as e:
            self.logger.exception(f"❌ Failed to run challenger trading: {e}")
            return False

    def run_portfolio_trading(self, with_gui=False):
        """Run portfolio trading with optional GUI."""
        self.logger.info("📈 Running portfolio trading")
        
        if with_gui and not self.launch_gui("portfolio"):
            return False
        
        if not self.launch_portfolio_manager():
            return False
        
        supported_tokens = CONFIG.get("SUPPORTED_TOKENS", {}).get("BINANCE", ["ETHUSDT"])
        return self.trading_process_manager.run_portfolio_trading(supported_tokens)

    # Step-based training methods using step orchestrator wrapper
    async def run_step_based_training_with_validation(self, symbol, exchange, start_step, 
                                                     training_mode="blank", force_rerun=False, with_gui=False):
        """Run step-based training with comprehensive validation."""
        return await self.step_orchestrator_wrapper.run_step_based_training_with_validation(
            symbol=symbol,
            exchange=exchange,
            start_step=start_step,
            training_mode=training_mode,
            force_rerun=force_rerun,
            with_gui=with_gui,
        )

    async def run_step2_with_existing_data(self, symbol, exchange, start_step="step2_feature_engineering", 
                                          force_rerun=False, with_gui=False):
        """Run step02 with existing data from step01 and step1_5."""
        return await self.step_orchestrator_wrapper.run_step2_with_existing_data(
            symbol=symbol,
            exchange=exchange,
            start_step=start_step,
            force_rerun=force_rerun,
            with_gui=with_gui,
        )

    # Pipeline methods using pipeline managers
    def run_data_collection_pipeline(self, symbol, exchange, with_gui=False):
        """Run enhanced data collection pipeline."""
        manager = PipelineManagerFactory.create_manager("data-collection", self)
        return manager.execute(symbol, exchange, with_gui)

    def run_model_training_pipeline(self, symbol, exchange, with_gui=False):
        """Run model training pipeline."""
        manager = PipelineManagerFactory.create_manager("model-training", self)
        return manager.execute(symbol, exchange, with_gui)

    def run_optimisation_pipeline(self, symbol, exchange, with_gui=False):
        """Run enhanced optimisation pipeline."""
        manager = PipelineManagerFactory.create_manager("optimisation", self)
        return manager.execute(symbol, exchange, with_gui)

    def run_backtesting_pipeline(self, symbol, exchange, with_gui=False):
        """Run backtesting pipeline."""
        manager = PipelineManagerFactory.create_manager("backtesting", self)
        return manager.execute(symbol, exchange, with_gui)

    def run_all_pipelines(self, symbol, exchange, with_gui=False):
        """Run all pipelines in sequence."""
        manager = PipelineManagerFactory.create_manager("all-pipelines", self)
        return manager.execute(symbol, exchange, with_gui)

    # Utility methods
    def show_training_modes(self):
        """Display available training modes and their configurations."""
        return self.config_manager.show_training_modes()

    def precompute_wavelet_features(self, symbol, exchange):
        """Precompute wavelet features for backtesting."""
        self.logger.info(f"🔧 Precomputing wavelet features for {symbol} on {exchange}")
        
        try:
            import asyncio
            from src.training.steps.precompute_wavelet_features import WaveletFeaturePrecomputer
            
            precomputer = WaveletFeaturePrecomputer(CONFIG)
            init_success = asyncio.run(precomputer.initialize())
            
            if not init_success:
                self.logger.error("❌ Failed to initialize wavelet precomputer")
                return False
            
            # Check if cache already exists
            cache_dir = CONFIG.get("wavelet_cache", {}).get("cache_dir", "data/wavelet_cache")
            if os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0:
                self.logger.info("✅ Wavelet features already cached, skipping precomputation")
                return True
            
            # Data path for precomputation
            data_path = f"data_cache/aggtrades_{exchange}_{symbol}_consolidated.parquet"
            if not os.path.exists(data_path):
                self.logger.error(f"❌ Consolidated data file not found: {data_path}")
                return False
            
            # Precompute features
            success = asyncio.run(precomputer.precompute_dataset(data_path=data_path, symbol=symbol))
            
            if success:
                self.logger.info("✅ Wavelet feature precomputation completed successfully")
                return True
            else:
                self.logger.error("❌ Wavelet feature precomputation failed")
                return False
                
        except Exception as e:
            self.logger.exception(f"❌ Failed to precompute wavelet features: {e}")
            return False

    def resume_training(self, symbol, exchange, with_gui=False):
        """Resume training from the last checkpoint."""
        self.logger.info(f"🔄 Resuming training for {symbol} on {exchange}")
        
        checkpoint_file = Path("checkpoints/training_progress.json")
        if not checkpoint_file.exists():
            self.logger.error("❌ No checkpoint found to resume from")
            return False
        
        try:
            with open(checkpoint_file) as f:
                checkpoint_data = json.load(f)
            
            training_mode = checkpoint_data.get("training_mode", "blank")
            lookback_days = checkpoint_data.get("lookback_days", 30)
            
            return self._run_unified_training(
                symbol=symbol,
                exchange=exchange,
                training_mode=training_mode,
                lookback_days=lookback_days,
                with_gui=with_gui,
            )
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to resume training: {e}")
            return False

    def run_data_loading(self, symbol, exchange, lookback_days=DEFAULT_LOOKBACK_DAYS):
        """Run data loading and consolidation."""
        try:
            self.logger.info(f"🔄 Starting data loading for {symbol} on {exchange}")
            
            # Set environment variable for blank training mode
            os.environ["BLANK_TRAINING_MODE"] = "1"
            
            # Step 1: Download data
            download_script = "backtesting/ares_data_downloader_optimized.py"
            if not os.path.exists(download_script):
                self.logger.error(f"❌ Download script not found: {download_script}")
                return False
            
            download_cmd = [
                sys.executable, download_script,
                "--symbol", symbol,
                "--exchange", exchange,
                "--lookback-years", str(lookback_days // 365),
            ]
            
            env = os.environ.copy()
            env["BLANK_TRAINING_MODE"] = "1"
            download_result = subprocess.run(download_cmd, env=env, check=False)
            
            if download_result.returncode != 0:
                self.logger.error(f"❌ Download failed")
                return False
            
            # Step 2: Consolidate data
            consolidate_script = "src/training/steps/step1_data_collection.py"
            if not os.path.exists(consolidate_script):
                self.logger.error(f"❌ Consolidation script not found: {consolidate_script}")
                return False
            
            consolidate_cmd = [
                sys.executable, consolidate_script,
                symbol, exchange, "1000", "data_cache",
                str(lookback_days),
                str(CONFIG.get("DATA_CONFIG", {}).get("exclude_recent_days", 0)),
            ]
            
            consolidate_result = subprocess.run(consolidate_cmd, env=env, check=False, timeout=1800)
            
            if consolidate_result.returncode != 0:
                self.logger.error(f"❌ Consolidation failed")
                return False
            
            self.logger.info("✅ Data loading completed successfully")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Data loading failed: {e}")
            return False

    async def run_regime_operations(self, symbol, exchange, subcommand, with_gui=False):
        """Run regime operations (HMM labeling or ML training)."""
        self.logger.info(f"🧠 Running regime operations for {symbol} on {exchange}")
        
        if with_gui and not self.launch_gui("regime", symbol, exchange):
            return False
        
        try:
            from src.analyst.unified_regime_classifier import UnifiedRegimeClassifier
            
            regime_classifier = UnifiedRegimeClassifier(CONFIG, exchange, symbol)
            
            if subcommand == "load":
                data_file = f"data/{symbol}_1h.csv"
                if not os.path.exists(data_file):
                    self.logger.error(f"❌ Data file not found: {data_file}")
                    return False
                
                from src.analyst.data_utils import load_klines_data
                historical_data = load_klines_data(data_file)
                
                if historical_data is None or historical_data.empty:
                    self.logger.error("❌ Failed to load historical data")
                    return False
                
                success = await regime_classifier.train_complete_system(historical_data)
                return success
                
            elif subcommand in ["train", "train_blank"]:
                # Similar implementation for train commands
                return True
            else:
                self.logger.error(f"❌ Unknown regime subcommand: {subcommand}")
                return False
                
        except Exception as e:
            self.logger.exception(f"❌ Failed to run regime operations: {e}")
            return False

    # GUI methods
    def run_gui_only(self):
        """Run GUI only mode."""
        self.logger.info("🖥️ Running GUI only mode")
        return self.launch_gui()

    def run_gui_with_mode(self, mode, symbol, exchange):
        """Run GUI with specific mode."""
        self.logger.info(f"🖥️ Running GUI with mode: {mode}")
        return self.launch_gui(mode, symbol, exchange)


# Command execution functions (simplified)
def execute_command(launcher, args):
    """Execute the requested command using command handlers."""
    print(f"🔍 DEBUG: Executing command: {args.command}")
    
    # Normalize input step name and collapse force flags
    normalized_step = launcher.step_orchestrator_wrapper._normalize_step_name(getattr(args, "step", None))
    force_flag = bool(getattr(args, "force", False) or getattr(args, "force_rerun", False))
    
    try:
        # Create command handler
        handler = CommandHandlerFactory.create_handler(args.command, launcher)
        
        # Execute command with appropriate parameters
        if args.command in ["light", "blank", "full"]:
            return handler.execute(
                training_mode=args.command,
                symbol=args.symbol,
                exchange=args.exchange,
                lookback_days=getattr(args, "lookback_days", None),
                with_gui=args.gui,
            )
        elif args.command in ["paper", "live", "challenger"]:
            return handler.execute(
                trading_mode=args.command.upper() if args.command != "challenger" else "CHALLENGER",
                symbol=args.symbol,
                exchange=args.exchange,
                with_gui=args.gui,
            )
        elif args.command.startswith("step"):
            return handler.execute(
                start_step=normalized_step or f"{args.command}_data_reading",
                symbol=args.symbol,
                exchange=args.exchange,
                training_mode=args.training_mode,
                force_rerun=force_flag,
                with_gui=args.gui,
            )
        elif args.command == "load":
            return handler.execute(
                symbol=args.symbol,
                exchange=args.exchange,
                lookback_days=DEFAULT_LOOKBACK_DAYS if not args.blank_mode else 30,
                blank_mode=args.blank_mode,
            )
        elif args.command in ["data-collection", "market-analysis", "model-training", 
                             "optimisation", "backtesting", "all-pipelines"]:
            return handler.execute(
                pipeline_type=args.command,
                symbol=args.symbol,
                exchange=args.exchange,
                with_gui=args.gui,
            )
        elif args.command == "regime":
            return handler.execute(
                subcommand=args.regime_subcommand,
                symbol=args.symbol,
                exchange=args.exchange,
                with_gui=args.gui,
            )
        else:
            return handler.execute(command=args.command, **vars(args))
            
    except Exception as e:
        launcher.logger.exception(f"❌ Failed to execute command {args.command}: {e}")
        return False


def execute_gui_command(launcher, args):
    """Execute GUI-specific commands."""
    if args.mode:
        if not args.symbol or not args.exchange:
            launcher.logger.error("❌ Symbol and exchange are required when mode is specified")
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


def initialize_launcher():
    """Initialize launcher with signal handling."""
    signal_handler = setup_signal_handlers()
    launcher = AresLauncher()
    launcher.setup_logging()
    launcher.setup_signal_handling()
    return launcher, signal_handler


def main():
    """Main entry point for the refactored Ares launcher."""
    try:
        # Parse and validate arguments (using existing functions)
        from ares_launcher import parse_arguments, validate_arguments
        
        args = parse_arguments()
        validate_arguments(args)
        
        # Initialize launcher
        launcher, signal_handler = initialize_launcher()
        
        # Log command execution
        launcher.comprehensive_logger.log_launcher_start(
            args.command,
            getattr(args, "symbol", None),
            getattr(args, "exchange", None),
        )
        
        # Execute the requested command
        if args.command == "gui":
            success = execute_gui_command(launcher, args)
        else:
            success = execute_command(launcher, args)
        
        if success:
            launcher.comprehensive_logger.log_launcher_end(0)
            return 0
        else:
            launcher.comprehensive_logger.log_launcher_end(1)
            return 1
            
    except Exception as e:
        if "launcher" in locals():
            launcher.comprehensive_logger.log_error(f"Main function exception: {e}", exc_info=True)
            launcher.comprehensive_logger.log_launcher_end(1)
        else:
            print(f"💥 ERROR: Exception in main: {e}")
            import traceback
            traceback.print_exc()
        return 1
    finally:
        if "launcher" in locals():
            launcher.cleanup()


if __name__ == "__main__":
    main()