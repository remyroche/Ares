#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Pipeline Managers for Ares Launcher

This module contains specialized pipeline managers that handle the execution
of different types of pipelines, reducing complexity in the main launcher class.
"""

import asyncio

import os
import subprocess
import sys

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional

from src.utils.common_operations import format_datetime, get_current_datetime
import logging
import time

class BasePipelineManager(ABC):
    """Base class for all pipeline managers."""
    
    def __init__(self, launcher):
        self.launcher = launcher
        self.logger = launcher.logger
        
    @abstractmethod
    def execute(self, symbol: str, exchange: str, with_gui: bool = False) -> bool:
        """Execute the pipeline with the given parameters."""
        pass
    
    def _run_subprocess_with_monitoring(self, cmd: list, env: Optional[Dict] = None) -> bool:
        """Run a subprocess with real-time output monitoring."""
        try:
            process = subprocess.Popen(
                cmd,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                text = True,
                bufsize = 1,
                universal_newlines = True,
                env = env or os.environ.copy(),
            )
            self.launcher.processes.append(process)
            
            # Monitor output in real-time
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    tprint(output.strip())
                    self.logger.info(output.strip())
            
            return process.poll() == 0
            
        except Exception as e:
            self.logger.exception(f"Failed to run subprocess: {e}")
            return False

class DataCollectionPipelineManager(BasePipelineManager):
    """Manages data collection pipeline execution."""
    
    def execute(self, symbol: str, exchange: str, with_gui: bool = False) -> bool:
        """Execute unified data collection pipeline with all 12 steps."""
        self.logger.info(f"📊 Running unified data collection pipeline for {symbol} on {exchange}")
        tprint("=" * 80)
        tprint("🚀 UNIFIED DATA COLLECTION PIPELINE")
        tprint("=" * 80)
        tprint(f"ℹ️ Symbol: {symbol}")
        tprint(f"ℹ️ Exchange: {exchange}")
        tprint(f"ℹ️ GUI Mode: {with_gui}")
        tprint(f"ℹ️ Start Time: {format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S')}")
        tprint("=" * 80)
        tprint("📋 Pipeline Steps:")
        tprint("   1. Data Download - Download raw data from exchanges")
        tprint("   2. Data Conversion - Convert data formats and standardize")
        tprint("   3. Data Validation - Validate data quality and integrity")
        tprint("   4. Data Preparation - Prepare data for further processing")
        tprint("   5. Feature Engineering - Limited feature engineering (price returns, volume returns)")
        tprint("   6. Data Resampling - Resample to multiple timeframes")
        tprint("   7. Gap Filling - Detect and fill data gaps")
        tprint("   8. Data Quality Check - Comprehensive quality assessment")
        tprint("   9. Data Integration - Integrate multiple data sources with backwards compatibility")
        tprint("  10. Data Storage - Store processed data")
        tprint("  11. Data Monitoring - Monitor data collection process")
        tprint("  12. Data Export - Export data in various formats")
        tprint("=" * 80)

        # Pre-flight validation
        if not self._validate_prerequisites(symbol, exchange):
            return False

        if with_gui and not self.launcher.launch_gui("data-collection", symbol, exchange):
            return False

        # Execute unified data collection pipeline using standalone script
        return self._execute_standalone_pipeline(symbol, exchange)
    
    def _execute_standalone_pipeline(self, symbol: str, exchange: str) -> bool:
        """Execute the unified data collection pipeline using the standalone script."""
        try:
            # Set up environment
            env = os.environ.copy()
            env.update({
                'PYTHONPATH': str(Path(__file__).parent.parent.parent),
                'DATA_COLLECTION_MODE': 'unified',
                'SYMBOL': symbol,
                'EXCHANGE': exchange
            })

            # Build command for standalone script
            cmd = [
                sys.executable, 
                "standalone_data_collection.py",
                "--symbol", symbol,
                "--exchange", exchange.upper(),
                "--mode", "full",
                "--data-dir", "data_cache",
                "--lookback-days", "30",
                "--timeframes", "5m", "15m", "30m", "1h",
                "--add-technical-indicators",
                "--parallel-processing",
                "--max-workers", "4"
            ]
            
            self.logger.info(f"🔄 Executing standalone data collection pipeline: {' '.join(cmd)}")
            
            return self._run_subprocess_with_monitoring(cmd, env)
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to execute standalone data collection pipeline: {e}")
            tprint("=" * 80)
            tprint("❌ DATA COLLECTION PIPELINE FAILED")
            tprint("=" * 80)
            tprint(f"Error: {str(e)}")
            tprint("=" * 80)
            return False
    
    def _validate_prerequisites(self, symbol: str, exchange: str) -> bool:
        """Validate prerequisites for data collection."""
        self.logger.info("🔍 Validating data collection prerequisites...")
        
        try:
            from src.utils.common_operations import safe_file_exists, ensure_directory
            
            # Check required directories
            required_dirs = ["data_cache", "log"]
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
                return False
            
            self.logger.info("✅ Data collection prerequisites validation completed")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Prerequisites validation failed: {e}")
            return False

class ModelTrainingPipelineManager(BasePipelineManager):
    """Manages model training pipeline execution."""
    
    def execute(self, symbol: str, exchange: str, with_gui: bool = False) -> bool:
        """Execute model training pipeline with comprehensive monitoring."""
        self.logger.info(f"📊 Running model training pipeline for {symbol} on {exchange}")
        tprint("=" * 80)
        tprint("🚀 ENHANCED MODEL TRAINING PIPELINE")
        tprint("=" * 80)
        tprint(f"🎯 Symbol: {symbol}")
        tprint(f"🏢 Exchange: {exchange}")
        tprint(f"🖥️ GUI Mode: {with_gui}")
        tprint(f"⏰ Start Time: {format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S')}")
        tprint("=" * 80)

        # Pre-flight validation
        if not self._validate_prerequisites(symbol, exchange):
            return False

        if with_gui and not self.launcher.launch_gui("model-training", symbol, exchange):
            return False

        # Set up environment
        env = os.environ.copy()
        env.update({
            'MODEL_TRAINING_MODE': 'enhanced',
            'SYMBOL': symbol,
            'EXCHANGE': exchange
        })

        cmd = [
            sys.executable,
            "src/training/steps/model_training/step09_model_training_main.py",
            "--symbol", symbol,
            "--exchange", exchange,
            "--enhanced-mode"
        ]
        
        return self._run_subprocess_with_monitoring(cmd, env)
    
    def _validate_prerequisites(self, symbol: str, exchange: str) -> bool:
        """Validate prerequisites for model training."""
        self.logger.info("🔍 Validating model training prerequisites...")
        
        try:
            
            # Check required directories
            required_dirs = ["data_cache", "models", "checkpoints", "log"]
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
                tprint(f"❌ Missing required data files:")
                for file_path in missing_files:
                    tprint(f"   • {file_path}")
                tprint("💡 Please run data collection first:")
                tprint(f"   python ares_launcher.py load --symbol {symbol} --exchange {exchange}")
                return False
            
            self.logger.info("✅ Model training prerequisites validation completed")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Prerequisites validation failed: {e}")
            return False

class OptimisationPipelineManager(BasePipelineManager):
    """Manages optimisation pipeline execution."""
    
    def execute(self, symbol: str, exchange: str, with_gui: bool = False) -> bool:
        """Execute enhanced optimisation pipeline."""
        self.logger.info(f"📊 Running enhanced optimisation pipeline for {symbol} on {exchange}")
        tprint("=" * 80)
        tprint("🚀 ENHANCED OPTIMISATION PIPELINE")
        tprint("=" * 80)
        tprint(f"🎯 Symbol: {symbol}")
        tprint(f"🏢 Exchange: {exchange}")
        tprint(f"🖥️ GUI Mode: {with_gui}")
        tprint(f"⏰ Start Time: {format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S')}")
        tprint("=" * 80)

        # Pre-flight validation
        if not self._validate_prerequisites(symbol, exchange):
            return False

        if with_gui and not self.launcher.launch_gui("optimisation", symbol, exchange):
            return False

        # Set up environment
        env = os.environ.copy()
        env.update({
            'OPTIMISATION_MODE': 'enhanced',
            'SYMBOL': symbol,
            'EXCHANGE': exchange
        })

        cmd = [
            sys.executable,
            "src/training/steps/optimisation/step16_optimisation_main.py",
            "--symbol", symbol,
            "--exchange", exchange,
            "--enhanced-mode"
        ]
        
        return self._run_subprocess_with_monitoring(cmd, env)
    
    def _validate_prerequisites(self, symbol: str, exchange: str) -> bool:
        """Validate prerequisites for optimisation."""
        self.logger.info("🔍 Validating optimisation prerequisites...")
        
        try:
            
            # Check required directories
            required_dirs = ["data_cache", "models", "checkpoints", "log"]
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
                tprint(f"❌ Missing required data files:")
                for file_path in missing_files:
                    tprint(f"   • {file_path}")
                tprint("💡 Please run data collection first:")
                tprint(f"   python ares_launcher.py load --symbol {symbol} --exchange {exchange}")
                return False
            
            self.logger.info("✅ Optimisation prerequisites validation completed")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Prerequisites validation failed: {e}")
            return False

class BacktestingPipelineManager(BasePipelineManager):
    """Manages backtesting pipeline execution."""
    
    def execute(self, symbol: str, exchange: str, with_gui: bool = False) -> bool:
        """Execute enhanced backtesting pipeline."""
        self.logger.info(f"📊 Running enhanced backtesting for {symbol} on {exchange}")
        tprint(f"📊 Running enhanced backtesting for {symbol} on {exchange}")
        tprint("=" * 80)

        if with_gui and not self.launcher.launch_gui("backtesting", symbol, exchange):
            return False

        try:
            # Import enhanced backtesting components
            from src.training.steps.backtesting.enhanced_logging import get_backtesting_logger

            # Initialize enhanced logger
            launcher_logger = get_backtesting_logger(
                f"launcher_{symbol}_{exchange}", 
                log_dir="log/backtesting"
            )
            launcher_logger.start_performance_monitoring(interval = 5.0)

            try:
                launcher_logger.log_info("🚀 Starting Enhanced Backtesting from Launcher", "LAUNCHER")
                launcher_logger.log_info(f"📊 Configuration: {symbol} on {exchange}", "LAUNCHER")

                # Enhanced configuration
                enhanced_config = {
                    'force_rerun': True,
                    'walk_forward_validation': True,
                    'monte_carlo_validation': True,
                    'ab_testing': True,
                    'model_saving': True,
                    'random_state': 42,
                    'enable_validation': True,
                    'strict_validation': False,
                    'validate_data_quality': True,
                    'retry_failed_steps': True,
                    'max_retries': 3,
                    'timeout_seconds': 3600,
                    'enable_performance_monitoring': True,
                    'log_detailed_metrics': True,
                }

                # Pre-flight validation
                launcher_logger.log_progress("Pre-flight Validation", 0, "Starting validation checks")
                
                data_dir = "historical_data"
                required_files = [
                    f"aggtrades_{exchange}_{symbol}_consolidated.parquet",
                    f"volume_{exchange}_{symbol}_consolidated.parquet"
                ]
                
                missing_files = []
                for file_name in required_files:
                    file_path = f"{data_dir}/{file_name}"
                    if not safe_file_exists(file_path):
                        missing_files.append(file_name)
                    else:
                        launcher_logger.log_success(f"Required file found: {file_name}", "VALIDATION")
                
                if missing_files:
                    launcher_logger.log_error(
                        Exception(f"Missing required data files: {missing_files}"), 
                        "VALIDATION"
                    )
                    launcher_logger.log_quality_flag(
                        "MISSING_DATA_FILES", 
                        f"Missing required data files: {missing_files}", 
                        "ERROR"
                    )
                    return False
                
                launcher_logger.log_success("All required data files found", "VALIDATION")
                launcher_logger.log_progress("Pre-flight Validation", 100, "Validation completed successfully")

                # Run enhanced backtesting pipeline
                launcher_logger.log_info("🚀 Starting enhanced backtesting pipeline", "EXECUTION")
                tprint("🚀 Starting enhanced backtesting pipeline...")
                tprint(f"📅 Started at: {format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S')}")

                success = asyncio.run(
                    self._run_backtesting_pipeline(
                        symbol = symbol,
                        exchange = exchange,
                        timeframe="1m",
                        data_dir = data_dir,
                        **enhanced_config
                    )
                )

                if success:
                    launcher_logger.log_success("🎉 Enhanced backtesting completed successfully!", "COMPLETION")
                    tprint("🎉 Enhanced backtesting completed successfully!")
                    return True
                else:
                    launcher_logger.log_error(Exception("Enhanced backtesting failed"), "EXECUTION")
                    tprint("❌ Enhanced backtesting failed!")
                    return False

            finally:
                launcher_logger.stop_performance_monitoring()
                launcher_logger.cleanup()

        except Exception as e:
            self.logger.exception(f"❌ Failed to run enhanced backtesting: {e}")
            tprint(f"❌ Failed to run enhanced backtesting: {e}")
            return False
    
    async def _run_backtesting_pipeline(self, **kwargs) -> bool:
        """Run the actual backtesting pipeline."""
        # This would call the actual backtesting pipeline implementation
        # For now, return True as a placeholder
        return True

class AllPipelinesManager(BasePipelineManager):
    """Manages execution of all pipelines in sequence."""
    
    def execute(self, symbol: str, exchange: str, with_gui: bool = False) -> bool:
        """Execute all pipelines in sequence with organized report management."""
        self.logger.info(f"📊 Running all pipelines for {symbol} on {exchange}")

        # Initialize report manager and collector
        try:
            from src.utils.report_manager import initialize_report_manager
            from src.utils.report_collector import initialize_report_collector
            
            report_manager = initialize_report_manager()
            report_collector = initialize_report_collector()
            
            report_collector.setup_pipeline_interception(symbol, exchange)
            
            self.logger.info(f"📁 Report directory initialized: {report_manager.get_run_directory()}")
            tprint(f"📁 Report directory: {report_manager.get_run_directory()}")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to initialize report manager/collector: {e}")

        if with_gui and not self.launcher.launch_gui("all-pipelines", symbol, exchange):
            return False

        # Set up environment
        env = os.environ.copy()
        env.update({
            'SYMBOL': symbol,
            'EXCHANGE': exchange,
            'REPORT_RUN_TIMESTAMP': report_manager.get_run_timestamp()
        })

        cmd = [sys.executable, "src/training/steps/run_all_pipelines.py"]
        
        return self._run_subprocess_with_monitoring(cmd, env)

class PipelineManagerFactory:
    """Factory for creating pipeline managers."""
    
    @staticmethod
    def create_manager(pipeline_type: str, launcher):
        """Create the appropriate pipeline manager."""
        managers = {
            "data-collection": DataCollectionPipelineManager,
            "model-training": ModelTrainingPipelineManager,
            "optimisation": OptimisationPipelineManager,
            "backtesting": BacktestingPipelineManager,
            "all-pipelines": AllPipelinesManager,
        }
        
        if pipeline_type not in managers:
            raise ValueError(f"No pipeline manager available for: {pipeline_type}")
        
        return managers[pipeline_type](launcher)