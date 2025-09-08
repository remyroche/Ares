#!/usr/bin/env python3
from src.utils.logger import system_logger
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
from src.config.environment import get_environment_settings
from ..standardized_parquet_handler import standardized_parquet_handler

"""Step Validators for Optimisation Pipeline.

This module provides comprehensive step-by-step validators for each optimisation component
with enhanced data protection, validation, and error handling.
"""

from pathlib import Path
# Fallback imports for missing utils modules
try:
    from .utils.common_operations import (
        format_datetime, get_current_datetime, safe_file_exists,
    )
except ImportError:
    # Fallback implementations
    def format_datetime(dt):
        return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else None

    def get_current_datetime():
        from datetime import datetime
        return datetime.now()

    def safe_file_exists(path):
        return os.path.exists(path) if path else False

try:
    from .utils.data_quality_framework import DataQualityFramework
except ImportError:
    class DataQualityFramework:
        def __init__(self):
            pass

try:
    from .utils.base_validator import BaseValidator
except ImportError:
    class BaseValidator:
        def __init__(self):
            pass

from src.utils.logger import system_logger

# Import decorators
try:
    from src.core.decorators.validate import validates
except ImportError:
    # Fallback decorator
    def validates(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

import psutil
import pandas as pd
import logging
from typing import Dict, Any, List, Optional


logger = system_logger.getChild('OptimisationStepValidators')

# Get dynamic symbol configuration
_settings = get_environment_settings()

def get_default_symbol() -> str:
    """Get the default trading symbol from configuration."""
    return _settings.get_default_symbol('ETHUSDT')

class ConfidenceCalibrationStepValidator(BaseValidator):
    """Validator for confidence calibration step with comprehensive data protection."""
    @log_important_calls
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("confidence_calibration_step", config)
        self.dq_framework = DataQualityFramework()
    
    @validates()
    async def validate(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any],
    ) -> bool:
        """Validate confidence calibration step prerequisites and data."""
        self.logger.info("🔍 Validating confidence calibration step...")

        try:
            symbol = training_input.get('symbol', get_default_symbol())
            exchange = training_input.get('exchange', 'BINANCE')
            timeframe = training_input.get('timeframe', '1m')
            data_dir = training_input.get('data_dir', 'data_cache')
            
            # Validate input parameters
            if not self._validate_input_parameters(symbol, exchange, timeframe, data_dir):
                return False
            
            # Validate tactician specialist data availability
            if not await self._validate_tactician_data_availability(symbol, exchange, timeframe, data_dir):
                return False
            
            # Validate regime classification data
            if not await self._validate_regime_data_availability(symbol, exchange, data_dir):
                return False
            
            # Validate output directory permissions
            if not self._validate_output_permissions():
                return False
            
            self.logger.info("✅ Confidence calibration step validation passed")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Confidence calibration step validation failed: {e}")
            return False
    @log_all_calls
    
    def _validate_input_parameters(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> bool:
        """Validate input parameters for confidence calibration."""
        self.logger.info("🔍 Validating input parameters...")
        
        # Validate symbol
        if not symbol or not isinstance(symbol, str) or len(symbol) < 3:
            self.logger.error(f"❌ Invalid symbol: {symbol}")
            return False
        
        # Validate exchange
        valid_exchanges = ['BINANCE', 'MEXC', 'GATEIO']
        if exchange not in valid_exchanges:
            self.logger.error(f"❌ Invalid exchange: {exchange}. Valid: {valid_exchanges}")
            return False
        
        # Validate timeframe
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h']
        if timeframe not in valid_timeframes:
            self.logger.error(f"❌ Invalid timeframe: {timeframe}. Valid: {valid_timeframes}")
            return False
        
        # Validate data directory
        if not safe_file_exists(data_dir):
            self.logger.warning(f"⚠️ Data directory does not exist: {data_dir}")
        
        self.logger.info("✅ Input parameters validation passed")
        return True
    
    async def _validate_tactician_data_availability(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> bool:
        """Validate tactician specialist data availability."""
        self.logger.info("🔍 Validating tactician data availability...")
        
        # Check for tactician specialist files
        tactician_files = [
            f"{data_dir}/tactician_specialist_{symbol}_{exchange}_{timeframe}.parquet",
            f"models/{symbol}_{exchange}_tactician_specialist.pkl"
        ]
        
        missing_files = []
        for file_path in tactician_files:
            if not safe_file_exists(file_path):
                missing_files.append(file_path)
            else:
                self.logger.info(f"✅ Tactician file available: {file_path}")
        
        if missing_files:
            self.logger.error(f"❌ Missing tactician files: {missing_files}")
            return False
        
        # Validate data quality
        try:
            tactician_data = standardized_parquet_handler.read_parquet_standardized(tactician_files[0])
            if tactician_data.empty:
                self.logger.error("❌ Tactician data is empty")
                return False
            
            # Check for required columns
            required_columns = ['timestamp', 'confidence', 'prediction']
            missing_columns = [col for col in required_columns if col not in tactician_data.columns]
            
            if missing_columns:
                self.logger.error(f"❌ Missing required columns in tactician data: {missing_columns}")
                return False
            
            self.logger.info("✅ Tactician data validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to validate tactician data: {e}")
            return False
    
    async def _validate_regime_data_availability(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """Validate regime classification data availability."""
        self.logger.info("🔍 Validating regime data availability...")
        
        # Check for regime classification files
        regime_files = [
            f"models/{symbol}_{exchange}_regime_classifier.pkl",
            f"{data_dir}/regime_labels_{symbol}_{exchange}.parquet"
        ]
        
        missing_files = []
        for file_path in regime_files:
            if not safe_file_exists(file_path):
                missing_files.append(file_path)
            else:
                self.logger.info(f"✅ Regime file available: {file_path}")
        
        if missing_files:
            self.logger.warning(f"⚠️ Some regime files missing: {missing_files}")
            self.logger.warning("⚠️ Confidence calibration will use default regime handling")
        
        return True
    @log_all_calls
    
    def _validate_output_permissions(self) -> bool:
        """Validate output directory permissions."""
        self.logger.info("🔍 Validating output permissions...")
        
        try:
            # Test write permissions to models directory
            models_dir = Path("models")
            
            # Test file creation
            test_file = models_dir / "test_write_permission.tmp"
            test_file.write_text("test")
            test_file.unlink()
            
            self.logger.info("✅ Output permissions validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Output permissions validation failed: {e}")
            return False

class FinalParametersOptimizationStepValidator(BaseValidator):
    """Validator for final parameters optimization step with comprehensive data protection."""
    @log_important_calls
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("final_parameters_optimization_step", config)
        self.dq_framework = DataQualityFramework()
    
    @validates()
    async def validate(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any],
    ) -> bool:
        """Validate final parameters optimization step prerequisites and data."""
        self.logger.info("🔍 Validating final parameters optimization step...")

        try:
            symbol = training_input.get('symbol', get_default_symbol())
            exchange = training_input.get('exchange', 'BINANCE')
            timeframe = training_input.get('timeframe', '1m')
            data_dir = training_input.get('data_dir', 'data_cache')
            
            # Validate input parameters
            if not self._validate_input_parameters(symbol, exchange, timeframe, data_dir):
                return False
            
            # Validate confidence calibration results
            if not await self._validate_confidence_calibration_results(symbol, exchange):
                return False
            
            # Validate configuration parameters
            if not self._validate_optimization_configuration():
                return False
            
            # Validate computational resources
            if not self._validate_computational_resources():
                return False
            
            self.logger.info("✅ Final parameters optimization step validation passed")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Final parameters optimization step validation failed: {e}")
            return False
    @log_all_calls
    
    def _validate_input_parameters(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> bool:
        """Validate input parameters for final parameters optimization."""
        self.logger.info("🔍 Validating input parameters...")
        
        # Reuse validation logic from confidence calibration
        validator = ConfidenceCalibrationStepValidator(self.config)
        return validator._validate_input_parameters(symbol, exchange, timeframe, data_dir)
    
    async def _validate_confidence_calibration_results(self, symbol: str, exchange: str) -> bool:
        """Validate confidence calibration results are available."""
        self.logger.info("🔍 Validating confidence calibration results...")
        
        # Check for confidence calibration results
        calibration_files = [
            f"models/{symbol}_{exchange}_confidence_calibration.json",
            f"models/{symbol}_{exchange}_confidence_calibration_regime_0.json",
            f"models/{symbol}_{exchange}_confidence_calibration_regime_1.json"
        ]
        
        found_files = []
        for file_path in calibration_files:
            if safe_file_exists(file_path):
                found_files.append(file_path)
                self.logger.info(f"✅ Confidence calibration file found: {file_path}")
        
        if not found_files:
            self.logger.warning("⚠️ No confidence calibration results found")
            self.logger.warning("⚠️ Final parameters optimization will use default confidence parameters")
        
        return True
    @log_all_calls
    
    def _validate_optimization_configuration(self) -> bool:
        """Validate optimization configuration parameters."""
        self.logger.info("🔍 Validating optimization configuration...")
        
        # Check for required configuration parameters
        required_params = [
            'random_state',
            'n_trials',
            'max_trials',
            'timeout'
        ]
        
        missing_params = []
        for param in required_params:
            if param not in self.config:
                missing_params.append(param)
        
        if missing_params:
            self.logger.warning(f"⚠️ Missing optimization parameters: {missing_params}")
            self.logger.warning("⚠️ Using default optimization parameters")
        
        # Validate parameter ranges
        if 'n_trials' in self.config:
            n_trials = self.config['n_trials']
            if not isinstance(n_trials, int) or n_trials < 1 or n_trials > 1000:
                self.logger.error(f"❌ Invalid n_trials: {n_trials}. Must be between 1 and 1000")
                return False
        
        if 'max_trials' in self.config:
            max_trials = self.config['max_trials']
            if not isinstance(max_trials, int) or max_trials < 1 or max_trials > 10000:
                self.logger.error(f"❌ Invalid max_trials: {max_trials}. Must be between 1 and 10000")
                return False
        
        self.logger.info("✅ Optimization configuration validation passed")
        return True
    @log_all_calls
    
    def _validate_computational_resources(self) -> bool:
        """Validate computational resources are sufficient."""
        self.logger.info("🔍 Validating computational resources...")
        
        try:
            
            # Check available memory
            available_memory = psutil.virtual_memory().available
            required_memory = 2 * 1024 * 1024 * 1024  # 2GB minimum
            
            if available_memory < required_memory:
                self.logger.warning(f"⚠️ Low available memory: {available_memory / (1024**3):.1f}GB")
                self.logger.warning("⚠️ Optimization may be slower or fail with large datasets")
            
            # Check CPU cores
            cpu_cores = psutil.cpu_count()
            if cpu_cores < 2:
                self.logger.warning(f"⚠️ Limited CPU cores: {cpu_cores}")
                self.logger.warning("⚠️ Optimization will use single-threaded mode")
            
            self.logger.info("✅ Computational resources validation passed")
            return True
            
        except ImportError:
            self.logger.warning("⚠️ psutil not available, skipping resource validation")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ Resource validation failed: {e}")
            return True

class OptimisationPipelineStepValidator(BaseValidator):
    """Comprehensive validator for the entire optimisation pipeline."""
    @log_important_calls
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("optimisation_pipeline", config)
        self.confidence_validator = ConfidenceCalibrationStepValidator(config)
        self.parameters_validator = FinalParametersOptimizationStepValidator(config)
    
    @validates()
    async def validate(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any],
    ) -> bool:
        """Validate the entire optimisation pipeline."""
        self.logger.info("🔍 Validating optimisation pipeline...")
        
        try:
            # Validate confidence calibration step
            confidence_valid = await self.confidence_validator.validate(training_input, pipeline_state)
            
            # Validate final parameters optimization step
            parameters_valid = await self.parameters_validator.validate(training_input, pipeline_state)
            
            # Overall validation result
            overall_valid = confidence_valid and parameters_valid
            
            if overall_valid:
                self.logger.info("✅ Optimisation pipeline validation passed")
            else:
                self.logger.error("❌ Optimisation pipeline validation failed")
            
            return overall_valid
            
        except Exception as e:
            self.logger.exception(f"❌ Optimisation pipeline validation failed: {e}")
            return False

# Factory function for creating validators
def create_optimisation_validator(step_name: str, config: Dict[str, Any]) -> BaseValidator:
    """Create appropriate validator for optimisation step."""
    validators = {
        'confidence_calibration': ConfidenceCalibrationStepValidator,
        'final_parameters_optimization': FinalParametersOptimizationStepValidator,
        'optimisation_pipeline': OptimisationPipelineStepValidator
    }
    
    if step_name not in validators:
        raise ValueError(f"Unknown optimisation step: {step_name}")
    
    return validators[step_name](config)