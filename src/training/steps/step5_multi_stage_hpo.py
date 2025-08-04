# src/training/steps/step5_multi_stage_hpo.py
"""
Multi-stage hyperparameter optimization step for the Ares trading system.

This module implements a 4-stage hyperparameter optimization pipeline:
1. Stage 1: Ultra-Coarse (5 trials) - Very wide ranges
2. Stage 2: Coarse (20 trials) - Narrowed ranges from Stage 1
3. Stage 3: Medium (30 trials) - Further narrowed ranges from Stage 2
4. Stage 4: Fine (50 trials) - Final optimization with precise ranges

The optimization uses adaptive range narrowing based on best results from each stage.
"""

import asyncio
import gc
import json
import logging
import os
import pickle
import psutil
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np
import pandas as pd
from functools import partial
from collections import deque
import hashlib
import copy
from enum import Enum

import aiofiles
from pydantic import BaseModel, validator
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import joblib

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import CONFIG
from src.database.sqlite_manager import SQLiteManager
from src.supervisor.optimizer import Optimizer
from src.utils.error_handler import (
    handle_errors, 
    handle_specific_errors,
    handle_data_processing_errors,
    handle_file_operations
)
from src.utils.logger import setup_logging, system_logger


# ==============================================================================
# Type Definitions
# ==============================================================================

class TrainingData(TypedDict):
    """Type definition for training data structure."""
    klines: pd.DataFrame
    agg_trades: pd.DataFrame
    futures: pd.DataFrame


class StageResult(TypedDict):
    """Type definition for stage optimization results."""
    stage: int
    trials: int
    duration: float
    optimization_time: float
    result: Dict[str, Any]


class OptimizationMetrics(TypedDict):
    """Type definition for optimization metrics."""
    total_duration: float
    stage_breakdown: Dict[int, float]
    optimization_vs_overhead: float
    memory_usage_mb: Optional[float]


# ==============================================================================
# Main MultiStageHPO Class
# ==============================================================================

class MultiStageHPO:
    """
    Multi-stage hyperparameter optimization class.
    
    This class orchestrates the 4-stage hyperparameter optimization process:
    1. Stage 1: Ultra-Coarse (5 trials) - Very wide ranges
    2. Stage 2: Coarse (20 trials) - Narrowed ranges from Stage 1
    3. Stage 3: Medium (30 trials) - Further narrowed ranges from Stage 2
    4. Stage 4: Fine (50 trials) - Final optimization with precise ranges
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MultiStageHPO.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("MultiStageHPO")
        
        # Initialize configuration
        self.hpo_config = MultiStageHPOConfig.from_environment()
        
        # Initialize components
        self.data_loader = DataLoader()
        self.hpo_ranges_manager = HPORangesManager(self.hpo_config)
        self.optimization_executor = OptimizationExecutor(self.hpo_config)
        self.results_manager = ResultsManager()
        
        # State tracking
        self.is_initialized = False
        self.optimization_history: List[Dict[str, Any]] = []
        
        self.logger.info("MultiStageHPO initialized")

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError),
        default_return=False,
        context="multi-stage HPO initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize MultiStageHPO components.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing MultiStageHPO...")
            
            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid MultiStageHPO configuration")
                return False
            
            self.is_initialized = True
            self.logger.info("✅ MultiStageHPO initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MultiStageHPO: {e}")
            return False

    def _validate_configuration(self) -> bool:
        """
        Validate the MultiStageHPO configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Check stage trials
            if not self.hpo_config.stage_trials or len(self.hpo_config.stage_trials) == 0:
                self.logger.error("No stage trials configured")
                return False
            
            # Check narrowing factors
            if not self.hpo_config.narrowing_factors:
                self.logger.error("No narrowing factors configured")
                return False
            
            # Check default HPO ranges
            if not self.hpo_config.default_hpo_ranges:
                self.logger.error("No default HPO ranges configured")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError),
        default_return=None,
        context="multi-stage HPO execution",
    )
    async def run_optimization(
        self,
        symbol: str,
        data_dir: str,
        data_file_path: str,
        timeframe: str = "1m"
    ) -> Optional[Dict[str, Any]]:
        """
        Run multi-stage hyperparameter optimization.
        
        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            data_dir: Directory containing training data
            data_file_path: Path to pickle file with collected data
            timeframe: Timeframe for analysis (default: '1m')
            
        Returns:
            Optional[Dict[str, Any]]: Optimization results or None if failed
        """
        try:
            if not self.is_initialized:
                self.logger.error("MultiStageHPO not initialized")
                return None
            
            self.logger.info(f"Starting multi-stage HPO for {symbol}")
            
            # Load training data
            training_data = await self.data_loader.load_training_data(data_file_path)
            if not training_data:
                self.logger.error("Failed to load training data")
                return None
            
            # Load HPO ranges
            initial_hpo_ranges = await self.hpo_ranges_manager.load_hpo_ranges(data_dir, symbol)
            if not initial_hpo_ranges:
                self.logger.error("Failed to load HPO ranges")
                return None
            
            # Initialize optimizer
            optimizer = Optimizer(config=self.config)
            
            # Execute optimization stages
            stage_results = await self.optimization_executor.execute_stages(
                optimizer, training_data, initial_hpo_ranges, symbol
            )
            
            if not stage_results:
                self.logger.error("Optimization stages failed")
                return None
            
            # Save results
            total_duration = sum(stage['duration'] for stage in stage_results)
            success = await self.results_manager.save_results(
                stage_results, initial_hpo_ranges, data_dir, symbol, total_duration
            )
            
            if not success:
                self.logger.warning("Failed to save results")
            
            # Create result summary
            result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'stage_results': stage_results,
                'total_duration': total_duration,
                'status': 'completed'
            }
            
            self.optimization_history.append(result)
            self.logger.info(f"✅ Multi-stage HPO completed for {symbol}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to run optimization: {e}")
            return None

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """
        Get optimization history.
        
        Returns:
            List[Dict[str, Any]]: List of optimization results
        """
        return self.optimization_history.copy()

    def cleanup(self) -> None:
        """
        Clean up resources.
        """
        try:
            self.logger.info("Cleaning up MultiStageHPO...")
            # Add any cleanup logic here if needed
            self.logger.info("✅ MultiStageHPO cleanup completed")
        except Exception as e:
            self.logger.error(f"Failed to cleanup MultiStageHPO: {e}")


# ==============================================================================
# Configuration Models
# ==============================================================================

class HPOParameterConfig(BaseModel):
    """Configuration for individual HPO parameter."""
    min: float
    max: float
    type: str
    
    @validator('max')
    def max_must_be_greater_than_min(cls, v: float, values: Dict[str, Any]) -> float:
        if 'min' in values and v <= values['min']:
            raise ValueError('max must be greater than min')
        return v
    
    @validator('type')
    def type_must_be_valid(cls, v: str) -> str:
        if v not in ['float', 'int']:
            raise ValueError('type must be float or int')
        return v


class HPORangesConfig(BaseModel):
    """Configuration for HPO parameter ranges."""
    parameters: Dict[str, HPOParameterConfig]
    
    def validate_ranges(self) -> bool:
        """Validate all parameter ranges."""
        for name, config in self.parameters.items():
            if config.max <= config.min:
                raise ValueError(f"Invalid range for {name}")
        return True


@dataclass
class MultiStageHPOConfig:
    """Configuration for multi-stage hyperparameter optimization."""
    
    # Stage configuration
    stage_trials: List[int]
    narrowing_factors: Dict[int, float]
    min_range_factor: float = 0.1
    
    # Default HPO ranges
    default_hpo_ranges: Optional[Dict[str, Dict[str, Any]]] = None
    
    def __post_init__(self) -> None:
        if self.default_hpo_ranges is None:
            self.default_hpo_ranges = {
                "learning_rate": {"min": 0.001, "max": 0.1, "type": "float"},
                "max_depth": {"min": 3, "max": 12, "type": "int"},
                "n_estimators": {"min": 50, "max": 500, "type": "int"},
                "subsample": {"min": 0.6, "max": 1.0, "type": "float"},
                "colsample_bytree": {"min": 0.6, "max": 1.0, "type": "float"},
                "reg_alpha": {"min": 1e-8, "max": 10.0, "type": "float", "log": True},  # L1 regularization
                "reg_lambda": {"min": 1e-8, "max": 10.0, "type": "float", "log": True},  # L2 regularization
                "min_child_weight": {"min": 1, "max": 10, "type": "int"},
                "gamma": {"min": 0.0, "max": 1.0, "type": "float"},
                "scale_pos_weight": {"min": 0.5, "max": 2.0, "type": "float"},
                "model_type": {"min": 0, "max": 6, "type": "int"},  # 0-6 for different model types
                "confidence_threshold": {"min": 0.6, "max": 0.95, "type": "float"},
                "tp_multiplier": {"min": 1.2, "max": 5.0, "type": "float"},
                "sl_multiplier": {"min": 0.8, "max": 2.5, "type": "float"},
                "position_size": {"min": 0.02, "max": 0.3, "type": "float"},
                # Position division optimization parameters
                "entry_confidence_threshold": {"min": 0.6, "max": 0.85, "type": "float"},
                "additional_position_threshold": {"min": 0.7, "max": 0.9, "type": "float"},
                "division_confidence_threshold": {"min": 0.75, "max": 0.95, "type": "float"},
                "max_division_ratio": {"min": 0.8, "max": 1.2, "type": "float"},
                "max_positions": {"min": 2, "max": 5, "type": "int"},
            }
    
    @classmethod
    def from_environment(cls) -> 'MultiStageHPOConfig':
        """Create configuration based on environment variables."""
        if os.environ.get("BLANK_TRAINING_MODE") == "1":
            print("🔧 BLANK TRAINING MODE DETECTED - Using minimal configuration")
            return cls(
                stage_trials=[1, 1, 1, 1],  # Total: 4 trials for ultra-fast testing
                narrowing_factors={1: 0.5, 2: 0.3, 3: 0.2}
            )
        else:
            return cls(
                stage_trials=[5, 20, 30, 50],  # Total: 105 trials
                narrowing_factors={1: 0.5, 2: 0.3, 3: 0.2}
            )


@dataclass
class OptimizationProgress:
    """Track optimization progress and timing."""
    current_stage: int
    total_stages: int
    current_trial: int
    total_trials: int
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    
    @property
    def progress_percentage(self) -> float:
        """Calculate overall progress percentage."""
        return (self.current_stage - 1) / self.total_stages * 100
    
    def update_progress(self, stage: int, trial: int) -> None:
        """Update progress tracking."""
        self.current_stage = stage
        self.current_trial = trial
        # Update estimated completion time based on current progress
        if self.progress_percentage > 0:
            elapsed = datetime.now() - self.start_time
            estimated_total = elapsed / (self.progress_percentage / 100)
            self.estimated_completion = self.start_time + estimated_total


# ==============================================================================
# Utility Classes
# ==============================================================================

class MemoryManager:
    """Manage memory usage and cleanup."""
    
    @staticmethod
    def log_memory_usage(logger: logging.Logger, context: str) -> Optional[float]:
        """Log current memory usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            logger.info(f"Memory usage ({context}): {memory_mb:.2f} MB")
            return memory_mb
        except Exception as e:
            logger.warning(f"Could not get memory usage: {e}")
            return None
    
    @staticmethod
    def cleanup_dataframes(*dataframes: Optional[pd.DataFrame]) -> None:
        """Explicitly cleanup DataFrames to free memory."""
        for df in dataframes:
            if df is not None:
                del df
        gc.collect()


class StructuredLogger:
    """Enhanced logger with structured logging and correlation tracking."""
    
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.correlation_id = str(uuid.uuid4())
    
    @asynccontextmanager
    async def stage_context(self, stage_num: int, trials: int):
        """Context manager for stage logging."""
        start_time = time.time()
        self.logger.info(f"STAGE_{stage_num}_START", extra={
            "stage": stage_num,
            "trials": trials,
            "correlation_id": self.correlation_id,
            "timestamp": datetime.now().isoformat()
        })
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.logger.info(f"STAGE_{stage_num}_COMPLETE", extra={
                "stage": stage_num,
                "duration": duration,
                "correlation_id": self.correlation_id,
                "timestamp": datetime.now().isoformat()
            })


# ==============================================================================
# Core Processing Classes
# ==============================================================================

class MultiStageHPOProcessor:
    """Handles multi-stage hyperparameter optimization processing."""
    
    def __init__(self, config: MultiStageHPOConfig) -> None:
        self.config = config
        self.logger = system_logger.getChild("MultiStageHPOProcessor")
    
    def _narrow_parameter_ranges(
        self, 
        stage_result: Dict[str, Any], 
        current_ranges: Dict[str, Any], 
        stage_num: int
    ) -> Dict[str, Any]:
        """
        Narrow parameter ranges based on stage results.
        
        Args:
            stage_result: Results from the current stage
            current_ranges: Current parameter ranges
            stage_num: Current stage number (1-3)
            
        Returns:
            dict: Narrowed parameter ranges for the next stage
        """
        narrowed_ranges = current_ranges.copy()
        
        if not stage_result or 'best_params' not in stage_result:
            return narrowed_ranges
        
        best_params = stage_result['best_params']
        narrowing_factor = self.config.narrowing_factors.get(stage_num, 0.3)
        
        for param_name, param_config in current_ranges.items():
            if param_name in best_params:
                best_value = best_params[param_name]
                
                # Handle both 'min'/'max' and 'low'/'high' key formats
                has_min_max = all(key in param_config for key in ['min', 'max'])
                has_low_high = all(key in param_config for key in ['low', 'high'])
                
                if has_min_max:
                    min_key, max_key = 'min', 'max'
                elif has_low_high:
                    min_key, max_key = 'low', 'high'
                else:
                    # Skip parameters without valid range keys
                    continue
                
                current_min = param_config.get(min_key, 0)
                current_max = param_config.get(max_key, 1)
                param_type = param_config.get('type', 'float')
                
                # Calculate new range around the best value
                range_size = current_max - current_min
                new_range_size = range_size * narrowing_factor
                
                new_min = max(current_min, best_value - new_range_size / 2)
                new_max = min(current_max, best_value + new_range_size / 2)
                
                # Ensure minimum range size
                min_range = range_size * self.config.min_range_factor
                if new_max - new_min < min_range:
                    center = (new_min + new_max) / 2
                    new_min = center - min_range / 2
                    new_max = center + min_range / 2
                
                # Update the range - always use 'min'/'max' format for consistency
                narrowed_ranges[param_name] = {
                    'min': new_min,
                    'max': new_max,
                    'type': param_type
                }
        
        return narrowed_ranges


class DataLoader:
    """Handles data loading operations with validation."""
    
    def __init__(self) -> None:
        self.logger = system_logger.getChild("DataLoader")
        self.memory_manager = MemoryManager()
    
    @handle_specific_errors(
        error_handlers={
            FileNotFoundError: (None, "Data file not found"),
            pickle.PickleError: (None, "Invalid pickle data"),
            KeyError: (None, "Missing required data keys"),
        },
        default_return=None,
        context="data loading"
    )
    async def load_training_data(self, data_file_path: str) -> Optional[TrainingData]:
        """
        Load training data from pickle file with validation.
        
        Args:
            data_file_path: Path to the pickle file containing training data
            
        Returns:
            TrainingData: Validated training data or None if loading fails
        """
        self.logger.info("📊 Loading data from pickle file...")
        data_load_start = time.time()
        
        # Log initial memory usage
        initial_memory = self.memory_manager.log_memory_usage(self.logger, "data_loading_start")
        
        if not os.path.exists(data_file_path):
            self.logger.error(f"❌ Data file not found: {data_file_path}")
            return None
        
        try:
            async with aiofiles.open(data_file_path, 'rb') as f:
                content = await f.read()
                collected_data = pickle.loads(content)
        except Exception as e:
            self.logger.error(f"❌ Failed to load pickle data: {e}")
            return None
        
        # Extract and validate data
        klines_df = collected_data.get("klines")
        agg_trades_df = collected_data.get("agg_trades")
        futures_df = collected_data.get("futures")
        
        # Validate data integrity
        if not self._validate_training_data(klines_df, agg_trades_df, futures_df):
            return None
        
        data_load_duration = time.time() - data_load_start
        self.logger.info(f"⏱️  Data loading completed in {data_load_duration:.2f} seconds")
        
        # Log final memory usage
        final_memory = self.memory_manager.log_memory_usage(self.logger, "data_loading_complete")
        if initial_memory and final_memory:
            memory_increase = final_memory - initial_memory
            self.logger.info(f"📈 Memory increase: {memory_increase:.2f} MB")
        
        return {
            "klines": klines_df,
            "agg_trades": agg_trades_df,
            "futures": futures_df
        }
    
    def _validate_training_data(
        self, 
        klines_df: Optional[pd.DataFrame], 
        agg_trades_df: Optional[pd.DataFrame], 
        futures_df: Optional[pd.DataFrame]
    ) -> bool:
        """
        Validate that training data meets requirements.
        
        Args:
            klines_df: Klines data
            agg_trades_df: Aggregated trades data
            futures_df: Futures data
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        if klines_df is None or klines_df.empty:
            self.logger.error("❌ Klines data is empty or None")
            return False
        
        if klines_df.isnull().any().any():
            self.logger.error("❌ Klines data contains null values")
            return False
        
        # Check for minimum required data
        if len(klines_df) < 1000:  # Minimum data points
            self.logger.warning("⚠️  Klines data seems small, may affect optimization quality")
        
        self.logger.info(f"✅ Data validation passed - Klines shape: {klines_df.shape}")
        return True


class HPORangesManager:
    """Manages HPO ranges loading and saving with validation."""
    
    def __init__(self, config: MultiStageHPOConfig) -> None:
        self.config = config
        self.logger = system_logger.getChild("HPORangesManager")
    
    @handle_specific_errors(
        error_handlers={
            FileNotFoundError: (None, "HPO ranges file not found"),
            json.JSONDecodeError: (None, "Invalid JSON in HPO ranges file"),
            ValueError: (None, "Invalid HPO ranges configuration"),
        },
        default_return=None,
        context="HPO ranges loading"
    )
    async def load_hpo_ranges(self, data_dir: str, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Load HPO ranges from file or create default with validation.
        
        Args:
            data_dir: Directory containing HPO ranges file
            symbol: Trading symbol for file naming
            
        Returns:
            Dict: Validated HPO ranges or None if loading fails
        """
        self.logger.info("🎯 Loading HPO ranges...")
        hpo_load_start = time.time()
        
        hpo_ranges_path = os.path.join(data_dir, f"{symbol}_hpo_ranges.json")
        
        if not os.path.exists(hpo_ranges_path):
            self.logger.warning(f"⚠️  HPO ranges file not found: {hpo_ranges_path}")
            self.logger.info("🔧 Creating default HPO ranges file...")
            
            # Save default HPO ranges
            try:
                async with aiofiles.open(hpo_ranges_path, "w") as f:
                    await f.write(json.dumps(self.config.default_hpo_ranges, indent=4))
                self.logger.info(f"✅ Created default HPO ranges file: {hpo_ranges_path}")
                initial_hpo_ranges = self.config.default_hpo_ranges
            except Exception as e:
                self.logger.error(f"❌ Failed to create default HPO ranges: {e}")
                return None
        else:
            try:
                async with aiofiles.open(hpo_ranges_path, "r") as f:
                    content = await f.read()
                    initial_hpo_ranges = json.loads(content)
            except Exception as e:
                self.logger.error(f"❌ Failed to load HPO ranges: {e}")
                return None
        
        # Validate HPO ranges
        if not self._validate_hpo_ranges(initial_hpo_ranges):
            return None
        
        hpo_load_duration = time.time() - hpo_load_start
        self.logger.info(f"⏱️  HPO ranges loading completed in {hpo_load_duration:.2f} seconds")
        self.logger.info(f"✅ Loaded initial HPO ranges: {len(initial_hpo_ranges)} parameters")
        
        return initial_hpo_ranges
    
    def _validate_hpo_ranges(self, hpo_ranges: Dict[str, Any]) -> bool:
        """
        Validate HPO ranges configuration.
        
        Args:
            hpo_ranges: HPO ranges to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            for param_name, param_config in hpo_ranges.items():
                if not isinstance(param_config, dict):
                    self.logger.error(f"❌ Invalid parameter config for {param_name}")
                    return False
                
                # Check for either 'min'/'max' or 'low'/'high' keys
                has_min_max = all(key in param_config for key in ['min', 'max'])
                has_low_high = all(key in param_config for key in ['low', 'high'])
                
                if not (has_min_max or has_low_high):
                    self.logger.error(f"❌ Missing required keys for {param_name}")
                    return False
                
                # Determine which keys to use for validation
                if has_min_max:
                    min_key, max_key = 'min', 'max'
                else:
                    min_key, max_key = 'low', 'high'
                
                if param_config[max_key] < param_config[min_key]:
                    self.logger.error(f"❌ Invalid range for {param_name}: {max_key} < {min_key}")
                    return False
                
                if param_config['type'] not in ['float', 'int']:
                    self.logger.error(f"❌ Invalid type for {param_name}: {param_config['type']}")
                    return False
            
            self.logger.info("✅ HPO ranges validation passed")
            return True
        except Exception as e:
            self.logger.error(f"❌ HPO ranges validation failed: {e}")
            return False


class OptimizationExecutor:
    """Handles the execution of optimization stages with monitoring."""
    
    def __init__(self, config: MultiStageHPOConfig) -> None:
        self.config = config
        self.processor = MultiStageHPOProcessor(config)
        self.logger = system_logger.getChild("OptimizationExecutor")
        self.structured_logger = StructuredLogger(self.logger)
        self.memory_manager = MemoryManager()
    
    @handle_specific_errors(
        error_handlers={
            Exception: (None, "Optimization execution failed"),
        },
        default_return=None,
        context="optimization execution"
    )
    async def execute_stages(
        self,
        optimizer: Optimizer,
        data: TrainingData,
        initial_ranges: Dict[str, Any],
        symbol: str
    ) -> Optional[List[StageResult]]:
        """
        Execute all optimization stages with progress tracking.
        
        Args:
            optimizer: Optimizer instance
            data: Training data
            initial_ranges: Initial HPO ranges
            symbol: Trading symbol
            
        Returns:
            List[StageResult]: Stage results or None if execution fails
        """
        self.logger.info("🎯 Running 4-stage hyperparameter optimization...")
        optimization_start = time.time()
        
        checkpoint_file_path = os.path.join(
            CONFIG["CHECKPOINT_DIR"],
            f"{symbol}_multi_stage_optimization_checkpoint.pkl",
        )
        
        current_ranges = initial_ranges.copy()
        stage_results = []
        
        # Initialize progress tracking
        progress = OptimizationProgress(
            current_stage=1,
            total_stages=len(self.config.stage_trials),
            current_trial=0,
            total_trials=sum(self.config.stage_trials),
            start_time=datetime.now()
        )
        
        for stage_num, trials in enumerate(self.config.stage_trials, 1):
            progress.update_progress(stage_num, 0)
            
            stage_result = await self._execute_single_stage(
                optimizer, data, current_ranges, checkpoint_file_path, 
                stage_num, trials, progress
            )
            
            if not stage_result:
                return None
            
            stage_results.append(stage_result)
            
            # For stages 1-3, narrow the parameter ranges based on results
            if stage_num < 4:
                current_ranges = self.processor._narrow_parameter_ranges(
                    stage_result["result"], current_ranges, stage_num
                )
                self.logger.info(f"      - Range narrowing completed for stage {stage_num}")
        
        optimization_duration = time.time() - optimization_start
        self.logger.info(f"⏱️  4-stage optimization completed in {optimization_duration:.2f} seconds")
        
        return stage_results
    
    async def _execute_single_stage(
        self,
        optimizer: Optimizer,
        data: TrainingData,
        current_ranges: Dict[str, Any],
        checkpoint_file_path: str,
        stage_num: int,
        trials: int,
        progress: OptimizationProgress
    ) -> Optional[StageResult]:
        """
        Execute a single optimization stage with detailed monitoring.
        
        Args:
            optimizer: Optimizer instance
            data: Training data
            current_ranges: Current parameter ranges
            checkpoint_file_path: Path to checkpoint file
            stage_num: Current stage number
            trials: Number of trials for this stage
            progress: Progress tracking object
            
        Returns:
            StageResult: Stage result or None if execution fails
        """
        async with self.structured_logger.stage_context(stage_num, trials):
            stage_start_time = time.time()
            self.logger.info(f"🔄 STAGE {stage_num}/4: Running {trials} trials...")
            self.logger.info(f"   - Trials: {trials}")
            self.logger.info(f"   - Parameters: {len(current_ranges)}")
            self.logger.info(f"   - Parameter keys: {list(current_ranges.keys())}")
            self.logger.info(f"   - Progress: {progress.progress_percentage:.1f}%")
            
            # Log memory usage before stage
            pre_memory = self.memory_manager.log_memory_usage(self.logger, f"stage_{stage_num}_start")
            
            # Enhanced stage execution with detailed logging
            self.logger.info(f"   🔍 STAGE {stage_num} DETAILED ANALYSIS:")
            self.logger.info(f"      - Start time: {time.strftime('%H:%M:%S', time.localtime(stage_start_time))}")
            self.logger.info(f"      - Checkpoint file: {checkpoint_file_path}")
            self.logger.info(f"      - Data shapes:")
            self.logger.info(f"         * Klines: {data['klines'].shape if data['klines'] is not None else 'None'}")
            self.logger.info(f"         * Agg trades: {data['agg_trades'].shape if data['agg_trades'] is not None else 'None'}")
            self.logger.info(f"         * Futures: {data['futures'].shape if data['futures'] is not None else 'None'}")
            
            # Run optimization for this stage with enhanced timing
            self.logger.info(f"      - Starting optimization...")
            optimization_start = time.time()
            
            try:
                stage_result = await optimizer.implement_global_system_optimization(
                    historical_pnl_data=pd.DataFrame(),
                    strategy_breakdown_data={},
                    checkpoint_file_path=checkpoint_file_path,
                    hpo_ranges=current_ranges,
                    klines_df=data['klines'],
                    agg_trades_df=data['agg_trades'],
                    futures_df=data['futures'],
                )
            except Exception as e:
                self.logger.error(f"      ❌ Stage {stage_num} optimization failed: {e}")
                return None
            
            optimization_time = time.time() - optimization_start
            self.logger.info(f"      - Optimization completed in {optimization_time:.4f} seconds")
            
            if not stage_result:
                self.logger.error(f"      ❌ Stage {stage_num} optimization failed")
                self.logger.error(f"      - Stage result: {stage_result}")
                return None
            
            # Calculate detailed timing
            stage_duration = time.time() - stage_start_time
            self.logger.info(f"      - Total stage duration: {stage_duration:.4f} seconds")
            self.logger.info(f"      - Optimization vs total: {optimization_time/stage_duration*100:.1f}%")
            
            # Ensure minimum timing for very fast operations
            if stage_duration < 0.001:
                self.logger.warning(f"      ⚠️  Stage completed too quickly ({stage_duration:.6f}s), adding minimum timing")
                stage_duration = max(stage_duration, 0.001)  # Minimum 1ms for logging purposes
            
            # Log memory usage after stage
            post_memory = self.memory_manager.log_memory_usage(self.logger, f"stage_{stage_num}_end")
            if pre_memory and post_memory:
                memory_change = post_memory - pre_memory
                self.logger.info(f"      - Memory change: {memory_change:+.2f} MB")
            
            self.logger.info(f"      ✅ Stage {stage_num} completed successfully")
            self.logger.info(f"      - Final duration: {stage_duration:.4f} seconds")
            self.logger.info(f"      - Result type: {type(stage_result).__name__}")
            self.logger.info(f"      - Result keys: {list(stage_result.keys()) if isinstance(stage_result, dict) else 'N/A'}")
            
            return {
                "stage": stage_num,
                "trials": trials,
                "duration": stage_duration,
                "optimization_time": optimization_time,
                "result": stage_result,
            }


class ResultsManager:
    """Handles saving and managing optimization results with metrics."""
    
    def __init__(self) -> None:
        self.logger = system_logger.getChild("ResultsManager")
        self.memory_manager = MemoryManager()
    
    @handle_specific_errors(
        error_handlers={
            IOError: (False, "Failed to save results file"),
            json.JSONDecodeError: (False, "Failed to serialize results"),
        },
        default_return=False,
        context="results saving"
    )
    async def save_results(
        self,
        stage_results: List[StageResult],
        current_ranges: Dict[str, Any],
        data_dir: str,
        symbol: str,
        total_duration: float
    ) -> bool:
        """
        Save final optimization results with comprehensive metrics.
        
        Args:
            stage_results: Results from all optimization stages
            current_ranges: Final parameter ranges
            data_dir: Directory to save results
            symbol: Trading symbol
            total_duration: Total optimization duration
            
        Returns:
            bool: True if saving succeeded, False otherwise
        """
        self.logger.info("💾 Saving final results...")
        save_start = time.time()
        
        # Calculate comprehensive metrics
        metrics = self._calculate_optimization_metrics(stage_results, total_duration)
        
        results_file = os.path.join(data_dir, f"{symbol}_multi_stage_hpo_results.json")
        
        try:
            async with aiofiles.open(results_file, "w") as f:
                results_data = {
                    "stage_results": stage_results,
                    "total_trials": sum(result["trials"] for result in stage_results),
                    "total_duration": total_duration,
                    "final_ranges": current_ranges,
                    "metrics": metrics,
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                }
                await f.write(json.dumps(results_data, indent=4, default=str))
        except Exception as e:
            self.logger.error(f"❌ Failed to save results: {e}")
            return False
        
        save_duration = time.time() - save_start
        self.logger.info(f"⏱️  Results saved in {save_duration:.2f} seconds")
        self.logger.info(f"📁 Results saved to: {results_file}")
        
        # Log final metrics
        self._log_optimization_metrics(metrics)
        
        return True
    
    def _calculate_optimization_metrics(
        self, 
        stage_results: List[StageResult], 
        total_duration: float
    ) -> OptimizationMetrics:
        """
        Calculate comprehensive optimization metrics.
        
        Args:
            stage_results: Results from all stages
            total_duration: Total optimization duration
            
        Returns:
            OptimizationMetrics: Calculated metrics
        """
        stage_breakdown = {}
        total_optimization_time = 0
        
        for result in stage_results:
            stage_num = result["stage"]
            stage_breakdown[stage_num] = result["duration"]
            total_optimization_time += result["optimization_time"]
        
        optimization_vs_overhead = (total_optimization_time / total_duration) * 100 if total_duration > 0 else 0
        
        # Get current memory usage
        memory_usage = self.memory_manager.log_memory_usage(self.logger, "final_metrics")
        
        return {
            "total_duration": total_duration,
            "stage_breakdown": stage_breakdown,
            "optimization_vs_overhead": optimization_vs_overhead,
            "memory_usage_mb": memory_usage
        }
    
    def _log_optimization_metrics(self, metrics: OptimizationMetrics) -> None:
        """Log comprehensive optimization metrics."""
        self.logger.info("📊 OPTIMIZATION METRICS:")
        self.logger.info(f"   - Total duration: {metrics['total_duration']:.2f} seconds")
        self.logger.info(f"   - Optimization efficiency: {metrics['optimization_vs_overhead']:.1f}%")
        if metrics['memory_usage_mb']:
            self.logger.info(f"   - Final memory usage: {metrics['memory_usage_mb']:.2f} MB")
        
        self.logger.info("   - Stage breakdown:")
        for stage, duration in metrics['stage_breakdown'].items():
            percentage = (duration / metrics['total_duration']) * 100
            self.logger.info(f"     * Stage {stage}: {duration:.2f}s ({percentage:.1f}%)")


# ==============================================================================
# Main Execution Function
# ==============================================================================

@handle_errors(
    exceptions=(Exception,),
    default_return=False,
    context="multi_stage_hpo_step",
)
async def run_step(
    symbol: str,
    data_dir: str,
    data_file_path: str,
    timeframe: str = "1m",
) -> bool:
    """
    Execute 4-stage hyperparameter optimization pipeline.
    
    This function orchestrates a multi-stage optimization process:
    1. Stage 1: Ultra-Coarse (5 trials) - Very wide ranges
    2. Stage 2: Coarse (20 trials) - Narrowed ranges from Stage 1
    3. Stage 3: Medium (30 trials) - Further narrowed ranges from Stage 2
    4. Stage 4: Fine (50 trials) - Final optimization with precise ranges
    
    The optimization uses adaptive range narrowing based on best results from each stage.
    
    Args:
        symbol: Trading symbol (e.g., 'ETHUSDT')
        data_dir: Directory containing training data
        data_file_path: Path to pickle file with collected data
        timeframe: Timeframe for analysis (default: '1m')
    
    Returns:
        bool: True if optimization completed successfully, False otherwise
    
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data validation fails
        RuntimeError: If optimization process fails
    
    Example:
        >>> success = await run_step('ETHUSDT', '/data', '/data/eth_data.pkl')
        >>> print(f"Optimization {'succeeded' if success else 'failed'}")
    """
    setup_logging()
    logger = system_logger.getChild("Step5MultiStageHPO")

    start_time = time.time()
    logger.info("=" * 80)
    logger.info("🚀 STEP 5: 4-STAGE HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🎯 Symbol: {symbol}")
    logger.info(f"📁 Data directory: {data_dir}")
    logger.info(f"📦 Data file path: {data_file_path}")

    try:
        # Initialize configuration
        config = MultiStageHPOConfig.from_environment()
        logger.info(f"🔢 Stage trials: {config.stage_trials} (Total: {sum(config.stage_trials)} trials)")

        # Initialize components
        data_loader = DataLoader()
        hpo_ranges_manager = HPORangesManager(config)
        optimization_executor = OptimizationExecutor(config)
        results_manager = ResultsManager()

        # Step 5.1: Load data from pickle file
        logger.info("📊 STEP 5.1: Loading data from pickle file...")
        data_load_start = time.time()

        training_data = await data_loader.load_training_data(data_file_path)
        if not training_data:
            logger.error("❌ Failed to load training data")
            return False

        data_load_duration = time.time() - data_load_start
        logger.info(f"⏱️  Data loading completed in {data_load_duration:.2f} seconds")

        # Step 5.2: Load initial HPO ranges
        logger.info("🎯 STEP 5.2: Loading initial HPO ranges...")
        hpo_load_start = time.time()

        initial_hpo_ranges = await hpo_ranges_manager.load_hpo_ranges(data_dir, symbol)
        if not initial_hpo_ranges:
            logger.error("❌ Failed to load HPO ranges")
            return False

        hpo_load_duration = time.time() - hpo_load_start
        logger.info(f"⏱️  HPO ranges loading completed in {hpo_load_duration:.2f} seconds")

        # Step 5.3: Initialize database manager
        logger.info("🗄️  STEP 5.3: Initializing database manager...")
        db_init_start = time.time()

        db_manager = SQLiteManager({})
        await db_manager.initialize()

        db_init_duration = time.time() - db_init_start
        logger.info(f"⏱️  Database initialization completed in {db_init_duration:.2f} seconds")

        # Step 5.4: Initialize optimizer
        logger.info("🔧 STEP 5.4: Initializing optimizer...")
        optimizer_init_start = time.time()

        optimizer = Optimizer(config=CONFIG)

        optimizer_init_duration = time.time() - optimizer_init_start
        logger.info(f"⏱️  Optimizer initialization completed in {optimizer_init_duration:.2f} seconds")

        # Step 5.5: Run 4-stage hyperparameter optimization
        logger.info("🎯 STEP 5.5: Running 4-stage hyperparameter optimization...")
        optimization_start = time.time()

        stage_results = await optimization_executor.execute_stages(
            optimizer, training_data, initial_hpo_ranges, symbol
        )

        if not stage_results:
            logger.error("❌ Optimization stages failed")
            return False

        optimization_duration = time.time() - optimization_start
        logger.info(f"⏱️  4-stage optimization completed in {optimization_duration:.2f} seconds")

        # Step 5.6: Save final results
        logger.info("💾 STEP 5.6: Saving final results...")
        save_start = time.time()

        # Get final ranges from the last stage
        final_ranges = initial_hpo_ranges.copy()
        for stage_result in stage_results[:-1]:  # All except last stage
            final_ranges = optimization_executor.processor._narrow_parameter_ranges(
                stage_result["result"], final_ranges, stage_result["stage"]
            )

        success = await results_manager.save_results(
            stage_results, final_ranges, data_dir, symbol, optimization_duration
        )

        if not success:
            logger.error("❌ Failed to save results")
            return False

        save_duration = time.time() - save_start
        logger.info(f"⏱️  Results saved in {save_duration:.2f} seconds")

        # Final summary
        total_duration = time.time() - start_time
        logger.info("=" * 80)
        logger.info("🎉 STEP 5: 4-STAGE HYPERPARAMETER OPTIMIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total duration: {total_duration:.2f} seconds")
        logger.info("📊 Performance breakdown:")
        logger.info(
            f"   - Data loading: {data_load_duration:.2f}s ({(data_load_duration/total_duration)*100:.1f}%)",
        )
        logger.info(
            f"   - HPO loading: {hpo_load_duration:.2f}s ({(hpo_load_duration/total_duration)*100:.1f}%)",
        )
        logger.info(
            f"   - DB initialization: {db_init_duration:.2f}s ({(db_init_duration/total_duration)*100:.1f}%)",
        )
        logger.info(
            f"   - Optimizer setup: {optimizer_init_duration:.2f}s ({(optimizer_init_duration/total_duration)*100:.1f}%)",
        )
        logger.info(
            f"   - 4-stage optimization: {optimization_duration:.2f}s ({(optimization_duration/total_duration)*100:.1f}%)",
        )
        logger.info(
            f"   - Saving: {save_duration:.2f}s ({(save_duration/total_duration)*100:.1f}%)",
        )

        logger.info("📈 Stage breakdown:")
        for stage_result in stage_results:
            logger.info(
                f"   - Stage {stage_result['stage']}: {stage_result['trials']} trials in {stage_result['duration']:.2f}s",
            )

        logger.info("✅ Success: True")

        return True

    except Exception as e:
        total_duration = time.time() - start_time
        logger.error("=" * 80)
        logger.error("❌ STEP 5: 4-STAGE HYPERPARAMETER OPTIMIZATION FAILED")
        logger.error("=" * 80)
        logger.error(f"⏱️  Duration before failure: {total_duration:.2f} seconds")
        logger.error(f"💥 Error: {e}")
        logger.error("📋 Full traceback:", exc_info=True)
        return False


if __name__ == "__main__":
    # Command-line arguments: symbol, data_dir, data_file_path
    symbol = sys.argv[1]
    data_dir = sys.argv[2]
    data_file_path = sys.argv[3]

    success = asyncio.run(run_step(symbol, data_dir, data_file_path))

    if not success:
        sys.exit(1)  # Indicate failure
    sys.exit(0)  # Indicate success
