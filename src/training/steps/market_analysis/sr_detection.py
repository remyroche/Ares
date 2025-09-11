"""SR Detection Stage: Detect Support/Resistance levels using Enhanced SR Detection."""

import asyncio
import sys
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
try:
    from collections.abc import Iterable
except ImportError:
    from typing import Iterable
import time
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import traceback
import logging
import random
import gc

# Core imports
try:
    from src.training.base_step import BaseStep
except ImportError:
    # Fallback BaseStep class
    class BaseStep:
        def __init__(self, config):
            self.config = config
        
        async def execute(self, data):
            pass
        
        def validate_config(self):
            pass
        
        def get_status(self):
            return {}

from src.utils.logger import system_logger

# Initialize logger early to avoid usage before definition
logger = system_logger.getChild('SRDetection')

# Required utility modules - Comprehensive Integration
from src.utils.common_operations import (
    safe_json_load, safe_json_dump, safe_read_parquet, safe_to_parquet,
    ensure_directory, create_fallback_logger, create_fallback_decorator,
    safe_mean, safe_std, safe_float, safe_int, safe_append, safe_extend,
    safe_dict_get, safe_dict_items, safe_lower, safe_upper, safe_join,
    get_current_datetime, format_datetime, create_empty_dataframe,
    safe_fillna, safe_rolling, safe_copy, safe_deepcopy, safe_sleep,
    safe_gather, create_async_task, get_logger, setup_basic_logging,
    safe_exception_handler, suggest_float_uniform, suggest_int_uniform,
    validate_dataframe, validate_numeric_range, optimize_dataframe_dtypes,
    timed_operation, format_bytes, chunked_iterable, parallel_map,
    safe_log_metric, safe_log_params, safe_log_artifact, get_common_operations_health_status
)

# Core decorators and errors
from src.core.decorators import handles_errors, error_boundary, converts_errors
from src.core.errors import (
    AppError, ValidationError, DataIntegrityError, 
    NotFoundError, BusinessRuleError
)

# Pipeline standards and utilities
from src.utils.pipeline_standards import PipelineStandards
from src.utils.monitoring_utils import (
    global_monitor, function_tracker, logging_patterns
)
from src.utils.comprehensive_function_logger import (
    log_step_functions, log_important_calls, log_all_calls, 
    log_internal_call, log_step_progress, log_data_operation
)

# Enhanced SR Detection imports
try:
    from src.tactician.sr_levels.enhanced_sr_detection import EnhancedSRDetector, SRLevel
    ENHANCED_SR_DETECTOR_AVAILABLE = True
    logger.info('✅ Enhanced SR Detector available')
except ImportError as e:
    EnhancedSRDetector = None
    SRLevel = None
    ENHANCED_SR_DETECTOR_AVAILABLE = False
    logger.warning(f'⚠️ Enhanced SR Detector not available: {e}')

# M1 Optimization Utilities - Integrated via Common Operations
try:
    from src.utils.common_operations import (
        integrate_with_m1_optimizers, get_m1_gpu_manager, get_m1_memory_optimizer,
        get_m1_cpu_optimizer, cleanup_m1_optimizers, memory_checkpoint, gpu_context,
        optimize_memory, get_memory_usage
    )

    # Initialize M1 integration through common operations
    m1_integration_result = integrate_with_m1_optimizers()
    M1_GPU_AVAILABLE = m1_integration_result.get('gpu_manager', False)
    M1_MEMORY_AVAILABLE = m1_integration_result.get('memory_optimizer', False)
    M1_CPU_AVAILABLE = m1_integration_result.get('cpu_optimizer', False)
    M1_BATCH_AVAILABLE = M1_CPU_AVAILABLE  # Batch processor available if CPU optimizer is

    integration_status = m1_integration_result.get('integration_status', 'unknown')
    if integration_status == 'success':
        logger.info("✅ Complete M1 utilities integration successful")
    elif integration_status == 'partial':
        logger.info("⚠️ Partial M1 utilities integration - some components available")
    else:
        logger.warning("❌ M1 utilities integration failed")

except ImportError as e:
    M1_GPU_AVAILABLE = False
    M1_MEMORY_AVAILABLE = False
    M1_CPU_AVAILABLE = False
    M1_BATCH_AVAILABLE = False
    logger.warning(f"M1 utilities integration not available: {e}")
except Exception as e:
    M1_GPU_AVAILABLE = False
    M1_MEMORY_AVAILABLE = False
    M1_CPU_AVAILABLE = False
    M1_BATCH_AVAILABLE = False
    logger.error(f"Unexpected error in M1 utilities integration: {e}")

# Utility functions for memory management and validation
def get_memory_usage():
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        return 0

def format_bytes(bytes_val):
    return f"{bytes_val / 1024 / 1024:.1f} MB"

def memory_checkpoint(name):
    pass

def optimize_dataframe_dtypes(df):
    return df

def validate_dataframe(df):
    return True

# Import standardized math validation utilities
from src.utils.math_validation import validate_finite, safe_divide


class SRDetectionStep(BaseStep):
    """SR Detection Stage: Detect Support/Resistance levels using Enhanced SR Detection."""

    @log_important_calls
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize SR detection step."""
        super().__init__(config)
        self.logger = system_logger.getChild('SRDetectionStep')
        self.standards = PipelineStandards(self.logger)
        self.sr_optimization_config = config.get('sr_optimization', {
            'min_touches': 2, 
            'tolerance_pct': 0.5, 
            'lookback_periods': 100
        })

        # Adjust configuration for LIGHT mode
        training_mode = os.environ.get('LIGHT_TRAINING_MODE', '')
        if training_mode == '1' or config.get('training_mode') == 'light':
            self.sr_optimization_config['lookback_periods'] = 10
            self.logger.info('💡 LIGHT mode: Adjusted lookback_periods to 10 (was 100)')
        
        # Configurable proximity threshold for SR classification
        self.proximity_threshold = config.get('sr_optimization', {}).get('proximity_threshold', 0.002)  # Default 0.2%
        
        # Min/max SR ratio configuration
        self.min_sr_ratio = config.get('sr_optimization', {}).get('min_sr_ratio', 0.15)  # Default 15% minimum SR ratio
        self.max_sr_ratio = config.get('sr_optimization', {}).get('max_sr_ratio', 0.30)  # Default 30% maximum SR ratio
        self.sr_ratio_adjustment_attempts = config.get('sr_optimization', {}).get('sr_ratio_adjustment_attempts', 5)  # Max adjustment attempts
        
        # Initialize automatic memory management
        try:
            from src.utils.hardware.memory_optimization import get_memory_manager, MemoryContext as memory_context
            self.memory_manager = get_memory_manager()
            self.memory_manager.start_monitoring()
            self.logger.info("🧠 Memory management initialized")
        except Exception as e:
            self.logger.warning(f"Memory manager initialization failed: {e}")
            # Fallback memory manager
            class FallbackMemoryManager:
                def start_monitoring(self):
                    pass
                def stop_monitoring(self):
                    pass
            self.memory_manager = FallbackMemoryManager()

    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the SR detection stage."""
        self.logger.info('🎯 Starting SR Detection Stage execution')
        start_time = time.time()

        try:
            # Get data from pipeline state
            data = pipeline_state.get('dataframe')
            if data is None:
                raise ValueError("No dataframe found in pipeline state")

            self.logger.info(f'📊 Data loaded: {data.shape[0]:,} rows, {data.shape[1]} columns')

            # Detect SR levels
            sr_levels = self._detect_sr_levels(data)
            
            execution_time = time.time() - start_time
            self.logger.info(f'✅ SR Detection completed in {execution_time:.2f} seconds')

            return {
                'success': True,
                'sr_levels': sr_levels,
                'execution_time': execution_time,
                'stage': 'sr_detection'
            }

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f'❌ SR Detection failed: {e}')
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'stage': 'sr_detection'
            }

    def _detect_sr_levels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect support and resistance levels using Enhanced SR Detection."""
        self.logger.info('🎯 ===== STARTING SR DETECTION PROCESS =====')
        self.logger.info('🎯 Using Enhanced SR Detection with multiple advanced algorithms...')
        detection_start_time = time.time()

        # CRITICAL: Validate input data before S/R detection
        if data is None:
            self.logger.error('❌ CRITICAL: Input data is None for S/R detection. Cannot proceed.')
            raise ValueError("CRITICAL: Input data is None for S/R detection. Cannot proceed.")

        if data.empty:
            self.logger.error('❌ CRITICAL: Input data is empty for S/R detection. Cannot proceed.')
            raise ValueError("CRITICAL: Input data is empty for S/R detection. Cannot proceed.")

        self.logger.info(f'📊 Input data shape: {data.shape[0]:,} rows × {data.shape[1]} columns')
        self.logger.info(f'📊 Input data columns: {list(data.columns)}')
        self.logger.info(f'📊 Input data memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB')

        # Use comprehensive data validation
        self.logger.info('🔍 Starting comprehensive data validation for SR detection...')
        validation_start = time.time()
        clean_data = self._validate_price_data_quality(data)
        validation_time = time.time() - validation_start

        self.logger.info(f'✅ S/R detection input validation passed: {len(clean_data)} rows, {len(clean_data.columns)} columns')
        self.logger.info(f'⏱️ Data validation took: {validation_time:.2f} seconds')
        self.logger.info(f'📊 Data reduction: {len(data) - len(clean_data)} rows removed ({(len(data) - len(clean_data))/len(data)*100:.1f}%)')

        try:
            # First, get basic SR levels using existing detector
            if not ENHANCED_SR_DETECTOR_AVAILABLE or EnhancedSRDetector is None or SRLevel is None:
                self.logger.error('❌ Enhanced SR Detector not available for initial detection.')
                raise RuntimeError("Enhanced SR Detector not available for initial detection.")
            
            self.logger.info('✅ Enhanced SR Detector is available, proceeding with detection...')
            
            # Create basic SR detector for initial detection
            sr_config = {
                'min_touches': getattr(self, 'min_touches', 2),
                'tolerance_pct': getattr(self, 'tolerance_pct', 0.5),
                'lookback_periods': getattr(self, 'lookback_periods', 100),
                'memory_efficient': True,
                'use_parallel': getattr(self, 'enable_parallel_processing', False),
            }
            
            self.logger.info(f'🔧 SR Detection Configuration:')
            self.logger.info(f'   • Min touches: {sr_config["min_touches"]}')
            self.logger.info(f'   • Tolerance %: {sr_config["tolerance_pct"]}%')
            self.logger.info(f'   • Lookback periods: {sr_config["lookback_periods"]}')
            self.logger.info(f'   • Memory efficient: {sr_config["memory_efficient"]}')
            self.logger.info(f'   • Parallel processing: {sr_config["use_parallel"]}')
            
            self.logger.info('🎯 Creating Enhanced SR Detector...')
            detector_creation_start = time.time()
            detector = EnhancedSRDetector(sr_config)
            detector_creation_time = time.time() - detector_creation_start
            self.logger.info(f'✅ Enhanced SR Detector created in {detector_creation_time:.2f} seconds')
            
            self.logger.info('🔍 Starting basic SR level detection...')
            basic_detection_start = time.time()
            basic_sr_levels = detector.detect_sr_levels(clean_data)
            basic_detection_time = time.time() - basic_detection_start
            self.logger.info(f'✅ Basic SR level detection completed in {basic_detection_time:.2f} seconds')
            
            # Convert to list format for further processing
            self.logger.info('🔄 Converting basic SR levels to list format...')
            if isinstance(basic_sr_levels, dict):
                support_levels = basic_sr_levels.get('support_levels', [])
                resistance_levels = basic_sr_levels.get('resistance_levels', [])
                all_levels = support_levels + resistance_levels
                self.logger.info(f'📊 Basic detection results: {len(support_levels)} support levels, {len(resistance_levels)} resistance levels')
            else:
                all_levels = basic_sr_levels if isinstance(basic_sr_levels, list) else []
                self.logger.info(f'📊 Basic detection results: {len(all_levels)} total levels (format: list)')
            
            if not all_levels:
                self.logger.error('❌ No basic SR levels detected.')
                raise RuntimeError("No basic SR levels detected.")
            
            self.logger.info(f'✅ Total basic SR levels detected: {len(all_levels)}')
            
            # Convert levels to dict format for consistency
            self.logger.info('🔄 Converting SR levels to dictionary format...')
            levels_dict = []
            conversion_start = time.time()
            
            for i, level in enumerate(all_levels):
                if hasattr(level, 'price'):
                    level_dict = {
                        'price': level.price,
                        'strength': getattr(level, 'strength', 0.5),
                        'level_type': getattr(level, 'type', 'support'),
                        'touch_count': getattr(level, 'touch_count', 2),
                        'first_touch': getattr(level, 'first_touch', datetime.now() - timedelta(days=30)),
                        'last_touch': getattr(level, 'last_touch', datetime.now() - timedelta(days=1))
                    }
                    levels_dict.append(level_dict)
                    
                    # Log every 10th level for progress tracking
                    if (i + 1) % 10 == 0 or i == len(all_levels) - 1:
                        self.logger.info(f'   📊 Converted {i + 1}/{len(all_levels)} levels ({(i + 1)/len(all_levels)*100:.1f}%)')
            
            conversion_time = time.time() - conversion_start
            self.logger.info(f'✅ Level conversion completed in {conversion_time:.2f} seconds')
            self.logger.info(f'📊 Converted {len(levels_dict)} levels to dictionary format')
            
            # Separate support and resistance levels
            support_levels = [level for level in levels_dict if level.get('level_type', '').lower() == 'support']
            resistance_levels = [level for level in levels_dict if level.get('level_type', '').lower() == 'resistance']
            
            detection_time = time.time() - detection_start_time
            
            self.logger.info('🎯 ===== SR DETECTION PROCESS COMPLETED =====')
            self.logger.info(f'✅ Total detection time: {detection_time:.2f} seconds')
            self.logger.info(f'📊 Final results: {len(support_levels)} support levels, {len(resistance_levels)} resistance levels')
            
            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'all_levels': levels_dict,
                'detection_time': detection_time,
                'detection_config': sr_config,
                'data_shape': clean_data.shape,
                'validation_time': validation_time
            }

        except Exception as e:
            detection_time = time.time() - detection_start_time
            self.logger.error(f'❌ SR Detection failed after {detection_time:.2f} seconds: {e}')
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            raise

    def _validate_price_data_quality(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean price data for SR detection."""
        self.logger.info('🔍 Starting price data quality validation...')
        
        # Check for required OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required OHLCV columns: {missing_cols}")
        
        # Create a copy to avoid modifying original data
        clean_data = data.copy()
        
        # Remove rows with NaN values in critical columns
        initial_rows = len(clean_data)
        clean_data = clean_data.dropna(subset=required_cols)
        removed_rows = initial_rows - len(clean_data)
        
        if removed_rows > 0:
            self.logger.info(f'🧹 Removed {removed_rows} rows with NaN values in OHLCV columns')
        
        # Validate price relationships (high >= low, etc.)
        invalid_high_low = clean_data['high'] < clean_data['low']
        invalid_open_close = (clean_data['open'] <= 0) | (clean_data['close'] <= 0)
        invalid_volume = clean_data['volume'] < 0
        
        invalid_rows = invalid_high_low | invalid_open_close | invalid_volume
        if invalid_rows.any():
            invalid_count = invalid_rows.sum()
            self.logger.warning(f'⚠️ Found {invalid_count} rows with invalid price relationships')
            clean_data = clean_data[~invalid_rows]
        
        # Ensure positive prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (clean_data[col] <= 0).any():
                self.logger.warning(f'⚠️ Found non-positive values in {col} column')
                clean_data = clean_data[clean_data[col] > 0]
        
        final_rows = len(clean_data)
        total_removed = initial_rows - final_rows
        
        if total_removed > 0:
            self.logger.info(f'✅ Data validation completed: {total_removed} rows removed ({total_removed/initial_rows*100:.1f}%)')
            self.logger.info(f'📊 Final dataset: {final_rows:,} rows')
        else:
            self.logger.info('✅ Data validation completed: No rows removed')
        
        return clean_data

    def _get_fallback_sr_levels(self) -> Dict[str, Any]:
        """Get fallback SR levels when detection fails."""
        self.logger.warning('⚠️ Using fallback SR levels due to detection failure')
        
        return {
            'support_levels': [],
            'resistance_levels': [],
            'all_levels': [],
            'detection_time': 0.0,
            'detection_config': {},
            'data_shape': (0, 0),
            'validation_time': 0.0,
            'fallback': True
        }