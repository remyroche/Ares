"""Step 2.5: S/R Detection Optimization - Main orchestrator combining all three stages."""

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
logger = system_logger.getChild('SROptimizationOrchestrator')

# Import the three stages
from .sr_detection import SRDetectionStep
from .sr_clustering import SRClusteringStep
from .sr_ml_learning import SRMLLearningStep

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

def validate_finite(arr):
    return np.all(np.isfinite(arr))

def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)


class SROptimizationStep(BaseStep):
    """Step 2.5: S/R Detection Optimization - Main orchestrator combining all three stages."""

    @log_important_calls
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize SR optimization step with comprehensive utility integration."""
        super().__init__(config)
        self.logger = system_logger.getChild('SROptimizationStep')
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
        
        # Initialize the three stages
        self.sr_detection_step = SRDetectionStep(config)
        self.sr_clustering_step = SRClusteringStep(config)
        self.sr_ml_learning_step = SRMLLearningStep(config)
        
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
        """Execute the complete SR optimization pipeline with all three stages."""
        self.logger.info('🎯 Starting Step 2.5 execution with comprehensive monitoring')
        start_time = time.time()

        try:
            # Stage 1: SR Detection
            self.logger.info('🎯 ===== STAGE 1: SR DETECTION =====')
            detection_result = await self.sr_detection_step.execute(training_input, pipeline_state)
            
            if not detection_result.get('success', False):
                self.logger.error('❌ SR Detection stage failed')
                return {
                    'success': False,
                    'error': f"SR Detection failed: {detection_result.get('error', 'Unknown error')}",
                    'stage': 'sr_detection',
                    'execution_time': time.time() - start_time
                }
            
            # Update pipeline state with detection results
            pipeline_state['sr_levels'] = detection_result['sr_levels']
            
            # Stage 2: SR Clustering
            self.logger.info('🚀 ===== STAGE 2: SR CLUSTERING =====')
            clustering_result = await self.sr_clustering_step.execute(training_input, pipeline_state)
            
            if not clustering_result.get('success', False):
                self.logger.error('❌ SR Clustering stage failed')
                return {
                    'success': False,
                    'error': f"SR Clustering failed: {clustering_result.get('error', 'Unknown error')}",
                    'stage': 'sr_clustering',
                    'execution_time': time.time() - start_time
                }
            
            # Update pipeline state with clustering results
            pipeline_state['clustered_levels'] = clustering_result['clustered_levels']
            
            # Stage 3: SR ML Learning
            self.logger.info('🤖 ===== STAGE 3: SR ML LEARNING =====')
            ml_learning_result = await self.sr_ml_learning_step.execute(training_input, pipeline_state)
            
            if not ml_learning_result.get('success', False):
                self.logger.error('❌ SR ML Learning stage failed')
                return {
                    'success': False,
                    'error': f"SR ML Learning failed: {ml_learning_result.get('error', 'Unknown error')}",
                    'stage': 'sr_ml_learning',
                    'execution_time': time.time() - start_time
                }
            
            # Combine all results
            execution_time = time.time() - start_time
            
            self.logger.info('🎯 ===== STEP 2.5 COMPLETED SUCCESSFULLY =====')
            self.logger.info(f'✅ Total execution time: {execution_time:.2f} seconds')
            self.logger.info(f'📊 Detection time: {detection_result.get("execution_time", 0):.2f} seconds')
            self.logger.info(f'📊 Clustering time: {clustering_result.get("execution_time", 0):.2f} seconds')
            self.logger.info(f'📊 ML Learning time: {ml_learning_result.get("execution_time", 0):.2f} seconds')
            
            return {
                'success': True,
                'sr_levels': detection_result['sr_levels'],
                'clustered_levels': clustering_result['clustered_levels'],
                'ml_results': ml_learning_result['ml_results'],
                'execution_time': execution_time,
                'stage_times': {
                    'detection': detection_result.get('execution_time', 0),
                    'clustering': clustering_result.get('execution_time', 0),
                    'ml_learning': ml_learning_result.get('execution_time', 0)
                },
                'stage': 'complete_sr_optimization'
            }

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f'❌ Step 2.5 execution failed: {e}')
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'stage': 'complete_sr_optimization'
            }

    def validate_config(self):
        """Validate the configuration for all stages."""
        self.logger.info('🔍 Validating configuration for all SR optimization stages...')
        
        try:
            # Validate detection stage
            self.sr_detection_step.validate_config()
            self.logger.info('✅ SR Detection stage configuration validated')
            
            # Validate clustering stage
            self.sr_clustering_step.validate_config()
            self.logger.info('✅ SR Clustering stage configuration validated')
            
            # Validate ML learning stage
            self.sr_ml_learning_step.validate_config()
            self.logger.info('✅ SR ML Learning stage configuration validated')
            
            self.logger.info('✅ All stage configurations validated successfully')
            return True
            
        except Exception as e:
            self.logger.error(f'❌ Configuration validation failed: {e}')
            return False

    def get_status(self):
        """Get the status of all stages."""
        return {
            'detection_stage': self.sr_detection_step.get_status(),
            'clustering_stage': self.sr_clustering_step.get_status(),
            'ml_learning_stage': self.sr_ml_learning_step.get_status(),
            'overall_status': 'ready'
        }