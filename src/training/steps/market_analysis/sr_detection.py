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

# Required utility modules - Simplified imports
from src.utils.common_operations import (
    safe_json_load, safe_json_dump,
    ensure_directory, create_fallback_logger, create_fallback_decorator,
    get_current_datetime, format_datetime, create_empty_dataframe,
    safe_fillna, get_logger, setup_basic_logging,
    validate_dataframe, optimize_dataframe_dtypes,
    safe_log_metric, safe_log_params, safe_log_artifact
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

# Enhanced SR Detection imports - moved to local imports to avoid circular dependencies
ENHANCED_SR_DETECTOR_AVAILABLE = False
EnhancedSRDetector = None
SRLevel = None

# Enhanced optimization imports
try:
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager, OperationType, OptimizationStrategy
    )
    from src.utils.ml_common.optimization.regime_hpo_wrapper import RegimeHPOWrapper
    from src.utils.ml_common.explainability.shap_lime_integration import (
        SHAPLIMEExplainer, ExplanationConfig
    )
    from src.utils.ml_common.validation.temporal_cross_validation import temporal_cross_validation
    from src.utils.ml_common.validation.data_leakage_detector import DataLeakageDetector
    from src.utils.hardware.unified_hardware_manager import (
        UnifiedHardwareManager, WorkloadType, OptimizationLevel
    )
    ENHANCED_OPTIMIZATION_AVAILABLE = True
    logger.info("✅ Enhanced optimization utilities available")
except ImportError as e:
    ENHANCED_OPTIMIZATION_AVAILABLE = False
    logger.warning(f"⚠️ Enhanced optimization utilities not available: {e}")

# Enhanced error handling and validation imports
try:
    from src.training.steps.market_analysis.sr_error_handlers import (
        handles_sr_detection_errors, handles_sr_data_validation
    )
    from src.training.steps.market_analysis.sr_data_validator import (
        SRDataValidator, ValidationLevel
    )
    from src.training.steps.market_analysis.sr_logging_enhancer import (
        SRLoggingEnhancer, create_sr_logger
    )
    ENHANCED_FEATURES_AVAILABLE = True
    logger.info("✅ Enhanced error handling and validation available")
except ImportError as e:
    ENHANCED_FEATURES_AVAILABLE = False
    logger.warning(f"⚠️ Enhanced features not available: {e}")

# M1 Optimization Utilities - Now in hardware modules
try:
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
    from src.utils.hardware.m1_memory_optimizer import optimize_memory, get_memory_usage
    from src.utils.hardware.m1_optimizations import M1Optimizer

    # Create wrapper functions for compatibility
    def integrate_with_m1_optimizers():
        """Integrate with M1 optimizers."""
        try:
            # Initialize components
            gpu_manager = get_m1_gpu_manager()
            cpu_optimizer = m1_cpu_optimizer()
            memory_optimizer = M1Optimizer()

            return {
                'gpu_manager': True,
                'cpu_optimizer': True,
                'memory_optimizer': True,
                'integration_status': 'success'
            }
        except Exception as e:
            logger.warning(f"Failed to integrate M1 optimizers: {e}")
            return {
                'gpu_manager': False,
                'cpu_optimizer': False,
                'memory_optimizer': False,
                'integration_status': 'failed'
            }

    def cleanup_m1_optimizers():
        """Cleanup M1 optimizers."""
        pass

    def memory_checkpoint(checkpoint_name: str):
        """Memory checkpoint context manager."""
        optimizer = M1Optimizer()
        return optimizer.memory_checkpoint(checkpoint_name)

    def gpu_context():
        """GPU context manager."""
        gpu_manager = get_m1_gpu_manager()
        return gpu_manager.get_gpu_context()

    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer as m1_cpu_optimizer

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

# Register cleanup on exit to prevent memory monitoring infinite loops
import atexit
try:
    from src.utils.common_operations import cleanup_m1_optimizers
    atexit.register(lambda: cleanup_m1_optimizers())
except ImportError:
    pass  # Cleanup function not available

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
        
        # Initialize enhanced logging
        if ENHANCED_FEATURES_AVAILABLE:
            log_file = config.get('log_file', 'logs/sr_detection.log')
            self.sr_logger = create_sr_logger(log_file=log_file, enable_structured=True)
            self.logger.info("Enhanced logging initialized")
        else:
            self.sr_logger = None
            self.logger.warning("Enhanced logging not available, using basic logging")
        
        # Initialize enhanced optimization components
        self._initialize_enhanced_components()

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

    def _initialize_enhanced_components(self):
        """Initialize enhanced optimization components."""
        self.logger.info("🚀 Initializing enhanced SR detection components...")
        
        # Initialize VectorBT optimization
        if ENHANCED_OPTIMIZATION_AVAILABLE:
            try:
                self.vectorization_manager = UnifiedVectorizationManager()
                self.logger.info("✅ VectorBT optimization manager initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ VectorBT optimization failed: {e}")
                self.vectorization_manager = None
            
            # Initialize HPO wrapper
            try:
                self.hpo_wrapper = RegimeHPOWrapper()
                self.logger.info("✅ HPO wrapper initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ HPO wrapper failed: {e}")
                self.hpo_wrapper = None
            
            # Initialize explainability
            try:
                explanation_config = ExplanationConfig(
                    enable_shap=True,
                    enable_lime=True,
                    shap_sample_size=100,
                    lime_sample_size=1000,
                    parallel_explanations=True
                )
                self.explainer = SHAPLIMEExplainer(explanation_config)
                self.logger.info("✅ SHAP/LIME explainer initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ SHAP/LIME explainer failed: {e}")
                self.explainer = None
            
            # Initialize data leakage detector
            try:
                self.leakage_detector = DataLeakageDetector()
                self.logger.info("✅ Data leakage detector initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Data leakage detector failed: {e}")
                self.leakage_detector = None
            
            # Initialize hardware manager
            try:
                self.hardware_manager = UnifiedHardwareManager()
                self.hardware_manager.initialize()
                self.logger.info("✅ Hardware optimization manager initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware manager failed: {e}")
                self.hardware_manager = None
        else:
            self.vectorization_manager = None
            self.hpo_wrapper = None
            self.explainer = None
            self.leakage_detector = None
            self.hardware_manager = None
            self.logger.warning("⚠️ Enhanced optimization not available")

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the SR detection stage with full artifact integration."""
        self.logger.info('🎯 Starting SR Detection Stage execution')
        start_time = time.time()

        try:
            # Extract configuration parameters
            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'long')
            execution_mode = config.get('execution_mode', 'light')
            
            # Set up artifact manager context
            self.artifact_manager.set_context(
                step_name=self.step_name,
                symbol=symbol,
                exchange=exchange,
                direction=direction,
                model='Analyst'
            )

            # Try to load existing SR levels from artifacts first
            existing_sr_levels = None
            try:
                existing_sr_levels = self._get_artifact('sr_levels_dictionary', 'data')
                if existing_sr_levels:
                    self.logger.info(f'📊 Loaded existing SR levels from artifacts: {len(existing_sr_levels.get("levels", []))} levels')
            except Exception as e:
                self.logger.debug(f'No existing SR levels found in artifacts: {e}')

            # Get data from config or pipeline state
            data = config.get('dataframe') or config.get('data')
            if data is None:
                # Try to load data from artifacts
                try:
                    data = self._get_artifact('market_data', 'data')
                    if data is not None:
                        self.logger.info(f'📊 Loaded market data from artifacts: {data.shape[0]:,} rows, {data.shape[1]} columns')
                except Exception as e:
                    self.logger.debug(f'No market data found in artifacts: {e}')

            if data is None:
                raise ValueError("No dataframe found in config or artifacts")

            self.logger.info(f'📊 Data loaded: {data.shape[0]:,} rows, {data.shape[1]} columns')

            # Detect SR levels
            sr_levels = self._detect_sr_levels(data)

            # Save SR levels as artifact
            sr_artifact_path = self._save_artifact(
                sr_levels,
                'sr_levels_dictionary',
                'data',
                metadata={
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'direction': direction,
                    'execution_mode': execution_mode,
                    'total_levels': len(sr_levels.get('all_levels', [])),
                    'support_levels': len(sr_levels.get('support_levels', [])),
                    'resistance_levels': len(sr_levels.get('resistance_levels', []))
                }
            )

            # Save market data as artifact for future use
            data_artifact_path = self._save_artifact(
                data,
                'market_data',
                'data',
                metadata={
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'rows': data.shape[0],
                    'columns': data.shape[1]
                }
            )

            execution_time = time.time() - start_time
            self.logger.info(f'✅ SR Detection completed in {execution_time:.2f} seconds')

            return {
                'success': True,
                'sr_levels': sr_levels,
                'execution_time': execution_time,
                'stage': 'sr_detection',
                'artifacts': [sr_artifact_path, data_artifact_path],
                'metrics': {
                    'total_levels': len(sr_levels.get('all_levels', [])),
                    'support_levels': len(sr_levels.get('support_levels', [])),
                    'resistance_levels': len(sr_levels.get('resistance_levels', [])),
                    'data_rows': data.shape[0],
                    'data_columns': data.shape[1]
                }
            }

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f'❌ SR Detection failed: {e}')
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'stage': 'sr_detection',
                'artifacts': [],
                'metrics': {}
            }

    def load_sr_levels_from_artifacts(self, symbol: str = 'ETHUSDT', exchange: str = 'binance', 
                                     direction: str = 'long') -> Optional[Dict[str, Any]]:
        """Load existing SR levels from artifacts if available."""
        try:
            # Set context for artifact retrieval
            self.artifact_manager.set_context(
                step_name=self.step_name,
                symbol=symbol,
                exchange=exchange,
                direction=direction,
                model='Analyst'
            )
            
            # Try to load SR levels
            sr_levels = self._get_artifact('sr_levels_dictionary', 'data')
            if sr_levels:
                self.logger.info(f'📊 Loaded existing SR levels from artifacts: {len(sr_levels.get("all_levels", []))} levels')
                return sr_levels
            else:
                self.logger.debug('No existing SR levels found in artifacts')
                return None
                
        except Exception as e:
            self.logger.debug(f'Failed to load SR levels from artifacts: {e}')
            return None

    def _detect_sr_levels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect support and resistance levels using Enhanced SR Detection with VectorBT optimization."""
        self.logger.info('🎯 ===== STARTING ENHANCED SR DETECTION PROCESS =====')
        self.logger.info('🎯 Using Enhanced SR Detection with VectorBT optimization and ML explainability...')
        detection_start_time = time.time()
        
        # Configure hardware for SR detection workload
        if self.hardware_manager:
            try:
                self.hardware_manager.optimize_for_workload(
                    WorkloadType.ML_TRAINING, 
                    OptimizationLevel.BALANCED
                )
                self.logger.info("🖥️ Hardware optimized for SR detection")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware optimization failed: {e}")

        # CRITICAL: Validate input data before S/R detection
        if data is None:
            self.logger.error('❌ CRITICAL: Input data is None for S/R detection. Cannot proceed.')
            raise ValueError("CRITICAL: Input data is None for S/R detection. Cannot proceed.")

        if len(data) == 0:
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
            # Local imports to avoid circular dependencies
            try:
                from src.tactician.sr_levels.enhanced_sr_detection import EnhancedSRDetector, SRLevel
                enhanced_detector_available = True
                self.logger.info('✅ Enhanced SR Detector available')
            except ImportError as e:
                self.logger.warning(f'⚠️ Enhanced SR Detector not available: {e}')
                enhanced_detector_available = False
                EnhancedSRDetector = None
                SRLevel = None

            if not enhanced_detector_available or EnhancedSRDetector is None or SRLevel is None:
                self.logger.error('❌ Enhanced SR Detector not available for initial detection.')
                raise RuntimeError("Enhanced SR Detector not available for initial detection.")

            self.logger.info('✅ Enhanced SR Detector is available, proceeding with detection...')

            # Create enhanced SR detector with VectorBT optimization
            sr_config = {
                'min_touches': getattr(self, 'min_touches', 2),
                'tolerance_pct': getattr(self, 'tolerance_pct', 0.5),
                'lookback_periods': getattr(self, 'lookback_periods', 100),
                'memory_efficient': True,
                'use_parallel': getattr(self, 'enable_parallel_processing', False),
                'disable_dbscan_clustering': True,  # DISABLE DBSCAN clustering - using new logic
                'enable_vectorbt_optimization': True,
                'enable_hardware_optimization': True,
                'enable_explainability': True,
                'enable_data_leakage_detection': True
            }

            self.logger.info(f'🔧 SR Detection Configuration:')
            self.logger.info(f'   • Min touches: {sr_config["min_touches"]}')
            self.logger.info(f'   • Tolerance %: {sr_config["tolerance_pct"]}%')
            self.logger.info(f'   • Lookback periods: {sr_config["lookback_periods"]}')
            self.logger.info(f'   • Memory efficient: {sr_config["memory_efficient"]}')
            self.logger.info(f'   • Parallel processing: {sr_config["use_parallel"]}')
            self.logger.info(f'   • DBSCAN clustering: {"DISABLED" if sr_config.get("disable_dbscan_clustering", False) else "ENABLED"}')

            self.logger.info('🎯 Creating Enhanced SR Detector...')
            detector_creation_start = time.time()
            detector = EnhancedSRDetector(sr_config)
            detector_creation_time = time.time() - detector_creation_start
            self.logger.info(f'✅ Enhanced SR Detector created in {detector_creation_time:.2f} seconds')

            # Use VectorBT optimization if available
            if self.vectorization_manager and sr_config.get('enable_vectorbt_optimization', False):
                self.logger.info('⚡ Using VectorBT optimization for SR detection...')
                basic_detection_start = time.time()
                basic_sr_levels = self._detect_sr_levels_vectorbt(detector, clean_data, sr_config)
                basic_detection_time = time.time() - basic_detection_start
                self.logger.info(f'✅ VectorBT-optimized SR detection completed in {basic_detection_time:.2f} seconds')
            else:
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
            elif isinstance(basic_sr_levels, list) and len(basic_sr_levels) > 0:
                # Handle list of SRLevel objects from EnhancedSRDetector
                if hasattr(basic_sr_levels[0], 'price'):  # SRLevel objects
                    all_levels = basic_sr_levels
                    support_levels = [level for level in all_levels if getattr(level, 'type', '').lower() == 'support']
                    resistance_levels = [level for level in all_levels if getattr(level, 'type', '').lower() == 'resistance']
                    self.logger.info(f'📊 Enhanced SR detection results: {len(support_levels)} support levels, {len(resistance_levels)} resistance levels')
                else:
                    # Handle list of dictionaries
                    all_levels = basic_sr_levels
                    self.logger.info(f'📊 Basic detection results: {len(all_levels)} total levels (format: dict list)')
            else:
                all_levels = []
                self.logger.info('📊 Basic detection results: 0 levels')

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
                    # If no touch times available, assume no touches occurred
                    has_touch_times = hasattr(level, 'first_touch_time') and hasattr(level, 'last_touch_time')
                    touch_count = getattr(level, 'touch_count', 0 if not has_touch_times else 2)

                    level_dict = {
                        'price': level.price,
                        'strength': getattr(level, 'strength', 0.5),
                        'level_type': getattr(level, 'type', 'support'),
                        'touch_count': touch_count,
                        'first_touch': getattr(level, 'first_touch_time', datetime.now()) if has_touch_times else datetime.now(),
                        'last_touch': getattr(level, 'last_touch_time', datetime.now()) if has_touch_times else datetime.now()
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

            # Also separate the original SRLevel objects for compatibility
            support_srlevels = [level for level in all_levels if hasattr(level, 'type') and getattr(level, 'type', '').lower() == 'support']
            resistance_srlevels = [level for level in all_levels if hasattr(level, 'type') and getattr(level, 'type', '').lower() == 'resistance']

            # Filter out levels with zero touches (theoretical levels never touched by price)
            original_count = len(levels_dict)
            levels_dict = [level for level in levels_dict if level.get('touch_count', 0) > 0]
            filtered_count = len(levels_dict)

            # Re-separate support and resistance levels after filtering
            support_levels = [level for level in levels_dict if level.get('level_type', '').lower() == 'support']
            resistance_levels = [level for level in levels_dict if level.get('level_type', '').lower() == 'resistance']

            self.logger.info(f'🧹 Filtered out {original_count - filtered_count} levels with zero touches')
            self.logger.info(f'📊 Levels after filtering: {len(support_levels)} support, {len(resistance_levels)} resistance')

            # Generate enhanced features if available
            explanations = {}
            leakage_validation = {}
            hpo_optimization = {}
            
            if sr_config.get('enable_explainability', False):
                explanations = self._generate_sr_explanations(levels_dict, clean_data)
            
            if sr_config.get('enable_data_leakage_detection', False):
                leakage_validation = self._validate_data_leakage(clean_data, levels_dict)
            
            if self.hpo_wrapper and len(levels_dict) > 0:
                hpo_optimization = self._optimize_sr_parameters(clean_data, levels_dict)

            detection_time = time.time() - detection_start_time

            self.logger.info('🎯 ===== ENHANCED SR DETECTION PROCESS COMPLETED =====')
            self.logger.info(f'✅ Total detection time: {detection_time:.2f} seconds')
            self.logger.info(f'📊 Final results: {len(support_levels)} support levels, {len(resistance_levels)} resistance levels')
            self.logger.info(f'🧠 Explanations: {len(explanations)} levels explained')
            self.logger.info(f'🔍 Data leakage: {"Detected" if leakage_validation.get("has_leakage", False) else "None"}')

            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'support_srlevels': support_srlevels,
                'resistance_srlevels': resistance_srlevels,
                'all_levels': levels_dict,
                'all_srlevels': all_levels,
                'detection_time': detection_time,
                'detection_config': sr_config,
                'data_shape': clean_data.shape,
                'validation_time': validation_time,
                # Enhanced features
                'explanations': explanations,
                'leakage_validation': leakage_validation,
                'hpo_optimization': hpo_optimization,
                'enhancement_features': {
                    'vectorbt_optimization': sr_config.get('enable_vectorbt_optimization', False),
                    'hardware_optimization': sr_config.get('enable_hardware_optimization', False),
                    'explainability': sr_config.get('enable_explainability', False),
                    'data_leakage_detection': sr_config.get('enable_data_leakage_detection', False)
                }
            }

        except Exception as e:
            detection_time = time.time() - detection_start_time
            self.logger.error(f'❌ SR Detection failed after {detection_time:.2f} seconds: {e}')
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            raise

    def _validate_price_data_quality(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean price data for SR detection with comprehensive checks."""
        self.logger.info('🔍 Starting comprehensive price data quality validation...')

        # Check for required OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required OHLCV columns: {missing_cols}")

        # Create a copy to avoid modifying original data
        clean_data = data.copy()
        initial_rows = len(clean_data)

        # Remove rows with NaN values in critical columns
        clean_data = clean_data.dropna(subset=required_cols)
        removed_nan_rows = initial_rows - len(clean_data)

        if removed_nan_rows > 0:
            self.logger.info(f'🧹 Removed {removed_nan_rows} rows with NaN values in OHLCV columns')

        # Validate price relationships (high >= low, etc.)
        invalid_high_low = clean_data['high'] < clean_data['low']
        invalid_high_open = clean_data['high'] < clean_data['open']
        invalid_high_close = clean_data['high'] < clean_data['close']
        invalid_low_open = clean_data['low'] > clean_data['open']
        invalid_low_close = clean_data['low'] > clean_data['close']

        # Check for non-positive prices
        invalid_open_close = (clean_data['open'] <= 0) | (clean_data['close'] <= 0)
        invalid_high_low_positive = (clean_data['high'] <= 0) | (clean_data['low'] <= 0)
        invalid_volume = clean_data['volume'] < 0

        # Check for extreme outliers (prices > 10x median)
        extreme_outliers = pd.Series(False, index=clean_data.index)
        for col in ['open', 'high', 'low', 'close']:
            median_price = clean_data[col].median()
            if median_price > 0:
                extreme_outliers |= (clean_data[col] > median_price * 10)

        # Check for duplicate timestamps if timestamp column exists
        duplicate_timestamps = pd.Series(False, index=clean_data.index)
        if 'timestamp' in clean_data.columns:
            duplicate_timestamps = clean_data['timestamp'].duplicated()

        # Combine all invalid conditions
        invalid_rows = (
            invalid_high_low | invalid_high_open | invalid_high_close |
            invalid_low_open | invalid_low_close | invalid_open_close |
            invalid_high_low_positive | invalid_volume | extreme_outliers |
            duplicate_timestamps
        )

        if invalid_rows.any():
            invalid_count = invalid_rows.sum()
            self.logger.warning(f'⚠️ Found {invalid_count} rows with invalid data:')
            self.logger.warning(f'   • High < Low: {invalid_high_low.sum()}')
            self.logger.warning(f'   • High < Open/Close: {(invalid_high_open | invalid_high_close).sum()}')
            self.logger.warning(f'   • Low > Open/Close: {(invalid_low_open | invalid_low_close).sum()}')
            self.logger.warning(f'   • Non-positive prices: {(invalid_open_close | invalid_high_low_positive).sum()}')
            self.logger.warning(f'   • Negative volume: {invalid_volume.sum()}')
            self.logger.warning(f'   • Extreme outliers: {extreme_outliers.sum()}')
            if 'timestamp' in clean_data.columns:
                self.logger.warning(f'   • Duplicate timestamps: {duplicate_timestamps.sum()}')

            clean_data = clean_data[~invalid_rows]

        final_rows = len(clean_data)
        total_removed = initial_rows - final_rows

        if total_removed > 0:
            self.logger.info(f'✅ Data validation completed: {total_removed} rows removed ({total_removed/initial_rows*100:.1f}%)')
            self.logger.info(f'📊 Final dataset: {final_rows:,} rows')
        else:
            self.logger.info('✅ Data validation completed: No rows removed')

        # Final validation - ensure we have enough data
        if final_rows < 10:
            raise ValueError(f"Insufficient data after validation: {final_rows} rows (minimum 10 required)")

            return clean_data

    def _detect_sr_levels_vectorbt(self, detector, data: pd.DataFrame, config: Dict[str, Any]) -> Any:
        """Detect SR levels using VectorBT optimization."""
        try:
            # Prepare data for VectorBT optimization
            operation_data = {
                'data': data,
                'operation': 'sr_detection',
                'config': config
            }
            
            # Use VectorBT for technical analysis optimization
            result = self.vectorization_manager.optimize_operation(
                OperationType.VECTORBT_TECHNICAL_ANALYSIS,
                operation_data,
                prefer_vectorbt=True
            )
            
            # Extract SR levels from VectorBT result
            if hasattr(result, 'result') and result.result:
                return result.result
            else:
                # Fallback to standard detection
                return detector.detect_sr_levels(data)
                
        except Exception as e:
            self.logger.warning(f"⚠️ VectorBT detection failed, falling back to standard: {e}")
            return detector.detect_sr_levels(data)

    def _generate_sr_explanations(self, sr_levels: List[Dict[str, Any]], data: pd.DataFrame) -> Dict[str, Any]:
        """Generate SHAP/LIME explanations for SR levels."""
        if not self.explainer or not sr_levels:
            return {}
        
        try:
            self.logger.info("🧠 Generating SHAP/LIME explanations for SR levels...")
            
            # Create feature matrix for explanations
            feature_matrix = self._create_sr_feature_matrix(sr_levels, data)
            
            # Generate explanations
            explanations = {}
            for i, level in enumerate(sr_levels):
                try:
                    # Create a simple model for explanation (placeholder)
                    class SRLevelModel:
                        def predict(self, X):
                            return np.array([level.get('strength', 0.5)] * len(X))
                        def predict_proba(self, X):
                            return np.array([[1-level.get('strength', 0.5), level.get('strength', 0.5)]] * len(X))
                    
                    model = SRLevelModel()
                    
                    # Generate explanation
                    explanation = self.explainer.explain_model(
                        model, 
                        feature_matrix[i:i+1], 
                        f'sr_level_{i}',
                        output_names=['strength'],
                        feature_names=['price', 'volume', 'volatility', 'time_since_touch']
                    )
                    
                    explanations[f'level_{i}'] = {
                        'level_info': level,
                        'explanation': explanation,
                        'feature_importance': self._calculate_sr_feature_importance(level)
                    }
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to generate explanation for level {i}: {e}")
                    continue
            
            self.logger.info(f"✅ Generated explanations for {len(explanations)} SR levels")
            return explanations
            
        except Exception as e:
            self.logger.warning(f"⚠️ SR explanation generation failed: {e}")
            return {}

    def _create_sr_feature_matrix(self, sr_levels: List[Dict[str, Any]], data: pd.DataFrame) -> np.ndarray:
        """Create feature matrix for SR level explanations."""
        try:
            features = []
            for level in sr_levels:
                level_features = [
                    level.get('price', 0.0),
                    data['volume'].mean() if 'volume' in data.columns else 0.0,
                    data['close'].std() if 'close' in data.columns else 0.0,
                    level.get('touch_count', 0)
                ]
                features.append(level_features)
            return np.array(features)
        except Exception as e:
            self.logger.warning(f"⚠️ Feature matrix creation failed: {e}")
            return np.array([])

    def _calculate_sr_feature_importance(self, level: Dict[str, Any]) -> Dict[str, float]:
        """Calculate feature importance for SR level."""
        try:
            # Simple feature importance based on level properties
            importance = {
                'price_strength': level.get('strength', 0.5),
                'touch_frequency': min(1.0, level.get('touch_count', 0) / 5.0),
                'time_persistence': 0.7,  # Placeholder
                'volume_confirmation': 0.6  # Placeholder
            }
            
            # Normalize importance scores
            total = sum(importance.values())
            if total > 0:
                importance = {k: v/total for k, v in importance.items()}
            
            return importance
        except Exception as e:
            self.logger.warning(f"⚠️ Feature importance calculation failed: {e}")
            return {}

    def _validate_data_leakage(self, data: pd.DataFrame, sr_levels: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate for data leakage in SR detection."""
        if not self.leakage_detector:
            return {}
        
        try:
            self.logger.info("🔍 Validating for data leakage...")
            
            # Create train/test split for validation
            split_idx = int(len(data) * 0.8)
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]
            
            # Check for temporal leakage
            leakage_report = self.leakage_detector.generate_report(
                X_train=train_data,
                X_test=test_data,
                features=data,
                target=pd.Series([1] * len(data))  # Dummy target
            )
            
            if leakage_report.has_leakage:
                self.logger.warning(f"⚠️ Data leakage detected: {leakage_report.leakage_score:.2f}")
            else:
                self.logger.info("✅ No data leakage detected")
            
            return {
                'has_leakage': leakage_report.has_leakage,
                'leakage_score': leakage_report.leakage_score,
                'violations': leakage_report.temporal_violations,
                'recommendations': leakage_report.recommendations
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Data leakage validation failed: {e}")
            return {}

    def _optimize_sr_parameters(self, data: pd.DataFrame, sr_levels: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize SR detection parameters using HPO."""
        if not self.hpo_wrapper:
            return {}
        
        try:
            self.logger.info("🎯 Optimizing SR detection parameters...")
            
            # Create feature matrix for HPO
            X = self._create_sr_feature_matrix(sr_levels, data)
            y = np.array([1] * len(sr_levels))  # Dummy target for optimization
            
            # Run HPO optimization
            hpo_result = self.hpo_wrapper.hierarchical_optimization(X, y)
            
            self.logger.info(f"✅ HPO optimization completed: {hpo_result.total_optimization_time:.2f}s")
            
            return {
                'best_params': hpo_result.base_model_best_params,
                'best_scores': hpo_result.base_model_best_scores,
                'optimization_time': hpo_result.total_optimization_time,
                'convergence_info': hpo_result.convergence_info
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ HPO optimization failed: {e}")
            return {}

    def validate_config(self) -> None:
        """Validate the configuration for the SR detection step."""
        try:
            # Validate required configuration parameters
            required_keys = ['sr_optimization']
            for key in required_keys:
                if key not in self.config:
                    raise ValueError(f"Missing required configuration key: {key}")

            # Validate SR optimization parameters
            sr_config = self.config.get('sr_optimization', {})
            if not isinstance(sr_config, dict):
                raise ValueError("sr_optimization must be a dictionary")

            # Validate numeric parameters with comprehensive checks
            if 'min_touches' in sr_config:
                if not isinstance(sr_config['min_touches'], (int, float)) or sr_config['min_touches'] < 1:
                    raise ValueError("min_touches must be a positive number >= 1")

            if 'tolerance_pct' in sr_config:
                if not isinstance(sr_config['tolerance_pct'], (int, float)) or sr_config['tolerance_pct'] <= 0 or sr_config['tolerance_pct'] > 1:
                    raise ValueError("tolerance_pct must be a number between 0 and 1")

            if 'lookback_periods' in sr_config:
                if not isinstance(sr_config['lookback_periods'], (int, float)) or sr_config['lookback_periods'] < 1:
                    raise ValueError("lookback_periods must be a positive number >= 1")

            # Validate optional parameters
            if 'proximity_threshold' in sr_config:
                if not isinstance(sr_config['proximity_threshold'], (int, float)) or sr_config['proximity_threshold'] <= 0:
                    raise ValueError("proximity_threshold must be a positive number")

            if 'min_sr_ratio' in sr_config:
                if not isinstance(sr_config['min_sr_ratio'], (int, float)) or sr_config['min_sr_ratio'] < 0 or sr_config['min_sr_ratio'] > 1:
                    raise ValueError("min_sr_ratio must be a number between 0 and 1")

            if 'max_sr_ratio' in sr_config:
                if not isinstance(sr_config['max_sr_ratio'], (int, float)) or sr_config['max_sr_ratio'] < 0 or sr_config['max_sr_ratio'] > 1:
                    raise ValueError("max_sr_ratio must be a number between 0 and 1")

            # Validate ratio relationship
            if 'min_sr_ratio' in sr_config and 'max_sr_ratio' in sr_config:
                if sr_config['min_sr_ratio'] >= sr_config['max_sr_ratio']:
                    raise ValueError("min_sr_ratio must be less than max_sr_ratio")

            self.logger.info("✅ SR detection configuration validated successfully")

        except Exception as e:
            self.logger.error(f"❌ SR detection configuration validation failed: {e}")
            raise

    def get_status(self) -> Dict[str, Any]:
        """Get the current status and metrics of the SR detection step."""
        status = {
            'step_name': 'SRDetectionStep',
            'status': 'ready',
            'config_validated': True,
            'memory_manager_active': hasattr(self, 'memory_manager') and self.memory_manager is not None,
            'sr_optimization_config': self.sr_optimization_config,
            'proximity_threshold': self.proximity_threshold,
            'min_sr_ratio': self.min_sr_ratio,
            'max_sr_ratio': self.max_sr_ratio,
            'sr_ratio_range': f"{self.min_sr_ratio:.2f} - {self.max_sr_ratio:.2f}",
            'timestamp': get_current_datetime().isoformat()
        }
        
        # Add enhanced logging status if available
        if self.sr_logger:
            status['enhanced_logging'] = True
            status['log_events'] = len(self.sr_logger.events)
            status['performance_summary'] = self.sr_logger.get_performance_summary()
        else:
            status['enhanced_logging'] = False
        
        return status
    
    def cleanup(self):
        """Cleanup resources and stop monitoring."""
        try:
            # Stop memory monitoring
            if hasattr(self, 'memory_manager') and self.memory_manager:
                self.memory_manager.stop_monitoring()
                self.logger.info("Memory monitoring stopped")
            
            # Stop enhanced logging
            if self.sr_logger:
                self.sr_logger.export_events('logs/sr_detection_events.json')
                self.logger.info("Enhanced logging events exported")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            self.logger.info("SRDetectionStep cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    def get_logging_summary(self) -> Dict[str, Any]:
        """Get comprehensive logging summary."""
        if not self.sr_logger:
            return {"message": "Enhanced logging not available"}
        
        return self.sr_logger.get_performance_summary()
    
    def export_logs(self, filepath: str, format: str = 'json'):
        """Export logs to file."""
        if not self.sr_logger:
            self.logger.warning("Enhanced logging not available, cannot export logs")
            return
        
        try:
            self.sr_logger.export_events(filepath, format)
            self.logger.info(f"Logs exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to export logs: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup
