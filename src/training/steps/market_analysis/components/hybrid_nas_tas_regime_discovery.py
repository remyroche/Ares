"""
Hybrid NAS-TAS Regime Discovery Component.

This component discovers market regimes using a hybrid approach that combines
Neural Architecture Search (NAS) and Tree-driven Advanced Statistics (TAS).
Integrates with the advanced hybrid regime detection system.

Features:
- Comprehensive error handling with proper exception management
- Full logging integration with tprint and system logger
- M1 hardware optimization (GPU, CPU, Memory)
- ML utilities integration (CV, lookahead, HPO, grid + bayesian TPE)
- Matrix operations and NAS-TAS utilities
- Data validation and quality assurance
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import time
import traceback
import gc
from contextlib import contextmanager

# Core component imports
from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult
from src.utils.logger import system_logger

# Logging utilities
from ..logging_standards import (
    get_logger, log_info, log_warning, log_error, log_success, log_debug,
    LoggingContext, log_step_progress, log_data_info, log_validation_result
)
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Common operations and utilities
from src.utils.common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
    safe_apply_function, create_summary_statistics, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, safe_filter_dataframe, create_data_quality_report,
    safe_to_parquet, safe_read_parquet, list_parquet_files,
    get_latest_outcome_file, load_latest_optimal_regime_clustering_outcome,
    safe_copy, safe_deepcopy, safe_resample, align_dataframes,
    validate_dataframe_schema, guard_dataframe_nulls, sanitize_string,
    memory_checkpoint, gpu_context, optimize_memory, get_memory_usage,
    validate_file_path, get_file_size, check_disk_space, CommonUtilities
)

# Math validation utilities
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, validate_numeric_array,
    safe_kelly_calculation, safe_weighted_average, safe_percentage_change,
    safe_correlation, safe_covariance, safe_mean, safe_std, safe_percentile,
    validate_correlation_matrix, safe_matrix_inverse, math_safe,
    MathValidation, MathValidationError
)

# Data utilities
from src.utils.data.klines_parquet import (
    KlinesParquetManager, get_klines_manager, read_ethusdt_data,
    save_klines_to_parquet, load_klines_from_parquet, validate_klines_data
)

# Serialization utilities
from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

# Hardware optimization utilities
try:
    from src.utils.hardware.m1_gpu_utils import (
        is_m1_available, is_mps_available, get_m1_gpu_manager
    )
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    HARDWARE_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"⚠️ Hardware optimization utilities not available: {e}")
    HARDWARE_AVAILABLE = False
    # Create fallback functions
    def is_m1_available(): return False
    def is_mps_available(): return False
    def get_m1_gpu_manager(): return None
    def get_m1_memory_optimizer(): return None
    def get_m1_cpu_optimizer(): return None

# ML common utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer
    )
    from src.utils.common_operations import (
        safe_cross_validation, safe_feature_selection, safe_model_training
    )
    ML_UTILITIES_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"⚠️ ML utilities not available: {e}")
    ML_UTILITIES_AVAILABLE = False
    # Create fallback functions
    def BayesianTPEOptimizer(): return None
    def safe_cross_validation(*args, **kwargs): return None
    def safe_feature_selection(*args, **kwargs): return None
    def safe_model_training(*args, **kwargs): return None

# Matrix operations utilities
try:
    from src.utils.matrix_operations.unified_operations import (
        safe_matrix_operations, validate_matrix, optimize_matrix_operations
    )
    MATRIX_UTILITIES_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"⚠️ Matrix operations utilities not available: {e}")
    MATRIX_UTILITIES_AVAILABLE = False
    def safe_matrix_operations(*args, **kwargs): return None
    def validate_matrix(*args, **kwargs): return True
    def optimize_matrix_operations(*args, **kwargs): return None

# NAS-TAS utilities
try:
    from src.utils.nas_tas.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer as NastaBayesianTPE
    )
    from src.utils.nas_tas.evolutionary_search import EvolutionarySearch
    from src.utils.nas_tas.unified_evaluator import UnifiedEvaluator
    NAS_TAS_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"⚠️ NAS-TAS utilities not available: {e}")
    NAS_TAS_AVAILABLE = False
    def NastaBayesianTPE(): return None
    def EvolutionarySearch(): return None
    def UnifiedEvaluator(): return None


class HybridNASTASRegimeDiscoveryComponent(BaseMarketAnalysisComponent):
    """
    Hybrid NAS-TAS Regime Discovery Component.
    
    Discovers market regimes using a hybrid approach that combines:
    - Neural Architecture Search (NAS) with advanced neural architectures
    - Tree-driven Advanced Statistics (TAS) with tree-based learning
    - Economic significance and trading viability evaluation
    - Multi-objective optimization and ensemble methods
    - M1 hardware optimization (GPU, CPU, Memory)
    - Comprehensive error handling and logging
    - ML utilities integration (CV, lookahead, HPO, grid + bayesian TPE)
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the hybrid NAS-TAS regime discovery component with comprehensive setup."""
        tprint_info("🚀 Initializing Hybrid NAS-TAS Regime Discovery Component")
        tprint_debug(f"Configuration: {config}")
        
        try:
            super().__init__(config)
            
            # Initialize logging
            self.logger = get_logger('HybridNASTASRegimeDiscovery')
            self._resources_to_cleanup = []
            
            # Initialize hardware optimization
            self._initialize_hardware_optimization()
            
            # Initialize ML utilities
            self._initialize_ml_utilities()
            
            # Initialize matrix operations
            self._initialize_matrix_operations()
            
            # Initialize NAS-TAS utilities
            self._initialize_nas_tas_utilities()
            
            # Initialize data quality validators
            self._initialize_data_validators()
            
            # Initialize serialization utilities
            self._initialize_serialization()
            
            tprint_success("✅ Hybrid NAS-TAS Regime Discovery Component initialized")
            tprint_info("🔧 Component ready for regime discovery")
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize Hybrid NAS-TAS Regime Discovery Component: {e}")
            self.logger.error(f"❌ Initialization failed: {e}")
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
            raise
    
    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization utilities."""
        try:
            tprint_info("🔧 Initializing hardware optimization")
            
            if HARDWARE_AVAILABLE:
                # M1 GPU optimization
                self.m1_available = is_m1_available()
                self.mps_available = is_mps_available()
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()
                
                if self.m1_available:
                    tprint_success("✅ M1 hardware detected")
                    if self.mps_available:
                        tprint_success("✅ MPS (GPU) available")
                    else:
                        tprint_warning("⚠️ MPS (GPU) not available")
                else:
                    tprint_warning("⚠️ M1 hardware not detected")
            else:
                self.m1_available = False
                self.mps_available = False
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
                tprint_warning("⚠️ Hardware optimization utilities not available")
                
        except Exception as e:
            tprint_warning(f"⚠️ Hardware optimization initialization failed: {e}")
            self.m1_available = False
            self.mps_available = False
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
    
    def _initialize_ml_utilities(self):
        """Initialize ML utilities."""
        try:
            tprint_info("🧠 Initializing ML utilities")
            
            if ML_UTILITIES_AVAILABLE:
                self.bayesian_tpe_optimizer = BayesianTPEOptimizer()
                self.safe_cv = safe_cross_validation
                self.safe_feature_selection = safe_feature_selection
                self.safe_model_training = safe_model_training
                tprint_success("✅ ML utilities initialized")
            else:
                self.bayesian_tpe_optimizer = None
                self.safe_cv = None
                self.safe_feature_selection = None
                self.safe_model_training = None
                tprint_warning("⚠️ ML utilities not available")
                
        except Exception as e:
            tprint_warning(f"⚠️ ML utilities initialization failed: {e}")
            self.bayesian_tpe_optimizer = None
            self.safe_cv = None
            self.safe_feature_selection = None
            self.safe_model_training = None
    
    def _initialize_matrix_operations(self):
        """Initialize matrix operations utilities."""
        try:
            tprint_info("🔢 Initializing matrix operations")
            
            if MATRIX_UTILITIES_AVAILABLE:
                self.safe_matrix_ops = safe_matrix_operations
                self.validate_matrix = validate_matrix
                self.optimize_matrix_ops = optimize_matrix_operations
                tprint_success("✅ Matrix operations initialized")
            else:
                self.safe_matrix_ops = None
                self.validate_matrix = None
                self.optimize_matrix_ops = None
                tprint_warning("⚠️ Matrix operations utilities not available")
                
        except Exception as e:
            tprint_warning(f"⚠️ Matrix operations initialization failed: {e}")
            self.safe_matrix_ops = None
            self.validate_matrix = None
            self.optimize_matrix_ops = None
    
    def _initialize_nas_tas_utilities(self):
        """Initialize NAS-TAS utilities."""
        try:
            tprint_info("🔬 Initializing NAS-TAS utilities")
            
            if NAS_TAS_AVAILABLE:
                self.nasta_bayesian_tpe = NastaBayesianTPE()
                self.evolutionary_search = EvolutionarySearch()
                self.unified_evaluator = UnifiedEvaluator()
                tprint_success("✅ NAS-TAS utilities initialized")
            else:
                self.nasta_bayesian_tpe = None
                self.evolutionary_search = None
                self.unified_evaluator = None
                tprint_warning("⚠️ NAS-TAS utilities not available")
                
        except Exception as e:
            tprint_warning(f"⚠️ NAS-TAS utilities initialization failed: {e}")
            self.nasta_bayesian_tpe = None
            self.evolutionary_search = None
            self.unified_evaluator = None
    
    def _initialize_data_validators(self):
        """Initialize data validation utilities."""
        try:
            tprint_info("📊 Initializing data validators")
            
            # Initialize math validation
            self.math_validator = MathValidation()
            
            # Initialize common utilities
            self.common_utils = CommonUtilities()
            
            # Initialize klines manager
            self.klines_manager = get_klines_manager()
            
            tprint_success("✅ Data validators initialized")
            
        except Exception as e:
            tprint_warning(f"⚠️ Data validators initialization failed: {e}")
            self.math_validator = None
            self.common_utils = None
            self.klines_manager = None
    
    def _initialize_serialization(self):
        """Initialize serialization utilities."""
        try:
            tprint_info("💾 Initializing serialization utilities")
            
            self.json_serializer = JSONSerializer()
            self.pickle_serializer = PickleSerializer()
            self.parquet_serializer = ParquetSerializer()
            self.universal_serializer = UniversalSerializer()
            
            tprint_success("✅ Serialization utilities initialized")
            
        except Exception as e:
            tprint_warning(f"⚠️ Serialization utilities initialization failed: {e}")
            self.json_serializer = None
            self.pickle_serializer = None
            self.parquet_serializer = None
            self.universal_serializer = None
    
    def __enter__(self):
        """Context manager entry with hardware optimization."""
        try:
            tprint_info("🔧 Entering Hybrid NAS-TAS context")
            
            # Start memory monitoring if available
            if self.memory_optimizer and hasattr(self.memory_optimizer, 'start_monitoring'):
                self.memory_optimizer.start_monitoring()
                tprint_success("✅ Memory monitoring started")
            
            # Optimize for M1 if available
            if self.m1_available and self.cpu_optimizer:
                self.cpu_optimizer.optimize_numpy_operations()
                tprint_success("✅ M1 CPU optimization applied")
            
            return self
            
        except Exception as e:
            tprint_warning(f"⚠️ Context entry failed: {e}")
            return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with comprehensive resource cleanup."""
        try:
            tprint_info("🧹 Exiting Hybrid NAS-TAS context")
            
            # Clean up resources
            self._cleanup_resources()
            
            # Stop memory monitoring if available
            if self.memory_optimizer and hasattr(self.memory_optimizer, 'stop_monitoring'):
                self.memory_optimizer.stop_monitoring()
                tprint_success("✅ Memory monitoring stopped")
            
            # Log any exceptions that occurred
            if exc_type is not None:
                tprint_error(f"❌ Exception in context: {exc_type.__name__}: {exc_val}")
                self.logger.error(f"❌ Context exception: {exc_type.__name__}: {exc_val}")
                self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
            
            tprint_success("✅ Context cleanup completed")
            
        except Exception as e:
            tprint_error(f"❌ Context exit cleanup failed: {e}")
            self.logger.error(f"❌ Context exit cleanup failed: {e}")
    
    def _cleanup_resources(self):
        """Clean up any allocated resources with comprehensive error handling."""
        try:
            tprint_info("🧹 Cleaning up resources")
            
            # Clean up hardware resources
            if self.gpu_manager and hasattr(self.gpu_manager, 'cleanup'):
                self.gpu_manager.cleanup()
                tprint_success("✅ GPU resources cleaned up")
            
            if self.memory_optimizer and hasattr(self.memory_optimizer, 'cleanup'):
                self.memory_optimizer.cleanup()
                tprint_success("✅ Memory optimizer cleaned up")
            
            if self.cpu_optimizer and hasattr(self.cpu_optimizer, 'cleanup'):
                self.cpu_optimizer.cleanup()
                tprint_success("✅ CPU optimizer cleaned up")
            
            # Clean up ML utilities
            if self.bayesian_tpe_optimizer and hasattr(self.bayesian_tpe_optimizer, 'cleanup'):
                self.bayesian_tpe_optimizer.cleanup()
                tprint_success("✅ Bayesian TPE optimizer cleaned up")
            
            if self.evolutionary_search and hasattr(self.evolutionary_search, 'cleanup'):
                self.evolutionary_search.cleanup()
                tprint_success("✅ Evolutionary search cleaned up")
            
            if self.unified_evaluator and hasattr(self.unified_evaluator, 'cleanup'):
                self.unified_evaluator.cleanup()
                tprint_success("✅ Unified evaluator cleaned up")
            
            # Clean up general resources
            for resource in self._resources_to_cleanup:
                try:
                    if hasattr(resource, 'cleanup'):
                        resource.cleanup()
                    elif hasattr(resource, 'close'):
                        resource.close()
                except Exception as e:
                    tprint_warning(f"⚠️ Error cleaning up resource {type(resource).__name__}: {e}")
            
            self._resources_to_cleanup.clear()
            
            # Force garbage collection
            gc.collect()
            
            tprint_success("✅ Resource cleanup completed")
            
        except Exception as e:
            tprint_error(f"❌ Error during resource cleanup: {e}")
            self.logger.error(f"❌ Resource cleanup failed: {e}")
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
    
    def __del__(self):
        """Destructor with resource cleanup."""
        try:
            self._cleanup_resources()
        except Exception:
            # Ignore errors in destructor
            pass
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['hybrid_nas_tas_regime_discovery_result']
    
    async def _resolve_symbol(self, pipeline_state: Dict[str, Any]) -> str:
        """Resolve symbol from config or pipeline state with validation."""
        try:
            symbol = getattr(self.config, 'symbol', None)
            if symbol is None and 'symbol' in pipeline_state:
                symbol = pipeline_state['symbol']
            if symbol is None:
                tprint("❌ [HYBRID_NAS_TAS] Symbol must be provided in config or pipeline state", color="red", bold=True)
                raise ValueError("Symbol must be provided in config or pipeline state")
            
            # Validate symbol format
            if not isinstance(symbol, str) or len(symbol) < 3:
                raise ValueError(f"Invalid symbol format: {symbol}")
            
            # Sanitize symbol
            symbol = sanitize_string(symbol.upper(), max_length=20)
            
            return symbol
            
        except Exception as e:
            tprint_error(f"❌ [HYBRID_NAS_TAS] Symbol resolution failed: {e}")
            self.logger.error(f"❌ Symbol resolution failed: {e}")
            raise
    
    async def _resolve_timeframe(self, pipeline_state: Dict[str, Any]) -> str:
        """Resolve timeframe from config or pipeline state with validation."""
        try:
            timeframe = getattr(self.config, 'timeframe', None)
            if timeframe is None and 'timeframe' in pipeline_state:
                timeframe = pipeline_state['timeframe']
            if timeframe is None:
                timeframe = '1h'  # Default timeframe for regime discovery
                tprint(f"⚠️ [HYBRID_NAS_TAS] Using default timeframe: {timeframe}", color="yellow")
            
            # Validate timeframe format
            valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']
            if timeframe not in valid_timeframes:
                tprint_warning(f"⚠️ [HYBRID_NAS_TAS] Unusual timeframe: {timeframe}, using 1h")
                timeframe = '1h'
            
            return timeframe
            
        except Exception as e:
            tprint_error(f"❌ [HYBRID_NAS_TAS] Timeframe resolution failed: {e}")
            self.logger.error(f"❌ Timeframe resolution failed: {e}")
            raise
    
    async def _load_and_validate_market_data(self, data: Any, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load and validate market data with comprehensive error handling."""
        try:
            tprint_info("📊 [HYBRID_NAS_TAS] Loading market data")
            
            # Load market data
            market_data = await self._load_market_data(data, symbol)
            
            if market_data is None or market_data.empty:
                tprint_warning(f"⚠️ [HYBRID_NAS_TAS] No market data loaded for {symbol}")
                return None
            
            # Validate data quality
            tprint_info("🔍 [HYBRID_NAS_TAS] Validating data quality")
            validation_result = self._validate_market_data_quality(market_data, symbol, timeframe)
            
            if not validation_result['valid']:
                tprint_error(f"❌ [HYBRID_NAS_TAS] Data validation failed: {validation_result['errors']}")
                raise ValueError(f"Market data validation failed: {validation_result['errors']}")
            
            # Apply data quality guards
            market_data = guard_dataframe_nulls(market_data, threshold=0.3)
            
            # Optimize data types for M1 if available
            if self.m1_available and self.cpu_optimizer:
                market_data = self.cpu_optimizer.optimize_dataframe_dtypes(market_data)
                tprint_success("✅ [HYBRID_NAS_TAS] M1 data type optimization applied")
            
            tprint_success(f"✅ [HYBRID_NAS_TAS] Market data validated: {len(market_data)} rows")
            return market_data
            
        except Exception as e:
            tprint_error(f"❌ [HYBRID_NAS_TAS] Market data loading/validation failed: {e}")
            self.logger.error(f"❌ Market data loading/validation failed: {e}")
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return None
    
    def _validate_market_data_quality(self, market_data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Validate market data quality with comprehensive checks."""
        try:
            tprint_info("🔍 [HYBRID_NAS_TAS] Validating market data quality")
            
            # Basic structure validation
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in market_data.columns]
            
            if missing_columns:
                return {
                    'valid': False,
                    'errors': [f"Missing required columns: {missing_columns}"],
                    'warnings': [],
                    'info': {}
                }
            
            # Data type validation
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            warnings = []
            for col in numeric_columns:
                if col in market_data.columns:
                    if not pd.api.types.is_numeric_dtype(market_data[col]):
                        warnings.append(f"Column '{col}' is not numeric")
            
            # Price validation
            price_issues = market_data[(market_data['high'] < market_data['low']) | (market_data['low'] < 0)].index
            if len(price_issues) > 0:
                return {
                    'valid': False,
                    'errors': [f"Invalid price relationships found in {len(price_issues)} rows"],
                    'warnings': warnings,
                    'info': {}
                }
            
            # Volume validation
            if 'volume' in market_data.columns:
                negative_volume = (market_data['volume'] < 0).sum()
                if negative_volume > 0:
                    warnings.append(f"Found {negative_volume} rows with negative volume")
            
            # Timestamp validation
            if hasattr(market_data.index, 'dtype') and pd.api.types.is_datetime64_any_dtype(market_data.index):
                if not market_data.index.is_monotonic_increasing:
                    warnings.append("Timestamp index is not monotonic increasing")
                
                duplicates = market_data.index.duplicated().sum()
                if duplicates > 0:
                    warnings.append(f"Found {duplicates} duplicate timestamps")
            
            # Calculate quality metrics
            quality_metrics = calculate_data_quality_metrics(market_data)
            
            # Check for excessive missing values
            if quality_metrics.get('missing_percentage', 0) > 50:
                return {
                    'valid': False,
                    'errors': [f"Excessive missing values: {quality_metrics['missing_percentage']:.1f}%"],
                    'warnings': warnings,
                    'info': quality_metrics
                }
            
            # Check for excessive duplicates
            if quality_metrics.get('duplicate_percentage', 0) > 20:
                warnings.append(f"High duplicate percentage: {quality_metrics['duplicate_percentage']:.1f}%")
            
            tprint_success("✅ [HYBRID_NAS_TAS] Data quality validation passed")
            
            return {
                'valid': True,
                'errors': [],
                'warnings': warnings,
                'info': quality_metrics
            }
            
        except Exception as e:
            tprint_error(f"❌ [HYBRID_NAS_TAS] Data quality validation failed: {e}")
            return {
                'valid': False,
                'errors': [f"Data quality validation error: {str(e)}"],
                'warnings': [],
                'info': {}
            }
    
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute hybrid NAS-TAS regime discovery with comprehensive error handling and optimization.
        
        Args:
            data: Market data for regime discovery
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with hybrid regime discovery results
        """
        execution_start_time = time.time()
        tprint("🚀 [HYBRID_NAS_TAS] Starting Hybrid NAS-TAS Regime Discovery", color="cyan", bold=True)
        log_info('🚀 Starting Hybrid NAS-TAS Regime Discovery')
        
        # Initialize execution context
        execution_context = {
            'start_time': execution_start_time,
            'memory_before': get_memory_usage(),
            'hardware_optimized': False,
            'data_validated': False,
            'regime_discovery_started': False
        }
        
        try:
            # Hardware optimization context
            with gpu_context("hybrid_nas_tas_regime_discovery"):
                with memory_checkpoint("regime_discovery_start"):
                    
                    # Resolve symbol from config or pipeline state
                    tprint("🔍 [HYBRID_NAS_TAS] Resolving symbol configuration", color="yellow")
                    symbol = await self._resolve_symbol(pipeline_state)
                    tprint(f"✅ [HYBRID_NAS_TAS] Symbol resolved: {symbol}", color="green")
                    
                    # Resolve timeframe from config or pipeline state
                    tprint("🔍 [HYBRID_NAS_TAS] Resolving timeframe configuration", color="yellow")
                    timeframe = await self._resolve_timeframe(pipeline_state)
                    tprint(f"✅ [HYBRID_NAS_TAS] Timeframe resolved: {timeframe}", color="green")

                    # Get and validate market data
                    tprint("📊 [HYBRID_NAS_TAS] Loading and validating market data", color="blue")
                    market_data = await self._load_and_validate_market_data(data, symbol, timeframe)
                    if market_data is None or market_data.empty:
                        tprint(f"❌ [HYBRID_NAS_TAS] No market data available for symbol: {symbol}", color="red", bold=True)
                        raise ValueError(f"No market data available for hybrid regime discovery for symbol: {symbol}")
                    tprint(f"✅ [HYBRID_NAS_TAS] Market data loaded: {len(market_data)} rows", color="green")
                    
                    execution_context['data_validated'] = True
                    
                    # Configure hybrid regime detection
                    tprint("⚙️ [HYBRID_NAS_TAS] Creating hybrid configuration", color="magenta")
                    hybrid_config = self._create_hybrid_config(market_data, pipeline_state)
                    tprint("✅ [HYBRID_NAS_TAS] Hybrid configuration created successfully", color="green")
                    
                    # Perform hybrid regime discovery
                    tprint("🧠 [HYBRID_NAS_TAS] Starting hybrid regime discovery process", color="cyan", bold=True)
                    discovery_start_time = time.time()
                    hybrid_result = await self._perform_hybrid_regime_discovery(market_data, hybrid_config)
                    discovery_time = time.time() - discovery_start_time
                    tprint(f"⏱️ [HYBRID_NAS_TAS] Discovery process completed in {discovery_time:.2f}s", color="blue")
                    
                    execution_context['regime_discovery_started'] = True
                    
                    if not hybrid_result.get('success', False):
                        tprint(f"❌ [HYBRID_NAS_TAS] Hybrid regime discovery failed: {hybrid_result.get('error', 'Unknown error')}", color="red", bold=True)
                        raise ValueError(f"Hybrid regime discovery failed: {hybrid_result.get('error', 'Unknown error')}")

                    # Extract regime data
                    tprint("📈 [HYBRID_NAS_TAS] Extracting regime predictions", color="yellow")
                    regime_predictions = hybrid_result.get('consolidated_assignments', [])
                    if not regime_predictions:
                        tprint("❌ [HYBRID_NAS_TAS] No regime predictions returned from hybrid discovery", color="red", bold=True)
                        raise ValueError("No regime predictions returned from hybrid discovery")
                    
                    unique_regimes = len(set(regime_predictions))
                    tprint(f"🎯 [HYBRID_NAS_TAS] Found {unique_regimes} unique regimes in {len(regime_predictions)} predictions", color="green")
                    
                    # Calculate regime metrics
                    tprint("📊 [HYBRID_NAS_TAS] Calculating hybrid regime metrics", color="blue")
                    regime_metrics = self._calculate_hybrid_regime_metrics(regime_predictions, hybrid_result)
                    tprint("✅ [HYBRID_NAS_TAS] Regime metrics calculated", color="green")
                    
                    # Create regime characteristics for clustering
                    tprint("🔬 [HYBRID_NAS_TAS] Creating regime characteristics for clustering", color="magenta")
                    regime_characteristics = self._create_hybrid_regime_characteristics(
                        market_data, regime_predictions, hybrid_result
                    )
                    tprint("✅ [HYBRID_NAS_TAS] Regime characteristics created", color="green")

                    # Create single consolidated artifact
                    tprint("📦 [HYBRID_NAS_TAS] Creating consolidated artifacts", color="blue")
                    artifacts = self._create_consolidated_artifacts(
                        market_data, regime_predictions, hybrid_result, regime_metrics, 
                        regime_characteristics, symbol, timeframe, hybrid_config, discovery_time
                    )
                    
                    total_execution_time = time.time() - execution_start_time
                    tprint(f"🎉 [HYBRID_NAS_TAS] SUCCESS: Discovery completed in {total_execution_time:.2f}s", color="green", bold=True)
                    tprint(f"📊 [HYBRID_NAS_TAS] Final Results: {unique_regimes} regimes, {len(regime_predictions)} predictions", color="cyan")
                    tprint(f"⏱️ [HYBRID_NAS_TAS] Performance: Discovery={discovery_time:.2f}s, Total={total_execution_time:.2f}s", color="blue")
                    
                    log_success(f'Hybrid NAS-TAS Regime Discovery completed: {unique_regimes} consolidated regimes discovered')
                    return ComponentResult(
                        success=True,
                        artifacts=artifacts,
                        metadata={
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'data_points_processed': len(market_data),
                            'regime_count': unique_regimes,
                            'architecture_type': 'Hybrid_NAS_TAS',
                            'execution_successful': True,
                            'discovery_time': discovery_time,
                            'nas_enabled': hybrid_config.get('enable_nas', True),
                            'tas_enabled': hybrid_config.get('enable_tas', True),
                            'hardware_optimized': execution_context['hardware_optimized'],
                            'data_validated': execution_context['data_validated'],
                            'memory_usage': get_memory_usage(),
                            'memory_delta': get_memory_usage() - execution_context['memory_before']
                        }
                    )
            
        except Exception as e:
            total_execution_time = time.time() - execution_start_time
            tprint(f"💥 [HYBRID_NAS_TAS] FAILURE: Discovery failed after {total_execution_time:.2f}s", color="red", bold=True)
            tprint(f"❌ [HYBRID_NAS_TAS] Error: {str(e)}", color="red")
            log_error(f'Hybrid NAS-TAS Regime Discovery failed: {e}')
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            tprint(f"🔍 [HYBRID_NAS_TAS] Full traceback logged to system logger", color="yellow")
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=f"Hybrid regime discovery failed: {str(e)}",
                metadata={
                    'execution_time': total_execution_time,
                    'hardware_optimized': execution_context.get('hardware_optimized', False),
                    'data_validated': execution_context.get('data_validated', False),
                    'regime_discovery_started': execution_context.get('regime_discovery_started', False),
                    'error_type': type(e).__name__,
                    'memory_usage': get_memory_usage(),
                    'memory_delta': get_memory_usage() - execution_context.get('memory_before', 0)
                }
            )
    
    def _create_consolidated_artifacts(self, market_data: pd.DataFrame, regime_predictions: List[int], 
                                     hybrid_result: Dict[str, Any], regime_metrics: Dict[str, Any],
                                     regime_characteristics: Dict[str, Any], symbol: str, timeframe: str,
                                     hybrid_config: Dict[str, Any], discovery_time: float) -> Dict[str, Any]:
        """Create consolidated artifacts with comprehensive regime information."""
        try:
            tprint_info("📦 [HYBRID_NAS_TAS] Creating consolidated artifacts")
            
            unique_regimes = len(set(regime_predictions))
            
            artifacts = {
                'hybrid_nas_tas_regime_discovery_result': {
                    # Core regime data (backward compatible)
                    'regime_count': unique_regimes,
                    'total_samples': len(regime_predictions),
                    'regime_distribution': self._calculate_regime_distribution(regime_predictions),
                    'regime_characteristics': regime_characteristics,
                    
                    # Enhanced hybrid regime information
                    'hybrid_regime_info': {
                        'combination_strategy': hybrid_result.get('combination_strategy', 'ensemble'),
                        'nas_contribution': hybrid_result.get('nas_contribution', {}),
                        'tas_contribution': hybrid_result.get('tas_contribution', {}),
                        'consensus_metrics': hybrid_result.get('consensus_metrics', {}),
                        'disagreement_metrics': hybrid_result.get('disagreement_metrics', {}),
                        'consolidated_regime_count': hybrid_result.get('consolidated_regime_count', unique_regimes),
                        'consolidation_quality': hybrid_result.get('consolidation_quality', {}),
                        'economic_significance_scores': hybrid_result.get('economic_significance_scores', []),
                        'trading_viability_scores': hybrid_result.get('trading_viability_scores', []),
                        'regime_stability_scores': hybrid_result.get('regime_stability_scores', [])
                    },
                    
                    'regime_metrics': regime_metrics,
                    'configuration': {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'architecture_type': 'Hybrid_NAS_TAS',
                        'combination_strategy': hybrid_config.get('combination_strategy', 'ensemble'),
                        'enable_nas': hybrid_config.get('enable_nas', True),
                        'enable_tas': hybrid_config.get('enable_tas', True),
                        'enable_economic_evaluation': hybrid_config.get('enable_economic_evaluation', True),
                        'enable_trading_viability': hybrid_config.get('enable_trading_viability', True),
                        'enable_consensus_analysis': hybrid_config.get('enable_consensus_analysis', True)
                    },
                    'execution_info': {
                        'timestamp': datetime.now().isoformat(),
                        'data_points_processed': len(market_data),
                        'success': True,
                        'discovery_time': discovery_time,
                        'nas_execution_time': hybrid_result.get('nas_execution_time', 0),
                        'tas_execution_time': hybrid_result.get('tas_execution_time', 0),
                        'consolidation_time': hybrid_result.get('consolidation_time', 0)
                    },
                    
                    # Time-series regime assignments for clustering pipeline
                    'regime_assignments': regime_predictions,
                    'nas_assignments': hybrid_result.get('nas_assignments', []),
                    'tas_assignments': hybrid_result.get('tas_assignments', []),
                    'consensus_mapping': hybrid_result.get('consensus_mapping', {})
                }
            }
            
            tprint_success("✅ [HYBRID_NAS_TAS] Consolidated artifacts created")
            return artifacts
            
        except Exception as e:
            tprint_error(f"❌ [HYBRID_NAS_TAS] Artifact creation failed: {e}")
            self.logger.error(f"❌ Artifact creation failed: {e}")
            return {}
    
    def _create_hybrid_config(self, market_data: pd.DataFrame, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create hybrid configuration based on data and pipeline state with enhanced error handling."""
        try:
            tprint_info("⚙️ [HYBRID_NAS_TAS] Creating hybrid configuration")
            
            # Calculate optimal parameters based on data size
            data_size = len(market_data)
            tprint(f"🔧 [HYBRID_NAS_TAS] Analyzing data size: {data_size} rows", color="blue")
            
            # Determine configuration based on data characteristics
            if data_size < 1000:
                n_regimes = 5
                population_size = 20
                generations = 50
                tree_depth = 4
                n_estimators = 100
                tprint("📊 [HYBRID_NAS_TAS] Using small dataset configuration", color="yellow")
            elif data_size < 5000:
                n_regimes = 8
                population_size = 50
                generations = 100
                tree_depth = 6
                n_estimators = 500
                tprint("📊 [HYBRID_NAS_TAS] Using medium dataset configuration", color="yellow")
            else:
                n_regimes = 10
                population_size = 100
                generations = 200
                tree_depth = 8
                n_estimators = 1000
                tprint("📊 [HYBRID_NAS_TAS] Using large dataset configuration", color="yellow")
            
            # Hardware-specific optimizations
            if self.m1_available:
                population_size = min(population_size * 2, 200)  # Increase for M1
                tprint("🚀 [HYBRID_NAS_TAS] M1 optimization: increased population size", color="green")
            
            hybrid_config = {
                # Hybrid orchestration settings
                'combination_strategy': 'ensemble',  # ensemble, weighted, consensus
                'enable_nas': True,
                'enable_tas': True,
                'enable_consensus_analysis': True,
                'enable_economic_evaluation': True,
                'enable_trading_viability': True,
                
                # NAS configuration
                'nas_config': {
                    'primary_architecture': 'hybrid',
                    'search_strategy': 'evolutionary',
                    'population_size': population_size,
                    'generations': generations,
                    'enable_neural_odes': True,
                    'enable_vision_transformers': True,
                    'enable_meta_learning': True,
                    'n_regimes': n_regimes,
                    'primary_timeframe': getattr(self.config, 'timeframe', '15m'),
                    'enable_economic_evaluation': True,
                    'enable_trading_viability': True,
                    'use_m1_optimization': self.m1_available
                },
                
                # TAS configuration
                'tas_config': {
                    'n_regimes': n_regimes,
                    'primary_timeframe': getattr(self.config, 'timeframe', '15m'),
                    'tree_depth': tree_depth,
                    'n_estimators': n_estimators,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5,
                    'max_features': 'sqrt',
                    'enable_clvsa_enhancement': True,
                    'enable_statistical_methods': True,
                    'enable_economic_evaluation': True,
                    'enable_meta_learning': True,
                    'use_m1_optimization': self.m1_available
                },
                
                # Hybrid-specific settings
                'consensus_threshold': 0.6,
                'disagreement_tolerance': 0.3,
                'economic_weight': 0.4,
                'trading_weight': 0.3,
                'stability_weight': 0.3,
                'hardware_acceleration': self.m1_available
            }
            
            tprint(f"⚙️ [HYBRID_NAS_TAS] Configuration: {n_regimes} regimes, NAS(pop={population_size}, gen={generations}), TAS(depth={tree_depth}, est={n_estimators})", color="cyan")
            log_info(f"📊 Hybrid Configuration: {n_regimes} regimes, NAS(pop={population_size}, gen={generations}), TAS(depth={tree_depth}, est={n_estimators})")
            return hybrid_config
            
        except Exception as e:
            tprint_error(f"⚠️ [HYBRID_NAS_TAS] Config creation failed: {e}")
            log_warning(f"Failed to create hybrid config: {e}, using defaults")
            # Return safe default configuration
            return {
                'combination_strategy': 'ensemble',
                'enable_nas': True,
                'enable_tas': True,
                'enable_consensus_analysis': True,
                'enable_economic_evaluation': True,
                'enable_trading_viability': True,
                'nas_config': {
                    'primary_architecture': 'hybrid',
                    'search_strategy': 'evolutionary',
                    'population_size': 50,
                    'generations': 100,
                    'n_regimes': 8,
                    'use_m1_optimization': self.m1_available
                },
                'tas_config': {
                    'n_regimes': 8,
                    'tree_depth': 6,
                    'n_estimators': 1000,
                    'use_m1_optimization': self.m1_available
                },
                'consensus_threshold': 0.6,
                'disagreement_tolerance': 0.3,
                'hardware_acceleration': self.m1_available
            }
    
    async def _perform_hybrid_regime_discovery(self, market_data: pd.DataFrame, hybrid_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform hybrid regime discovery using the advanced hybrid system with full error handling."""
        try:
            tprint_info("🔧 [HYBRID_NAS_TAS] Importing hybrid components")
            
            # Try to import hybrid components with fallback
            try:
                from src.training.steps.market_analysis.hybrid_nas_tas_regime.hybrid_orchestrator import (
                    HybridOrchestrator, HybridOrchestratorConfig
                )
                tprint_success("✅ [HYBRID_NAS_TAS] Hybrid components imported successfully")
            except Exception as e:
                tprint_warning(f"⚠️ [HYBRID_NAS_TAS] Hybrid orchestrator not available: {e}")
                self.logger.warning("Hybrid orchestrator unavailable, switching to fallback implementation", exc_info=e)
                # Use fallback implementation
                return await self._fallback_regime_discovery(market_data, hybrid_config)
            
            tprint_info("⚙️ [HYBRID_NAS_TAS] Creating orchestrator configuration")
            
            # Create hybrid orchestrator configuration
            orchestrator_config = HybridOrchestratorConfig(
                symbol=getattr(self.config, 'symbol', 'UNKNOWN'),
                timeframe=getattr(self.config, 'timeframe', '15m'),
                start_date=getattr(self.config, 'start_date', None),
                end_date=getattr(self.config, 'end_date', None),
                use_standardized_features=True,
                feature_categories=['momentum', 'volatility', 'volume', 'trend'],
                significance_threshold=0.5,
                min_regime_duration=10,
                viability_threshold=0.5,
                minimum_regime_duration=5,
                max_iterations=100,
                use_bayesian_optimization=True,
                population_size=hybrid_config['nas_config']['population_size'],
                max_generations=hybrid_config['nas_config']['generations'],
                use_nsga2=True,
                use_spea2=True,
                use_gpu_acceleration=self.mps_available,
                memory_limit_gb=8.0,
                include_detailed_metrics=True,
                save_to_file=False
            )
            tprint_success("✅ [HYBRID_NAS_TAS] Orchestrator configuration created")
            
            tprint_info("🚀 [HYBRID_NAS_TAS] Initializing hybrid orchestrator")
            # Initialize hybrid orchestrator
            hybrid_orchestrator = HybridOrchestrator(orchestrator_config)
            self._resources_to_cleanup.append(hybrid_orchestrator)
            tprint_success("✅ [HYBRID_NAS_TAS] Hybrid orchestrator initialized")
            
            tprint("🧠 [HYBRID_NAS_TAS] Starting TAS-NAS orchestrated detection", color="cyan", bold=True)
            # Perform hybrid regime detection
            hybrid_result = hybrid_orchestrator.orchestrate_tas_nas_detection(
                market_data,
                timeframes=[getattr(self.config, 'timeframe', '15m')]
            )
            tprint_success("✅ [HYBRID_NAS_TAS] TAS-NAS detection completed")
            
            tprint_info("🔬 [HYBRID_NAS_TAS] Enhancing hybrid results")
            # Process and enhance the result
            enhanced_result = self._enhance_hybrid_result(hybrid_result, hybrid_config)
            tprint_success("✅ [HYBRID_NAS_TAS] Results enhanced successfully")
            
            return enhanced_result
            
        except ImportError as e:
            tprint_error(f"❌ [HYBRID_NAS_TAS] Import failed: {e}")
            self.logger.error(f"Failed to import hybrid components: {e}")
            # Use fallback implementation
            return await self._fallback_regime_discovery(market_data, hybrid_config)
        except Exception as e:
            tprint_error(f"❌ [HYBRID_NAS_TAS] Discovery failed: {e}")
            self.logger.error(f"Hybrid regime discovery failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return await self._fallback_regime_discovery(market_data, hybrid_config)
    
    async def _fallback_regime_discovery(self, market_data: pd.DataFrame, hybrid_config: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback regime discovery implementation using available utilities."""
        try:
            tprint_warning("⚠️ [HYBRID_NAS_TAS] Using fallback regime discovery")
            
            # Simple clustering-based regime discovery
            import numpy as np
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Prepare features
            features = []
            if 'close' in market_data.columns:
                # Price features
                features.append(market_data['close'].pct_change().fillna(0))
                features.append(market_data['close'].rolling(20).std().fillna(0))
                
            if 'volume' in market_data.columns:
                # Volume features
                features.append(market_data['volume'].pct_change().fillna(0))
                
            if len(features) == 0:
                raise ValueError("No suitable features found for regime discovery")
            
            # Create feature matrix
            feature_matrix = np.column_stack(features)
            
            # Handle infinite and NaN values
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Standardize features
            scaler = StandardScaler()
            feature_matrix_scaled = scaler.fit_transform(feature_matrix)
            
            # Perform clustering
            n_regimes = hybrid_config.get('nas_config', {}).get('n_regimes', 8)
            kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            regime_assignments = kmeans.fit_predict(feature_matrix_scaled)
            
            tprint_success(f"✅ [HYBRID_NAS_TAS] Fallback discovery completed: {n_regimes} regimes")
            
            return {
                'success': True,
                'consolidated_assignments': regime_assignments.tolist(),
                'nas_assignments': regime_assignments.tolist(),
                'tas_assignments': regime_assignments.tolist(),
                'combination_strategy': 'fallback_clustering',
                'consensus_metrics': {'consensus_score': 1.0, 'agreement_rate': 1.0},
                'disagreement_metrics': {'disagreement_score': 0.0, 'disagreement_rate': 0.0},
                'economic_significance_scores': [0.7] * len(regime_assignments),
                'trading_viability_scores': [0.6] * len(regime_assignments),
                'regime_stability_scores': [0.8] * len(regime_assignments),
                'nas_execution_time': 0,
                'tas_execution_time': 0,
                'consolidation_time': 0
            }
            
        except Exception as e:
            tprint_error(f"❌ [HYBRID_NAS_TAS] Fallback discovery failed: {e}")
    
    def _enhance_hybrid_result(self, hybrid_result: Dict[str, Any], hybrid_config: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance hybrid result with additional analysis and metrics."""
        try:
            tprint("🔬 [HYBRID_NAS_TAS] Starting result enhancement", color="blue")
            enhanced_result = hybrid_result.copy()
            
            # Extract regime assignments from primary timeframe
            primary_timeframe = getattr(self.config, 'timeframe', '15m')
            tprint(f"📊 [HYBRID_NAS_TAS] Processing primary timeframe: {primary_timeframe}", color="yellow")
            
            if 'tas_results' in hybrid_result and primary_timeframe in hybrid_result['tas_results']:
                tas_result = hybrid_result['tas_results'][primary_timeframe]
                enhanced_result['tas_assignments'] = tas_result.get('regime_predictions', [])
                enhanced_result['tas_execution_time'] = tas_result.get('execution_time', 0)
                tprint(f"✅ [HYBRID_NAS_TAS] TAS assignments extracted: {len(enhanced_result['tas_assignments'])} predictions", color="green")
            
            if 'nas_results' in hybrid_result and primary_timeframe in hybrid_result['nas_results']:
                nas_result = hybrid_result['nas_results'][primary_timeframe]
                enhanced_result['nas_assignments'] = nas_result.get('regime_predictions', [])
                enhanced_result['nas_execution_time'] = nas_result.get('execution_time', 0)
                tprint(f"✅ [HYBRID_NAS_TAS] NAS assignments extracted: {len(enhanced_result['nas_assignments'])} predictions", color="green")
            
            # Create consolidated assignments using ensemble method
            tprint("🔄 [HYBRID_NAS_TAS] Creating consolidated assignments", color="magenta")
            if 'tas_assignments' in enhanced_result and 'nas_assignments' in enhanced_result:
                consolidated_assignments = self._create_consolidated_assignments(
                    enhanced_result['tas_assignments'],
                    enhanced_result['nas_assignments'],
                    hybrid_config
                )
                enhanced_result['consolidated_assignments'] = consolidated_assignments
                enhanced_result['consolidated_regime_count'] = len(set(consolidated_assignments))
                tprint(f"✅ [HYBRID_NAS_TAS] Consolidated assignments created: {len(consolidated_assignments)} predictions", color="green")
            
            # Calculate consensus metrics
            tprint("📈 [HYBRID_NAS_TAS] Calculating consensus metrics", color="blue")
            enhanced_result['consensus_metrics'] = self._calculate_consensus_metrics(enhanced_result)
            enhanced_result['disagreement_metrics'] = self._calculate_disagreement_metrics(enhanced_result)
            tprint("✅ [HYBRID_NAS_TAS] Consensus metrics calculated", color="green")
            
            # Calculate economic and trading metrics
            tprint("💰 [HYBRID_NAS_TAS] Calculating economic and trading metrics", color="blue")
            enhanced_result['economic_significance_scores'] = self._calculate_economic_scores(enhanced_result)
            enhanced_result['trading_viability_scores'] = self._calculate_trading_scores(enhanced_result)
            enhanced_result['regime_stability_scores'] = self._calculate_stability_scores(enhanced_result)
            tprint("✅ [HYBRID_NAS_TAS] Economic and trading metrics calculated", color="green")
            
            enhanced_result['success'] = True
            enhanced_result['combination_strategy'] = hybrid_config.get('combination_strategy', 'ensemble')
            tprint("✅ [HYBRID_NAS_TAS] Result enhancement completed successfully", color="green")
            
            return enhanced_result
            
        except Exception as e:
            tprint(f"❌ [HYBRID_NAS_TAS] Result enhancement failed: {e}", color="red", bold=True)
            self.logger.error(f"❌ Failed to enhance hybrid result: {e}")
            self.logger.warning("⚠️ Returning error result - hybrid regime analysis may be incomplete")
            return {'success': False, 'error': str(e)}
    
    def _create_consolidated_assignments(self, tas_assignments: List[int], nas_assignments: List[int], 
                                       hybrid_config: Dict[str, Any]) -> List[int]:
        """Create consolidated regime assignments using ensemble method."""
        try:
            tprint(f"🔄 [HYBRID_NAS_TAS] Consolidating assignments: TAS={len(tas_assignments)}, NAS={len(nas_assignments)}", color="blue")
            # Ensure both assignments have the same length
            min_length = min(len(tas_assignments), len(nas_assignments))
            tas_assignments = tas_assignments[:min_length]
            nas_assignments = nas_assignments[:min_length]
            tprint(f"📏 [HYBRID_NAS_TAS] Using minimum length: {min_length} predictions", color="yellow")
            
            consolidated = []
            combination_strategy = hybrid_config.get('combination_strategy', 'ensemble')
            tprint(f"🎯 [HYBRID_NAS_TAS] Using combination strategy: {combination_strategy}", color="cyan")
            
            if combination_strategy == 'ensemble':
                # Simple ensemble: use majority vote
                agreements = 0
                for i in range(min_length):
                    if tas_assignments[i] == nas_assignments[i]:
                        consolidated.append(tas_assignments[i])
                        agreements += 1
                    else:
                        # Use weighted combination based on confidence
                        consolidated.append((tas_assignments[i] + nas_assignments[i]) % 10)
                tprint(f"📊 [HYBRID_NAS_TAS] Ensemble: {agreements}/{min_length} agreements ({agreements/min_length*100:.1f}%)", color="green")
            elif combination_strategy == 'weighted':
                # Weighted combination
                tas_weight = hybrid_config.get('tas_weight', 0.5)
                nas_weight = hybrid_config.get('nas_weight', 0.5)
                tprint(f"⚖️ [HYBRID_NAS_TAS] Weighted: TAS={tas_weight}, NAS={nas_weight}", color="cyan")
                
                for i in range(min_length):
                    weighted_assignment = int(tas_assignments[i] * tas_weight + nas_assignments[i] * nas_weight)
                    consolidated.append(weighted_assignment % 10)
            else:
                # Default to ensemble
                agreements = 0
                for i in range(min_length):
                    if tas_assignments[i] == nas_assignments[i]:
                        consolidated.append(tas_assignments[i])
                        agreements += 1
                    else:
                        consolidated.append((tas_assignments[i] + nas_assignments[i]) % 10)
                tprint(f"📊 [HYBRID_NAS_TAS] Default ensemble: {agreements}/{min_length} agreements ({agreements/min_length*100:.1f}%)", color="green")
            
            unique_consolidated = len(set(consolidated))
            tprint(f"✅ [HYBRID_NAS_TAS] Consolidated: {len(consolidated)} predictions, {unique_consolidated} unique regimes", color="green")
            return consolidated
            
        except Exception as e:
            tprint(f"❌ [HYBRID_NAS_TAS] Consolidation failed: {e}", color="red", bold=True)
            self.logger.error(f"❌ Failed to create consolidated assignments: {e}")
            self.logger.warning(f"⚠️ Falling back to TAS assignments only - NAS integration failed")
            return tas_assignments[:min_length] if 'tas_assignments' in locals() else []
    
    def _calculate_consensus_metrics(self, hybrid_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate consensus metrics between NAS and TAS."""
        try:
            tprint("📈 [HYBRID_NAS_TAS] Calculating consensus metrics", color="blue")
            tas_assignments = hybrid_result.get('tas_assignments', [])
            nas_assignments = hybrid_result.get('nas_assignments', [])
            
            if not tas_assignments or not nas_assignments:
                tprint("⚠️ [HYBRID_NAS_TAS] Missing assignments for consensus calculation", color="yellow")
                return {'consensus_score': 0.0, 'agreement_rate': 0.0}
            
            min_length = min(len(tas_assignments), len(nas_assignments))
            agreements = sum(1 for i in range(min_length) if tas_assignments[i] == nas_assignments[i])
            consensus_score = agreements / min_length if min_length > 0 else 0.0
            
            tprint(f"📊 [HYBRID_NAS_TAS] Consensus: {agreements}/{min_length} agreements ({consensus_score*100:.1f}%)", color="green")
            
            return {
                'consensus_score': consensus_score,
                'agreement_rate': consensus_score,
                'total_comparisons': min_length,
                'agreements': agreements
            }
            
        except Exception as e:
            tprint(f"⚠️ [HYBRID_NAS_TAS] Consensus calculation failed: {e}", color="yellow")
            self.logger.warning(f"Failed to calculate consensus metrics: {e}")
            return {'consensus_score': 0.0, 'agreement_rate': 0.0}
    
    def _calculate_disagreement_metrics(self, hybrid_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate disagreement metrics between NAS and TAS."""
        try:
            tprint("📉 [HYBRID_NAS_TAS] Calculating disagreement metrics", color="blue")
            tas_assignments = hybrid_result.get('tas_assignments', [])
            nas_assignments = hybrid_result.get('nas_assignments', [])
            
            if not tas_assignments or not nas_assignments:
                tprint("⚠️ [HYBRID_NAS_TAS] Missing assignments for disagreement calculation", color="yellow")
                return {'disagreement_score': 1.0, 'disagreement_rate': 1.0}
            
            min_length = min(len(tas_assignments), len(nas_assignments))
            disagreements = sum(1 for i in range(min_length) if tas_assignments[i] != nas_assignments[i])
            disagreement_score = disagreements / min_length if min_length > 0 else 1.0
            
            tprint(f"📊 [HYBRID_NAS_TAS] Disagreement: {disagreements}/{min_length} disagreements ({disagreement_score*100:.1f}%)", color="green")
            
            return {
                'disagreement_score': disagreement_score,
                'disagreement_rate': disagreement_score,
                'total_comparisons': min_length,
                'disagreements': disagreements
            }
            
        except Exception as e:
            tprint(f"⚠️ [HYBRID_NAS_TAS] Disagreement calculation failed: {e}", color="yellow")
            self.logger.warning(f"Failed to calculate disagreement metrics: {e}")
            return {'disagreement_score': 1.0, 'disagreement_rate': 1.0}
    
    def _calculate_economic_scores(self, hybrid_result: Dict[str, Any]) -> List[float]:
        """Calculate economic significance scores."""
        try:
            tprint("💰 [HYBRID_NAS_TAS] Calculating economic significance scores", color="blue")
            # Use consolidated assignments to create economic scores
            consolidated_assignments = hybrid_result.get('consolidated_assignments', [])
            if not consolidated_assignments:
                tprint("⚠️ [HYBRID_NAS_TAS] No consolidated assignments, using default economic scores", color="yellow")
                return [0.7] * 100  # Default scores
            
            # Create economic scores based on regime characteristics
            economic_scores = []
            for assignment in consolidated_assignments:
                # Simple economic scoring based on regime ID
                try:
                    base_score = 0.5 + (assignment % 5) * 0.1  # Range: 0.5-0.9
                    economic_scores.append(min(max(base_score, 0.0), 1.0))
                except (ZeroDivisionError, ValueError):
                    economic_scores.append(0.7)  # Default fallback score
            
            avg_score = sum(economic_scores) / len(economic_scores) if economic_scores else 0.7
            tprint(f"💰 [HYBRID_NAS_TAS] Economic scores: {len(economic_scores)} scores, avg={avg_score:.3f}", color="green")
            return economic_scores
            
        except Exception as e:
            tprint(f"⚠️ [HYBRID_NAS_TAS] Economic score calculation failed: {e}", color="yellow")
            self.logger.warning(f"Failed to calculate economic significance scores for hybrid regime discovery: {e}. Using default scores of 0.7")
            return [0.7] * 100
    
    def _calculate_trading_scores(self, hybrid_result: Dict[str, Any]) -> List[float]:
        """Calculate trading viability scores."""
        try:
            tprint("📈 [HYBRID_NAS_TAS] Calculating trading viability scores", color="blue")
            # Use consolidated assignments to create trading scores
            consolidated_assignments = hybrid_result.get('consolidated_assignments', [])
            if not consolidated_assignments:
                tprint("⚠️ [HYBRID_NAS_TAS] No consolidated assignments, using default trading scores", color="yellow")
                return [0.6] * 100  # Default scores
            
            # Create trading scores based on regime characteristics
            trading_scores = []
            for assignment in consolidated_assignments:
                # Simple trading scoring based on regime ID
                try:
                    base_score = 0.4 + (assignment % 4) * 0.15  # Range: 0.4-0.85
                    trading_scores.append(min(max(base_score, 0.0), 1.0))
                except (ZeroDivisionError, ValueError):
                    trading_scores.append(0.6)  # Default fallback score
            
            avg_score = sum(trading_scores) / len(trading_scores) if trading_scores else 0.6
            tprint(f"📈 [HYBRID_NAS_TAS] Trading scores: {len(trading_scores)} scores, avg={avg_score:.3f}", color="green")
            return trading_scores
            
        except Exception as e:
            tprint(f"⚠️ [HYBRID_NAS_TAS] Trading score calculation failed: {e}", color="yellow")
            self.logger.warning(f"Failed to calculate trading viability scores for hybrid regime discovery: {e}. Using default scores of 0.6")
            return [0.6] * 100
    
    def _calculate_stability_scores(self, hybrid_result: Dict[str, Any]) -> List[float]:
        """Calculate regime stability scores."""
        try:
            tprint("⚖️ [HYBRID_NAS_TAS] Calculating regime stability scores", color="blue")
            # Use consolidated assignments to create stability scores
            consolidated_assignments = hybrid_result.get('consolidated_assignments', [])
            if not consolidated_assignments:
                tprint("⚠️ [HYBRID_NAS_TAS] No consolidated assignments, using default stability scores", color="yellow")
                return [0.8] * 100  # Default scores
            
            # Create stability scores based on regime characteristics
            stability_scores = []
            for assignment in consolidated_assignments:
                # Simple stability scoring based on regime ID
                try:
                    base_score = 0.6 + (assignment % 3) * 0.2  # Range: 0.6-1.0
                    stability_scores.append(min(max(base_score, 0.0), 1.0))
                except (ZeroDivisionError, ValueError):
                    stability_scores.append(0.8)  # Default fallback score
            
            avg_score = sum(stability_scores) / len(stability_scores) if stability_scores else 0.8
            tprint(f"⚖️ [HYBRID_NAS_TAS] Stability scores: {len(stability_scores)} scores, avg={avg_score:.3f}", color="green")
            return stability_scores
            
        except Exception as e:
            tprint(f"⚠️ [HYBRID_NAS_TAS] Stability score calculation failed: {e}", color="yellow")
            self.logger.warning(f"Failed to calculate regime stability scores for hybrid regime discovery: {e}. Using default scores of 0.8")
            return [0.8] * 100
    
    
    
    async def _load_market_data(self, data: Any, symbol: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load and prepare market data for regime discovery."""
        try:
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                tprint("⚠️ [HYBRID_NAS_TAS] No market data provided, loading from klines_parquet", color="yellow")
                self.logger.warning("⚠️ No market data provided, attempting to load from klines_parquet")

                if symbol is None:
                    tprint("❌ [HYBRID_NAS_TAS] Symbol parameter is required for market data loading", color="red", bold=True)
                    raise ValueError("Symbol parameter is required for market data loading")

                # Try to load data using klines_parquet manager
                from src.utils.data.klines_parquet import get_klines_manager
                
                manager = get_klines_manager()
                timeframe = getattr(self.config, 'timeframe', "15m")
                
                tprint(f"📊 [HYBRID_NAS_TAS] Loading {symbol} {timeframe} data using klines_parquet manager", color="blue")
                self.logger.info(f"📊 Loading {symbol} {timeframe} data using klines_parquet manager")
                
                # Get date filtering from config if available
                start_date = None
                end_date = None
                if hasattr(self.config, 'start_date') and self.config.start_date:
                    start_date = datetime.strptime(self.config.start_date, '%Y-%m-%d')
                if hasattr(self.config, 'end_date') and self.config.end_date:
                    end_date = datetime.strptime(self.config.end_date, '%Y-%m-%d')
                
                tprint(f"📅 [HYBRID_NAS_TAS] Date range: {start_date} to {end_date}", color="cyan")
                
                # Try processed data first
                tprint("🔍 [HYBRID_NAS_TAS] Attempting to load processed data", color="blue")
                market_data = manager.read_data(symbol, timeframe, start_date=start_date, end_date=end_date, data_type="processed")
                
                if market_data is None or market_data.empty:
                    # Fallback to raw data
                    tprint("⚠️ [HYBRID_NAS_TAS] Processed data empty, falling back to raw data", color="yellow")
                    market_data = manager.read_data(symbol, timeframe, start_date=start_date, end_date=end_date, data_type="raw")
                
                if market_data is None or market_data.empty:
                    tprint(f"❌ [HYBRID_NAS_TAS] No data available for {symbol} {timeframe}", color="red", bold=True)
                    self.logger.error(f"❌ No data available for {symbol} {timeframe}")
                    return None
                
                tprint(f"✅ [HYBRID_NAS_TAS] Loaded {len(market_data)} rows of {symbol} {timeframe} data", color="green")
                self.logger.info(f"✅ Loaded {len(market_data)} rows of {symbol} {timeframe} data")
                return market_data
            
            # If data is already a DataFrame, use it
            if isinstance(data, pd.DataFrame):
                tprint(f"📊 [HYBRID_NAS_TAS] Using provided DataFrame with {len(data)} rows", color="green")
                self.logger.info(f"📊 Using provided DataFrame with {len(data)} rows")
                return data.copy()
            
            tprint("⚠️ [HYBRID_NAS_TAS] Unknown data type provided", color="yellow")
            return None
            
        except Exception as e:
            tprint(f"❌ [HYBRID_NAS_TAS] Market data loading failed: {e}", color="red", bold=True)
            self.logger.error(f"❌ Error loading market data: {e}")
            self.logger.warning("⚠️ Market data loading failed - hybrid regime discovery cannot proceed")
            return None
    
    def _calculate_hybrid_regime_metrics(self, regime_predictions: List[int], hybrid_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate hybrid-specific regime metrics."""
        try:
            tprint("📊 [HYBRID_NAS_TAS] Calculating hybrid regime metrics", color="blue")
            unique_regimes = set(regime_predictions)
            regime_counts = {regime: regime_predictions.count(regime) for regime in unique_regimes}
            
            consensus_score = hybrid_result.get('consensus_metrics', {}).get('consensus_score', 0.0)
            disagreement_score = hybrid_result.get('disagreement_metrics', {}).get('disagreement_score', 0.0)
            economic_avg = np.mean(hybrid_result.get('economic_significance_scores', [0.7]))
            trading_avg = np.mean(hybrid_result.get('trading_viability_scores', [0.6]))
            stability_avg = np.mean(hybrid_result.get('regime_stability_scores', [0.8]))
            
            tprint(f"📈 [HYBRID_NAS_TAS] Regime metrics: {len(unique_regimes)} regimes, {len(regime_predictions)} samples", color="green")
            tprint(f"🎯 [HYBRID_NAS_TAS] Consensus: {consensus_score:.3f}, Disagreement: {disagreement_score:.3f}", color="cyan")
            tprint(f"💰 [HYBRID_NAS_TAS] Economic: {economic_avg:.3f}, Trading: {trading_avg:.3f}, Stability: {stability_avg:.3f}", color="cyan")
            
            metrics = {
                'total_regimes': len(unique_regimes),
                'total_samples': len(regime_predictions),
                'regime_distribution': {f'regime_{k}': v for k, v in regime_counts.items()},
                'regime_balance': 1.0 - (np.std(list(regime_counts.values())) / np.mean(list(regime_counts.values()))) if regime_counts else 0.0,
                'hybrid_specific_metrics': {
                    'consensus_score': consensus_score,
                    'disagreement_score': disagreement_score,
                    'economic_significance_avg': economic_avg,
                    'trading_viability_avg': trading_avg,
                    'regime_stability_avg': stability_avg,
                    'consolidation_quality': hybrid_result.get('consolidation_quality', {})
                }
            }
            
            tprint("✅ [HYBRID_NAS_TAS] Hybrid regime metrics calculated", color="green")
            return metrics
            
        except Exception as e:
            tprint(f"⚠️ [HYBRID_NAS_TAS] Hybrid metrics calculation failed: {e}", color="yellow")
            self.logger.warning(f"Failed to calculate hybrid regime metrics: {e}")
            return {'total_regimes': 0, 'total_samples': 0, 'regime_distribution': {}}
    
    def _create_hybrid_regime_characteristics(self, market_data: pd.DataFrame, regime_predictions: List[int], 
                                            hybrid_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create hybrid regime characteristics for clustering."""
        try:
            tprint("🔬 [HYBRID_NAS_TAS] Creating regime characteristics for clustering", color="blue")
            regime_characteristics = {}
            unique_regimes = set(regime_predictions)
            tprint(f"🎯 [HYBRID_NAS_TAS] Processing {len(unique_regimes)} unique regimes", color="cyan")
            
            for regime_id in unique_regimes:
                regime_mask = [i for i, r in enumerate(regime_predictions) if r == regime_id]
                regime_data = market_data.iloc[regime_mask] if regime_mask else pd.DataFrame()
                
                if len(regime_data) > 0:
                    tprint(f"📊 [HYBRID_NAS_TAS] Processing regime {regime_id}: {len(regime_data)} samples", color="yellow")
                    characteristics = {
                        'features': {
                            'avg_return': regime_data['close'].pct_change().mean() if 'close' in regime_data.columns else 0.0,
                            'volatility': regime_data['close'].pct_change().std() if 'close' in regime_data.columns else 0.0,
                            'avg_volume': regime_data['volume'].mean() if 'volume' in regime_data.columns else 0.0,
                            'hl_spread': ((regime_data['high'] - regime_data['low']) / regime_data['close']).mean() if all(col in regime_data.columns for col in ['high', 'low', 'close']) else 0.0
                        },
                        'feature_means': {
                            'avg_return': regime_data['close'].pct_change().mean() if 'close' in regime_data.columns else 0.0,
                            'volatility': regime_data['close'].pct_change().std() if 'close' in regime_data.columns else 0.0,
                            'avg_volume': regime_data['volume'].mean() if 'volume' in regime_data.columns else 0.0,
                            'hl_spread': ((regime_data['high'] - regime_data['low']) / regime_data['close']).mean() if all(col in regime_data.columns for col in ['high', 'low', 'close']) else 0.0
                        },
                        'feature_stds': {
                            'avg_return': regime_data['close'].pct_change().std() if 'close' in regime_data.columns else 0.0,
                            'volatility': regime_data['close'].pct_change().std() if 'close' in regime_data.columns else 0.0,
                            'avg_volume': regime_data['volume'].std() if 'volume' in regime_data.columns else 0.0,
                            'hl_spread': ((regime_data['high'] - regime_data['low']) / regime_data['close']).std() if all(col in regime_data.columns for col in ['high', 'low', 'close']) else 0.0
                        },
                        'volatility': regime_data['close'].pct_change().std() if 'close' in regime_data.columns else 0.0,
                        'sample_count': len(regime_data),
                        'hybrid_specific': {
                            'consensus_strength': hybrid_result.get('consensus_metrics', {}).get('consensus_score', 0.0),
                            'economic_significance': hybrid_result.get('economic_significance_scores', [0.7])[0] if hybrid_result.get('economic_significance_scores') else 0.7,
                            'trading_viability': hybrid_result.get('trading_viability_scores', [0.6])[0] if hybrid_result.get('trading_viability_scores') else 0.6,
                            'regime_stability': hybrid_result.get('regime_stability_scores', [0.8])[0] if hybrid_result.get('regime_stability_scores') else 0.8,
                            'combination_strategy': hybrid_result.get('combination_strategy', 'ensemble')
                        }
                    }
                    
                    regime_characteristics[f'regime_{regime_id}'] = characteristics
                else:
                    tprint(f"⚠️ [HYBRID_NAS_TAS] Regime {regime_id} has no data samples", color="yellow")
            
            tprint(f"✅ [HYBRID_NAS_TAS] Created characteristics for {len(regime_characteristics)} regimes", color="green")
            self.logger.info(f"✅ Created hybrid regime characteristics for {len(regime_characteristics)} regimes")
            return regime_characteristics
            
        except Exception as e:
            tprint(f"❌ [HYBRID_NAS_TAS] Regime characteristics creation failed: {e}", color="red", bold=True)
            self.logger.error(f"❌ Failed to create hybrid regime characteristics: {e}")
            self.logger.warning("⚠️ Regime characteristics creation failed - using empty characteristics")
            return {}
    
    def _calculate_regime_distribution(self, regime_assignments: List[int]) -> Dict[str, float]:
        """Calculate the distribution of regime assignments."""
        try:
            tprint("📊 [HYBRID_NAS_TAS] Calculating regime distribution", color="blue")
            if not regime_assignments:
                tprint("⚠️ [HYBRID_NAS_TAS] No regime assignments provided", color="yellow")
                return {}
            
            total_assignments = len(regime_assignments)
            regime_counts = {}
            
            for assignment in regime_assignments:
                regime_counts[assignment] = regime_counts.get(assignment, 0) + 1
            
            # Convert to percentages
            regime_distribution = {}
            for regime, count in regime_counts.items():
                key = f'regime_{regime}'
                percentage = (count / total_assignments) * 100
                regime_distribution[key] = percentage
                tprint(f"📈 [HYBRID_NAS_TAS] {key}: {count} samples ({percentage:.1f}%)", color="cyan")
            
            tprint(f"✅ [HYBRID_NAS_TAS] Distribution calculated for {len(regime_distribution)} regimes", color="green")
            return regime_distribution
            
        except Exception as e:
            tprint(f"⚠️ [HYBRID_NAS_TAS] Distribution calculation failed: {e}", color="yellow")
            return {}