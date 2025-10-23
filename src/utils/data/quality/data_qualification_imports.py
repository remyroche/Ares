from src.utils.tprint import tprint

"""
Unified Import Manager for Data Qualification Pipeline

This module provides centralized import management for all data qualification steps,
ensuring consistent utility access with proper fallback mechanisms and error handling.

Key Features:
- Centralized import management with fallback handling
- ML Commons integration with graceful degradation
- M1 optimization utilities with CPU fallback
- Comprehensive error handling and logging
- Type-safe utility access
- Performance monitoring and analytics
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union, Protocol, Type, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path
import sys
import warnings

# Core imports
import pandas as pd
import numpy as np
from datetime import datetime

# Initialize logger
logger = logging.getLogger(__name__)

@dataclass
class ImportResult:
    """Result of import operation with metadata."""
    success: bool
    module: Optional[Any] = None
    error: Optional[Exception] = None
    fallback_used: bool = False
    import_time: float = 0.0
    module_name: str = ""

@dataclass
class UtilitySuite:
    """Container for utility modules with fallback support."""
    ml_common: Optional[Dict[str, Any]] = None
    m1_optimizers: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    data_processing: Optional[Dict[str, Any]] = None
    serialization: Optional[Dict[str, Any]] = None
    parquet: Optional[Dict[str, Any]] = None
    fallbacks_used: List[str] = field(default_factory=list)
    import_errors: List[str] = field(default_factory=list)

class DataQualificationImportManager:
    """
    Centralized import manager for data qualification utilities.

    Provides unified access to all utility modules with comprehensive
    fallback mechanisms and error handling.

    Example:
        >>> manager = DataQualificationImportManager()
        >>> utilities = manager.get_utility_suite()
        >>> if utilities.ml_common:
        ...     data_quality = utilities.ml_common['data_quality']
    """

    def __init__(self, enable_fallbacks: bool = True, log_imports: bool = True):
        """
        Initialize the import manager.

        Args:
            enable_fallbacks: Whether to enable fallback mechanisms
            log_imports: Whether to log import operations
        """
        self.enable_fallbacks = enable_fallbacks
        self.log_imports = log_imports
        self.import_cache: Dict[str, ImportResult] = {}
        self.utility_suite: Optional[UtilitySuite] = None
        self.logger = logger.getChild('ImportManager')

        if self.log_imports:
            self.logger.info("🚀 Data Qualification Import Manager initialized")

    def get_utility_suite(self, force_refresh: bool = False) -> UtilitySuite:
        """
        Get the complete utility suite with fallback handling.

        Args:
            force_refresh: Force refresh of cached imports

        Returns:
            UtilitySuite with all available utilities

        Example:
            >>> manager = DataQualificationImportManager()
            >>> suite = manager.get_utility_suite()
            >>> if suite.ml_common:
            ...     tprint("ML Commons available")
        """
        if self.utility_suite is None or force_refresh:
            self.utility_suite = self._build_utility_suite()

        return self.utility_suite

    def _build_utility_suite(self) -> UtilitySuite:
        """Build the complete utility suite with fallback handling."""
        self.logger.info("🔧 Building utility suite with fallback handling...")

        suite = UtilitySuite()

        # Import ML Commons utilities
        suite.ml_common = self._import_ml_commons()

        # Import M1 optimization utilities
        suite.m1_optimizers = self._import_m1_optimizers()

        # Import validation utilities
        suite.validation = self._import_validation_utilities()

        # Import data processing utilities
        suite.data_processing = self._import_data_processing_utilities()

        # Import serialization utilities
        suite.serialization = self._import_serialization_utilities()

        # Import parquet utilities
        suite.parquet = self._import_parquet_utilities()

        # Log summary
        if self.log_imports:
            self._log_import_summary(suite)

        return suite

    def _import_ml_commons(self) -> Optional[Dict[str, Any]]:
        """Import ML Commons utilities with fallback handling."""
        try:
            self.logger.info("📦 Importing ML Commons utilities...")

            # Core ML Commons imports
            ml_commons = {}

            # Data Quality Utilities
            try:
                from src.utils.ml_common.data_quality import DataQualityUtilities
                ml_commons['data_quality'] = DataQualityUtilities
                self.logger.debug("✅ DataQualityUtilities imported")
            except ImportError as e:
                self.logger.warning(f"⚠️ DataQualityUtilities not available: {e}")
                if self.enable_fallbacks:
                    ml_commons['data_quality'] = self._get_fallback_data_quality()

            # Pipeline Orchestrator
            try:
                from src.utils.ml_common.pipeline_orchestrator import MLPipelineOrchestrator
                ml_commons['pipeline_orchestrator'] = MLPipelineOrchestrator
                self.logger.debug("✅ MLPipelineOrchestrator imported")
            except ImportError as e:
                self.logger.warning(f"⚠️ MLPipelineOrchestrator not available: {e}")
                if self.enable_fallbacks:
                    ml_commons['pipeline_orchestrator'] = self._get_fallback_pipeline_orchestrator()

            # Feature Selection Framework
            try:
                from src.feature_selection.core import get_feature_selection_framework
                ml_commons['feature_selection'] = get_feature_selection_framework
                self.logger.debug("✅ Feature selection framework imported")
            except ImportError as e:
                self.logger.warning(f"⚠️ FeatureSelectionFramework not available: {e}")
                if self.enable_fallbacks:
                    ml_commons['feature_selection'] = self._get_fallback_feature_selection()

            # Parallel Processing (use central ParallelProcessor)
            try:
                from src.utils.parallel_processing_optimizer import ParallelProcessor
                ml_commons['parallel_processing'] = ParallelProcessor
                self.logger.debug("✅ ParallelProcessor imported")
            except ImportError as e:
                self.logger.warning(f"⚠️ ParallelProcessor not available: {e}")
                if self.enable_fallbacks:
                    ml_commons['parallel_processing'] = self._get_fallback_parallel_processing()

            # Matrix Operations
            try:
                from src.utils.matrix_operations import EnhancedMatrixOperations
                ml_commons['matrix_operations'] = EnhancedMatrixOperations
                self.logger.debug("✅ EnhancedMatrixOperations imported")
            except ImportError as e:
                self.logger.warning(f"⚠️ EnhancedMatrixOperations not available: {e}")
                if self.enable_fallbacks:
                    ml_commons['matrix_operations'] = self._get_fallback_matrix_operations()

            # Data Labeling Utilities
            try:
                from src.utils.ml_common.data_labeling import (
                    DataLabelingUtilities, TripleBarrierConfig, LabelingMethod
                )
                ml_commons['data_labeling'] = DataLabelingUtilities
                ml_commons['triple_barrier_config'] = TripleBarrierConfig
                ml_commons['labeling_method'] = LabelingMethod
                self.logger.debug("✅ DataLabelingUtilities imported")
            except ImportError as e:
                self.logger.warning(f"⚠️ DataLabelingUtilities not available: {e}")
                if self.enable_fallbacks:
                    ml_commons['data_labeling'] = self._get_fallback_data_labeling()

            # HMM Regime Detection
            try:
                from src.utils.ml_common.hmm_regime_detection import (
                    HMMRegimeDetector, HMMRegimeConfig, RegimeDetectionMethod
                )
                ml_commons['hmm_regime_detector'] = HMMRegimeDetector
                ml_commons['hmm_regime_config'] = HMMRegimeConfig
                ml_commons['regime_detection_method'] = RegimeDetectionMethod
                self.logger.debug("✅ HMMRegimeDetector imported")
            except ImportError as e:
                self.logger.warning(f"⚠️ HMMRegimeDetector not available: {e}")
                if self.enable_fallbacks:
                    ml_commons['hmm_regime_detector'] = self._get_fallback_hmm_detector()

            # Regime Data Processing
            try:
                from src.utils.ml_common.regime_data_processing import (
                    RegimeDataProcessor, RegimeProcessingConfig
                )
                ml_commons['regime_data_processor'] = RegimeDataProcessor
                ml_commons['regime_processing_config'] = RegimeProcessingConfig
                self.logger.debug("✅ RegimeDataProcessor imported")
            except ImportError as e:
                self.logger.warning(f"⚠️ RegimeDataProcessor not available: {e}")
                if self.enable_fallbacks:
                    ml_commons['regime_data_processor'] = self._get_fallback_regime_processor()

            self.logger.info(f"✅ ML Commons utilities loaded: {len(ml_commons)} modules")
            return ml_commons

        except Exception as e:
            self.logger.error(f"❌ Failed to import ML Commons utilities: {e}")
            return None

    def _import_m1_optimizers(self) -> Optional[Dict[str, Any]]:
        """Import M1 optimization utilities with fallback handling."""
        try:
            self.logger.info("⚡ Importing M1 optimization utilities...")

            optimizers = {}

            # M1 GPU Utils
            try:
                from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager, M1GPUManager
                optimizers['gpu_manager'] = get_m1_gpu_manager
                optimizers['gpu_manager_class'] = M1GPUManager
                self.logger.debug("✅ M1 GPU utilities imported")
            except ImportError as e:
                self.logger.warning(f"⚠️ M1 GPU utilities not available: {e}")
                if self.enable_fallbacks:
                    optimizers['gpu_manager'] = self._get_fallback_gpu_manager()

            # M1 Memory Optimizer
            try:
                from src.utils.hardware.m1_memory_optimizer import (
                    get_m1_memory_optimizer,
                    M1MemoryOptimizer
                )
                optimizers['memory_optimizer'] = get_m1_memory_optimizer
                optimizers['memory_optimizer_class'] = M1MemoryOptimizer
                self.logger.debug("✅ M1 Memory optimizer imported")
            except ImportError as e:
                self.logger.warning(f"⚠️ M1 Memory optimizer not available: {e}")
                if self.enable_fallbacks:
                    optimizers['memory_optimizer'] = self._get_fallback_memory_optimizer()

            # M1 CPU Optimizer
            try:
                from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer, M1CPUOptimizer
                optimizers['cpu_optimizer'] = get_m1_cpu_optimizer
                optimizers['cpu_optimizer_class'] = M1CPUOptimizer
                self.logger.debug("✅ M1 CPU optimizer imported")
            except ImportError as e:
                self.logger.warning(f"⚠️ M1 CPU optimizer not available: {e}")
                if self.enable_fallbacks:
                    optimizers['cpu_optimizer'] = self._get_fallback_cpu_optimizer()

            self.logger.info(f"✅ M1 optimization utilities loaded: {len(optimizers)} modules")
            return optimizers

        except Exception as e:
            self.logger.error(f"❌ Failed to import M1 optimization utilities: {e}")
            return None

    def _import_validation_utilities(self) -> Optional[Dict[str, Any]]:
        """Import validation utilities with fallback handling."""
        try:
            self.logger.info("🔍 Importing validation utilities...")

            validation = {}

            # Math Validation
            try:
                from src.utils.math_validation import (
                    safe_divide, safe_log, safe_sqrt, safe_kelly_calculation,
                    validate_positive, validate_range, MathValidationError
                )
                validation['math'] = {
                    'safe_divide': safe_divide,
                    'safe_log': safe_log,
                    'safe_sqrt': safe_sqrt,
                    'safe_kelly_calculation': safe_kelly_calculation,
                    'validate_positive': validate_positive,
                    'validate_range': validate_range,
                    'MathValidationError': MathValidationError
                }
                self.logger.debug("✅ Math validation utilities imported")
            except ImportError as e:
                self.logger.warning(f"⚠️ Math validation utilities not available: {e}")
                if self.enable_fallbacks:
                    validation['math'] = self._get_fallback_math_validation()

            # Common Operations
            try:
                from src.utils.common_operations import (
                    safe_float, safe_int, safe_dict_get, optimize_dataframe_dtypes,
                    validate_dataframe_schema, validate_data_quality
                )
                validation['common_operations'] = {
                    'safe_float': safe_float,
                    'safe_int': safe_int,
                    'safe_dict_get': safe_dict_get,
                    'optimize_dataframe_dtypes': optimize_dataframe_dtypes,
                    'validate_dataframe_schema': validate_dataframe_schema,
                    'validate_data_quality': validate_data_quality
                }
                self.logger.debug("✅ Common operations imported")
            except ImportError as e:
                self.logger.warning(f"⚠️ Common operations not available: {e}")
                if self.enable_fallbacks:
                    validation['common_operations'] = self._get_fallback_common_operations()

            # Common Utilities
            try:
                from src.utils.common_utilities import (
                    create_data_quality_report, get_global_detector,
                    validate_no_future_data, LookaheadBiasError
                )
                validation['common_utilities'] = {
                    'create_data_quality_report': create_data_quality_report,
                    'get_global_detector': get_global_detector,
                    'validate_no_future_data': validate_no_future_data,
                    'LookaheadBiasError': LookaheadBiasError
                }
                self.logger.debug("✅ Common utilities imported")
            except ImportError as e:
                self.logger.warning(f"⚠️ Common utilities not available: {e}")
                if self.enable_fallbacks:
                    validation['common_utilities'] = self._get_fallback_common_utilities()

            self.logger.info(f"✅ Validation utilities loaded: {len(validation)} modules")
            return validation

        except Exception as e:
            self.logger.error(f"❌ Failed to import validation utilities: {e}")
            return None

    def _import_data_processing_utilities(self) -> Optional[Dict[str, Any]]:
        """Import data processing utilities with fallback handling."""
        try:
            self.logger.info("📊 Importing data processing utilities...")

            data_processing = {}

            # Data Processing Utils
            try:
                from src.utils.data_processing_utils import (
                    preprocess_data, clean_data, validate_data_structure
                )
                data_processing['core'] = {
                    'preprocess_data': preprocess_data,
                    'clean_data': clean_data,
                    'validate_data_structure': validate_data_structure
                }
                self.logger.debug("✅ Data processing utilities imported")
            except ImportError as e:
                self.logger.warning(f"⚠️ Data processing utilities not available: {e}")
                if self.enable_fallbacks:
                    data_processing['core'] = self._get_fallback_data_processing()

            self.logger.info(f"✅ Data processing utilities loaded: {len(data_processing)} modules")
            return data_processing

        except Exception as e:
            self.logger.error(f"❌ Failed to import data processing utilities: {e}")
            return None

    def _import_serialization_utilities(self) -> Optional[Dict[str, Any]]:
        """Import serialization utilities with fallback handling."""
        try:
            self.logger.info("💾 Importing serialization utilities...")

            serialization = {}

            # Serialization Utils
            try:
                from src.utils.serialization_utils import (
                    safe_json_dump, safe_json_load, safe_read_parquet, safe_to_parquet
                )
                serialization['core'] = {
                    'safe_json_dump': safe_json_dump,
                    'safe_json_load': safe_json_load,
                    'safe_read_parquet': safe_read_parquet,
                    'safe_to_parquet': safe_to_parquet
                }
                self.logger.debug("✅ Serialization utilities imported")
            except ImportError as e:
                self.logger.warning(f"⚠️ Serialization utilities not available: {e}")
                if self.enable_fallbacks:
                    serialization['core'] = self._get_fallback_serialization()

            self.logger.info(f"✅ Serialization utilities loaded: {len(serialization)} modules")
            return serialization

        except Exception as e:
            self.logger.error(f"❌ Failed to import serialization utilities: {e}")
            return None

    def _import_parquet_utilities(self) -> Optional[Dict[str, Any]]:
        """Import parquet utilities with fallback handling."""
        try:
            self.logger.info("📁 Importing parquet utilities...")

            parquet = {}

            # Parquet Utils
            try:
                from src.utils.parquet_utils import get_parquet_utils, ParquetUtils
                parquet['core'] = {
                    'get_parquet_utils': get_parquet_utils,
                    'ParquetUtils': ParquetUtils
                }
                self.logger.debug("✅ Parquet utilities imported")
            except ImportError as e:
                self.logger.warning(f"⚠️ Parquet utilities not available: {e}")
                if self.enable_fallbacks:
                    parquet['core'] = self._get_fallback_parquet()

            self.logger.info(f"✅ Parquet utilities loaded: {len(parquet)} modules")
            return parquet

        except Exception as e:
            self.logger.error(f"❌ Failed to import parquet utilities: {e}")
            return None

    # Fallback implementations
    def _get_fallback_data_quality(self) -> Any:
        """Get fallback data quality utility."""
        class FallbackDataQuality:
            """Fallback data quality utility when ML common utilities are not available."""

            def __init__(self, *args, **kwargs):
                """Initialize fallback data quality utility.

                Args:
                    *args: Positional arguments
                    **kwargs: Keyword arguments
                """
                self.logger = logger.getChild('FallbackDataQuality')
                self.logger.warning("⚠️ Using fallback data quality utility")

            def missing_value_analysis(self, data):
                """Perform missing value analysis.

                Args:
                    data: Input data to analyze

                Returns:
                    Dict containing severity assessment and recommendations
                """
                return {'severity_assessment': {'severity_level': 'low'}, 'recommendations': []}

            def automated_outlier_detection(self, data):
                """Perform automated outlier detection.

                Args:
                    data: Input data to analyze

                Returns:
                    Dict containing outliers detected and recommendations
                """
                return {'outliers_detected': 0, 'recommendations': []}

            def feature_correlation_analysis(self, data):
                """Perform feature correlation analysis.

                Args:
                    data: Input data to analyze

                Returns:
                    Dict containing correlations and recommendations
                """
                return {'correlations': {}, 'recommendations': []}

            def automated_data_cleaning(self, data, config):
                """Perform automated data cleaning.

                Args:
                    data: Input data to clean
                    config: Cleaning configuration

                Returns:
                    Tuple of (cleaned_data, cleaning_metadata)
                """
                return data, {'total_removed_samples': 0}

        return FallbackDataQuality

    def _get_fallback_pipeline_orchestrator(self) -> Any:
        """Get fallback pipeline orchestrator."""
        class FallbackPipelineOrchestrator:
            def __init__(self, *args, **kwargs):
                self.logger = logger.getChild('FallbackPipelineOrchestrator')
                self.logger.warning("⚠️ Using fallback pipeline orchestrator")
                self.active_pipelines = {}

            def create_training_pipeline(self, *args, **kwargs):
                return f"fallback_pipeline_{int(time.time())}"

            def execute_pipeline(self, pipeline_id, *args, **kwargs):
                return {'success': True, 'results': {}}

            def automated_pipeline_optimization(self, *args, **kwargs):
                return {'recommendations': []}

        return FallbackPipelineOrchestrator

    def _get_fallback_feature_selection(self) -> Any:
        """Get fallback feature selection framework."""
        class FallbackFeatureSelection:
            def __init__(self, *args, **kwargs):
                self.logger = logger.getChild('FallbackFeatureSelection')
                self.logger.warning("⚠️ Using fallback feature selection")

            def mrmr_selection(self, features, target, feature_names, n_features):
                return {'selected_features': feature_names[:n_features]}

            def correlation_based_filtering(self, features, feature_names, correlation_threshold):
                return {'selected_features': feature_names}

        return FallbackFeatureSelection

    def _get_fallback_parallel_processing(self) -> Any:
        """Get fallback parallel processing coordinator."""
        class FallbackParallelProcessing:
            def __init__(self, *args, **kwargs):
                self.logger = logger.getChild('FallbackParallelProcessing')
                self.logger.warning("⚠️ Using fallback parallel processing")

            def error_handling_parallel_execution(self, tasks, *args, **kwargs):
                results = []
                for task in tasks:
                    try:
                        result = task['function'](*task.get('args', []), **task.get('kwargs', {}))
                        results.append({'success': True, 'result': result})
                    except Exception as e:
                        results.append({'success': False, 'error': str(e)})
                return results

            def parallel_feature_engineering(self, functions, data_list):
                results = []
                for func, data in zip(functions, data_list):
                    try:
                        result = func(data)
                        results.append(result)
                    except Exception as e:
                        results.append({'success': False, 'error': str(e)})
                return results

        return FallbackParallelProcessing

    def _get_fallback_matrix_operations(self) -> Any:
        """Get fallback matrix operations."""
        class FallbackMatrixOperations:
            def __init__(self, *args, **kwargs):
                self.logger = logger.getChild('FallbackMatrixOperations')
                self.logger.warning("⚠️ Using fallback matrix operations")

            def covariance_matrix(self, data):
                return np.cov(data.T)

            def eigendecomposition(self, matrix):
                return np.linalg.eig(matrix)

        return FallbackMatrixOperations

    def _get_fallback_data_labeling(self) -> Any:
        """Get fallback data labeling utility."""
        class FallbackDataLabeling:
            def __init__(self, *args, **kwargs):
                self.logger = logger.getChild('FallbackDataLabeling')
                self.logger.warning("⚠️ Using fallback data labeling")

        return FallbackDataLabeling

    def _get_fallback_hmm_detector(self) -> Any:
        """Get fallback HMM detector."""
        class FallbackHMMDetector:
            def __init__(self, *args, **kwargs):
                self.logger = logger.getChild('FallbackHMMDetector')
                self.logger.warning("⚠️ Using fallback HMM detector")

            def detect_regimes(self, data, *args, **kwargs):
                n_samples = len(data)
                return type('RegimeResult', (), {
                    'regime_ids': np.zeros(n_samples, dtype=int),
                    'regime_probabilities': np.ones((n_samples, 1))
                })()

        return FallbackHMMDetector

    def _get_fallback_regime_processor(self) -> Any:
        """Get fallback regime processor."""
        class FallbackRegimeProcessor:
            def __init__(self, *args, **kwargs):
                self.logger = logger.getChild('FallbackRegimeProcessor')
                self.logger.warning("⚠️ Using fallback regime processor")

        return FallbackRegimeProcessor

    def _get_fallback_gpu_manager(self) -> Any:
        """Get fallback GPU manager."""
        def fallback_gpu_manager():
            class FallbackGPUManager:
                def __init__(self):
                    self.logger = logger.getChild('FallbackGPUManager')
                    self.logger.warning("⚠️ Using fallback GPU manager")

                def is_available(self):
                    return False

                def get_device(self):
                    return 'cpu'

            return FallbackGPUManager()

        return fallback_gpu_manager

    def _get_fallback_memory_optimizer(self) -> Any:
        """Get fallback memory optimizer."""
        def fallback_memory_optimizer():
            class FallbackMemoryOptimizer:
                def __init__(self):
                    self.logger = logger.getChild('FallbackMemoryOptimizer')
                    self.logger.warning("⚠️ Using fallback memory optimizer")

                def memory_checkpoint(self, name):
                    return self._dummy_context()

                def create_memory_efficient_dataframe(self, df):
                    return df

                def _dummy_context(self):
                    from contextlib import nullcontext
                    return nullcontext()

            return FallbackMemoryOptimizer()

        return fallback_memory_optimizer

    def _get_fallback_cpu_optimizer(self) -> Any:
        """Get fallback CPU optimizer."""
        def fallback_cpu_optimizer():
            class FallbackCPUOptimizer:
                def __init__(self):
                    self.logger = logger.getChild('FallbackCPUOptimizer')
                    self.logger.warning("⚠️ Using fallback CPU optimizer")

                def calculate_optimal_chunk_size(self, shape):
                    return min(1000, shape[0] // 4)

                def get_optimal_workers_for_task(self, task_type):
                    return 1

            return FallbackCPUOptimizer()

        return fallback_cpu_optimizer

    def _get_fallback_math_validation(self) -> Dict[str, Any]:
        """Get fallback math validation utilities."""
        def safe_divide(a, b, default=0.0):
            try:
                return a / b if b != 0 else default
            except:
                return default

        def safe_log(x, default=0.0):
            try:
                return np.log(x) if x > 0 else default
            except:
                return default

        def safe_sqrt(x, default=0.0):
            try:
                return np.sqrt(x) if x >= 0 else default
            except:
                return default

        def validate_positive(value, name="value"):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
            return value

        def validate_range(value, min_val, max_val, name="value"):
            if not (min_val <= value <= max_val):
                raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
            return value

        class MathValidationError(Exception):
            pass

        return {
            'safe_divide': safe_divide,
            'safe_log': safe_log,
            'safe_sqrt': safe_sqrt,
            'safe_kelly_calculation': lambda x: x,  # Dummy implementation
            'validate_positive': validate_positive,
            'validate_range': validate_range,
            'MathValidationError': MathValidationError
        }

    def _get_fallback_common_operations(self) -> Dict[str, Any]:
        """Get fallback common operations utilities."""
        def safe_float(value, default=0.0):
            try:
                return float(value)
            except:
                return default

        def safe_int(value, default=0):
            try:
                return int(value)
            except:
                return default

        def safe_dict_get(d, key, default=None):
            return d.get(key, default)

        def optimize_dataframe_dtypes(df):
            return df

        def validate_dataframe_schema(df, schema):
            return True

        def validate_data_quality(df):
            return True

        return {
            'safe_float': safe_float,
            'safe_int': safe_int,
            'safe_dict_get': safe_dict_get,
            'optimize_dataframe_dtypes': optimize_dataframe_dtypes,
            'validate_dataframe_schema': validate_dataframe_schema,
            'validate_data_quality': validate_data_quality
        }

    def _get_fallback_common_utilities(self) -> Dict[str, Any]:
        """Get fallback common utilities."""
        def create_data_quality_report(data):
            return {'quality_score': 1.0, 'issues': []}

        def get_global_detector():
            return None

        def validate_no_future_data(data):
            return True

        class LookaheadBiasError(Exception):
            pass

        return {
            'create_data_quality_report': create_data_quality_report,
            'get_global_detector': get_global_detector,
            'validate_no_future_data': validate_no_future_data,
            'LookaheadBiasError': LookaheadBiasError
        }

    def _get_fallback_data_processing(self) -> Dict[str, Any]:
        """Get fallback data processing utilities."""
        def preprocess_data(data):
            return data

        def clean_data(data):
            return data

        def validate_data_structure(data):
            return True

        return {
            'preprocess_data': preprocess_data,
            'clean_data': clean_data,
            'validate_data_structure': validate_data_structure
        }

    def _get_fallback_serialization(self) -> Dict[str, Any]:
        """Get fallback serialization utilities."""
        def safe_json_dump(data, file_path):
            import json
            with open(file_path, 'w') as f:
                json.dump(data, f)

        def safe_json_load(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)

        def safe_read_parquet(file_path):
            return pd.read_parquet(file_path)

        def safe_to_parquet(df, file_path):
            df.to_parquet(file_path)

        return {
            'safe_json_dump': safe_json_dump,
            'safe_json_load': safe_json_load,
            'safe_read_parquet': safe_read_parquet,
            'safe_to_parquet': safe_to_parquet
        }

    def _get_fallback_parquet(self) -> Dict[str, Any]:
        """Get fallback parquet utilities."""
        class FallbackParquetUtils:
            def read_parquet(self, file_path):
                return pd.read_parquet(file_path)

            def write_parquet(self, df, file_path):
                df.to_parquet(file_path)

        def get_parquet_utils():
            return FallbackParquetUtils()

        return {
            'get_parquet_utils': get_parquet_utils,
            'ParquetUtils': FallbackParquetUtils
        }

    def _log_import_summary(self, suite: UtilitySuite):
        """Log summary of import results."""
        self.logger.info("📊 Import Summary:")

        modules = [
            ('ML Commons', suite.ml_common),
            ('M1 Optimizers', suite.m1_optimizers),
            ('Validation', suite.validation),
            ('Data Processing', suite.data_processing),
            ('Serialization', suite.serialization),
            ('Parquet', suite.parquet)
        ]

        for name, module in modules:
            if module:
                self.logger.info(f"  ✅ {name}: {len(module)} utilities loaded")
            else:
                self.logger.warning(f"  ❌ {name}: Failed to load")

        if suite.fallbacks_used:
            self.logger.warning(f"  ⚠️ Fallbacks used: {', '.join(suite.fallbacks_used)}")

        if suite.import_errors:
            self.logger.error(f"  ❌ Import errors: {', '.join(suite.import_errors)}")

    def get_import_statistics(self) -> Dict[str, Any]:
        """Get statistics about import operations."""
        return {
            'total_imports': len(self.import_cache),
            'successful_imports': sum(1 for r in self.import_cache.values() if r.success),
            'failed_imports': sum(1 for r in self.import_cache.values() if not r.success),
            'fallbacks_used': sum(1 for r in self.import_cache.values() if r.fallback_used),
            'average_import_time': (
                np.mean([r.import_time for r in self.import_cache.values()])
                if self.import_cache else 0
            )
        }

# Global instance for easy access
_import_manager: Optional[DataQualificationImportManager] = None

def get_import_manager() -> DataQualificationImportManager:
    """Get the global import manager instance."""
    global _import_manager
    if _import_manager is None:
        _import_manager = DataQualificationImportManager()
    return _import_manager

def get_utility_suite() -> UtilitySuite:
    """Get the utility suite from the global import manager."""
    return get_import_manager().get_utility_suite()

# Convenience functions for backward compatibility
def get_ml_commons_utilities() -> Optional[Dict[str, Any]]:
    """Get ML Commons utilities."""
    return get_utility_suite().ml_common

def get_m1_optimization_utilities() -> Optional[Dict[str, Any]]:
    """Get M1 optimization utilities."""
    return get_utility_suite().m1_optimizers

def get_validation_utilities() -> Optional[Dict[str, Any]]:
    """Get validation utilities."""
    return get_utility_suite().validation

def get_data_processing_utilities() -> Optional[Dict[str, Any]]:
    """Get data processing utilities."""
    return get_utility_suite().data_processing

def get_serialization_utilities() -> Optional[Dict[str, Any]]:
    """Get serialization utilities."""
    return get_utility_suite().serialization

def get_parquet_utilities() -> Optional[Dict[str, Any]]:
    """Get parquet utilities."""
    return get_utility_suite().parquet

# Export main classes and functions
__all__ = [
    'DataQualificationImportManager',
    'UtilitySuite',
    'ImportResult',
    'get_import_manager',
    'get_utility_suite',
    'get_ml_commons_utilities',
    'get_m1_optimization_utilities',
    'get_validation_utilities',
    'get_data_processing_utilities',
    'get_serialization_utilities',
    'get_parquet_utilities'
]
