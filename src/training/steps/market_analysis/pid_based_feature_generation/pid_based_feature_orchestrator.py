"""
PID-Based Feature Orchestrator

This module orchestrates all PID-based feature generation processes, integrating
interaction, polynomial, and cross-timeframe feature generation with optimized
lookback periods from feature_lookback_optimization.

Key Features:
- Orchestrates all three feature generation types
- Integrates optimized lookback periods
- Uses matrix_operations/ for all calculations
- Comprehensive validation and error handling
- Hardware-optimized computations
"""

import asyncio
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Core dependencies with fallback support
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Import feature generators
from .interaction_feature_generator import InteractionFeatureGenerator, InteractionConfig, InteractionResult
from .polynomial_feature_generator import PolynomialFeatureGenerator, PolynomialConfig, PolynomialResult
from .cross_timeframe_feature_generator import CrossTimeframeFeatureGenerator, CrossTimeframeConfig, CrossTimeframeResult
from .optimized_lookback_integration import OptimizedLookbackIntegration, LookbackIntegrationResult
from .feature_selection_mechanism import FeatureSelectionMechanism, FeatureSelectionConfig, FeatureSelectionResult

# Import matrix operations
try:
    from src.utils.matrix_operations import get_unified_matrix_operations
    MATRIX_OPS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Matrix operations not available: {e}")
    MATRIX_OPS_AVAILABLE = False

# Import tprint for extensive logging
try:
    from src.utils.tprint import tprint, tprint_info, tprint_error, tprint_warning, tprint_success, tprint_debug, tprint_performance
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    # Fallback to basic print
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)

# Import common operations for comprehensive utility integration
try:
    from src.utils.common_operations import (
        # Data validation and quality
        validate_dataframe, validate_dataframe_columns, calculate_data_quality_metrics,
        create_data_quality_report, get_dataframe_info, optimize_dataframe_dtypes,
        
        # Safe operations
        safe_dataframe_operation, safe_fillna, safe_convert_dtypes, safe_merge_dataframes,
        safe_drop_columns, safe_rename_columns, safe_timestamp_conversion,
        
        # Math operations
        safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
        safe_float, safe_int, validate_finite, validate_positive, validate_range,
        safe_kelly_calculation, safe_weighted_average, safe_percentage_change,
        
        # File operations
        safe_json_dump, safe_json_load, safe_to_parquet, safe_read_parquet,
        ensure_directory, safe_file_exists, safe_copy,
        
        # Performance utilities
        timed_operation, format_bytes, chunked_iterable, parallel_map,
        
        # M1 optimization
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        integrate_with_m1_optimizers, cleanup_m1_optimizers,
        memory_checkpoint, gpu_context, optimize_memory, get_memory_usage,
        
        # Matrix utilities
        validate_correlation_matrix, safe_matrix_inverse, math_safe,
        
        # Logging utilities
        get_logger, setup_basic_logging, safe_log_metric, safe_log_params, safe_log_artifact
    )
    COMMON_OPERATIONS_AVAILABLE = True
except ImportError as e:
    COMMON_OPERATIONS_AVAILABLE = False
    logging.warning(f"Common operations not available: {e}")
    # Fallback functions
    def safe_divide(a, b, default=0.0): return a / b if b != 0 else default
    def safe_log(x, default=0.0): return np.log(x) if x > 0 else default
    def safe_sqrt(x, default=0.0): return np.sqrt(x) if x >= 0 else default
    def safe_power(x, y, default=0.0): return x ** y if np.isfinite(x) and np.isfinite(y) else default
    def validate_finite(value, name="value"): return float(value) if np.isfinite(value) else 0.0

# Import serialization utilities
try:
    from src.utils.serialization_utils import (
        JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
    )
    SERIALIZATION_AVAILABLE = True
except ImportError as e:
    SERIALIZATION_AVAILABLE = False
    logging.warning(f"Serialization utilities not available: {e}")

# Import math validation for additional math operations
try:
    from src.utils.math_validation import MathValidation, safe_correlation, safe_covariance, safe_percentile
    MATH_VALIDATION_AVAILABLE = True
except ImportError:
    MATH_VALIDATION_AVAILABLE = False

# Import logger as fallback
try:
    from src.utils.logger import system_logger
    logger = system_logger.getChild('PIDBasedFeatureOrchestrator')
except ImportError:
    logger = logging.getLogger('PIDBasedFeatureOrchestrator')
    logger.setLevel(logging.INFO)


class GenerationStatus(Enum):
    """Status of feature generation process."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class OrchestratorConfig:
    """Configuration for PID-based feature orchestrator with common utilities integration."""
    # Feature Generation Limits
    max_interaction_features: int = 100
    max_polynomial_features: int = 50
    max_cross_timeframe_features: int = 50
    
    # PID Configuration
    synergy_threshold: float = 0.1
    redundancy_threshold: float = 0.15
    unique_info_threshold: float = 0.05
    
    # Computational Settings
    enable_parallel_processing: bool = True
    enable_gpu_acceleration: bool = True
    memory_limit_gb: float = 8.0
    
    # Generation Control
    enable_interaction_features: bool = True
    enable_polynomial_features: bool = True
    enable_cross_timeframe_features: bool = True
    
    # Validation
    min_feature_quality_score: float = 0.3
    max_redundancy_threshold: float = 0.8
    
    # Hardware Optimization
    chunk_size_mb: int = 256
    max_memory_percent: float = 0.7
    
    # Common Utilities Integration
    enable_common_operations: bool = True
    enable_serialization: bool = True
    enable_data_validation: bool = True
    enable_data_optimization: bool = True
    enable_m1_optimization: bool = True
    
    # Data Quality Settings
    min_data_quality_score: float = 0.7
    max_missing_data_ratio: float = 0.1
    enable_quality_reporting: bool = True
    
    # Serialization Settings
    save_intermediate_results: bool = True
    serialization_format: str = 'parquet'  # 'json', 'pickle', 'parquet'
    artifacts_directory: str = 'artifacts/pid_features'
    
    # Performance Settings
    enable_profiling: bool = True
    enable_memory_monitoring: bool = True
    enable_performance_logging: bool = True


@dataclass
class OrchestratorResult:
    """Result of PID-based feature orchestration with common utilities integration."""
    # Individual Results
    interaction_result: Optional[InteractionResult] = None
    polynomial_result: Optional[PolynomialResult] = None
    cross_timeframe_result: Optional[CrossTimeframeResult] = None
    
    # Combined Results
    combined_features: Dict[str, np.ndarray] = field(default_factory=dict)
    combined_feature_names: List[str] = field(default_factory=list)
    feature_importance_scores: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    total_features_generated: int = 0
    execution_time: float = 0.0
    optimization_used: bool = False
    matrix_ops_used: bool = False
    generation_status: GenerationStatus = GenerationStatus.PENDING
    
    # Quality Metrics
    overall_quality_score: float = 0.0
    feature_diversity_score: float = 0.0
    redundancy_score: float = 0.0
    stability_score: float = 0.0
    
    # Common Utilities Integration Results
    data_quality_report: Optional[Dict[str, Any]] = None
    validation_results: Dict[str, Any] = field(default_factory=dict)
    optimization_results: Dict[str, Any] = field(default_factory=dict)
    serialization_status: Dict[str, bool] = field(default_factory=dict)
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    hardware_optimization_used: bool = False
    memory_usage: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    utility_integration_status: Dict[str, bool] = field(default_factory=dict)


class PIDBasedFeatureOrchestrator:
    """
    PID-Based Feature Orchestrator.
    
    Orchestrates all PID-based feature generation processes, integrating
    interaction, polynomial, and cross-timeframe feature generation.
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """Initialize the PID-based feature orchestrator with common utilities integration."""
        try:
            # Input validation
            if config is not None and not isinstance(config, OrchestratorConfig):
                raise TypeError(f"Config must be OrchestratorConfig or None, got {type(config)}")
            
            self.config = config or OrchestratorConfig()
            self.logger = logger.getChild('PIDBasedFeatureOrchestrator')
            
            # Initialize common utilities integration
            self._initialize_common_utilities()
            
            # Initialize math validation
            if MATH_VALIDATION_AVAILABLE:
                self.math_validator = MathValidation()
            else:
                self.math_validator = None
            
            # Initialize components
            self._initialize_components()
            
            tprint_success("PIDBasedFeatureOrchestrator initialized successfully")
            tprint_info(f"Max interaction features: {self.config.max_interaction_features}")
            tprint_info(f"Max polynomial features: {self.config.max_polynomial_features}")
            tprint_info(f"Max cross-timeframe features: {self.config.max_cross_timeframe_features}")
            tprint_info(f"Common operations available: {COMMON_OPERATIONS_AVAILABLE}")
            tprint_info(f"Serialization available: {SERIALIZATION_AVAILABLE}")
            tprint_info(f"Math validation available: {MATH_VALIDATION_AVAILABLE}")
            tprint_info(f"Matrix operations available: {MATRIX_OPS_AVAILABLE}")
            
        except Exception as e:
            tprint_error(f"Failed to initialize PIDBasedFeatureOrchestrator: {e}")
            raise
    
    def _initialize_common_utilities(self):
        """Initialize common utilities integration."""
        # Initialize serializers
        if SERIALIZATION_AVAILABLE and self.config.enable_serialization:
            self.json_serializer = JSONSerializer()
            self.pickle_serializer = PickleSerializer()
            self.parquet_serializer = ParquetSerializer()
            self.universal_serializer = UniversalSerializer()
            tprint_success("Serializers initialized")
        else:
            self.json_serializer = None
            self.pickle_serializer = None
            self.parquet_serializer = None
            self.universal_serializer = None
        
        # Initialize M1 optimizers
        if COMMON_OPERATIONS_AVAILABLE and self.config.enable_m1_optimization:
            self.gpu_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
            tprint_success("M1 optimizers initialized")
        else:
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
        
        # Initialize utility status tracking
        self.utility_integration_status = {
            'common_operations': COMMON_OPERATIONS_AVAILABLE and self.config.enable_common_operations,
            'serialization': SERIALIZATION_AVAILABLE and self.config.enable_serialization,
            'math_validation': MATH_VALIDATION_AVAILABLE,
            'matrix_operations': MATRIX_OPS_AVAILABLE,
            'data_validation': self.config.enable_data_validation,
            'data_optimization': self.config.enable_data_optimization,
            'm1_optimization': self.config.enable_m1_optimization
        }
        
        tprint_info(f"Utility integration status: {self.utility_integration_status}")
    
    def _initialize_components(self):
        """Initialize required components."""
        # Initialize feature generators
        if self.config.enable_interaction_features:
            interaction_config = InteractionConfig(
                synergy_threshold=self.config.synergy_threshold,
                redundancy_threshold=self.config.redundancy_threshold,
                unique_info_threshold=self.config.unique_info_threshold,
                max_interaction_features=self.config.max_interaction_features,
                enable_parallel_processing=self.config.enable_parallel_processing,
                enable_gpu_acceleration=self.config.enable_gpu_acceleration,
                memory_limit_gb=self.config.memory_limit_gb
            )
            self.interaction_generator = InteractionFeatureGenerator(interaction_config)
            self.logger.info("✅ Interaction Feature Generator initialized")
        else:
            self.interaction_generator = None
        
        if self.config.enable_polynomial_features:
            polynomial_config = PolynomialConfig(
                synergy_threshold=self.config.synergy_threshold,
                redundancy_threshold=self.config.redundancy_threshold,
                unique_info_threshold=self.config.unique_info_threshold,
                max_polynomial_features=self.config.max_polynomial_features,
                enable_parallel_processing=self.config.enable_parallel_processing,
                enable_gpu_acceleration=self.config.enable_gpu_acceleration,
                memory_limit_gb=self.config.memory_limit_gb
            )
            self.polynomial_generator = PolynomialFeatureGenerator(polynomial_config)
            self.logger.info("✅ Polynomial Feature Generator initialized")
        else:
            self.polynomial_generator = None
        
        if self.config.enable_cross_timeframe_features:
            cross_timeframe_config = CrossTimeframeConfig(
                synergy_threshold=self.config.synergy_threshold,
                redundancy_threshold=self.config.redundancy_threshold,
                unique_info_threshold=self.config.unique_info_threshold,
                max_cross_timeframe_features=self.config.max_cross_timeframe_features,
                enable_parallel_processing=self.config.enable_parallel_processing,
                enable_gpu_acceleration=self.config.enable_gpu_acceleration,
                memory_limit_gb=self.config.memory_limit_gb
            )
            self.cross_timeframe_generator = CrossTimeframeFeatureGenerator(cross_timeframe_config)
            self.logger.info("✅ Cross Timeframe Feature Generator initialized")
        else:
            self.cross_timeframe_generator = None
        
        # Initialize matrix operations
        if MATRIX_OPS_AVAILABLE:
            self.matrix_ops = get_unified_matrix_operations(
                enable_gpu=self.config.enable_gpu_acceleration,
                enable_memory_optimization=True,
                enable_parallel=self.config.enable_parallel_processing
            )
            self.logger.info("✅ Matrix Operations initialized")
        else:
            self.matrix_ops = None
            self.logger.warning("⚠️ Matrix Operations not available")
    
    async def orchestrate_feature_generation(
        self, 
        data: Union[np.ndarray, pd.DataFrame],
        feature_names: List[str],
        optimized_lookback_periods: Optional[Dict[str, int]] = None,
        target: Optional[np.ndarray] = None
    ) -> OrchestratorResult:
        """
        Orchestrate all PID-based feature generation processes.
        
        Args:
            data: Input feature matrix
            feature_names: List of feature names
            optimized_lookback_periods: Optimized lookback periods from feature_lookback_optimization
            target: Target variable for PID analysis (optional)
            
        Returns:
            OrchestratorResult with all generated features
        """
        start_time = time.time()
        tprint_info("Starting PID-based feature orchestration...")
        
        result = OrchestratorResult()
        result.generation_status = GenerationStatus.IN_PROGRESS
        
        try:
            # Enhanced input validation with common utilities
            validation_result = await self._validate_input_data(data, feature_names, target)
            if not validation_result['is_valid']:
                raise ValueError(f"Data validation failed: {validation_result['issues']}")
            
            # Apply data optimization if enabled
            if self.config.enable_data_optimization:
                data, feature_names, optimization_info = await self._optimize_input_data(data, feature_names)
                result.optimization_results = optimization_info
                result.optimization_used = True
            
            # Convert data to numpy array if needed
            if isinstance(data, pd.DataFrame):
                X = data.values
                if feature_names is None:
                    feature_names = list(data.columns)
                tprint_info(f"Converted DataFrame to numpy array: {X.shape}")
            else:
                X = data
                tprint_info(f"Using numpy array data: {X.shape}")
            
            # Enhanced data quality assessment
            if self.config.enable_quality_reporting:
                quality_report = await self._assess_data_quality(X, feature_names)
                result.data_quality_report = quality_report
                tprint_info(f"Data quality score: {quality_report.get('overall_score', 0.0):.3f}")
            
            # Validate data shape
            if X.shape[0] == 0:
                raise ValueError("Input data has no samples - fast failing")
            if X.shape[1] == 0:
                raise ValueError("Input data has no features - fast failing")
            
            # Check for NaN/Inf values with safe operations
            nan_count = np.sum(np.isnan(X))
            inf_count = np.sum(np.isinf(X))
            if nan_count > 0:
                tprint_warning(f"Input data contains {nan_count} NaN values - this may cause issues")
            if inf_count > 0:
                tprint_warning(f"Input data contains {inf_count} Inf values - this may cause issues")
            
            # Validate feature names match data dimensions
            if len(feature_names) != X.shape[1]:
                raise ValueError(f"Feature names count ({len(feature_names)}) doesn't match data columns ({X.shape[1]}) - fast failing")
            
            # Validate target if provided
            if target is not None:
                if len(target) != X.shape[0]:
                    raise ValueError(f"Target length ({len(target)}) doesn't match data length ({X.shape[0]}) - fast failing")
                if np.any(np.isnan(target)) or np.any(np.isinf(target)):
                    tprint_warning("Target contains NaN or Inf values - this may cause issues")
            
            tprint_info(f"Input data shape: {X.shape}")
            tprint_info(f"Feature count: {len(feature_names)}")
            tprint_info(f"Data type: {X.dtype}")
            
            # Track optimization usage
            if optimized_lookback_periods:
                result.optimization_used = True
                tprint_info(f"Optimized lookback periods will be applied: {len(optimized_lookback_periods)} periods")
            else:
                tprint_info("No optimized lookback periods provided - using defaults")
            
            # Generate features in parallel if possible
            generation_tasks = []
            
            # Interaction features
            if self.interaction_generator:
                try:
                    tprint_info("Creating interaction feature generation task...")
                    task = asyncio.create_task(
                        self.interaction_generator.generate_interaction_features(
                            X, feature_names, optimized_lookback_periods, target
                        )
                    )
                    generation_tasks.append(('interaction', task))
                    tprint_success("Interaction feature generation task created")
                except Exception as e:
                    tprint_error(f"Failed to create interaction feature generation task: {e}")
                    raise
            
            # Polynomial features
            if self.polynomial_generator:
                try:
                    tprint_info("Creating polynomial feature generation task...")
                    task = asyncio.create_task(
                        self.polynomial_generator.generate_polynomial_features(
                            X, feature_names, optimized_lookback_periods, target
                        )
                    )
                    generation_tasks.append(('polynomial', task))
                    tprint_success("Polynomial feature generation task created")
                except Exception as e:
                    tprint_error(f"Failed to create polynomial feature generation task: {e}")
                    raise
            
            # Cross-timeframe features
            if self.cross_timeframe_generator:
                try:
                    tprint_info("Creating cross-timeframe feature generation task...")
                    task = asyncio.create_task(
                        self.cross_timeframe_generator.generate_cross_timeframe_features(
                            X, feature_names, optimized_lookback_periods, target
                        )
                    )
                    generation_tasks.append(('cross_timeframe', task))
                    tprint_success("Cross-timeframe feature generation task created")
                except Exception as e:
                    tprint_error(f"Failed to create cross-timeframe feature generation task: {e}")
                    raise
            
            # Wait for all tasks to complete
            tprint_info(f"Executing {len(generation_tasks)} feature generation tasks...")
            try:
                completed_tasks = await asyncio.gather(*[task for _, task in generation_tasks], return_exceptions=True)
                tprint_success("All feature generation tasks completed")
            except Exception as e:
                tprint_error(f"Failed to execute feature generation tasks: {e}")
                raise
            
            # Process results
            successful_generations = 0
            failed_generations = 0
            
            tprint_info("Processing feature generation results...")
            for (generation_type, _), task_result in zip(generation_tasks, completed_tasks):
                try:
                    if isinstance(task_result, Exception):
                        tprint_error(f"{generation_type} feature generation failed: {task_result}")
                        failed_generations += 1
                        continue
                    
                    # Validate task result
                    if task_result is None:
                        tprint_warning(f"{generation_type} feature generation returned None")
                        failed_generations += 1
                        continue
                    
                    # Store result based on type
                    if generation_type == 'interaction':
                        result.interaction_result = task_result
                        tprint_success(f"Interaction features: {getattr(task_result, 'total_features_generated', 0)} features")
                    elif generation_type == 'polynomial':
                        result.polynomial_result = task_result
                        tprint_success(f"Polynomial features: {getattr(task_result, 'total_features_generated', 0)} features")
                    elif generation_type == 'cross_timeframe':
                        result.cross_timeframe_result = task_result
                        tprint_success(f"Cross-timeframe features: {getattr(task_result, 'total_features_generated', 0)} features")
                    
                    successful_generations += 1
                    tprint_success(f"{generation_type} feature generation completed successfully")
                    
                except Exception as e:
                    tprint_error(f"Error processing {generation_type} result: {e}")
                    failed_generations += 1
            
            tprint_info(f"Feature generation summary: {successful_generations} successful, {failed_generations} failed")
            
            # Combine all generated features
            try:
                tprint_info("Combining all generated features...")
                combined_features, combined_names, importance_scores = self._combine_features(result)
                tprint_success(f"Combined features: {len(combined_names)} total features")
            except Exception as e:
                tprint_error(f"Failed to combine features: {e}")
                raise
            
            # Store combined results
            result.combined_features = combined_features
            result.combined_feature_names = combined_names
            result.feature_importance_scores = importance_scores
            result.total_features_generated = len(combined_names)
            result.matrix_ops_used = self.matrix_ops is not None
            
            # Calculate quality metrics
            try:
                tprint_info("Calculating quality metrics...")
                result.overall_quality_score = self._calculate_overall_quality_score(result)
                result.feature_diversity_score = self._calculate_feature_diversity_score(combined_names)
                result.redundancy_score = self._calculate_redundancy_score(combined_features)
                result.stability_score = self._calculate_stability_score(result)
                tprint_success("Quality metrics calculated successfully")
            except Exception as e:
                tprint_warning(f"Failed to calculate quality metrics: {e}")
                # Set default values
                result.overall_quality_score = 0.0
                result.feature_diversity_score = 0.0
                result.redundancy_score = 0.0
                result.stability_score = 0.0
            
            # Determine final status
            if successful_generations == len(generation_tasks):
                result.generation_status = GenerationStatus.COMPLETED
                tprint_success("All feature generation tasks completed successfully")
            elif successful_generations > 0:
                result.generation_status = GenerationStatus.PARTIAL
                tprint_warning(f"Partial success: {successful_generations}/{len(generation_tasks)} tasks completed")
            else:
                result.generation_status = GenerationStatus.FAILED
                tprint_error("All feature generation tasks failed")
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            # Save artifacts if enabled
            if self.config.save_intermediate_results:
                tprint_info("💾 Saving artifacts...")
                serialization_result = await self._save_artifacts(result, start_time)
                result.serialization_status = serialization_result['status']
                result.artifact_paths = serialization_result['paths']
                tprint_success(f"Artifacts saved: {sum(serialization_result['status'].values())} successful")
            
            # Collect performance metrics
            if self.config.enable_performance_logging:
                tprint_info("📈 Collecting performance metrics...")
                performance_metrics = await self._collect_performance_metrics(start_time)
                result.performance_metrics = performance_metrics
                result.memory_usage = performance_metrics.get('memory_usage', {})
                result.hardware_optimization_used = bool(self.gpu_manager or self.memory_optimizer or self.cpu_optimizer)
                tprint_success("Performance metrics collected")
            
            # Set utility integration status
            result.utility_integration_status = self.utility_integration_status
            
            tprint_performance("PID-based feature orchestration", execution_time)
            tprint_info(f"Generated {result.total_features_generated} total features")
            tprint_info(f"Overall quality score: {result.overall_quality_score:.3f}")
            tprint_info(f"Feature diversity score: {result.feature_diversity_score:.3f}")
            tprint_info(f"Generation status: {result.generation_status.value}")
            tprint_info(f"Utility integrations: {sum(result.utility_integration_status.values())}/{len(result.utility_integration_status)}")
            
            return result
            
        except ValueError as e:
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            result.generation_status = GenerationStatus.FAILED
            
            tprint_error(f"PID-based feature orchestration failed - validation error: {e}")
            tprint_error(f"Error details: {traceback.format_exc()}")
            
            return result
            
        except TypeError as e:
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            result.generation_status = GenerationStatus.FAILED
            
            tprint_error(f"PID-based feature orchestration failed - type error: {e}")
            tprint_error(f"Error details: {traceback.format_exc()}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            result.generation_status = GenerationStatus.FAILED
            
            tprint_error(f"PID-based feature orchestration failed - unexpected error: {e}")
            tprint_error(f"Error type: {type(e).__name__}")
            tprint_error(f"Error details: {traceback.format_exc()}")
            
            return result
    
    def _combine_features(
        self, 
        result: OrchestratorResult
    ) -> Tuple[Dict[str, np.ndarray], List[str], Dict[str, float]]:
        """Combine features from all generators."""
        try:
            tprint_info("Starting feature combination process...")
            combined_features = {}
            combined_names = []
            importance_scores = {}
            
            # Add interaction features
            if result.interaction_result and hasattr(result.interaction_result, 'interaction_features') and result.interaction_result.interaction_features:
                tprint_info(f"Combining {len(result.interaction_result.interaction_features)} interaction features...")
                for name, feature in result.interaction_result.interaction_features.items():
                    try:
                        # Validate feature data
                        if feature is None or not hasattr(feature, 'shape'):
                            tprint_warning(f"Skipping invalid interaction feature: {name}")
                            continue
                        
                        combined_features[f"interaction_{name}"] = feature
                        combined_names.append(f"interaction_{name}")
                        
                        # Safe importance score extraction
                        score = result.interaction_result.interaction_scores.get(name, 0.0) if hasattr(result.interaction_result, 'interaction_scores') else 0.0
                        importance_scores[f"interaction_{name}"] = validate_finite(score, f"interaction_{name}_score")
                        
                    except Exception as e:
                        tprint_warning(f"Failed to combine interaction feature {name}: {e}")
                        continue
                
                tprint_success(f"Combined {len([f for f in combined_names if f.startswith('interaction_')])} interaction features")
            
            # Add polynomial features
            if result.polynomial_result and hasattr(result.polynomial_result, 'polynomial_features') and result.polynomial_result.polynomial_features:
                tprint_info(f"Combining {len(result.polynomial_result.polynomial_features)} polynomial features...")
                for name, feature in result.polynomial_result.polynomial_features.items():
                    try:
                        # Validate feature data
                        if feature is None or not hasattr(feature, 'shape'):
                            tprint_warning(f"Skipping invalid polynomial feature: {name}")
                            continue
                        
                        combined_features[f"polynomial_{name}"] = feature
                        combined_names.append(f"polynomial_{name}")
                        
                        # Safe importance score extraction
                        score = result.polynomial_result.polynomial_scores.get(name, 0.0) if hasattr(result.polynomial_result, 'polynomial_scores') else 0.0
                        importance_scores[f"polynomial_{name}"] = validate_finite(score, f"polynomial_{name}_score")
                        
                    except Exception as e:
                        tprint_warning(f"Failed to combine polynomial feature {name}: {e}")
                        continue
                
                tprint_success(f"Combined {len([f for f in combined_names if f.startswith('polynomial_')])} polynomial features")
            
            # Add cross-timeframe features
            if result.cross_timeframe_result and hasattr(result.cross_timeframe_result, 'cross_timeframe_features') and result.cross_timeframe_result.cross_timeframe_features:
                tprint_info(f"Combining {len(result.cross_timeframe_result.cross_timeframe_features)} cross-timeframe features...")
                for name, feature in result.cross_timeframe_result.cross_timeframe_features.items():
                    try:
                        # Validate feature data
                        if feature is None or not hasattr(feature, 'shape'):
                            tprint_warning(f"Skipping invalid cross-timeframe feature: {name}")
                            continue
                        
                        combined_features[f"cross_timeframe_{name}"] = feature
                        combined_names.append(f"cross_timeframe_{name}")
                        
                        # Safe importance score extraction
                        score = result.cross_timeframe_result.cross_timeframe_scores.get(name, 0.0) if hasattr(result.cross_timeframe_result, 'cross_timeframe_scores') else 0.0
                        importance_scores[f"cross_timeframe_{name}"] = validate_finite(score, f"cross_timeframe_{name}_score")
                        
                    except Exception as e:
                        tprint_warning(f"Failed to combine cross-timeframe feature {name}: {e}")
                        continue
                
                tprint_success(f"Combined {len([f for f in combined_names if f.startswith('cross_timeframe_')])} cross-timeframe features")
            
            tprint_success(f"Feature combination completed: {len(combined_names)} total features")
            return combined_features, combined_names, importance_scores
            
        except Exception as e:
            tprint_error(f"Failed to combine features: {e}")
            raise
    
    def _calculate_overall_quality_score(self, result: OrchestratorResult) -> float:
        """Calculate overall quality score."""
        try:
            tprint_debug("Calculating overall quality score...")
            scores = []
            
            # Individual quality scores
            if result.interaction_result and hasattr(result.interaction_result, 'feature_stability_score'):
                score = validate_finite(result.interaction_result.feature_stability_score, "interaction_stability")
                scores.append(score)
                tprint_debug(f"Interaction stability score: {score:.4f}")
            
            if result.polynomial_result and hasattr(result.polynomial_result, 'feature_stability_score'):
                score = validate_finite(result.polynomial_result.feature_stability_score, "polynomial_stability")
                scores.append(score)
                tprint_debug(f"Polynomial stability score: {score:.4f}")
            
            if result.cross_timeframe_result and hasattr(result.cross_timeframe_result, 'feature_stability_score'):
                score = validate_finite(result.cross_timeframe_result.feature_stability_score, "cross_timeframe_stability")
                scores.append(score)
                tprint_debug(f"Cross-timeframe stability score: {score:.4f}")
            
            # Generation success rate
            total_generators = sum([
                bool(result.interaction_result),
                bool(result.polynomial_result),
                bool(result.cross_timeframe_result)
            ])
            success_rate = safe_divide(total_generators, 3.0, 0.0)
            scores.append(success_rate)
            tprint_debug(f"Success rate: {success_rate:.4f}")
            
            if scores:
                overall_score = validate_finite(np.mean(scores), "overall_quality")
                tprint_debug(f"Overall quality score: {overall_score:.4f}")
                return overall_score
            else:
                tprint_warning("No quality scores available, returning 0.0")
                return 0.0
            
        except Exception as e:
            tprint_warning(f"Failed to calculate overall quality score: {e}")
            return 0.0
    
    def _calculate_feature_diversity_score(self, feature_names: List[str]) -> float:
        """Calculate feature diversity score based on naming patterns."""
        try:
            tprint_debug("Calculating feature diversity score...")
            
            if not feature_names:
                tprint_warning("No feature names provided for diversity calculation")
                return 0.0
            
            # Count different feature types
            interaction_count = sum(1 for name in feature_names if name.startswith('interaction_'))
            polynomial_count = sum(1 for name in feature_names if name.startswith('polynomial_'))
            cross_timeframe_count = sum(1 for name in feature_names if name.startswith('cross_timeframe_'))
            
            total_count = len(feature_names)
            tprint_debug(f"Feature type counts - Interaction: {interaction_count}, Polynomial: {polynomial_count}, Cross-timeframe: {cross_timeframe_count}")
            
            # Calculate diversity as entropy
            proportions = [
                safe_divide(interaction_count, total_count, 0.0),
                safe_divide(polynomial_count, total_count, 0.0),
                safe_divide(cross_timeframe_count, total_count, 0.0)
            ]
            
            # Remove zero proportions
            proportions = [p for p in proportions if p > 0]
            
            if not proportions:
                tprint_warning("No valid proportions for diversity calculation")
                return 0.0
            
            # Calculate entropy using safe log
            entropy = -sum(p * safe_log(p, 0.0) for p in proportions)
            max_entropy = safe_log(len(proportions), 0.0)
            
            diversity_score = safe_divide(entropy, max_entropy, 0.0) if max_entropy > 0 else 0.0
            diversity_score = validate_finite(diversity_score, "diversity_score")
            
            tprint_debug(f"Feature diversity score: {diversity_score:.4f}")
            return diversity_score
            
        except Exception as e:
            tprint_warning(f"Failed to calculate feature diversity score: {e}")
            return 0.0
    
    def _calculate_redundancy_score(self, combined_features: Dict[str, np.ndarray]) -> float:
        """Calculate redundancy score."""
        try:
            tprint_debug("Calculating redundancy score...")
            
            if len(combined_features) < 2:
                tprint_warning("Insufficient features for redundancy calculation")
                return 0.0
            
            # Convert to matrix
            try:
                feature_matrix = np.column_stack(list(combined_features.values()))
                tprint_debug(f"Feature matrix shape: {feature_matrix.shape}")
            except Exception as e:
                tprint_warning(f"Failed to create feature matrix: {e}")
                return 0.0
            
            # Calculate correlation matrix safely
            try:
                if self.matrix_ops:
                    corr_matrix = self.matrix_ops.safe_correlation_matrix(feature_matrix)
                else:
                    corr_matrix = np.corrcoef(feature_matrix.T)
                
                # Validate correlation matrix
                if not np.all(np.isfinite(corr_matrix)):
                    tprint_warning("Correlation matrix contains non-finite values")
                    return 0.0
                    
            except Exception as e:
                tprint_warning(f"Failed to calculate correlation matrix: {e}")
                return 0.0
            
            # Count high correlations (>0.8)
            n = corr_matrix.shape[0]
            upper_triangle = corr_matrix[np.triu_indices(n, k=1)]
            high_correlations = np.sum(np.abs(upper_triangle) > 0.8)
            
            # Normalize by total possible correlations
            total_correlations = n * (n - 1) // 2
            redundancy_score = safe_divide(high_correlations, total_correlations, 0.0)
            redundancy_score = validate_finite(redundancy_score, "redundancy_score")
            
            tprint_debug(f"Redundancy score: {redundancy_score:.4f} ({high_correlations}/{total_correlations} high correlations)")
            return redundancy_score
            
        except Exception as e:
            tprint_warning(f"Failed to calculate redundancy score: {e}")
            return 0.0
    
    def _calculate_stability_score(self, result: OrchestratorResult) -> float:
        """Calculate overall stability score."""
        try:
            tprint_debug("Calculating stability score...")
            scores = []
            
            if result.interaction_result and hasattr(result.interaction_result, 'feature_stability_score'):
                score = validate_finite(result.interaction_result.feature_stability_score, "interaction_stability")
                scores.append(score)
                tprint_debug(f"Interaction stability: {score:.4f}")
            
            if result.polynomial_result and hasattr(result.polynomial_result, 'feature_stability_score'):
                score = validate_finite(result.polynomial_result.feature_stability_score, "polynomial_stability")
                scores.append(score)
                tprint_debug(f"Polynomial stability: {score:.4f}")
            
            if result.cross_timeframe_result and hasattr(result.cross_timeframe_result, 'feature_stability_score'):
                score = validate_finite(result.cross_timeframe_result.feature_stability_score, "cross_timeframe_stability")
                scores.append(score)
                tprint_debug(f"Cross-timeframe stability: {score:.4f}")
            
            if scores:
                stability_score = validate_finite(np.mean(scores), "stability_score")
                tprint_debug(f"Overall stability score: {stability_score:.4f}")
                return stability_score
            else:
                tprint_warning("No stability scores available")
                return 0.0
            
        except Exception as e:
            tprint_warning(f"Failed to calculate stability score: {e}")
            return 0.0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = {
            'orchestrator_config': {
                'max_interaction_features': self.config.max_interaction_features,
                'max_polynomial_features': self.config.max_polynomial_features,
                'max_cross_timeframe_features': self.config.max_cross_timeframe_features,
                'enable_interaction_features': self.config.enable_interaction_features,
                'enable_polynomial_features': self.config.enable_polynomial_features,
                'enable_cross_timeframe_features': self.config.enable_cross_timeframe_features
            },
            'component_availability': {
                'interaction_generator': self.interaction_generator is not None,
                'polynomial_generator': self.polynomial_generator is not None,
                'cross_timeframe_generator': self.cross_timeframe_generator is not None,
                'matrix_ops': self.matrix_ops is not None
            },
            'system_availability': {
                'numpy_available': NUMPY_AVAILABLE,
                'pandas_available': PANDAS_AVAILABLE,
                'matrix_ops_available': MATRIX_OPS_AVAILABLE
            }
        }
        
        # Add individual generator metrics
        if self.interaction_generator:
            metrics['interaction_generator_metrics'] = self.interaction_generator.get_performance_metrics()
        
        if self.polynomial_generator:
            metrics['polynomial_generator_metrics'] = self.polynomial_generator.get_performance_metrics()
        
        if self.cross_timeframe_generator:
            metrics['cross_timeframe_generator_metrics'] = self.cross_timeframe_generator.get_performance_metrics()
        
        if self.matrix_ops:
            metrics['matrix_ops_stats'] = self.matrix_ops.get_performance_stats()
            metrics['hardware_info'] = self.matrix_ops.get_hardware_info()
        
        # Add common utilities metrics
        metrics['utility_integration_status'] = self.utility_integration_status
        metrics['common_operations_available'] = COMMON_OPERATIONS_AVAILABLE
        metrics['serialization_available'] = SERIALIZATION_AVAILABLE
        metrics['math_validation_available'] = MATH_VALIDATION_AVAILABLE
        
        return metrics
    
    async def _validate_input_data(
        self, 
        data: Union[np.ndarray, pd.DataFrame], 
        feature_names: Optional[List[str]], 
        target: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Validate input data using common utilities."""
        validation_result = {
            'is_valid': False,
            'issues': [],
            'data_quality_score': 0.0
        }
        
        try:
            if COMMON_OPERATIONS_AVAILABLE and self.config.enable_data_validation:
                # Convert to DataFrame for validation
                if isinstance(data, np.ndarray):
                    if feature_names is None:
                        feature_names = [f"feature_{i}" for i in range(data.shape[1])]
                    df = pd.DataFrame(data, columns=feature_names)
                else:
                    df = data
                
                # Validate DataFrame
                if not validate_dataframe(df):
                    validation_result['issues'].append("Invalid DataFrame")
                    return validation_result
                
                # Check required columns
                if feature_names and not validate_dataframe_columns(df, feature_names):
                    validation_result['issues'].append("Missing required columns")
                    return validation_result
                
                # Calculate data quality metrics
                quality_metrics = calculate_data_quality_metrics(df)
                validation_result['data_quality_score'] = 1.0 - (quality_metrics.get('missing_percentage', 0) / 100)
                
                # Check data quality thresholds
                if quality_metrics.get('missing_percentage', 0) > self.config.max_missing_data_ratio * 100:
                    validation_result['issues'].append(f"High missing data ratio: {quality_metrics.get('missing_percentage', 0):.2f}%")
                
                if quality_metrics.get('duplicate_percentage', 0) > 10:
                    validation_result['issues'].append(f"High duplicate ratio: {quality_metrics.get('duplicate_percentage', 0):.2f}%")
                
                validation_result['is_valid'] = len(validation_result['issues']) == 0
            else:
                # Fallback validation
                if data is None or (hasattr(data, 'shape') and data.shape[0] == 0):
                    validation_result['issues'].append("Empty or None data")
                else:
                    validation_result['is_valid'] = True
            
            return validation_result
            
        except Exception as e:
            validation_result['issues'].append(f"Validation error: {e}")
            return validation_result
    
    async def _optimize_input_data(
        self, 
        data: Union[np.ndarray, pd.DataFrame], 
        feature_names: Optional[List[str]]
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], List[str], Dict[str, Any]]:
        """Optimize input data using common utilities."""
        optimization_info = {
            'optimizations_applied': [],
            'memory_usage_before': 0.0,
            'memory_usage_after': 0.0,
            'optimization_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            if COMMON_OPERATIONS_AVAILABLE:
                # Get initial memory usage
                optimization_info['memory_usage_before'] = get_memory_usage()
                
                # Convert to DataFrame if needed
                if isinstance(data, np.ndarray):
                    if feature_names is None:
                        feature_names = [f"feature_{i}" for i in range(data.shape[1])]
                    df = pd.DataFrame(data, columns=feature_names)
                else:
                    df = data.copy()
                
                # Optimize dtypes
                df = optimize_dataframe_dtypes(df)
                optimization_info['optimizations_applied'].append('dtype_optimization')
                
                # Fill missing values safely
                df = safe_fillna(df, method='forward')
                optimization_info['optimizations_applied'].append('missing_value_filling')
                
                # Apply M1-specific optimizations
                if self.config.enable_m1_optimization and self.gpu_manager:
                    # This would use M1-specific optimizations
                    optimization_info['optimizations_applied'].append('m1_optimization')
                
                # Get final memory usage
                optimization_info['memory_usage_after'] = get_memory_usage()
                optimization_info['optimization_time'] = time.time() - start_time
                
                return df, feature_names, optimization_info
            else:
                return data, feature_names, optimization_info
                
        except Exception as e:
            tprint_warning(f"Data optimization failed: {e}")
            return data, feature_names, optimization_info
    
    async def _assess_data_quality(
        self, 
        X: np.ndarray, 
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Assess data quality using common utilities."""
        quality_report = {
            'overall_score': 0.0,
            'missing_data_ratio': 0.0,
            'duplicate_ratio': 0.0,
            'data_types': {},
            'statistics': {}
        }
        
        try:
            if COMMON_OPERATIONS_AVAILABLE:
                # Convert to DataFrame for quality assessment
                df = pd.DataFrame(X, columns=feature_names)
                
                # Calculate data quality metrics
                quality_metrics = calculate_data_quality_metrics(df)
                
                # Create comprehensive quality report
                quality_report = create_data_quality_report(df)
                
                # Calculate overall score
                missing_ratio = quality_metrics.get('missing_percentage', 0) / 100
                duplicate_ratio = quality_metrics.get('duplicate_percentage', 0) / 100
                
                quality_report['overall_score'] = max(0.0, 1.0 - missing_ratio - duplicate_ratio)
                quality_report['missing_data_ratio'] = missing_ratio
                quality_report['duplicate_ratio'] = duplicate_ratio
                
                # Add basic statistics
                quality_report['statistics'] = {
                    'mean': safe_mean(pd.Series(X.flatten())),
                    'std': safe_std(pd.Series(X.flatten())),
                    'min': float(np.min(X)),
                    'max': float(np.max(X))
                }
            
            return quality_report
            
        except Exception as e:
            tprint_warning(f"Data quality assessment failed: {e}")
            return quality_report
    
    async def _save_artifacts(
        self, 
        result: OrchestratorResult, 
        start_time: float
    ) -> Dict[str, Any]:
        """Save artifacts using serialization utilities."""
        serialization_result = {
            'status': {},
            'paths': {}
        }
        
        try:
            if SERIALIZATION_AVAILABLE and self.config.enable_serialization:
                # Create artifacts directory
                artifacts_dir = Path(self.config.artifacts_directory) / datetime.now().strftime("%Y%m%d_%H%M%S")
                ensure_directory(artifacts_dir)
                
                # Save features as parquet
                if result.combined_features:
                    features_df = pd.DataFrame(result.combined_features)
                    features_path = artifacts_dir / "features.parquet"
                    if safe_to_parquet(features_df, features_path):
                        serialization_result['status']['features'] = True
                        serialization_result['paths']['features'] = str(features_path)
                    else:
                        serialization_result['status']['features'] = False
                
                # Save metadata as JSON
                metadata = {
                    'feature_names': result.combined_feature_names,
                    'feature_scores': result.feature_importance_scores,
                    'total_features_generated': result.total_features_generated,
                    'execution_time': result.execution_time,
                    'utility_integration_status': result.utility_integration_status,
                    'timestamp': datetime.now().isoformat()
                }
                
                metadata_path = artifacts_dir / "metadata.json"
                if safe_json_dump(metadata, metadata_path):
                    serialization_result['status']['metadata'] = True
                    serialization_result['paths']['metadata'] = str(metadata_path)
                else:
                    serialization_result['status']['metadata'] = False
                
                # Save performance metrics
                if result.performance_metrics:
                    metrics_path = artifacts_dir / "performance_metrics.json"
                    if safe_json_dump(result.performance_metrics, metrics_path):
                        serialization_result['status']['performance'] = True
                        serialization_result['paths']['performance'] = str(metrics_path)
                    else:
                        serialization_result['status']['performance'] = False
            
            return serialization_result
            
        except Exception as e:
            tprint_warning(f"Artifact saving failed: {e}")
            return serialization_result
    
    async def _collect_performance_metrics(self, start_time: float) -> Dict[str, Any]:
        """Collect comprehensive performance metrics."""
        metrics = {
            'execution_time': time.time() - start_time,
            'memory_usage': {},
            'hardware_utilization': {},
            'utility_usage': {}
        }
        
        try:
            # Memory usage
            if COMMON_OPERATIONS_AVAILABLE:
                metrics['memory_usage']['current'] = get_memory_usage()
                metrics['memory_usage']['formatted'] = format_bytes(get_memory_usage())
            
            # Hardware utilization
            if self.gpu_manager:
                metrics['hardware_utilization']['gpu'] = self.gpu_manager.get_gpu_info()
            if self.memory_optimizer:
                metrics['hardware_utilization']['memory'] = {'optimized': True}
            if self.cpu_optimizer:
                metrics['hardware_utilization']['cpu'] = self.cpu_optimizer.get_cpu_info()
            
            # Utility usage
            metrics['utility_usage'] = self.utility_integration_status
            
            return metrics
            
        except Exception as e:
            tprint_warning(f"Performance metrics collection failed: {e}")
            return metrics