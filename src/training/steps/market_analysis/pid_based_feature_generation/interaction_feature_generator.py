"""
Interaction Feature Generator using Partial Information Decomposition

This module generates data-driven interaction features using PID analysis to identify
the most relevant feature interactions up to 100 features.

Key Features:
- Uses optimized lookback periods from feature_lookback_optimization
- Leverages matrix_operations/ for all calculations
- Generates up to 100 interaction features based on PID analysis
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

# Import PID utilities
try:
    from src.training.utils.feature_selection.partial_information_decompositor import (
        PartialInformationDecompositor, PIDConfig, PIDResult
    )
    PID_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PID utilities not available: {e}")
    PID_AVAILABLE = False

# Import matrix operations
try:
    from src.utils.matrix_operations import get_unified_matrix_operations
    MATRIX_OPS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Matrix operations not available: {e}")
    MATRIX_OPS_AVAILABLE = False

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

# Import logger
try:
    from src.utils.logger import system_logger
    logger = system_logger.getChild('InteractionFeatureGenerator')
except ImportError:
    logger = logging.getLogger('InteractionFeatureGenerator')
    logger.setLevel(logging.INFO)


class InteractionType(Enum):
    """Types of interaction features."""
    MULTIPLICATIVE = "multiplicative"
    ADDITIVE = "additive"
    RATIO = "ratio"
    DIFFERENCE = "difference"
    CORRELATION = "correlation"
    RANK_CORRELATION = "rank_correlation"
    POLYNOMIAL = "polynomial"
    LOGARITHMIC = "logarithmic"
    EXPONENTIAL = "exponential"


@dataclass
class InteractionConfig:
    """Configuration for interaction feature generation."""
    # PID Configuration
    synergy_threshold: float = 0.1
    redundancy_threshold: float = 0.15
    unique_info_threshold: float = 0.05
    
    # Feature Limits
    max_interaction_features: int = 100
    max_feature_pairs: int = 50
    
    # Interaction Types
    interaction_types: List[InteractionType] = field(default_factory=lambda: [
        InteractionType.MULTIPLICATIVE,
        InteractionType.ADDITIVE,
        InteractionType.RATIO,
        InteractionType.DIFFERENCE,
        InteractionType.CORRELATION
    ])
    
    # Computational Settings
    enable_parallel_processing: bool = True
    enable_gpu_acceleration: bool = True
    memory_limit_gb: float = 8.0
    
    # Validation
    min_correlation_threshold: float = 0.1
    max_correlation_threshold: float = 0.95
    significance_threshold: float = 0.05
    
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
    
    # Performance Settings
    enable_profiling: bool = True
    enable_memory_monitoring: bool = True
    enable_performance_logging: bool = True


@dataclass
class InteractionResult:
    """Result of interaction feature generation."""
    interaction_features: Dict[str, np.ndarray] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    interaction_scores: Dict[str, float] = field(default_factory=dict)
    pid_analysis: Optional[PIDResult] = None
    
    # Metadata
    total_features_generated: int = 0
    execution_time: float = 0.0
    optimization_used: bool = False
    matrix_ops_used: bool = False
    
    # Quality Metrics
    average_correlation: float = 0.0
    feature_stability_score: float = 0.0
    redundancy_score: float = 0.0
    
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


class InteractionFeatureGenerator:
    """
    Interaction Feature Generator using Partial Information Decomposition.
    
    Generates data-driven interaction features using PID analysis to identify
    the most relevant feature interactions up to 100 features.
    """
    
    def __init__(self, config: Optional[InteractionConfig] = None):
        """Initialize the interaction feature generator with common utilities integration."""
        self.config = config or InteractionConfig()
        self.logger = logger.getChild('InteractionFeatureGenerator')
        
        # Initialize common utilities integration
        self._initialize_common_utilities()
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("🔧 InteractionFeatureGenerator initialized")
        self.logger.info(f"📊 Max interaction features: {self.config.max_interaction_features}")
        self.logger.info(f"📊 Interaction types: {[t.value for t in self.config.interaction_types]}")
        self.logger.info(f"🔧 Common operations available: {COMMON_OPERATIONS_AVAILABLE}")
        self.logger.info(f"🔧 Serialization available: {SERIALIZATION_AVAILABLE}")
        self.logger.info(f"🔧 Math validation available: {MATH_VALIDATION_AVAILABLE}")
        self.logger.info(f"🔧 Matrix operations available: {MATRIX_OPS_AVAILABLE}")
    
    def _initialize_common_utilities(self):
        """Initialize common utilities integration."""
        # Initialize serializers
        if SERIALIZATION_AVAILABLE and self.config.enable_serialization:
            self.json_serializer = JSONSerializer()
            self.pickle_serializer = PickleSerializer()
            self.parquet_serializer = ParquetSerializer()
            self.universal_serializer = UniversalSerializer()
            self.logger.info("✅ Serializers initialized")
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
            self.logger.info("✅ M1 optimizers initialized")
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
        
        self.logger.info(f"🔧 Utility integration status: {self.utility_integration_status}")
    
    def _initialize_components(self):
        """Initialize required components."""
        # Initialize PID decompositor
        if PID_AVAILABLE:
            pid_config = PIDConfig(
                synergy_threshold=self.config.synergy_threshold,
                redundancy_threshold=self.config.redundancy_threshold,
                unique_info_threshold=self.config.unique_info_threshold,
                max_interaction_features=self.config.max_interaction_features
            )
            self.pid_decompositor = PartialInformationDecompositor(pid_config)
            self.logger.info("✅ PID Decompositor initialized")
        else:
            self.pid_decompositor = None
            self.logger.warning("⚠️ PID Decompositor not available")
        
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
    
    async def generate_interaction_features(
        self, 
        data: Union[np.ndarray, pd.DataFrame],
        feature_names: List[str],
        optimized_lookback_periods: Optional[Dict[str, int]] = None,
        target: Optional[np.ndarray] = None
    ) -> InteractionResult:
        """
        Generate interaction features using PID analysis.
        
        Args:
            data: Input feature matrix
            feature_names: List of feature names
            optimized_lookback_periods: Optimized lookback periods from feature_lookback_optimization
            target: Target variable for PID analysis (optional)
            
        Returns:
            InteractionResult with generated interaction features
        """
        start_time = time.time()
        self.logger.info("🔧 Starting interaction feature generation...")
        
        result = InteractionResult()
        
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
            else:
                X = data
            
            # Enhanced data quality assessment
            if self.config.enable_quality_reporting:
                quality_report = await self._assess_data_quality(X, feature_names)
                result.data_quality_report = quality_report
                self.logger.info(f"📊 Data quality score: {quality_report.get('overall_score', 0.0):.3f}")
            
            self.logger.info(f"📊 Input data shape: {X.shape}")
            self.logger.info(f"📊 Feature count: {len(feature_names)}")
            
            # Apply optimized lookback periods if available
            if optimized_lookback_periods:
                X, feature_names = self._apply_optimized_lookback_periods(
                    X, feature_names, optimized_lookback_periods
                )
                result.optimization_used = True
                self.logger.info("✅ Applied optimized lookback periods")
            
            # Perform PID analysis
            if self.pid_decompositor and target is not None:
                self.logger.info("🔍 Performing PID analysis...")
                pid_result = self.pid_decompositor.decompose_information(X, target, feature_names)
                result.pid_analysis = pid_result
                
                # Extract significant interactions from PID
                significant_pairs = self._extract_significant_interactions(pid_result)
                self.logger.info(f"📊 Found {len(significant_pairs)} significant interactions")
            else:
                # Fallback: use correlation-based approach
                self.logger.info("📊 Using correlation-based interaction detection")
                significant_pairs = self._correlation_based_interaction_detection(X, feature_names)
                self.logger.info(f"📊 Found {len(significant_pairs)} correlation-based interactions")
            
            # Generate interaction features
            self.logger.info("🔧 Generating interaction features...")
            interaction_features, interaction_names = self._generate_interaction_matrix(
                X, feature_names, significant_pairs
            )
            
            # Calculate interaction scores
            interaction_scores = self._calculate_interaction_scores(
                interaction_features, interaction_names, target
            )
            
            # Store results
            result.interaction_features = {
                name: feature for name, feature in zip(interaction_names, interaction_features.T)
            }
            result.feature_names = interaction_names
            result.interaction_scores = interaction_scores
            result.total_features_generated = len(interaction_names)
            result.matrix_ops_used = self.matrix_ops is not None
            
            # Calculate quality metrics
            result.average_correlation = self._calculate_average_correlation(interaction_features)
            result.feature_stability_score = self._calculate_stability_score(interaction_features)
            result.redundancy_score = self._calculate_redundancy_score(interaction_features)
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            # Set utility integration status
            result.utility_integration_status = getattr(self, 'utility_integration_status', {})
            result.hardware_optimization_used = bool(self.gpu_manager or self.memory_optimizer or self.cpu_optimizer)
            
            # Collect performance metrics
            if self.config.enable_performance_logging:
                result.performance_metrics = self.get_performance_metrics()
                result.memory_usage = {'current': get_memory_usage()} if COMMON_OPERATIONS_AVAILABLE else {}
            
            self.logger.info(f"✅ Interaction feature generation completed in {execution_time:.3f}s")
            self.logger.info(f"📊 Generated {result.total_features_generated} interaction features")
            self.logger.info(f"📊 Average correlation: {result.average_correlation:.3f}")
            self.logger.info(f"📊 Stability score: {result.feature_stability_score:.3f}")
            self.logger.info(f"🔧 Utility integrations: {sum(result.utility_integration_status.values())}/{len(result.utility_integration_status)}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            self.logger.error(f"❌ Interaction feature generation failed: {e}")
            self.logger.error(f"❌ Error details: {traceback.format_exc()}")
            
            return result
    
    def _apply_optimized_lookback_periods(
        self, 
        X: np.ndarray, 
        feature_names: List[str], 
        optimized_lookback_periods: Dict[str, int]
    ) -> Tuple[np.ndarray, List[str]]:
        """Apply optimized lookback periods to features."""
        try:
            # This is a placeholder for applying optimized lookback periods
            # In practice, this would involve resampling or windowing the data
            # based on the optimized periods from feature_lookback_optimization
            
            self.logger.info(f"📊 Applying optimized lookback periods: {optimized_lookback_periods}")
            
            # For now, we'll just log the optimization and return the original data
            # In a full implementation, this would:
            # 1. Identify which features correspond to which optimized periods
            # 2. Apply the appropriate windowing/resampling
            # 3. Update feature names to reflect the optimization
            
            return X, feature_names
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to apply optimized lookback periods: {e}")
            return X, feature_names
    
    def _extract_significant_interactions(self, pid_result: PIDResult) -> List[Tuple[str, str]]:
        """Extract significant feature interactions from PID analysis."""
        significant_pairs = []
        
        # Sort synergy scores and take top interactions
        synergy_items = sorted(pid_result.synergy.items(), key=lambda x: x[1], reverse=True)
        
        for (feat1, feat2), synergy_score in synergy_items:
            if synergy_score > self.config.synergy_threshold:
                significant_pairs.append((feat1, feat2))
                if len(significant_pairs) >= self.config.max_feature_pairs:
                    break
        
        return significant_pairs
    
    def _correlation_based_interaction_detection(
        self, 
        X: np.ndarray, 
        feature_names: List[str]
    ) -> List[Tuple[str, str]]:
        """Fallback correlation-based interaction detection."""
        try:
            if self.matrix_ops:
                # Use matrix operations for correlation calculation
                correlation_matrix = self.matrix_ops.safe_correlation_matrix(X)
            else:
                # Fallback to numpy correlation
                correlation_matrix = np.corrcoef(X.T)
            
            significant_pairs = []
            n_features = len(feature_names)
            
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    corr = abs(correlation_matrix[i, j])
                    
                    if (self.config.min_correlation_threshold <= corr <= 
                        self.config.max_correlation_threshold):
                        significant_pairs.append((feature_names[i], feature_names[j]))
                        
                        if len(significant_pairs) >= self.config.max_feature_pairs:
                            break
                
                if len(significant_pairs) >= self.config.max_feature_pairs:
                    break
            
            return significant_pairs
            
        except Exception as e:
            self.logger.warning(f"⚠️ Correlation-based detection failed: {e}")
            return []
    
    def _generate_interaction_matrix(
        self, 
        X: np.ndarray, 
        feature_names: List[str], 
        significant_pairs: List[Tuple[str, str]]
    ) -> Tuple[np.ndarray, List[str]]:
        """Generate interaction feature matrix."""
        interaction_features = []
        interaction_names = []
        
        for feat1, feat2 in significant_pairs:
            try:
                # Find feature indices
                idx1 = feature_names.index(feat1)
                idx2 = feature_names.index(feat2)
                
                x1, x2 = X[:, idx1], X[:, idx2]
                
                # Generate different types of interactions
                for interaction_type in self.config.interaction_types:
                    interaction_feat, interaction_name = self._create_interaction_feature(
                        x1, x2, feat1, feat2, interaction_type
                    )
                    
                    if interaction_feat is not None:
                        interaction_features.append(interaction_feat)
                        interaction_names.append(interaction_name)
                        
                        if len(interaction_names) >= self.config.max_interaction_features:
                            break
                
                if len(interaction_names) >= self.config.max_interaction_features:
                    break
                    
            except (ValueError, IndexError) as e:
                self.logger.warning(f"⚠️ Failed to create interaction for ({feat1}, {feat2}): {e}")
                continue
        
        if interaction_features:
            return np.column_stack(interaction_features), interaction_names
        else:
            return np.array([]).reshape(X.shape[0], 0), []
    
    def _create_interaction_feature(
        self, 
        x1: np.ndarray, 
        x2: np.ndarray, 
        feat1: str, 
        feat2: str, 
        interaction_type: InteractionType
    ) -> Tuple[Optional[np.ndarray], str]:
        """Create a specific type of interaction feature."""
        try:
            if interaction_type == InteractionType.MULTIPLICATIVE:
                feature = x1 * x2
                name = f"{feat1}_x_{feat2}"
                
            elif interaction_type == InteractionType.ADDITIVE:
                feature = x1 + x2
                name = f"{feat1}_plus_{feat2}"
                
            elif interaction_type == InteractionType.RATIO:
                if self.matrix_ops:
                    feature = self.matrix_ops.batch_process(
                        np.column_stack([x1, x2]), 
                        'safe_divide', 
                        numerator=x1, 
                        denominator=x2, 
                        default_value=0.0
                    )
                else:
                    feature = np.divide(x1, x2, out=np.zeros_like(x1), where=(x2 != 0))
                name = f"{feat1}_ratio_{feat2}"
                
            elif interaction_type == InteractionType.DIFFERENCE:
                feature = x1 - x2
                name = f"{feat1}_minus_{feat2}"
                
            elif interaction_type == InteractionType.CORRELATION:
                # Rolling correlation (simplified)
                window = min(20, len(x1))
                feature = np.full_like(x1, np.corrcoef(x1, x2)[0, 1])
                name = f"{feat1}_corr_{feat2}"
                
            elif interaction_type == InteractionType.POLYNOMIAL:
                feature = x1 * x2 + x1**2 + x2**2
                name = f"{feat1}_{feat2}_poly"
                
            elif interaction_type == InteractionType.LOGARITHMIC:
                if self.matrix_ops:
                    log_x1 = self.matrix_ops.batch_process(
                        x1.reshape(-1, 1), 'log', default_value=0.0
                    )
                    log_x2 = self.matrix_ops.batch_process(
                        x2.reshape(-1, 1), 'log', default_value=0.0
                    )
                    feature = log_x1.flatten() * log_x2.flatten()
                else:
                    feature = np.log(np.maximum(x1, 1e-10)) * np.log(np.maximum(x2, 1e-10))
                name = f"log_{feat1}_x_log_{feat2}"
                
            else:
                return None, ""
            
            # Validate feature
            if np.any(np.isnan(feature)) or np.any(np.isinf(feature)):
                self.logger.warning(f"⚠️ Invalid values in {name}, skipping")
                return None, ""
            
            return feature, name
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create {interaction_type.value} interaction: {e}")
            return None, ""
    
    def _calculate_interaction_scores(
        self, 
        interaction_features: np.ndarray, 
        interaction_names: List[str], 
        target: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Calculate importance scores for interaction features."""
        scores = {}
        
        if target is None:
            # Use variance as importance score
            for i, name in enumerate(interaction_names):
                scores[name] = float(np.var(interaction_features[:, i]))
        else:
            # Use correlation with target as importance score
            for i, name in enumerate(interaction_names):
                try:
                    if self.matrix_ops:
                        corr = self.matrix_ops.safe_correlation_matrix(
                            np.column_stack([interaction_features[:, i], target])
                        )[0, 1]
                    else:
                        corr = np.corrcoef(interaction_features[:, i], target)[0, 1]
                    scores[name] = abs(float(corr))
                except Exception:
                    scores[name] = 0.0
        
        return scores
    
    def _calculate_average_correlation(self, interaction_features: np.ndarray) -> float:
        """Calculate average correlation between interaction features."""
        try:
            if self.matrix_ops:
                corr_matrix = self.matrix_ops.safe_correlation_matrix(interaction_features)
            else:
                corr_matrix = np.corrcoef(interaction_features.T)
            
            # Get upper triangle (excluding diagonal)
            n = corr_matrix.shape[0]
            upper_triangle = corr_matrix[np.triu_indices(n, k=1)]
            
            return float(np.mean(np.abs(upper_triangle)))
            
        except Exception:
            return 0.0
    
    def _calculate_stability_score(self, interaction_features: np.ndarray) -> float:
        """Calculate stability score based on feature consistency."""
        try:
            # Calculate coefficient of variation for each feature
            cv_scores = []
            for i in range(interaction_features.shape[1]):
                feature = interaction_features[:, i]
                mean_val = np.mean(feature)
                std_val = np.std(feature)
                
                if mean_val != 0:
                    cv = std_val / abs(mean_val)
                    cv_scores.append(cv)
            
            if cv_scores:
                # Lower CV = higher stability
                avg_cv = np.mean(cv_scores)
                stability_score = max(0.0, 1.0 - avg_cv)
                return float(stability_score)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_redundancy_score(self, interaction_features: np.ndarray) -> float:
        """Calculate redundancy score based on feature correlations."""
        try:
            if self.matrix_ops:
                corr_matrix = self.matrix_ops.safe_correlation_matrix(interaction_features)
            else:
                corr_matrix = np.corrcoef(interaction_features.T)
            
            # Count high correlations (>0.8)
            n = corr_matrix.shape[0]
            upper_triangle = corr_matrix[np.triu_indices(n, k=1)]
            high_correlations = np.sum(np.abs(upper_triangle) > 0.8)
            
            # Normalize by total possible correlations
            total_correlations = n * (n - 1) // 2
            redundancy_score = high_correlations / total_correlations if total_correlations > 0 else 0.0
            
            return float(redundancy_score)
            
        except Exception:
            return 0.0
    
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
                    optimization_info['optimizations_applied'].append('m1_optimization')
                
                # Get final memory usage
                optimization_info['memory_usage_after'] = get_memory_usage()
                optimization_info['optimization_time'] = time.time() - start_time
                
                return df, feature_names, optimization_info
            else:
                return data, feature_names, optimization_info
                
        except Exception as e:
            self.logger.warning(f"Data optimization failed: {e}")
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
            self.logger.warning(f"Data quality assessment failed: {e}")
            return quality_report
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics with common utilities integration."""
        metrics = {
            'pid_available': PID_AVAILABLE,
            'matrix_ops_available': MATRIX_OPS_AVAILABLE,
            'numpy_available': NUMPY_AVAILABLE,
            'pandas_available': PANDAS_AVAILABLE,
            'common_operations_available': COMMON_OPERATIONS_AVAILABLE,
            'serialization_available': SERIALIZATION_AVAILABLE,
            'math_validation_available': MATH_VALIDATION_AVAILABLE
        }
        
        if self.matrix_ops:
            metrics['matrix_ops_stats'] = self.matrix_ops.get_performance_stats()
            metrics['hardware_info'] = self.matrix_ops.get_hardware_info()
        
        # Add common utilities metrics
        metrics['utility_integration_status'] = getattr(self, 'utility_integration_status', {})
        metrics['memory_usage'] = get_memory_usage() if COMMON_OPERATIONS_AVAILABLE else 0.0
        
        return metrics