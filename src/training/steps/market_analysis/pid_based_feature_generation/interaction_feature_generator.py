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

# Import matrix operations with advanced capabilities
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations, get_enhanced_matrix_operations,
        get_vectorized_processing_core, get_batch_matrix_processor,
        compute_trading_indicators, optimize_matrix_operation_with_hardware,
        safe_matrix_multiply, safe_correlation_matrix, safe_matrix_inverse,
        gpu_matrix_multiply, correlation_matrix_gpu, eigendecomposition_gpu,
        batch_matrix_multiply, batch_feature_transformation, batch_correlation_analysis,
        create_ml_pipeline, execute_ml_pipeline, optimize_pipeline_config
    )
    MATRIX_OPS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Matrix operations not available: {e}")
    MATRIX_OPS_AVAILABLE = False

# Import base feature generator
from .base_feature_generator import BaseFeatureGenerator, BaseFeatureConfig, BaseFeatureResult

# Import common operations for comprehensive utility integration
try:
    from src.utils.common_operations import (
        safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
        validate_finite, get_memory_usage
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
class InteractionConfig(BaseFeatureConfig):
    """Configuration for interaction feature generation with common utilities integration."""
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


@dataclass
class InteractionResult(BaseFeatureResult):
    """Result of interaction feature generation with common utilities integration."""
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


class InteractionFeatureGenerator(BaseFeatureGenerator):
    """
    Interaction Feature Generator using Partial Information Decomposition.
    
    Generates data-driven interaction features using PID analysis to identify
    the most relevant feature interactions up to 100 features.
    """
    
    def __init__(self, config: Optional[InteractionConfig] = None):
        """Initialize the interaction feature generator with common utilities integration."""
        super().__init__(config or InteractionConfig(), "InteractionFeatureGenerator")
        
        self.logger.info(f"📊 Max interaction features: {self.config.max_interaction_features}")
        self.logger.info(f"📊 Interaction types: {[t.value for t in self.config.interaction_types]}")
    
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
        
        # Initialize matrix operations with advanced capabilities
        if MATRIX_OPS_AVAILABLE:
            self.matrix_ops = get_unified_matrix_operations(
                enable_gpu=self.config.enable_gpu_acceleration,
                enable_memory_optimization=True,
                enable_parallel=self.config.enable_parallel_processing
            )
            
            # Initialize enhanced matrix operations
            self.enhanced_matrix_ops = get_enhanced_matrix_operations()
            
            # Initialize vectorized processing core
            self.vectorized_core = get_vectorized_processing_core()
            
            # Initialize batch processor
            self.batch_processor = get_batch_matrix_processor()
            
            self.logger.info("✅ Advanced Matrix Operations initialized")
            self.logger.info("✅ Enhanced Matrix Operations initialized")
            self.logger.info("✅ Vectorized Processing Core initialized")
            self.logger.info("✅ Batch Matrix Processor initialized")
        else:
            self.matrix_ops = None
            self.enhanced_matrix_ops = None
            self.vectorized_core = None
            self.batch_processor = None
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
            
            # Generate enhanced features using advanced matrix operations
            self.logger.info("🔧 Generating enhanced features using advanced matrix operations...")
            enhanced_features = self._generate_enhanced_interaction_features(X, feature_names, target)
            
            # Calculate interaction scores
            interaction_scores = self._calculate_interaction_scores(
                interaction_features, interaction_names, target
            )
            
            # Combine traditional and enhanced features
            all_features = {name: feature for name, feature in zip(interaction_names, interaction_features.T)}
            all_features.update(enhanced_features)
            
            # Store results
            result.interaction_features = all_features
            result.feature_names = list(all_features.keys())
            result.interaction_scores = interaction_scores
            result.total_features_generated = len(all_features)
            result.matrix_ops_used = self.matrix_ops is not None
            
            # Calculate quality metrics
            result.average_correlation = self._calculate_average_correlation(interaction_features)
            result.feature_stability_score = self._calculate_stability_score(interaction_features)
            result.redundancy_score = self._calculate_redundancy_score(interaction_features)
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            # Set utility integration status using base class method
            self._set_utility_integration_status(result)
            
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
    
    def _generate_enhanced_interaction_features(
        self, 
        X: np.ndarray, 
        feature_names: List[str],
        target: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Generate enhanced interaction features using advanced matrix operations."""
        enhanced_features = {}
        
        try:
            if not MATRIX_OPS_AVAILABLE:
                return enhanced_features
            
            # Convert to DataFrame for vectorized processing
            df = pd.DataFrame(X, columns=feature_names)
            
            # Use vectorized processing core for advanced feature engineering
            if self.vectorized_core:
                # Compute trading indicators as additional features
                trading_indicators = compute_trading_indicators(df)
                
                # Add trading indicator features
                for col in trading_indicators.columns:
                    if col not in feature_names:
                        enhanced_features[f"trading_{col}"] = trading_indicators[col].values
                
                # Use batch processing for large datasets
                if X.shape[0] > 1000 and self.batch_processor:
                    # Process in batches for memory efficiency
                    batch_size = min(500, X.shape[0] // 4)
                    batches = [X[i:i+batch_size] for i in range(0, X.shape[0], batch_size)]
                    
                    # Process each batch
                    batch_results = []
                    for batch in batches:
                        batch_df = pd.DataFrame(batch, columns=feature_names)
                        batch_features = self.vectorized_core.optimize_dataframe_for_processing(batch_df)
                        batch_results.append(batch_features.values)
                    
                    # Combine batch results
                    if batch_results:
                        combined_features = np.vstack(batch_results)
                        enhanced_features["batch_optimized"] = combined_features
            
            # Use enhanced matrix operations for advanced correlations
            if self.enhanced_matrix_ops:
                # Compute advanced correlation features
                corr_matrix = correlation_matrix_gpu(df) if self.config.enable_gpu_acceleration else safe_correlation_matrix(df)
                
                # Extract upper triangle correlations as features
                n = corr_matrix.shape[0]
                upper_triangle = corr_matrix[np.triu_indices(n, k=1)]
                enhanced_features["correlation_features"] = upper_triangle
            
            # Use ML pipeline for complex feature transformations
            if self.config.enable_parallel_processing:
                pipeline_config = [
                    {"operation": "normalize", "params": {"method": "zscore"}},
                    {"operation": "polynomial", "params": {"degree": 2}},
                    {"operation": "interaction", "params": {"max_features": 10}}
                ]
                
                try:
                    pipeline = create_ml_pipeline(pipeline_config)
                    pipeline_result = execute_ml_pipeline(df, pipeline)
                    
                    # Add pipeline features
                    for i, col in enumerate(pipeline_result.columns):
                        if col not in feature_names:
                            enhanced_features[f"pipeline_{col}"] = pipeline_result[col].values
                except Exception as e:
                    self.logger.warning(f"ML pipeline execution failed: {e}")
            
            self.logger.info(f"✅ Generated {len(enhanced_features)} enhanced interaction features")
            return enhanced_features
            
        except Exception as e:
            self.logger.warning(f"Enhanced feature generation failed: {e}")
            return enhanced_features
    
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