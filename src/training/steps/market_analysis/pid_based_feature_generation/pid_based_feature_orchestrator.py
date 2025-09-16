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

# Import math validation for safe operations
try:
    from src.utils.math_validation import MathValidation, safe_divide, safe_log, safe_sqrt, safe_power, validate_finite
    MATH_VALIDATION_AVAILABLE = True
except ImportError:
    MATH_VALIDATION_AVAILABLE = False
    # Fallback functions
    def safe_divide(a, b, default=0.0): return a / b if b != 0 else default
    def safe_log(x, default=0.0): return np.log(x) if x > 0 else default
    def safe_sqrt(x, default=0.0): return np.sqrt(x) if x >= 0 else default
    def safe_power(x, y, default=0.0): return x ** y if np.isfinite(x) and np.isfinite(y) else default
    def validate_finite(value, name="value"): return float(value) if np.isfinite(value) else 0.0

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
    """Configuration for PID-based feature orchestrator."""
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


@dataclass
class OrchestratorResult:
    """Result of PID-based feature orchestration."""
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


class PIDBasedFeatureOrchestrator:
    """
    PID-Based Feature Orchestrator.
    
    Orchestrates all PID-based feature generation processes, integrating
    interaction, polynomial, and cross-timeframe feature generation.
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """Initialize the PID-based feature orchestrator."""
        try:
            # Input validation
            if config is not None and not isinstance(config, OrchestratorConfig):
                raise TypeError(f"Config must be OrchestratorConfig or None, got {type(config)}")
            
            self.config = config or OrchestratorConfig()
            self.logger = logger.getChild('PIDBasedFeatureOrchestrator')
            
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
            tprint_info(f"tprint available: {TPRINT_AVAILABLE}")
            tprint_info(f"Math validation available: {MATH_VALIDATION_AVAILABLE}")
            
        except Exception as e:
            tprint_error(f"Failed to initialize PIDBasedFeatureOrchestrator: {e}")
            raise
    
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
            # Fast-fail input validation
            if data is None:
                raise ValueError("Data cannot be None - fast failing")
            
            if feature_names is None or len(feature_names) == 0:
                raise ValueError("Feature names cannot be None or empty - fast failing")
            
            # Convert data to numpy array if needed
            if isinstance(data, pd.DataFrame):
                if data.empty:
                    raise ValueError("Input DataFrame is empty - fast failing")
                X = data.values
                if feature_names is None:
                    feature_names = list(data.columns)
                tprint_info(f"Converted DataFrame to numpy array: {X.shape}")
            else:
                if not hasattr(data, 'shape'):
                    raise TypeError(f"Data must be array-like, got {type(data)} - fast failing")
                X = data
                tprint_info(f"Using numpy array data: {X.shape}")
            
            # Validate data shape
            if X.shape[0] == 0:
                raise ValueError("Input data has no samples - fast failing")
            if X.shape[1] == 0:
                raise ValueError("Input data has no features - fast failing")
            
            # Check for NaN/Inf values
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
            
            tprint_performance("PID-based feature orchestration", execution_time)
            tprint_info(f"Generated {result.total_features_generated} total features")
            tprint_info(f"Overall quality score: {result.overall_quality_score:.3f}")
            tprint_info(f"Feature diversity score: {result.feature_diversity_score:.3f}")
            tprint_info(f"Generation status: {result.generation_status.value}")
            
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
        
        return metrics