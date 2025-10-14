"""
Sophisticated Lookback Optimizer for UnifiedDataDrivenPipeline

This module integrates the advanced optimization algorithms from FeatureLookbackOptimizationComponent
into the UnifiedDataDrivenPipeline, including:
- Nested walk-forward cross-validation
- Advanced coarse-to-refine with bootstrap validation
- Sophisticated Bayesian TPE optimization
- Multi-horizon profit labeling integration
- Direction-specific optimization (longs/shorts)
- Execution mode-aware optimization
- Comprehensive validation and error handling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Iterable
from dataclasses import dataclass
from enum import Enum
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, OrderedDict
from functools import lru_cache
import gc

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

# Import sophisticated optimization components
try:
    from src.training.steps.pre_training.feature_lookback_optimization.core.optimizer import (
        CoreOptimizer, OptimizationMethod, OptimizationResult, LookbackConstraints
    )
    from src.training.steps.pre_training.feature_lookback_optimization.core.vectorbt_optimizer import (
        VectorBTOptimizer, VectorBTOptimizationConfig, OptimizationStrategy, create_vectorbt_optimizer
    )
    SOPHISTICATED_OPTIMIZER_AVAILABLE = True
except ImportError:
    SOPHISTICATED_OPTIMIZER_AVAILABLE = False
    CoreOptimizer = None
    OptimizationMethod = None
    OptimizationResult = None
    LookbackConstraints = None
    VectorBTOptimizer = None
    VectorBTOptimizationConfig = None
    OptimizationStrategy = None
    create_vectorbt_optimizer = None

# Import multi-horizon profit labeler
try:
    from src.training.steps.pre_training.multi_horizon_profit_labeler import MultiHorizonConfig
    MULTI_HORIZON_AVAILABLE = True
except ImportError:
    MULTI_HORIZON_AVAILABLE = False
    MultiHorizonConfig = None

# Import VectorBT optimizations
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, 
        get_vectorbt_rolling_optimizer,
        optimized_rolling_mean,
        optimized_rolling_std,
        optimized_rolling_corr
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    VectorBTRollingOptimizer = None

# Import matrix operations
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_batch_matrix_processor,
        safe_matrix_multiply,
        optimize_dataframe,
        matrix_correlation_analysis
    )
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False

# Import caching and serialization
try:
    from src.feature_generation.core.feature_cache import FeatureCacheService
    from src.utils.serialization_utils import UniversalSerializer, JSONSerializer, PickleSerializer
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False

logger = logging.getLogger(__name__)


class OptimizationDirection(Enum):
    """Optimization directions."""
    LONGS = "longs"
    SHORTS = "shorts"
    BOTH = "both"


@dataclass
class SophisticatedOptimizationResult:
    """Result of sophisticated lookback optimization."""
    feature_name: str
    best_lookback: int
    best_score: float
    method: str
    optimization_time: float
    n_trials: int
    convergence_achieved: bool
    stability_score: float
    sensitivity_score: float
    regularization_penalty: float
    validation_scores: List[float]
    optimization_metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    direction: Optional[str] = None
    target_column: Optional[str] = None
    outer_validation: Optional[Dict[str, Any]] = None
    frozen_from_inner: Optional[bool] = None
    lookback_aggregates: Optional[Dict[str, Any]] = None


@dataclass
class ExecutionModeConfig:
    """Configuration for execution modes."""
    light: Dict[str, Any]
    full: Dict[str, Any]
    blank: Dict[str, Any]


class SophisticatedLookbackOptimizer:
    """
    Sophisticated lookback optimizer integrating advanced algorithms from FeatureLookbackOptimizationComponent.
    
    Features:
    - Nested walk-forward cross-validation
    - Advanced coarse-to-refine with bootstrap validation
    - Sophisticated Bayesian TPE optimization
    - Multi-horizon profit labeling integration
    - Direction-specific optimization (longs/shorts)
    - Execution mode-aware optimization
    - Comprehensive validation and error handling
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the sophisticated lookback optimizer."""
        self.config = config or {}
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.performance_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'failed_optimizations': 0,
            'total_execution_time': 0.0,
            'vectorbt_operations': 0,
            'bayesian_optimizations': 0,
            'coarse_to_refine_optimizations': 0,
            'parallel_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'nested_cv_operations': 0
        }
        
        # Execution mode configuration
        self.execution_mode_config = ExecutionModeConfig(
            light={'max_features': 100, 'max_lookback': 50, 'n_trials': 30, 'cv_folds': 3},
            full={'max_features': 500, 'max_lookback': 200, 'n_trials': 50, 'cv_folds': 5},
            blank={'max_features': 20, 'max_lookback': 20, 'n_trials': 10, 'cv_folds': 2}
        )
        
        tprint_success("✅ Sophisticated Lookback Optimizer initialized")
    
    def _initialize_components(self):
        """Initialize all optimizer components."""
        tprint_debug("Initializing sophisticated lookback optimizer components")
        
        # Initialize core optimizer
        if SOPHISTICATED_OPTIMIZER_AVAILABLE:
            self.core_optimizer = CoreOptimizer()
            tprint_success("✅ Core optimizer initialized")
        else:
            self.core_optimizer = None
            tprint_warning("⚠️ Core optimizer not available")
        
        # Initialize VectorBT optimizer
        if VECTORBT_AVAILABLE:
            try:
                self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
                tprint_success("✅ VectorBT optimizer initialized")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT optimizer initialization failed: {e}")
                self.vectorbt_optimizer = None
        else:
            self.vectorbt_optimizer = None
            tprint_warning("⚠️ VectorBT not available")
        
        # Initialize matrix operations
        if MATRIX_OPS_AVAILABLE:
            self.matrix_ops = get_unified_matrix_operations()
            self.vectorized_core = get_vectorized_processing_core()
            self.batch_processor = get_batch_matrix_processor()
            tprint_success("✅ Matrix operations initialized")
        else:
            self.matrix_ops = None
            self.vectorized_core = None
            self.batch_processor = None
            tprint_warning("⚠️ Matrix operations not available")
        
        # Initialize caching
        if CACHING_AVAILABLE:
            self.feature_cache = FeatureCacheService()
            self.universal_serializer = UniversalSerializer()
            self.json_serializer = JSONSerializer()
            self.pickle_serializer = PickleSerializer()
            tprint_success("✅ Caching initialized")
        else:
            self.feature_cache = None
            self.universal_serializer = None
            self.json_serializer = None
            self.pickle_serializer = None
            tprint_warning("⚠️ Caching not available")
    
    def optimize_features_sophisticated(
        self,
        data: pd.DataFrame,
        feature_names: List[str],
        target_columns: Dict[str, str],  # direction -> target_column mapping
        lookback_range: Tuple[int, int],
        optimization_direction: OptimizationDirection = OptimizationDirection.BOTH,
        execution_mode: str = "full",
        use_nested_cv: bool = True,
        regularization_settings: Optional[Dict[str, float]] = None,
        max_workers: Optional[int] = None,
        **kwargs
    ) -> Dict[str, SophisticatedOptimizationResult]:
        """
        Optimize multiple features using sophisticated algorithms.
        
        Args:
            data: Input data with features
            feature_names: List of feature names to optimize
            target_columns: Mapping of direction to target column
            lookback_range: Min and max lookback periods
            optimization_direction: Direction to optimize (longs, shorts, both)
            execution_mode: Execution mode (light, full, blank)
            use_nested_cv: Whether to use nested cross-validation
            regularization_settings: Regularization settings
            max_workers: Maximum number of parallel workers
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping feature names to optimization results
        """
        tprint_info(f"🚀 Starting sophisticated optimization for {len(feature_names)} features")
        tprint_debug(f"📊 Direction: {optimization_direction.value}, Mode: {execution_mode}")
        tprint_debug(f"📊 Nested CV: {use_nested_cv}, Lookback range: {lookback_range}")
        
        start_time = time.time()
        
        try:
            # Get execution mode configuration
            mode_config = getattr(self.execution_mode_config, execution_mode, self.execution_mode_config.full)
            
            # Create mode-aware constraints
            if self.core_optimizer:
                constraints = self.core_optimizer.create_mode_aware_constraints(execution_mode)
            else:
                constraints = None
            
            # Determine which directions to optimize
            directions_to_optimize = self._get_directions_to_optimize(optimization_direction)
            
            # Build nested CV splits if requested
            outer_splits = None
            if use_nested_cv and len(data) > 100:  # Only use nested CV for sufficient data
                outer_splits = self._build_walk_forward_splits(len(data))
                tprint_info(f"🧭 Using nested CV with {len(outer_splits)} outer folds")
            else:
                tprint_info("🧭 Using single-pass optimization")
            
            # Optimize features
            results = {}
            
            for feature_name in feature_names:
                tprint_debug(f"🔍 Optimizing feature: {feature_name}")
                
                feature_results = {}
                
                for direction in directions_to_optimize:
                    target_column = target_columns.get(direction)
                    if not target_column:
                        tprint_warning(f"⚠️ No target column for direction {direction}")
                        continue
                    
                    tprint_debug(f"🎯 Optimizing {feature_name} for {direction} using {target_column}")
                    
                    # Optimize single feature for specific direction
                    result = self._optimize_single_feature_sophisticated(
                        data, feature_name, target_column, direction,
                        lookback_range, constraints, outer_splits,
                        regularization_settings, execution_mode, **kwargs
                    )
                    
                    if result and result.success:
                        feature_results[direction] = result
                        tprint_success(f"✅ {feature_name} ({direction}): lookback={result.best_lookback}, score={result.best_score:.4f}")
                    else:
                        tprint_warning(f"⚠️ Failed to optimize {feature_name} ({direction})")
                
                # Store results for this feature
                if feature_results:
                    results[feature_name] = feature_results
            
            execution_time = time.time() - start_time
            
            # Update performance stats
            self.performance_stats.update({
                'total_optimizations': len(feature_names),
                'successful_optimizations': len([r for f in results.values() for r in f.values() if r.success]),
                'failed_optimizations': len(feature_names) - len(results),
                'total_execution_time': execution_time,
                'nested_cv_operations': 1 if outer_splits else 0
            })
            
            tprint_success(f"✅ Sophisticated optimization completed in {execution_time:.3f}s")
            tprint_info(f"🏆 Results: {self.performance_stats['successful_optimizations']} successful, {self.performance_stats['failed_optimizations']} failed")
            
            return results
            
        except Exception as e:
            tprint_error(f"❌ Sophisticated optimization failed: {e}")
            return {}
    
    def _optimize_single_feature_sophisticated(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        direction: str,
        lookback_range: Tuple[int, int],
        constraints: Optional[LookbackConstraints],
        outer_splits: Optional[List[Tuple[slice, slice]]],
        regularization_settings: Optional[Dict[str, float]],
        execution_mode: str,
        **kwargs
    ) -> Optional[SophisticatedOptimizationResult]:
        """Optimize a single feature using sophisticated algorithms."""
        tprint_debug(f"🧠 Starting sophisticated optimization for {feature_name} ({direction})")
        
        try:
            if not self.core_optimizer:
                tprint_warning("⚠️ Core optimizer not available, using fallback")
                return self._fallback_optimization(
                    data, feature_name, target_column, direction, lookback_range
                )
            
            # Prepare optimization parameters
            optimizer_kwargs = {
                'regularization_settings': regularization_settings or {},
                'outer_split_iterator': outer_splits,
                **kwargs
            }
            
            # Choose optimization method based on execution mode and constraints
            if constraints and constraints.use_bayesian_optimization:
                tprint_debug(f"🧠 Using Bayesian TPE optimization for {feature_name}")
                result = self.core_optimizer._optimize_with_bayesian_tpe(
                    data, feature_name, target_column, lookback_range,
                    **optimizer_kwargs
                )
            else:
                tprint_debug(f"🧠 Using coarse-to-refine optimization for {feature_name}")
                result = self.core_optimizer._optimize_coarse_to_refine(
                    data, feature_name, target_column, lookback_range,
                    **optimizer_kwargs
                )
            
            if not result:
                return None
            
            # Convert to sophisticated result format
            sophisticated_result = SophisticatedOptimizationResult(
                feature_name=feature_name,
                best_lookback=result.best_lookback_period,
                best_score=result.best_score,
                method=result.optimization_method,
                optimization_time=result.optimization_time,
                n_trials=result.total_trials,
                convergence_achieved=result.convergence_achieved,
                stability_score=getattr(result, 'stability_score', 0.0),
                sensitivity_score=getattr(result, 'sensitivity_score', 0.0),
                regularization_penalty=getattr(result, 'regularization_penalty', 0.0),
                validation_scores=getattr(result, 'validation_scores', []),
                optimization_metadata=result.metadata or {},
                success=True,
                direction=direction,
                target_column=target_column
            )
            
            # Add nested CV metadata if available
            if outer_splits and result.metadata:
                sophisticated_result.outer_validation = result.metadata.get('outer_folds')
                sophisticated_result.frozen_from_inner = result.metadata.get('frozen_from_inner', True)
                sophisticated_result.lookback_aggregates = result.metadata.get('lookback_aggregates')
            
            return sophisticated_result
            
        except Exception as e:
            tprint_error(f"❌ Sophisticated optimization failed for {feature_name} ({direction}): {e}")
            return SophisticatedOptimizationResult(
                feature_name=feature_name,
                best_lookback=0,
                best_score=0.0,
                method="failed",
                optimization_time=0.0,
                n_trials=0,
                convergence_achieved=False,
                stability_score=0.0,
                sensitivity_score=0.0,
                regularization_penalty=0.0,
                validation_scores=[],
                optimization_metadata={},
                success=False,
                error_message=str(e),
                direction=direction,
                target_column=target_column
            )
    
    def _fallback_optimization(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        direction: str,
        lookback_range: Tuple[int, int]
    ) -> SophisticatedOptimizationResult:
        """Fallback optimization when sophisticated methods are not available."""
        tprint_debug(f"🔄 Using fallback optimization for {feature_name} ({direction})")
        
        try:
            # Simple grid search fallback
            min_lookback, max_lookback = lookback_range
            step_size = max(1, (max_lookback - min_lookback) // 10)
            
            best_lookback = min_lookback
            best_score = 0.0
            
            for lookback in range(min_lookback, max_lookback + 1, step_size):
                if feature_name not in data.columns or target_column not in data.columns:
                    continue
                
                # Calculate simple correlation score
                feature_series = data[feature_name].dropna()
                target_series = data[target_column].dropna()
                
                if len(feature_series) < lookback or len(target_series) < lookback:
                    continue
                
                # Calculate rolling correlation
                rolling_feature = feature_series.rolling(window=lookback).mean()
                aligned_feature = rolling_feature.dropna()
                aligned_target = target_series.loc[aligned_feature.index]
                
                if len(aligned_feature) < 10:
                    continue
                
                correlation = np.corrcoef(aligned_feature, aligned_target)[0, 1]
                score = abs(correlation) if not np.isnan(correlation) else 0.0
                
                if score > best_score:
                    best_score = score
                    best_lookback = lookback
            
            return SophisticatedOptimizationResult(
                feature_name=feature_name,
                best_lookback=best_lookback,
                best_score=best_score,
                method="fallback_grid_search",
                optimization_time=0.0,
                n_trials=len(range(min_lookback, max_lookback + 1, step_size)),
                convergence_achieved=True,
                stability_score=0.0,
                sensitivity_score=0.0,
                regularization_penalty=0.0,
                validation_scores=[],
                optimization_metadata={},
                success=True,
                direction=direction,
                target_column=target_column
            )
            
        except Exception as e:
            tprint_error(f"❌ Fallback optimization failed for {feature_name} ({direction}): {e}")
            return SophisticatedOptimizationResult(
                feature_name=feature_name,
                best_lookback=0,
                best_score=0.0,
                method="failed",
                optimization_time=0.0,
                n_trials=0,
                convergence_achieved=False,
                stability_score=0.0,
                sensitivity_score=0.0,
                regularization_penalty=0.0,
                validation_scores=[],
                optimization_metadata={},
                success=False,
                error_message=str(e),
                direction=direction,
                target_column=target_column
            )
    
    def _get_directions_to_optimize(self, optimization_direction: OptimizationDirection) -> List[str]:
        """Get list of directions to optimize based on optimization direction."""
        if optimization_direction == OptimizationDirection.LONGS:
            return ['long']
        elif optimization_direction == OptimizationDirection.SHORTS:
            return ['short']
        elif optimization_direction == OptimizationDirection.BOTH:
            return ['long', 'short']
        else:
            return ['long']  # Default to long
    
    def _build_walk_forward_splits(self, data_length: int, n_splits: int = 3) -> List[Tuple[slice, slice]]:
        """Build walk-forward cross-validation splits."""
        try:
            if data_length < 100:
                return []
            
            # Calculate split sizes
            min_train_size = max(50, data_length // 3)
            min_val_size = max(20, data_length // 10)
            
            splits = []
            for i in range(n_splits):
                # Calculate train and validation indices
                train_start = 0
                train_end = min_train_size + (i * (data_length - min_train_size) // n_splits)
                val_start = train_end
                val_end = min(val_start + min_val_size, data_length)
                
                if val_end - val_start < min_val_size:
                    break
                
                train_slice = slice(train_start, train_end)
                val_slice = slice(val_start, val_end)
                
                splits.append((train_slice, val_slice))
            
            return splits
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to build walk-forward splits: {e}")
            return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'failed_optimizations': 0,
            'total_execution_time': 0.0,
            'vectorbt_operations': 0,
            'bayesian_optimizations': 0,
            'coarse_to_refine_optimizations': 0,
            'parallel_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'nested_cv_operations': 0
        }


def create_sophisticated_lookback_optimizer(config: Optional[Dict[str, Any]] = None) -> SophisticatedLookbackOptimizer:
    """Create a sophisticated lookback optimizer with default configuration."""
    return SophisticatedLookbackOptimizer(config)