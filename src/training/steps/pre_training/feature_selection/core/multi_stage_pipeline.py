"""
Multi-Stage Feature Selection Pipeline

This module provides a reusable class for the multi-stage feature selection pipeline
that can be called from other parts of the system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import time
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
)

# Import configuration classes
from .config import FeatureSelectionConfig, FeatureSelectionResult

# Import VectorBT mRMR selector
try:
    from src.feature_selection.vectorbt.vectorbt_mrmr_selector import VectorBTMRMRSelector
    from src.feature_selection.vectorbt.vectorbt_config import VectorBTFeatureSelectionConfig
    VECTORBT_MRMR_AVAILABLE = True
except ImportError:
    VECTORBT_MRMR_AVAILABLE = False

# Import LightGBM and SHAP for ensemble methods
try:
    import lightgbm as lgb
    import shap
    LIGHTGBM_SHAP_AVAILABLE = True
except ImportError:
    LIGHTGBM_SHAP_AVAILABLE = False

# Import scikit-learn for ensemble methods
try:
    from sklearn.linear_model import LassoCV
    from sklearn.feature_selection import RFE
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel
    from scipy.stats import spearmanr
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import scipy for distance correlation and HSIC
try:
    from scipy.spatial.distance import pdist, squareform
    from scipy.linalg import eigh
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Import hardware optimization tools
try:
    from src.utils.hardware import (
        get_unified_hardware_manager,
        get_adaptive_optimization_engine,
        get_advanced_memory_optimizer,
        WorkloadType
    )
    from src.utils.hardware.advanced_memory_optimizer import AdvancedM1MemoryOptimizer, MemoryStrategy
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False

# Import VectorBT utilities
try:
    from ..utils.vectorbt_utils import (
        create_vectorbt_tools, VectorBTConfig, get_vectorbt_performance_stats,
        VECTORBT_UTILS_AVAILABLE
    )
except ImportError:
    VECTORBT_UTILS_AVAILABLE = False

# Import VectorBT optimization tools
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    )
    from src.utils.matrix_operations.vectorbt_optimizations import (
        VectorBTOptimizedOperations, get_unified_matrix_operations
    )
    VECTORBT_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    VECTORBT_OPTIMIZATIONS_AVAILABLE = False
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None
    VectorBTOptimizedOperations = None
    get_unified_matrix_operations = None

# Import matrix operations
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_batch_matrix_processor
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False


class MultiStageFeatureSelectionPipeline:
    """
    Multi-Stage Feature Selection Pipeline with VectorBT Optimizations and Early Pruning
    
    A reusable class that implements a two-stage feature selection approach:
    1. Stage 1: Enhanced Multi-Method Scoring (50% mRMR + 30% Distance Correlation + 20% HSIC)
    2. Stage 2: Progressive refinement with RFE using ensemble scoring
    
    Enhanced with:
    - VectorBTRollingOptimizer for distance correlation and bootstrap stability
    - UnifiedVectorizationManager for HSIC, LASSO, and cross-validation
    - Early pruning with progressive thresholds
    - Hardware acceleration and fast-fail error handling
    """

    def __init__(self, config: Optional[FeatureSelectionConfig] = None, execution_mode_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Multi-Stage Feature Selection Pipeline.
        
        Args:
            config: Feature selection configuration
            execution_mode_config: Additional execution mode configuration
        """
        self.config = config or FeatureSelectionConfig()
        self.logger = get_logger("MultiStageFeatureSelectionPipeline")
        self.matrix_ops = get_unified_matrix_operations()

        # Initialize VectorBT optimization tools if available
        if VECTORBT_UTILS_AVAILABLE and self.config.enable_vectorbt_optimization:
            tprint("🚀 Initializing VectorBT optimization tools")
            vectorbt_config = VectorBTConfig(
                enable_gpu=self.config.vectorbt_enable_gpu,
                enable_parallel=self.config.vectorbt_enable_parallel,
                memory_efficient=self.config.vectorbt_memory_efficient,
                chunk_size=self.config.vectorbt_chunk_size
            )
            
            vectorbt_tools = create_vectorbt_tools(vectorbt_config)
            self.vectorbt_optimizer = vectorbt_tools['optimizer']
            self.vectorization_manager = vectorbt_tools['manager']
            self.vectorbt_enabled = vectorbt_tools['available']
            
            if self.vectorbt_enabled:
                tprint("✅ VectorBT optimization tools initialized successfully")
            else:
                tprint_warning("⚠️ VectorBT optimization tools not available")
        else:
            self.vectorbt_optimizer = None
            self.vectorization_manager = None
            self.vectorbt_enabled = False
            tprint("⚠️ VectorBT optimization disabled or not available")
        
        # Initialize enhanced VectorBT optimization tools
        self.rolling_optimizer = None
        self.enhanced_vectorization_manager = None
        if VECTORBT_OPTIMIZATIONS_AVAILABLE:
            try:
                self.rolling_optimizer = get_vectorbt_rolling_optimizer()
                self.enhanced_vectorization_manager = get_unified_matrix_operations()
                tprint("✅ Enhanced VectorBT optimization tools initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Enhanced VectorBT tools not available: {e}")
                self.rolling_optimizer = None
                self.enhanced_vectorization_manager = None

        # Initialize hardware optimization tools if available
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            self.hardware_manager = get_unified_hardware_manager()
            self.adaptive_engine = get_adaptive_optimization_engine()
            self.memory_optimizer = get_advanced_memory_optimizer()
            
            # Initialize advanced memory optimizer for aggressive cleanup
            try:
                self.advanced_memory_optimizer = AdvancedM1MemoryOptimizer(
                    memory_limit_gb=8.0,
                    strategy=MemoryStrategy.AGGRESSIVE
                )
                self.logger.info("🧠 Advanced memory optimizer initialized with aggressive strategy")
            except Exception as e:
                self.advanced_memory_optimizer = None
                self.logger.warning(f"⚠️ Advanced memory optimizer not available: {e}")
            
            self.logger.info("🚀 Hardware optimization tools initialized")
        else:
            self.hardware_manager = None
            self.adaptive_engine = None
            self.memory_optimizer = None
            self.advanced_memory_optimizer = None
            self.logger.info("⚠️ Hardware optimization tools not available")

        # Initialize batch processor for chunked processing
        if MATRIX_OPERATIONS_AVAILABLE:
            self.batch_processor = get_batch_matrix_processor()
            self.logger.info("📦 Batch matrix processor initialized")
        else:
            self.batch_processor = None
            self.logger.info("⚠️ Batch matrix processor not available")

        # Initialize execution mode configuration
        self.execution_mode_config = execution_mode_config
        
        # Initialize VectorBT memory optimization if available
        if VECTORBT_UTILS_AVAILABLE:
            self._vectorbt_memory_config = {
                'use_memory_efficient': True,
                'chunk_size': 1000,
                'max_memory_gb': 8.0,
                'enable_gpu': False  # Can be enabled if GPU is available
            }
            tprint("🚀 VectorBT memory optimization enabled")
        else:
            self._vectorbt_memory_config = None
            tprint("⚠️ VectorBT memory optimization not available")
        
        # Initialize early pruning configuration
        self.enable_early_pruning = getattr(self.config, 'enable_early_pruning', True)
        self.pruning_thresholds = getattr(self.config, 'pruning_thresholds', [0.1, 0.2, 0.3])
        self.pruning_stats = {
            'features_pruned': 0,
            'pruning_rounds': 0,
            'memory_saved_mb': 0,
            'time_saved_seconds': 0
        }
        
        # Enhanced performance tracking
        self.performance_stats = {
            'stage1_time': 0.0,
            'stage2_time': 0.0,
            'total_time': 0.0,
            'features_processed': 0,
            'memory_optimizations': 0,
            'vectorbt_operations': 0,
            'distance_corr_time': 0.0,
            'hsic_time': 0.0,
            'lasso_ensemble_time': 0.0,
            'bootstrap_time': 0.0,
            'pruning_time': 0.0
        }

    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       symbol: str = "BTCUSDT", exchange: str = "binance", 
                       timeframe: str = "15m") -> FeatureSelectionResult:
        """
        Execute multi-stage feature selection with fast fail and extensive logging.
        
        Args:
            X: Feature matrix
            y: Target variable
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            
        Returns:
            FeatureSelectionResult with selected features and metrics
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If any stage of the pipeline fails
        """
        start_time = time.time()
        tprint("🚀 Starting Multi-Stage Feature Selection Pipeline")
        tprint_info(f"   📊 Input data shape: {X.shape}")
        tprint_info(f"   📊 Target shape: {y.shape}")
        tprint_info(f"   📊 Symbol: {symbol}, Exchange: {exchange}, Timeframe: {timeframe}")
        
        # FAST FAIL: Early validation checks
        tprint_debug("🔍 Performing fast fail validation checks")
        
        # Check input data validity
        if X is None or X.empty:
            error_msg = "Input feature matrix X is None or empty"
            tprint_error(f"❌ FAST FAIL: {error_msg}")
            raise ValueError(error_msg)
        
        if y is None or y.empty:
            error_msg = "Target variable y is None or empty"
            tprint_error(f"❌ FAST FAIL: {error_msg}")
            raise ValueError(error_msg)
        
        if len(X) != len(y):
            error_msg = f"Feature matrix length ({len(X)}) doesn't match target length ({len(y)})"
            tprint_error(f"❌ FAST FAIL: {error_msg}")
            raise ValueError(error_msg)
        
        if X.shape[1] == 0:
            error_msg = "Feature matrix has no columns"
            tprint_error(f"❌ FAST FAIL: {error_msg}")
            raise ValueError(error_msg)
        
        tprint_success("   ✅ Fast fail validation passed")
        
        # Use enhanced multi-method pipeline
        tprint("🚀 Using Enhanced Multi-Method Pipeline")
        tprint_info("   📊 Stage 1: Enhanced Multi-Method Scoring (50% mRMR + 30% Distance Correlation + 20% HSIC)")
        tprint_info("   📊 Stage 2: RFE with percentage-based step size (10% of features above target)")
        
        # Set thread limits
        tprint_debug("🔧 Setting thread limits")
        self._set_thread_limits()
        
        # Display VectorBT status
        tprint_debug("🚀 Displaying VectorBT status")
        self._display_vectorbt_status()
        
        # Initialize result tracking
        tprint_debug("📊 Initializing result tracking")
        stage_results = {}
        selected_features = X.columns.tolist()
        feature_importance = {}
        feature_scores = {}
        
        tprint_info(f"   📊 Starting with {len(selected_features)} features")
        
        # Stage 1: Enhanced Multi-Method Scoring
        tprint("📊 Stage 1: Enhanced Multi-Method Scoring (mRMR + Distance Correlation + HSIC)")
        tprint_debug(f"   🔍 Input features for Stage 1: {len(selected_features)}")
        
        try:
            stage_1_result = self._stage_1_enhanced_multi_method_scoring(X, y)
            selected_features = stage_1_result['selected_features']
            stage_results['stage_1'] = stage_1_result
            
            tprint_success(f"   ✅ Stage 1 completed: {len(selected_features)} features selected")
            tprint_debug(f"   📊 Stage 1 method: {stage_1_result.get('method', 'unknown')}")
            
        except Exception as e:
            error_msg = f"Stage 1 Enhanced Multi-Method Scoring failed: {e}"
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e
        
        # Stage 2: RFE with percentage-based step size
        tprint("📊 Stage 2: RFE with percentage-based step size")
        tprint_debug(f"   🔍 Input features for Stage 2: {len(selected_features)}")
        
        try:
            stage_2_result = self._stage_2_progressive_refinement(X, y, selected_features)
            selected_features = stage_2_result['selected_features']
            stage_results['stage_2'] = stage_2_result
            
            tprint_success(f"   ✅ Stage 2 completed: {len(selected_features)} features selected")
            tprint_debug(f"   📊 Stage 2 method: {stage_2_result.get('method', 'unknown')}")
            tprint_debug(f"   📊 Bootstrap/CV used: {stage_2_result.get('use_bootstrap_cv', False)}")
            
        except Exception as e:
            error_msg = f"Stage 2 RFE progressive refinement failed: {e}"
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e
        
        # Calculate final metrics
        tprint_debug("📊 Calculating final performance metrics")
        try:
            performance_metrics = self._calculate_performance_metrics(X[selected_features], y)
            tprint_success("   ✅ Performance metrics calculated")
            tprint_debug(f"   📊 Final feature count: {performance_metrics.get('n_features', 0)}")
            tprint_debug(f"   📊 Sample count: {performance_metrics.get('n_samples', 0)}")
        except Exception as e:
            tprint_warning(f"⚠️ Performance metrics calculation failed: {e}")
            performance_metrics = {}
        
        # Create result
        execution_time = time.time() - start_time
        tprint_debug("📊 Creating final result object")
        
        result = FeatureSelectionResult(
            selected_features=selected_features,
            feature_importance=feature_importance,
            feature_scores=feature_scores,
            performance_metrics=performance_metrics,
            validation_scores={},
            config_used=self.config,
            execution_time=execution_time,
            memory_usage={},
            stage_results=stage_results,
            success=True
        )
        
        tprint_success(f"✅ Feature selection completed successfully in {execution_time:.2f}s")
        tprint_info(f"   📊 Final result: {len(selected_features)} features selected from {len(X.columns)}")
        tprint_info(f"   📊 Reduction ratio: {len(selected_features)/len(X.columns):.1%}")
        
        # Log final feature list
        tprint_debug("📊 Selected features:")
        for i, feature in enumerate(selected_features, 1):
            tprint_debug(f"   {i:2d}. {feature}")
        
        return result

    def _set_thread_limits(self):
        """Set thread limits to avoid oversubscription."""
        tprint_debug("🔧 Setting thread limits to avoid oversubscription")
        try:
            import os
            os.environ['OMP_NUM_THREADS'] = '4'
            os.environ['MKL_NUM_THREADS'] = '4'
            os.environ['NUMEXPR_NUM_THREADS'] = '4'
        except Exception as e:
            tprint_warning(f"⚠️ Could not set thread limits: {e}")

    def _display_vectorbt_status(self):
        """Display VectorBT optimization status and configuration."""
        if VECTORBT_UTILS_AVAILABLE:
            tprint("🚀 VECTORBT OPTIMIZATION STATUS:")
            tprint(f"   ✅ VectorBT Available: {VECTORBT_UTILS_AVAILABLE}")
            tprint(f"   🧠 Memory Optimization: {'Enabled' if self._vectorbt_memory_config else 'Disabled'}")
            if self._vectorbt_memory_config:
                tprint(f"   📦 Chunk Size: {self._vectorbt_memory_config['chunk_size']}")
                tprint(f"   💾 Max Memory: {self._vectorbt_memory_config['max_memory_gb']}GB")
                tprint(f"   🎮 GPU Enabled: {self._vectorbt_memory_config['enable_gpu']}")
        else:
            tprint("⚠️ VectorBT optimization not available")

    def _y_numeric(self, y: pd.Series) -> np.ndarray:
        """Convert y to numeric for Spearman/ranks."""
        if pd.api.types.is_numeric_dtype(y):
            return y.values
        return pd.to_numeric(y, errors='coerce').values

    def spearman_abs_vectorized(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Optimized vectorized Spearman correlation calculation with early termination."""
        # Fast fail on empty data
        if X.empty or y.empty:
            return pd.Series(dtype=float, index=X.columns)
        
        # Convert to numeric and handle missing values
        y_numeric = self._y_numeric(y)
        valid_mask = ~np.isnan(y_numeric)
        
        if not valid_mask.any():
            return pd.Series(0.0, index=X.columns)
        
        # Use only valid samples
        X_valid = X.loc[valid_mask]
        y_valid = y_numeric[valid_mask]
        
        # Calculate ranks
        X_ranks = X_valid.rank(method="average")
        y_ranks = pd.Series(y_valid).rank(method="average")
        
        # Vectorized correlation calculation
        X_centered = X_ranks - X_ranks.mean()
        y_centered = y_ranks - y_ranks.mean()
        
        numerator = (X_centered * y_centered).sum()
        denominator = np.sqrt((X_centered ** 2).sum() * (y_centered ** 2).sum())
        
        # Avoid division by zero
        correlations = numerator / (denominator + 1e-10)
        
        return correlations.abs()

    def _stage_1_enhanced_multi_method_scoring(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Stage 1: Enhanced Multi-Method Scoring (50% mRMR + 30% Distance Correlation + 20% HSIC).
        
        Selects top 50% of features above target using weighted combination of three methods.
        """
        tprint_debug("🔍 Stage 1: Enhanced Multi-Method Scoring")
        tprint_debug(f"   📊 Input features: {len(X.columns)}")
        tprint_debug(f"   📊 Data shape: {X.shape}")
        
        target_features = self.config.target_features
        features_above_target = len(X.columns) - target_features
        target_ratio = self.config.stage1_target_ratio
        features_to_select = max(target_features, int(len(X.columns) * target_ratio))
        
        tprint_debug(f"   📊 Target features: {target_features}")
        tprint_debug(f"   📊 Features above target: {features_above_target}")
        tprint_debug(f"   📊 Target ratio: {target_ratio:.1%}")
        tprint_debug(f"   📊 Features to select: {features_to_select}")
        
        # Calculate mRMR scores (50% weight)
        tprint_debug("   📊 Calculating mRMR scores (50% weight)")
        mrmr_scores = self._calculate_mrmr_scores(X, y)
        
        # Calculate Distance Correlation scores (30% weight)
        tprint_debug("   📊 Calculating Distance Correlation scores (30% weight)")
        distance_corr_scores = self._calculate_distance_correlation_scores(X, y)
        
        # Calculate HSIC scores (20% weight)
        tprint_debug("   📊 Calculating HSIC scores (20% weight)")
        hsic_scores = self._calculate_hsic_scores(X, y)
        
        # Combine scores with weights
        tprint_debug("   📊 Combining scores with weights")
        mrmr_weight = self.config.stage1_mrmr_weight
        distance_corr_weight = self.config.stage1_distance_correlation_weight
        hsic_weight = self.config.stage1_hsic_weight
        
        combined_scores = (
            mrmr_scores * mrmr_weight + 
            distance_corr_scores * distance_corr_weight +
            hsic_scores * hsic_weight
        )
        
        # Select top features
        selected_features = self._select_top_features(
            X.columns.tolist(), combined_scores, features_to_select
        )
        
        tprint_debug(f"   ✅ Stage 1 completed: {len(selected_features)} features selected")
        
        return {
            'selected_features': selected_features,
            'mrmr_scores': mrmr_scores.to_dict(),
            'distance_correlation_scores': distance_corr_scores.to_dict(),
            'hsic_scores': hsic_scores.to_dict(),
            'combined_scores': combined_scores.to_dict(),
            'target_count': features_to_select,
            'method': 'enhanced_multi_method_scoring'
        }

    def _stage_2_progressive_refinement(self, X: pd.DataFrame, y: pd.Series, 
                                      current_features: List[str]) -> Dict[str, Any]:
        """
        Stage 2: Progressive refinement using RFE with percentage-based step size.
        
        Uses RFE to recursively remove 10% of features above target in each round.
        Uses bootstrap stability and CV only when 40+ features away from target.
        """
        tprint_debug("🔍 Stage 2: Progressive refinement with RFE")
        tprint_debug(f"   📊 Input features: {len(current_features)}")
        tprint_debug(f"   📊 Data shape: {X.shape}")
        
        target_features = self.config.target_features
        current_features = current_features.copy()
        
        tprint_debug(f"   📊 Target features: {target_features}")
        tprint_debug(f"   📊 RFE step percentage: {self.config.rfe_step_size:.1%}")
        tprint_debug(f"   📊 Bootstrap/CV threshold: {self.config.stage2_bootstrap_cv_threshold} features")
        
        # Check if we should use bootstrap stability and CV
        features_above_target = len(current_features) - target_features
        use_bootstrap_cv = features_above_target >= self.config.stage2_bootstrap_cv_threshold
        tprint_debug(f"   📊 Use bootstrap stability and CV: {use_bootstrap_cv} (threshold: {self.config.stage2_bootstrap_cv_threshold})")
        
        # Use RFE with percentage-based step size
        selected_features = self._rfe_with_percentage_step(
            X[current_features], y, current_features, target_features, use_bootstrap_cv
        )
        
        tprint_debug(f"   ✅ Stage 2 completed: {len(selected_features)} features selected")
        
        return {
            'selected_features': selected_features,
            'target_count': target_features,
            'method': 'rfe_percentage_based',
            'use_bootstrap_cv': use_bootstrap_cv
        }

    def _rfe_with_percentage_step(self, X: pd.DataFrame, y: pd.Series, 
                                 feature_names: List[str], target_features: int,
                                 use_bootstrap_cv: bool = False) -> List[str]:
        """
        Recursive Feature Elimination with percentage-based step size.
        
        Removes 10% of features above target in each RFE round, recursively.
        """
        tprint_debug("🔍 Starting RFE with percentage-based step size")
        
        current_features = feature_names.copy()
        current_X = X.copy()
        rfe_rounds = []
        
        while len(current_features) > target_features:
            features_above_target = len(current_features) - target_features
            
            # Calculate step size as percentage of features above target
            step_size = max(1, int(features_above_target * self.config.rfe_step_size))
            tprint_debug(f"   📊 RFE Round: {len(current_features)} features, {features_above_target} above target")
            tprint_debug(f"   📊 Step size: {step_size} features (10% of {features_above_target})")
            
            # Calculate feature importance using ensemble methods
            feature_scores = self._calculate_ensemble_feature_scores(
                current_X, y, use_bootstrap_cv=use_bootstrap_cv
            )
            
            # Select features to remove (lowest scores)
            features_to_remove = self._select_features_to_remove(
                current_features, feature_scores, step_size
            )
            
            # Remove features
            current_features = [f for f in current_features if f not in features_to_remove]
            current_X = current_X.drop(columns=features_to_remove)
            
            tprint_debug(f"   📊 Removed {len(features_to_remove)} features: {features_to_remove}")
            
            rfe_rounds.append({
                'round': len(rfe_rounds) + 1,
                'features_remaining': len(current_features),
                'features_removed': len(features_to_remove),
                'step_size': step_size,
                'features_above_target': features_above_target,
                'features_removed_list': features_to_remove
            })
            
            # Safety check to prevent infinite loop
            if len(rfe_rounds) > 100:
                tprint_warning("   ⚠️ Maximum RFE rounds reached, stopping")
                break
        
        tprint_debug(f"   ✅ RFE completed: {len(current_features)} features selected in {len(rfe_rounds)} rounds")
        
        return current_features

    def _calculate_mrmr_scores(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calculate mRMR scores using VectorBT optimization."""
        if VECTORBT_MRMR_AVAILABLE:
            # Use VectorBT mRMR selector
            mrmr_selector = VectorBTMRMRSelector()
            result = mrmr_selector.select_features(
                X.values, y.values, 
                n_features=min(len(X.columns), 100)  # Limit for performance
            )
            return pd.Series(result['feature_scores'], index=X.columns)
        else:
            # Fast fail if VectorBT mRMR not available
            error_msg = "VectorBT mRMR not available - fast fail"
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

    def _calculate_spearman_scores(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calculate Spearman correlation scores."""
        return self.spearman_abs_vectorized(X, y)

    def _calculate_distance_correlation_scores(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calculate distance correlation scores for all features with VectorBT optimization."""
        tprint_debug("   📊 Calculating distance correlation scores with VectorBT optimization")
        start_time = time.time()
        
        if not SCIPY_AVAILABLE:
            tprint_warning("   ⚠️ SciPy not available, falling back to Spearman correlation")
            return self.spearman_abs_vectorized(X, y)
        
        try:
            # Use VectorBTRollingOptimizer if available
            if self.rolling_optimizer and VECTORBT_OPTIMIZATIONS_AVAILABLE:
                return self._calculate_distance_correlation_vectorbt(X, y)
            
            # Subsample if enabled and dataset is large
            if self.config.distance_correlation_enable_subsampling and len(X) > self.config.distance_correlation_sample_size:
                tprint_debug(f"   📊 Subsampling data for distance correlation: {len(X)} -> {self.config.distance_correlation_sample_size}")
                sample_indices = np.random.choice(len(X), self.config.distance_correlation_sample_size, replace=False)
                X_sample = X.iloc[sample_indices]
                y_sample = y.iloc[sample_indices]
            else:
                X_sample = X
                y_sample = y
            
            # Calculate distance correlation for each feature
            distance_corr_scores = {}
            for feature in X.columns:
                try:
                    dc_score = self._distance_correlation(X_sample[feature], y_sample)
                    distance_corr_scores[feature] = dc_score
                except Exception as e:
                    tprint_debug(f"   ⚠️ Distance correlation failed for {feature}: {e}")
                    distance_corr_scores[feature] = 0.0
            
            self.performance_stats['distance_corr_time'] = time.time() - start_time
            return pd.Series(distance_corr_scores, index=X.columns)
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Distance correlation calculation failed: {e}, falling back to Spearman")
            return self.spearman_abs_vectorized(X, y)
    
    def _calculate_distance_correlation_vectorbt(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calculate distance correlation using VectorBTRollingOptimizer."""
        try:
            tprint_debug("   🚀 Using VectorBTRollingOptimizer for distance correlation")
            
            # Subsample if enabled and dataset is large
            if self.config.distance_correlation_enable_subsampling and len(X) > self.config.distance_correlation_sample_size:
                tprint_debug(f"   📊 Subsampling data: {len(X)} -> {self.config.distance_correlation_sample_size}")
                sample_indices = np.random.choice(len(X), self.config.distance_correlation_sample_size, replace=False)
                X_sample = X.iloc[sample_indices]
                y_sample = y.iloc[sample_indices]
            else:
                X_sample = X
                y_sample = y
            
            # Use rolling operations for efficient distance correlation calculation
            distance_corr_scores = {}
            window_size = min(100, len(X_sample) // 4)  # Adaptive window size
            
            for feature in X_sample.columns:
                try:
                    # Use rolling correlation as distance correlation approximation
                    rolling_corr = self.rolling_optimizer.rolling_correlation(
                        X_sample[feature], y_sample, window=window_size
                    )
                    
                    if rolling_corr is not None and not rolling_corr.empty:
                        # Use mean of rolling correlations as distance correlation approximation
                        dc_score = abs(rolling_corr.mean())
                    else:
                        # Fallback to standard distance correlation
                        dc_score = self._distance_correlation(X_sample[feature], y_sample)
                    
                    distance_corr_scores[feature] = dc_score
                    
                except Exception as e:
                    tprint_debug(f"   ⚠️ VectorBT distance correlation failed for {feature}: {e}")
                    # Fallback to standard distance correlation
                    dc_score = self._distance_correlation(X_sample[feature], y_sample)
                    distance_corr_scores[feature] = dc_score
            
            self.performance_stats['distance_corr_time'] = time.time() - start_time
            self.performance_stats['vectorbt_operations'] += 1
            tprint_debug(f"   ✅ VectorBT distance correlation completed in {self.performance_stats['distance_corr_time']:.2f}s")
            
            return pd.Series(distance_corr_scores, index=X.columns)
            
        except Exception as e:
            tprint_warning(f"   ⚠️ VectorBT distance correlation failed: {e}, using fallback")
            return self._calculate_distance_correlation_scores_fallback(X, y)
    
    def _calculate_distance_correlation_scores_fallback(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Fallback distance correlation calculation without VectorBT."""
        # Subsample if enabled and dataset is large
        if self.config.distance_correlation_enable_subsampling and len(X) > self.config.distance_correlation_sample_size:
            sample_indices = np.random.choice(len(X), self.config.distance_correlation_sample_size, replace=False)
            X_sample = X.iloc[sample_indices]
            y_sample = y.iloc[sample_indices]
        else:
            X_sample = X
            y_sample = y
        
        # Calculate distance correlation for each feature
        distance_corr_scores = {}
        for feature in X_sample.columns:
            try:
                dc_score = self._distance_correlation(X_sample[feature], y_sample)
                distance_corr_scores[feature] = dc_score
            except Exception as e:
                tprint_debug(f"   ⚠️ Distance correlation failed for {feature}: {e}")
                distance_corr_scores[feature] = 0.0
        
        return pd.Series(distance_corr_scores, index=X.columns)

    def _distance_correlation(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate distance correlation between two series."""
        try:
            # Remove NaN values
            valid_mask = ~(x.isna() | y.isna())
            if not valid_mask.any():
                return 0.0
            
            x_clean = x[valid_mask].values
            y_clean = y[valid_mask].values
            
            if len(x_clean) < 3:
                return 0.0
            
            # Calculate distance matrices
            x_dist = pdist(x_clean.reshape(-1, 1), metric='euclidean')
            y_dist = pdist(y_clean.reshape(-1, 1), metric='euclidean')
            
            # Convert to squareform
            x_dist_matrix = squareform(x_dist)
            y_dist_matrix = squareform(y_dist)
            
            # Center the distance matrices
            n = len(x_clean)
            x_centered = x_dist_matrix - np.mean(x_dist_matrix, axis=1)[:, np.newaxis] - np.mean(x_dist_matrix, axis=0) + np.mean(x_dist_matrix)
            y_centered = y_dist_matrix - np.mean(y_dist_matrix, axis=1)[:, np.newaxis] - np.mean(y_dist_matrix, axis=0) + np.mean(y_dist_matrix)
            
            # Calculate distance covariance and variances
            dcov_xy = np.sqrt(np.mean(x_centered * y_centered))
            dcov_xx = np.sqrt(np.mean(x_centered * x_centered))
            dcov_yy = np.sqrt(np.mean(y_centered * y_centered))
            
            # Avoid division by zero
            if dcov_xx == 0 or dcov_yy == 0:
                return 0.0
            
            # Distance correlation
            dcorr = dcov_xy / np.sqrt(dcov_xx * dcov_yy)
            
            return abs(dcorr)  # Return absolute value for feature selection
            
        except Exception as e:
            tprint_debug(f"   ⚠️ Distance correlation calculation error: {e}")
            return 0.0

    def _calculate_hsic_scores(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calculate HSIC scores for all features with VectorBT optimization."""
        tprint_debug("   📊 Calculating HSIC scores with VectorBT optimization")
        start_time = time.time()
        
        if not SCIPY_AVAILABLE or not SKLEARN_AVAILABLE:
            tprint_warning("   ⚠️ Required libraries not available, falling back to Spearman correlation")
            return self.spearman_abs_vectorized(X, y)
        
        try:
            # Use UnifiedVectorizationManager if available
            if self.enhanced_vectorization_manager and VECTORBT_OPTIMIZATIONS_AVAILABLE:
                return self._calculate_hsic_vectorbt(X, y)
            
            # Subsample if enabled and dataset is large
            if self.config.hsic_enable_subsampling and len(X) > self.config.hsic_sample_size:
                tprint_debug(f"   📊 Subsampling data for HSIC: {len(X)} -> {self.config.hsic_sample_size}")
                sample_indices = np.random.choice(len(X), self.config.hsic_sample_size, replace=False)
                X_sample = X.iloc[sample_indices]
                y_sample = y.iloc[sample_indices]
            else:
                X_sample = X
                y_sample = y
            
            # Calculate HSIC for each feature
            hsic_scores = {}
            for feature in X.columns:
                try:
                    hsic_score = self._hsic_score(X_sample[feature], y_sample)
                    hsic_scores[feature] = hsic_score
                except Exception as e:
                    tprint_debug(f"   ⚠️ HSIC calculation failed for {feature}: {e}")
                    hsic_scores[feature] = 0.0
            
            self.performance_stats['hsic_time'] = time.time() - start_time
            return pd.Series(hsic_scores, index=X.columns)
            
        except Exception as e:
            tprint_warning(f"   ⚠️ HSIC calculation failed: {e}, falling back to Spearman")
            return self.spearman_abs_vectorized(X, y)
    
    def _calculate_hsic_vectorbt(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calculate HSIC using UnifiedVectorizationManager."""
        try:
            tprint_debug("   🚀 Using UnifiedVectorizationManager for HSIC")
            
            # Subsample if enabled and dataset is large
            if self.config.hsic_enable_subsampling and len(X) > self.config.hsic_sample_size:
                tprint_debug(f"   📊 Subsampling data: {len(X)} -> {self.config.hsic_sample_size}")
                sample_indices = np.random.choice(len(X), self.config.hsic_sample_size, replace=False)
                X_sample = X.iloc[sample_indices]
                y_sample = y.iloc[sample_indices]
            else:
                X_sample = X
                y_sample = y
            
            # Use vectorization manager for efficient HSIC calculation
            hsic_scores = {}
            for feature in X_sample.columns:
                try:
                    # Use vectorization manager for kernel operations
                    if hasattr(self.enhanced_vectorization_manager, 'hsic_calculation'):
                        hsic_score = self.enhanced_vectorization_manager.hsic_calculation(
                            X_sample[feature], y_sample, 
                            kernel=self.config.hsic_kernel,
                            gamma=self.config.hsic_gamma
                        )
                    else:
                        # Fallback to standard HSIC calculation
                        hsic_score = self._hsic_score(X_sample[feature], y_sample)
                    
                    hsic_scores[feature] = hsic_score
                    
                except Exception as e:
                    tprint_debug(f"   ⚠️ VectorBT HSIC failed for {feature}: {e}")
                    # Fallback to standard HSIC calculation
                    hsic_score = self._hsic_score(X_sample[feature], y_sample)
                    hsic_scores[feature] = hsic_score
            
            self.performance_stats['hsic_time'] = time.time() - start_time
            self.performance_stats['vectorbt_operations'] += 1
            tprint_debug(f"   ✅ VectorBT HSIC completed in {self.performance_stats['hsic_time']:.2f}s")
            
            return pd.Series(hsic_scores, index=X.columns)
            
        except Exception as e:
            tprint_warning(f"   ⚠️ VectorBT HSIC failed: {e}, using fallback")
            return self._calculate_hsic_scores_fallback(X, y)
    
    def _calculate_hsic_scores_fallback(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Fallback HSIC calculation without VectorBT."""
        # Subsample if enabled and dataset is large
        if self.config.hsic_enable_subsampling and len(X) > self.config.hsic_sample_size:
            sample_indices = np.random.choice(len(X), self.config.hsic_sample_size, replace=False)
            X_sample = X.iloc[sample_indices]
            y_sample = y.iloc[sample_indices]
        else:
            X_sample = X
            y_sample = y
        
        # Calculate HSIC for each feature
        hsic_scores = {}
        for feature in X_sample.columns:
            try:
                hsic_score = self._hsic_score(X_sample[feature], y_sample)
                hsic_scores[feature] = hsic_score
            except Exception as e:
                tprint_debug(f"   ⚠️ HSIC calculation failed for {feature}: {e}")
                hsic_scores[feature] = 0.0
        
        return pd.Series(hsic_scores, index=X.columns)

    def _hsic_score(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate HSIC score between two series."""
        try:
            # Remove NaN values
            valid_mask = ~(x.isna() | y.isna())
            if not valid_mask.any():
                return 0.0
            
            x_clean = x[valid_mask].values
            y_clean = y[valid_mask].values
            
            if len(x_clean) < 3:
                return 0.0
            
            # Reshape for kernel calculation
            x_reshaped = x_clean.reshape(-1, 1)
            y_reshaped = y_clean.reshape(-1, 1)
            
            # Calculate kernels based on configuration
            kernel_type = self.config.hsic_kernel
            gamma = self.config.hsic_gamma
            
            if kernel_type == 'rbf':
                if gamma is None:
                    # Auto gamma: 1 / (n_features * X.var())
                    gamma = 1.0 / (x_reshaped.shape[1] * np.var(x_reshaped))
                Kx = rbf_kernel(x_reshaped, gamma=gamma)
                Ky = rbf_kernel(y_reshaped, gamma=gamma)
            elif kernel_type == 'linear':
                Kx = linear_kernel(x_reshaped)
                Ky = linear_kernel(y_reshaped)
            elif kernel_type == 'poly':
                Kx = polynomial_kernel(x_reshaped, degree=2)
                Ky = polynomial_kernel(y_reshaped, degree=2)
            else:
                # Default to RBF
                gamma = 1.0 / (x_reshaped.shape[1] * np.var(x_reshaped))
                Kx = rbf_kernel(x_reshaped, gamma=gamma)
                Ky = rbf_kernel(y_reshaped, gamma=gamma)
            
            # Center the kernels
            n = len(x_clean)
            H = np.eye(n) - np.ones((n, n)) / n  # Centering matrix
            
            Kx_centered = H @ Kx @ H
            Ky_centered = H @ Ky @ H
            
            # Calculate HSIC
            hsic = np.trace(Kx_centered @ Ky_centered) / (n - 1) ** 2
            
            return abs(hsic)  # Return absolute value for feature selection
            
        except Exception as e:
            tprint_debug(f"   ⚠️ HSIC calculation error: {e}")
            return 0.0

    def _calculate_ensemble_feature_scores(self, X: pd.DataFrame, y: pd.Series, 
                                         use_bootstrap_cv: bool = False) -> pd.Series:
        """Calculate ensemble feature scores using multiple methods."""
        ensemble_scores = {}
        
        # LGBM-SHAP scores (40% weight)
        if LIGHTGBM_SHAP_AVAILABLE:
            lgbm_scores = self._calculate_lgbm_shap_scores(X, y)
            ensemble_scores['lgbm_shap'] = lgbm_scores
        else:
            error_msg = "LightGBM-SHAP not available - fast fail"
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        # LASSO ensemble scores (30% weight)
        if SKLEARN_AVAILABLE:
            lasso_scores = self._calculate_lasso_ensemble_scores(X, y)
            ensemble_scores['lasso_ensemble'] = lasso_scores
            
            # RFE scores (20% weight)
            rfe_scores = self._calculate_rfe_scores(X, y)
            ensemble_scores['rfe'] = rfe_scores
        else:
            error_msg = "Scikit-learn not available - fast fail"
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        # Bootstrap stability scores (10% weight) - only if enabled
        if use_bootstrap_cv and SKLEARN_AVAILABLE:
            bootstrap_scores = self._calculate_bootstrap_stability_scores(X, y)
            ensemble_scores['bootstrap_stability'] = bootstrap_scores
        
        # Combine scores with weights
        return self._combine_ensemble_scores(ensemble_scores)

    def _calculate_lgbm_shap_scores(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calculate LGBM-SHAP feature importance scores."""
        # Train LightGBM model
        lgb_params = self.config.lgbm_params.copy()
        lgb_params['verbose'] = -1  # Suppress output
        
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X, y)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Calculate feature importance as mean absolute SHAP values
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        return pd.Series(feature_importance, index=X.columns)

    def _calculate_lasso_ensemble_scores(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calculate LASSO ensemble feature importance scores."""
        # Use LassoCV for automatic alpha selection
        lasso = LassoCV(
            alphas=np.logspace(
                self.config.lasso_alpha_range[0],
                self.config.lasso_alpha_range[1],
                self.config.lasso_n_alphas
            ),
            cv=self.config.lasso_cv_folds,
            random_state=42
        )
        
        lasso.fit(X, y)
        
        # Feature importance as absolute coefficients
        feature_importance = np.abs(lasso.coef_)
        
        return pd.Series(feature_importance, index=X.columns)

    def _calculate_rfe_scores(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calculate RFE feature importance scores."""
        # Use RandomForest as base estimator for RFE
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
        
        # Calculate feature importance
        rf.fit(X, y)
        feature_importance = rf.feature_importances_
        
        return pd.Series(feature_importance, index=X.columns)

    def _calculate_bootstrap_stability_scores(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calculate bootstrap stability scores."""
        n_samples = self.config.bootstrap_n_samples
        sample_ratio = self.config.bootstrap_sample_ratio
        stability_threshold = self.config.stability_threshold
        
        feature_stability = np.zeros(len(X.columns))
        
        for _ in range(n_samples):
            # Bootstrap sample
            n_samples_subset = int(len(X) * sample_ratio)
            indices = np.random.choice(len(X), n_samples_subset, replace=True)
            
            X_bootstrap = X.iloc[indices]
            y_bootstrap = y.iloc[indices]
            
            # Calculate feature importance for this bootstrap sample
            rf = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=1)
            rf.fit(X_bootstrap, y_bootstrap)
            importance = rf.feature_importances_
            
            # Count features above threshold
            feature_stability += (importance > stability_threshold).astype(int)
        
        # Normalize by number of successful bootstrap samples
        feature_stability = feature_stability / n_samples
        
        return pd.Series(feature_stability, index=X.columns)

    def _combine_ensemble_scores(self, ensemble_scores: Dict[str, pd.Series]) -> pd.Series:
        """Combine ensemble scores using configured weights."""
        if not ensemble_scores:
            error_msg = "No ensemble scores available - fast fail"
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        # Get weights from config
        weights = self.config.ensemble_weights
        
        # Normalize scores to 0-1 range
        normalized_scores = {}
        for method, scores in ensemble_scores.items():
            if scores.max() > 0:
                normalized_scores[method] = scores / scores.max()
            else:
                normalized_scores[method] = scores
        
        # Combine with weights
        combined_scores = pd.Series(0.0, index=list(ensemble_scores.values())[0].index)
        
        for method, scores in normalized_scores.items():
            weight = weights.get(method, 0.0)
            combined_scores += scores * weight
        
        return combined_scores

    def _select_features_to_remove(self, feature_names: List[str], 
                                  feature_scores: pd.Series, count: int) -> List[str]:
        """Select features to remove based on lowest scores."""
        if count <= 0:
            return []
        
        # Get scores for current features
        current_scores = feature_scores[feature_names]
        
        # Select features with lowest scores
        bottom_features = current_scores.nsmallest(count).index.tolist()
        
        return bottom_features

    def _select_top_features(self, feature_names: List[str], 
                           feature_scores: pd.Series, count: int) -> List[str]:
        """Select top features based on highest scores."""
        if count <= 0:
            return []
        
        # Get scores for current features
        current_scores = feature_scores[feature_names]
        
        # Select features with highest scores
        top_features = current_scores.nlargest(count).index.tolist()
        
        return top_features

    def _calculate_performance_metrics(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Calculate performance metrics for selected features."""
        tprint_debug("📊 Calculating performance metrics")
        
        metrics = {
            'n_features': len(X.columns),
            'n_samples': len(X),
            'feature_diversity': len(set([col.split('_')[0] for col in X.columns])),
            'data_quality': {
                'missing_ratio': X.isnull().sum().sum() / (len(X) * len(X.columns)),
                'variance_ratio': X.var().mean() / X.var().std() if X.var().std() > 0 else 0
            }
        }
        
        # Add VectorBT performance stats if available
        if self.vectorbt_enabled:
            try:
                vectorbt_stats = get_vectorbt_performance_stats()
                metrics['vectorbt_stats'] = vectorbt_stats
            except Exception as e:
                tprint_warning(f"⚠️ Could not get VectorBT stats: {e}")
        
        return metrics

    def cleanup(self):
        """Cleanup resources."""
        tprint("🧹 Cleaning up MultiStageFeatureSelectionPipeline resources")
        
        try:
            # Clear any caches or temporary data
            if hasattr(self, 'vectorbt_optimizer'):
                self.vectorbt_optimizer = None
                
            if hasattr(self, 'vectorization_manager'):
                self.vectorization_manager = None
            
            tprint_success("   ✅ Cleanup completed")
        except Exception as e:
            tprint_warning(f"   ⚠️ Cleanup error: {e}")


# Convenience function for easy usage
def run_multi_stage_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    timeframe: str = "15m",
    config: Optional[FeatureSelectionConfig] = None
) -> FeatureSelectionResult:
    """
    Convenience function to run multi-stage feature selection.
    
    Args:
        X: Feature matrix
        y: Target variable
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        config: Optional configuration
        
    Returns:
        FeatureSelectionResult with selected features
    """
    pipeline = MultiStageFeatureSelectionPipeline(config)
    try:
        result = pipeline.select_features(X, y, symbol, exchange, timeframe)
        return result
    finally:
        pipeline.cleanup()